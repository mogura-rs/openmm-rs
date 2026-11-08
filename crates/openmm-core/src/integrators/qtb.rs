mod bath;
mod spectrum;

use super::{random::GaussianStream, Integrator};
use crate::constraints::{ConstraintError, ConstraintSolver};
use crate::System;
use bath::{Bath, Configuration};
use burn::prelude::*;
pub use spectrum::quantum_thermal_bath_spectrum;
use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

/// Adaptive quantum thermal bath with a frequency-dependent Gaussian force.
pub struct QTBIntegrator {
    temperature: f64,
    friction: f64,
    dt: f64,
    segment_length: f64,
    cutoff: f64,
    rate: f64,
    tolerance: f64,
    groups: u32,
    types: BTreeMap<i32, i32>,
    rates: BTreeMap<i32, f64>,
    random: GaussianStream,
    bath: Mutex<Option<Bath>>,
    captured_dt: Option<f64>,
}

impl QTBIntegrator {
    pub fn new(temperature: f64, friction: f64, step_size: f64) -> Self {
        assert!(temperature.is_finite() && temperature >= 0.0);
        assert!(friction.is_finite() && friction >= 0.0);
        assert!(step_size.is_finite() && step_size > 0.0);
        Self {
            temperature,
            friction,
            dt: step_size,
            segment_length: 1.0,
            cutoff: 500.0,
            rate: 0.01,
            tolerance: 1e-5,
            groups: u32::MAX,
            types: BTreeMap::new(),
            rates: BTreeMap::new(),
            random: GaussianStream::default(),
            bath: Mutex::new(None),
            captured_dt: None,
        }
    }
    pub fn temperature(&self) -> f64 {
        self.temperature
    }
    pub fn set_temperature(&mut self, value: f64) {
        assert!(value.is_finite() && value >= 0.0);
        self.temperature = value;
    }
    pub fn friction(&self) -> f64 {
        self.friction
    }
    pub fn set_friction(&mut self, value: f64) {
        assert!(value.is_finite() && value >= 0.0);
        self.friction = value;
    }
    pub fn step_size(&self) -> f64 {
        self.dt
    }
    pub fn set_step_size(&mut self, value: f64) {
        assert!(value.is_finite() && value > 0.0);
        self.dt = value;
    }
    pub fn segment_length(&self) -> f64 {
        self.segment_length
    }
    pub fn set_segment_length(&mut self, value: f64) {
        assert!(value.is_finite() && value > 0.0);
        self.segment_length = value;
    }
    pub fn cutoff_frequency(&self) -> f64 {
        self.cutoff
    }
    pub fn set_cutoff_frequency(&mut self, value: f64) {
        assert!(value.is_finite() && value > 0.0);
        self.cutoff = value;
    }
    pub fn default_adaptation_rate(&self) -> f64 {
        self.rate
    }
    pub fn set_default_adaptation_rate(&mut self, value: f64) {
        assert!(value.is_finite() && value >= 0.0);
        self.rate = value;
    }
    pub fn constraint_tolerance(&self) -> f64 {
        self.tolerance
    }
    pub fn set_constraint_tolerance(&mut self, value: f64) {
        assert!(value.is_finite() && value > 0.0 && value < 1.0);
        self.tolerance = value;
    }
    pub fn integration_force_groups(&self) -> u32 {
        self.groups
    }
    pub fn set_integration_force_groups(&mut self, value: u32) {
        self.groups = value;
    }
    pub fn random_number_seed(&self) -> i32 {
        self.random.seed()
    }
    pub fn set_random_number_seed(&mut self, value: i32) {
        self.random.set_seed(value);
    }
    pub fn particle_types(&self) -> &BTreeMap<i32, i32> {
        &self.types
    }
    pub fn set_particle_type(&mut self, index: i32, value: i32) {
        self.types.insert(index, value);
    }
    pub fn type_adaptation_rates(&self) -> &BTreeMap<i32, f64> {
        &self.rates
    }
    pub fn set_type_adaptation_rate(&mut self, kind: i32, value: f64) {
        assert!(value.is_finite() && value >= 0.0);
        self.rates.insert(kind, value);
    }
    pub fn adapted_friction(&self, particle: usize) -> Result<Vec<f64>, ConstraintError> {
        let bath = self.bath.lock().unwrap();
        let bath = bath
            .as_ref()
            .ok_or_else(|| ConstraintError("QTBIntegrator is not initialized".into()))?;
        bath.adapted_friction(particle)
    }
    fn configure<B: Backend>(&self, system: &System<B>) -> Result<Configuration, ConstraintError> {
        let steps = (self.segment_length / self.dt).ceil();
        if self.friction <= 0.0
            || !steps.is_finite()
            || steps < 1.0
            || steps > (i32::MAX / 3) as f64
        {
            return Err(ConstraintError(
                "QTB requires positive friction and a representable segment length".into(),
            ));
        }
        let steps = steps as usize;
        let count = system.num_particles();
        if count
            .checked_mul(3)
            .and_then(|n| n.checked_mul(steps))
            .is_none_or(|n| n > isize::MAX as usize / 8)
        {
            return Err(ConstraintError("QTB segment storage is too large".into()));
        }
        let masses = system.masses();
        let mut type_index = HashMap::new();
        let mut particle_types = vec![0; count];
        let mut particles: Vec<Vec<usize>> = Vec::new();
        let mut rates = Vec::new();
        let mut type_masses = Vec::new();
        for (&particle, &kind) in &self.types {
            if particle < 0 || particle as usize >= count {
                return Err(ConstraintError(
                    "QTB particle type index is out of range".into(),
                ));
            }
            let particle = particle as usize;
            let index = *type_index.entry(kind).or_insert_with(|| {
                let index = particles.len();
                particles.push(Vec::new());
                rates.push(*self.rates.get(&kind).unwrap_or(&self.rate));
                type_masses.push(masses[particle]);
                index
            });
            if type_masses[index] != masses[particle] {
                return Err(ConstraintError(
                    "all QTB particles of the same type must have the same mass".into(),
                ));
            }
            particle_types[particle] = index;
            particles[index].push(particle);
        }
        for particle in 0..count {
            if !self.types.contains_key(&(particle as i32)) {
                particle_types[particle] = particles.len();
                particles.push(vec![particle]);
                rates.push(self.rate);
                type_masses.push(masses[particle]);
            }
        }
        Ok(Configuration {
            dt: self.dt,
            friction: self.friction,
            cutoff: self.cutoff,
            steps,
            particle_types,
            particles,
            rates,
            masses: type_masses,
        })
    }
    fn advance<B: Backend>(
        &self,
        p: &mut Tensor<B, 2>,
        v: &mut Tensor<B, 2>,
        f: &Tensor<B, 2>,
        m: &Tensor<B, 1>,
        constraints: Option<&ConstraintSolver>,
    ) -> Result<(), ConstraintError> {
        let mut bath = self.bath.lock().unwrap();
        let bath = bath.as_mut().ok_or_else(|| {
            ConstraintError("QTBIntegrator must be initialized with a System".into())
        })?;
        bath.prepare(self.temperature)?;
        let constrained = constraints.filter(|value| !value.is_empty());
        let previous = constrained.map(|_| p.clone());
        let update = |p: &mut Tensor<B, 2>, v: &mut Tensor<B, 2>, bath: &mut Bath, phase: u8| {
            let run =
                |p: &mut [f64], v: &mut [f64], f: &[f64], m: &[f64]| bath.update(p, v, f, m, phase);
            if super::try_host_step(p, v, f, m, run).is_none() {
                let mut position = crate::tensor::to_f64_vec(p);
                let mut velocity = crate::tensor::to_f64_vec(v);
                bath.update(
                    &mut position,
                    &mut velocity,
                    &crate::tensor::to_f64_vec(f),
                    &crate::tensor::to_f64_vec(m),
                    phase,
                );
                *p = crate::tensor::from_f64_vec(position, p.dims(), &p.device());
                *v = crate::tensor::from_f64_vec(velocity, v.dims(), &v.device());
            }
        };
        if let Some(constraints) = constrained {
            update(p, v, bath, 1);
            constraints.apply_velocities(p, v, self.tolerance)?;
            update(p, v, bath, 2);
            let proposed = p.clone();
            constraints.apply_positions(previous.as_ref().unwrap(), p, self.tolerance)?;
            super::velocities_from_positions(p, &proposed, v, m, bath.config.dt, true);
        } else {
            update(p, v, bath, 0);
        }
        bath.finish_step();
        Ok(())
    }
}

impl<B: Backend> Integrator<B> for QTBIntegrator {
    fn validate_system(&self, system: &System<B>) -> Result<(), ConstraintError> {
        self.configure(system).map(|_| ())
    }
    fn initialize_system(&mut self, system: &System<B>) {
        let config = self.configure(system).expect("invalid QTB configuration");
        self.captured_dt = Some(config.dt);
        self.random.initialize();
        *self.bath.get_mut().unwrap() = Some(self.random.with_random(|rng| Bath::new(config, rng)));
    }
    fn step_size(&self) -> f64 {
        self.dt
    }
    fn actual_step_size(&self) -> f64 {
        self.captured_dt.unwrap_or(self.dt)
    }
    fn set_step_size(&mut self, value: f64) {
        Self::set_step_size(self, value);
    }
    fn constraint_tolerance(&self) -> f64 {
        self.tolerance
    }
    fn set_constraint_tolerance(&mut self, value: f64) {
        Self::set_constraint_tolerance(self, value);
    }
    fn integration_force_groups(&self) -> u32 {
        self.groups
    }
    fn set_integration_force_groups(&mut self, value: u32) {
        self.groups = value;
    }
    fn velocity_time_offset(&self) -> f64 {
        0.5 * self.captured_dt.unwrap_or(self.dt)
    }
    fn kinetic_energy_time_offset(&self) -> f64 {
        0.0
    }
    fn step(&self, p: &mut Tensor<B, 2>, v: &mut Tensor<B, 2>, f: &Tensor<B, 2>, m: &Tensor<B, 1>) {
        self.advance(p, v, f, m, None).expect("QTB step failed");
    }
    fn step_constrained(
        &self,
        p: &mut Tensor<B, 2>,
        v: &mut Tensor<B, 2>,
        f: &Tensor<B, 2>,
        m: &Tensor<B, 1>,
        constraints: &ConstraintSolver,
    ) -> Result<(), ConstraintError> {
        self.advance(p, v, f, m, Some(constraints))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Context, StateFlags};
    use approx::assert_relative_eq;
    use burn::backend::NdArray;
    type B = NdArray<f64>;

    #[test]
    fn all_fft_bins_are_filtered_and_fixed_particles_remain_fixed() {
        for length in [1, 2, 3, 4, 5] {
            let mut system = System::<B>::new();
            system.add_particle(1.0);
            system.add_particle(0.0);
            let mut integrator = QTBIntegrator::new(0.0, 1.0, 0.001);
            integrator.set_segment_length(length as f64 * 0.001);
            integrator.set_cutoff_frequency(1e-5);
            integrator.set_default_adaptation_rate(0.0);
            integrator.set_random_number_seed(71);
            let mut context = Context::new(system, Box::new(integrator), Default::default());
            context.set_positions(
                crate::tensor::from_floats::<B, 1>(
                    [[0.0; 3], [2.0; 3]].concat().as_slice(),
                    &Default::default(),
                )
                .reshape([2, 3]),
            );
            context.set_velocities(
                crate::tensor::from_floats::<B, 1>(
                    [[1.0, 0.0, 0.0], [3.0; 3]].concat().as_slice(),
                    &Default::default(),
                )
                .reshape([2, 3]),
            );
            context.integrator_mut().set_step_size(0.002);
            context.step(12);
            let state = context.get_state(StateFlags::all());
            let p = crate::tensor::to_f64_vec(state.positions.as_ref().unwrap());
            let v = crate::tensor::to_f64_vec(state.velocities.as_ref().unwrap());
            assert_relative_eq!(state.time, 0.012, epsilon = 1e-16);
            assert_relative_eq!(v[0], (-0.012_f64).exp(), epsilon = 1e-14);
            assert_eq!(&v[1..3], &[0.0; 2]);
            assert_eq!(&p[3..6], &[2.0; 3]);
            assert_eq!(&v[3..6], &[3.0; 3]);
            assert_relative_eq!(
                state.kinetic_energy.unwrap(),
                0.5 * v[0] * v[0],
                epsilon = 1e-14
            );
        }
    }

    #[test]
    fn type_validation_and_reinitialization_capture_settings() {
        let mut system = System::<B>::new();
        system.add_particle(1.0);
        system.add_particle(2.0);
        let mut integrator = QTBIntegrator::new(300.0, 1.0, 0.001);
        integrator.set_segment_length(0.002);
        integrator.set_particle_type(0, -7);
        integrator.set_particle_type(1, -7);
        assert!(integrator.validate_system(&system).is_err());
        integrator.set_particle_type(1, 12345);
        integrator.initialize_system(&system);
        assert_eq!(integrator.adapted_friction(0).unwrap(), vec![1.0; 4]);
        integrator.set_friction(2.0);
        integrator.set_segment_length(0.004);
        assert_eq!(integrator.adapted_friction(0).unwrap(), vec![1.0; 4]);
        integrator.initialize_system(&system);
        assert_eq!(integrator.adapted_friction(0).unwrap(), vec![2.0; 7]);
        integrator.set_particle_type(-1, 0);
        assert!(integrator.validate_system(&system).is_err());
        assert!(integrator.adapted_friction(2).is_err());
    }

    #[test]
    fn single_precision_uses_the_same_captured_dynamics() {
        type Single = NdArray<f32>;
        let mut system = System::<Single>::new();
        system.add_particle(2.0);
        let mut integrator = QTBIntegrator::new(0.0, 10.0, 0.001);
        integrator.set_segment_length(0.002);
        integrator.set_cutoff_frequency(1e-5);
        integrator.set_default_adaptation_rate(0.0);
        let mut context = Context::new(system, Box::new(integrator), Default::default());
        context.set_velocities(crate::tensor::from_f64_vec(
            vec![1.0, 2.0, 3.0],
            [1, 3],
            &Default::default(),
        ));
        context.integrator_mut().set_step_size(0.002);
        context.step(10);
        let state = context.get_state(StateFlags::all());
        for (actual, initial) in crate::tensor::to_f64_vec(state.velocities.as_ref().unwrap())
            .into_iter()
            .zip([1.0, 2.0, 3.0])
        {
            assert_relative_eq!(actual, initial * (-0.1_f64).exp(), epsilon = 8e-7);
        }
        assert_relative_eq!(state.time, 0.01, epsilon = 1e-15);
    }
}
