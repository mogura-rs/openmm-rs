// OpenMM LFMiddle: full kick followed by drift, thermostat, and drift.
// Reference test: openmm/tests/TestLangevinMiddleIntegrator.h
// Reference impl: openmm/platforms/reference/src/SimTKReference/ReferenceLangevinMiddleDynamics.cpp
use super::random::GaussianStream;
use super::{fixed_particles, inverse_masses, Integrator};
use crate::constraints::{ConstraintError, ConstraintSolver};
use burn::prelude::*;
use rand::Rng;
use rand_distr::StandardNormal;

/// Boltzmann constant in kJ/mol/K.
pub const BOLTZMANN: f64 = 1.380649e-23 * 6.02214076e23 / 1000.0;

pub struct LangevinMiddleIntegrator {
    /// Step size in ps.
    dt: f64,
    /// Temperature in K.
    temperature: f64,
    /// Friction coefficient in ps^-1.
    friction: f64,
    constraint_tolerance: f64,
    integration_force_groups: u32,
    random: GaussianStream,
}

impl LangevinMiddleIntegrator {
    pub fn random_number_seed(&self) -> i32 {
        self.random.seed()
    }
    /// Set the seed for initialization; zero selects OS entropy.
    /// Changing the seed does not reset an initialized random stream.
    pub fn set_random_number_seed(&mut self, seed: i32) {
        self.random.set_seed(seed);
    }

    pub fn integration_force_groups(&self) -> u32 {
        self.integration_force_groups
    }
    pub fn set_integration_force_groups(&mut self, groups: u32) {
        self.integration_force_groups = groups;
    }

    pub fn new(temperature: f64, friction: f64, step_size: f64) -> Self {
        assert!(
            temperature.is_finite() && temperature >= 0.0,
            "temperature must be nonnegative and finite"
        );
        assert!(
            friction.is_finite() && friction >= 0.0,
            "friction must be nonnegative and finite"
        );
        assert!(
            step_size.is_finite() && step_size > 0.0,
            "step size must be positive and finite"
        );
        Self {
            dt: step_size,
            temperature,
            friction,
            constraint_tolerance: 1e-5,
            integration_force_groups: u32::MAX,
            random: GaussianStream::default(),
        }
    }

    pub fn temperature(&self) -> f64 {
        self.temperature
    }
    pub fn set_constraint_tolerance(&mut self, tolerance: f64) {
        assert!(
            tolerance.is_finite() && tolerance > 0.0 && tolerance < 1.0,
            "constraint tolerance must be between zero and one"
        );
        self.constraint_tolerance = tolerance;
    }
    pub fn friction(&self) -> f64 {
        self.friction
    }
    pub fn set_temperature(&mut self, t: f64) {
        assert!(
            t.is_finite() && t >= 0.0,
            "temperature must be nonnegative and finite"
        );
        self.temperature = t;
    }
    pub fn set_friction(&mut self, f: f64) {
        assert!(
            f.is_finite() && f >= 0.0,
            "friction must be nonnegative and finite"
        );
        self.friction = f;
    }
    pub fn set_step_size(&mut self, dt: f64) {
        assert!(
            dt.is_finite() && dt > 0.0,
            "step size must be positive and finite"
        );
        self.dt = dt;
    }
    fn try_host_update<B: Backend>(
        &self,
        positions: &mut Tensor<B, 2>,
        velocities: &mut Tensor<B, 2>,
        forces: &Tensor<B, 2>,
        masses: &Tensor<B, 1>,
        kick: bool,
    ) -> Option<()> {
        let dt = self.dt;
        let half_dt = 0.5 * dt;
        let decay = (-self.friction * dt).exp();
        let variance = BOLTZMANN * self.temperature * -(-2.0 * self.friction * dt).exp_m1();
        super::try_host_step(
            positions,
            velocities,
            forces,
            masses,
            |position, velocity, force, mass| {
                self.random.with_random(|random| {
                    for (atom, &mass) in mass.iter().enumerate() {
                        if mass != 0.0 {
                            let inverse_mass = mass.recip();
                            let noise_scale = (variance * inverse_mass).sqrt();
                            for axis in 0..3 {
                                let i = 3 * atom + axis;
                                if kick {
                                    velocity[i] += force[i] * inverse_mass * dt;
                                }
                                let midpoint = position[i] + half_dt * velocity[i];
                                velocity[i] = decay * velocity[i]
                                    + noise_scale * random.sample::<f64, _>(StandardNormal);
                                position[i] = midpoint + half_dt * velocity[i];
                            }
                        }
                    }
                });
            },
        )
    }
}

impl<B: Backend> Integrator<B> for LangevinMiddleIntegrator {
    fn initialize(&mut self) {
        self.random.initialize();
    }

    fn integration_force_groups(&self) -> u32 {
        self.integration_force_groups
    }
    fn set_integration_force_groups(&mut self, groups: u32) {
        self.integration_force_groups = groups;
    }

    fn step(
        &self,
        positions: &mut Tensor<B, 2>,
        velocities: &mut Tensor<B, 2>,
        forces: &Tensor<B, 2>,
        masses: &Tensor<B, 1>,
    ) {
        if self
            .try_host_update(positions, velocities, forces, masses, true)
            .is_some()
        {
            return;
        }
        let dt = self.dt;
        let half_dt = dt * 0.5;

        let inv_mass = inverse_masses(masses);
        let fixed = fixed_particles(masses);

        let accel = forces.clone() * inv_mass.clone();
        *velocities = velocities.clone() + accel * dt;

        *positions =
            positions.clone() + (velocities.clone() * half_dt).mask_fill(fixed.clone(), 0.0);

        let a = (-self.friction * self.dt).exp();
        let kt = BOLTZMANN * self.temperature;

        let variance = -(-2.0 * self.friction * self.dt).exp_m1();
        let b_per_particle = (inv_mass * kt * variance).sqrt();

        let noise = self.random.tensor(masses);
        *velocities = (velocities.clone() * a + noise * b_per_particle)
            .mask_where(fixed.clone(), velocities.clone());

        *positions = positions.clone() + (velocities.clone() * half_dt).mask_fill(fixed, 0.0);
    }

    fn step_size(&self) -> f64 {
        self.dt
    }

    fn constraint_tolerance(&self) -> f64 {
        self.constraint_tolerance
    }

    fn step_constrained(
        &self,
        positions: &mut Tensor<B, 2>,
        velocities: &mut Tensor<B, 2>,
        forces: &Tensor<B, 2>,
        masses: &Tensor<B, 1>,
        constraints: &ConstraintSolver,
    ) -> Result<(), ConstraintError>
    where
        B: Backend,
    {
        if constraints.is_empty() {
            self.step(positions, velocities, forces, masses);
            return Ok(());
        }
        let previous = positions.clone();
        if super::try_host_step(
            positions,
            velocities,
            forces,
            masses,
            |_, velocity, force, mass| {
                for (atom, &mass) in mass.iter().enumerate() {
                    if mass != 0.0 {
                        let inverse_mass = mass.recip();
                        for axis in 0..3 {
                            let i = 3 * atom + axis;
                            velocity[i] += force[i] * inverse_mass * self.dt;
                        }
                    }
                }
            },
        )
        .is_some()
        {
            constraints.apply_velocities(positions, velocities, self.constraint_tolerance)?;
            self.try_host_update(positions, velocities, forces, masses, false)
                .unwrap();
            let proposed = positions.clone();
            constraints.apply_positions(&previous, positions, self.constraint_tolerance)?;
            super::velocities_from_positions(
                positions, &proposed, velocities, masses, self.dt, true,
            );
            return Ok(());
        }
        let dt = self.dt;
        let fixed = fixed_particles(masses);
        let inverse_masses = inverse_masses(masses);
        *velocities = velocities.clone() + forces.clone() * inverse_masses.clone() * dt;
        constraints.apply_velocities(positions, velocities, self.constraint_tolerance)?;
        *positions =
            positions.clone() + (velocities.clone() * (0.5 * dt)).mask_fill(fixed.clone(), 0.0);
        let decay = (-self.friction * self.dt).exp();
        let variance = BOLTZMANN * self.temperature * -(-2.0 * self.friction * self.dt).exp_m1();
        let noise = self.random.tensor(masses);
        *velocities = (velocities.clone() * decay + noise * (inverse_masses * variance).sqrt())
            .mask_where(fixed.clone(), velocities.clone());
        *positions = positions.clone() + (velocities.clone() * (0.5 * dt)).mask_fill(fixed, 0.0);
        let proposed = positions.clone();
        constraints.apply_positions(&previous, positions, self.constraint_tolerance)?;
        super::velocities_from_positions(positions, &proposed, velocities, masses, dt, true);
        Ok(())
    }
}

/// Generate independent standard normal samples on the selected backend.
pub fn gaussian_noise<B: Backend>(shape: [usize; 2], device: &B::Device) -> Tensor<B, 2> {
    Tensor::random(shape, burn::tensor::Distribution::Normal(0.0, 1.0), device)
}

/// Historical name for the LFMiddle integrator.
pub type LangevinIntegrator = LangevinMiddleIntegrator;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use burn::backend::NdArray;

    type B = NdArray;

    // Free particle with thermostat: kinetic energy should fluctuate around 0.5*kT per DOF.
    // Statistical test with many steps.
    #[test]
    fn test_temperature() {
        let num_particles = 32;
        let temperature = 300.0;
        let integrator = LangevinMiddleIntegrator::new(temperature, 3.0, 0.01);
        let device: <B as Backend>::Device = Default::default();

        let mass = 12.0f32; // carbon
        let mut pos = Tensor::<B, 2>::zeros([num_particles, 3], &device);
        let mut vel = Tensor::<B, 2>::zeros([num_particles, 3], &device);
        let forces = Tensor::<B, 2>::zeros([num_particles, 3], &device);
        let masses = Tensor::<B, 1>::from_floats(vec![mass; num_particles].as_slice(), &device);

        // Equilibrate.
        for _ in 0..5000 {
            integrator.step(&mut pos, &mut vel, &forces, &masses);
        }

        // Measure average kinetic energy over many thermostat updates.
        // E_k = 0.5 * sum_i m_i * |v_i|^2; expected = 0.5 * N_dof * kT.
        let mut total_ek = 0.0f64;
        let steps = 2000;
        for _ in 0..steps {
            integrator.step(&mut pos, &mut vel, &forces, &masses);
            let v2: f32 = (vel.clone() * vel.clone()).sum().into_scalar();
            total_ek += 0.5 * mass as f64 * v2 as f64;
        }
        let avg_ek = total_ek / steps as f64;
        let expected_ek = 0.5 * (num_particles * 3) as f64 * BOLTZMANN * temperature;

        // Many independent particles keep the estimator variance manageable.
        assert_relative_eq!(avg_ek, expected_ek, epsilon = expected_ek * 0.12);
    }
}
