use super::langevin::LangevinMiddleIntegrator;
use super::{fixed_particles, inverse_masses, Integrator};
use crate::constraints::{ConstraintError, ConstraintSolver};
use burn::prelude::*;
use std::sync::Mutex;

#[derive(Default)]
struct StepHistory {
    previous: f64,
    reported: f64,
    signature: Option<[f64; 3]>,
}

struct AdaptiveStep {
    error_tolerance: f64,
    maximum_step_size: f64,
    constraint_tolerance: f64,
    groups: u32,
    history: Mutex<StepHistory>,
}

impl AdaptiveStep {
    fn new(error_tolerance: f64) -> Self {
        assert!(error_tolerance.is_finite() && error_tolerance > 0.0);
        Self {
            error_tolerance,
            maximum_step_size: 0.0,
            constraint_tolerance: 1e-5,
            groups: u32::MAX,
            history: Mutex::new(StepHistory::default()),
        }
    }

    fn initialize(&mut self) {
        let history = self.history.get_mut().unwrap();
        history.previous = 0.0;
        history.signature = None;
    }

    fn select<B: Backend>(
        &self,
        forces: &Tensor<B, 2>,
        masses: &Tensor<B, 1>,
        limit: f64,
        temperature: f64,
        friction: f64,
    ) -> Result<(f64, f64), ConstraintError> {
        let error = crate::tensor::with_f64_data(forces, |forces| {
            crate::tensor::with_f64_data(masses, |masses| {
                let mut error = 0.0;
                for (atom, &mass) in masses.iter().enumerate() {
                    if mass != 0.0 {
                        let inverse_mass = mass.recip();
                        for axis in 0..3 {
                            let acceleration = forces[3 * atom + axis] * inverse_mass;
                            error += acceleration * acceleration;
                        }
                    }
                }
                (error / (3 * masses.len()) as f64).sqrt()
            })
        });
        let signature = [self.error_tolerance, temperature, friction];
        let mut history = self.history.lock().unwrap();
        let previous = if history.signature == Some(signature) {
            history.previous
        } else {
            0.0
        };
        let mut dt = (self.error_tolerance / error).sqrt();
        if previous > 0.0 {
            dt = dt.min(2.0 * previous);
        }
        if dt > previous && dt < f64::from(1.2_f32) * previous {
            dt = previous;
        }
        let limit = if self.maximum_step_size > 0.0 {
            limit.min(self.maximum_step_size)
        } else {
            limit
        };
        if dt > limit {
            dt = limit;
        }
        if !dt.is_finite() || dt <= 0.0 || !error.is_finite() {
            return Err(ConstraintError(
                "adaptive step size must be positive and finite; set a maximum step size for zero forces".into(),
            ));
        }
        history.previous = dt;
        history.signature = Some(signature);
        Ok((dt, previous))
    }
}

/// Variable-step leapfrog using the RMS acceleration error estimate.
pub struct VariableVerletIntegrator {
    adaptive: AdaptiveStep,
}

impl VariableVerletIntegrator {
    pub fn new(error_tolerance: f64) -> Self {
        Self {
            adaptive: AdaptiveStep::new(error_tolerance),
        }
    }

    fn advance<B: Backend>(
        &self,
        positions: &mut Tensor<B, 2>,
        velocities: &mut Tensor<B, 2>,
        forces: &Tensor<B, 2>,
        masses: &Tensor<B, 1>,
        constraints: &ConstraintSolver,
        limit: f64,
    ) -> Result<(), ConstraintError> {
        let (dt, previous_dt) = self.adaptive.select(forces, masses, limit, 0.0, 0.0)?;
        let vstep = 0.5 * (dt + previous_dt);
        let previous = (!constraints.is_empty()).then(|| positions.clone());
        let updated = super::try_host_step(positions, velocities, forces, masses, |p, v, f, m| {
            for (atom, &mass) in m.iter().enumerate() {
                if mass != 0.0 {
                    let inverse_mass = mass.recip();
                    for axis in 0..3 {
                        let i = 3 * atom + axis;
                        let old = p[i];
                        p[i] += (v[i] + inverse_mass * f[i] * vstep) * dt;
                        v[i] = (p[i] - old) / dt;
                    }
                }
            }
        });
        if updated.is_none() {
            let old = positions.clone();
            let velocity = velocities.clone() + forces.clone() * inverse_masses(masses) * vstep;
            *positions =
                positions.clone() + (velocity * dt).mask_fill(fixed_particles(masses), 0.0);
            super::velocities_from_positions(positions, &old, velocities, masses, dt, false);
        }
        if let Some(previous) = previous {
            constraints.apply_positions(
                &previous,
                positions,
                self.adaptive.constraint_tolerance,
            )?;
            super::velocities_from_positions(positions, &previous, velocities, masses, dt, false);
        }
        self.adaptive.history.lock().unwrap().reported = dt;
        Ok(())
    }

    fn initialize_state(&mut self) {
        self.adaptive.initialize();
    }
}

/// Variable-step LFMiddle with an independent seeded random stream.
pub struct VariableLangevinIntegrator {
    adaptive: AdaptiveStep,
    inner: Mutex<LangevinMiddleIntegrator>,
}

impl VariableLangevinIntegrator {
    pub fn new(temperature: f64, friction: f64, error_tolerance: f64) -> Self {
        Self {
            adaptive: AdaptiveStep::new(error_tolerance),
            inner: Mutex::new(LangevinMiddleIntegrator::new(temperature, friction, 1.0)),
        }
    }
    pub fn temperature(&self) -> f64 {
        self.inner.lock().unwrap().temperature()
    }
    pub fn set_temperature(&mut self, value: f64) {
        self.inner.get_mut().unwrap().set_temperature(value);
    }
    pub fn friction(&self) -> f64 {
        self.inner.lock().unwrap().friction()
    }
    pub fn set_friction(&mut self, value: f64) {
        self.inner.get_mut().unwrap().set_friction(value);
    }
    pub fn random_number_seed(&self) -> i32 {
        self.inner.lock().unwrap().random_number_seed()
    }
    pub fn set_random_number_seed(&mut self, seed: i32) {
        self.inner.get_mut().unwrap().set_random_number_seed(seed);
    }

    fn initialize_state(&mut self) {
        self.adaptive.initialize();
        self.inner.get_mut().unwrap().initialize_random();
    }

    fn advance<B: Backend>(
        &self,
        positions: &mut Tensor<B, 2>,
        velocities: &mut Tensor<B, 2>,
        forces: &Tensor<B, 2>,
        masses: &Tensor<B, 1>,
        constraints: &ConstraintSolver,
        limit: f64,
    ) -> Result<(), ConstraintError> {
        let mut inner = self.inner.lock().unwrap();
        let (dt, _) =
            self.adaptive
                .select(forces, masses, limit, inner.temperature(), inner.friction())?;
        inner.set_step_size(dt);
        inner.set_constraint_tolerance(self.adaptive.constraint_tolerance);
        inner.step_constrained(positions, velocities, forces, masses, constraints)?;
        self.adaptive.history.lock().unwrap().reported = dt;
        Ok(())
    }
}

macro_rules! adaptive_integrator {
    ($name:ident) => {
        impl $name {
            pub fn error_tolerance(&self) -> f64 {
                self.adaptive.error_tolerance
            }
            pub fn set_error_tolerance(&mut self, value: f64) {
                assert!(value.is_finite() && value > 0.0);
                self.adaptive.error_tolerance = value;
            }
            pub fn maximum_step_size(&self) -> f64 {
                self.adaptive.maximum_step_size
            }
            pub fn set_maximum_step_size(&mut self, value: f64) {
                assert!(value.is_finite() && value >= 0.0);
                self.adaptive.maximum_step_size = value;
            }
            pub fn set_constraint_tolerance(&mut self, value: f64) {
                assert!(value.is_finite() && value > 0.0 && value < 1.0);
                self.adaptive.constraint_tolerance = value;
            }
            pub fn integration_force_groups(&self) -> u32 {
                self.adaptive.groups
            }
            pub fn set_integration_force_groups(&mut self, value: u32) {
                self.adaptive.groups = value;
            }
            pub fn set_step_size(&mut self, value: f64) {
                assert!(value.is_finite() && value >= 0.0);
                self.adaptive.history.get_mut().unwrap().reported = value;
            }
        }
        impl<B: Backend> Integrator<B> for $name {
            fn set_step_size(&mut self, value: f64) {
                $name::set_step_size(self, value);
            }
            fn set_constraint_tolerance(&mut self, value: f64) {
                $name::set_constraint_tolerance(self, value);
            }
            fn initialize(&mut self) {
                self.initialize_state();
            }
            fn integration_force_groups(&self) -> u32 {
                self.adaptive.groups
            }
            fn set_integration_force_groups(&mut self, value: u32) {
                self.adaptive.groups = value;
            }
            fn step_size(&self) -> f64 {
                self.adaptive.history.lock().unwrap().reported
            }
            fn constraint_tolerance(&self) -> f64 {
                self.adaptive.constraint_tolerance
            }
            fn kinetic_energy_time_offset(&self) -> f64 {
                0.5 * self.adaptive.history.lock().unwrap().reported
            }
            fn supports_step_to(&self) -> bool {
                true
            }
            fn step(
                &self,
                p: &mut Tensor<B, 2>,
                v: &mut Tensor<B, 2>,
                f: &Tensor<B, 2>,
                m: &Tensor<B, 1>,
            ) {
                let constraints = ConstraintSolver::new(&[], &[]).unwrap();
                self.advance(p, v, f, m, &constraints, f64::INFINITY)
                    .unwrap_or_else(|e| panic!("{e}"));
            }
            fn step_constrained(
                &self,
                p: &mut Tensor<B, 2>,
                v: &mut Tensor<B, 2>,
                f: &Tensor<B, 2>,
                m: &Tensor<B, 1>,
                c: &ConstraintSolver,
            ) -> Result<(), ConstraintError> {
                self.advance(p, v, f, m, c, f64::INFINITY)
            }
            fn step_with_limit(
                &self,
                p: &mut Tensor<B, 2>,
                v: &mut Tensor<B, 2>,
                f: &Tensor<B, 2>,
                m: &Tensor<B, 1>,
                c: &ConstraintSolver,
                limit: f64,
            ) -> Result<(), ConstraintError> {
                self.advance(p, v, f, m, c, limit)
            }
        }
    };
}

adaptive_integrator!(VariableVerletIntegrator);
adaptive_integrator!(VariableLangevinIntegrator);
