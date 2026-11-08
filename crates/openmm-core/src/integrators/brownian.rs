// BrownianIntegrator: overdamped Langevin (no inertia).
//   x(t+dt) = x(t) + F/(m*gamma)*dt + sqrt(2*kT*dt/(m*gamma)) * noise
//
// Reference test: openmm/tests/TestBrownianIntegrator.h
use super::random::GaussianStream;
use super::{fixed_particles, inverse_masses, Integrator};
use crate::constraints::{ConstraintError, ConstraintSolver};
use crate::integrators::langevin::BOLTZMANN;
use burn::prelude::*;
use rand::Rng;
use rand_distr::StandardNormal;

pub struct BrownianIntegrator {
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

impl BrownianIntegrator {
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
            friction.is_finite() && friction > 0.0,
            "friction must be positive and finite"
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
            f.is_finite() && f > 0.0,
            "friction must be positive and finite"
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
}

impl<B: Backend> Integrator<B> for BrownianIntegrator {
    fn set_step_size(&mut self, value: f64) {
        BrownianIntegrator::set_step_size(self, value);
    }
    fn set_constraint_tolerance(&mut self, value: f64) {
        BrownianIntegrator::set_constraint_tolerance(self, value);
    }
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
        let force_scale = self.dt / self.friction;
        let noise_amplitude = (2.0 * BOLTZMANN * self.temperature * force_scale).sqrt();
        if super::try_host_step(
            positions,
            velocities,
            forces,
            masses,
            |position, velocity, force, mass| {
                self.random.with_random(|random| {
                    for (atom, &mass) in mass.iter().enumerate() {
                        if mass != 0.0 {
                            let inverse_mass = mass.recip();
                            let noise_scale = noise_amplitude * inverse_mass.sqrt();
                            for axis in 0..3 {
                                let i = 3 * atom + axis;
                                let previous = position[i];
                                position[i] = previous
                                    + force_scale * inverse_mass * force[i]
                                    + noise_scale * random.sample::<f64, _>(StandardNormal);
                                velocity[i] = (position[i] - previous) / self.dt;
                            }
                        }
                    }
                });
            },
        )
        .is_some()
        {
            return;
        }
        let dt = self.dt;

        // Mobility = 1 / (m * gamma)
        let mobility = inverse_masses(masses) / self.friction;

        // Drift term: F / (m*gamma) * dt
        let drift = forces.clone() * mobility.clone() * dt;

        // Diffusion term: sqrt(2*kT*dt/(m*gamma)) * noise
        let kt = BOLTZMANN * self.temperature;
        let diff_scale = (mobility * (2.0 * kt * dt)).sqrt();

        let noise = self.random.tensor(masses);
        let diffusion = noise * diff_scale;

        let old_positions = positions.clone();
        *positions = positions.clone() + drift + diffusion;
        *velocities = ((positions.clone() - old_positions) / dt)
            .mask_where(fixed_particles(masses), velocities.clone());
    }

    fn step_constrained(
        &self,
        positions: &mut Tensor<B, 2>,
        velocities: &mut Tensor<B, 2>,
        forces: &Tensor<B, 2>,
        masses: &Tensor<B, 1>,
        constraints: &ConstraintSolver,
    ) -> Result<(), ConstraintError> {
        if constraints.is_empty() {
            self.step(positions, velocities, forces, masses);
            return Ok(());
        }
        let previous = positions.clone();
        self.step(positions, velocities, forces, masses);
        constraints.apply_positions(&previous, positions, self.constraint_tolerance)?;
        super::velocities_from_positions(positions, &previous, velocities, masses, self.dt, false);
        Ok(())
    }

    fn step_size(&self) -> f64 {
        self.dt
    }

    fn constraint_tolerance(&self) -> f64 {
        self.constraint_tolerance
    }
}
