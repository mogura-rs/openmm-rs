// BrownianIntegrator: overdamped Langevin (no inertia).
//   x(t+dt) = x(t) + F/(m*gamma)*dt + sqrt(2*kT*dt/(m*gamma)) * noise
//
// Reference test: openmm/tests/TestBrownianIntegrator.h
use super::{fixed_particles, inverse_masses, Integrator};
use crate::integrators::langevin::{gaussian_noise, BOLTZMANN};
use burn::prelude::*;

pub struct BrownianIntegrator {
    /// Step size in ps.
    dt: f64,
    /// Temperature in K.
    temperature: f64,
    /// Friction coefficient in ps^-1.
    friction: f64,
    constraint_tolerance: f64,
}

impl BrownianIntegrator {
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
    fn step(
        &self,
        positions: &mut Tensor<B, 2>,
        velocities: &mut Tensor<B, 2>,
        forces: &Tensor<B, 2>,
        masses: &Tensor<B, 1>,
    ) {
        let dt = self.dt as f32;
        let n = masses.dims()[0];
        let device = masses.device();

        // Mobility = 1 / (m * gamma)
        let mobility = inverse_masses(masses) / self.friction as f32;

        // Drift term: F / (m*gamma) * dt
        let drift = forces.clone() * mobility.clone() * dt;

        // Diffusion term: sqrt(2*kT*dt/(m*gamma)) * noise
        let kt = BOLTZMANN * self.temperature;
        let diff_scale = (mobility * (2.0 * kt as f32 * dt)).sqrt();

        let noise = gaussian_noise::<B>([n, 3], &device);
        let diffusion = noise * diff_scale;

        let old_positions = positions.clone();
        *positions = positions.clone() + drift + diffusion;
        *velocities = ((positions.clone() - old_positions) / dt)
            .mask_where(fixed_particles(masses), velocities.clone());
    }

    fn step_size(&self) -> f64 {
        self.dt
    }

    fn constraint_tolerance(&self) -> f64 {
        self.constraint_tolerance
    }
}
