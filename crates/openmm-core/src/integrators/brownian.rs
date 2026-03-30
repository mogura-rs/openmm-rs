// BrownianIntegrator: overdamped Langevin (no inertia).
//   x(t+dt) = x(t) + F/(m*gamma)*dt + sqrt(2*kT*dt/(m*gamma)) * noise
//
// Reference test: openmm/tests/TestBrownianIntegrator.h
use super::Integrator;
use crate::integrators::langevin::{gaussian_noise, BOLTZMANN};
use burn::prelude::*;

pub struct BrownianIntegrator {
    /// Step size in ps.
    dt: f64,
    /// Temperature in K.
    temperature: f64,
    /// Friction coefficient in ps^-1.
    friction: f64,
}

impl BrownianIntegrator {
    pub fn new(temperature: f64, friction: f64, step_size: f64) -> Self {
        Self {
            dt: step_size,
            temperature,
            friction,
        }
    }

    pub fn temperature(&self) -> f64 {
        self.temperature
    }
    pub fn friction(&self) -> f64 {
        self.friction
    }
    pub fn set_temperature(&mut self, t: f64) {
        self.temperature = t;
    }
    pub fn set_friction(&mut self, f: f64) {
        self.friction = f;
    }
    pub fn set_step_size(&mut self, dt: f64) {
        self.dt = dt;
    }
}

impl<B: Backend> Integrator<B> for BrownianIntegrator {
    fn step(
        &self,
        positions: &mut Tensor<B, 2>,
        _velocities: &mut Tensor<B, 2>,
        forces: &Tensor<B, 2>,
        masses: &Tensor<B, 1>,
    ) {
        let dt = self.dt as f32;
        let n = masses.dims()[0];
        let device = masses.device();

        // Mobility = 1 / (m * gamma)
        let mobility = (masses.clone() * (self.friction as f32))
            .recip()
            .reshape([n, 1]);

        // Drift term: F / (m*gamma) * dt
        let drift = forces.clone() * mobility.clone() * dt;

        // Diffusion term: sqrt(2*kT*dt/(m*gamma)) * noise
        let kt = BOLTZMANN * self.temperature;
        let diff_scale = ((masses.clone() * (self.friction as f32)).recip()
            * (2.0 * kt as f32 * dt))
            .sqrt()
            .reshape([n, 1]);

        let noise = gaussian_noise::<B>([n, 3], &device);
        let diffusion = noise * diff_scale;

        *positions = positions.clone() + drift + diffusion;
    }

    fn step_size(&self) -> f64 {
        self.dt
    }
}
