// LangevinMiddleIntegrator: BAOAB splitting scheme (O(dt^3) accuracy).
//   B: v += F/m * dt/2
//   A: x += v * dt/2
//   O: Ornstein-Uhlenbeck (thermostat)  v = a*v + b*noise
//   A: x += v * dt/2
//   B: v += F/m * dt/2
//
// Thermostat coefficients:
//   a = exp(-gamma * dt)
//   b = sqrt(kT/m * (1 - a^2))
//
// Reference test: openmm/tests/TestLangevinMiddleIntegrator.h
// Reference impl: openmm/platforms/reference/src/SimTKReference/ReferenceLangevinMiddleDynamics.cpp
use super::Integrator;
use burn::prelude::*;

/// Boltzmann constant in kJ/mol/K.
pub const BOLTZMANN: f64 = 0.008314462618;

pub struct LangevinMiddleIntegrator {
    /// Step size in ps.
    dt: f64,
    /// Temperature in K.
    temperature: f64,
    /// Friction coefficient in ps^-1.
    friction: f64,
}

impl LangevinMiddleIntegrator {
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

impl<B: Backend> Integrator<B> for LangevinMiddleIntegrator {
    fn step(
        &self,
        positions: &mut Tensor<B, 2>,
        velocities: &mut Tensor<B, 2>,
        forces: &Tensor<B, 2>,
        masses: &Tensor<B, 1>,
    ) {
        let dt = self.dt as f32;
        let half_dt = dt * 0.5;
        let n = masses.dims()[0];
        let device = masses.device();

        let inv_mass = masses.clone().recip().reshape([n, 1]);

        // --- B: v += F/m * dt/2 ---
        let accel = forces.clone() * inv_mass.clone();
        *velocities = velocities.clone() + accel.clone() * half_dt;

        // --- A: x += v * dt/2 ---
        *positions = positions.clone() + velocities.clone() * half_dt;

        // --- O: Ornstein-Uhlenbeck thermostat ---
        let a = (-self.friction * self.dt).exp() as f32;
        let kt = (BOLTZMANN * self.temperature) as f32;

        // b[i] = sqrt(kT/m[i] * (1 - a^2)), broadcast over xyz.
        let b_per_particle = ((masses.clone().recip() * kt * (1.0 - a * a)).sqrt()).reshape([n, 1]);

        // Gaussian noise: Box-Muller via two uniform samples.
        let noise = gaussian_noise::<B>([n, 3], &device);
        *velocities = velocities.clone() * a + noise * b_per_particle;

        // --- A: x += v * dt/2 ---
        *positions = positions.clone() + velocities.clone() * half_dt;

        // --- B: v += F/m * dt/2 ---
        *velocities = velocities.clone() + accel * half_dt;
    }

    fn step_size(&self) -> f64 {
        self.dt
    }
}

/// Box-Muller transform to generate standard normal samples as a [shape] tensor.
pub fn gaussian_noise<B: Backend>(shape: [usize; 2], device: &B::Device) -> Tensor<B, 2> {
    Tensor::random(shape, burn::tensor::Distribution::Normal(0.0, 1.0), device)
}

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
