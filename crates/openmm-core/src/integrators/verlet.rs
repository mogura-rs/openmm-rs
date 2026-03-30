// VerletIntegrator: velocity Verlet algorithm.
// Assumes forces are already computed for the current positions.
// The step() call here implements the "leapfrog" variant:
//   v(t+dt/2) = v(t-dt/2) + F(t)/m * dt
//   x(t+dt)   = x(t) + v(t+dt/2) * dt
//
// Context::step() provides forces at x(t); the half-step velocity is stored
// in the velocities tensor between steps.
//
// Reference test: openmm/tests/TestVerletIntegrator.h
use super::Integrator;
use burn::prelude::*;

pub struct VerletIntegrator {
    /// Step size in ps.
    dt: f64,
}

impl VerletIntegrator {
    pub fn new(step_size: f64) -> Self {
        Self { dt: step_size }
    }
}

impl<B: Backend> Integrator<B> for VerletIntegrator {
    fn step(
        &self,
        positions: &mut Tensor<B, 2>,
        velocities: &mut Tensor<B, 2>,
        forces: &Tensor<B, 2>,
        masses: &Tensor<B, 1>,
    ) {
        let dt = self.dt as f32;

        // a = F / m,  where masses are [N] and forces/velocities are [N, 3].
        // Expand masses to [N, 1] for broadcasting.
        let n = masses.dims()[0];
        let inv_mass = masses.clone().recip().reshape([n, 1]);

        // Leapfrog: v(t+dt/2) = v(t-dt/2) + a(t) * dt
        let accel = forces.clone() * inv_mass.clone();
        *velocities = velocities.clone() + accel * dt;

        // x(t+dt) = x(t) + v(t+dt/2) * dt
        *positions = positions.clone() + velocities.clone() * dt;
    }

    fn step_size(&self) -> f64 {
        self.dt
    }

    fn velocity_time_offset(&self) -> f64 {
        self.dt * 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use burn::backend::NdArray;

    type B = NdArray;

    // A free particle (no force) should move at constant velocity.
    #[test]
    fn test_free_particle() {
        let integrator = VerletIntegrator::new(0.001); // 1 fs

        let device = Default::default();
        let mut pos = Tensor::<B, 1>::from_floats([0.0f32, 0.0, 0.0], &device).reshape([1, 3]);
        let mut vel = Tensor::<B, 1>::from_floats([1.0f32, 0.0, 0.0], &device).reshape([1, 3]);
        let forces = Tensor::<B, 1>::from_floats([0.0f32, 0.0, 0.0], &device).reshape([1, 3]);
        let masses = Tensor::<B, 1>::from_floats([1.0f32], &device);

        integrator.step(&mut pos, &mut vel, &forces, &masses);

        let pos_data: Vec<f32> = pos.to_data().to_vec::<f32>().unwrap();
        assert_relative_eq!(pos_data[0], 0.001_f32, epsilon = 1e-5);
        assert_relative_eq!(pos_data[1], 0.0, epsilon = 1e-5);
    }
}
