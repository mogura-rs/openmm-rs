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
use super::{fixed_particles, inverse_masses, Integrator};
use burn::prelude::*;

pub struct VerletIntegrator {
    /// Step size in ps.
    dt: f64,
    constraint_tolerance: f64,
    integration_force_groups: u32,
}

impl VerletIntegrator {
    pub fn integration_force_groups(&self) -> u32 {
        self.integration_force_groups
    }
    pub fn set_integration_force_groups(&mut self, groups: u32) {
        self.integration_force_groups = groups;
    }

    pub fn new(step_size: f64) -> Self {
        assert!(
            step_size.is_finite() && step_size > 0.0,
            "step size must be positive and finite"
        );
        Self {
            dt: step_size,
            constraint_tolerance: 1e-5,
            integration_force_groups: u32::MAX,
        }
    }

    pub fn set_constraint_tolerance(&mut self, tolerance: f64) {
        assert!(
            tolerance.is_finite() && tolerance > 0.0 && tolerance < 1.0,
            "constraint tolerance must be between zero and one"
        );
        self.constraint_tolerance = tolerance;
    }

    pub fn set_step_size(&mut self, step_size: f64) {
        assert!(
            step_size.is_finite() && step_size > 0.0,
            "step size must be positive and finite"
        );
        self.dt = step_size;
    }
}

impl<B: Backend> Integrator<B> for VerletIntegrator {
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
        let dt = self.dt;
        let updated = crate::tensor::with_ndarray_f64(forces, |force| {
            crate::tensor::with_ndarray_f64(masses, |mass| {
                crate::tensor::with_mut_ndarray_f64(velocities, |velocity| {
                    crate::tensor::with_mut_ndarray_f64(positions, |position| {
                        for (atom, &mass) in mass.iter().enumerate() {
                            if mass != 0.0 {
                                let inverse_mass = mass.recip();
                                for axis in 0..3 {
                                    let index = 3 * atom + axis;
                                    velocity[index] += force[index] * inverse_mass * dt;
                                    position[index] += velocity[index] * dt;
                                }
                            }
                        }
                    })
                })
                .flatten()
            })
            .flatten()
        })
        .flatten();
        if updated.is_some() {
            return;
        }

        // a = F / m,  where masses are [N] and forces/velocities are [N, 3].
        // Expand masses to [N, 1] for broadcasting.
        let inv_mass = inverse_masses(masses);

        // Leapfrog: v(t+dt/2) = v(t-dt/2) + a(t) * dt
        let accel = forces.clone() * inv_mass.clone();
        *velocities = velocities.clone() + accel * dt;

        // x(t+dt) = x(t) + v(t+dt/2) * dt
        *positions =
            positions.clone() + (velocities.clone() * dt).mask_fill(fixed_particles(masses), 0.0);
    }

    fn step_size(&self) -> f64 {
        self.dt
    }

    fn constraint_tolerance(&self) -> f64 {
        self.constraint_tolerance
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
