pub mod brownian;
pub mod compound;
pub mod dpd;
pub mod langevin;
pub mod nose_hoover;
pub mod qtb;
mod random;
pub mod variable;
pub mod verlet;

use crate::constraints::{ConstraintError, ConstraintSolver};
use burn::prelude::*;

pub(crate) fn inverse_masses<B: Backend>(masses: &Tensor<B, 1>) -> Tensor<B, 2> {
    let n = masses.dims()[0];
    masses
        .clone()
        .recip()
        .mask_fill(masses.clone().equal_elem(0.0), 0.0)
        .reshape([n, 1])
}

pub(crate) fn fixed_particles<B: Backend>(masses: &Tensor<B, 1>) -> Tensor<B, 2, Bool> {
    let n = masses.dims()[0];
    masses
        .clone()
        .equal_elem(0.0)
        .reshape([n, 1])
        .expand([n, 3])
}

pub type IntegratorComponent<B> = std::sync::Arc<std::sync::Mutex<Box<dyn Integrator<B>>>>;

#[derive(Clone, Copy)]
pub struct StepEnvironment {
    pub box_vectors: Option<[[f64; 3]; 3]>,
    pub max_step_size: f64,
}

/// Time-stepping algorithm. Operates on raw tensors (nm, nm/ps).
pub trait Integrator<B: Backend>: Send + Sync {
    /// Initialize internal state when attaching to a Context.
    fn initialize(&mut self) {}

    fn validate_system(&self, _system: &crate::System<B>) -> Result<(), ConstraintError> {
        Ok(())
    }

    fn initialize_system(&mut self, _system: &crate::System<B>) {
        self.initialize();
    }

    fn integration_force_groups(&self) -> u32 {
        u32::MAX
    }

    fn set_integration_force_groups(&mut self, _groups: u32) {
        panic!("this integrator does not support changing force groups");
    }

    /// Advance positions and velocities by one step given current forces and masses.
    /// positions:  [N, 3] nm
    /// velocities: [N, 3] nm/ps
    /// forces:     [N, 3] kJ/mol/nm
    /// masses:     [N]    Da
    fn step(
        &self,
        positions: &mut Tensor<B, 2>,
        velocities: &mut Tensor<B, 2>,
        forces: &Tensor<B, 2>,
        masses: &Tensor<B, 1>,
    );

    fn selected_integrator(&self) -> Option<usize> {
        None
    }

    fn select_integrator(&mut self, _index: usize) -> Result<(), ConstraintError> {
        Err(ConstraintError(
            "this integrator has no child selection".into(),
        ))
    }

    fn component(&self, _path: &[usize]) -> Option<IntegratorComponent<B>> {
        None
    }

    fn set_step_size(&mut self, _value: f64) {
        panic!("this integrator does not support changing step size");
    }

    fn set_constraint_tolerance(&mut self, _value: f64) {
        panic!("this integrator does not support changing constraint tolerance");
    }

    /// Step size in ps.
    fn step_size(&self) -> f64;

    /// Physical duration of a step when settings were captured at initialization.
    fn actual_step_size(&self) -> f64 {
        self.step_size()
    }

    fn constraint_tolerance(&self) -> f64 {
        1e-5
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
        self.step(positions, velocities, forces, masses);
        let proposed = positions.clone();
        constraints.apply_positions(&previous, positions, self.constraint_tolerance())?;
        velocities_from_positions(
            positions,
            &proposed,
            velocities,
            masses,
            self.step_size(),
            true,
        );
        Ok(())
    }

    fn supports_step_to(&self) -> bool {
        false
    }

    fn step_with_limit(
        &self,
        positions: &mut Tensor<B, 2>,
        velocities: &mut Tensor<B, 2>,
        forces: &Tensor<B, 2>,
        masses: &Tensor<B, 1>,
        constraints: &ConstraintSolver,
        _limit: f64,
    ) -> Result<(), ConstraintError> {
        self.step_constrained(positions, velocities, forces, masses, constraints)
    }

    #[allow(clippy::too_many_arguments)]
    fn step_in_environment(
        &self,
        positions: &mut Tensor<B, 2>,
        velocities: &mut Tensor<B, 2>,
        forces: &Tensor<B, 2>,
        masses: &Tensor<B, 1>,
        constraints: &ConstraintSolver,
        environment: StepEnvironment,
    ) -> Result<(), ConstraintError> {
        self.step_with_limit(
            positions,
            velocities,
            forces,
            masses,
            constraints,
            environment.max_step_size,
        )
    }

    /// Force correction used when reporting kinetic energy.
    fn kinetic_energy_time_offset(&self) -> f64 {
        self.velocity_time_offset()
    }

    /// Force correction used when initializing thermal velocities.
    fn velocity_time_offset(&self) -> f64 {
        0.0
    }
}

#[inline]
pub(crate) fn try_host_step<B: Backend, R>(
    positions: &mut Tensor<B, 2>,
    velocities: &mut Tensor<B, 2>,
    forces: &Tensor<B, 2>,
    masses: &Tensor<B, 1>,
    update: impl FnOnce(&mut [f64], &mut [f64], &[f64], &[f64]) -> R,
) -> Option<R> {
    crate::tensor::with_ndarray_f64(forces, |force| {
        crate::tensor::with_ndarray_f64(masses, |mass| {
            crate::tensor::with_mut_ndarray_f64(velocities, |velocity| {
                crate::tensor::with_mut_ndarray_f64(positions, |position| {
                    update(position, velocity, force, mass)
                })
            })
            .flatten()
        })
        .flatten()
    })
    .flatten()
}

pub(crate) fn velocities_from_positions<B: Backend>(
    positions: &Tensor<B, 2>,
    previous: &Tensor<B, 2>,
    velocities: &mut Tensor<B, 2>,
    masses: &Tensor<B, 1>,
    step_size: f64,
    add: bool,
) {
    let updated = crate::tensor::with_ndarray_f64(positions, |position| {
        crate::tensor::with_ndarray_f64(previous, |previous| {
            crate::tensor::with_ndarray_f64(masses, |mass| {
                crate::tensor::with_mut_ndarray_f64(velocities, |velocity| {
                    for (atom, &mass) in mass.iter().enumerate() {
                        if mass != 0.0 {
                            for axis in 0..3 {
                                let i = 3 * atom + axis;
                                let change = (position[i] - previous[i]) / step_size;
                                velocity[i] = if add { velocity[i] + change } else { change };
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
    if updated.is_none() {
        let change = (positions.clone() - previous.clone()) / step_size;
        *velocities = if add {
            velocities.clone() + change
        } else {
            change.mask_where(fixed_particles(masses), velocities.clone())
        };
    }
}
