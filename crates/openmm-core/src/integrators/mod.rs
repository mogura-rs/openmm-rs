pub mod brownian;
pub mod langevin;
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

/// Time-stepping algorithm. Operates on raw tensors (nm, nm/ps).
pub trait Integrator<B: Backend>: Send + Sync {
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

    /// Step size in ps.
    fn step_size(&self) -> f64;

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
        *velocities = velocities.clone() + (positions.clone() - proposed) / self.step_size();
        Ok(())
    }

    /// Offset between positions and stored velocities in ps.
    fn velocity_time_offset(&self) -> f64 {
        0.0
    }
}
