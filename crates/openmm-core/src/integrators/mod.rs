pub mod brownian;
pub mod langevin;
pub mod verlet;

use burn::prelude::*;

/// Time-stepping algorithm. Operates on raw tensors (nm, nm/ps).
pub trait Integrator<B: Backend>: Send + Sync {
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

    /// Offset between positions and stored velocities in ps.
    fn velocity_time_offset(&self) -> f64 {
        0.0
    }
}
