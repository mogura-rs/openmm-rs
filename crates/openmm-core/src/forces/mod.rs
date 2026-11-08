pub mod custom_angle;
pub mod custom_bond;
pub mod custom_external;
pub mod custom_nonbonded;
pub mod custom_torsion;
mod ewald;
pub mod harmonic_angle;
pub mod harmonic_bond;
mod neighbors;
pub mod nonbonded;
pub mod periodic_torsion;
mod pme;
pub mod rb_torsion;

use burn::prelude::*;
use std::any::Any;
use std::collections::HashMap;

pub trait AsAny {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: Any> AsAny for T {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub trait ForceClone<B: Backend> {
    fn clone_box(&self) -> Box<dyn Force<B>>;
}

impl<B: Backend, T> ForceClone<B> for T
where
    T: 'static + Force<B> + Clone,
{
    fn clone_box(&self) -> Box<dyn Force<B>> {
        Box::new(self.clone())
    }
}

/// Core trait implemented by every force field term.
///
/// Positions tensor: shape [N, 3], units nm.
/// Returned forces tensor: shape [N, 3], units kJ/mol/nm.
/// Returned energy: kJ/mol.
pub trait Force<B: Backend>: Send + Sync + AsAny + ForceClone<B> {
    fn initialize(&mut self, _num_particles: usize, _box_vectors: Option<&[[f64; 3]; 3]>) {}

    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        box_vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64);

    fn uses_periodic_boundary_conditions(&self) -> bool {
        false
    }

    fn global_parameters(&self) -> HashMap<String, f64> {
        HashMap::new()
    }

    fn set_global_parameter(&mut self, _name: &str, _value: f64) -> bool {
        false
    }

    fn energy_parameter_derivatives(
        &self,
        _positions: &Tensor<B, 2>,
        _box_vectors: Option<&[[f64; 3]; 3]>,
    ) -> HashMap<String, f64> {
        HashMap::new()
    }
}

impl<B: Backend> Clone for Box<dyn Force<B>> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
