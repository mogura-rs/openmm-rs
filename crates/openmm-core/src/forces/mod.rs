pub mod alignment;
pub mod andersen;
pub mod anisotropic_barostat;
pub mod barostat;
pub mod cm_motion;
pub mod cmap_torsion;
pub mod custom_angle;
pub mod custom_bond;
pub mod custom_centroid;
pub mod custom_compound;
pub mod custom_cv;
mod custom_evaluator;
pub mod custom_external;
pub mod custom_hbond;
pub mod custom_nonbonded;
pub mod custom_torsion;
pub mod custom_volume;
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

#[derive(Clone)]
pub(crate) struct ForceMetadata {
    group: u32,
    name: String,
}

impl ForceMetadata {
    fn new(name: &str) -> Self {
        Self {
            group: 0,
            name: name.to_owned(),
        }
    }
}

macro_rules! force_metadata {
    () => {
        pub fn force_group(&self) -> u32 {
            self.metadata.group
        }
        pub fn set_force_group(&mut self, group: u32) {
            assert!(group < 32, "force group must be between zero and 31");
            self.metadata.group = group;
        }
        pub fn name(&self) -> &str {
            &self.metadata.name
        }
        pub fn set_name(&mut self, name: &str) {
            self.metadata.name = name.to_owned();
        }
    };
}
pub(crate) use force_metadata;

macro_rules! force_metadata_impl {
    () => {
        fn force_group(&self) -> u32 {
            self.metadata.group
        }
        fn name(&self) -> &str {
            &self.metadata.name
        }
    };
}
pub(crate) use force_metadata_impl;

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

type HostAccumulator<'a> = Box<dyn FnOnce(f64, &mut [f64]) + 'a>;

/// An energy value and deferred addition of its scaled host forces.
pub struct HostForceEvaluation<'a> {
    pub energy: f64,
    accumulate: HostAccumulator<'a>,
}

impl<'a> HostForceEvaluation<'a> {
    pub fn new(energy: f64, accumulate: impl FnOnce(f64, &mut [f64]) + 'a) -> Self {
        Self {
            energy,
            accumulate: Box::new(accumulate),
        }
    }

    pub fn accumulate(self, coefficient: f64, forces: &mut [f64]) {
        (self.accumulate)(coefficient, forces);
    }
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
    fn requires_periodic_system(&self) -> bool {
        false
    }

    fn propose_context_box_update(
        &mut self,
        _box_vectors: &crate::units::BoxVectors,
    ) -> Option<barostat::BoxScaleProposal> {
        None
    }

    fn accept_context_box_update(
        &mut self,
        _proposal: &barostat::BoxScaleProposal,
        _energy_change: f64,
        _scaled_particles: usize,
    ) -> bool {
        false
    }

    fn initialize_masses(&mut self, _masses: &[f64]) {}

    fn initialize_constraints(
        &mut self,
        _num_particles: usize,
        _constraints: &[crate::Constraint],
    ) {
    }

    fn update_context_velocities(
        &mut self,
        _velocities: &mut Tensor<B, 2>,
        _masses: &[f64],
        _step_size: f64,
        _step_count: u64,
    ) {
    }

    fn initialize(&mut self, _num_particles: usize, _box_vectors: Option<&[[f64; 3]; 3]>) {}

    fn force_group(&self) -> u32 {
        0
    }

    fn name(&self) -> &str {
        "Force"
    }

    fn bonded_particles(&self) -> Vec<(usize, usize)> {
        Vec::new()
    }

    fn calc_forces_and_energy_groups(
        &self,
        positions: &Tensor<B, 2>,
        box_vectors: Option<&[[f64; 3]; 3]>,
        groups: u32,
    ) -> Option<(Tensor<B, 2>, f64)> {
        (groups & (1 << self.force_group()) != 0)
            .then(|| self.calc_forces_and_energy(positions, box_vectors))
    }

    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        box_vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64);

    /// Optionally evaluate energy and prepare direct host force accumulation.
    /// The caller may drop the result if it needs only energy.
    fn prepare_host_evaluation<'a>(
        &'a self,
        _positions: &'a Tensor<B, 2>,
        _box_vectors: Option<&'a [[f64; 3]; 3]>,
    ) -> Option<HostForceEvaluation<'a>> {
        None
    }

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
