// Snapshot of simulation state. Mirrors OpenMM::State.
use crate::units::BoxVectors;
use burn::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct StateFlags {
    pub positions: bool,
    pub velocities: bool,
    pub forces: bool,
    pub energy: bool,
    pub parameters: bool,
    pub parameter_derivatives: bool,
}

impl StateFlags {
    pub fn none() -> Self {
        Self {
            positions: false,
            velocities: false,
            forces: false,
            energy: false,
            parameters: false,
            parameter_derivatives: false,
        }
    }

    pub fn all() -> Self {
        Self {
            positions: true,
            velocities: true,
            forces: true,
            energy: true,
            parameters: true,
            parameter_derivatives: true,
        }
    }
}

impl Default for StateFlags {
    fn default() -> Self {
        Self::none()
    }
}

/// Immutable snapshot returned by Context::get_state().
pub struct State<B: Backend> {
    pub time: f64, // ps
    pub step_count: u64,
    pub parameters: Option<HashMap<String, f64>>,
    pub energy_parameter_derivatives: Option<HashMap<String, f64>>,
    pub periodic_box_vectors: Option<BoxVectors>,
    pub positions: Option<Tensor<B, 2>>,  // [N, 3] nm
    pub velocities: Option<Tensor<B, 2>>, // [N, 3] nm/ps
    pub forces: Option<Tensor<B, 2>>,     // [N, 3] kJ/mol/nm
    pub potential_energy: Option<f64>,    // kJ/mol
    pub kinetic_energy: Option<f64>,      // kJ/mol
}
