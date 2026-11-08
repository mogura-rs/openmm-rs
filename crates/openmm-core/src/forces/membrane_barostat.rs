use super::Force;
use crate::integrators::langevin::BOLTZMANN;
use crate::units::BoxVectors;
use burn::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};

use super::barostat::{scaled_box_axes, volume, BoxScaleProposal, BAR_NM3_TO_KJ_PER_MOL};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum MembraneXYMode {
    Isotropic = 0,
    Anisotropic = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum MembraneZMode {
    Free = 0,
    Fixed = 1,
    ConstantVolume = 2,
}

#[derive(Clone)]
pub struct MonteCarloMembraneBarostat {
    metadata: super::ForceMetadata,
    default_pressure: f64,
    default_temperature: f64,
    pressure: f64,
    default_surface_tension: f64,
    surface_tension: f64,
    xy_mode: Arc<AtomicU32>,
    z_mode: Arc<AtomicU32>,
    proposal_area_change: f64,
    proposal_axis: usize,
    temperature: f64,
    frequency: Arc<AtomicU32>,
    scale_molecules_as_rigid: bool,
    random_number_seed: i32,
    random: Option<ChaCha12Rng>,
    step: u64,
    volume_scale: [f64; 3],
    attempted: [u64; 3],
    accepted: [u64; 3],
}

impl MonteCarloMembraneBarostat {
    super::force_metadata!();

    pub const PRESSURE: &'static str = "MembraneMonteCarloPressure";
    pub const SURFACE_TENSION: &'static str = "MembraneMonteCarloSurfaceTension";
    pub const TEMPERATURE: &'static str = "MembraneMonteCarloTemperature";

    pub fn new(
        default_pressure: f64,
        default_surface_tension: f64,
        default_temperature: f64,
        xy_mode: MembraneXYMode,
        z_mode: MembraneZMode,
        frequency: u32,
    ) -> Self {
        assert!(default_pressure.is_finite(), "pressure must be finite");
        assert!(
            default_temperature.is_finite() && default_temperature >= 0.0,
            "temperature must be nonnegative and finite"
        );
        assert!(
            default_surface_tension.is_finite(),
            "surface tension must be finite"
        );
        Self {
            metadata: super::ForceMetadata::new("MonteCarloMembraneBarostat"),
            default_pressure,
            default_temperature,
            pressure: default_pressure,
            default_surface_tension,
            surface_tension: default_surface_tension,
            xy_mode: Arc::new(AtomicU32::new(xy_mode as u32)),
            z_mode: Arc::new(AtomicU32::new(z_mode as u32)),
            proposal_area_change: 0.0,
            proposal_axis: 0,
            temperature: default_temperature,
            frequency: Arc::new(AtomicU32::new(frequency)),
            scale_molecules_as_rigid: true,
            random_number_seed: 0,
            random: None,
            step: 0,
            volume_scale: [0.0; 3],
            attempted: [0; 3],
            accepted: [0; 3],
        }
    }

    pub fn default_pressure(&self) -> f64 {
        self.default_pressure
    }
    pub fn set_default_pressure(&mut self, pressure: f64) {
        assert!(pressure.is_finite(), "pressure must be finite");
        self.default_pressure = pressure;
    }
    pub fn default_surface_tension(&self) -> f64 {
        self.default_surface_tension
    }
    pub fn set_default_surface_tension(&mut self, tension: f64) {
        assert!(tension.is_finite(), "surface tension must be finite");
        self.default_surface_tension = tension;
    }
    pub fn xy_mode(&self) -> MembraneXYMode {
        match self.xy_mode.load(Ordering::Relaxed) {
            0 => MembraneXYMode::Isotropic,
            _ => MembraneXYMode::Anisotropic,
        }
    }
    pub fn set_xy_mode(&self, mode: MembraneXYMode) {
        self.xy_mode.store(mode as u32, Ordering::Relaxed);
    }
    pub fn z_mode(&self) -> MembraneZMode {
        match self.z_mode.load(Ordering::Relaxed) {
            0 => MembraneZMode::Free,
            1 => MembraneZMode::Fixed,
            _ => MembraneZMode::ConstantVolume,
        }
    }
    pub fn set_z_mode(&self, mode: MembraneZMode) {
        self.z_mode.store(mode as u32, Ordering::Relaxed);
    }
    pub fn default_temperature(&self) -> f64 {
        self.default_temperature
    }
    pub fn set_default_temperature(&mut self, temperature: f64) {
        assert!(
            temperature.is_finite() && temperature >= 0.0,
            "temperature must be nonnegative and finite"
        );
        self.default_temperature = temperature;
    }
    pub fn frequency(&self) -> u32 {
        self.frequency.load(Ordering::Relaxed)
    }
    /// Frequency changes are shared with existing Contexts and clones.
    pub fn set_frequency(&self, frequency: u32) {
        self.frequency.store(frequency, Ordering::Relaxed);
    }
    pub fn scale_molecules_as_rigid(&self) -> bool {
        self.scale_molecules_as_rigid
    }
    /// Change the mode captured when a new Context is initialized.
    pub fn set_scale_molecules_as_rigid(&mut self, rigid: bool) {
        self.scale_molecules_as_rigid = rigid;
    }
    pub fn random_number_seed(&self) -> i32 {
        self.random_number_seed
    }
    pub fn set_random_number_seed(&mut self, seed: i32) {
        self.random_number_seed = seed;
    }
    pub fn uses_periodic_boundary_conditions(&self) -> bool {
        false
    }

    fn propose(&mut self, vectors: &BoxVectors) -> Option<BoxScaleProposal> {
        self.step = self.step.saturating_add(1);
        let frequency = self.frequency();
        if frequency == 0 || self.step < u64::from(frequency) {
            return None;
        }
        self.step = 0;
        let old_volume = volume(vectors);
        let xy_mode = self.xy_mode();
        let z_mode = self.z_mode();
        let random = self.random.as_mut().expect("barostat must be initialized");
        let axis = loop {
            let candidate = (random.random::<f64>() * 3.0) as usize;
            if candidate < 2 {
                break if xy_mode == MembraneXYMode::Isotropic {
                    0
                } else {
                    candidate
                };
            }
            if z_mode == MembraneZMode::Free {
                break 2;
            }
        };
        self.proposal_axis = axis;
        let delta_volume = self.volume_scale[axis] * 2.0 * (random.random::<f64>() - 0.5);
        let mut new_volume = old_volume + delta_volume;
        if !new_volume.is_finite() || new_volume <= 0.0 {
            self.record_attempt(false, old_volume);
            return None;
        }
        let mut scale = [1.0; 3];
        if axis < 2 && xy_mode == MembraneXYMode::Isotropic {
            scale[0] = (new_volume / old_volume).sqrt();
            scale[1] = scale[0];
        } else {
            scale[axis] = new_volume / old_volume;
        }
        if z_mode == MembraneZMode::ConstantVolume {
            scale[2] = 1.0 / (scale[0] * scale[1]);
            new_volume = old_volume;
        }
        self.proposal_area_change = vectors[0][0] * vectors[1][1] * (scale[0] * scale[1] - 1.0);
        Some(BoxScaleProposal {
            box_vectors: scaled_box_axes(vectors, scale),
            transform: [
                [scale[0], 0.0, 0.0],
                [0.0, scale[1], 0.0],
                [0.0, 0.0, scale[2]],
            ],
            scale_molecules_as_rigid: self.scale_molecules_as_rigid,
            old_volume,
            new_volume,
        })
    }

    fn accept(
        &mut self,
        proposal: &BoxScaleProposal,
        energy_change: f64,
        scaled_particles: usize,
    ) -> bool {
        let kt = BOLTZMANN * self.temperature;
        let work = energy_change
            + self.pressure * BAR_NM3_TO_KJ_PER_MOL * (proposal.new_volume - proposal.old_volume)
            - self.surface_tension * BAR_NM3_TO_KJ_PER_MOL * self.proposal_area_change
            - scaled_particles as f64 * kt * (proposal.new_volume / proposal.old_volume).ln();
        let accepted = if work.is_nan() {
            false
        } else if work <= 0.0 {
            true
        } else {
            let uniform = self
                .random
                .as_mut()
                .expect("barostat must be initialized")
                .random::<f64>();
            kt > 0.0 && uniform <= (-work / kt).exp()
        };
        self.record_attempt(accepted, proposal.old_volume);
        accepted
    }

    fn record_attempt(&mut self, accepted: bool, previous_volume: f64) {
        let axis = self.proposal_axis;
        self.attempted[axis] += 1;
        self.accepted[axis] += u64::from(accepted);
        if self.attempted[axis] >= 10 {
            let fraction = self.accepted[axis] as f64 / self.attempted[axis] as f64;
            if fraction < 0.25 {
                self.volume_scale[axis] /= 1.1;
            } else if fraction > 0.75 {
                self.volume_scale[axis] =
                    (self.volume_scale[axis] * 1.1).min(previous_volume * 0.3);
            } else {
                return;
            }
            self.attempted[axis] = 0;
            self.accepted[axis] = 0;
        }
    }
}

impl<B: Backend> Force<B> for MonteCarloMembraneBarostat {
    super::force_metadata_impl!();

    fn requires_periodic_system(&self) -> bool {
        true
    }

    fn initialize(&mut self, _num_particles: usize, box_vectors: Option<&BoxVectors>) {
        let vectors = box_vectors.expect("a barostat requires a periodic box");
        self.pressure = self.default_pressure;
        self.surface_tension = self.default_surface_tension;
        self.temperature = self.default_temperature;
        self.volume_scale = [0.01 * volume(vectors); 3];
        self.step = 0;
        self.attempted = [0; 3];
        self.accepted = [0; 3];
        self.random = Some(if self.random_number_seed == 0 {
            ChaCha12Rng::from_os_rng()
        } else {
            ChaCha12Rng::seed_from_u64(self.random_number_seed as u32 as u64)
        });
    }

    fn propose_context_box_update(&mut self, box_vectors: &BoxVectors) -> Option<BoxScaleProposal> {
        self.propose(box_vectors)
    }

    fn accept_context_box_update(
        &mut self,
        proposal: &BoxScaleProposal,
        energy_change: f64,
        scaled_particles: usize,
    ) -> bool {
        self.accept(proposal, energy_change, scaled_particles)
    }

    fn global_parameters(&self) -> HashMap<String, f64> {
        HashMap::from([
            (Self::PRESSURE.to_owned(), self.default_pressure),
            (
                Self::SURFACE_TENSION.to_owned(),
                self.default_surface_tension,
            ),
            (Self::TEMPERATURE.to_owned(), self.default_temperature),
        ])
    }

    fn set_global_parameter(&mut self, name: &str, value: f64) -> bool {
        match name {
            Self::PRESSURE => {
                assert!(value.is_finite(), "pressure must be finite");
                self.pressure = value;
            }
            Self::SURFACE_TENSION => {
                assert!(value.is_finite(), "surface tension must be finite");
                self.surface_tension = value;
            }
            Self::TEMPERATURE => {
                assert!(
                    value.is_finite() && value >= 0.0,
                    "temperature must be nonnegative and finite"
                );
                self.temperature = value;
            }
            _ => return false,
        }
        true
    }

    fn calc_forces_and_energy_groups(
        &self,
        _positions: &Tensor<B, 2>,
        _box_vectors: Option<&BoxVectors>,
        _groups: u32,
    ) -> Option<(Tensor<B, 2>, f64)> {
        None
    }

    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        _box_vectors: Option<&BoxVectors>,
    ) -> (Tensor<B, 2>, f64) {
        (
            crate::tensor::zeros(positions.dims(), &positions.device()),
            0.0,
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn invalid_volume_proposals_are_rejected_without_nonfinite_box_vectors() {
        let mut barostat = MonteCarloMembraneBarostat::new(
            1.0,
            7.0,
            300.0,
            MembraneXYMode::Isotropic,
            MembraneZMode::ConstantVolume,
            1,
        );
        barostat.set_random_number_seed(13);
        <MonteCarloMembraneBarostat as Force<NdArray<f64>>>::initialize(
            &mut barostat,
            1,
            Some(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        );
        let small_box = [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]];
        let mut rejected = 0;
        for _ in 0..100 {
            if let Some(proposal) = barostat.propose(&small_box) {
                assert!(proposal.new_volume > 0.0);
                assert!(proposal.box_vectors.iter().flatten().all(|v| v.is_finite()));
            } else {
                rejected += 1;
            }
        }
        assert!(rejected > 10);
    }
}
