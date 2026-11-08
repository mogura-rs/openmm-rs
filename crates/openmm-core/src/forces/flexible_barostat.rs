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

use super::barostat::{volume, BoxScaleProposal, BAR_NM3_TO_KJ_PER_MOL};

pub(crate) fn reduced_box(mut vectors: BoxVectors) -> Option<BoxVectors> {
    if !vectors.iter().flatten().all(|value| value.is_finite()) {
        return None;
    }
    for (axis, vector) in vectors.iter_mut().enumerate() {
        if vector[axis] == 0.0 {
            return None;
        }
        if vector[axis] < 0.0 {
            for value in vector {
                *value = -*value;
            }
        }
    }
    for (target, source, component) in [(2, 1, 1), (2, 0, 0), (1, 0, 0)] {
        let shift = (vectors[target][component] / vectors[source][component]).round();
        let basis = vectors[source];
        for (value, basis_value) in vectors[target].iter_mut().zip(basis) {
            *value -= shift * basis_value;
        }
    }
    vectors
        .iter()
        .flatten()
        .all(|value| value.is_finite())
        .then_some(vectors)
}

#[derive(Clone)]
pub struct MonteCarloFlexibleBarostat {
    metadata: super::ForceMetadata,
    default_pressure: f64,
    default_temperature: f64,
    pressure: f64,
    temperature: f64,
    frequency: Arc<AtomicU32>,
    scale_molecules_as_rigid: bool,
    random_number_seed: i32,
    random: Option<ChaCha12Rng>,
    step: u64,
    length_scale: f64,
    proposal_log_shape_ratio: f64,
    attempted: u64,
    accepted: u64,
}

impl MonteCarloFlexibleBarostat {
    super::force_metadata!();

    pub const PRESSURE: &'static str = "MonteCarloPressure";
    pub const TEMPERATURE: &'static str = "MonteCarloTemperature";

    pub fn new(default_pressure: f64, default_temperature: f64, frequency: u32) -> Self {
        assert!(default_pressure.is_finite(), "pressure must be finite");
        assert!(
            default_temperature.is_finite() && default_temperature >= 0.0,
            "temperature must be nonnegative and finite"
        );
        Self {
            metadata: super::ForceMetadata::new("MonteCarloFlexibleBarostat"),
            default_pressure,
            default_temperature,
            pressure: default_pressure,
            temperature: default_temperature,
            frequency: Arc::new(AtomicU32::new(frequency)),
            scale_molecules_as_rigid: true,
            random_number_seed: 0,
            random: None,
            step: 0,
            length_scale: 0.0,
            proposal_log_shape_ratio: 0.0,
            attempted: 0,
            accepted: 0,
        }
    }

    pub fn default_pressure(&self) -> f64 {
        self.default_pressure
    }
    pub fn set_default_pressure(&mut self, pressure: f64) {
        assert!(pressure.is_finite(), "pressure must be finite");
        self.default_pressure = pressure;
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
        let random = self.random.as_mut().expect("barostat must be initialized");
        let mut trial = *vectors;
        for (row, vector) in trial.iter_mut().enumerate() {
            for value in vector.iter_mut().take(row + 1) {
                *value += self.length_scale * 2.0 * (random.random::<f64>() - 0.5);
            }
        }
        let Some(trial) = reduced_box(trial) else {
            self.record_attempt(false, old_volume);
            return None;
        };
        let new_volume = volume(&trial);
        if !new_volume.is_finite() || new_volume <= 0.0 {
            self.record_attempt(false, old_volume);
            return None;
        }
        let scales: [f64; 3] = std::array::from_fn(|axis| trial[axis][axis] / vectors[axis][axis]);
        self.proposal_log_shape_ratio = 2.0 * scales[0].ln() + scales[1].ln();
        Some(BoxScaleProposal {
            box_vectors: trial,
            transform: [
                [scales[0], 0.0, 0.0],
                [0.0, scales[1], 0.0],
                [0.0, 0.0, scales[2]],
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
            - (scaled_particles as f64 - 2.0)
                * kt
                * (proposal.new_volume / proposal.old_volume).ln()
            - kt * self.proposal_log_shape_ratio;
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
        self.record_attempt(accepted, proposal.new_volume);
        accepted
    }

    fn record_attempt(&mut self, accepted: bool, previous_volume: f64) {
        self.attempted += 1;
        self.accepted += u64::from(accepted);
        if self.attempted >= 10 {
            let fraction = self.accepted as f64 / self.attempted as f64;
            if fraction < 0.25 {
                self.length_scale /= 1.1;
            } else if fraction > 0.75 {
                self.length_scale = (self.length_scale * 1.1).min(previous_volume.cbrt() * 0.3);
            } else {
                return;
            }
            self.attempted = 0;
            self.accepted = 0;
        }
    }
}

impl<B: Backend> Force<B> for MonteCarloFlexibleBarostat {
    super::force_metadata_impl!();

    fn requires_periodic_system(&self) -> bool {
        true
    }

    fn initialize(&mut self, _num_particles: usize, box_vectors: Option<&BoxVectors>) {
        let vectors = box_vectors.expect("a barostat requires a periodic box");
        self.pressure = self.default_pressure;
        self.temperature = self.default_temperature;
        self.length_scale = 0.01 * volume(vectors).cbrt();
        self.step = 0;
        self.attempted = 0;
        self.accepted = 0;
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
            (Self::TEMPERATURE.to_owned(), self.default_temperature),
        ])
    }

    fn set_global_parameter(&mut self, name: &str, value: f64) -> bool {
        match name {
            Self::PRESSURE => {
                assert!(value.is_finite(), "pressure must be finite");
                self.pressure = value;
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
    #[test]
    fn reduction_preserves_the_lattice_and_rejects_singular_boxes() {
        let original = [[-2.0, 0.0, 0.0], [3.1, -2.2, 0.0], [-4.3, 4.8, -3.4]];
        let reduced = reduced_box(original).unwrap();
        crate::system::validate_box_vectors(&reduced);
        assert_eq!(volume(&reduced), volume(&original).abs());
        for vector in original {
            let z = vector[2] / reduced[2][2];
            let y = (vector[1] - z * reduced[2][1]) / reduced[1][1];
            let x = (vector[0] - y * reduced[1][0] - z * reduced[2][0]) / reduced[0][0];
            for coefficient in [x, y, z] {
                assert!((coefficient - coefficient.round()).abs() < 1e-12);
            }
        }
        let mut singular = reduced;
        singular[1][1] = 0.0;
        assert!(reduced_box(singular).is_none());
        let mut invalid = reduced;
        invalid[2][0] = f64::INFINITY;
        assert!(reduced_box(invalid).is_none());
    }
}
