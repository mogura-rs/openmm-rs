use super::Force;
use crate::constraints::Constraint;
use crate::integrators::langevin::BOLTZMANN;
use burn::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use rand_distr::StandardNormal;
use std::collections::HashMap;

#[derive(Clone)]
pub struct AndersenThermostat {
    metadata: super::ForceMetadata,
    default_temperature: f64,
    default_collision_frequency: f64,
    temperature: f64,
    collision_frequency: f64,
    random_number_seed: i32,
    random: Option<ChaCha12Rng>,
    atoms: Vec<usize>,
    group_offsets: Vec<usize>,
}

impl AndersenThermostat {
    super::force_metadata!();

    pub const TEMPERATURE: &'static str = "AndersenTemperature";
    pub const COLLISION_FREQUENCY: &'static str = "AndersenCollisionFrequency";

    pub fn new(default_temperature: f64, default_collision_frequency: f64) -> Self {
        Self {
            metadata: super::ForceMetadata::new("AndersenThermostat"),
            default_temperature,
            default_collision_frequency,
            temperature: default_temperature,
            collision_frequency: default_collision_frequency,
            random_number_seed: 0,
            random: None,
            atoms: Vec::new(),
            group_offsets: Vec::new(),
        }
    }

    pub fn default_temperature(&self) -> f64 {
        self.default_temperature
    }

    pub fn set_default_temperature(&mut self, temperature: f64) {
        self.default_temperature = temperature;
    }

    pub fn default_collision_frequency(&self) -> f64 {
        self.default_collision_frequency
    }

    pub fn set_default_collision_frequency(&mut self, frequency: f64) {
        self.default_collision_frequency = frequency;
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

    fn prepare_groups(&mut self, num_particles: usize, constraints: &[Constraint]) {
        fn root(parent: &mut [usize], mut index: usize) -> usize {
            while parent[index] != index {
                parent[index] = parent[parent[index]];
                index = parent[index];
            }
            index
        }
        let mut parent: Vec<_> = (0..num_particles).collect();
        let mut size = vec![1; num_particles];
        for constraint in constraints {
            assert!(
                constraint.particle1 < num_particles && constraint.particle2 < num_particles,
                "constraint particle indices must be valid"
            );
            let mut a = root(&mut parent, constraint.particle1);
            let mut b = root(&mut parent, constraint.particle2);
            if a == b {
                continue;
            }
            if size[a] < size[b] {
                std::mem::swap(&mut a, &mut b);
            }
            parent[b] = a;
            size[a] += size[b];
        }
        let mut group_ids = vec![usize::MAX; num_particles];
        let mut groups = vec![0; num_particles];
        self.group_offsets = vec![0];
        for (atom, group) in groups.iter_mut().enumerate() {
            let representative = root(&mut parent, atom);
            if group_ids[representative] == usize::MAX {
                group_ids[representative] = self.group_offsets.len() - 1;
                self.group_offsets.push(0);
            }
            *group = group_ids[representative];
            self.group_offsets[*group + 1] += 1;
        }
        for i in 1..self.group_offsets.len() {
            self.group_offsets[i] += self.group_offsets[i - 1];
        }
        let mut cursors = self.group_offsets.clone();
        self.atoms.resize(num_particles, 0);
        for (atom, group) in groups.into_iter().enumerate() {
            self.atoms[cursors[group]] = atom;
            cursors[group] += 1;
        }
    }
}

impl<B: Backend> Force<B> for AndersenThermostat {
    super::force_metadata_impl!();

    fn initialize_constraints(&mut self, num_particles: usize, constraints: &[Constraint]) {
        self.prepare_groups(num_particles, constraints);
    }

    fn initialize(&mut self, num_particles: usize, _box_vectors: Option<&[[f64; 3]; 3]>) {
        assert!(
            self.default_temperature >= 0.0,
            "temperature must be nonnegative"
        );
        assert!(
            self.default_collision_frequency >= 0.0,
            "collision frequency must be nonnegative"
        );
        self.temperature = self.default_temperature;
        self.collision_frequency = self.default_collision_frequency;
        self.random = Some(if self.random_number_seed == 0 {
            ChaCha12Rng::from_os_rng()
        } else {
            ChaCha12Rng::seed_from_u64(self.random_number_seed as u32 as u64)
        });
        if self.atoms.len() != num_particles {
            self.prepare_groups(num_particles, &[]);
        }
    }

    fn global_parameters(&self) -> HashMap<String, f64> {
        HashMap::from([
            (Self::TEMPERATURE.to_owned(), self.default_temperature),
            (
                Self::COLLISION_FREQUENCY.to_owned(),
                self.default_collision_frequency,
            ),
        ])
    }

    fn set_global_parameter(&mut self, name: &str, value: f64) -> bool {
        match name {
            Self::TEMPERATURE => self.temperature = value,
            Self::COLLISION_FREQUENCY => self.collision_frequency = value,
            _ => return false,
        }
        true
    }

    fn update_context_velocities(
        &mut self,
        velocities: &mut Tensor<B, 2>,
        masses: &[f64],
        step_size: f64,
        _step_count: u64,
    ) {
        let probability = 1.0 - (-self.collision_frequency * step_size).exp();
        let random = self
            .random
            .as_mut()
            .expect("thermostat must be initialized");
        if probability <= 0.0 || probability.is_nan() {
            for _ in self.group_offsets.windows(2) {
                let _ = random.random::<f64>();
            }
            return;
        }
        let mut update = Some(|values: &mut [f64]| {
            for bounds in self.group_offsets.windows(2) {
                if random.random::<f64>() < probability {
                    for &atom in &self.atoms[bounds[0]..bounds[1]] {
                        if masses[atom] != 0.0 {
                            let scale = (BOLTZMANN * self.temperature / masses[atom]).sqrt();
                            for axis in 0..3 {
                                values[3 * atom + axis] =
                                    scale * random.sample::<f64, _>(StandardNormal);
                            }
                        }
                    }
                }
            }
        });
        if crate::tensor::with_mut_ndarray_f64(velocities, |values| update.take().unwrap()(values))
            .is_none()
        {
            let mut values = crate::tensor::to_f64_vec(velocities);
            update.take().unwrap()(&mut values);
            *velocities =
                crate::tensor::from_f64_vec(values, [masses.len(), 3], &velocities.device());
        }
    }

    fn calc_forces_and_energy_groups(
        &self,
        _positions: &Tensor<B, 2>,
        _box_vectors: Option<&[[f64; 3]; 3]>,
        _groups: u32,
    ) -> Option<(Tensor<B, 2>, f64)> {
        None
    }

    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        _box_vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64) {
        (Tensor::zeros(positions.shape(), &positions.device()), 0.0)
    }

    fn prepare_host_evaluation<'a>(
        &'a self,
        _positions: &'a Tensor<B, 2>,
        _box_vectors: Option<&'a [[f64; 3]; 3]>,
    ) -> Option<super::HostForceEvaluation<'a>> {
        Some(super::HostForceEvaluation::new(0.0, |_, _| {}))
    }
}
