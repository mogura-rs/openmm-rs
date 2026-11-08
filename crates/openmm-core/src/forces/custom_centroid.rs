use super::{custom_compound::CustomCompoundBondForce, Force};
use burn::prelude::*;
use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
};

#[derive(Clone)]
pub struct CustomCentroidBondForce {
    inner: CustomCompoundBondForce,
    groups: Vec<Group>,
    normalized_weights: Vec<Vec<f64>>,
}

#[derive(Clone)]
struct Group {
    particles: Vec<usize>,
    weights: Vec<f64>,
}

impl Deref for CustomCentroidBondForce {
    type Target = CustomCompoundBondForce;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl DerefMut for CustomCentroidBondForce {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl CustomCentroidBondForce {
    pub fn new(
        groups_per_bond: usize,
        energy: &str,
    ) -> Result<Self, crate::expression::ParseError> {
        let mut inner = CustomCompoundBondForce::new_with_prefix(groups_per_bond, energy, 'g')?;
        inner.set_name("CustomCentroidBondForce");
        Ok(Self {
            inner,
            groups: Vec::new(),
            normalized_weights: Vec::new(),
        })
    }
    pub fn num_groups_per_bond(&self) -> usize {
        self.inner.num_particles_per_bond()
    }
    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }
    pub fn add_group(&mut self, particles: Vec<usize>, weights: Vec<f64>) -> usize {
        assert!(
            weights.is_empty() || weights.len() == particles.len(),
            "group weight count must match particles"
        );
        self.groups.push(Group { particles, weights });
        self.normalized_weights.clear();
        self.groups.len() - 1
    }
    pub fn group_parameters(&self, index: usize) -> (&[usize], &[f64]) {
        (&self.groups[index].particles, &self.groups[index].weights)
    }
    pub fn set_group_parameters(&mut self, index: usize, particles: Vec<usize>, weights: Vec<f64>) {
        assert!(
            weights.is_empty() || weights.len() == particles.len(),
            "group weight count must match particles"
        );
        self.groups[index] = Group { particles, weights };
        self.normalized_weights.clear();
    }
    pub fn update_parameters_from(&mut self, source: &Self) {
        self.inner.update_parameters_from(&source.inner);
    }
    fn evaluate<B: Backend>(
        &self,
        positions: &Tensor<B, 2>,
        vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64, HashMap<String, f64>) {
        assert_eq!(
            self.groups.len(),
            self.normalized_weights.len(),
            "centroid weights have not been initialized"
        );
        let n = positions.dims()[0];
        let pos = positions
            .to_data()
            .convert::<f64>()
            .to_vec::<f64>()
            .unwrap();
        let mut centers = vec![0.0; 3 * self.groups.len()];
        for (index, (group, weights)) in
            self.groups.iter().zip(&self.normalized_weights).enumerate()
        {
            for (&particle, &weight) in group.particles.iter().zip(weights) {
                for axis in 0..3 {
                    centers[3 * index + axis] += weight * pos[3 * particle + axis];
                }
            }
        }
        let (group_forces, energy, derivatives) =
            self.inner
                .evaluate_host(&centers, self.groups.len(), vectors);
        let mut forces = vec![0.0; 3 * n];
        for (index, (group, weights)) in
            self.groups.iter().zip(&self.normalized_weights).enumerate()
        {
            for (&particle, &weight) in group.particles.iter().zip(weights) {
                for axis in 0..3 {
                    forces[3 * particle + axis] += weight * group_forces[3 * index + axis];
                }
            }
        }
        (
            crate::tensor::from_floats::<B, 1>(forces.as_slice(), &positions.device())
                .reshape([n, 3]),
            energy,
            derivatives,
        )
    }
}

impl<B: Backend> Force<B> for CustomCentroidBondForce {
    fn bonded_particles(&self) -> Vec<(usize, usize)> {
        let mut bonds = Vec::new();
        for index in 0..self.inner.num_bonds() {
            let (groups, _) = self.inner.bond_parameters(index);
            if groups.len() < 2 {
                continue;
            }
            let mut particles = groups
                .iter()
                .flat_map(|&group| self.groups[group].particles.iter().copied());
            if let Some(anchor) = particles.next() {
                bonds.extend(
                    particles
                        .filter(|&particle| particle != anchor)
                        .map(|particle| (anchor, particle)),
                );
            }
        }
        bonds
    }
    fn initialize_masses(&mut self, masses: &[f64]) {
        self.normalized_weights = self
            .groups
            .iter()
            .map(|group| {
                assert!(
                    group.particles.iter().all(|&p| p < masses.len()),
                    "centroid particle index out of range"
                );
                let mut weights = if group.weights.is_empty() {
                    group
                        .particles
                        .iter()
                        .map(|&p| masses[p])
                        .collect::<Vec<_>>()
                } else {
                    group.weights.clone()
                };
                let total: f64 = weights.iter().sum();
                assert!(total != 0.0, "centroid weights sum to zero");
                for value in &mut weights {
                    *value /= total;
                }
                weights
            })
            .collect();
    }
    fn initialize(&mut self, _num_particles: usize, vectors: Option<&[[f64; 3]; 3]>) {
        <CustomCompoundBondForce as Force<B>>::initialize(
            &mut self.inner,
            self.groups.len(),
            vectors,
        );
    }
    fn force_group(&self) -> u32 {
        self.inner.force_group()
    }
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.inner.uses_periodic_boundary_conditions()
    }
    fn global_parameters(&self) -> HashMap<String, f64> {
        <CustomCompoundBondForce as Force<B>>::global_parameters(&self.inner)
    }
    fn set_global_parameter(&mut self, name: &str, value: f64) -> bool {
        <CustomCompoundBondForce as Force<B>>::set_global_parameter(&mut self.inner, name, value)
    }
    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64) {
        let (forces, energy, _) = self.evaluate(positions, vectors);
        (forces, energy)
    }
    fn energy_parameter_derivatives(
        &self,
        positions: &Tensor<B, 2>,
        vectors: Option<&[[f64; 3]; 3]>,
    ) -> HashMap<String, f64> {
        self.evaluate(positions, vectors).2
    }
}
