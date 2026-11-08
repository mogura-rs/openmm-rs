use super::harmonic_bond::apply_minimum_image_ortho;
use super::periodic_torsion::{sub3_pos, torsion_geometry};
use super::Force;
use crate::tabulated::{Continuous2DFunction, TableError};
use burn::prelude::*;
use std::{f64::consts::TAU, sync::Arc};

#[derive(Clone)]
struct Map {
    size: usize,
    energy: Arc<Vec<f64>>,
    spline: Continuous2DFunction,
}

impl Map {
    fn new(size: usize, energy: Vec<f64>) -> Result<Self, TableError> {
        if size < 2 || size.checked_mul(size) != Some(energy.len()) {
            return Err(TableError(
                "CMAP requires a square map with at least two points per axis".into(),
            ));
        }
        let width = size + 1;
        let mut values = Vec::with_capacity(width * width);
        for j in 0..width {
            for i in 0..width {
                values.push(energy[i % size + size * (j % size)]);
            }
        }
        Ok(Self {
            size,
            energy: Arc::new(energy),
            spline: Continuous2DFunction::new([width, width], values, [[0.0, TAU]; 2], true)?,
        })
    }
}

#[derive(Debug, Clone)]
pub struct CMAPTorsionParameters {
    pub map: usize,
    pub particles: [usize; 8],
}

#[derive(Clone)]
pub struct CMAPTorsionForce {
    metadata: super::ForceMetadata,
    maps: Vec<Map>,
    torsions: Vec<CMAPTorsionParameters>,
    use_pbc: bool,
}

impl Default for CMAPTorsionForce {
    fn default() -> Self {
        Self::new()
    }
}

impl CMAPTorsionForce {
    super::force_metadata!();
    pub fn new() -> Self {
        Self {
            metadata: super::ForceMetadata::new("CMAPTorsionForce"),
            maps: Vec::new(),
            torsions: Vec::new(),
            use_pbc: false,
        }
    }
    pub fn add_map(&mut self, size: usize, energy: Vec<f64>) -> Result<usize, TableError> {
        self.maps.push(Map::new(size, energy)?);
        Ok(self.maps.len() - 1)
    }
    pub fn num_maps(&self) -> usize {
        self.maps.len()
    }
    pub fn map_parameters(&self, index: usize) -> (usize, &[f64]) {
        (self.maps[index].size, &self.maps[index].energy)
    }
    pub fn set_map_parameters(
        &mut self,
        index: usize,
        size: usize,
        energy: Vec<f64>,
    ) -> Result<(), TableError> {
        self.maps[index] = Map::new(size, energy)?;
        Ok(())
    }
    pub fn add_torsion(&mut self, map: usize, particles: [usize; 8]) -> usize {
        self.torsions.push(CMAPTorsionParameters { map, particles });
        self.torsions.len() - 1
    }
    pub fn num_torsions(&self) -> usize {
        self.torsions.len()
    }
    pub fn torsion_parameters(&self, index: usize) -> &CMAPTorsionParameters {
        &self.torsions[index]
    }
    pub fn set_torsion_parameters(&mut self, index: usize, map: usize, particles: [usize; 8]) {
        self.torsions[index] = CMAPTorsionParameters { map, particles };
    }
    pub fn uses_periodic_boundary_conditions(&self) -> bool {
        self.use_pbc
    }
    pub fn set_uses_periodic_boundary_conditions(&mut self, periodic: bool) {
        self.use_pbc = periodic;
    }

    pub fn update_parameters_from(&mut self, source: &Self) {
        assert_eq!(
            self.maps.len(),
            source.maps.len(),
            "CMAP map count must match"
        );
        assert_eq!(
            self.torsions.len(),
            source.torsions.len(),
            "CMAP torsion count must match"
        );
        for (old, new) in self.maps.iter().zip(&source.maps) {
            assert_eq!(old.size, new.size, "CMAP map size must match");
        }
        for (old, new) in self.torsions.iter().zip(&source.torsions) {
            assert_eq!(
                old.particles, new.particles,
                "CMAP torsion particles must match"
            );
            assert!(new.map < self.maps.len(), "CMAP map index out of range");
        }
        self.maps = source.maps.clone();
        self.torsions = source.torsions.clone();
    }
}

impl<B: Backend> Force<B> for CMAPTorsionForce {
    super::force_metadata_impl!();
    fn initialize(&mut self, num_particles: usize, _box_vectors: Option<&[[f64; 3]; 3]>) {
        for torsion in &self.torsions {
            assert!(torsion.map < self.maps.len(), "CMAP map index out of range");
            assert!(
                torsion.particles.iter().all(|&p| p < num_particles),
                "CMAP particle index out of range"
            );
        }
    }
    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.use_pbc
    }
    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        box_vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64) {
        let pos = positions
            .to_data()
            .convert::<f64>()
            .to_vec::<f64>()
            .unwrap();
        let mut forces = vec![0.0; pos.len()];
        let mut energy = 0.0;
        for torsion in &self.torsions {
            let mut angles = [0.0; 2];
            let mut directions = [[[0.0; 3]; 4]; 2];
            let mut valid = true;
            for angle in 0..2 {
                let particles = &torsion.particles[4 * angle..4 * angle + 4];
                let mut bonds: [[f64; 3]; 3] =
                    std::array::from_fn(|i| sub3_pos(&pos, particles[i + 1], particles[i]));
                if self.use_pbc {
                    if let Some(vectors) = box_vectors {
                        for bond in &mut bonds {
                            apply_minimum_image_ortho(bond, vectors);
                        }
                    }
                }
                if let Some((phi, gradient)) = torsion_geometry(&bonds[0], &bonds[1], &bonds[2]) {
                    angles[angle] = phi;
                    directions[angle] = gradient;
                } else {
                    valid = false;
                }
            }
            if !valid {
                continue;
            }
            let (value, derivatives) = self.maps[torsion.map].spline.evaluate(angles);
            energy += value;
            for angle in 0..2 {
                for (atom, direction) in directions[angle].iter().enumerate() {
                    let particle = torsion.particles[4 * angle + atom];
                    for axis in 0..3 {
                        forces[3 * particle + axis] += derivatives[angle] * direction[axis];
                    }
                }
            }
        }
        (
            crate::tensor::from_floats::<B, 1>(forces.as_slice(), &positions.device())
                .reshape(positions.shape()),
            energy,
        )
    }
}
