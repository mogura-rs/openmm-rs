// CustomNonbondedForce: user-defined pairwise energy as a function of r
// and per-particle parameters.
// Reference test: openmm/tests/TestCustomNonbondedForce.h
use super::custom_evaluator::CustomEvaluator;
use super::neighbors::NeighborCells;
use super::Force;
use crate::expression::Expression;
use crate::forces::harmonic_bond::apply_minimum_image_ortho;
use burn::prelude::*;
use std::collections::HashMap;

#[derive(Clone)]
pub struct CustomNonbondedForce {
    metadata: super::ForceMetadata,
    energy_expr: Expression,
    per_particle_params: Vec<String>,
    paired_parameter_names: Vec<[String; 2]>,
    particles: Vec<Vec<f64>>,
    exclusions: Vec<(usize, usize)>,
    global_param_names: Vec<String>,
    global_params: HashMap<String, f64>,
    energy_param_deriv_names: Vec<String>,
    use_pbc: bool,
    cutoff_distance: Option<f64>,
    use_neighbor_cells: bool,
    neighbor_cells: NeighborCells,
}

impl CustomNonbondedForce {
    super::force_metadata!();
    pub fn new(energy: &str) -> Result<Self, crate::expression::ParseError> {
        Ok(Self {
            metadata: super::ForceMetadata::new("CustomNonbondedForce"),
            energy_expr: Expression::parse(energy)?,
            per_particle_params: Vec::new(),
            paired_parameter_names: Vec::new(),
            particles: Vec::new(),
            exclusions: Vec::new(),
            global_param_names: Vec::new(),
            global_params: HashMap::new(),
            energy_param_deriv_names: Vec::new(),
            use_pbc: false,
            cutoff_distance: None,
            use_neighbor_cells: true,
            neighbor_cells: NeighborCells::default(),
        })
    }

    pub fn add_per_particle_parameter(&mut self, name: &str) -> usize {
        self.per_particle_params.push(name.to_string());
        self.paired_parameter_names
            .push([format!("{name}1"), format!("{name}2")]);
        self.per_particle_params.len() - 1
    }

    pub fn num_per_particle_parameters(&self) -> usize {
        self.per_particle_params.len()
    }

    pub fn per_particle_parameter_name(&self, index: usize) -> &str {
        &self.per_particle_params[index]
    }

    pub fn add_global_parameter(&mut self, name: &str, value: f64) {
        if !self.global_params.contains_key(name) {
            self.global_param_names.push(name.to_string());
        }
        self.global_params.insert(name.to_string(), value);
    }

    pub fn num_global_parameters(&self) -> usize {
        self.global_param_names.len()
    }

    pub fn global_parameter_name(&self, index: usize) -> &str {
        &self.global_param_names[index]
    }

    pub fn global_parameter_default_value(&self, index: usize) -> f64 {
        let name = &self.global_param_names[index];
        self.global_params[name]
    }

    pub fn set_global_parameter_default_value(&mut self, index: usize, value: f64) {
        let name = self.global_param_names[index].clone();
        *self
            .global_params
            .get_mut(&name)
            .expect("unknown global parameter") = value;
    }

    pub fn add_energy_parameter_derivative(&mut self, name: &str) {
        assert!(
            self.global_params.contains_key(name),
            "unknown global parameter"
        );
        if !self
            .energy_param_deriv_names
            .iter()
            .any(|existing| existing == name)
        {
            self.energy_param_deriv_names.push(name.to_string());
        }
    }

    pub fn num_energy_parameter_derivatives(&self) -> usize {
        self.energy_param_deriv_names.len()
    }

    pub fn energy_parameter_derivative_name(&self, index: usize) -> &str {
        &self.energy_param_deriv_names[index]
    }

    /// Add a particle. Returns particle index.
    pub fn add_particle(&mut self, params: Vec<f64>) -> usize {
        assert_eq!(params.len(), self.per_particle_params.len());
        self.particles.push(params);
        self.particles.len() - 1
    }

    pub fn num_particles(&self) -> usize {
        self.particles.len()
    }

    pub fn particle_parameters(&self, index: usize) -> &[f64] {
        &self.particles[index]
    }

    pub fn set_particle_parameters(&mut self, index: usize, params: Vec<f64>) {
        assert_eq!(params.len(), self.per_particle_params.len());
        self.particles[index] = params;
    }

    pub fn add_exclusion(&mut self, p1: usize, p2: usize) {
        let key = if p1 < p2 { (p1, p2) } else { (p2, p1) };
        self.exclusions.push(key);
    }

    pub fn num_exclusions(&self) -> usize {
        self.exclusions.len()
    }

    pub fn exclusion_particles(&self, index: usize) -> (usize, usize) {
        self.exclusions[index]
    }

    pub fn set_exclusion_particles(&mut self, index: usize, p1: usize, p2: usize) {
        self.exclusions[index] = if p1 < p2 { (p1, p2) } else { (p2, p1) };
    }

    pub fn set_uses_periodic_boundary_conditions(&mut self, use_pbc: bool) {
        self.use_pbc = use_pbc;
    }

    pub fn uses_periodic_boundary_conditions(&self) -> bool {
        self.use_pbc
    }

    pub fn set_use_neighbor_cells(&mut self, enabled: bool) {
        self.use_neighbor_cells = enabled;
    }

    pub fn set_cutoff_distance(&mut self, cutoff: f64) {
        assert!(
            cutoff.is_finite() && cutoff > 0.0,
            "cutoff distance must be positive and finite"
        );
        self.cutoff_distance = Some(cutoff);
    }

    pub fn cutoff_distance(&self) -> Option<f64> {
        self.cutoff_distance
    }

    pub fn update_parameters_from(&mut self, source: &Self) {
        assert_eq!(
            self.per_particle_params, source.per_particle_params,
            "per-particle parameter definitions must match"
        );
        assert_eq!(
            self.particles.len(),
            source.particles.len(),
            "particle count must match"
        );
        assert_eq!(
            self.exclusions.len(),
            source.exclusions.len(),
            "exclusion count must match"
        );
        self.particles = source.particles.clone();
        self.exclusions = source.exclusions.clone();
    }
}

impl CustomNonbondedForce {
    fn evaluate<B: Backend>(
        &self,
        positions: &Tensor<B, 2>,
        box_vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64, HashMap<String, f64>) {
        let device = positions.device();
        let n = positions.dims()[0];
        let pos: Vec<f64> = positions
            .to_data()
            .convert::<f64>()
            .to_vec::<f64>()
            .unwrap();

        let mut forces = vec![0.0f64; n * 3];
        let mut energy = 0.0f64;
        let mut evaluator = CustomEvaluator::new(
            &self.energy_expr,
            &self.global_params,
            self.paired_parameter_names
                .iter()
                .flatten()
                .map(String::as_str),
            &["r"],
            &self.energy_param_deriv_names,
        );

        let excluded: std::collections::HashSet<(usize, usize)> =
            self.exclusions.iter().cloned().collect();
        let cutoff2 = self.cutoff_distance.map(|c| c * c);
        let mut pair = |i: usize, j: usize| {
            let key = (i, j);
            if excluded.contains(&key) {
                return;
            }

            let mut delta = [
                pos[j * 3] - pos[i * 3],
                pos[j * 3 + 1] - pos[i * 3 + 1],
                pos[j * 3 + 2] - pos[i * 3 + 2],
            ];
            if self.use_pbc {
                if let Some(bv) = box_vectors {
                    apply_minimum_image_ortho(&mut delta, bv);
                }
            }

            let r2 = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
            if cutoff2.map(|c2| r2 >= c2).unwrap_or(false) {
                return;
            }
            let r = r2.sqrt();

            evaluator.set_coordinates(&[r]);
            evaluator.set_parameters(
                self.particles[i]
                    .iter()
                    .zip(&self.particles[j])
                    .flat_map(|(&a, &b)| [a, b]),
            );
            energy += evaluator.evaluate();
            let de_dr = evaluator.coordinate_derivative(0);

            let f_over_r = de_dr / r;
            for d in 0..3 {
                let f = f_over_r * delta[d];
                forces[i * 3 + d] += f;
                forces[j * 3 + d] -= f;
            }
        };
        if let Some(cutoff) = self.cutoff_distance.filter(|_| self.use_neighbor_cells) {
            self.neighbor_cells
                .visit(&pos, cutoff, box_vectors.filter(|_| self.use_pbc), pair);
        } else {
            for i in 0..n {
                for j in i + 1..n {
                    pair(i, j);
                }
            }
        }

        (
            crate::tensor::from_floats::<B, 1>(forces.as_slice(), &device).reshape([n, 3]),
            energy,
            evaluator.finish(),
        )
    }
}

impl<B: Backend> Force<B> for CustomNonbondedForce {
    fn initialize(&mut self, num_particles: usize, _box_vectors: Option<&[[f64; 3]; 3]>) {
        let _ = CustomEvaluator::new(
            &self.energy_expr,
            &self.global_params,
            self.paired_parameter_names
                .iter()
                .flatten()
                .map(String::as_str),
            &["r"],
            &self.energy_param_deriv_names,
        );
        assert_eq!(
            self.particles.len(),
            num_particles,
            "custom nonbonded particle count must match system"
        );
    }
    super::force_metadata_impl!();
    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        box_vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64) {
        let (forces, energy, _) = self.evaluate::<B>(positions, box_vectors);
        (forces, energy)
    }

    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.use_pbc
    }

    fn global_parameters(&self) -> HashMap<String, f64> {
        self.global_params.clone()
    }

    fn set_global_parameter(&mut self, name: &str, value: f64) -> bool {
        match self.global_params.get_mut(name) {
            Some(parameter) => {
                *parameter = value;
                true
            }
            None => false,
        }
    }

    fn energy_parameter_derivatives(
        &self,
        positions: &Tensor<B, 2>,
        box_vectors: Option<&[[f64; 3]; 3]>,
    ) -> HashMap<String, f64> {
        self.evaluate::<B>(positions, box_vectors).2
    }
}
