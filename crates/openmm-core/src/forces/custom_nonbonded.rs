// CustomNonbondedForce: user-defined pairwise energy as a function of r
// and per-particle parameters.
// Reference test: openmm/tests/TestCustomNonbondedForce.h
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

    pub fn set_cutoff_distance(&mut self, cutoff: f64) {
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

impl<B: Backend> Force<B> for CustomNonbondedForce {
    super::force_metadata_impl!();
    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        box_vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64) {
        let device = positions.device();
        let n = positions.dims()[0];
        let pos: Vec<f64> = positions
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(|v| v as f64)
            .collect();

        let mut forces = vec![0.0f64; n * 3];
        let mut energy = 0.0f64;

        let excluded: std::collections::HashSet<(usize, usize)> =
            self.exclusions.iter().cloned().collect();
        let cutoff2 = self.cutoff_distance.map(|c| c * c);
        let mut vars: HashMap<&str, f64> = HashMap::with_capacity(
            1 + self.global_params.len() + 2 * self.paired_parameter_names.len(),
        );
        for (name, &value) in &self.global_params {
            vars.insert(name.as_str(), value);
        }

        for i in 0..n {
            for j in (i + 1)..n {
                let key = (i, j);
                if excluded.contains(&key) {
                    continue;
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
                    continue;
                }
                let r = r2.sqrt();

                vars.insert("r", r);
                let pi = &self.particles[i];
                let pj = &self.particles[j];
                for (idx, names) in self.paired_parameter_names.iter().enumerate() {
                    vars.insert(names[0].as_str(), pi[idx]);
                    vars.insert(names[1].as_str(), pj[idx]);
                }

                energy += self.energy_expr.compute(&vars).unwrap_or(0.0);

                // Numerical dE/dr.
                let dr = 1e-5_f64;
                vars.insert("r", r + dr);
                let ep = self.energy_expr.compute(&vars).unwrap_or(0.0);
                vars.insert("r", r - dr);
                let em = self.energy_expr.compute(&vars).unwrap_or(0.0);
                let de_dr = (ep - em) / (2.0 * dr);

                let f_over_r = de_dr / r;
                for d in 0..3 {
                    let f = f_over_r * delta[d];
                    forces[i * 3 + d] += f;
                    forces[j * 3 + d] -= f;
                }
            }
        }

        let f32v: Vec<f32> = forces.iter().map(|&v| v as f32).collect();
        (
            Tensor::<B, 1>::from_floats(f32v.as_slice(), &device).reshape([n, 3]),
            energy,
        )
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
        let mut derivatives = HashMap::new();
        let step = 1e-6;
        for name in &self.energy_param_deriv_names {
            let value = *self
                .global_params
                .get(name)
                .expect("unknown global parameter");
            let mut plus = self.clone();
            plus.global_params.insert(name.clone(), value + step);
            let energy_plus = Force::<B>::calc_forces_and_energy(&plus, positions, box_vectors).1;

            let mut minus = self.clone();
            minus.global_params.insert(name.clone(), value - step);
            let energy_minus = Force::<B>::calc_forces_and_energy(&minus, positions, box_vectors).1;

            derivatives.insert(name.clone(), (energy_plus - energy_minus) / (2.0 * step));
        }
        derivatives
    }
}
