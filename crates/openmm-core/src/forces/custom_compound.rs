use super::custom_evaluator::CustomEvaluator;
use super::Force;
use crate::expression::{Expression, ParseError};
use burn::prelude::*;
use std::collections::HashMap;

#[derive(Clone)]
pub struct CustomCompoundBondForce {
    metadata: super::ForceMetadata,
    particles_per_bond: usize,
    symbol_prefix: char,
    coordinate_names: Vec<String>,
    energy_expr: Expression,
    energy_source: String,
    per_bond_params: Vec<String>,
    bonds: Vec<BondRecord>,
    global_param_names: Vec<String>,
    global_params: HashMap<String, f64>,
    energy_param_deriv_names: Vec<String>,
    use_pbc: bool,
}

#[derive(Clone)]
struct BondRecord {
    particles: Vec<usize>,
    params: Vec<f64>,
}

impl CustomCompoundBondForce {
    super::force_metadata!();
    pub fn new(particles_per_bond: usize, energy: &str) -> Result<Self, ParseError> {
        Self::new_with_prefix(particles_per_bond, energy, 'p')
    }
    pub(super) fn new_with_prefix(
        particles_per_bond: usize,
        energy: &str,
        symbol_prefix: char,
    ) -> Result<Self, ParseError> {
        Ok(Self {
            metadata: super::ForceMetadata::new("CustomCompoundBondForce"),
            particles_per_bond,
            symbol_prefix,
            coordinate_names: (1..=particles_per_bond)
                .flat_map(|index| ['x', 'y', 'z'].map(|axis| format!("{axis}{index}")))
                .collect(),
            energy_expr: Expression::parse_geometric(
                energy,
                Some((symbol_prefix, particles_per_bond)),
            )?,
            energy_source: energy.to_owned(),
            per_bond_params: Vec::new(),
            bonds: Vec::new(),
            global_param_names: Vec::new(),
            global_params: HashMap::new(),
            energy_param_deriv_names: Vec::new(),
            use_pbc: false,
        })
    }
    pub fn num_particles_per_bond(&self) -> usize {
        self.particles_per_bond
    }
    pub fn energy_function(&self) -> &str {
        &self.energy_source
    }
    pub fn set_energy_function(&mut self, energy: &str) -> Result<(), String> {
        let mut expression = Expression::parse_geometric(
            energy,
            Some((self.symbol_prefix, self.particles_per_bond)),
        )
        .map_err(|e| e.to_string())?;
        for index in 0..self.num_tabulated_functions() {
            expression
                .add_tabulated_function(
                    self.tabulated_function_name(index),
                    self.tabulated_function(index).clone(),
                )
                .map_err(|e| e.to_string())?;
        }
        self.energy_expr = expression;
        self.energy_source = energy.to_owned();
        Ok(())
    }
    pub fn add_tabulated_function(
        &mut self,
        name: &str,
        function: impl Into<crate::TabulatedFunction>,
    ) -> Result<usize, crate::expression::EvalError> {
        self.energy_expr.add_tabulated_function(name, function)
    }
    pub fn num_tabulated_functions(&self) -> usize {
        self.energy_expr.num_tabulated_functions()
    }
    pub fn tabulated_function_name(&self, index: usize) -> &str {
        self.energy_expr.tabulated_function_name(index)
    }
    pub fn tabulated_function(&self, index: usize) -> &crate::TabulatedFunction {
        self.energy_expr.tabulated_function(index)
    }
    pub fn set_tabulated_function(
        &mut self,
        index: usize,
        function: impl Into<crate::TabulatedFunction>,
    ) -> Result<(), crate::expression::EvalError> {
        self.energy_expr.set_tabulated_function(index, function)
    }
    pub fn add_per_bond_parameter(&mut self, name: &str) -> usize {
        self.per_bond_params.push(name.to_owned());
        self.per_bond_params.len() - 1
    }
    pub fn num_per_bond_parameters(&self) -> usize {
        self.per_bond_params.len()
    }

    pub fn per_bond_parameter_name(&self, index: usize) -> &str {
        &self.per_bond_params[index]
    }

    /// Add a global (constant) parameter.
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

    pub fn set_global_parameter_value(&mut self, name: &str, value: f64) {
        *self
            .global_params
            .get_mut(name)
            .expect("unknown global parameter") = value;
    }

    pub fn add_bond(&mut self, particles: Vec<usize>, params: Vec<f64>) -> usize {
        assert_eq!(
            particles.len(),
            self.particles_per_bond,
            "incorrect number of particles in bond"
        );
        assert_eq!(
            params.len(),
            self.per_bond_params.len(),
            "incorrect number of bond parameters"
        );
        self.bonds.push(BondRecord { particles, params });
        self.bonds.len() - 1
    }
    pub fn num_bonds(&self) -> usize {
        self.bonds.len()
    }
    pub fn bond_parameters(&self, index: usize) -> (&[usize], &[f64]) {
        (&self.bonds[index].particles, &self.bonds[index].params)
    }
    pub fn set_bond_parameters(&mut self, index: usize, particles: Vec<usize>, params: Vec<f64>) {
        assert_eq!(
            particles.len(),
            self.particles_per_bond,
            "incorrect number of particles in bond"
        );
        assert_eq!(
            params.len(),
            self.per_bond_params.len(),
            "incorrect number of bond parameters"
        );
        self.bonds[index] = BondRecord { particles, params };
    }
    pub fn set_uses_periodic_boundary_conditions(&mut self, periodic: bool) {
        self.use_pbc = periodic;
    }
    pub fn uses_periodic_boundary_conditions(&self) -> bool {
        self.use_pbc
    }
    pub fn update_parameters_from(&mut self, source: &Self) {
        assert_eq!(
            self.bonds.len(),
            source.bonds.len(),
            "bond count must match"
        );
        assert_eq!(
            self.per_bond_params, source.per_bond_params,
            "bond parameter definitions must match"
        );
        for (bond, other) in self.bonds.iter().zip(&source.bonds) {
            assert_eq!(
                bond.particles, other.particles,
                "bond particles cannot change in a Context"
            );
        }
        self.energy_expr
            .update_tabulated_functions_from(&source.energy_expr);
        for (bond, other) in self.bonds.iter_mut().zip(&source.bonds) {
            bond.params.clone_from(&other.params);
        }
    }
    pub(super) fn evaluate_host(
        &self,
        positions: &[f64],
        n: usize,
        vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Vec<f64>, f64, HashMap<String, f64>) {
        let coordinates: Vec<_> = self.coordinate_names.iter().map(String::as_str).collect();
        let mut evaluator = CustomEvaluator::new(
            &self.energy_expr,
            &self.global_params,
            self.per_bond_params.iter().map(String::as_str),
            &coordinates,
            &self.energy_param_deriv_names,
        );
        evaluator.set_periodic_box(vectors.filter(|_| self.use_pbc));
        let mut forces = vec![0.0; 3 * n];
        let mut energy = 0.0;
        for bond in &self.bonds {
            evaluator.set_coordinate_values(
                bond.particles.iter().flat_map(|&particle| {
                    positions[3 * particle..3 * particle + 3].iter().copied()
                }),
            );
            evaluator.set_parameters(bond.params.iter().copied());
            energy += evaluator.evaluate();
            for (index, &particle) in bond.particles.iter().enumerate() {
                for axis in 0..3 {
                    forces[3 * particle + axis] -=
                        evaluator.coordinate_derivative(3 * index + axis);
                }
            }
        }
        (forces, energy, evaluator.finish())
    }
    fn evaluate<B: Backend>(
        &self,
        positions: &Tensor<B, 2>,
        vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64, HashMap<String, f64>) {
        let n = positions.dims()[0];
        let pos = positions
            .to_data()
            .convert::<f64>()
            .to_vec::<f64>()
            .unwrap();
        let (forces, energy, derivatives) = self.evaluate_host(&pos, n, vectors);
        (
            crate::tensor::from_floats::<B, 1>(forces.as_slice(), &positions.device())
                .reshape([n, 3]),
            energy,
            derivatives,
        )
    }
}

impl<B: Backend> Force<B> for CustomCompoundBondForce {
    super::force_metadata_impl!();
    fn initialize(&mut self, num_particles: usize, _vectors: Option<&[[f64; 3]; 3]>) {
        let coordinates: Vec<_> = self.coordinate_names.iter().map(String::as_str).collect();
        let _ = CustomEvaluator::new(
            &self.energy_expr,
            &self.global_params,
            self.per_bond_params.iter().map(String::as_str),
            &coordinates,
            &self.energy_param_deriv_names,
        );
        for bond in &self.bonds {
            assert!(
                bond.particles.iter().all(|&p| p < num_particles),
                "custom bond particle index out of range"
            );
        }
    }
    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64) {
        let (forces, energy, _) = self.evaluate(positions, vectors);
        (forces, energy)
    }
    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.use_pbc
    }
    fn global_parameters(&self) -> HashMap<String, f64> {
        self.global_params.clone()
    }
    fn set_global_parameter(&mut self, name: &str, value: f64) -> bool {
        if let Some(parameter) = self.global_params.get_mut(name) {
            *parameter = value;
            true
        } else {
            false
        }
    }
    fn energy_parameter_derivatives(
        &self,
        positions: &Tensor<B, 2>,
        vectors: Option<&[[f64; 3]; 3]>,
    ) -> HashMap<String, f64> {
        self.evaluate(positions, vectors).2
    }
}
