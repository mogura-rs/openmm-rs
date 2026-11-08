use super::{custom_evaluator::CustomEvaluator, Force};
use crate::expression::{Expression, ParseError};
use burn::prelude::*;
use std::collections::HashMap;

const BOX_VARIABLES: [&str; 7] = ["v", "ax", "bx", "by", "cx", "cy", "cz"];

#[derive(Clone)]
pub struct CustomVolumeForce {
    metadata: super::ForceMetadata,
    energy: String,
    expression: Expression,
    global_names: Vec<String>,
    globals: HashMap<String, f64>,
}
impl CustomVolumeForce {
    super::force_metadata!();
    pub fn new(energy: &str) -> Result<Self, ParseError> {
        Ok(Self {
            metadata: super::ForceMetadata::new("CustomVolumeForce"),
            energy: energy.to_owned(),
            expression: Expression::parse(energy)?,
            global_names: Vec::new(),
            globals: HashMap::new(),
        })
    }
    pub fn energy_function(&self) -> &str {
        &self.energy
    }
    pub fn set_energy_function(&mut self, energy: &str) -> Result<(), ParseError> {
        let expression = Expression::parse(energy)?;
        self.expression = expression;
        self.energy = energy.to_owned();
        Ok(())
    }
    pub fn add_global_parameter(&mut self, name: &str, value: f64) -> usize {
        if !self.globals.contains_key(name) {
            self.global_names.push(name.to_owned());
        }
        self.globals.insert(name.to_owned(), value);
        self.global_names.iter().position(|n| n == name).unwrap()
    }
    pub fn num_global_parameters(&self) -> usize {
        self.global_names.len()
    }
    pub fn global_parameter_name(&self, index: usize) -> &str {
        &self.global_names[index]
    }
    pub fn global_parameter_default_value(&self, index: usize) -> f64 {
        self.globals[&self.global_names[index]]
    }
    pub fn set_global_parameter_default_value(&mut self, index: usize, value: f64) {
        self.globals.insert(self.global_names[index].clone(), value);
    }
    pub fn uses_periodic_boundary_conditions(&self) -> bool {
        true
    }
}
impl<B: Backend> Force<B> for CustomVolumeForce {
    super::force_metadata_impl!();
    fn initialize(&mut self, _n: usize, _vectors: Option<&[[f64; 3]; 3]>) {
        self.expression
            .validate_variables(
                self.global_names
                    .iter()
                    .map(String::as_str)
                    .chain(BOX_VARIABLES),
            )
            .unwrap_or_else(|error| panic!("{error}"));
    }
    fn uses_periodic_boundary_conditions(&self) -> bool {
        true
    }
    fn global_parameters(&self) -> HashMap<String, f64> {
        self.globals.clone()
    }
    fn set_global_parameter(&mut self, name: &str, value: f64) -> bool {
        if let Some(current) = self.globals.get_mut(name) {
            *current = value;
            true
        } else {
            false
        }
    }
    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64) {
        let vectors = vectors.expect("CustomVolumeForce requires periodic box vectors");
        let mut evaluator =
            CustomEvaluator::new(&self.expression, &self.globals, BOX_VARIABLES, &[], &[]);
        evaluator.set_parameters([
            vectors[0][0] * vectors[1][1] * vectors[2][2],
            vectors[0][0],
            vectors[1][0],
            vectors[1][1],
            vectors[2][0],
            vectors[2][1],
            vectors[2][2],
        ]);
        (
            Tensor::zeros(positions.shape(), &positions.device()),
            evaluator.evaluate(),
        )
    }
}
