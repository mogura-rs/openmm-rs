use super::{custom_evaluator::CustomEvaluator, Force};
use crate::expression::{EvalError, Expression, ParseError};
use burn::prelude::*;
use std::collections::HashMap;

#[derive(Clone)]
pub struct CustomCVForce<B: Backend> {
    metadata: super::ForceMetadata,
    expression: Expression,
    energy: String,
    variables: Vec<(String, Box<dyn Force<B>>)>,
    global_names: Vec<String>,
    globals: HashMap<String, f64>,
    parameters: Option<HashMap<String, f64>>,
    derivative_names: Vec<String>,
    force_cache_bytes: usize,
}

impl<B: Backend> CustomCVForce<B> {
    super::force_metadata!();
    pub fn new(energy: &str) -> Result<Self, ParseError> {
        Ok(Self {
            metadata: super::ForceMetadata::new("CustomCVForce"),
            expression: Expression::parse_unbound(energy)?,
            energy: energy.to_owned(),
            variables: Vec::new(),
            global_names: Vec::new(),
            globals: HashMap::new(),
            parameters: None,
            derivative_names: Vec::new(),
            force_cache_bytes: 16 * 1024 * 1024,
        })
    }
    pub fn energy_function(&self) -> &str {
        &self.energy
    }
    pub fn set_energy_function(&mut self, energy: &str) -> Result<(), String> {
        let mut expression = Expression::parse_unbound(energy).map_err(|e| e.to_string())?;
        for index in 0..self.num_tabulated_functions() {
            expression
                .add_tabulated_function(
                    self.tabulated_function_name(index),
                    self.tabulated_function(index).clone(),
                )
                .map_err(|e| e.to_string())?;
        }
        self.expression = expression;
        self.energy = energy.to_owned();
        Ok(())
    }
    pub fn add_collective_variable(&mut self, name: &str, force: Box<dyn Force<B>>) -> usize {
        assert!(
            self.variables.len() < 32,
            "CustomCVForce cannot have more than 32 collective variables"
        );
        self.variables.push((name.to_owned(), force));
        self.parameters = None;
        self.variables.len() - 1
    }
    pub fn num_collective_variables(&self) -> usize {
        self.variables.len()
    }
    pub fn collective_variable_name(&self, index: usize) -> &str {
        &self.variables[index].0
    }
    pub fn collective_variable(&self, index: usize) -> &(dyn Force<B> + 'static) {
        self.variables[index].1.as_ref()
    }
    pub fn collective_variable_mut(&mut self, index: usize) -> &mut (dyn Force<B> + 'static) {
        self.variables[index].1.as_mut()
    }
    pub fn replace_collective_variable(&mut self, index: usize, force: Box<dyn Force<B>>) {
        self.variables[index].1 = force;
        self.parameters = None;
    }
    pub fn collective_variable_values(
        &self,
        positions: &Tensor<B, 2>,
        vectors: Option<&[[f64; 3]; 3]>,
    ) -> Vec<f64> {
        self.variables
            .iter()
            .map(|(_, force)| {
                force
                    .prepare_host_evaluation(positions, vectors)
                    .map_or_else(
                        || force.calc_forces_and_energy(positions, vectors).1,
                        |prepared| prepared.energy,
                    )
            })
            .collect()
    }
    pub fn force_cache_bytes(&self) -> usize {
        self.force_cache_bytes
    }
    pub fn set_force_cache_bytes(&mut self, bytes: usize) {
        self.force_cache_bytes = bytes;
    }
    pub fn add_global_parameter(&mut self, name: &str, value: f64) -> usize {
        if !self.globals.contains_key(name) {
            self.global_names.push(name.to_owned());
        }
        self.globals.insert(name.to_owned(), value);
        self.parameters = None;
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
        self.parameters = None;
    }
    pub fn add_energy_parameter_derivative(&mut self, name: &str) {
        assert!(self.globals.contains_key(name), "unknown global parameter");
        if !self.derivative_names.iter().any(|n| n == name) {
            self.derivative_names.push(name.to_owned());
        }
    }
    pub fn num_energy_parameter_derivatives(&self) -> usize {
        self.derivative_names.len()
    }
    pub fn energy_parameter_derivative_name(&self, index: usize) -> &str {
        &self.derivative_names[index]
    }
    pub fn add_tabulated_function(
        &mut self,
        name: &str,
        function: impl Into<crate::TabulatedFunction>,
    ) -> Result<usize, EvalError> {
        self.expression.add_tabulated_function(name, function)
    }
    pub fn num_tabulated_functions(&self) -> usize {
        self.expression.num_tabulated_functions()
    }
    pub fn tabulated_function_name(&self, index: usize) -> &str {
        self.expression.tabulated_function_name(index)
    }
    pub fn tabulated_function(&self, index: usize) -> &crate::TabulatedFunction {
        self.expression.tabulated_function(index)
    }
    pub fn set_tabulated_function(
        &mut self,
        index: usize,
        function: impl Into<crate::TabulatedFunction>,
    ) -> Result<(), EvalError> {
        self.expression.set_tabulated_function(index, function)
    }
    pub fn update_parameters_from(&mut self, source: &Self) {
        self.expression
            .update_tabulated_functions_from(&source.expression);
    }
    pub fn uses_periodic_boundary_conditions(&self) -> bool {
        self.variables
            .iter()
            .any(|(_, force)| force.uses_periodic_boundary_conditions())
    }

    fn parameter_values(&self) -> HashMap<String, f64> {
        let mut values = HashMap::new();
        for (_, force) in &self.variables {
            for (name, value) in force.global_parameters() {
                if let Some(previous) = values.insert(name.clone(), value) {
                    assert!(previous == value, "collective variable global parameter '{name}' has conflicting default values");
                }
            }
        }
        values.extend(
            self.globals
                .iter()
                .map(|(name, &value)| (name.clone(), value)),
        );
        values
    }

    fn evaluate(
        &self,
        positions: &Tensor<B, 2>,
        vectors: Option<&[[f64; 3]; 3]>,
        want_forces: bool,
        want_derivatives: bool,
    ) -> (Option<Tensor<B, 2>>, f64, HashMap<String, f64>) {
        let n = positions.dims()[0];
        let bytes_per_force = n.saturating_mul(3 * std::mem::size_of::<f64>());
        let cache_forces = want_forces
            && bytes_per_force.saturating_mul(self.variables.len()) <= self.force_cache_bytes;
        let mut values = Vec::with_capacity(self.variables.len());
        let mut cached = Vec::with_capacity(self.variables.len());
        let mut prepared = Vec::with_capacity(self.variables.len());
        for (_, force) in &self.variables {
            if let Some(evaluation) = force.prepare_host_evaluation(positions, vectors) {
                values.push(evaluation.energy);
                cached.push(None);
                prepared.push(Some(evaluation));
                continue;
            }
            prepared.push(None);
            let (forces, energy) = force.calc_forces_and_energy(positions, vectors);
            values.push(energy);
            if cache_forces {
                cached.push(Some(forces));
            } else {
                cached.push(None);
            }
        }
        let names = self
            .variables
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();
        let requested = if want_derivatives {
            self.derivative_names.as_slice()
        } else {
            &[]
        };
        let mut evaluator =
            CustomEvaluator::new(&self.expression, &self.globals, [], &names, requested);
        evaluator.set_coordinates(&values);
        let energy = evaluator.evaluate();
        let mut total = None;
        let mut host_total = if want_forces
            && crate::tensor::with_ndarray_f64(positions, |_| true).unwrap_or(false)
        {
            Some(vec![0.0; 3 * n])
        } else {
            None
        };
        let mut inherited_derivatives = HashMap::new();
        for (index, (((_, force), cached), prepared)) in
            self.variables.iter().zip(cached).zip(prepared).enumerate()
        {
            let coefficient = evaluator.coordinate_derivative(index);
            if coefficient == 0.0 {
                continue;
            }
            if want_forces {
                let accumulated = if let (Some(prepared), Some(total)) = (prepared, &mut host_total)
                {
                    prepared.accumulate(coefficient, total);
                    true
                } else {
                    false
                };
                if !accumulated {
                    let forces = cached
                        .unwrap_or_else(|| force.calc_forces_and_energy(positions, vectors).0);
                    if let Some(total) = &mut host_total {
                        crate::tensor::with_f64_data(&forces, |values| {
                            for (total, &value) in total.iter_mut().zip(values) {
                                *total += coefficient * value;
                            }
                        });
                    } else {
                        let scaled = forces * coefficient;
                        total = Some(match total {
                            Some(total) => total + scaled,
                            None => scaled,
                        });
                    }
                }
            }
            if want_derivatives && !self.derivative_names.is_empty() {
                for (name, value) in force.energy_parameter_derivatives(positions, vectors) {
                    *inherited_derivatives.entry(name).or_insert(0.0) += coefficient * value;
                }
            }
        }
        if let Some(host_total) = host_total {
            total = Some(crate::tensor::from_f64_vec(
                host_total,
                [n, 3],
                &positions.device(),
            ));
        }
        if want_forces && total.is_none() {
            total = Some(Tensor::zeros(positions.shape(), &positions.device()));
        }
        let mut derivatives = evaluator.finish();
        for (name, value) in inherited_derivatives {
            *derivatives.entry(name).or_insert(0.0) += value;
        }
        (total, energy, derivatives)
    }
}

impl<B: Backend> Force<B> for CustomCVForce<B> {
    super::force_metadata_impl!();
    fn initialize_masses(&mut self, masses: &[f64]) {
        for (_, force) in &mut self.variables {
            force.initialize_masses(masses);
        }
    }
    fn initialize(&mut self, n: usize, vectors: Option<&[[f64; 3]; 3]>) {
        self.expression
            .validate_functions()
            .unwrap_or_else(|e| panic!("{e}"));
        self.expression
            .validate_variables(
                self.global_names
                    .iter()
                    .map(String::as_str)
                    .chain(self.variables.iter().map(|(name, _)| name.as_str())),
            )
            .unwrap_or_else(|e| panic!("{e}"));
        for (_, force) in &mut self.variables {
            force.initialize(n, vectors);
        }
        let values = self.parameter_values();
        self.parameters = Some(values.clone());
        for (name, value) in values {
            self.set_global_parameter(&name, value);
        }
    }
    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.uses_periodic_boundary_conditions()
    }
    fn bonded_particles(&self) -> Vec<(usize, usize)> {
        self.variables
            .iter()
            .flat_map(|(_, force)| force.bonded_particles())
            .collect()
    }
    fn global_parameters(&self) -> HashMap<String, f64> {
        self.parameters
            .clone()
            .unwrap_or_else(|| self.parameter_values())
    }
    fn set_global_parameter(&mut self, name: &str, value: f64) -> bool {
        let mut known = false;
        if let Some(current) = self
            .parameters
            .as_mut()
            .and_then(|parameters| parameters.get_mut(name))
        {
            *current = value;
            known = true;
        }
        if let Some(current) = self.globals.get_mut(name) {
            *current = value;
            known = true;
        }
        for (_, force) in &mut self.variables {
            known |= force.set_global_parameter(name, value);
        }
        known
    }
    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64) {
        let (forces, energy, _) = self.evaluate(positions, vectors, true, false);
        (forces.unwrap(), energy)
    }
    fn energy_parameter_derivatives(
        &self,
        positions: &Tensor<B, 2>,
        vectors: Option<&[[f64; 3]; 3]>,
    ) -> HashMap<String, f64> {
        self.evaluate(positions, vectors, false, true).2
    }
}

pub(crate) fn update_force_parameters<B: Backend>(
    target: &mut dyn Force<B>,
    source: &dyn Force<B>,
) -> Result<(), String> {
    macro_rules! update {
        ($force:ty) => {
            if let Some(source) = source.as_any().downcast_ref::<$force>() {
                let target = target
                    .as_any_mut()
                    .downcast_mut::<$force>()
                    .ok_or_else(|| "force types must match".to_owned())?;
                target.update_parameters_from(source);
                return Ok(());
            }
        };
    }
    update!(crate::RMSDForce);
    update!(crate::OrientationRestraintForce);
    update!(crate::CMAPTorsionForce);
    update!(crate::HarmonicBondForce);
    update!(crate::HarmonicAngleForce);
    update!(crate::PeriodicTorsionForce);
    update!(crate::RBTorsionForce);
    update!(crate::NonbondedForce);
    update!(crate::GBSAOBCForce);
    update!(crate::CustomExternalForce);
    update!(crate::CustomBondForce);
    update!(crate::CustomAngleForce);
    update!(crate::CustomTorsionForce);
    update!(crate::CustomNonbondedForce);
    update!(crate::CustomCompoundBondForce);
    update!(crate::CustomCentroidBondForce);
    update!(crate::CustomHbondForce);
    update!(CustomCVForce<B>);
    Err("this force does not support parameter updates".to_owned())
}
