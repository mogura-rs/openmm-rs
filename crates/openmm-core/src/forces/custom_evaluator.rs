use crate::expression::{Expression, Workspace};
use std::collections::HashMap;

pub(super) struct CustomEvaluator<'a> {
    expression: &'a Expression,
    workspace: Workspace,
    coordinates: Vec<Option<usize>>,
    parameters: Vec<Option<usize>>,
    derivatives: Vec<(&'a str, Option<usize>, f64)>,
}

impl<'a> CustomEvaluator<'a> {
    pub fn new(
        expression: &'a Expression,
        globals: &HashMap<String, f64>,
        parameters: impl IntoIterator<Item = &'a str>,
        coordinates: &[&str],
        derivative_names: &'a [String],
    ) -> Self {
        let parameters: Vec<_> = parameters.into_iter().collect();
        expression
            .validate_variables(
                globals
                    .keys()
                    .map(String::as_str)
                    .chain(parameters.iter().copied())
                    .chain(coordinates.iter().copied()),
            )
            .unwrap_or_else(|error| panic!("{error}"));
        let mut workspace = expression.workspace_with_derivatives(
            coordinates
                .iter()
                .copied()
                .chain(derivative_names.iter().map(String::as_str))
                .filter_map(|name| expression.variable_index(name)),
        );
        for (name, &value) in globals {
            workspace.set(expression.variable_index(name), value);
        }
        Self {
            expression,
            workspace,
            coordinates: coordinates
                .iter()
                .map(|name| expression.variable_index(name))
                .collect(),
            parameters: parameters
                .iter()
                .map(|name| expression.variable_index(name))
                .collect(),
            derivatives: derivative_names
                .iter()
                .map(|name| (name.as_str(), expression.variable_index(name), 0.0))
                .collect(),
        }
    }

    #[inline]
    pub fn set_coordinates(&mut self, values: &[f64]) {
        for (&index, &value) in self.coordinates.iter().zip(values) {
            self.workspace.set(index, value);
        }
    }

    pub fn set_parameters(&mut self, values: impl IntoIterator<Item = f64>) {
        for (&index, value) in self.parameters.iter().zip(values) {
            self.workspace.set(index, value);
        }
    }

    #[inline]
    pub fn set_parameter_pairs(&mut self, offset: usize, first: &[f64], second: &[f64]) {
        for i in 0..first.len() {
            self.workspace
                .set(self.parameters[2 * (offset + i)], first[i]);
            self.workspace
                .set(self.parameters[2 * (offset + i) + 1], second[i]);
        }
    }

    #[inline]
    pub fn evaluate(&mut self) -> f64 {
        self.evaluate_scaled(1.0)
    }

    #[inline]
    pub fn evaluate_scaled(&mut self, scale: f64) -> f64 {
        let energy = self.expression.evaluate(&mut self.workspace);
        for (_, index, total) in &mut self.derivatives {
            *total += scale * self.workspace.derivative(*index);
        }
        energy
    }

    #[inline]
    pub fn parameter_derivative(&self, index: usize) -> f64 {
        self.workspace.derivative(self.derivatives[index].1)
    }

    #[inline]
    pub fn coordinate_derivative(&self, index: usize) -> f64 {
        self.workspace.derivative(self.coordinates[index])
    }

    pub fn finish(self) -> HashMap<String, f64> {
        self.derivatives
            .into_iter()
            .map(|(name, _, total)| (name.to_owned(), total))
            .collect()
    }
}
