mod compiler;
mod runtime;
use super::{ContextProgram, Integrator};
use crate::{constraints::ConstraintError, tabulated::TabulatedFunction};
use burn::prelude::*;
use compiler::Compiled;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ComputationType {
    ComputeGlobal,
    ComputePerDof,
    ComputeSum,
    ConstrainPositions,
    ConstrainVelocities,
    UpdateContextState,
    IfBlockStart,
    WhileBlockStart,
    BlockEnd,
}

#[derive(Clone, Debug)]
pub struct ComputationStep {
    pub kind: ComputationType,
    pub variable: String,
    pub expression: String,
}

struct Variables {
    globals: Vec<(String, f64)>,
    per_dof: Vec<(String, Vec<[f64; 3]>)>,
    step_size: f64,
}

struct Data {
    variables: Variables,
    tolerance: f64,
    groups: u32,
    computations: Vec<ComputationStep>,
    tables: Vec<(String, TabulatedFunction)>,
    kinetic: String,
    random: super::random::GaussianStream,
    particles: Option<usize>,
    compiled: Option<Compiled>,
}

pub struct CustomIntegrator {
    data: Arc<Mutex<Data>>,
}

fn failure(message: impl Into<String>) -> ConstraintError {
    ConstraintError(message.into())
}

impl CustomIntegrator {
    pub fn new(step_size: f64) -> Self {
        assert!(
            step_size >= 0.0 && step_size.is_finite(),
            "step size must be nonnegative and finite"
        );
        Self {
            data: Arc::new(Mutex::new(Data {
                variables: Variables {
                    globals: Vec::new(),
                    per_dof: Vec::new(),
                    step_size,
                },
                tolerance: 1e-5,
                groups: u32::MAX,
                computations: Vec::new(),
                tables: Vec::new(),
                kinetic: "m*v*v/2".into(),
                random: Default::default(),
                particles: None,
                compiled: None,
            })),
        }
    }
    fn structural(data: &Data) -> Result<(), ConstraintError> {
        if data.particles.is_some() {
            Err(failure(
                "the integrator cannot be modified after binding to a Context",
            ))
        } else {
            Ok(())
        }
    }
    pub fn add_global_variable(
        &mut self,
        name: &str,
        value: f64,
    ) -> Result<usize, ConstraintError> {
        let mut data = self.data.lock().unwrap();
        Self::structural(&data)?;
        data.variables.globals.push((name.into(), value));
        Ok(data.variables.globals.len() - 1)
    }
    pub fn num_global_variables(&self) -> usize {
        self.data.lock().unwrap().variables.globals.len()
    }
    pub fn global_variable_name(&self, index: usize) -> Result<String, ConstraintError> {
        self.data
            .lock()
            .unwrap()
            .variables
            .globals
            .get(index)
            .map(|v| v.0.clone())
            .ok_or_else(|| failure("global variable index out of range"))
    }
    pub fn global_variable(&self, index: usize) -> Result<f64, ConstraintError> {
        self.data
            .lock()
            .unwrap()
            .variables
            .globals
            .get(index)
            .map(|v| v.1)
            .ok_or_else(|| failure("global variable index out of range"))
    }
    pub fn set_global_variable(&mut self, index: usize, value: f64) -> Result<(), ConstraintError> {
        self.data
            .lock()
            .unwrap()
            .variables
            .globals
            .get_mut(index)
            .ok_or_else(|| failure("global variable index out of range"))?
            .1 = value;
        Ok(())
    }
    pub fn global_variable_index(&self, name: &str) -> Result<usize, ConstraintError> {
        self.data
            .lock()
            .unwrap()
            .variables
            .globals
            .iter()
            .position(|v| v.0 == name)
            .ok_or_else(|| failure(format!("unknown global variable: {name}")))
    }
    pub fn add_per_dof_variable(
        &mut self,
        name: &str,
        value: f64,
    ) -> Result<usize, ConstraintError> {
        let mut data = self.data.lock().unwrap();
        Self::structural(&data)?;
        data.variables.per_dof.push((name.into(), vec![[value; 3]]));
        Ok(data.variables.per_dof.len() - 1)
    }
    pub fn num_per_dof_variables(&self) -> usize {
        self.data.lock().unwrap().variables.per_dof.len()
    }
    pub fn per_dof_variable_name(&self, index: usize) -> Result<String, ConstraintError> {
        self.data
            .lock()
            .unwrap()
            .variables
            .per_dof
            .get(index)
            .map(|v| v.0.clone())
            .ok_or_else(|| failure("per-DOF variable index out of range"))
    }
    pub fn per_dof_variable(&self, index: usize) -> Result<Vec<[f64; 3]>, ConstraintError> {
        self.data
            .lock()
            .unwrap()
            .variables
            .per_dof
            .get(index)
            .map(|v| v.1.clone())
            .ok_or_else(|| failure("per-DOF variable index out of range"))
    }
    pub fn set_per_dof_variable(
        &mut self,
        index: usize,
        values: Vec<[f64; 3]>,
    ) -> Result<(), ConstraintError> {
        let mut data = self.data.lock().unwrap();
        if data.particles.is_some_and(|n| n != values.len()) {
            return Err(failure("wrong number of per-DOF values"));
        }
        data.variables
            .per_dof
            .get_mut(index)
            .ok_or_else(|| failure("per-DOF variable index out of range"))?
            .1 = values;
        Ok(())
    }
    pub fn per_dof_variable_index(&self, name: &str) -> Result<usize, ConstraintError> {
        self.data
            .lock()
            .unwrap()
            .variables
            .per_dof
            .iter()
            .position(|v| v.0 == name)
            .ok_or_else(|| failure(format!("unknown per-DOF variable: {name}")))
    }
    pub fn add_computation(
        &mut self,
        kind: ComputationType,
        variable: &str,
        expression: &str,
    ) -> Result<usize, ConstraintError> {
        let mut data = self.data.lock().unwrap();
        Self::structural(&data)?;
        data.computations.push(ComputationStep {
            kind,
            variable: variable.into(),
            expression: expression.into(),
        });
        Ok(data.computations.len() - 1)
    }
    pub fn add_compute_global(
        &mut self,
        variable: &str,
        expression: &str,
    ) -> Result<usize, ConstraintError> {
        self.add_computation(ComputationType::ComputeGlobal, variable, expression)
    }
    pub fn add_compute_per_dof(
        &mut self,
        variable: &str,
        expression: &str,
    ) -> Result<usize, ConstraintError> {
        self.add_computation(ComputationType::ComputePerDof, variable, expression)
    }
    pub fn add_compute_sum(
        &mut self,
        variable: &str,
        expression: &str,
    ) -> Result<usize, ConstraintError> {
        self.add_computation(ComputationType::ComputeSum, variable, expression)
    }
    pub fn add_constrain_positions(&mut self) -> Result<usize, ConstraintError> {
        self.add_computation(ComputationType::ConstrainPositions, "", "")
    }
    pub fn add_constrain_velocities(&mut self) -> Result<usize, ConstraintError> {
        self.add_computation(ComputationType::ConstrainVelocities, "", "")
    }
    pub fn add_update_context_state(&mut self) -> Result<usize, ConstraintError> {
        self.add_computation(ComputationType::UpdateContextState, "", "")
    }
    pub fn begin_if_block(&mut self, expression: &str) -> Result<usize, ConstraintError> {
        self.add_computation(ComputationType::IfBlockStart, "", expression)
    }
    pub fn begin_while_block(&mut self, expression: &str) -> Result<usize, ConstraintError> {
        self.add_computation(ComputationType::WhileBlockStart, "", expression)
    }
    pub fn end_block(&mut self) -> Result<usize, ConstraintError> {
        self.add_computation(ComputationType::BlockEnd, "", "")
    }
    pub fn num_computations(&self) -> usize {
        self.data.lock().unwrap().computations.len()
    }
    pub fn computation_step(&self, index: usize) -> Result<ComputationStep, ConstraintError> {
        self.data
            .lock()
            .unwrap()
            .computations
            .get(index)
            .cloned()
            .ok_or_else(|| failure("computation index out of range"))
    }
    pub fn add_tabulated_function(
        &mut self,
        name: &str,
        function: impl Into<TabulatedFunction>,
    ) -> usize {
        let mut data = self.data.lock().unwrap();
        data.tables.push((name.into(), function.into()));
        data.tables.len() - 1
    }
    pub fn num_tabulated_functions(&self) -> usize {
        self.data.lock().unwrap().tables.len()
    }
    pub fn tabulated_function_name(&self, index: usize) -> Result<String, ConstraintError> {
        self.data
            .lock()
            .unwrap()
            .tables
            .get(index)
            .map(|v| v.0.clone())
            .ok_or_else(|| failure("tabulated function index out of range"))
    }
    pub fn tabulated_function(&self, index: usize) -> Result<TabulatedFunction, ConstraintError> {
        self.data
            .lock()
            .unwrap()
            .tables
            .get(index)
            .map(|v| v.1.clone())
            .ok_or_else(|| failure("tabulated function index out of range"))
    }
    pub fn is_prepared(&self) -> bool {
        self.data.lock().unwrap().compiled.is_some()
    }
    pub fn set_tabulated_function(
        &mut self,
        index: usize,
        function: impl Into<TabulatedFunction>,
    ) -> Result<(), ConstraintError> {
        self.data
            .lock()
            .unwrap()
            .tables
            .get_mut(index)
            .ok_or_else(|| failure("tabulated function index out of range"))?
            .1 = function.into();
        Ok(())
    }
    pub fn kinetic_energy_expression(&self) -> String {
        self.data.lock().unwrap().kinetic.clone()
    }
    pub fn set_kinetic_energy_expression(
        &mut self,
        expression: &str,
    ) -> Result<(), ConstraintError> {
        crate::expression::Expression::parse(expression).map_err(|e| failure(e.to_string()))?;
        self.data.lock().unwrap().kinetic = expression.into();
        Ok(())
    }
    pub fn random_number_seed(&self) -> i32 {
        self.data.lock().unwrap().random.seed()
    }
    pub fn set_random_number_seed(&mut self, seed: i32) {
        self.data.lock().unwrap().random.set_seed(seed);
    }
    pub fn step_size(&self) -> f64 {
        self.data.lock().unwrap().variables.step_size
    }
    pub fn set_step_size(&mut self, dt: f64) {
        assert!(
            dt >= 0.0 && dt.is_finite(),
            "step size must be nonnegative and finite"
        );
        self.data.lock().unwrap().variables.step_size = dt;
    }
    pub fn constraint_tolerance(&self) -> f64 {
        self.data.lock().unwrap().tolerance
    }
    pub fn set_constraint_tolerance(&mut self, tolerance: f64) {
        assert!(
            tolerance > 0.0 && tolerance.is_finite(),
            "constraint tolerance must be positive and finite"
        );
        self.data.lock().unwrap().tolerance = tolerance;
    }
    pub fn integration_force_groups(&self) -> u32 {
        self.data.lock().unwrap().groups
    }
    pub fn set_integration_force_groups(&mut self, groups: u32) {
        self.data.lock().unwrap().groups = groups;
    }
}

impl<B: Backend> Integrator<B> for CustomIntegrator {
    fn context_program(&self) -> Option<Arc<dyn ContextProgram<B>>> {
        Some(self.data.clone())
    }
    fn validate_system(&self, system: &crate::System<B>) -> Result<(), ConstraintError> {
        let data = self.data.lock().unwrap();
        let params: std::collections::HashMap<_, _> = system
            .forces()
            .iter()
            .flat_map(|force| force.global_parameters())
            .collect();
        let mut names = std::collections::HashSet::new();
        for name in data
            .variables
            .globals
            .iter()
            .map(|v| &v.0)
            .chain(data.variables.per_dof.iter().map(|v| &v.0))
        {
            if !names.insert(name) {
                return Err(failure(format!("duplicate integrator variable: {name}")));
            }
            if params.contains_key(name) {
                return Err(failure(format!(
                    "integrator variable conflicts with Context parameter: {name}"
                )));
            }
        }
        for (_, values) in &data.variables.per_dof {
            if values.len() != 1 && values.len() != system.num_particles() {
                return Err(failure("wrong number of per-DOF values"));
            }
        }
        for step in &data.computations {
            if step.kind == ComputationType::ComputeGlobal
                && step.variable != "dt"
                && !params.contains_key(&step.variable)
                && !data.variables.globals.iter().any(|v| v.0 == step.variable)
            {
                return Err(failure(format!(
                    "unknown global variable: {}",
                    step.variable
                )));
            }
        }
        Ok(())
    }
    fn initialize_system(&mut self, system: &crate::System<B>) {
        let mut data = self.data.lock().unwrap();
        let n = system.num_particles();
        for (_, values) in &mut data.variables.per_dof {
            if values.len() == 1 {
                values.resize(n, values[0]);
            }
        }
        data.particles = Some(n);
        data.compiled = None;
        data.random.initialize();
    }
    fn step_size(&self) -> f64 {
        self.step_size()
    }
    fn set_step_size(&mut self, dt: f64) {
        self.set_step_size(dt);
    }
    fn constraint_tolerance(&self) -> f64 {
        self.constraint_tolerance()
    }
    fn set_constraint_tolerance(&mut self, value: f64) {
        self.set_constraint_tolerance(value);
    }
    fn integration_force_groups(&self) -> u32 {
        self.integration_force_groups()
    }
    fn set_integration_force_groups(&mut self, groups: u32) {
        self.set_integration_force_groups(groups);
    }
    fn step(&self, _: &mut Tensor<B, 2>, _: &mut Tensor<B, 2>, _: &Tensor<B, 2>, _: &Tensor<B, 1>) {
        panic!("CustomIntegrator requires a Context");
    }
}

#[cfg(all(test, feature = "ndarray"))]
mod tests;
