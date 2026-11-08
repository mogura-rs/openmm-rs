use super::{Integrator, IntegratorComponent};
use crate::constraints::{ConstraintError, ConstraintSolver};
use burn::prelude::*;
use std::sync::{Arc, Mutex};

/// A collection of integrators sharing one Context and its stored velocities.
pub struct CompoundIntegrator<B: Backend> {
    children: Vec<IntegratorComponent<B>>,
    current: usize,
    initialized: bool,
}

impl<B: Backend> Default for CompoundIntegrator<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> CompoundIntegrator<B> {
    pub fn new() -> Self {
        Self {
            children: Vec::new(),
            current: 0,
            initialized: false,
        }
    }

    pub fn add_integrator(&mut self, integrator: Box<dyn Integrator<B>>) -> usize {
        assert!(
            !self.initialized,
            "cannot add an integrator after initialization"
        );
        self.children.push(Arc::new(Mutex::new(integrator)));
        self.children.len() - 1
    }

    pub fn num_integrators(&self) -> usize {
        self.children.len()
    }

    pub fn integrator(&self, index: usize) -> IntegratorComponent<B> {
        self.children[index].clone()
    }

    pub fn current_integrator(&self) -> usize {
        self.current
    }

    pub fn set_current_integrator(&mut self, index: usize) {
        assert!(index < self.children.len(), "integrator index out of range");
        self.current = index;
    }
}

impl<B: Backend> Integrator<B> for CompoundIntegrator<B> {
    fn validate_system(&self, system: &crate::System<B>) -> Result<(), ConstraintError> {
        if self.children.is_empty() {
            return Err(ConstraintError(
                "CompoundIntegrator requires at least one child".into(),
            ));
        }
        for child in &self.children {
            child.lock().unwrap().validate_system(system)?;
        }
        Ok(())
    }
    fn initialize_system(&mut self, system: &crate::System<B>) {
        self.validate_system(system)
            .expect("invalid compound configuration");
        for child in &self.children {
            child.lock().unwrap().initialize_system(system);
        }
        self.initialized = true;
    }
    fn initialize(&mut self) {
        assert!(
            !self.children.is_empty(),
            "CompoundIntegrator requires at least one child"
        );
        for child in &self.children {
            child.lock().unwrap().initialize();
        }
        self.initialized = true;
    }

    fn selected_integrator(&self) -> Option<usize> {
        Some(self.current)
    }

    fn select_integrator(&mut self, index: usize) -> Result<(), ConstraintError> {
        if index >= self.children.len() {
            return Err(ConstraintError("integrator index out of range".into()));
        }
        self.current = index;
        Ok(())
    }

    fn component(&self, path: &[usize]) -> Option<IntegratorComponent<B>> {
        let (&index, rest) = path.split_first()?;
        let child = self.children.get(index)?;
        if rest.is_empty() {
            Some(child.clone())
        } else {
            child.lock().unwrap().component(rest)
        }
    }

    fn step_size(&self) -> f64 {
        self.children[self.current].lock().unwrap().step_size()
    }

    fn set_step_size(&mut self, value: f64) {
        self.children[self.current]
            .lock()
            .unwrap()
            .set_step_size(value);
    }

    fn constraint_tolerance(&self) -> f64 {
        self.children[self.current]
            .lock()
            .unwrap()
            .constraint_tolerance()
    }

    fn set_constraint_tolerance(&mut self, value: f64) {
        self.children[self.current]
            .lock()
            .unwrap()
            .set_constraint_tolerance(value);
    }

    fn integration_force_groups(&self) -> u32 {
        self.children[self.current]
            .lock()
            .unwrap()
            .integration_force_groups()
    }

    fn set_integration_force_groups(&mut self, groups: u32) {
        for child in &self.children {
            child.lock().unwrap().set_integration_force_groups(groups);
        }
    }

    fn kinetic_energy_time_offset(&self) -> f64 {
        self.children[self.current]
            .lock()
            .unwrap()
            .kinetic_energy_time_offset()
    }

    fn velocity_time_offset(&self) -> f64 {
        self.children[0].lock().unwrap().velocity_time_offset()
    }

    fn step(&self, p: &mut Tensor<B, 2>, v: &mut Tensor<B, 2>, f: &Tensor<B, 2>, m: &Tensor<B, 1>) {
        self.children[self.current].lock().unwrap().step(p, v, f, m);
    }

    fn step_constrained(
        &self,
        p: &mut Tensor<B, 2>,
        v: &mut Tensor<B, 2>,
        f: &Tensor<B, 2>,
        m: &Tensor<B, 1>,
        c: &ConstraintSolver,
    ) -> Result<(), ConstraintError> {
        self.children[self.current]
            .lock()
            .unwrap()
            .step_constrained(p, v, f, m, c)
    }

    fn step_with_limit(
        &self,
        p: &mut Tensor<B, 2>,
        v: &mut Tensor<B, 2>,
        f: &Tensor<B, 2>,
        m: &Tensor<B, 1>,
        c: &ConstraintSolver,
        limit: f64,
    ) -> Result<(), ConstraintError> {
        self.children[self.current]
            .lock()
            .unwrap()
            .step_with_limit(p, v, f, m, c, limit)
    }
}
