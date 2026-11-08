use super::{err, extract_force_snapshot, tabulated, B};
use openmm_core::CustomCVForce;
use pyo3::prelude::*;

#[pyclass(unsendable, name = "CustomCVForce")]
pub(crate) struct PyCustomCVForce {
    inner: CustomCVForce<B>,
    variables: Vec<Py<PyAny>>,
    force_owner: super::ownership::ForceOwner,
    tables: Vec<tabulated::SharedTable>,
}

impl PyCustomCVForce {
    pub(crate) fn snapshot(&self, py: Python<'_>) -> PyResult<CustomCVForce<B>> {
        let mut force = self.inner.clone();
        for (index, variable) in self.variables.iter().enumerate() {
            force.replace_collective_variable(index, extract_force_snapshot(variable.bind(py))?);
        }
        for (index, table) in self.tables.iter().enumerate() {
            force
                .set_tabulated_function(index, table.borrow().clone())
                .map_err(|e| err(&e.to_string()))?;
        }
        Ok(force)
    }
}

#[pymethods]
impl PyCustomCVForce {
    #[new]
    fn new(energy: &str) -> PyResult<Self> {
        Ok(Self {
            inner: CustomCVForce::new(energy).map_err(|e| err(&e.to_string()))?,
            variables: Vec::new(),
            force_owner: super::ownership::ForceOwner::default(),
            tables: Vec::new(),
        })
    }
    fn get_force_group(&self) -> u32 {
        self.inner.force_group()
    }
    fn set_force_group(&mut self, group: u32) {
        self.inner.set_force_group(group);
    }
    fn get_name(&self) -> &str {
        self.inner.name()
    }
    fn set_name(&mut self, name: &str) {
        self.inner.set_name(name);
    }
    fn get_energy_function(&self) -> &str {
        self.inner.energy_function()
    }
    fn set_energy_function(&mut self, energy: &str) -> PyResult<()> {
        self.inner.set_energy_function(energy).map_err(|e| err(&e))
    }
    fn uses_periodic_boundary_conditions(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(self.snapshot(py)?.uses_periodic_boundary_conditions())
    }
    fn add_collective_variable(&mut self, name: &str, force: &Bound<'_, PyAny>) -> PyResult<usize> {
        if self.variables.len() >= 32 {
            return Err(err(
                "CustomCVForce cannot have more than 32 collective variables",
            ));
        }
        let snapshot = extract_force_snapshot(force)?;
        self.force_owner.claim(force)?;
        let index = self.inner.add_collective_variable(name, snapshot);
        self.variables.push(force.clone().unbind());
        Ok(index)
    }
    fn get_num_collective_variables(&self) -> usize {
        self.inner.num_collective_variables()
    }
    fn get_collective_variable_name(&self, index: usize) -> &str {
        self.inner.collective_variable_name(index)
    }
    fn get_collective_variable(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
        self.variables
            .get(index)
            .map(|variable| variable.clone_ref(py))
            .ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err("collective variable index out of range")
            })
    }
    fn get_force_cache_bytes(&self) -> usize {
        self.inner.force_cache_bytes()
    }
    fn set_force_cache_bytes(&mut self, bytes: usize) {
        self.inner.set_force_cache_bytes(bytes);
    }
    fn add_global_parameter(&mut self, name: &str, value: f64) -> usize {
        self.inner.add_global_parameter(name, value)
    }
    fn get_num_global_parameters(&self) -> usize {
        self.inner.num_global_parameters()
    }
    fn get_global_parameter_name(&self, index: usize) -> &str {
        self.inner.global_parameter_name(index)
    }
    fn get_global_parameter_default_value(&self, index: usize) -> f64 {
        self.inner.global_parameter_default_value(index)
    }
    fn set_global_parameter_default_value(&mut self, index: usize, value: f64) {
        self.inner.set_global_parameter_default_value(index, value);
    }
    fn add_energy_parameter_derivative(&mut self, name: &str) {
        self.inner.add_energy_parameter_derivative(name);
    }
    fn get_num_energy_parameter_derivatives(&self) -> usize {
        self.inner.num_energy_parameter_derivatives()
    }
    fn get_energy_parameter_derivative_name(&self, index: usize) -> &str {
        self.inner.energy_parameter_derivative_name(index)
    }
    fn add_tabulated_function(
        &mut self,
        name: &str,
        function: &Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        let table = tabulated::extract(function)?;
        let index = self
            .inner
            .add_tabulated_function(name, table.borrow().clone())
            .map_err(|e| err(&e.to_string()))?;
        self.tables.push(table);
        Ok(index)
    }
    fn get_num_tabulated_functions(&self) -> usize {
        self.tables.len()
    }
    fn get_tabulated_function_name(&self, index: usize) -> &str {
        self.inner.tabulated_function_name(index)
    }
    fn get_tabulated_function(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
        let table = self.tables.get(index).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err("tabulated function index out of range")
        })?;
        tabulated::wrap(py, table.clone())
    }
}
