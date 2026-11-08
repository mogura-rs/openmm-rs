use super::{err, tabulated};
use openmm_core::CustomCompoundBondForce;
use pyo3::prelude::*;

#[pyclass(unsendable, name = "CustomCompoundBondForce")]
pub(crate) struct PyCustomCompoundBondForce {
    inner: CustomCompoundBondForce,
    tables: Vec<tabulated::SharedTable>,
}
impl PyCustomCompoundBondForce {
    pub(crate) fn snapshot(&self) -> CustomCompoundBondForce {
        let mut force = self.inner.clone();
        for (index, table) in self.tables.iter().enumerate() {
            force
                .set_tabulated_function(index, table.borrow().clone())
                .unwrap();
        }
        force
    }
    fn inner_mut(&mut self) -> &mut CustomCompoundBondForce {
        &mut self.inner
    }

    fn inner_ref(&self) -> &CustomCompoundBondForce {
        &self.inner
    }
}

#[pymethods]
impl PyCustomCompoundBondForce {
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

    #[new]
    fn new(particles_per_bond: usize, energy: &str) -> PyResult<Self> {
        let inner = CustomCompoundBondForce::new(particles_per_bond, energy)
            .map_err(|e| err(&e.to_string()))?;
        Ok(Self {
            inner,
            tables: Vec::new(),
        })
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
            .map_err(|error| err(&error.to_string()))?;
        self.tables.push(table);
        Ok(index)
    }
    fn get_num_tabulated_functions(&self) -> usize {
        self.tables.len()
    }
    fn get_tabulated_function_name(&self, index: usize) -> PyResult<String> {
        self.tables.get(index).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err("tabulated function index out of range")
        })?;
        Ok(self.inner.tabulated_function_name(index).to_owned())
    }
    fn get_tabulated_function(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
        let table = self.tables.get(index).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err("tabulated function index out of range")
        })?;
        tabulated::wrap(py, table.clone())
    }

    fn add_per_bond_parameter(&mut self, name: &str) -> usize {
        self.inner_mut().add_per_bond_parameter(name)
    }

    fn num_per_bond_parameters(&mut self) -> usize {
        self.inner_mut().num_per_bond_parameters()
    }

    fn get_per_bond_parameter_name(&mut self, index: usize) -> String {
        self.inner_mut().per_bond_parameter_name(index).to_string()
    }

    fn add_global_parameter(&mut self, name: &str, value: f64) {
        self.inner_mut().add_global_parameter(name, value);
    }

    fn num_global_parameters(&mut self) -> usize {
        self.inner_mut().num_global_parameters()
    }

    fn get_global_parameter_name(&mut self, index: usize) -> String {
        self.inner_mut().global_parameter_name(index).to_string()
    }

    fn get_global_parameter_default_value(&mut self, index: usize) -> f64 {
        self.inner_mut().global_parameter_default_value(index)
    }

    fn set_global_parameter_default_value(&mut self, index: usize, value: f64) {
        self.inner_mut()
            .set_global_parameter_default_value(index, value);
    }

    fn add_energy_parameter_derivative(&mut self, name: &str) {
        self.inner_mut().add_energy_parameter_derivative(name);
    }

    fn num_energy_parameter_derivatives(&mut self) -> usize {
        self.inner_mut().num_energy_parameter_derivatives()
    }

    fn get_energy_parameter_derivative_name(&mut self, index: usize) -> String {
        self.inner_mut()
            .energy_parameter_derivative_name(index)
            .to_string()
    }

    fn get_energy_function(&self) -> &str {
        self.inner.energy_function()
    }
    fn set_energy_function(&mut self, energy: &str) -> PyResult<()> {
        self.inner.set_energy_function(energy).map_err(|e| err(&e))
    }
    fn get_num_particles_per_bond(&self) -> usize {
        self.inner.num_particles_per_bond()
    }
    fn add_bond(&mut self, particles: Vec<usize>, parameters: Vec<f64>) -> usize {
        self.inner.add_bond(particles, parameters)
    }
    fn get_num_bonds(&self) -> usize {
        self.inner.num_bonds()
    }
    fn get_bond_parameters(&self, index: usize) -> (Vec<usize>, Vec<f64>) {
        let (particles, parameters) = self.inner.bond_parameters(index);
        (particles.to_vec(), parameters.to_vec())
    }
    fn set_bond_parameters(&mut self, index: usize, particles: Vec<usize>, parameters: Vec<f64>) {
        self.inner.set_bond_parameters(index, particles, parameters);
    }
    fn set_uses_periodic_boundary_conditions(&mut self, periodic: bool) {
        self.inner.set_uses_periodic_boundary_conditions(periodic);
    }
    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.inner_ref().uses_periodic_boundary_conditions()
    }
}
