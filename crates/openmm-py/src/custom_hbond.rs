use super::{err, tabulated};
use openmm_core::{CustomHbondForce, CustomHbondMethod};
use pyo3::prelude::*;

#[pyclass(unsendable, name = "CustomHbondForce")]
pub(crate) struct PyCustomHbondForce {
    inner: CustomHbondForce,
    tables: Vec<tabulated::SharedTable>,
}
impl PyCustomHbondForce {
    pub(crate) fn snapshot(&self) -> CustomHbondForce {
        let mut force = self.inner.clone();
        for (index, table) in self.tables.iter().enumerate() {
            force
                .set_tabulated_function(index, table.borrow().clone())
                .unwrap();
        }
        force
    }
    fn inner_mut(&mut self) -> &mut CustomHbondForce {
        &mut self.inner
    }

    fn inner_ref(&self) -> &CustomHbondForce {
        &self.inner
    }
}

#[pymethods]
impl PyCustomHbondForce {
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
    fn new(energy: &str) -> PyResult<Self> {
        let inner = CustomHbondForce::new(energy).map_err(|e| err(&e.to_string()))?;
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

    #[classattr]
    #[pyo3(name = "NoCutoff")]
    const NO_CUTOFF: i32 = 0;
    #[classattr]
    #[pyo3(name = "CutoffNonPeriodic")]
    const CUTOFF_NON_PERIODIC: i32 = 1;
    #[classattr]
    #[pyo3(name = "CutoffPeriodic")]
    const CUTOFF_PERIODIC: i32 = 2;
    fn get_energy_function(&self) -> &str {
        self.inner.energy_function()
    }
    fn set_energy_function(&mut self, energy: &str) -> PyResult<()> {
        self.inner.set_energy_function(energy).map_err(|e| err(&e))
    }
    fn set_nonbonded_method(&mut self, method: i32) -> PyResult<()> {
        self.inner.set_nonbonded_method(match method {
            0 => CustomHbondMethod::NoCutoff,
            1 => CustomHbondMethod::CutoffNonPeriodic,
            2 => CustomHbondMethod::CutoffPeriodic,
            _ => return Err(err("unknown hydrogen bond method")),
        });
        Ok(())
    }
    fn get_nonbonded_method(&self) -> i32 {
        match self.inner.nonbonded_method() {
            CustomHbondMethod::NoCutoff => 0,
            CustomHbondMethod::CutoffNonPeriodic => 1,
            CustomHbondMethod::CutoffPeriodic => 2,
        }
    }
    fn set_cutoff_distance(&mut self, distance: f64) {
        self.inner.set_cutoff_distance(distance);
    }
    fn get_cutoff_distance(&self) -> f64 {
        self.inner.cutoff_distance()
    }
    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.inner_ref().uses_periodic_boundary_conditions()
    }
    fn set_use_neighbor_cells(&mut self, enabled: bool) {
        self.inner.set_use_neighbor_cells(enabled);
    }
    fn add_exclusion(&mut self, donor: usize, acceptor: usize) -> usize {
        self.inner.add_exclusion(donor, acceptor)
    }
    fn get_num_exclusions(&self) -> usize {
        self.inner.num_exclusions()
    }
    fn get_exclusion_particles(&self, index: usize) -> (usize, usize) {
        self.inner.exclusion_particles(index)
    }
    fn set_exclusion_particles(&mut self, index: usize, donor: usize, acceptor: usize) {
        self.inner.set_exclusion_particles(index, donor, acceptor);
    }

    fn add_per_donor_parameter(&mut self, name: &str) -> usize {
        self.inner.add_per_donor_parameter(name)
    }
    fn get_num_per_donor_parameters(&self) -> usize {
        self.inner.num_per_donor_parameters()
    }
    fn get_per_donor_parameter_name(&self, index: usize) -> &str {
        self.inner.per_donor_parameter_name(index)
    }
    fn add_donor(
        &mut self,
        first: usize,
        second: isize,
        third: isize,
        parameters: Vec<f64>,
    ) -> PyResult<usize> {
        if second < -1 || third < -1 {
            return Err(err("optional particle indices must be -1 or nonnegative"));
        }
        Ok(self.inner.add_donor(
            first,
            usize::try_from(second).ok(),
            usize::try_from(third).ok(),
            parameters,
        ))
    }
    fn get_num_donors(&self) -> usize {
        self.inner.num_donors()
    }
    fn get_donor_parameters(&self, index: usize) -> (usize, isize, isize, Vec<f64>) {
        let (atoms, parameters) = self.inner.donor_parameters(index);
        (
            atoms[0].unwrap(),
            atoms[1].map_or(-1, |p| p as isize),
            atoms[2].map_or(-1, |p| p as isize),
            parameters.to_vec(),
        )
    }
    fn set_donor_parameters(
        &mut self,
        index: usize,
        first: usize,
        second: isize,
        third: isize,
        parameters: Vec<f64>,
    ) -> PyResult<()> {
        if second < -1 || third < -1 {
            return Err(err("optional particle indices must be -1 or nonnegative"));
        }
        self.inner.set_donor_parameters(
            index,
            first,
            usize::try_from(second).ok(),
            usize::try_from(third).ok(),
            parameters,
        );
        Ok(())
    }

    fn add_per_acceptor_parameter(&mut self, name: &str) -> usize {
        self.inner.add_per_acceptor_parameter(name)
    }
    fn get_num_per_acceptor_parameters(&self) -> usize {
        self.inner.num_per_acceptor_parameters()
    }
    fn get_per_acceptor_parameter_name(&self, index: usize) -> &str {
        self.inner.per_acceptor_parameter_name(index)
    }
    fn add_acceptor(
        &mut self,
        first: usize,
        second: isize,
        third: isize,
        parameters: Vec<f64>,
    ) -> PyResult<usize> {
        if second < -1 || third < -1 {
            return Err(err("optional particle indices must be -1 or nonnegative"));
        }
        Ok(self.inner.add_acceptor(
            first,
            usize::try_from(second).ok(),
            usize::try_from(third).ok(),
            parameters,
        ))
    }
    fn get_num_acceptors(&self) -> usize {
        self.inner.num_acceptors()
    }
    fn get_acceptor_parameters(&self, index: usize) -> (usize, isize, isize, Vec<f64>) {
        let (atoms, parameters) = self.inner.acceptor_parameters(index);
        (
            atoms[0].unwrap(),
            atoms[1].map_or(-1, |p| p as isize),
            atoms[2].map_or(-1, |p| p as isize),
            parameters.to_vec(),
        )
    }
    fn set_acceptor_parameters(
        &mut self,
        index: usize,
        first: usize,
        second: isize,
        third: isize,
        parameters: Vec<f64>,
    ) -> PyResult<()> {
        if second < -1 || third < -1 {
            return Err(err("optional particle indices must be -1 or nonnegative"));
        }
        self.inner.set_acceptor_parameters(
            index,
            first,
            usize::try_from(second).ok(),
            usize::try_from(third).ok(),
            parameters,
        );
        Ok(())
    }
}
