use super::{err, tabulated, PyContext};
use openmm_core::{CustomGBForce, CustomGBMethod, GBComputationType};
use pyo3::prelude::*;

#[pyclass(unsendable, name = "CustomGBForce")]
pub(crate) struct PyCustomGBForce {
    inner: CustomGBForce,
    tables: Vec<tabulated::SharedTable>,
}

impl PyCustomGBForce {
    pub(crate) fn snapshot(&self) -> PyResult<CustomGBForce> {
        let mut force = self.inner.clone();
        for (index, table) in self.tables.iter().enumerate() {
            force
                .set_tabulated_function(index, table.borrow().clone())
                .map_err(|e| err(&e))?;
        }
        Ok(force)
    }
}
fn computation(kind: u32) -> PyResult<GBComputationType> {
    match kind {
        0 => Ok(GBComputationType::SingleParticle),
        1 => Ok(GBComputationType::ParticlePair),
        2 => Ok(GBComputationType::ParticlePairNoExclusions),
        _ => Err(err("invalid CustomGB computation type")),
    }
}
fn check(index: usize, count: usize) -> PyResult<()> {
    if index < count {
        Ok(())
    } else {
        Err(pyo3::exceptions::PyIndexError::new_err(
            "CustomGB index out of range",
        ))
    }
}

#[pymethods]
impl PyCustomGBForce {
    #[new]
    fn new() -> Self {
        Self {
            inner: CustomGBForce::new(),
            tables: Vec::new(),
        }
    }
    fn add_computed_value(&mut self, name: &str, expression: &str, kind: u32) -> PyResult<usize> {
        self.inner
            .add_computed_value(name, expression, computation(kind)?)
            .map_err(|e| err(&e))
    }
    fn get_num_computed_values(&self) -> usize {
        self.inner.num_computed_values()
    }
    fn get_computed_value_parameters(&self, index: usize) -> PyResult<(String, String, u32)> {
        check(index, self.inner.num_computed_values())?;
        let (name, source, kind) = self.inner.computed_value_parameters(index);
        Ok((name.to_owned(), source.to_owned(), kind as u32))
    }
    fn set_computed_value_parameters(
        &mut self,
        index: usize,
        name: &str,
        expression: &str,
        kind: u32,
    ) -> PyResult<()> {
        check(index, self.inner.num_computed_values())?;
        self.inner
            .set_computed_value_parameters(index, name, expression, computation(kind)?)
            .map_err(|e| err(&e))
    }
    fn add_energy_term(&mut self, expression: &str, kind: u32) -> PyResult<usize> {
        self.inner
            .add_energy_term(expression, computation(kind)?)
            .map_err(|e| err(&e))
    }
    fn get_num_energy_terms(&self) -> usize {
        self.inner.num_energy_terms()
    }
    fn get_energy_term_parameters(&self, index: usize) -> PyResult<(String, u32)> {
        check(index, self.inner.num_energy_terms())?;
        let (source, kind) = self.inner.energy_term_parameters(index);
        Ok((source.to_owned(), kind as u32))
    }
    fn set_energy_term_parameters(
        &mut self,
        index: usize,
        expression: &str,
        kind: u32,
    ) -> PyResult<()> {
        check(index, self.inner.num_energy_terms())?;
        self.inner
            .set_energy_term_parameters(index, expression, computation(kind)?)
            .map_err(|e| err(&e))
    }
    fn add_per_particle_parameter(&mut self, name: &str) -> usize {
        self.inner.add_per_particle_parameter(name)
    }
    fn get_num_per_particle_parameters(&self) -> usize {
        self.inner.num_per_particle_parameters()
    }
    fn get_per_particle_parameter_name(&self, index: usize) -> PyResult<&str> {
        check(index, self.inner.num_per_particle_parameters())?;
        Ok(self.inner.per_particle_parameter_name(index))
    }
    fn set_per_particle_parameter_name(&mut self, index: usize, name: &str) -> PyResult<()> {
        check(index, self.inner.num_per_particle_parameters())?;
        self.inner.set_per_particle_parameter_name(index, name);
        Ok(())
    }
    fn add_global_parameter(&mut self, name: &str, value: f64) -> usize {
        self.inner.add_global_parameter(name, value)
    }
    fn get_num_global_parameters(&self) -> usize {
        self.inner.num_global_parameters()
    }
    fn get_global_parameter_name(&self, index: usize) -> PyResult<&str> {
        check(index, self.inner.num_global_parameters())?;
        Ok(self.inner.global_parameter_name(index))
    }
    fn set_global_parameter_name(&mut self, index: usize, name: &str) -> PyResult<()> {
        check(index, self.inner.num_global_parameters())?;
        self.inner.set_global_parameter_name(index, name);
        Ok(())
    }
    fn get_global_parameter_default_value(&self, index: usize) -> PyResult<f64> {
        check(index, self.inner.num_global_parameters())?;
        Ok(self.inner.global_parameter_default_value(index))
    }
    fn set_global_parameter_default_value(&mut self, index: usize, value: f64) -> PyResult<()> {
        check(index, self.inner.num_global_parameters())?;
        self.inner.set_global_parameter_default_value(index, value);
        Ok(())
    }
    fn add_energy_parameter_derivative(&mut self, name: &str) -> PyResult<()> {
        if !(0..self.inner.num_global_parameters())
            .any(|i| self.inner.global_parameter_name(i) == name)
        {
            return Err(err("unknown global parameter"));
        }
        self.inner.add_energy_parameter_derivative(name);
        Ok(())
    }
    fn get_num_energy_parameter_derivatives(&self) -> usize {
        self.inner.num_energy_parameter_derivatives()
    }
    fn get_energy_parameter_derivative_name(&self, index: usize) -> PyResult<&str> {
        check(index, self.inner.num_energy_parameter_derivatives())?;
        Ok(self.inner.energy_parameter_derivative_name(index))
    }
    fn add_particle(&mut self, parameters: Vec<f64>) -> usize {
        self.inner.add_particle(parameters)
    }
    fn get_num_particles(&self) -> usize {
        self.inner.num_particles()
    }
    fn get_particle_parameters(&self, index: usize) -> PyResult<Vec<f64>> {
        check(index, self.inner.num_particles())?;
        Ok(self.inner.particle_parameters(index).to_vec())
    }
    fn set_particle_parameters(&mut self, index: usize, parameters: Vec<f64>) -> PyResult<()> {
        check(index, self.inner.num_particles())?;
        self.inner.set_particle_parameters(index, parameters);
        Ok(())
    }
    fn add_exclusion(&mut self, first: usize, second: usize) -> usize {
        self.inner.add_exclusion(first, second)
    }
    fn get_num_exclusions(&self) -> usize {
        self.inner.num_exclusions()
    }
    fn get_exclusion_particles(&self, index: usize) -> PyResult<(usize, usize)> {
        check(index, self.inner.num_exclusions())?;
        Ok(self.inner.exclusion_particles(index))
    }
    fn set_exclusion_particles(
        &mut self,
        index: usize,
        first: usize,
        second: usize,
    ) -> PyResult<()> {
        check(index, self.inner.num_exclusions())?;
        self.inner.set_exclusion_particles(index, first, second);
        Ok(())
    }
    fn get_nonbonded_method(&self) -> u32 {
        self.inner.nonbonded_method() as u32
    }
    fn set_nonbonded_method(&mut self, method: u32) -> PyResult<()> {
        self.inner.set_nonbonded_method(match method {
            0 => CustomGBMethod::NoCutoff,
            1 => CustomGBMethod::CutoffNonPeriodic,
            2 => CustomGBMethod::CutoffPeriodic,
            _ => return Err(err("invalid CustomGB nonbonded method")),
        });
        Ok(())
    }
    fn get_cutoff_distance(&self) -> f64 {
        self.inner.cutoff_distance()
    }
    fn set_cutoff_distance(&mut self, value: f64) {
        self.inner.set_cutoff_distance(value);
    }
    fn get_force_group(&self) -> u32 {
        self.inner.force_group()
    }
    fn set_force_group(&mut self, group: u32) -> PyResult<()> {
        if group >= 32 {
            return Err(err("force group must be between zero and 31"));
        }
        self.inner.set_force_group(group);
        Ok(())
    }
    fn get_name(&self) -> &str {
        self.inner.name()
    }
    fn set_name(&mut self, name: &str) {
        self.inner.set_name(name);
    }
    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.inner.uses_periodic_boundary_conditions()
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
            .map_err(|e| err(&e))?;
        self.tables.push(table);
        Ok(index)
    }
    fn add_function(
        &mut self,
        name: &str,
        values: Vec<f64>,
        min: f64,
        max: f64,
    ) -> PyResult<usize> {
        let index = self
            .inner
            .add_function(name, values, min, max)
            .map_err(|e| err(&e))?;
        self.tables.push(std::rc::Rc::new(std::cell::RefCell::new(
            self.inner.tabulated_function(index).clone(),
        )));
        Ok(index)
    }
    fn get_num_functions(&self) -> usize {
        self.tables.len()
    }
    fn get_function_parameters(&self, index: usize) -> PyResult<(String, Vec<f64>, f64, f64)> {
        let snapshot = self.snapshot()?;
        let (name, values, min, max) = snapshot.function_parameters(index).map_err(|e| err(&e))?;
        Ok((name.to_owned(), values.to_vec(), min, max))
    }
    fn set_function_parameters(
        &mut self,
        index: usize,
        name: &str,
        values: Vec<f64>,
        min: f64,
        max: f64,
    ) -> PyResult<()> {
        check(index, self.tables.len())?;
        let mut snapshot = self.snapshot()?;
        snapshot
            .set_function_parameters(index, name, values, min, max)
            .map_err(|e| err(&e))?;
        *self.tables[index].borrow_mut() = snapshot.tabulated_function(index).clone();
        self.inner = snapshot;
        Ok(())
    }
    fn get_num_tabulated_functions(&self) -> usize {
        self.tables.len()
    }
    fn get_tabulated_function_name(&self, index: usize) -> PyResult<&str> {
        check(index, self.tables.len())?;
        Ok(self.inner.tabulated_function_name(index))
    }
    fn get_tabulated_function(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
        check(index, self.tables.len())?;
        tabulated::wrap(py, self.tables[index].clone())
    }
    fn update_parameters_in_context(
        slf: PyRef<'_, Self>,
        context: PyRef<'_, PyContext>,
    ) -> PyResult<()> {
        let index = context
            .force_objects
            .iter()
            .position(|force| force.as_ptr() == slf.as_ptr())
            .ok_or_else(|| err("this Force is not present in the Context"))?;
        context
            .inner
            .borrow_mut()
            .update_force_parameters_at_path(&[index], &slf.snapshot()?)
            .map_err(|e| err(&e))
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCustomGBForce>()?;
    let class = module.py().get_type::<PyCustomGBForce>();
    for (name, value) in [
        ("NoCutoff", 0),
        ("CutoffNonPeriodic", 1),
        ("CutoffPeriodic", 2),
        ("SingleParticle", 0),
        ("ParticlePair", 1),
        ("ParticlePairNoExclusions", 2),
    ] {
        class.setattr(name, value)?;
    }
    for (snake, camel) in [
        ("add_function", "addFunction"),
        ("get_num_functions", "getNumFunctions"),
        ("get_function_parameters", "getFunctionParameters"),
        ("set_function_parameters", "setFunctionParameters"),
        ("add_computed_value", "addComputedValue"),
        ("get_num_computed_values", "getNumComputedValues"),
        (
            "get_computed_value_parameters",
            "getComputedValueParameters",
        ),
        (
            "set_computed_value_parameters",
            "setComputedValueParameters",
        ),
        ("add_energy_term", "addEnergyTerm"),
        ("get_num_energy_terms", "getNumEnergyTerms"),
        ("get_energy_term_parameters", "getEnergyTermParameters"),
        ("set_energy_term_parameters", "setEnergyTermParameters"),
        ("add_per_particle_parameter", "addPerParticleParameter"),
        (
            "get_num_per_particle_parameters",
            "getNumPerParticleParameters",
        ),
        (
            "get_per_particle_parameter_name",
            "getPerParticleParameterName",
        ),
        (
            "set_per_particle_parameter_name",
            "setPerParticleParameterName",
        ),
        ("add_global_parameter", "addGlobalParameter"),
        ("get_num_global_parameters", "getNumGlobalParameters"),
        ("get_global_parameter_name", "getGlobalParameterName"),
        ("set_global_parameter_name", "setGlobalParameterName"),
        (
            "get_global_parameter_default_value",
            "getGlobalParameterDefaultValue",
        ),
        (
            "set_global_parameter_default_value",
            "setGlobalParameterDefaultValue",
        ),
        (
            "add_energy_parameter_derivative",
            "addEnergyParameterDerivative",
        ),
        (
            "get_num_energy_parameter_derivatives",
            "getNumEnergyParameterDerivatives",
        ),
        (
            "get_energy_parameter_derivative_name",
            "getEnergyParameterDerivativeName",
        ),
        ("add_particle", "addParticle"),
        ("get_num_particles", "getNumParticles"),
        ("get_particle_parameters", "getParticleParameters"),
        ("set_particle_parameters", "setParticleParameters"),
        ("add_exclusion", "addExclusion"),
        ("get_num_exclusions", "getNumExclusions"),
        ("get_exclusion_particles", "getExclusionParticles"),
        ("set_exclusion_particles", "setExclusionParticles"),
        ("get_nonbonded_method", "getNonbondedMethod"),
        ("set_nonbonded_method", "setNonbondedMethod"),
        ("get_cutoff_distance", "getCutoffDistance"),
        ("set_cutoff_distance", "setCutoffDistance"),
        ("get_force_group", "getForceGroup"),
        ("set_force_group", "setForceGroup"),
        ("get_name", "getName"),
        ("set_name", "setName"),
        (
            "uses_periodic_boundary_conditions",
            "usesPeriodicBoundaryConditions",
        ),
        ("add_tabulated_function", "addTabulatedFunction"),
        ("get_num_tabulated_functions", "getNumTabulatedFunctions"),
        ("get_tabulated_function_name", "getTabulatedFunctionName"),
        ("get_tabulated_function", "getTabulatedFunction"),
        ("update_parameters_in_context", "updateParametersInContext"),
    ] {
        class.setattr(camel, class.getattr(snake)?)?;
    }
    Ok(())
}
