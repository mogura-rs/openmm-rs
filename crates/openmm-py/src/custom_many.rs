use super::{err, tabulated, PyContext};
use openmm_core::{CustomManyParticleForce, CustomManyParticleMethod, PermutationMode};
use pyo3::prelude::*;

#[pyclass(unsendable, name = "CustomManyParticleForce")]
pub(crate) struct PyCustomManyParticleForce {
    inner: CustomManyParticleForce,
    tables: Vec<tabulated::SharedTable>,
}

impl PyCustomManyParticleForce {
    pub(crate) fn snapshot(&self) -> PyResult<CustomManyParticleForce> {
        let mut force = self.inner.clone();
        for (index, table) in self.tables.iter().enumerate() {
            force
                .set_tabulated_function(index, table.borrow().clone())
                .map_err(|e| err(&e))?;
        }
        Ok(force)
    }
}
fn check(index: usize, count: usize) -> PyResult<()> {
    if index < count {
        Ok(())
    } else {
        Err(pyo3::exceptions::PyIndexError::new_err(
            "CustomManyParticle index out of range",
        ))
    }
}

#[pymethods]
impl PyCustomManyParticleForce {
    #[new]
    fn new(particles_per_set: usize, energy: &str) -> PyResult<Self> {
        Ok(Self {
            inner: CustomManyParticleForce::new(particles_per_set, energy).map_err(|e| err(&e))?,
            tables: Vec::new(),
        })
    }
    fn get_num_particles_per_set(&self) -> usize {
        self.inner.num_particles_per_set()
    }
    fn get_energy_function(&self) -> &str {
        self.inner.energy_function()
    }
    fn set_energy_function(&mut self, energy: &str) -> PyResult<()> {
        self.inner.set_energy_function(energy).map_err(|e| err(&e))
    }
    #[pyo3(signature=(parameters=Vec::new(), r#type=0))]
    fn add_particle(&mut self, parameters: Vec<f64>, r#type: i32) -> usize {
        self.inner.add_particle(parameters, r#type)
    }
    fn get_particle_parameters(&self, index: usize) -> PyResult<(Vec<f64>, i32)> {
        check(index, self.inner.num_particles())?;
        let (p, t) = self.inner.particle_parameters(index);
        Ok((p.to_vec(), t))
    }
    fn set_particle_parameters(
        &mut self,
        index: usize,
        parameters: Vec<f64>,
        r#type: i32,
    ) -> PyResult<()> {
        check(index, self.inner.num_particles())?;
        self.inner
            .set_particle_parameters(index, parameters, r#type);
        Ok(())
    }
    fn get_type_filter(&self, index: usize) -> PyResult<std::collections::HashSet<i32>> {
        check(index, self.inner.num_particles_per_set())?;
        Ok(self.inner.type_filter(index).iter().copied().collect())
    }
    fn set_type_filter(
        &mut self,
        index: usize,
        types: std::collections::HashSet<i32>,
    ) -> PyResult<()> {
        check(index, self.inner.num_particles_per_set())?;
        self.inner
            .set_type_filter(index, types.into_iter().collect());
        Ok(())
    }
    fn get_permutation_mode(&self) -> u32 {
        self.inner.permutation_mode() as u32
    }
    fn set_permutation_mode(&mut self, mode: u32) -> PyResult<()> {
        self.inner.set_permutation_mode(match mode {
            0 => PermutationMode::SinglePermutation,
            1 => PermutationMode::UniqueCentralParticle,
            _ => return Err(err("invalid CustomManyParticle permutation mode")),
        });
        Ok(())
    }
    fn create_exclusions_from_bonds(
        &mut self,
        bonds: Vec<(usize, usize)>,
        bond_cutoff: i32,
    ) -> PyResult<()> {
        if bond_cutoff < 1 {
            return Ok(());
        }
        for &(i, j) in &bonds {
            check(i, self.inner.num_particles())?;
            check(j, self.inner.num_particles())?;
        }
        self.inner
            .create_exclusions_from_bonds(&bonds, bond_cutoff as usize);
        Ok(())
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

    fn get_num_particles(&self) -> usize {
        self.inner.num_particles()
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
            0 => CustomManyParticleMethod::NoCutoff,
            1 => CustomManyParticleMethod::CutoffNonPeriodic,
            2 => CustomManyParticleMethod::CutoffPeriodic,
            _ => return Err(err("invalid CustomManyParticle nonbonded method")),
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
    module.add_class::<PyCustomManyParticleForce>()?;
    let class = module.py().get_type::<PyCustomManyParticleForce>();
    for (name, value) in [
        ("NoCutoff", 0),
        ("CutoffNonPeriodic", 1),
        ("CutoffPeriodic", 2),
        ("SinglePermutation", 0),
        ("UniqueCentralParticle", 1),
    ] {
        class.setattr(name, value)?;
    }
    for (snake, camel) in [
        ("get_num_particles_per_set", "getNumParticlesPerSet"),
        ("get_energy_function", "getEnergyFunction"),
        ("set_energy_function", "setEnergyFunction"),
        ("add_particle", "addParticle"),
        ("get_particle_parameters", "getParticleParameters"),
        ("set_particle_parameters", "setParticleParameters"),
        ("get_type_filter", "getTypeFilter"),
        ("set_type_filter", "setTypeFilter"),
        ("get_permutation_mode", "getPermutationMode"),
        ("set_permutation_mode", "setPermutationMode"),
        ("create_exclusions_from_bonds", "createExclusionsFromBonds"),
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
        ("get_num_particles", "getNumParticles"),
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
