use super::barostat::{validate_pressure, validate_temperature};
use super::{err, PyContext};
use openmm_core::MonteCarloFlexibleBarostat;
use pyo3::prelude::*;

#[pyclass(unsendable, name = "MonteCarloFlexibleBarostat")]
pub(crate) struct PyMonteCarloFlexibleBarostat {
    pub(crate) inner: MonteCarloFlexibleBarostat,
}

#[pymethods]
impl PyMonteCarloFlexibleBarostat {
    #[new]
    #[pyo3(signature = (default_pressure, default_temperature, frequency=25, scale_molecules_as_rigid=true))]
    fn new(
        default_pressure: f64,
        default_temperature: f64,
        frequency: u32,
        scale_molecules_as_rigid: bool,
    ) -> PyResult<Self> {
        validate_pressure(default_pressure)?;
        validate_temperature(default_temperature)?;
        let mut inner =
            MonteCarloFlexibleBarostat::new(default_pressure, default_temperature, frequency);
        inner.set_scale_molecules_as_rigid(scale_molecules_as_rigid);
        Ok(Self { inner })
    }
    #[staticmethod]
    fn pressure() -> &'static str {
        MonteCarloFlexibleBarostat::PRESSURE
    }
    #[staticmethod]
    fn temperature() -> &'static str {
        MonteCarloFlexibleBarostat::TEMPERATURE
    }
    fn get_default_pressure(&self) -> f64 {
        self.inner.default_pressure()
    }
    fn set_default_pressure(&mut self, value: f64) -> PyResult<()> {
        validate_pressure(value)?;
        self.inner.set_default_pressure(value);
        Ok(())
    }
    fn get_default_temperature(&self) -> f64 {
        self.inner.default_temperature()
    }
    fn set_default_temperature(&mut self, value: f64) -> PyResult<()> {
        validate_temperature(value)?;
        self.inner.set_default_temperature(value);
        Ok(())
    }
    fn get_frequency(&self) -> u32 {
        self.inner.frequency()
    }
    fn set_frequency(&self, value: u32) {
        self.inner.set_frequency(value);
    }
    fn get_scale_molecules_as_rigid(&self) -> bool {
        self.inner.scale_molecules_as_rigid()
    }
    fn set_scale_molecules_as_rigid(&mut self, value: bool) {
        self.inner.set_scale_molecules_as_rigid(value);
    }
    fn get_random_number_seed(&self) -> i32 {
        self.inner.random_number_seed()
    }
    fn set_random_number_seed(&mut self, value: i32) {
        self.inner.set_random_number_seed(value);
    }
    fn get_name(&self) -> &str {
        self.inner.name()
    }
    fn set_name(&mut self, value: &str) {
        self.inner.set_name(value);
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
    fn uses_periodic_boundary_conditions(&self) -> bool {
        false
    }
    fn compute_current_pressure(
        slf: PyRef<'_, Self>,
        context: PyRef<'_, PyContext>,
    ) -> PyResult<[f64; 6]> {
        let index = context
            .force_objects
            .iter()
            .position(|force| force.as_ptr() == slf.as_ptr())
            .ok_or_else(|| err("this Force is not present in the Context"))?;
        context
            .inner
            .borrow()
            .compute_current_flexible_pressure(index)
            .map_err(|error| err(&error))
    }
    #[pyo3(signature=(context,include_kinetic=false))]
    fn compute_stress_tensor(
        slf: PyRef<'_, Self>,
        context: PyRef<'_, PyContext>,
        include_kinetic: bool,
    ) -> PyResult<[f64; 6]> {
        let index = context
            .force_objects
            .iter()
            .position(|force| force.as_ptr() == slf.as_ptr())
            .ok_or_else(|| err("this Force is not present in the Context"))?;
        context
            .inner
            .borrow()
            .compute_stress_tensor(index, include_kinetic)
            .map_err(|error| err(&error))
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMonteCarloFlexibleBarostat>()?;
    let class = module.py().get_type::<PyMonteCarloFlexibleBarostat>();
    for (snake, camel) in [
        ("pressure", "Pressure"),
        ("temperature", "Temperature"),
        ("get_default_pressure", "getDefaultPressure"),
        ("set_default_pressure", "setDefaultPressure"),
        ("get_default_temperature", "getDefaultTemperature"),
        ("set_default_temperature", "setDefaultTemperature"),
        ("get_frequency", "getFrequency"),
        ("set_frequency", "setFrequency"),
        ("get_scale_molecules_as_rigid", "getScaleMoleculesAsRigid"),
        ("set_scale_molecules_as_rigid", "setScaleMoleculesAsRigid"),
        ("get_random_number_seed", "getRandomNumberSeed"),
        ("set_random_number_seed", "setRandomNumberSeed"),
        ("get_name", "getName"),
        ("set_name", "setName"),
        ("get_force_group", "getForceGroup"),
        ("set_force_group", "setForceGroup"),
        (
            "uses_periodic_boundary_conditions",
            "usesPeriodicBoundaryConditions",
        ),
        ("compute_current_pressure", "computeCurrentPressure"),
        ("compute_stress_tensor", "computeStressTensor"),
    ] {
        class.setattr(camel, class.getattr(snake)?)?;
    }
    Ok(())
}
