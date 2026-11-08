use openmm_core::AndersenThermostat;
use pyo3::prelude::*;

#[pyclass(unsendable, name = "AndersenThermostat")]
pub(crate) struct PyAndersenThermostat {
    pub(crate) inner: AndersenThermostat,
}

#[pymethods]
impl PyAndersenThermostat {
    #[new]
    fn new(default_temperature: f64, default_collision_frequency: f64) -> Self {
        Self {
            inner: AndersenThermostat::new(default_temperature, default_collision_frequency),
        }
    }
    #[staticmethod]
    fn temperature() -> &'static str {
        AndersenThermostat::TEMPERATURE
    }
    #[staticmethod]
    fn collision_frequency() -> &'static str {
        AndersenThermostat::COLLISION_FREQUENCY
    }
    fn get_default_temperature(&self) -> f64 {
        self.inner.default_temperature()
    }
    fn set_default_temperature(&mut self, value: f64) {
        self.inner.set_default_temperature(value);
    }
    fn get_default_collision_frequency(&self) -> f64 {
        self.inner.default_collision_frequency()
    }
    fn set_default_collision_frequency(&mut self, value: f64) {
        self.inner.set_default_collision_frequency(value);
    }
    fn get_random_number_seed(&self) -> i32 {
        self.inner.random_number_seed()
    }
    fn set_random_number_seed(&mut self, seed: i32) {
        self.inner.set_random_number_seed(seed);
    }
    fn get_name(&self) -> &str {
        self.inner.name()
    }
    fn set_name(&mut self, name: &str) {
        self.inner.set_name(name);
    }
    fn get_force_group(&self) -> u32 {
        self.inner.force_group()
    }
    fn set_force_group(&mut self, group: u32) {
        self.inner.set_force_group(group);
    }
    fn uses_periodic_boundary_conditions(&self) -> bool {
        false
    }
}
