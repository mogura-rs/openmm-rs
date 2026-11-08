use super::barostat::{validate_pressure, validate_temperature};
use super::{err, PyContext};
use openmm_core::MonteCarloAnisotropicBarostat;
use pyo3::prelude::*;

#[pyclass(unsendable, name = "MonteCarloAnisotropicBarostat")]
pub(crate) struct PyMonteCarloAnisotropicBarostat {
    pub(crate) inner: MonteCarloAnisotropicBarostat,
}

#[pymethods]
impl PyMonteCarloAnisotropicBarostat {
    #[new]
    #[pyo3(signature = (default_pressure, default_temperature, scale_x=true, scale_y=true, scale_z=true, frequency=25, scale_molecules_as_rigid=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        default_pressure: [f64; 3],
        default_temperature: f64,
        scale_x: bool,
        scale_y: bool,
        scale_z: bool,
        frequency: u32,
        scale_molecules_as_rigid: bool,
    ) -> PyResult<Self> {
        for p in default_pressure {
            validate_pressure(p)?;
        }
        validate_temperature(default_temperature)?;
        let mut inner = MonteCarloAnisotropicBarostat::new(
            default_pressure,
            default_temperature,
            [scale_x, scale_y, scale_z],
            frequency,
        );
        inner.set_scale_molecules_as_rigid(scale_molecules_as_rigid);
        Ok(Self { inner })
    }
    #[staticmethod]
    fn pressure_x() -> &'static str {
        MonteCarloAnisotropicBarostat::PRESSURE_X
    }
    #[staticmethod]
    fn pressure_y() -> &'static str {
        MonteCarloAnisotropicBarostat::PRESSURE_Y
    }
    #[staticmethod]
    fn pressure_z() -> &'static str {
        MonteCarloAnisotropicBarostat::PRESSURE_Z
    }
    #[staticmethod]
    fn temperature() -> &'static str {
        MonteCarloAnisotropicBarostat::TEMPERATURE
    }
    fn get_default_pressure(&self) -> [f64; 3] {
        self.inner.default_pressure()
    }
    fn set_default_pressure(&mut self, value: [f64; 3]) -> PyResult<()> {
        for p in value {
            validate_pressure(p)?;
        }
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
    fn get_scale_x(&self) -> bool {
        self.inner.scale_axes()[0]
    }
    fn get_scale_y(&self) -> bool {
        self.inner.scale_axes()[1]
    }
    fn get_scale_z(&self) -> bool {
        self.inner.scale_axes()[2]
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
    ) -> PyResult<[f64; 3]> {
        let index = context
            .force_objects
            .iter()
            .position(|force| force.as_ptr() == slf.as_ptr())
            .ok_or_else(|| err("this Force is not present in the Context"))?;
        context
            .inner
            .borrow()
            .compute_current_anisotropic_pressure(index)
            .map_err(|error| err(&error))
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMonteCarloAnisotropicBarostat>()?;
    let class = module.py().get_type::<PyMonteCarloAnisotropicBarostat>();
    for (snake, camel) in [
        ("pressure_x", "PressureX"),
        ("pressure_y", "PressureY"),
        ("pressure_z", "PressureZ"),
        ("get_scale_x", "getScaleX"),
        ("get_scale_y", "getScaleY"),
        ("get_scale_z", "getScaleZ"),
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
    ] {
        class.setattr(camel, class.getattr(snake)?)?;
    }
    Ok(())
}
