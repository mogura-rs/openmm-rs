use super::barostat::{validate_pressure, validate_temperature};
use super::{err, PyContext};
use openmm_core::{MembraneXYMode, MembraneZMode, MonteCarloMembraneBarostat};
use pyo3::prelude::*;

#[pyclass(unsendable, name = "MonteCarloMembraneBarostat")]
pub(crate) struct PyMonteCarloMembraneBarostat {
    pub(crate) inner: MonteCarloMembraneBarostat,
}

fn parse_xy_mode(mode: u32) -> PyResult<MembraneXYMode> {
    match mode {
        0 => Ok(MembraneXYMode::Isotropic),
        1 => Ok(MembraneXYMode::Anisotropic),
        _ => Err(err("invalid XY mode")),
    }
}
fn parse_z_mode(mode: u32) -> PyResult<MembraneZMode> {
    match mode {
        0 => Ok(MembraneZMode::Free),
        1 => Ok(MembraneZMode::Fixed),
        2 => Ok(MembraneZMode::ConstantVolume),
        _ => Err(err("invalid Z mode")),
    }
}
fn validate_tension(value: f64) -> PyResult<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(err("surface tension must be finite"))
    }
}

#[pymethods]
impl PyMonteCarloMembraneBarostat {
    #[new]
    #[pyo3(signature = (default_pressure, default_surface_tension, default_temperature, xy_mode, z_mode, frequency=25, scale_molecules_as_rigid=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        default_pressure: f64,
        default_surface_tension: f64,
        default_temperature: f64,
        xy_mode: u32,
        z_mode: u32,
        frequency: u32,
        scale_molecules_as_rigid: bool,
    ) -> PyResult<Self> {
        validate_pressure(default_pressure)?;
        validate_temperature(default_temperature)?;
        validate_tension(default_surface_tension)?;
        let mut inner = MonteCarloMembraneBarostat::new(
            default_pressure,
            default_surface_tension,
            default_temperature,
            parse_xy_mode(xy_mode)?,
            parse_z_mode(z_mode)?,
            frequency,
        );
        inner.set_scale_molecules_as_rigid(scale_molecules_as_rigid);
        Ok(Self { inner })
    }
    #[staticmethod]
    fn pressure() -> &'static str {
        MonteCarloMembraneBarostat::PRESSURE
    }
    #[staticmethod]
    fn surface_tension() -> &'static str {
        MonteCarloMembraneBarostat::SURFACE_TENSION
    }
    #[staticmethod]
    fn temperature() -> &'static str {
        MonteCarloMembraneBarostat::TEMPERATURE
    }
    fn get_default_pressure(&self) -> f64 {
        self.inner.default_pressure()
    }
    fn set_default_pressure(&mut self, value: f64) -> PyResult<()> {
        validate_pressure(value)?;
        self.inner.set_default_pressure(value);
        Ok(())
    }
    fn get_default_surface_tension(&self) -> f64 {
        self.inner.default_surface_tension()
    }
    fn set_default_surface_tension(&mut self, value: f64) -> PyResult<()> {
        validate_tension(value)?;
        self.inner.set_default_surface_tension(value);
        Ok(())
    }
    fn get_xy_mode(&self) -> u32 {
        self.inner.xy_mode() as u32
    }
    fn set_xy_mode(&self, mode: u32) -> PyResult<()> {
        self.inner.set_xy_mode(parse_xy_mode(mode)?);
        Ok(())
    }
    fn get_z_mode(&self) -> u32 {
        self.inner.z_mode() as u32
    }
    fn set_z_mode(&self, mode: u32) -> PyResult<()> {
        self.inner.set_z_mode(parse_z_mode(mode)?);
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
    ) -> PyResult<[f64; 3]> {
        let index = context
            .force_objects
            .iter()
            .position(|force| force.as_ptr() == slf.as_ptr())
            .ok_or_else(|| err("this Force is not present in the Context"))?;
        context
            .inner
            .borrow()
            .compute_current_membrane_pressure(index)
            .map_err(|error| err(&error))
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMonteCarloMembraneBarostat>()?;
    let class = module.py().get_type::<PyMonteCarloMembraneBarostat>();
    for (name, value) in [
        ("XYIsotropic", 0),
        ("XYAnisotropic", 1),
        ("ZFree", 0),
        ("ZFixed", 1),
        ("ConstantVolume", 2),
    ] {
        class.setattr(name, value)?;
    }
    for (snake, camel) in [
        ("pressure", "Pressure"),
        ("surface_tension", "SurfaceTension"),
        ("get_default_surface_tension", "getDefaultSurfaceTension"),
        ("set_default_surface_tension", "setDefaultSurfaceTension"),
        ("get_xy_mode", "getXYMode"),
        ("set_xy_mode", "setXYMode"),
        ("get_z_mode", "getZMode"),
        ("set_z_mode", "setZMode"),
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
