use super::{err, PyContext};
use openmm_core::LCPOForce;
use pyo3::prelude::*;

#[pyclass(unsendable, name = "LCPOForce")]
pub(crate) struct PyLCPOForce {
    pub(crate) inner: LCPOForce,
}

#[pymethods]
impl PyLCPOForce {
    #[new]
    fn new() -> Self {
        Self {
            inner: LCPOForce::new(),
        }
    }
    fn add_particle(&mut self, radius: f64, p1: f64, p2: f64, p3: f64, p4: f64) -> usize {
        self.inner.add_particle(radius, p1, p2, p3, p4)
    }
    fn get_num_particles(&self) -> usize {
        self.inner.num_particles()
    }
    fn get_particle_parameters(&self, index: usize) -> PyResult<(f64, f64, f64, f64, f64)> {
        if index >= self.inner.num_particles() {
            return Err(err("LCPO particle index out of range"));
        }
        let p = self.inner.particle_parameters(index);
        Ok((p.radius, p.p1, p.p2, p.p3, p.p4))
    }
    fn set_particle_parameters(
        &mut self,
        index: usize,
        radius: f64,
        p1: f64,
        p2: f64,
        p3: f64,
        p4: f64,
    ) -> PyResult<()> {
        if index >= self.inner.num_particles() {
            return Err(err("LCPO particle index out of range"));
        }
        self.inner
            .set_particle_parameters(index, radius, p1, p2, p3, p4);
        Ok(())
    }
    fn get_surface_tension(&self) -> f64 {
        self.inner.surface_tension()
    }
    fn set_surface_tension(&mut self, value: f64) {
        self.inner.set_surface_tension(value);
    }
    fn set_uses_periodic_boundary_conditions(&mut self, value: bool) {
        self.inner.set_uses_periodic_boundary_conditions(value);
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
    fn set_name(&mut self, value: &str) {
        self.inner.set_name(value);
    }
    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.inner.uses_periodic_boundary_conditions()
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
        let mut context = context.inner.borrow_mut();
        context
            .update_force_parameters_at_path(&[index], &slf.inner)
            .map_err(|e| err(&e))
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyLCPOForce>()?;
    let class = module.py().get_type::<PyLCPOForce>();
    for (snake, camel) in [
        ("add_particle", "addParticle"),
        ("get_num_particles", "getNumParticles"),
        ("get_particle_parameters", "getParticleParameters"),
        ("set_particle_parameters", "setParticleParameters"),
        ("get_surface_tension", "getSurfaceTension"),
        ("set_surface_tension", "setSurfaceTension"),
        (
            "set_uses_periodic_boundary_conditions",
            "setUsesPeriodicBoundaryConditions",
        ),
        ("get_force_group", "getForceGroup"),
        ("set_force_group", "setForceGroup"),
        ("get_name", "getName"),
        ("set_name", "setName"),
        (
            "uses_periodic_boundary_conditions",
            "usesPeriodicBoundaryConditions",
        ),
        ("update_parameters_in_context", "updateParametersInContext"),
    ] {
        class.setattr(camel, class.getattr(snake)?)?;
    }
    class.setattr("num_particles", class.getattr("get_num_particles")?)?;
    Ok(())
}
