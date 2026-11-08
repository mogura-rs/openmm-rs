use super::{err, PyContext};
use openmm_core::{GBSAOBCForce, GBSAOBCMethod};
use pyo3::prelude::*;

#[pyclass(unsendable, name = "GBSAOBCForce")]
pub(crate) struct PyGBSAOBCForce {
    pub(crate) inner: GBSAOBCForce,
}

#[pymethods]
impl PyGBSAOBCForce {
    #[new]
    fn new() -> Self {
        Self {
            inner: GBSAOBCForce::new(),
        }
    }
    #[classattr]
    #[allow(non_upper_case_globals)]
    const NoCutoff: u32 = 0;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const CutoffNonPeriodic: u32 = 1;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const CutoffPeriodic: u32 = 2;
    fn add_particle(&mut self, charge: f64, radius: f64, scaling_factor: f64) -> usize {
        self.inner.add_particle(charge, radius, scaling_factor)
    }
    fn get_num_particles(&self) -> usize {
        self.inner.num_particles()
    }
    fn get_particle_parameters(&self, index: usize) -> PyResult<(f64, f64, f64)> {
        if index >= self.inner.num_particles() {
            return Err(err("GBSA particle index out of range"));
        }
        let p = self.inner.particle_parameters(index);
        Ok((p.charge, p.radius, p.scaling_factor))
    }
    fn set_particle_parameters(
        &mut self,
        index: usize,
        charge: f64,
        radius: f64,
        scaling_factor: f64,
    ) -> PyResult<()> {
        if index >= self.inner.num_particles() {
            return Err(err("GBSA particle index out of range"));
        }
        self.inner
            .set_particle_parameters(index, charge, radius, scaling_factor);
        Ok(())
    }
    fn get_nonbonded_method(&self) -> u32 {
        self.inner.nonbonded_method() as u32
    }
    fn set_nonbonded_method(&mut self, method: u32) -> PyResult<()> {
        self.inner.set_nonbonded_method(match method {
            0 => GBSAOBCMethod::NoCutoff,
            1 => GBSAOBCMethod::CutoffNonPeriodic,
            2 => GBSAOBCMethod::CutoffPeriodic,
            _ => return Err(err("invalid GBSA nonbonded method")),
        });
        Ok(())
    }
    fn get_cutoff_distance(&self) -> f64 {
        self.inner.cutoff_distance()
    }
    fn set_cutoff_distance(&mut self, value: f64) {
        self.inner.set_cutoff_distance(value);
    }
    fn get_solvent_dielectric(&self) -> f64 {
        self.inner.solvent_dielectric()
    }
    fn set_solvent_dielectric(&mut self, value: f64) {
        self.inner.set_solvent_dielectric(value);
    }
    fn get_solute_dielectric(&self) -> f64 {
        self.inner.solute_dielectric()
    }
    fn set_solute_dielectric(&mut self, value: f64) {
        self.inner.set_solute_dielectric(value);
    }
    fn get_surface_area_energy(&self) -> f64 {
        self.inner.surface_area_energy()
    }
    fn set_surface_area_energy(&mut self, value: f64) {
        self.inner.set_surface_area_energy(value);
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
    module.add_class::<PyGBSAOBCForce>()?;
    let class = module.py().get_type::<PyGBSAOBCForce>();
    for (snake, camel) in [
        ("add_particle", "addParticle"),
        ("get_num_particles", "getNumParticles"),
        ("get_particle_parameters", "getParticleParameters"),
        ("set_particle_parameters", "setParticleParameters"),
        ("get_nonbonded_method", "getNonbondedMethod"),
        ("set_nonbonded_method", "setNonbondedMethod"),
        ("get_cutoff_distance", "getCutoffDistance"),
        ("set_cutoff_distance", "setCutoffDistance"),
        ("get_solvent_dielectric", "getSolventDielectric"),
        ("set_solvent_dielectric", "setSolventDielectric"),
        ("get_solute_dielectric", "getSoluteDielectric"),
        ("set_solute_dielectric", "setSoluteDielectric"),
        ("get_surface_area_energy", "getSurfaceAreaEnergy"),
        ("set_surface_area_energy", "setSurfaceAreaEnergy"),
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
