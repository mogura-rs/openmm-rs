use super::{err, PyContext, Vec3};
use openmm_core::{ConstantPotentialForce, ConstantPotentialMethod};
use pyo3::prelude::*;
use std::collections::BTreeSet;

#[pyclass(unsendable, name = "ConstantPotentialForce")]
pub(crate) struct PyConstantPotentialForce {
    pub(crate) inner: ConstantPotentialForce,
}
fn check(index: usize, count: usize) -> PyResult<()> {
    if index < count {
        Ok(())
    } else {
        Err(pyo3::exceptions::PyIndexError::new_err(
            "constant-potential index out of range",
        ))
    }
}
fn force_index(slf: &PyRef<'_, PyConstantPotentialForce>, context: &PyContext) -> PyResult<usize> {
    context
        .force_objects
        .iter()
        .position(|force| force.as_ptr() == slf.as_ptr())
        .ok_or_else(|| err("this Force is not present in the Context"))
}

#[pymethods]
impl PyConstantPotentialForce {
    #[new]
    fn new() -> Self {
        Self {
            inner: ConstantPotentialForce::new(),
        }
    }
    #[classattr]
    const CG: u32 = 0;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const Matrix: u32 = 1;
    fn add_particle(&mut self, charge: f64) -> usize {
        self.inner.add_particle(charge)
    }
    fn get_num_particles(&self) -> usize {
        self.inner.num_particles()
    }
    fn get_particle_parameters(&self, index: usize) -> PyResult<f64> {
        check(index, self.inner.num_particles())?;
        Ok(self.inner.particle_parameters(index))
    }
    fn set_particle_parameters(&mut self, index: usize, charge: f64) -> PyResult<()> {
        check(index, self.inner.num_particles())?;
        self.inner.set_particle_parameters(index, charge);
        Ok(())
    }
    #[pyo3(signature=(particle1, particle2, charge_prod, replace=false))]
    fn add_exception(
        &mut self,
        particle1: usize,
        particle2: usize,
        charge_prod: f64,
        replace: bool,
    ) -> usize {
        self.inner
            .add_exception(particle1, particle2, charge_prod, replace)
    }
    fn get_num_exceptions(&self) -> usize {
        self.inner.num_exceptions()
    }
    fn get_exception_parameters(&self, index: usize) -> PyResult<(usize, usize, f64)> {
        check(index, self.inner.num_exceptions())?;
        let e = self.inner.exception_parameters(index);
        Ok((e.particle1, e.particle2, e.charge_prod))
    }
    fn set_exception_parameters(
        &mut self,
        index: usize,
        particle1: usize,
        particle2: usize,
        charge_prod: f64,
    ) -> PyResult<()> {
        check(index, self.inner.num_exceptions())?;
        self.inner
            .set_exception_parameters(index, particle1, particle2, charge_prod);
        Ok(())
    }
    fn create_exceptions_from_bonds(&mut self, bonds: Vec<(usize, usize)>, coulomb14scale: f64) {
        self.inner
            .create_exceptions_from_bonds(&bonds, coulomb14scale);
    }
    fn add_electrode(
        &mut self,
        particles: BTreeSet<usize>,
        potential: f64,
        gaussian_width: f64,
        thomas_fermi_scale: f64,
    ) -> usize {
        self.inner
            .add_electrode(particles, potential, gaussian_width, thomas_fermi_scale)
    }
    fn get_num_electrodes(&self) -> usize {
        self.inner.num_electrodes()
    }
    fn get_electrode_parameters(&self, index: usize) -> PyResult<(BTreeSet<usize>, f64, f64, f64)> {
        check(index, self.inner.num_electrodes())?;
        let e = self.inner.electrode_parameters(index);
        Ok((
            e.particles.clone(),
            e.potential,
            e.gaussian_width,
            e.thomas_fermi_scale,
        ))
    }
    fn set_electrode_parameters(
        &mut self,
        index: usize,
        particles: BTreeSet<usize>,
        potential: f64,
        gaussian_width: f64,
        thomas_fermi_scale: f64,
    ) -> PyResult<()> {
        check(index, self.inner.num_electrodes())?;
        self.inner.set_electrode_parameters(
            index,
            particles,
            potential,
            gaussian_width,
            thomas_fermi_scale,
        );
        Ok(())
    }
    fn get_constant_potential_method(&self) -> u32 {
        self.inner.constant_potential_method() as u32
    }
    fn set_constant_potential_method(&mut self, method: u32) -> PyResult<()> {
        self.inner.set_constant_potential_method(match method {
            0 => ConstantPotentialMethod::CG,
            1 => ConstantPotentialMethod::Matrix,
            _ => return Err(err("invalid constant-potential method")),
        });
        Ok(())
    }
    fn get_cutoff_distance(&self) -> f64 {
        self.inner.cutoff_distance()
    }
    fn set_cutoff_distance(&mut self, value: f64) {
        self.inner.set_cutoff_distance(value);
    }
    fn get_ewald_error_tolerance(&self) -> f64 {
        self.inner.ewald_error_tolerance()
    }
    fn set_ewald_error_tolerance(&mut self, value: f64) {
        self.inner.set_ewald_error_tolerance(value);
    }
    fn get_cg_error_tolerance(&self) -> f64 {
        self.inner.cg_error_tolerance()
    }
    fn set_cg_error_tolerance(&mut self, value: f64) {
        self.inner.set_cg_error_tolerance(value);
    }
    fn get_use_preconditioner(&self) -> bool {
        self.inner.use_preconditioner()
    }
    fn set_use_preconditioner(&mut self, value: bool) {
        self.inner.set_use_preconditioner(value);
    }
    fn get_use_charge_constraint(&self) -> bool {
        self.inner.use_charge_constraint()
    }
    fn set_use_charge_constraint(&mut self, value: bool) {
        self.inner.set_use_charge_constraint(value);
    }
    fn get_charge_constraint_target(&self) -> f64 {
        self.inner.charge_constraint_target()
    }
    fn set_charge_constraint_target(&mut self, value: f64) {
        self.inner.set_charge_constraint_target(value);
    }
    fn get_exceptions_use_periodic_boundary_conditions(&self) -> bool {
        self.inner.exceptions_use_periodic_boundary_conditions()
    }
    fn set_exceptions_use_periodic_boundary_conditions(&mut self, value: bool) {
        self.inner
            .set_exceptions_use_periodic_boundary_conditions(value);
    }
    fn get_pme_parameters(&self) -> (f64, usize, usize, usize) {
        let (alpha, [nx, ny, nz]) = self.inner.pme_parameters();
        (alpha, nx, ny, nz)
    }
    fn set_pme_parameters(&mut self, alpha: f64, nx: usize, ny: usize, nz: usize) {
        self.inner.set_pme_parameters(alpha, [nx, ny, nz]);
    }
    fn get_pme_parameters_in_context(
        slf: PyRef<'_, Self>,
        context: PyRef<'_, PyContext>,
    ) -> PyResult<(f64, usize, usize, usize)> {
        let index = force_index(&slf, &context)?;
        let (alpha, [nx, ny, nz]) = context
            .inner
            .borrow()
            .constant_potential_pme_parameters(&[index]);
        Ok((alpha, nx, ny, nz))
    }
    fn get_external_field(&self) -> Vec3 {
        self.inner.external_field().into()
    }
    fn set_external_field(&mut self, field: Vec3) {
        self.inner.set_external_field(field.into());
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
        true
    }
    fn get_charges(slf: PyRef<'_, Self>, context: PyRef<'_, PyContext>) -> PyResult<Vec<f64>> {
        let index = force_index(&slf, &context)?;
        Ok(context.inner.borrow().constant_potential_charges(&[index]))
    }
    fn update_parameters_in_context(
        slf: PyRef<'_, Self>,
        context: PyRef<'_, PyContext>,
    ) -> PyResult<()> {
        let index = force_index(&slf, &context)?;
        context
            .inner
            .borrow_mut()
            .update_force_parameters_at_path(&[index], &slf.inner)
            .map_err(|e| err(&e))
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyConstantPotentialForce>()?;
    let class = module.py().get_type::<PyConstantPotentialForce>();
    for (snake, camel) in ALIASES {
        class.setattr(*camel, class.getattr(*snake)?)?;
    }
    class.setattr("num_particles", class.getattr("get_num_particles")?)?;
    Ok(())
}

const ALIASES: &[(&str, &str)] = &[
    ("add_particle", "addParticle"),
    ("get_num_particles", "getNumParticles"),
    ("get_particle_parameters", "getParticleParameters"),
    ("set_particle_parameters", "setParticleParameters"),
    ("add_exception", "addException"),
    ("get_num_exceptions", "getNumExceptions"),
    ("get_exception_parameters", "getExceptionParameters"),
    ("set_exception_parameters", "setExceptionParameters"),
    ("create_exceptions_from_bonds", "createExceptionsFromBonds"),
    ("add_electrode", "addElectrode"),
    ("get_num_electrodes", "getNumElectrodes"),
    ("get_electrode_parameters", "getElectrodeParameters"),
    ("set_electrode_parameters", "setElectrodeParameters"),
    (
        "get_constant_potential_method",
        "getConstantPotentialMethod",
    ),
    (
        "set_constant_potential_method",
        "setConstantPotentialMethod",
    ),
    ("get_cutoff_distance", "getCutoffDistance"),
    ("set_cutoff_distance", "setCutoffDistance"),
    ("get_ewald_error_tolerance", "getEwaldErrorTolerance"),
    ("set_ewald_error_tolerance", "setEwaldErrorTolerance"),
    ("get_cg_error_tolerance", "getCGErrorTolerance"),
    ("set_cg_error_tolerance", "setCGErrorTolerance"),
    ("get_use_preconditioner", "getUsePreconditioner"),
    ("set_use_preconditioner", "setUsePreconditioner"),
    ("get_use_charge_constraint", "getUseChargeConstraint"),
    ("set_use_charge_constraint", "setUseChargeConstraint"),
    ("get_charge_constraint_target", "getChargeConstraintTarget"),
    ("set_charge_constraint_target", "setChargeConstraintTarget"),
    (
        "get_exceptions_use_periodic_boundary_conditions",
        "getExceptionsUsePeriodicBoundaryConditions",
    ),
    (
        "set_exceptions_use_periodic_boundary_conditions",
        "setExceptionsUsePeriodicBoundaryConditions",
    ),
    ("get_pme_parameters", "getPMEParameters"),
    ("set_pme_parameters", "setPMEParameters"),
    ("get_pme_parameters_in_context", "getPMEParametersInContext"),
    ("get_external_field", "getExternalField"),
    ("set_external_field", "setExternalField"),
    ("get_force_group", "getForceGroup"),
    ("set_force_group", "setForceGroup"),
    ("get_name", "getName"),
    ("set_name", "setName"),
    (
        "uses_periodic_boundary_conditions",
        "usesPeriodicBoundaryConditions",
    ),
    ("get_charges", "getCharges"),
    ("update_parameters_in_context", "updateParametersInContext"),
];
