use super::{err, PyContext};
use openmm_core::{DrudeForce, DrudeParticle};
use pyo3::prelude::*;

type ParticleParameters = (usize, usize, isize, isize, isize, f64, f64, f64, f64);
fn optional_index(value: isize) -> PyResult<Option<usize>> {
    if value == -1 {
        Ok(None)
    } else {
        usize::try_from(value)
            .map(Some)
            .map_err(|_| err("orientation index must be -1 or nonnegative"))
    }
}
#[allow(clippy::too_many_arguments)]
fn particle(
    particle: usize,
    particle1: usize,
    particle2: isize,
    particle3: isize,
    particle4: isize,
    charge: f64,
    polarizability: f64,
    aniso12: f64,
    aniso34: f64,
) -> PyResult<DrudeParticle> {
    Ok(DrudeParticle {
        particle,
        parent: particle1,
        axis2: optional_index(particle2)?,
        axis3: optional_index(particle3)?,
        axis4: optional_index(particle4)?,
        charge,
        polarizability,
        aniso12,
        aniso34,
    })
}
#[pyclass(unsendable, name = "DrudeForce")]
pub(crate) struct PyDrudeForce {
    pub(crate) inner: DrudeForce,
}
#[pymethods]
impl PyDrudeForce {
    #[new]
    fn new() -> Self {
        Self {
            inner: DrudeForce::new(),
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn add_particle(
        &mut self,
        particle: usize,
        particle1: usize,
        particle2: isize,
        particle3: isize,
        particle4: isize,
        charge: f64,
        polarizability: f64,
        aniso12: f64,
        aniso34: f64,
    ) -> PyResult<usize> {
        self.inner
            .add_particle(self::particle(
                particle,
                particle1,
                particle2,
                particle3,
                particle4,
                charge,
                polarizability,
                aniso12,
                aniso34,
            )?)
            .map_err(|e| err(&e))
    }
    fn get_num_particles(&self) -> usize {
        self.inner.num_particles()
    }
    fn get_particle_parameters(&self, index: usize) -> PyResult<ParticleParameters> {
        if index >= self.inner.num_particles() {
            return Err(err("Drude particle index out of range"));
        }
        let p = self.inner.particle_parameters(index);
        Ok((
            p.particle,
            p.parent,
            p.axis2.map_or(-1, |i| i as isize),
            p.axis3.map_or(-1, |i| i as isize),
            p.axis4.map_or(-1, |i| i as isize),
            p.charge,
            p.polarizability,
            p.aniso12,
            p.aniso34,
        ))
    }
    #[allow(clippy::too_many_arguments)]
    fn set_particle_parameters(
        &mut self,
        index: usize,
        particle: usize,
        particle1: usize,
        particle2: isize,
        particle3: isize,
        particle4: isize,
        charge: f64,
        polarizability: f64,
        aniso12: f64,
        aniso34: f64,
    ) -> PyResult<()> {
        self.inner
            .set_particle_parameters(
                index,
                self::particle(
                    particle,
                    particle1,
                    particle2,
                    particle3,
                    particle4,
                    charge,
                    polarizability,
                    aniso12,
                    aniso34,
                )?,
            )
            .map_err(|e| err(&e))
    }
    fn add_screened_pair(&mut self, particle1: usize, particle2: usize, thole: f64) -> usize {
        self.inner.add_screened_pair(particle1, particle2, thole)
    }
    fn get_num_screened_pairs(&self) -> usize {
        self.inner.num_screened_pairs()
    }
    fn get_screened_pair_parameters(&self, index: usize) -> PyResult<(usize, usize, f64)> {
        if index >= self.inner.num_screened_pairs() {
            return Err(err("screened pair index out of range"));
        }
        let p = self.inner.screened_pair_parameters(index);
        Ok((p.particle1, p.particle2, p.thole))
    }
    fn set_screened_pair_parameters(
        &mut self,
        index: usize,
        particle1: usize,
        particle2: usize,
        thole: f64,
    ) -> PyResult<()> {
        if index >= self.inner.num_screened_pairs() {
            return Err(err("screened pair index out of range"));
        }
        self.inner
            .set_screened_pair_parameters(index, particle1, particle2, thole);
        Ok(())
    }
    fn set_uses_periodic_boundary_conditions(&mut self, value: bool) {
        self.inner.set_uses_periodic_boundary_conditions(value);
    }
    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.inner.uses_periodic_boundary_conditions()
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
            .update_force_parameters_at_path(&[index], &slf.inner)
            .map_err(|e| err(&e))
    }
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDrudeForce>()?;
    let class = module.py().get_type::<PyDrudeForce>();
    for (snake, camel) in [
        ("add_particle", "addParticle"),
        ("get_num_particles", "getNumParticles"),
        ("get_particle_parameters", "getParticleParameters"),
        ("set_particle_parameters", "setParticleParameters"),
        ("add_screened_pair", "addScreenedPair"),
        ("get_num_screened_pairs", "getNumScreenedPairs"),
        ("get_screened_pair_parameters", "getScreenedPairParameters"),
        ("set_screened_pair_parameters", "setScreenedPairParameters"),
        (
            "set_uses_periodic_boundary_conditions",
            "setUsesPeriodicBoundaryConditions",
        ),
        (
            "uses_periodic_boundary_conditions",
            "usesPeriodicBoundaryConditions",
        ),
        ("get_force_group", "getForceGroup"),
        ("set_force_group", "setForceGroup"),
        ("get_name", "getName"),
        ("set_name", "setName"),
        ("update_parameters_in_context", "updateParametersInContext"),
    ] {
        class.setattr(camel, class.getattr(snake)?)?;
    }
    Ok(())
}
