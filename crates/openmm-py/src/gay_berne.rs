use super::{err, PyContext};
use openmm_core::{GayBerneForce, GayBerneMethod, GayBerneParticle};
use pyo3::prelude::*;

#[pyclass(unsendable, name = "GayBerneForce")]
pub(crate) struct PyGayBerneForce {
    pub(crate) inner: GayBerneForce,
}

#[pymethods]
impl PyGayBerneForce {
    #[new]
    fn new() -> Self {
        Self {
            inner: GayBerneForce::new(),
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
    #[allow(clippy::too_many_arguments)]
    fn add_particle(
        &mut self,
        sigma: f64,
        epsilon: f64,
        xparticle: i64,
        yparticle: i64,
        sx: f64,
        sy: f64,
        sz: f64,
        ex: f64,
        ey: f64,
        ez: f64,
    ) -> PyResult<usize> {
        self.inner
            .add_particle(particle(
                sigma, epsilon, xparticle, yparticle, sx, sy, sz, ex, ey, ez,
            )?)
            .map_err(|e| err(&e))
    }
    fn get_num_particles(&self) -> usize {
        self.inner.num_particles()
    }
    fn get_particle_parameters(&self, index: usize) -> PyResult<ParticleParameters> {
        if index >= self.inner.num_particles() {
            return Err(err("Gay-Berne particle index out of range"));
        }
        let p = self.inner.particle_parameters(index);
        Ok((
            p.sigma,
            p.epsilon,
            p.xparticle.map_or(-1, |i| i as i64),
            p.yparticle.map_or(-1, |i| i as i64),
            p.widths[0],
            p.widths[1],
            p.widths[2],
            p.scales[0],
            p.scales[1],
            p.scales[2],
        ))
    }
    #[allow(clippy::too_many_arguments)]
    fn set_particle_parameters(
        &mut self,
        index: usize,
        sigma: f64,
        epsilon: f64,
        xparticle: i64,
        yparticle: i64,
        sx: f64,
        sy: f64,
        sz: f64,
        ex: f64,
        ey: f64,
        ez: f64,
    ) -> PyResult<()> {
        if index >= self.inner.num_particles() {
            return Err(err("Gay-Berne particle index out of range"));
        }
        self.inner
            .set_particle_parameters(
                index,
                particle(sigma, epsilon, xparticle, yparticle, sx, sy, sz, ex, ey, ez)?,
            )
            .map_err(|e| err(&e))
    }
    fn get_num_exceptions(&self) -> usize {
        self.inner.num_exceptions()
    }
    #[pyo3(signature=(particle1,particle2,sigma,epsilon,replace=false))]
    fn add_exception(
        &mut self,
        particle1: usize,
        particle2: usize,
        sigma: f64,
        epsilon: f64,
        replace: bool,
    ) -> PyResult<usize> {
        self.inner
            .add_exception(particle1, particle2, sigma, epsilon, replace)
            .map_err(|e| err(&e))
    }
    fn get_exception_parameters(&self, index: usize) -> PyResult<(usize, usize, f64, f64)> {
        if index >= self.inner.num_exceptions() {
            return Err(err("Gay-Berne exception index out of range"));
        }
        let e = self.inner.exception_parameters(index);
        Ok((e.particles[0], e.particles[1], e.sigma, e.epsilon))
    }
    fn set_exception_parameters(
        &mut self,
        index: usize,
        particle1: usize,
        particle2: usize,
        sigma: f64,
        epsilon: f64,
    ) -> PyResult<()> {
        if index >= self.inner.num_exceptions() {
            return Err(err("Gay-Berne exception index out of range"));
        }
        self.inner
            .set_exception_parameters(index, particle1, particle2, sigma, epsilon);
        Ok(())
    }
    fn get_nonbonded_method(&self) -> u32 {
        self.inner.nonbonded_method() as u32
    }
    fn set_nonbonded_method(&mut self, method: u32) -> PyResult<()> {
        self.inner.set_nonbonded_method(match method {
            0 => GayBerneMethod::NoCutoff,
            1 => GayBerneMethod::CutoffNonPeriodic,
            2 => GayBerneMethod::CutoffPeriodic,
            _ => return Err(err("invalid Gay-Berne nonbonded method")),
        });
        Ok(())
    }
    fn get_cutoff_distance(&self) -> f64 {
        self.inner.cutoff_distance()
    }
    fn set_cutoff_distance(&mut self, value: f64) {
        self.inner.set_cutoff_distance(value);
    }
    fn get_use_switching_function(&self) -> bool {
        self.inner.use_switching_function()
    }
    fn set_use_switching_function(&mut self, value: bool) {
        self.inner.set_use_switching_function(value);
    }
    fn get_switching_distance(&self) -> f64 {
        self.inner.switching_distance()
    }
    fn set_switching_distance(&mut self, value: f64) {
        self.inner.set_switching_distance(value);
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
    module.add_class::<PyGayBerneForce>()?;
    let class = module.py().get_type::<PyGayBerneForce>();
    for (snake, camel) in [
        ("add_particle", "addParticle"),
        ("get_num_particles", "getNumParticles"),
        ("get_particle_parameters", "getParticleParameters"),
        ("set_particle_parameters", "setParticleParameters"),
        ("get_nonbonded_method", "getNonbondedMethod"),
        ("set_nonbonded_method", "setNonbondedMethod"),
        ("get_cutoff_distance", "getCutoffDistance"),
        ("set_cutoff_distance", "setCutoffDistance"),
        ("get_use_switching_function", "getUseSwitchingFunction"),
        ("set_use_switching_function", "setUseSwitchingFunction"),
        ("get_switching_distance", "getSwitchingDistance"),
        ("set_switching_distance", "setSwitchingDistance"),
        ("get_num_exceptions", "getNumExceptions"),
        ("get_exception_parameters", "getExceptionParameters"),
        ("set_exception_parameters", "setExceptionParameters"),
        ("add_exception", "addException"),
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

type ParticleParameters = (f64, f64, i64, i64, f64, f64, f64, f64, f64, f64);
#[allow(clippy::too_many_arguments)]
fn particle(
    sigma: f64,
    epsilon: f64,
    xparticle: i64,
    yparticle: i64,
    sx: f64,
    sy: f64,
    sz: f64,
    ex: f64,
    ey: f64,
    ez: f64,
) -> PyResult<GayBerneParticle> {
    let axis = |value: i64| {
        if value == -1 {
            Ok(None)
        } else {
            usize::try_from(value)
                .map(Some)
                .map_err(|_| err("Gay-Berne orientation index must be -1 or nonnegative"))
        }
    };
    Ok(GayBerneParticle {
        sigma,
        epsilon,
        xparticle: axis(xparticle)?,
        yparticle: axis(yparticle)?,
        widths: [sx, sy, sz],
        scales: [ex, ey, ez],
    })
}
