use super::{err, extract_force_snapshot, PyContext, Vec3, B};
use openmm_core::{ATMForce, ATMTransformation};
use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};

#[pyclass(name = "FixedDisplacement", frozen)]
pub(crate) struct PyFixedDisplacement {
    displacement1: Vec3,
    displacement0: Vec3,
}
#[pymethods]
impl PyFixedDisplacement {
    #[new]
    #[pyo3(signature=(displacement1, displacement0=(0.,0.,0.)))]
    fn new(displacement1: Vec3, displacement0: Vec3) -> Self {
        Self {
            displacement1,
            displacement0,
        }
    }
    fn get_fixed_displacement1(&self) -> Vec3 {
        self.displacement1
    }
    fn get_fixed_displacement0(&self) -> Vec3 {
        self.displacement0
    }
}
#[pyclass(name = "ParticleOffsetDisplacement", frozen)]
pub(crate) struct PyParticleOffsetDisplacement {
    destination1: usize,
    origin1: usize,
    destination0: i32,
    origin0: i32,
}
#[pymethods]
impl PyParticleOffsetDisplacement {
    #[new]
    #[pyo3(signature=(destination1, origin1, destination0=-1, origin0=-1))]
    fn new(destination1: usize, origin1: usize, destination0: i32, origin0: i32) -> PyResult<Self> {
        if (destination0 < 0) != (origin0 < 0) {
            return Err(err(
                "both state-zero displacement indices must be provided together",
            ));
        }
        Ok(Self {
            destination1,
            origin1,
            destination0,
            origin0,
        })
    }
    fn get_destination_particle1(&self) -> usize {
        self.destination1
    }
    fn get_origin_particle1(&self) -> usize {
        self.origin1
    }
    fn get_destination_particle0(&self) -> i32 {
        self.destination0
    }
    fn get_origin_particle0(&self) -> i32 {
        self.origin0
    }
}
fn transformation(object: &Bound<'_, PyAny>) -> PyResult<ATMTransformation> {
    if let Ok(value) = object.extract::<PyRef<'_, PyFixedDisplacement>>() {
        return Ok(ATMTransformation::fixed(
            value.displacement1.into(),
            value.displacement0.into(),
        ));
    }
    if let Ok(value) = object.extract::<PyRef<'_, PyParticleOffsetDisplacement>>() {
        let state0 = (value.destination0 >= 0 && value.origin0 >= 0)
            .then_some([value.destination0 as usize, value.origin0 as usize]);
        return Ok(ATMTransformation::particle_offset(
            value.destination1,
            value.origin1,
            state0,
        ));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected an ATM coordinate transformation",
    ))
}
fn check(index: usize, count: usize) -> PyResult<()> {
    if index < count {
        Ok(())
    } else {
        Err(pyo3::exceptions::PyIndexError::new_err(
            "ATM index out of range",
        ))
    }
}
#[pyclass(unsendable, name = "ATMForce")]
pub(crate) struct PyATMForce {
    inner: ATMForce<B>,
    forces: Vec<Py<PyAny>>,
    force_owner: super::ownership::ForceOwner,
}
impl PyATMForce {
    pub(crate) fn snapshot(&self, py: Python<'_>, live_group: bool) -> PyResult<ATMForce<B>> {
        let mut force = if live_group {
            self.inner.snapshot_for_context()
        } else {
            self.inner.clone()
        };
        for child in &self.forces {
            force.add_force(extract_force_snapshot(child.bind(py))?);
        }
        Ok(force)
    }
}
#[pymethods]
impl PyATMForce {
    #[new]
    #[pyo3(signature=(*args, **kwargs))]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut arguments: Vec<Bound<'_, PyAny>> = args.iter().collect();
        let names = [
            "lambda1",
            "lambda2",
            "alpha",
            "uh",
            "w0",
            "umax",
            "ubcore",
            "acore",
            "direction",
        ];
        if let Some(kwargs) = kwargs.filter(|values| !values.is_empty()) {
            if let Some(energy) = kwargs.get_item("energy")? {
                if !arguments.is_empty() || kwargs.len() != 1 {
                    return Err(err(
                        "energy cannot be combined with other ATM constructor arguments",
                    ));
                }
                arguments.push(energy);
            } else {
                let count = arguments.len();
                for (key, _) in kwargs.iter() {
                    let key: String = key.extract()?;
                    let index = names
                        .iter()
                        .position(|name| *name == key)
                        .ok_or_else(|| err("unknown ATM constructor keyword"))?;
                    if index < count {
                        return Err(err("duplicate ATM constructor argument"));
                    }
                }
                for name in names.iter().skip(count) {
                    arguments.push(
                        kwargs
                            .get_item(name)?
                            .ok_or_else(|| err("all nine ATM parameters are required"))?,
                    );
                }
            }
        }
        let inner = if arguments.len() == 1 {
            ATMForce::new(&arguments[0].extract::<String>()?).map_err(|e| err(&e.to_string()))?
        } else if arguments.len() == 9 {
            let mut values = [0.; 9];
            for (value, argument) in values.iter_mut().zip(arguments) {
                *value = argument.extract()?;
            }
            ATMForce::with_default_parameters(values).map_err(|e| err(&e))?
        } else {
            return Err(err(
                "ATMForce requires an energy expression or nine scalar parameters",
            ));
        };
        Ok(Self {
            inner,
            forces: Vec::new(),
            force_owner: super::ownership::ForceOwner::default(),
        })
    }
    fn get_force_group(&self) -> u32 {
        self.inner.force_group()
    }
    fn set_force_group(&mut self, group: u32) {
        self.inner.set_force_group(group);
    }
    fn get_name(&self) -> &str {
        self.inner.name()
    }
    fn set_name(&mut self, name: &str) {
        self.inner.set_name(name);
    }
    fn get_energy_function(&self) -> &str {
        self.inner.energy_function()
    }
    fn set_energy_function(&mut self, energy: &str) -> PyResult<()> {
        self.inner.set_energy_function(energy).map_err(|e| err(&e))
    }
    #[pyo3(signature=(displacement1=None, displacement0=(0.,0.,0.)))]
    fn add_particle(
        &mut self,
        displacement1: Option<&Bound<'_, PyAny>>,
        displacement0: Vec3,
    ) -> PyResult<usize> {
        let value = if let Some(displacement1) = displacement1 {
            if let Ok(vector) = displacement1.extract::<Vec3>() {
                ATMTransformation::fixed(vector.into(), displacement0.into())
            } else {
                transformation(displacement1)?
            }
        } else {
            ATMTransformation::default()
        };
        Ok(self.inner.add_particle(value))
    }
    fn get_num_particles(&self) -> usize {
        self.inner.num_particles()
    }
    fn get_particle_parameters(&self, index: usize) -> PyResult<(Vec3, Vec3)> {
        check(index, self.inner.num_particles())?;
        match self.inner.particle_transformation(index) {
            ATMTransformation::FixedDisplacement {
                displacement1,
                displacement0,
            } => Ok(((*displacement1).into(), (*displacement0).into())),
            _ => Err(err("the particle does not use FixedDisplacement")),
        }
    }
    #[pyo3(signature=(index, displacement1, displacement0=(0.,0.,0.)))]
    fn set_particle_parameters(
        &mut self,
        index: usize,
        displacement1: Vec3,
        displacement0: Vec3,
    ) -> PyResult<()> {
        check(index, self.inner.num_particles())?;
        self.inner.set_particle_transformation(
            index,
            ATMTransformation::fixed(displacement1.into(), displacement0.into()),
        );
        Ok(())
    }
    fn get_particle_transformation(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
        check(index, self.inner.num_particles())?;
        match self.inner.particle_transformation(index) {
            ATMTransformation::FixedDisplacement {
                displacement1,
                displacement0,
            } => Ok(Py::new(
                py,
                PyFixedDisplacement::new((*displacement1).into(), (*displacement0).into()),
            )?
            .into_any()),
            ATMTransformation::ParticleOffsetDisplacement {
                destination1,
                origin1,
                state0,
            } => {
                let [destination0, origin0] = state0.map_or([-1, -1], |p| p.map(|i| i as i32));
                Ok(Py::new(
                    py,
                    PyParticleOffsetDisplacement::new(
                        *destination1,
                        *origin1,
                        destination0,
                        origin0,
                    )?,
                )?
                .into_any())
            }
        }
    }
    fn set_particle_transformation(
        &mut self,
        index: usize,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        check(index, self.inner.num_particles())?;
        self.inner
            .set_particle_transformation(index, transformation(value)?);
        Ok(())
    }
    fn add_force(&mut self, force: &Bound<'_, PyAny>) -> PyResult<usize> {
        extract_force_snapshot(force)?;
        self.force_owner.claim(force)?;
        let index = self.forces.len();
        self.forces.push(force.clone().unbind());
        Ok(index)
    }
    fn get_num_forces(&self) -> usize {
        self.forces.len()
    }
    fn get_force(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
        check(index, self.forces.len())?;
        Ok(self.forces[index].clone_ref(py))
    }
    fn uses_periodic_boundary_conditions(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(self
            .snapshot(py, false)?
            .uses_periodic_boundary_conditions())
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
    fn add_energy_parameter_derivative(&mut self, name: &str) {
        self.inner.add_energy_parameter_derivative(name);
    }
    fn get_num_energy_parameter_derivatives(&self) -> usize {
        self.inner.num_energy_parameter_derivatives()
    }
    fn get_energy_parameter_derivative_name(&self, index: usize) -> PyResult<&str> {
        check(index, self.inner.num_energy_parameter_derivatives())?;
        Ok(self.inner.energy_parameter_derivative_name(index))
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
            .update_force_parameters_at_path(&[index], &slf.snapshot(slf.py(), false)?)
            .map_err(|e| err(&e))
    }
    fn get_perturbation_energy(
        slf: PyRef<'_, Self>,
        context: PyRef<'_, PyContext>,
    ) -> PyResult<(f64, f64, f64)> {
        let index = context
            .force_objects
            .iter()
            .position(|force| force.as_ptr() == slf.as_ptr())
            .ok_or_else(|| err("this Force is not present in the Context"))?;
        Ok(context.inner.borrow().atm_perturbation_energy(index))
    }
    #[staticmethod]
    #[pyo3(name = "Lambda1")]
    fn lambda1() -> &'static str {
        "Lambda1"
    }
    #[staticmethod]
    #[pyo3(name = "Lambda2")]
    fn lambda2() -> &'static str {
        "Lambda2"
    }
    #[staticmethod]
    #[pyo3(name = "Alpha")]
    fn alpha() -> &'static str {
        "Alpha"
    }
    #[staticmethod]
    #[pyo3(name = "Uh")]
    fn uh() -> &'static str {
        "Uh"
    }
    #[staticmethod]
    #[pyo3(name = "W0")]
    fn w0() -> &'static str {
        "W0"
    }
    #[staticmethod]
    #[pyo3(name = "Umax")]
    fn umax() -> &'static str {
        "Umax"
    }
    #[staticmethod]
    #[pyo3(name = "Ubcore")]
    fn ubcore() -> &'static str {
        "Ubcore"
    }
    #[staticmethod]
    #[pyo3(name = "Acore")]
    fn acore() -> &'static str {
        "Acore"
    }
    #[staticmethod]
    #[pyo3(name = "Direction")]
    fn direction() -> &'static str {
        "Direction"
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyATMForce>()?;
    let class = module.py().get_type::<PyATMForce>();
    class.setattr("getForceGroup", class.getattr("get_force_group")?)?;
    class.setattr("setForceGroup", class.getattr("set_force_group")?)?;
    class.setattr("getName", class.getattr("get_name")?)?;
    class.setattr("setName", class.getattr("set_name")?)?;
    class.setattr("getEnergyFunction", class.getattr("get_energy_function")?)?;
    class.setattr("setEnergyFunction", class.getattr("set_energy_function")?)?;
    class.setattr("addParticle", class.getattr("add_particle")?)?;
    class.setattr("getNumParticles", class.getattr("get_num_particles")?)?;
    class.setattr(
        "getParticleParameters",
        class.getattr("get_particle_parameters")?,
    )?;
    class.setattr(
        "setParticleParameters",
        class.getattr("set_particle_parameters")?,
    )?;
    class.setattr(
        "getParticleTransformation",
        class.getattr("get_particle_transformation")?,
    )?;
    class.setattr(
        "setParticleTransformation",
        class.getattr("set_particle_transformation")?,
    )?;
    class.setattr("addForce", class.getattr("add_force")?)?;
    class.setattr("getNumForces", class.getattr("get_num_forces")?)?;
    class.setattr("getForce", class.getattr("get_force")?)?;
    class.setattr(
        "usesPeriodicBoundaryConditions",
        class.getattr("uses_periodic_boundary_conditions")?,
    )?;
    class.setattr("addGlobalParameter", class.getattr("add_global_parameter")?)?;
    class.setattr(
        "getNumGlobalParameters",
        class.getattr("get_num_global_parameters")?,
    )?;
    class.setattr(
        "getGlobalParameterName",
        class.getattr("get_global_parameter_name")?,
    )?;
    class.setattr(
        "setGlobalParameterName",
        class.getattr("set_global_parameter_name")?,
    )?;
    class.setattr(
        "getGlobalParameterDefaultValue",
        class.getattr("get_global_parameter_default_value")?,
    )?;
    class.setattr(
        "setGlobalParameterDefaultValue",
        class.getattr("set_global_parameter_default_value")?,
    )?;
    class.setattr(
        "addEnergyParameterDerivative",
        class.getattr("add_energy_parameter_derivative")?,
    )?;
    class.setattr(
        "getNumEnergyParameterDerivatives",
        class.getattr("get_num_energy_parameter_derivatives")?,
    )?;
    class.setattr(
        "getEnergyParameterDerivativeName",
        class.getattr("get_energy_parameter_derivative_name")?,
    )?;
    class.setattr(
        "updateParametersInContext",
        class.getattr("update_parameters_in_context")?,
    )?;
    class.setattr(
        "getPerturbationEnergy",
        class.getattr("get_perturbation_energy")?,
    )?;
    module.add_class::<PyFixedDisplacement>()?;
    let class = module.py().get_type::<PyFixedDisplacement>();
    class.setattr(
        "getFixedDisplacement1",
        class.getattr("get_fixed_displacement1")?,
    )?;
    class.setattr(
        "getFixedDisplacement0",
        class.getattr("get_fixed_displacement0")?,
    )?;
    module.add_class::<PyParticleOffsetDisplacement>()?;
    let class = module.py().get_type::<PyParticleOffsetDisplacement>();
    class.setattr(
        "getDestinationParticle1",
        class.getattr("get_destination_particle1")?,
    )?;
    class.setattr("getOriginParticle1", class.getattr("get_origin_particle1")?)?;
    class.setattr(
        "getDestinationParticle0",
        class.getattr("get_destination_particle0")?,
    )?;
    class.setattr("getOriginParticle0", class.getattr("get_origin_particle0")?)?;
    Ok(())
}
