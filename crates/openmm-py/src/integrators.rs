use super::{err, group_mask, PyContext, B};
use burn::prelude::*;
use openmm_core::constraints::{ConstraintError, ConstraintSolver};
use openmm_core::{
    BrownianIntegrator, Context, Integrator, LangevinMiddleIntegrator, VariableLangevinIntegrator,
    VariableVerletIntegrator, VerletIntegrator,
};
use pyo3::prelude::*;
use std::{
    cell::RefCell,
    rc::{Rc, Weak},
    sync::{Arc, Mutex},
};

pub(crate) type SharedContext = Rc<RefCell<Context<B>>>;
#[derive(Clone, Default)]
pub(crate) struct ContextLink(Rc<RefCell<Weak<RefCell<Context<B>>>>>);

impl ContextLink {
    fn check_available(&self) -> PyResult<()> {
        if self.0.borrow().upgrade().is_some() {
            Err(err("this Integrator is already bound to a Context"))
        } else {
            Ok(())
        }
    }

    pub(crate) fn bind(&self, context: &SharedContext) {
        *self.0.borrow_mut() = Rc::downgrade(context);
    }

    fn step_to(&self, time: f64) -> PyResult<()> {
        let context = self
            .0
            .borrow()
            .upgrade()
            .ok_or_else(|| err("this Integrator is not bound to a Context"))?;
        let mut core = context
            .try_borrow_mut()
            .map_err(|_| err("the Context is already in use"))?;
        core.try_step_to(time).map_err(|e| err(&e.to_string()))
    }

    fn step(&self, steps: usize) -> PyResult<()> {
        let context = self
            .0
            .borrow()
            .upgrade()
            .ok_or_else(|| err("this Integrator is not bound to a Context"))?;
        let mut core = context
            .try_borrow_mut()
            .map_err(|_| err("the Context is already in use"))?;
        core.try_step(steps).map_err(|e| err(&e.to_string()))
    }
}

struct SharedIntegrator<T>(Arc<Mutex<T>>);

impl<T: Integrator<B>> Integrator<B> for SharedIntegrator<T> {
    fn initialize(&mut self) {
        self.0.lock().unwrap().initialize();
    }

    fn integration_force_groups(&self) -> u32 {
        self.0.lock().unwrap().integration_force_groups()
    }
    fn set_integration_force_groups(&mut self, groups: u32) {
        self.0.lock().unwrap().set_integration_force_groups(groups);
    }
    fn step_size(&self) -> f64 {
        self.0.lock().unwrap().step_size()
    }
    fn supports_step_to(&self) -> bool {
        self.0.lock().unwrap().supports_step_to()
    }
    fn step_with_limit(
        &self,
        p: &mut Tensor<B, 2>,
        v: &mut Tensor<B, 2>,
        f: &Tensor<B, 2>,
        m: &Tensor<B, 1>,
        c: &ConstraintSolver,
        limit: f64,
    ) -> Result<(), ConstraintError> {
        self.0.lock().unwrap().step_with_limit(p, v, f, m, c, limit)
    }
    fn constraint_tolerance(&self) -> f64 {
        self.0.lock().unwrap().constraint_tolerance()
    }
    fn kinetic_energy_time_offset(&self) -> f64 {
        self.0.lock().unwrap().kinetic_energy_time_offset()
    }
    fn velocity_time_offset(&self) -> f64 {
        self.0.lock().unwrap().velocity_time_offset()
    }
    fn step(
        &self,
        positions: &mut Tensor<B, 2>,
        velocities: &mut Tensor<B, 2>,
        forces: &Tensor<B, 2>,
        masses: &Tensor<B, 1>,
    ) {
        self.0
            .lock()
            .unwrap()
            .step(positions, velocities, forces, masses);
    }
    fn step_constrained(
        &self,
        positions: &mut Tensor<B, 2>,
        velocities: &mut Tensor<B, 2>,
        forces: &Tensor<B, 2>,
        masses: &Tensor<B, 1>,
        constraints: &ConstraintSolver,
    ) -> Result<(), ConstraintError> {
        self.0
            .lock()
            .unwrap()
            .step_constrained(positions, velocities, forces, masses, constraints)
    }
}

fn positive(value: f64, name: &str) -> PyResult<()> {
    if !value.is_finite() || value <= 0.0 {
        Err(err(&format!("{name} must be positive and finite")))
    } else {
        Ok(())
    }
}

fn nonnegative(value: f64, name: &str) -> PyResult<()> {
    if !value.is_finite() || value < 0.0 {
        Err(err(&format!("{name} must be nonnegative and finite")))
    } else {
        Ok(())
    }
}

macro_rules! integrator_class {
    ($wrapper:ident, $name:literal, $core:ty, {$($methods:tt)*}) => {
        #[pyclass(unsendable, subclass, name = $name)]
        pub(crate) struct $wrapper {
            inner: Arc<Mutex<$core>>,
            link: ContextLink,
        }
        impl $wrapper {
            fn from_core(inner: $core) -> Self {
                Self { inner: Arc::new(Mutex::new(inner)), link: ContextLink::default() }
            }
        }
        #[pymethods]
        impl $wrapper {
            fn step(&self, steps: usize) -> PyResult<()> { self.link.step(steps) }
            fn get_integration_force_groups(&self) -> i32 {
                self.inner.lock().unwrap().integration_force_groups() as i32
            }
            fn set_integration_force_groups(&self, groups: i64) -> PyResult<()> {
                let groups = group_mask(groups)?;
                self.inner.lock().unwrap().set_integration_force_groups(groups);
                Ok(())
            }
            fn get_constraint_tolerance(&self) -> f64 {
                <$core as Integrator<B>>::constraint_tolerance(&*self.inner.lock().unwrap())
            }
            fn set_constraint_tolerance(&self, tolerance: f64) -> PyResult<()> {
                positive(tolerance, "constraint tolerance")?;
                if tolerance >= 1.0 { return Err(err("constraint tolerance must be less than one")); }
                self.inner.lock().unwrap().set_constraint_tolerance(tolerance);
                Ok(())
            }
            #[getter]
            fn step_size(&self) -> f64 { self.read_step_size() }
            #[pyo3(name = "getStepSize")]
            fn read_step_size(&self) -> f64 {
                <$core as Integrator<B>>::step_size(&*self.inner.lock().unwrap())
            }
            fn set_step_size(&self, step_size: f64) -> PyResult<()> {
                if <$core as Integrator<B>>::supports_step_to(&*self.inner.lock().unwrap()) {
                    nonnegative(step_size, "step size")?;
                } else { positive(step_size, "step size")?; }
                self.inner.lock().unwrap().set_step_size(step_size);
                Ok(())
            }
            $($methods)*
        }
    };
}

integrator_class!(PyVerletIntegrator, "VerletIntegrator", VerletIntegrator, {
    #[new]
    fn new(step_size: f64) -> PyResult<Self> {
        positive(step_size, "step size")?;
        Ok(Self::from_core(VerletIntegrator::new(step_size)))
    }
});

integrator_class!(
    PyLangevinMiddleIntegrator,
    "LangevinMiddleIntegrator",
    LangevinMiddleIntegrator,
    {
        #[new]
        fn new(temperature: f64, friction: f64, step_size: f64) -> PyResult<Self> {
            nonnegative(temperature, "temperature")?;
            nonnegative(friction, "friction")?;
            positive(step_size, "step size")?;
            Ok(Self::from_core(LangevinMiddleIntegrator::new(
                temperature,
                friction,
                step_size,
            )))
        }
        fn get_random_number_seed(&self) -> i32 {
            self.inner.lock().unwrap().random_number_seed()
        }
        fn set_random_number_seed(&self, seed: i32) {
            self.inner.lock().unwrap().set_random_number_seed(seed);
        }
        #[getter]
        fn temperature(&self) -> f64 {
            self.read_temperature()
        }
        #[pyo3(name = "getTemperature")]
        fn read_temperature(&self) -> f64 {
            self.inner.lock().unwrap().temperature()
        }
        fn set_temperature(&self, temperature: f64) -> PyResult<()> {
            nonnegative(temperature, "temperature")?;
            self.inner.lock().unwrap().set_temperature(temperature);
            Ok(())
        }
        #[getter]
        fn friction(&self) -> f64 {
            self.read_friction()
        }
        #[pyo3(name = "getFriction")]
        fn read_friction(&self) -> f64 {
            self.inner.lock().unwrap().friction()
        }
        fn set_friction(&self, friction: f64) -> PyResult<()> {
            nonnegative(friction, "friction")?;
            self.inner.lock().unwrap().set_friction(friction);
            Ok(())
        }
    }
);

#[pyclass(unsendable, extends=PyLangevinMiddleIntegrator, name="LangevinIntegrator")]
pub(crate) struct PyLangevinIntegrator;

#[pymethods]
impl PyLangevinIntegrator {
    #[new]
    fn new(
        temperature: f64,
        friction: f64,
        step_size: f64,
    ) -> PyResult<(Self, PyLangevinMiddleIntegrator)> {
        Ok((
            Self,
            PyLangevinMiddleIntegrator::new(temperature, friction, step_size)?,
        ))
    }
}

integrator_class!(
    PyBrownianIntegrator,
    "BrownianIntegrator",
    BrownianIntegrator,
    {
        #[new]
        fn new(temperature: f64, friction: f64, step_size: f64) -> PyResult<Self> {
            nonnegative(temperature, "temperature")?;
            positive(friction, "friction")?;
            positive(step_size, "step size")?;
            Ok(Self::from_core(BrownianIntegrator::new(
                temperature,
                friction,
                step_size,
            )))
        }
        fn get_random_number_seed(&self) -> i32 {
            self.inner.lock().unwrap().random_number_seed()
        }
        fn set_random_number_seed(&self, seed: i32) {
            self.inner.lock().unwrap().set_random_number_seed(seed);
        }
        #[getter]
        fn temperature(&self) -> f64 {
            self.read_temperature()
        }
        #[pyo3(name = "getTemperature")]
        fn read_temperature(&self) -> f64 {
            self.inner.lock().unwrap().temperature()
        }
        fn set_temperature(&self, temperature: f64) -> PyResult<()> {
            nonnegative(temperature, "temperature")?;
            self.inner.lock().unwrap().set_temperature(temperature);
            Ok(())
        }
        #[getter]
        fn friction(&self) -> f64 {
            self.read_friction()
        }
        #[pyo3(name = "getFriction")]
        fn read_friction(&self) -> f64 {
            self.inner.lock().unwrap().friction()
        }
        fn set_friction(&self, friction: f64) -> PyResult<()> {
            positive(friction, "friction")?;
            self.inner.lock().unwrap().set_friction(friction);
            Ok(())
        }
    }
);

integrator_class!(
    PyVariableVerletIntegrator,
    "VariableVerletIntegrator",
    VariableVerletIntegrator,
    {
        #[new]
        fn new(error_tolerance: f64) -> PyResult<Self> {
            positive(error_tolerance, "error tolerance")?;
            Ok(Self::from_core(VariableVerletIntegrator::new(
                error_tolerance,
            )))
        }
        fn step_to(&self, time: f64) -> PyResult<()> {
            self.link.step_to(time)
        }
        fn get_error_tolerance(&self) -> f64 {
            self.inner.lock().unwrap().error_tolerance()
        }
        fn set_error_tolerance(&self, value: f64) -> PyResult<()> {
            positive(value, "error tolerance")?;
            self.inner.lock().unwrap().set_error_tolerance(value);
            Ok(())
        }
        fn get_maximum_step_size(&self) -> f64 {
            self.inner.lock().unwrap().maximum_step_size()
        }
        fn set_maximum_step_size(&self, value: f64) -> PyResult<()> {
            nonnegative(value, "maximum step size")?;
            self.inner.lock().unwrap().set_maximum_step_size(value);
            Ok(())
        }
    }
);

integrator_class!(
    PyVariableLangevinIntegrator,
    "VariableLangevinIntegrator",
    VariableLangevinIntegrator,
    {
        #[new]
        fn new(temperature: f64, friction: f64, error_tolerance: f64) -> PyResult<Self> {
            nonnegative(temperature, "temperature")?;
            nonnegative(friction, "friction")?;
            positive(error_tolerance, "error tolerance")?;
            Ok(Self::from_core(VariableLangevinIntegrator::new(
                temperature,
                friction,
                error_tolerance,
            )))
        }
        fn step_to(&self, time: f64) -> PyResult<()> {
            self.link.step_to(time)
        }
        fn get_error_tolerance(&self) -> f64 {
            self.inner.lock().unwrap().error_tolerance()
        }
        fn set_error_tolerance(&self, value: f64) -> PyResult<()> {
            positive(value, "error tolerance")?;
            self.inner.lock().unwrap().set_error_tolerance(value);
            Ok(())
        }
        fn get_maximum_step_size(&self) -> f64 {
            self.inner.lock().unwrap().maximum_step_size()
        }
        fn set_maximum_step_size(&self, value: f64) -> PyResult<()> {
            nonnegative(value, "maximum step size")?;
            self.inner.lock().unwrap().set_maximum_step_size(value);
            Ok(())
        }
        fn get_temperature(&self) -> f64 {
            self.inner.lock().unwrap().temperature()
        }
        fn set_temperature(&self, value: f64) -> PyResult<()> {
            nonnegative(value, "temperature")?;
            self.inner.lock().unwrap().set_temperature(value);
            Ok(())
        }
        fn get_friction(&self) -> f64 {
            self.inner.lock().unwrap().friction()
        }
        fn set_friction(&self, value: f64) -> PyResult<()> {
            nonnegative(value, "friction")?;
            self.inner.lock().unwrap().set_friction(value);
            Ok(())
        }
        fn get_random_number_seed(&self) -> i32 {
            self.inner.lock().unwrap().random_number_seed()
        }
        fn set_random_number_seed(&self, value: i32) {
            self.inner.lock().unwrap().set_random_number_seed(value);
        }
    }
);

pub(crate) fn prepare(
    object: &Bound<'_, PyAny>,
) -> PyResult<(Box<dyn Integrator<B>>, ContextLink)> {
    macro_rules! extract {
        ($class:ty) => {
            if let Ok(integrator) = object.extract::<PyRef<'_, $class>>() {
                integrator.link.check_available()?;
                return Ok((
                    Box::new(SharedIntegrator(integrator.inner.clone())),
                    integrator.link.clone(),
                ));
            }
        };
    }
    extract!(PyVariableVerletIntegrator);
    extract!(PyVariableLangevinIntegrator);
    extract!(PyVerletIntegrator);
    extract!(PyLangevinMiddleIntegrator);
    extract!(PyBrownianIntegrator);
    Err(err("unsupported integrator type"))
}

pub(crate) fn create_context(
    system: Py<super::PySystem>,
    integrator: &Bound<'_, PyAny>,
) -> PyResult<PyContext> {
    let py = integrator.py();
    let (core, link) = prepare(integrator)?;
    let snapshot = system.borrow(py).snapshot(py)?;
    if !snapshot.uses_periodic_boundary_conditions()
        && snapshot
            .forces()
            .iter()
            .any(|force| force.requires_periodic_system())
    {
        return Err(err("a barostat cannot be used with a non-periodic system"));
    }
    let force_objects = system
        .borrow(py)
        .force_objects
        .iter()
        .map(|force| force.clone_ref(py))
        .collect();
    let inner = Rc::new(RefCell::new(Context::new(
        snapshot,
        core,
        Default::default(),
    )));
    link.bind(&inner);
    Ok(PyContext {
        inner,
        system,
        integrator: integrator.clone().unbind(),
        force_objects,
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyVariableVerletIntegrator>()?;
    module.add_class::<PyVariableLangevinIntegrator>()?;
    module.add_class::<PyVerletIntegrator>()?;
    module.add_class::<PyLangevinMiddleIntegrator>()?;
    module.add_class::<PyLangevinIntegrator>()?;
    module.add_class::<PyBrownianIntegrator>()?;
    for class in [
        module.py().get_type::<PyVariableVerletIntegrator>(),
        module.py().get_type::<PyVariableLangevinIntegrator>(),
        module.py().get_type::<PyVerletIntegrator>(),
        module.py().get_type::<PyLangevinMiddleIntegrator>(),
        module.py().get_type::<PyBrownianIntegrator>(),
    ] {
        for (snake, camel) in [
            ("step_to", "stepTo"),
            ("get_error_tolerance", "getErrorTolerance"),
            ("set_error_tolerance", "setErrorTolerance"),
            ("get_maximum_step_size", "getMaximumStepSize"),
            ("set_maximum_step_size", "setMaximumStepSize"),
            ("get_step_size", "getStepSize"),
            ("get_random_number_seed", "getRandomNumberSeed"),
            ("set_random_number_seed", "setRandomNumberSeed"),
            ("get_temperature", "getTemperature"),
            ("get_friction", "getFriction"),
            ("set_step_size", "setStepSize"),
            ("set_temperature", "setTemperature"),
            ("set_friction", "setFriction"),
            ("get_integration_force_groups", "getIntegrationForceGroups"),
            ("set_integration_force_groups", "setIntegrationForceGroups"),
            ("get_constraint_tolerance", "getConstraintTolerance"),
            ("set_constraint_tolerance", "setConstraintTolerance"),
        ] {
            if class.hasattr(snake)? {
                class.setattr(camel, class.getattr(snake)?)?;
            } else if class.hasattr(camel)? {
                class.setattr(snake, class.getattr(camel)?)?;
            }
        }
    }
    Ok(())
}
