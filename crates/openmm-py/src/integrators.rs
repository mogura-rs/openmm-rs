use super::{err, group_mask, PyContext, B};
use burn::prelude::*;
use openmm_core::constraints::{ConstraintError, ConstraintSolver};
use openmm_core::integrators::IntegratorComponent;
use openmm_core::{
    BrownianIntegrator, CompoundIntegrator, Context, Integrator, LangevinMiddleIntegrator,
    VariableLangevinIntegrator, VariableVerletIntegrator, VerletIntegrator,
};
use pyo3::prelude::*;
use std::{
    cell::RefCell,
    rc::{Rc, Weak},
    sync::{Arc, Mutex},
};

pub(crate) type SharedContext = Rc<RefCell<Context<B>>>;
#[derive(Default)]
struct LinkState {
    context: Weak<RefCell<Context<B>>>,
    path: Vec<usize>,
    parent: Weak<RefCell<LinkState>>,
    ever_bound: bool,
}

#[derive(Clone, Default)]
pub(crate) struct ContextLink(Rc<RefCell<LinkState>>);

impl ContextLink {
    fn check_available(&self) -> PyResult<()> {
        let state = self.0.borrow();
        if state.context.upgrade().is_some() {
            Err(err("this Integrator is already bound to a Context"))
        } else if state.parent.upgrade().is_some() {
            Err(err(
                "this Integrator is already owned by a CompoundIntegrator",
            ))
        } else {
            Ok(())
        }
    }

    fn bind(&self, context: &SharedContext, path: Vec<usize>) {
        let mut state = self.0.borrow_mut();
        state.context = Rc::downgrade(context);
        state.path = path;
        state.ever_bound = true;
    }

    fn step_to(&self, time: f64) -> PyResult<()> {
        let state = self.0.borrow();
        let context = state
            .context
            .upgrade()
            .ok_or_else(|| err("this Integrator is not bound to a Context"))?;
        let mut core = context
            .try_borrow_mut()
            .map_err(|_| err("the Context is already in use"))?;
        core.try_step_to_component(&state.path, time)
            .map_err(|e| err(&e.to_string()))
    }

    fn step(&self, steps: usize) -> PyResult<()> {
        let state = self.0.borrow();
        let context = state
            .context
            .upgrade()
            .ok_or_else(|| err("this Integrator is not bound to a Context"))?;
        let mut core = context
            .try_borrow_mut()
            .map_err(|_| err("the Context is already in use"))?;
        core.try_step_component(&state.path, steps)
            .map_err(|e| err(&e.to_string()))
    }
}

struct SharedIntegrator<T>(Arc<Mutex<T>>);

impl<T: Integrator<B>> Integrator<B> for SharedIntegrator<T> {
    fn selected_integrator(&self) -> Option<usize> {
        self.0.lock().unwrap().selected_integrator()
    }
    fn select_integrator(&mut self, index: usize) -> Result<(), ConstraintError> {
        self.0.lock().unwrap().select_integrator(index)
    }
    fn component(&self, path: &[usize]) -> Option<IntegratorComponent<B>> {
        self.0.lock().unwrap().component(path)
    }
    fn set_step_size(&mut self, value: f64) {
        self.0.lock().unwrap().set_step_size(value);
    }
    fn set_constraint_tolerance(&mut self, value: f64) {
        self.0.lock().unwrap().set_constraint_tolerance(value);
    }

    fn validate_system(&self, system: &openmm_core::System<B>) -> Result<(), ConstraintError> {
        self.0.lock().unwrap().validate_system(system)
    }
    fn initialize_system(&mut self, system: &openmm_core::System<B>) {
        self.0.lock().unwrap().initialize_system(system);
    }
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

    fn actual_step_size(&self) -> f64 {
        self.0.lock().unwrap().actual_step_size()
    }
    fn supports_step_to(&self) -> bool {
        self.0.lock().unwrap().supports_step_to()
    }
    fn step_in_environment(
        &self,
        p: &mut Tensor<B, 2>,
        v: &mut Tensor<B, 2>,
        f: &Tensor<B, 2>,
        m: &Tensor<B, 1>,
        c: &ConstraintSolver,
        environment: openmm_core::integrators::StepEnvironment,
    ) -> Result<(), ConstraintError> {
        self.0
            .lock()
            .unwrap()
            .step_in_environment(p, v, f, m, c, environment)
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
            pub(super) inner: Arc<Mutex<$core>>,
            pub(super) link: ContextLink,
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

mod dpd;
use dpd::PyDPDIntegrator;
mod qtb;
use qtb::PyQTBIntegrator;
mod nose_hoover;
use nose_hoover::{PyNoseHooverChain, PyNoseHooverIntegrator};

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

#[pyclass(unsendable, name = "CompoundIntegrator")]
pub(crate) struct PyCompoundIntegrator {
    inner: Arc<Mutex<CompoundIntegrator<B>>>,
    link: ContextLink,
    children: Vec<Py<PyAny>>,
}

#[pymethods]
impl PyCompoundIntegrator {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(CompoundIntegrator::new())),
            link: ContextLink::default(),
            children: Vec::new(),
        }
    }

    fn add_integrator(mut slf: PyRefMut<'_, Self>, object: &Bound<'_, PyAny>) -> PyResult<usize> {
        if slf.link.0.borrow().ever_bound {
            return Err(err("cannot add an integrator after binding to a Context"));
        }
        if slf.as_ptr() == object.as_ptr() {
            return Err(err("integrator ownership cannot contain a cycle"));
        }
        let (core, link) = prepare_shared(object)?;
        link.check_available()?;
        let mut ancestor = Some(slf.link.0.clone());
        while let Some(parent) = ancestor {
            if Rc::ptr_eq(&parent, &link.0) {
                return Err(err("integrator ownership cannot contain a cycle"));
            }
            ancestor = parent.borrow().parent.upgrade();
        }
        let index = slf.inner.lock().unwrap().add_integrator(core);
        link.0.borrow_mut().parent = Rc::downgrade(&slf.link.0);
        slf.children.push(object.clone().unbind());
        Ok(index)
    }

    fn get_num_integrators(&self) -> usize {
        self.children.len()
    }
    fn get_integrator(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
        self.children
            .get(index)
            .map(|x| x.clone_ref(py))
            .ok_or_else(|| err("integrator index out of range"))
    }
    fn get_current_integrator(&self) -> usize {
        self.inner.lock().unwrap().current_integrator()
    }
    fn set_current_integrator(&self, index: usize) -> PyResult<()> {
        if index >= self.children.len() {
            return Err(err("integrator index out of range"));
        }
        self.inner.lock().unwrap().set_current_integrator(index);
        Ok(())
    }
    fn step(&self, steps: usize) -> PyResult<()> {
        self.link.step(steps)
    }
    #[pyo3(name = "getStepSize")]
    fn read_step_size(&self, py: Python<'_>) -> PyResult<f64> {
        self.check_nonempty(py)?;
        Ok(self.inner.lock().unwrap().step_size())
    }
    #[getter]
    fn step_size(&self, py: Python<'_>) -> PyResult<f64> {
        self.read_step_size(py)
    }
    fn set_step_size(&self, py: Python<'_>, value: f64) -> PyResult<()> {
        self.check_nonempty(py)?;
        self.children[self.get_current_integrator()]
            .bind(py)
            .call_method1("setStepSize", (value,))?;
        Ok(())
    }
    fn get_constraint_tolerance(&self, py: Python<'_>) -> PyResult<f64> {
        self.check_nonempty(py)?;
        Ok(self.inner.lock().unwrap().constraint_tolerance())
    }
    fn set_constraint_tolerance(&self, py: Python<'_>, value: f64) -> PyResult<()> {
        self.check_nonempty(py)?;
        positive(value, "constraint tolerance")?;
        if value >= 1.0 {
            return Err(err("constraint tolerance must be less than one"));
        }
        self.inner.lock().unwrap().set_constraint_tolerance(value);
        Ok(())
    }
    fn get_integration_force_groups(&self, py: Python<'_>) -> PyResult<i32> {
        self.check_nonempty(py)?;
        Ok(self.inner.lock().unwrap().integration_force_groups() as i32)
    }
    fn set_integration_force_groups(&self, groups: i64) -> PyResult<()> {
        let groups = group_mask(groups)?;
        self.inner
            .lock()
            .unwrap()
            .set_integration_force_groups(groups);
        Ok(())
    }
}

impl PyCompoundIntegrator {
    fn check_nonempty(&self, py: Python<'_>) -> PyResult<()> {
        if self.children.is_empty() {
            return Err(err("CompoundIntegrator requires at least one child"));
        }
        // Nested compounds may still be empty before a Context is created.
        let child = self.children[self.get_current_integrator()].bind(py);
        if let Ok(compound) = child.extract::<PyRef<'_, Self>>() {
            compound.check_nonempty(py)?;
        }
        Ok(())
    }
}

fn collect_bindings(
    object: &Bound<'_, PyAny>,
    path: Vec<usize>,
    bindings: &mut Vec<(ContextLink, Vec<usize>)>,
) -> PyResult<()> {
    let py = object.py();
    let (_, link) = prepare_shared(object)?;
    if link.0.borrow().context.upgrade().is_some() {
        return Err(err("this Integrator is already bound to a Context"));
    }
    bindings.push((link, path.clone()));
    if let Ok(compound) = object.extract::<PyRef<'_, PyCompoundIntegrator>>() {
        compound.check_nonempty(py)?;
        for (index, child) in compound.children.iter().enumerate() {
            let mut child_path = path.clone();
            child_path.push(index);
            collect_bindings(child.bind(object.py()), child_path, bindings)?;
        }
    }
    Ok(())
}

fn prepare_shared(object: &Bound<'_, PyAny>) -> PyResult<(Box<dyn Integrator<B>>, ContextLink)> {
    macro_rules! extract {
        ($class:ty) => {
            if let Ok(integrator) = object.extract::<PyRef<'_, $class>>() {
                return Ok((
                    Box::new(SharedIntegrator(integrator.inner.clone())),
                    integrator.link.clone(),
                ));
            }
        };
    }
    extract!(PyDPDIntegrator);
    extract!(PyQTBIntegrator);
    extract!(PyNoseHooverIntegrator);
    extract!(PyCompoundIntegrator);
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
    let (core, link) = prepare_shared(integrator)?;
    link.check_available()?;
    let mut bindings = Vec::new();
    collect_bindings(integrator, Vec::new(), &mut bindings)?;
    let snapshot = system.borrow(py).snapshot(py)?;
    if !snapshot.uses_periodic_boundary_conditions()
        && snapshot
            .forces()
            .iter()
            .any(|force| force.requires_periodic_system())
    {
        return Err(err("a barostat cannot be used with a non-periodic system"));
    }
    core.validate_system(&snapshot)
        .map_err(|e| err(&e.to_string()))?;
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
    for (link, path) in bindings {
        link.bind(&inner, path);
    }
    Ok(PyContext {
        inner,
        system,
        integrator: integrator.clone().unbind(),
        force_objects,
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDPDIntegrator>()?;
    module.add_class::<PyQTBIntegrator>()?;
    module.add_class::<PyNoseHooverIntegrator>()?;
    module.add_class::<PyNoseHooverChain>()?;
    module.add_class::<PyCompoundIntegrator>()?;
    module.add_class::<PyVariableVerletIntegrator>()?;
    module.add_class::<PyVariableLangevinIntegrator>()?;
    module.add_class::<PyVerletIntegrator>()?;
    module.add_class::<PyLangevinMiddleIntegrator>()?;
    module.add_class::<PyLangevinIntegrator>()?;
    module.add_class::<PyBrownianIntegrator>()?;
    for class in [
        module.py().get_type::<PyDPDIntegrator>(),
        module.py().get_type::<PyQTBIntegrator>(),
        module.py().get_type::<PyNoseHooverIntegrator>(),
        module.py().get_type::<PyNoseHooverChain>(),
        module.py().get_type::<PyCompoundIntegrator>(),
        module.py().get_type::<PyVariableVerletIntegrator>(),
        module.py().get_type::<PyVariableLangevinIntegrator>(),
        module.py().get_type::<PyVerletIntegrator>(),
        module.py().get_type::<PyLangevinMiddleIntegrator>(),
        module.py().get_type::<PyBrownianIntegrator>(),
    ] {
        for (snake, camel) in [
            ("get_segment_length", "getSegmentLength"),
            ("set_segment_length", "setSegmentLength"),
            ("get_cutoff_frequency", "getCutoffFrequency"),
            ("set_cutoff_frequency", "setCutoffFrequency"),
            ("get_default_adaptation_rate", "getDefaultAdaptationRate"),
            ("set_default_adaptation_rate", "setDefaultAdaptationRate"),
            ("get_type_adaptation_rates", "getTypeAdaptationRates"),
            ("set_type_adaptation_rate", "setTypeAdaptationRate"),
            ("get_adapted_friction", "getAdaptedFriction"),
            ("get_default_friction", "getDefaultFriction"),
            ("set_default_friction", "setDefaultFriction"),
            ("get_default_cutoff", "getDefaultCutoff"),
            ("set_default_cutoff", "setDefaultCutoff"),
            ("get_particle_type", "getParticleType"),
            ("set_particle_type", "setParticleType"),
            ("get_particle_types", "getParticleTypes"),
            ("get_num_type_pairs", "getNumTypePairs"),
            ("add_type_pair", "addTypePair"),
            ("get_type_pair_parameters", "getTypePairParameters"),
            ("set_type_pair_parameters", "setTypePairParameters"),
            ("add_thermostat", "addThermostat"),
            ("add_subsystem_thermostat", "addSubsystemThermostat"),
            ("get_num_thermostats", "getNumThermostats"),
            ("get_thermostat", "getThermostat"),
            ("has_subsystem_thermostats", "hasSubsystemThermostats"),
            ("get_maximum_pair_distance", "getMaximumPairDistance"),
            ("set_maximum_pair_distance", "setMaximumPairDistance"),
            ("compute_heat_bath_energy", "computeHeatBathEnergy"),
            ("get_relative_temperature", "getRelativeTemperature"),
            ("set_relative_temperature", "setRelativeTemperature"),
            ("get_collision_frequency", "getCollisionFrequency"),
            ("set_collision_frequency", "setCollisionFrequency"),
            (
                "get_relative_collision_frequency",
                "getRelativeCollisionFrequency",
            ),
            (
                "set_relative_collision_frequency",
                "setRelativeCollisionFrequency",
            ),
            ("get_num_degrees_of_freedom", "getNumDegreesOfFreedom"),
            ("set_num_degrees_of_freedom", "setNumDegreesOfFreedom"),
            ("get_chain_length", "getChainLength"),
            ("get_num_multi_time_steps", "getNumMultiTimeSteps"),
            (
                "get_num_yoshida_suzuki_time_steps",
                "getNumYoshidaSuzukiTimeSteps",
            ),
            ("get_chain_id", "getChainID"),
            ("get_thermostated_atoms", "getThermostatedAtoms"),
            ("set_thermostated_atoms", "setThermostatedAtoms"),
            ("get_thermostated_pairs", "getThermostatedPairs"),
            ("set_thermostated_pairs", "setThermostatedPairs"),
            ("get_yoshida_suzuki_weights", "getYoshidaSuzukiWeights"),
            (
                "uses_periodic_boundary_conditions",
                "usesPeriodicBoundaryConditions",
            ),
            (
                "get_all_thermostated_individual_particles",
                "getAllThermostatedIndividualParticles",
            ),
            ("get_all_thermostated_pairs", "getAllThermostatedPairs"),
            ("add_integrator", "addIntegrator"),
            ("get_num_integrators", "getNumIntegrators"),
            ("get_integrator", "getIntegrator"),
            ("get_current_integrator", "getCurrentIntegrator"),
            ("set_current_integrator", "setCurrentIntegrator"),
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
