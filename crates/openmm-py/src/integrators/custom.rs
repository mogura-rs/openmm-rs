use super::*;
use crate::tabulated::{self, SharedTable};
use openmm_core::{ComputationType, CustomIntegrator};

#[pyclass(unsendable, subclass, name = "CustomIntegrator")]
pub(crate) struct PyCustomIntegrator {
    pub(super) inner: Arc<Mutex<CustomIntegrator>>,
    pub(super) link: ContextLink,
    tables: Rc<RefCell<Vec<SharedTable>>>,
}

#[pymethods]
impl PyCustomIntegrator {
    #[new]
    fn new(step_size: f64) -> PyResult<Self> {
        nonnegative(step_size, "step size")?;
        let inner = Arc::new(Mutex::new(CustomIntegrator::new(step_size)));
        let tables: Rc<RefCell<Vec<SharedTable>>> = Rc::new(RefCell::new(Vec::new()));
        let link = ContextLink::default();
        let weak_inner = Arc::downgrade(&inner);
        let shared_tables = tables.clone();
        link.0.borrow_mut().local_preparation = Some(Rc::new(move || {
            if let Some(inner) = weak_inner.upgrade() {
                let mut core = inner.lock().unwrap();
                if !core.is_prepared() {
                    for (index, table) in shared_tables.borrow().iter().enumerate() {
                        core.set_tabulated_function(index, table.borrow().clone())
                            .map_err(|e| err(&e.to_string()))?;
                    }
                }
            }
            Ok(())
        }));
        Ok(Self {
            inner,
            link,
            tables,
        })
    }
    fn step(&self, steps: usize) -> PyResult<()> {
        self.link.step(steps)
    }
    #[getter]
    fn step_size(&self) -> f64 {
        self.read_step_size()
    }
    #[pyo3(name = "getStepSize")]
    fn read_step_size(&self) -> f64 {
        self.inner.lock().unwrap().step_size()
    }
    fn set_step_size(&self, value: f64) -> PyResult<()> {
        nonnegative(value, "step size")?;
        self.inner.lock().unwrap().set_step_size(value);
        Ok(())
    }
    fn get_constraint_tolerance(&self) -> f64 {
        self.inner.lock().unwrap().constraint_tolerance()
    }
    fn set_constraint_tolerance(&self, value: f64) -> PyResult<()> {
        positive(value, "constraint tolerance")?;
        if value >= 1.0 {
            return Err(err("constraint tolerance must be less than one"));
        }
        self.inner.lock().unwrap().set_constraint_tolerance(value);
        Ok(())
    }
    fn get_integration_force_groups(&self) -> i32 {
        self.inner.lock().unwrap().integration_force_groups() as i32
    }
    fn set_integration_force_groups(&self, value: i64) -> PyResult<()> {
        self.inner
            .lock()
            .unwrap()
            .set_integration_force_groups(group_mask(value)?);
        Ok(())
    }
    fn get_random_number_seed(&self) -> i32 {
        self.inner.lock().unwrap().random_number_seed()
    }
    fn set_random_number_seed(&self, value: i32) {
        self.inner.lock().unwrap().set_random_number_seed(value);
    }
    fn get_kinetic_energy_expression(&self) -> String {
        self.inner.lock().unwrap().kinetic_energy_expression()
    }
    fn set_kinetic_energy_expression(&self, expression: &str) -> PyResult<()> {
        self.inner
            .lock()
            .unwrap()
            .set_kinetic_energy_expression(expression)
            .map_err(|e| err(&e.to_string()))
    }
    fn add_tabulated_function(&self, name: &str, function: &Bound<'_, PyAny>) -> PyResult<usize> {
        let table = tabulated::extract(function)?;
        let index = self
            .inner
            .lock()
            .unwrap()
            .add_tabulated_function(name, table.borrow().clone());
        self.tables.borrow_mut().push(table);
        Ok(index)
    }
    fn get_tabulated_function_name(&self, index: usize) -> PyResult<String> {
        self.inner
            .lock()
            .unwrap()
            .tabulated_function_name(index)
            .map_err(|e| err(&e.to_string()))
    }
    fn get_tabulated_function(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
        let table = self
            .tables
            .borrow()
            .get(index)
            .cloned()
            .ok_or_else(|| err("tabulated function index out of range"))?;
        tabulated::wrap(py, table)
    }
    fn get_computation_step(&self, index: usize) -> PyResult<(u8, String, String)> {
        let step = self
            .inner
            .lock()
            .unwrap()
            .computation_step(index)
            .map_err(|e| err(&e.to_string()))?;
        Ok((step.kind as u8, step.variable, step.expression))
    }
    fn add_global_variable(&self, name: &str, value: f64) -> PyResult<usize> {
        self.inner
            .lock()
            .unwrap()
            .add_global_variable(name, value)
            .map_err(|e| err(&e.to_string()))
    }
    fn get_global_variable_name(&self, index: usize) -> PyResult<String> {
        self.inner
            .lock()
            .unwrap()
            .global_variable_name(index)
            .map_err(|e| err(&e.to_string()))
    }
    fn get_global_variable(&self, index: usize) -> PyResult<f64> {
        self.inner
            .lock()
            .unwrap()
            .global_variable(index)
            .map_err(|e| err(&e.to_string()))
    }
    fn set_global_variable(&self, index: usize, value: f64) -> PyResult<()> {
        self.inner
            .lock()
            .unwrap()
            .set_global_variable(index, value)
            .map_err(|e| err(&e.to_string()))
    }
    fn get_global_variable_by_name(&self, name: &str) -> PyResult<f64> {
        let core = self.inner.lock().unwrap();
        let index = core
            .global_variable_index(name)
            .map_err(|e| err(&e.to_string()))?;
        core.global_variable(index).map_err(|e| err(&e.to_string()))
    }
    fn set_global_variable_by_name(&self, name: &str, value: f64) -> PyResult<()> {
        let mut core = self.inner.lock().unwrap();
        let index = core
            .global_variable_index(name)
            .map_err(|e| err(&e.to_string()))?;
        core.set_global_variable(index, value)
            .map_err(|e| err(&e.to_string()))
    }
    fn add_per_dof_variable(&self, name: &str, value: f64) -> PyResult<usize> {
        self.inner
            .lock()
            .unwrap()
            .add_per_dof_variable(name, value)
            .map_err(|e| err(&e.to_string()))
    }
    fn get_per_dof_variable_name(&self, index: usize) -> PyResult<String> {
        self.inner
            .lock()
            .unwrap()
            .per_dof_variable_name(index)
            .map_err(|e| err(&e.to_string()))
    }
    fn get_per_dof_variable(&self, index: usize) -> PyResult<Vec<[f64; 3]>> {
        self.inner
            .lock()
            .unwrap()
            .per_dof_variable(index)
            .map_err(|e| err(&e.to_string()))
    }
    fn set_per_dof_variable(&self, index: usize, value: Vec<[f64; 3]>) -> PyResult<()> {
        self.inner
            .lock()
            .unwrap()
            .set_per_dof_variable(index, value)
            .map_err(|e| err(&e.to_string()))
    }
    fn get_per_dof_variable_by_name(&self, name: &str) -> PyResult<Vec<[f64; 3]>> {
        let core = self.inner.lock().unwrap();
        let index = core
            .per_dof_variable_index(name)
            .map_err(|e| err(&e.to_string()))?;
        core.per_dof_variable(index)
            .map_err(|e| err(&e.to_string()))
    }
    fn set_per_dof_variable_by_name(&self, name: &str, value: Vec<[f64; 3]>) -> PyResult<()> {
        let mut core = self.inner.lock().unwrap();
        let index = core
            .per_dof_variable_index(name)
            .map_err(|e| err(&e.to_string()))?;
        core.set_per_dof_variable(index, value)
            .map_err(|e| err(&e.to_string()))
    }
    fn get_num_global_variables(&self) -> usize {
        self.inner.lock().unwrap().num_global_variables()
    }
    fn get_num_per_dof_variables(&self) -> usize {
        self.inner.lock().unwrap().num_per_dof_variables()
    }
    fn get_num_computations(&self) -> usize {
        self.inner.lock().unwrap().num_computations()
    }
    fn get_num_tabulated_functions(&self) -> usize {
        self.inner.lock().unwrap().num_tabulated_functions()
    }
    #[classattr]
    #[pyo3(name = "ComputeGlobal")]
    const KIND_0: u8 = 0;
    fn add_compute_global(&self, variable: &str, expression: &str) -> PyResult<usize> {
        self.inner
            .lock()
            .unwrap()
            .add_computation(ComputationType::ComputeGlobal, variable, expression)
            .map_err(|e| err(&e.to_string()))
    }
    #[classattr]
    #[pyo3(name = "ComputePerDof")]
    const KIND_1: u8 = 1;
    fn add_compute_per_dof(&self, variable: &str, expression: &str) -> PyResult<usize> {
        self.inner
            .lock()
            .unwrap()
            .add_computation(ComputationType::ComputePerDof, variable, expression)
            .map_err(|e| err(&e.to_string()))
    }
    #[classattr]
    #[pyo3(name = "ComputeSum")]
    const KIND_2: u8 = 2;
    fn add_compute_sum(&self, variable: &str, expression: &str) -> PyResult<usize> {
        self.inner
            .lock()
            .unwrap()
            .add_computation(ComputationType::ComputeSum, variable, expression)
            .map_err(|e| err(&e.to_string()))
    }
    #[classattr]
    #[pyo3(name = "ConstrainPositions")]
    const KIND_3: u8 = 3;
    fn add_constrain_positions(&self) -> PyResult<usize> {
        self.inner
            .lock()
            .unwrap()
            .add_computation(ComputationType::ConstrainPositions, "", "")
            .map_err(|e| err(&e.to_string()))
    }
    #[classattr]
    #[pyo3(name = "ConstrainVelocities")]
    const KIND_4: u8 = 4;
    fn add_constrain_velocities(&self) -> PyResult<usize> {
        self.inner
            .lock()
            .unwrap()
            .add_computation(ComputationType::ConstrainVelocities, "", "")
            .map_err(|e| err(&e.to_string()))
    }
    #[classattr]
    #[pyo3(name = "UpdateContextState")]
    const KIND_5: u8 = 5;
    fn add_update_context_state(&self) -> PyResult<usize> {
        self.inner
            .lock()
            .unwrap()
            .add_computation(ComputationType::UpdateContextState, "", "")
            .map_err(|e| err(&e.to_string()))
    }
    #[classattr]
    #[pyo3(name = "IfBlockStart")]
    const KIND_6: u8 = 6;
    fn begin_if_block(&self, expression: &str) -> PyResult<usize> {
        self.inner
            .lock()
            .unwrap()
            .add_computation(ComputationType::IfBlockStart, "", expression)
            .map_err(|e| err(&e.to_string()))
    }
    #[classattr]
    #[pyo3(name = "WhileBlockStart")]
    const KIND_7: u8 = 7;
    fn begin_while_block(&self, expression: &str) -> PyResult<usize> {
        self.inner
            .lock()
            .unwrap()
            .add_computation(ComputationType::WhileBlockStart, "", expression)
            .map_err(|e| err(&e.to_string()))
    }
    #[classattr]
    #[pyo3(name = "BlockEnd")]
    const KIND_8: u8 = 8;
    fn end_block(&self) -> PyResult<usize> {
        self.inner
            .lock()
            .unwrap()
            .add_computation(ComputationType::BlockEnd, "", "")
            .map_err(|e| err(&e.to_string()))
    }
}
