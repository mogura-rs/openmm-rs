use openmm_core::CMMotionRemover;
use pyo3::prelude::*;

#[pyclass(unsendable, name = "CMMotionRemover")]
pub(crate) struct PyCMMotionRemover {
    pub(crate) inner: CMMotionRemover,
}

#[pymethods]
impl PyCMMotionRemover {
    #[new]
    #[pyo3(signature = (frequency=1))]
    fn new(frequency: u64) -> PyResult<Self> {
        if frequency == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "frequency must be positive",
            ));
        }
        Ok(Self {
            inner: CMMotionRemover::new(frequency),
        })
    }
    fn get_frequency(&self) -> u64 {
        self.inner.frequency()
    }
    fn set_frequency(&mut self, frequency: u64) -> PyResult<()> {
        if frequency == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "frequency must be positive",
            ));
        }
        self.inner.set_frequency(frequency);
        Ok(())
    }
    fn get_name(&self) -> &str {
        self.inner.name()
    }
    fn set_name(&mut self, name: &str) {
        self.inner.set_name(name);
    }
    fn get_force_group(&self) -> u32 {
        self.inner.force_group()
    }
    fn set_force_group(&mut self, group: u32) {
        self.inner.set_force_group(group);
    }
    fn uses_periodic_boundary_conditions(&self) -> bool {
        false
    }
}
