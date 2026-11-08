use openmm_core::tabulated::*;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use std::{cell::RefCell, rc::Rc};

pub type SharedTable = Rc<RefCell<TabulatedFunction>>;
fn error(error: TableError) -> PyErr {
    PyValueError::new_err(error.to_string())
}
#[pyclass(unsendable, name = "Continuous1DFunction")]
pub struct PyContinuous1DFunction {
    pub inner: SharedTable,
}
#[pymethods]
impl PyContinuous1DFunction {
    #[new]
    #[pyo3(signature=(values, min, max, periodic=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(values: Vec<f64>, min: f64, max: f64, periodic: bool) -> PyResult<Self> {
        Ok(Self {
            inner: Rc::new(RefCell::new(
                Continuous1DFunction::new([values.len()], values, [[min, max]], periodic)
                    .map_err(error)?
                    .into(),
            )),
        })
    }
    #[allow(clippy::type_complexity)]
    fn get_function_parameters(&self) -> (Vec<f64>, f64, f64) {
        let inner = self.inner.borrow();
        let TabulatedFunction::Continuous1D(table) = &*inner else {
            unreachable!()
        };
        (
            table.values().to_vec(),
            table.bounds()[0][0],
            table.bounds()[0][1],
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn set_function_parameters(&mut self, values: Vec<f64>, min: f64, max: f64) -> PyResult<()> {
        let mut inner = self.inner.borrow_mut();
        let TabulatedFunction::Continuous1D(table) = &mut *inner else {
            unreachable!()
        };
        table
            .set_function_parameters([values.len()], values, [[min, max]])
            .map_err(error)
    }
    fn get_periodic(&self) -> bool {
        let inner = self.inner.borrow();
        let TabulatedFunction::Continuous1D(table) = &*inner else {
            unreachable!()
        };
        table.periodic()
    }
    fn get_update_count(&self) -> u64 {
        let inner = self.inner.borrow();
        let TabulatedFunction::Continuous1D(table) = &*inner else {
            unreachable!()
        };
        table.update_count()
    }
}
#[pyclass(unsendable, name = "Continuous2DFunction")]
pub struct PyContinuous2DFunction {
    pub inner: SharedTable,
}
#[pymethods]
impl PyContinuous2DFunction {
    #[new]
    #[pyo3(signature=(xsize, ysize, values, xmin, xmax, ymin, ymax, periodic=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        xsize: usize,
        ysize: usize,
        values: Vec<f64>,
        xmin: f64,
        xmax: f64,
        ymin: f64,
        ymax: f64,
        periodic: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Rc::new(RefCell::new(
                Continuous2DFunction::new(
                    [xsize, ysize],
                    values,
                    [[xmin, xmax], [ymin, ymax]],
                    periodic,
                )
                .map_err(error)?
                .into(),
            )),
        })
    }
    #[allow(clippy::type_complexity)]
    fn get_function_parameters(&self) -> (usize, usize, Vec<f64>, f64, f64, f64, f64) {
        let inner = self.inner.borrow();
        let TabulatedFunction::Continuous2D(table) = &*inner else {
            unreachable!()
        };
        (
            table.shape()[0],
            table.shape()[1],
            table.values().to_vec(),
            table.bounds()[0][0],
            table.bounds()[0][1],
            table.bounds()[1][0],
            table.bounds()[1][1],
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn set_function_parameters(
        &mut self,
        xsize: usize,
        ysize: usize,
        values: Vec<f64>,
        xmin: f64,
        xmax: f64,
        ymin: f64,
        ymax: f64,
    ) -> PyResult<()> {
        let mut inner = self.inner.borrow_mut();
        let TabulatedFunction::Continuous2D(table) = &mut *inner else {
            unreachable!()
        };
        table
            .set_function_parameters([xsize, ysize], values, [[xmin, xmax], [ymin, ymax]])
            .map_err(error)
    }
    fn get_periodic(&self) -> bool {
        let inner = self.inner.borrow();
        let TabulatedFunction::Continuous2D(table) = &*inner else {
            unreachable!()
        };
        table.periodic()
    }
    fn get_update_count(&self) -> u64 {
        let inner = self.inner.borrow();
        let TabulatedFunction::Continuous2D(table) = &*inner else {
            unreachable!()
        };
        table.update_count()
    }
}
#[pyclass(unsendable, name = "Continuous3DFunction")]
pub struct PyContinuous3DFunction {
    pub inner: SharedTable,
}
#[pymethods]
impl PyContinuous3DFunction {
    #[new]
    #[pyo3(signature=(xsize, ysize, zsize, values, xmin, xmax, ymin, ymax, zmin, zmax, periodic=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        xsize: usize,
        ysize: usize,
        zsize: usize,
        values: Vec<f64>,
        xmin: f64,
        xmax: f64,
        ymin: f64,
        ymax: f64,
        zmin: f64,
        zmax: f64,
        periodic: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Rc::new(RefCell::new(
                Continuous3DFunction::new(
                    [xsize, ysize, zsize],
                    values,
                    [[xmin, xmax], [ymin, ymax], [zmin, zmax]],
                    periodic,
                )
                .map_err(error)?
                .into(),
            )),
        })
    }
    #[allow(clippy::type_complexity)]
    fn get_function_parameters(
        &self,
    ) -> (usize, usize, usize, Vec<f64>, f64, f64, f64, f64, f64, f64) {
        let inner = self.inner.borrow();
        let TabulatedFunction::Continuous3D(table) = &*inner else {
            unreachable!()
        };
        (
            table.shape()[0],
            table.shape()[1],
            table.shape()[2],
            table.values().to_vec(),
            table.bounds()[0][0],
            table.bounds()[0][1],
            table.bounds()[1][0],
            table.bounds()[1][1],
            table.bounds()[2][0],
            table.bounds()[2][1],
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn set_function_parameters(
        &mut self,
        xsize: usize,
        ysize: usize,
        zsize: usize,
        values: Vec<f64>,
        xmin: f64,
        xmax: f64,
        ymin: f64,
        ymax: f64,
        zmin: f64,
        zmax: f64,
    ) -> PyResult<()> {
        let mut inner = self.inner.borrow_mut();
        let TabulatedFunction::Continuous3D(table) = &mut *inner else {
            unreachable!()
        };
        table
            .set_function_parameters(
                [xsize, ysize, zsize],
                values,
                [[xmin, xmax], [ymin, ymax], [zmin, zmax]],
            )
            .map_err(error)
    }
    fn get_periodic(&self) -> bool {
        let inner = self.inner.borrow();
        let TabulatedFunction::Continuous3D(table) = &*inner else {
            unreachable!()
        };
        table.periodic()
    }
    fn get_update_count(&self) -> u64 {
        let inner = self.inner.borrow();
        let TabulatedFunction::Continuous3D(table) = &*inner else {
            unreachable!()
        };
        table.update_count()
    }
}
#[pyclass(unsendable, name = "Discrete1DFunction")]
pub struct PyDiscrete1DFunction {
    pub inner: SharedTable,
}
#[pymethods]
impl PyDiscrete1DFunction {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(values: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: Rc::new(RefCell::new(
                Discrete1DFunction::new([values.len()], values)
                    .map_err(error)?
                    .into(),
            )),
        })
    }
    #[allow(clippy::type_complexity)]
    fn get_function_parameters(&self) -> Vec<f64> {
        let inner = self.inner.borrow();
        let TabulatedFunction::Discrete1D(table) = &*inner else {
            unreachable!()
        };
        table.values().to_vec()
    }
    #[allow(clippy::too_many_arguments)]
    fn set_function_parameters(&mut self, values: Vec<f64>) -> PyResult<()> {
        let mut inner = self.inner.borrow_mut();
        let TabulatedFunction::Discrete1D(table) = &mut *inner else {
            unreachable!()
        };
        table
            .set_function_parameters([values.len()], values)
            .map_err(error)
    }
    fn get_periodic(&self) -> bool {
        false
    }
    fn get_update_count(&self) -> u64 {
        let inner = self.inner.borrow();
        let TabulatedFunction::Discrete1D(table) = &*inner else {
            unreachable!()
        };
        table.update_count()
    }
}
#[pyclass(unsendable, name = "Discrete2DFunction")]
pub struct PyDiscrete2DFunction {
    pub inner: SharedTable,
}
#[pymethods]
impl PyDiscrete2DFunction {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(xsize: usize, ysize: usize, values: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: Rc::new(RefCell::new(
                Discrete2DFunction::new([xsize, ysize], values)
                    .map_err(error)?
                    .into(),
            )),
        })
    }
    #[allow(clippy::type_complexity)]
    fn get_function_parameters(&self) -> (usize, usize, Vec<f64>) {
        let inner = self.inner.borrow();
        let TabulatedFunction::Discrete2D(table) = &*inner else {
            unreachable!()
        };
        (table.shape()[0], table.shape()[1], table.values().to_vec())
    }
    #[allow(clippy::too_many_arguments)]
    fn set_function_parameters(
        &mut self,
        xsize: usize,
        ysize: usize,
        values: Vec<f64>,
    ) -> PyResult<()> {
        let mut inner = self.inner.borrow_mut();
        let TabulatedFunction::Discrete2D(table) = &mut *inner else {
            unreachable!()
        };
        table
            .set_function_parameters([xsize, ysize], values)
            .map_err(error)
    }
    fn get_periodic(&self) -> bool {
        false
    }
    fn get_update_count(&self) -> u64 {
        let inner = self.inner.borrow();
        let TabulatedFunction::Discrete2D(table) = &*inner else {
            unreachable!()
        };
        table.update_count()
    }
}
#[pyclass(unsendable, name = "Discrete3DFunction")]
pub struct PyDiscrete3DFunction {
    pub inner: SharedTable,
}
#[pymethods]
impl PyDiscrete3DFunction {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(xsize: usize, ysize: usize, zsize: usize, values: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: Rc::new(RefCell::new(
                Discrete3DFunction::new([xsize, ysize, zsize], values)
                    .map_err(error)?
                    .into(),
            )),
        })
    }
    #[allow(clippy::type_complexity)]
    fn get_function_parameters(&self) -> (usize, usize, usize, Vec<f64>) {
        let inner = self.inner.borrow();
        let TabulatedFunction::Discrete3D(table) = &*inner else {
            unreachable!()
        };
        (
            table.shape()[0],
            table.shape()[1],
            table.shape()[2],
            table.values().to_vec(),
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn set_function_parameters(
        &mut self,
        xsize: usize,
        ysize: usize,
        zsize: usize,
        values: Vec<f64>,
    ) -> PyResult<()> {
        let mut inner = self.inner.borrow_mut();
        let TabulatedFunction::Discrete3D(table) = &mut *inner else {
            unreachable!()
        };
        table
            .set_function_parameters([xsize, ysize, zsize], values)
            .map_err(error)
    }
    fn get_periodic(&self) -> bool {
        false
    }
    fn get_update_count(&self) -> u64 {
        let inner = self.inner.borrow();
        let TabulatedFunction::Discrete3D(table) = &*inner else {
            unreachable!()
        };
        table.update_count()
    }
}
pub fn extract(value: &Bound<'_, PyAny>) -> PyResult<SharedTable> {
    if let Ok(value) = value.extract::<PyRef<'_, PyContinuous1DFunction>>() {
        return Ok(value.inner.clone());
    }
    if let Ok(value) = value.extract::<PyRef<'_, PyContinuous2DFunction>>() {
        return Ok(value.inner.clone());
    }
    if let Ok(value) = value.extract::<PyRef<'_, PyContinuous3DFunction>>() {
        return Ok(value.inner.clone());
    }
    if let Ok(value) = value.extract::<PyRef<'_, PyDiscrete1DFunction>>() {
        return Ok(value.inner.clone());
    }
    if let Ok(value) = value.extract::<PyRef<'_, PyDiscrete2DFunction>>() {
        return Ok(value.inner.clone());
    }
    if let Ok(value) = value.extract::<PyRef<'_, PyDiscrete3DFunction>>() {
        return Ok(value.inner.clone());
    }
    Err(PyTypeError::new_err("expected a tabulated function"))
}
pub fn wrap(py: Python<'_>, inner: SharedTable) -> PyResult<Py<PyAny>> {
    let snapshot = inner.borrow().clone();
    match snapshot {
        TabulatedFunction::Continuous1D(_) => {
            Ok(Py::new(py, PyContinuous1DFunction { inner })?.into_any())
        }
        TabulatedFunction::Continuous2D(_) => {
            Ok(Py::new(py, PyContinuous2DFunction { inner })?.into_any())
        }
        TabulatedFunction::Continuous3D(_) => {
            Ok(Py::new(py, PyContinuous3DFunction { inner })?.into_any())
        }
        TabulatedFunction::Discrete1D(_) => {
            Ok(Py::new(py, PyDiscrete1DFunction { inner })?.into_any())
        }
        TabulatedFunction::Discrete2D(_) => {
            Ok(Py::new(py, PyDiscrete2DFunction { inner })?.into_any())
        }
        TabulatedFunction::Discrete3D(_) => {
            Ok(Py::new(py, PyDiscrete3DFunction { inner })?.into_any())
        }
    }
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyContinuous1DFunction>()?;
    m.add_class::<PyContinuous2DFunction>()?;
    m.add_class::<PyContinuous3DFunction>()?;
    m.add_class::<PyDiscrete1DFunction>()?;
    m.add_class::<PyDiscrete2DFunction>()?;
    m.add_class::<PyDiscrete3DFunction>()?;
    Ok(())
}
