use openmm_core::VirtualSite;
use pyo3::exceptions::{PyIndexError, PyTypeError};
use pyo3::prelude::*;

type Vec3 = (f64, f64, f64);
fn array(v: Vec3) -> [f64; 3] {
    [v.0, v.1, v.2]
}
fn tuple(v: [f64; 3]) -> Vec3 {
    (v[0], v[1], v[2])
}

macro_rules! average_site {
    ($rust:ident, $name:literal, $variant:ident, $size:literal, ($($particle:ident),+), ($($weight:ident),+)) => {
        #[pyclass(unsendable, name=$name)]
        pub struct $rust { pub inner: VirtualSite }
        #[pymethods]
        impl $rust {
            #[new]
            fn new($($particle: usize,)+ $($weight: f64),+) -> Self {
                Self { inner: VirtualSite::$variant { particles: [$($particle),+], weights: [$($weight),+] } }
            }
            fn get_num_particles(&self) -> usize { $size }
            fn get_particle(&self, index: usize) -> PyResult<usize> { self.inner.particles().get(index).copied().ok_or_else(|| PyIndexError::new_err("virtual site particle index out of range")) }
            fn get_weight(&self, index: usize) -> PyResult<f64> {
                if let VirtualSite::$variant { weights, .. } = &self.inner {
                    weights.get(index).copied().ok_or_else(|| PyIndexError::new_err("virtual site weight index out of range"))
                } else { unreachable!() }
            }
        }
    }
}

average_site!(
    PyTwoParticleAverageSite,
    "TwoParticleAverageSite",
    TwoParticleAverage,
    2,
    (particle1, particle2),
    (weight1, weight2)
);
average_site!(
    PyThreeParticleAverageSite,
    "ThreeParticleAverageSite",
    ThreeParticleAverage,
    3,
    (particle1, particle2, particle3),
    (weight1, weight2, weight3)
);
average_site!(
    PyOutOfPlaneSite,
    "OutOfPlaneSite",
    OutOfPlane,
    3,
    (particle1, particle2, particle3),
    (weight12, weight13, weight_cross)
);

#[pyclass(unsendable, name = "LocalCoordinatesSite")]
pub struct PyLocalCoordinatesSite {
    pub inner: VirtualSite,
}

#[pymethods]
impl PyLocalCoordinatesSite {
    #[new]
    fn new(
        particles: Vec<usize>,
        origin_weights: Vec<f64>,
        x_weights: Vec<f64>,
        y_weights: Vec<f64>,
        local_position: Vec3,
    ) -> Self {
        Self {
            inner: VirtualSite::LocalCoordinates {
                particles,
                origin_weights,
                x_weights,
                y_weights,
                local_position: array(local_position),
            },
        }
    }
    fn get_num_particles(&self) -> usize {
        self.inner.particles().len()
    }
    fn get_particle(&self, index: usize) -> PyResult<usize> {
        self.inner
            .particles()
            .get(index)
            .copied()
            .ok_or_else(|| PyIndexError::new_err("virtual site particle index out of range"))
    }
    fn get_origin_weights(&self) -> Vec<f64> {
        if let VirtualSite::LocalCoordinates { origin_weights, .. } = &self.inner {
            origin_weights.clone()
        } else {
            unreachable!()
        }
    }
    fn get_x_weights(&self) -> Vec<f64> {
        if let VirtualSite::LocalCoordinates { x_weights, .. } = &self.inner {
            x_weights.clone()
        } else {
            unreachable!()
        }
    }
    fn get_y_weights(&self) -> Vec<f64> {
        if let VirtualSite::LocalCoordinates { y_weights, .. } = &self.inner {
            y_weights.clone()
        } else {
            unreachable!()
        }
    }
    fn get_local_position(&self) -> Vec3 {
        if let VirtualSite::LocalCoordinates { local_position, .. } = &self.inner {
            tuple(*local_position)
        } else {
            unreachable!()
        }
    }
}

#[pyclass(unsendable, name = "SymmetrySite")]
pub struct PySymmetrySite {
    pub inner: VirtualSite,
}

#[pymethods]
impl PySymmetrySite {
    #[new]
    #[pyo3(signature=(particle, rx, ry, rz, offset, use_box_vectors=false))]
    fn new(
        particle: usize,
        rx: Vec3,
        ry: Vec3,
        rz: Vec3,
        offset: Vec3,
        use_box_vectors: bool,
    ) -> Self {
        Self {
            inner: VirtualSite::Symmetry {
                particle,
                rotation: [array(rx), array(ry), array(rz)],
                offset: array(offset),
                use_box_vectors,
            },
        }
    }
    fn get_num_particles(&self) -> usize {
        1
    }
    fn get_particle(&self, index: usize) -> PyResult<usize> {
        self.inner
            .particles()
            .get(index)
            .copied()
            .ok_or_else(|| PyIndexError::new_err("virtual site particle index out of range"))
    }
    fn get_rotation_matrix(&self) -> (Vec3, Vec3, Vec3) {
        if let VirtualSite::Symmetry { rotation, .. } = &self.inner {
            (tuple(rotation[0]), tuple(rotation[1]), tuple(rotation[2]))
        } else {
            unreachable!()
        }
    }
    fn get_offset_vector(&self) -> Vec3 {
        if let VirtualSite::Symmetry { offset, .. } = &self.inner {
            tuple(*offset)
        } else {
            unreachable!()
        }
    }
    fn get_use_box_vectors(&self) -> bool {
        if let VirtualSite::Symmetry {
            use_box_vectors, ..
        } = &self.inner
        {
            *use_box_vectors
        } else {
            unreachable!()
        }
    }
}

pub fn from_python(value: &Bound<'_, PyAny>) -> PyResult<VirtualSite> {
    if let Ok(site) = value.extract::<PyRef<'_, PyTwoParticleAverageSite>>() {
        return Ok(site.inner.clone());
    }
    if let Ok(site) = value.extract::<PyRef<'_, PyThreeParticleAverageSite>>() {
        return Ok(site.inner.clone());
    }
    if let Ok(site) = value.extract::<PyRef<'_, PyOutOfPlaneSite>>() {
        return Ok(site.inner.clone());
    }
    if let Ok(site) = value.extract::<PyRef<'_, PyLocalCoordinatesSite>>() {
        return Ok(site.inner.clone());
    }
    if let Ok(site) = value.extract::<PyRef<'_, PySymmetrySite>>() {
        return Ok(site.inner.clone());
    }
    Err(PyTypeError::new_err("unsupported virtual site type"))
}

pub fn to_python(site: &VirtualSite, py: Python<'_>) -> PyResult<Py<PyAny>> {
    let inner = site.clone();
    match site {
        VirtualSite::TwoParticleAverage { .. } => {
            Ok(Py::new(py, PyTwoParticleAverageSite { inner })?.into_any())
        }
        VirtualSite::ThreeParticleAverage { .. } => {
            Ok(Py::new(py, PyThreeParticleAverageSite { inner })?.into_any())
        }
        VirtualSite::OutOfPlane { .. } => Ok(Py::new(py, PyOutOfPlaneSite { inner })?.into_any()),
        VirtualSite::LocalCoordinates { .. } => {
            Ok(Py::new(py, PyLocalCoordinatesSite { inner })?.into_any())
        }
        VirtualSite::Symmetry { .. } => Ok(Py::new(py, PySymmetrySite { inner })?.into_any()),
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTwoParticleAverageSite>()?;
    module.add_class::<PyThreeParticleAverageSite>()?;
    module.add_class::<PyOutOfPlaneSite>()?;
    module.add_class::<PyLocalCoordinatesSite>()?;
    module.add_class::<PySymmetrySite>()?;
    Ok(())
}
