use openmm_core::{CustomVolumeForce, OrientationRestraintForce, RGForce, RMSDForce};
use pyo3::prelude::*;

#[pyclass(unsendable, name = "RMSDForce")]
pub(crate) struct PyRMSDForce {
    pub(crate) inner: RMSDForce,
}
#[pymethods]
impl PyRMSDForce {
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
    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.inner.uses_periodic_boundary_conditions()
    }
    #[new]
    #[pyo3(signature = (reference_positions, particles=None))]
    fn new(reference_positions: Vec<[f64; 3]>, particles: Option<Vec<usize>>) -> Self {
        Self {
            inner: RMSDForce::new(reference_positions, particles.unwrap_or_default()),
        }
    }
    fn get_reference_positions(&self) -> Vec<[f64; 3]> {
        self.inner.reference_positions().to_vec()
    }
    fn set_reference_positions(&mut self, positions: Vec<[f64; 3]>) {
        self.inner.set_reference_positions(positions);
    }
    fn get_particles(&self) -> Vec<usize> {
        self.inner.particles().to_vec()
    }
    fn set_particles(&mut self, particles: Vec<usize>) {
        self.inner.set_particles(particles);
    }
}

#[pyclass(unsendable, name = "RGForce")]
pub(crate) struct PyRGForce {
    pub(crate) inner: RGForce,
}
#[pymethods]
impl PyRGForce {
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
    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.inner.uses_periodic_boundary_conditions()
    }
    #[new]
    #[pyo3(signature = (particles=None))]
    fn new(particles: Option<Vec<usize>>) -> Self {
        Self {
            inner: RGForce::new(particles.unwrap_or_default()),
        }
    }
    fn get_particles(&self) -> Vec<usize> {
        self.inner.particles().to_vec()
    }
    fn set_particles(&mut self, particles: Vec<usize>) {
        self.inner.set_particles(particles);
    }
}

#[pyclass(unsendable, name = "OrientationRestraintForce")]
pub(crate) struct PyOrientationRestraintForce {
    pub(crate) inner: OrientationRestraintForce,
}
#[pymethods]
impl PyOrientationRestraintForce {
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
    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.inner.uses_periodic_boundary_conditions()
    }
    #[new]
    #[pyo3(signature = (k, reference_positions, particles=None))]
    fn new(k: f64, reference_positions: Vec<[f64; 3]>, particles: Option<Vec<usize>>) -> Self {
        Self {
            inner: OrientationRestraintForce::new(
                k,
                reference_positions,
                particles.unwrap_or_default(),
            ),
        }
    }
    fn get_reference_positions(&self) -> Vec<[f64; 3]> {
        self.inner.reference_positions().to_vec()
    }
    fn set_reference_positions(&mut self, positions: Vec<[f64; 3]>) {
        self.inner.set_reference_positions(positions);
    }
    fn get_particles(&self) -> Vec<usize> {
        self.inner.particles().to_vec()
    }
    fn set_particles(&mut self, particles: Vec<usize>) {
        self.inner.set_particles(particles);
    }
    fn get_k(&self) -> f64 {
        self.inner.k()
    }
    fn set_k(&mut self, k: f64) {
        self.inner.set_k(k);
    }
}

#[pyclass(unsendable, name = "CustomVolumeForce")]
pub(crate) struct PyCustomVolumeForce {
    pub(crate) inner: CustomVolumeForce,
}
#[pymethods]
impl PyCustomVolumeForce {
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
    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.inner.uses_periodic_boundary_conditions()
    }
    #[new]
    fn new(energy: &str) -> PyResult<Self> {
        Ok(Self {
            inner: CustomVolumeForce::new(energy).map_err(|e| super::err(&e.to_string()))?,
        })
    }
    fn get_energy_function(&self) -> &str {
        self.inner.energy_function()
    }
    fn set_energy_function(&mut self, energy: &str) -> PyResult<()> {
        self.inner
            .set_energy_function(energy)
            .map_err(|e| super::err(&e.to_string()))
    }
    fn add_global_parameter(&mut self, name: &str, value: f64) -> usize {
        self.inner.add_global_parameter(name, value)
    }
    fn get_num_global_parameters(&self) -> usize {
        self.inner.num_global_parameters()
    }
    fn get_global_parameter_name(&self, index: usize) -> &str {
        self.inner.global_parameter_name(index)
    }
    fn get_global_parameter_default_value(&self, index: usize) -> f64 {
        self.inner.global_parameter_default_value(index)
    }
    fn set_global_parameter_default_value(&mut self, index: usize, value: f64) {
        self.inner.set_global_parameter_default_value(index, value);
    }
}
