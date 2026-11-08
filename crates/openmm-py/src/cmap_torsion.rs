use openmm_core::CMAPTorsionForce;
use pyo3::prelude::*;

type TorsionParameters = (
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
);

#[pyclass(unsendable, name = "CMAPTorsionForce")]
pub struct PyCMAPTorsionForce {
    pub inner: CMAPTorsionForce,
}

#[pymethods]
impl PyCMAPTorsionForce {
    #[new]
    fn new() -> Self {
        Self {
            inner: CMAPTorsionForce::new(),
        }
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
    fn add_map(&mut self, size: usize, energy: Vec<f64>) -> PyResult<usize> {
        self.inner
            .add_map(size, energy)
            .map_err(|error| super::err(&error.to_string()))
    }
    fn get_num_maps(&self) -> usize {
        self.inner.num_maps()
    }
    fn get_map_parameters(&self, index: usize) -> (usize, Vec<f64>) {
        let (size, values) = self.inner.map_parameters(index);
        (size, values.to_vec())
    }
    fn set_map_parameters(&mut self, index: usize, size: usize, energy: Vec<f64>) -> PyResult<()> {
        self.inner
            .set_map_parameters(index, size, energy)
            .map_err(|error| super::err(&error.to_string()))
    }
    #[allow(clippy::too_many_arguments)]
    fn add_torsion(
        &mut self,
        map: usize,
        a1: usize,
        a2: usize,
        a3: usize,
        a4: usize,
        b1: usize,
        b2: usize,
        b3: usize,
        b4: usize,
    ) -> usize {
        self.inner
            .add_torsion(map, [a1, a2, a3, a4, b1, b2, b3, b4])
    }
    fn get_num_torsions(&self) -> usize {
        self.inner.num_torsions()
    }
    fn get_torsion_parameters(&self, index: usize) -> TorsionParameters {
        let t = self.inner.torsion_parameters(index);
        let p = t.particles;
        (t.map, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])
    }
    #[allow(clippy::too_many_arguments)]
    fn set_torsion_parameters(
        &mut self,
        index: usize,
        map: usize,
        a1: usize,
        a2: usize,
        a3: usize,
        a4: usize,
        b1: usize,
        b2: usize,
        b3: usize,
        b4: usize,
    ) {
        self.inner
            .set_torsion_parameters(index, map, [a1, a2, a3, a4, b1, b2, b3, b4]);
    }
    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.inner.uses_periodic_boundary_conditions()
    }
    fn set_uses_periodic_boundary_conditions(&mut self, periodic: bool) {
        self.inner.set_uses_periodic_boundary_conditions(periodic);
    }
}
