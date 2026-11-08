mod custom_gb;
use custom_gb::PyCustomGBForce;
mod gay_berne;
use gay_berne::PyGayBerneForce;
mod lcpo;
use lcpo::PyLCPOForce;
mod gbsa_obc;
use gbsa_obc::PyGBSAOBCForce;
mod flexible_barostat;
use flexible_barostat::PyMonteCarloFlexibleBarostat;
mod membrane_barostat;
use membrane_barostat::PyMonteCarloMembraneBarostat;
mod anisotropic_barostat;
use anisotropic_barostat::PyMonteCarloAnisotropicBarostat;
mod andersen;
mod barostat;
mod cm_motion;
mod integrators;
mod ownership;
use andersen::PyAndersenThermostat;
use barostat::PyMonteCarloBarostat;
use cm_motion::PyCMMotionRemover;
mod custom_cv;
use custom_cv::PyCustomCVForce;
mod collective;
use collective::{PyCustomVolumeForce, PyOrientationRestraintForce, PyRGForce, PyRMSDForce};
mod custom_hbond;
use custom_hbond::PyCustomHbondForce;
mod custom_centroid;
use custom_centroid::PyCustomCentroidBondForce;
mod custom_compound;
use custom_compound::PyCustomCompoundBondForce;
// Thin pyo3 wrapper. All simulation logic lives in openmm-core.
// Backend is fixed to NdArray (CPU) at the Python boundary.
mod cmap_torsion;
mod tabulated;
use cmap_torsion::PyCMAPTorsionForce;
mod virtual_sites;
use burn::backend::NdArray;
use burn::prelude::{Backend, Tensor};
use openmm_core::forces::custom_angle::CustomAngleForce;
use openmm_core::forces::custom_bond::CustomBondForce;
use openmm_core::forces::custom_external::CustomExternalForce;
use openmm_core::forces::custom_nonbonded::{CustomNonbondedForce, CustomNonbondedMethod};
use openmm_core::forces::custom_torsion::CustomTorsionForce;
use openmm_core::forces::periodic_torsion::TorsionParam;
use openmm_core::forces::rb_torsion::RBTorsionParam;
use openmm_core::{
    HarmonicAngleForce, HarmonicBondForce, NonbondedForce, NonbondedMethod, PeriodicTorsionForce,
    RBTorsionForce, State, StateFlags, System,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::HashMap;

type B = NdArray<f64>;
type Device = <B as Backend>::Device;
type Vec3 = (f64, f64, f64);
type TorsionParticles = (usize, usize, usize, usize);
type PeriodicTorsionParameters = (TorsionParticles, i32, f64, f64);
type RBCoefficients = (f64, f64, f64, f64, f64, f64);
type RBTorsionParameters = (TorsionParticles, RBCoefficients);

fn err(msg: &str) -> PyErr {
    PyRuntimeError::new_err(msg.to_string())
}

fn to_box_vectors(a: Vec3, b: Vec3, c: Vec3) -> [[f64; 3]; 3] {
    [[a.0, a.1, a.2], [b.0, b.1, b.2], [c.0, c.1, c.2]]
}

fn vec3_list_to_box_vectors(vectors: &[Vec3]) -> PyResult<[[f64; 3]; 3]> {
    if vectors.len() != 3 {
        return Err(err("periodic_box_vectors must contain exactly 3 vectors"));
    }
    Ok([
        [vectors[0].0, vectors[0].1, vectors[0].2],
        [vectors[1].0, vectors[1].1, vectors[1].2],
        [vectors[2].0, vectors[2].1, vectors[2].2],
    ])
}

fn to_tensor3(data: &[Vec3], device: &Device) -> Tensor<B, 2> {
    let flat: Vec<f64> = data.iter().flat_map(|&(x, y, z)| [x, y, z]).collect();
    openmm_core::tensor::from_floats::<B, 1>(flat.as_slice(), device).reshape([data.len(), 3])
}

fn tensor_to_vec3(tensor: &Tensor<B, 2>) -> Vec<Vec3> {
    let data = tensor.clone().to_data().to_vec::<f64>().unwrap();
    data.chunks_exact(3)
        .map(|chunk| (chunk[0], chunk[1], chunk[2]))
        .collect()
}

#[pyclass(unsendable, name = "State")]
struct PyState {
    time: f64,
    step_count: u64,
    parameters: Option<HashMap<String, f64>>,
    energy_parameter_derivatives: Option<HashMap<String, f64>>,
    periodic_box_vectors: Option<Vec<Vec3>>,
    positions: Option<Vec<Vec3>>,
    velocities: Option<Vec<Vec3>>,
    forces: Option<Vec<Vec3>>,
    potential_energy: Option<f64>,
    kinetic_energy: Option<f64>,
}

impl PyState {
    fn from_core(state: State<B>) -> Self {
        Self {
            time: state.time,
            step_count: state.step_count,
            parameters: state.parameters,
            energy_parameter_derivatives: state.energy_parameter_derivatives,
            periodic_box_vectors: state.periodic_box_vectors.map(|vectors| {
                vectors
                    .into_iter()
                    .map(|v| (v[0], v[1], v[2]))
                    .collect::<Vec<_>>()
            }),
            positions: state.positions.as_ref().map(tensor_to_vec3),
            velocities: state.velocities.as_ref().map(tensor_to_vec3),
            forces: state.forces.as_ref().map(tensor_to_vec3),
            potential_energy: state.potential_energy,
            kinetic_energy: state.kinetic_energy,
        }
    }

    fn to_core(&self, device: &Device) -> PyResult<State<B>> {
        let periodic_box_vectors = self
            .periodic_box_vectors
            .as_ref()
            .map(|vectors| vec3_list_to_box_vectors(vectors))
            .transpose()?;

        Ok(State {
            time: self.time,
            step_count: self.step_count,
            parameters: self.parameters.clone(),
            energy_parameter_derivatives: self.energy_parameter_derivatives.clone(),
            periodic_box_vectors,
            positions: self
                .positions
                .as_ref()
                .map(|positions| to_tensor3(positions, device)),
            velocities: self
                .velocities
                .as_ref()
                .map(|velocities| to_tensor3(velocities, device)),
            forces: self
                .forces
                .as_ref()
                .map(|forces| to_tensor3(forces, device)),
            potential_energy: self.potential_energy,
            kinetic_energy: self.kinetic_energy,
        })
    }
}

#[pymethods]
impl PyState {
    #[getter]
    fn time(&self) -> f64 {
        self.time
    }

    #[getter]
    fn step_count(&self) -> u64 {
        self.step_count
    }

    #[getter]
    fn periodic_box_vectors(&self) -> Option<Vec<Vec3>> {
        self.periodic_box_vectors.clone()
    }

    #[getter]
    fn parameters(&self) -> Option<HashMap<String, f64>> {
        self.parameters.clone()
    }

    #[getter]
    fn energy_parameter_derivatives(&self) -> Option<HashMap<String, f64>> {
        self.energy_parameter_derivatives.clone()
    }

    #[getter]
    fn positions(&self) -> Option<Vec<Vec3>> {
        self.positions.clone()
    }

    #[getter]
    fn velocities(&self) -> Option<Vec<Vec3>> {
        self.velocities.clone()
    }

    #[getter]
    fn forces(&self) -> Option<Vec<Vec3>> {
        self.forces.clone()
    }

    #[getter]
    fn potential_energy(&self) -> Option<f64> {
        self.potential_energy
    }

    #[getter]
    fn kinetic_energy(&self) -> Option<f64> {
        self.kinetic_energy
    }
}

#[pyclass(unsendable, name = "NonbondedForce")]
struct PyNonbondedForce {
    inner: NonbondedForce,
}

impl PyNonbondedForce {
    fn inner_mut(&mut self) -> &mut NonbondedForce {
        &mut self.inner
    }

    fn inner_ref(&self) -> &NonbondedForce {
        &self.inner
    }
}

#[pymethods]
impl PyNonbondedForce {
    fn get_reciprocal_space_force_group(&self) -> i32 {
        self.inner.reciprocal_space_force_group()
    }
    fn set_reciprocal_space_force_group(&mut self, group: i32) {
        self.inner.set_reciprocal_space_force_group(group);
    }
    fn get_include_direct_space(&self) -> bool {
        self.inner.include_direct_space()
    }
    fn set_include_direct_space(&mut self, enabled: bool) {
        self.inner.set_include_direct_space(enabled);
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

    #[new]
    fn new() -> Self {
        Self {
            inner: NonbondedForce::new(),
        }
    }

    fn add_particle(&mut self, charge: f64, sigma: f64, epsilon: f64) -> usize {
        self.inner_mut().add_particle(charge, sigma, epsilon)
    }

    fn add_global_parameter(&mut self, name: &str, default: f64) -> usize {
        self.inner.add_global_parameter(name, default)
    }

    fn num_global_parameters(&self) -> usize {
        self.inner.num_global_parameters()
    }
    fn get_global_parameter_name(&self, index: usize) -> &str {
        self.inner.global_parameter_name(index)
    }
    fn get_global_parameter_default_value(&self, index: usize) -> f64 {
        self.inner.global_parameter_default_value(index)
    }
    fn set_global_parameter_name(&mut self, index: usize, name: &str) {
        self.inner.set_global_parameter_name(index, name);
    }
    fn set_global_parameter_default_value(&mut self, index: usize, value: f64) {
        self.inner.set_global_parameter_default_value(index, value);
    }

    fn add_particle_parameter_offset(
        &mut self,
        parameter: &str,
        index: usize,
        charge: f64,
        sigma: f64,
        epsilon: f64,
    ) -> usize {
        self.inner
            .add_particle_parameter_offset(parameter, index, charge, sigma, epsilon)
    }

    fn add_exception_parameter_offset(
        &mut self,
        parameter: &str,
        index: usize,
        charge: f64,
        sigma: f64,
        epsilon: f64,
    ) -> usize {
        self.inner
            .add_exception_parameter_offset(parameter, index, charge, sigma, epsilon)
    }

    fn num_particle_parameter_offsets(&self) -> usize {
        self.inner.num_particle_parameter_offsets()
    }
    fn num_exception_parameter_offsets(&self) -> usize {
        self.inner.num_exception_parameter_offsets()
    }

    fn get_particle_parameter_offset(&self, index: usize) -> (&str, usize, f64, f64, f64) {
        let o = self.inner.particle_parameter_offset(index);
        (
            &o.parameter,
            o.index,
            o.charge_scale,
            o.sigma_scale,
            o.epsilon_scale,
        )
    }

    fn get_exception_parameter_offset(&self, index: usize) -> (&str, usize, f64, f64, f64) {
        let o = self.inner.exception_parameter_offset(index);
        (
            &o.parameter,
            o.index,
            o.charge_scale,
            o.sigma_scale,
            o.epsilon_scale,
        )
    }

    fn set_particle_parameter_offset(
        &mut self,
        offset: usize,
        parameter: &str,
        index: usize,
        charge: f64,
        sigma: f64,
        epsilon: f64,
    ) {
        self.inner
            .set_particle_parameter_offset(offset, parameter, index, charge, sigma, epsilon);
    }

    fn set_exception_parameter_offset(
        &mut self,
        offset: usize,
        parameter: &str,
        index: usize,
        charge: f64,
        sigma: f64,
        epsilon: f64,
    ) {
        self.inner
            .set_exception_parameter_offset(offset, parameter, index, charge, sigma, epsilon);
    }

    fn add_exception(
        &mut self,
        particle1: usize,
        particle2: usize,
        charge_prod: f64,
        sigma: f64,
        epsilon: f64,
    ) -> usize {
        self.inner_mut()
            .add_exception(particle1, particle2, charge_prod, sigma, epsilon)
    }

    fn create_exceptions_from_bonds(
        &mut self,
        bonds: Vec<(usize, usize)>,
        coulomb14scale: f64,
        lj14scale: f64,
    ) {
        self.inner_mut()
            .create_exceptions_from_bonds(&bonds, coulomb14scale, lj14scale);
    }

    fn set_cutoff_distance(&mut self, cutoff: f64) {
        self.inner_mut().set_cutoff_distance(cutoff);
    }

    fn set_reaction_field_dielectric(&mut self, dielectric: f64) {
        self.inner_mut().set_reaction_field_dielectric(dielectric);
    }

    fn set_ewald_error_tolerance(&mut self, tolerance: f64) {
        self.inner_mut().set_ewald_error_tolerance(tolerance);
    }

    fn set_use_dispersion_correction(&mut self, enabled: bool) {
        self.inner_mut().set_use_dispersion_correction(enabled);
    }
    fn set_use_switching_function(&mut self, enabled: bool) {
        self.inner_mut().set_use_switching_function(enabled);
    }
    fn set_switching_distance(&mut self, distance: f64) {
        self.inner_mut().set_switching_distance(distance);
    }
    fn set_exceptions_use_periodic_boundary_conditions(&mut self, enabled: bool) {
        self.inner_mut()
            .set_exceptions_use_periodic_boundary_conditions(enabled);
    }
    fn set_pme_parameters(&mut self, alpha: f64, nx: usize, ny: usize, nz: usize) {
        self.inner_mut().set_pme_parameters(alpha, [nx, ny, nz]);
    }
    fn get_pme_parameters(&self) -> (f64, usize, usize, usize) {
        let (alpha, [nx, ny, nz]) = self.inner_ref().pme_parameters();
        (alpha, nx, ny, nz)
    }

    fn set_ljpme_parameters(&mut self, alpha: f64, nx: usize, ny: usize, nz: usize) {
        self.inner_mut().set_ljpme_parameters(alpha, [nx, ny, nz]);
    }

    fn get_ljpme_parameters(&self) -> (f64, usize, usize, usize) {
        let (alpha, [nx, ny, nz]) = self.inner_ref().ljpme_parameters();
        (alpha, nx, ny, nz)
    }

    fn set_nonbonded_method(&mut self, method: &str) -> PyResult<()> {
        let method =
            match method {
                "NoCutoff" => NonbondedMethod::NoCutoff,
                "CutoffNonPeriodic" => NonbondedMethod::CutoffNonPeriodic,
                "CutoffPeriodic" => NonbondedMethod::CutoffPeriodic,
                "Ewald" => NonbondedMethod::Ewald,
                "PME" => NonbondedMethod::PME,
                "LJPME" => NonbondedMethod::LJPME,
                _ => return Err(err(
                    "unknown method; expected NoCutoff, CutoffNonPeriodic, CutoffPeriodic, Ewald, PME, or LJPME",
                )),
            };
        self.inner_mut().set_nonbonded_method(method);
        Ok(())
    }

    fn num_particles(&self) -> usize {
        self.inner_ref().num_particles()
    }

    fn num_exceptions(&self) -> usize {
        self.inner_ref().num_exceptions()
    }

    fn get_particle_parameters(&self, index: usize) -> (f64, f64, f64) {
        let params = self.inner_ref().particle_parameters(index);
        (params.charge, params.sigma, params.epsilon)
    }

    fn set_particle_parameters(&mut self, index: usize, charge: f64, sigma: f64, epsilon: f64) {
        self.inner_mut()
            .set_particle_parameters(index, charge, sigma, epsilon);
    }

    fn get_exception_parameters(&self, index: usize) -> (usize, usize, f64, f64, f64) {
        let params = self.inner_ref().exception_parameters(index);
        (
            params.particle1,
            params.particle2,
            params.charge_prod,
            params.sigma,
            params.epsilon,
        )
    }

    fn set_exception_parameters(
        &mut self,
        index: usize,
        particle1: usize,
        particle2: usize,
        charge_prod: f64,
        sigma: f64,
        epsilon: f64,
    ) {
        self.inner_mut().set_exception_parameters(
            index,
            particle1,
            particle2,
            charge_prod,
            sigma,
            epsilon,
        );
    }

    fn nonbonded_method(&self) -> String {
        let method = match self.inner_ref().nonbonded_method() {
            NonbondedMethod::NoCutoff => "NoCutoff",
            NonbondedMethod::CutoffNonPeriodic => "CutoffNonPeriodic",
            NonbondedMethod::CutoffPeriodic => "CutoffPeriodic",
            NonbondedMethod::Ewald => "Ewald",
            NonbondedMethod::PME => "PME",
            NonbondedMethod::LJPME => "LJPME",
        };
        method.to_string()
    }

    fn cutoff_distance(&self) -> f64 {
        self.inner_ref().cutoff_distance()
    }

    fn reaction_field_dielectric(&self) -> f64 {
        self.inner_ref().reaction_field_dielectric()
    }

    fn ewald_error_tolerance(&self) -> f64 {
        self.inner_ref().ewald_error_tolerance()
    }

    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.inner_ref().uses_periodic_boundary_conditions()
    }
}

#[pyclass(unsendable, name = "CustomExternalForce")]
struct PyCustomExternalForce {
    inner: CustomExternalForce,
}

impl PyCustomExternalForce {
    fn inner_mut(&mut self) -> &mut CustomExternalForce {
        &mut self.inner
    }

    fn inner_ref(&self) -> &CustomExternalForce {
        &self.inner
    }
}

#[pymethods]
impl PyCustomExternalForce {
    fn get_energy_function(&self) -> &str {
        self.inner.energy_function()
    }
    fn set_energy_function(&mut self, energy: &str) -> PyResult<()> {
        self.inner
            .set_energy_function(energy)
            .map_err(|e| err(&e.to_string()))
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

    #[new]
    fn new(energy: &str) -> PyResult<Self> {
        let inner = CustomExternalForce::new(energy).map_err(|e| err(&e.to_string()))?;
        Ok(Self { inner })
    }

    fn add_per_particle_parameter(&mut self, name: &str) -> usize {
        self.inner_mut().add_per_particle_parameter(name)
    }

    fn num_per_particle_parameters(&self) -> usize {
        self.inner_ref().num_per_particle_parameters()
    }

    fn get_per_particle_parameter_name(&self, index: usize) -> String {
        self.inner_ref()
            .per_particle_parameter_name(index)
            .to_string()
    }

    fn add_global_parameter(&mut self, name: &str, value: f64) {
        self.inner_mut().add_global_parameter(name, value);
    }

    fn num_global_parameters(&self) -> usize {
        self.inner_ref().num_global_parameters()
    }

    fn get_global_parameter_name(&self, index: usize) -> String {
        self.inner_ref().global_parameter_name(index).to_string()
    }

    fn get_global_parameter_default_value(&self, index: usize) -> f64 {
        self.inner_ref().global_parameter_default_value(index)
    }

    fn set_global_parameter_default_value(&mut self, index: usize, value: f64) {
        self.inner_mut()
            .set_global_parameter_default_value(index, value);
    }

    fn add_energy_parameter_derivative(&mut self, name: &str) {
        self.inner_mut().add_energy_parameter_derivative(name);
    }

    fn num_energy_parameter_derivatives(&self) -> usize {
        self.inner_ref().num_energy_parameter_derivatives()
    }

    fn get_energy_parameter_derivative_name(&self, index: usize) -> String {
        self.inner_ref()
            .energy_parameter_derivative_name(index)
            .to_string()
    }

    fn add_particle(&mut self, index: usize, params: Vec<f64>) -> usize {
        self.inner_mut().add_particle(index, params)
    }

    fn num_particles(&self) -> usize {
        self.inner_ref().num_particles()
    }

    fn get_particle_parameters(&self, index: usize) -> (usize, Vec<f64>) {
        let (particle, params) = self.inner_ref().particle_parameters(index);
        (particle, params.to_vec())
    }

    fn set_particle_parameters(&mut self, record_index: usize, index: usize, params: Vec<f64>) {
        self.inner_mut()
            .set_particle_parameters(record_index, index, params);
    }

    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.inner_ref().uses_periodic_boundary_conditions()
    }
}

#[pyclass(unsendable, name = "CustomBondForce")]
struct PyCustomBondForce {
    inner: CustomBondForce,
}

impl PyCustomBondForce {
    fn inner_mut(&mut self) -> &mut CustomBondForce {
        &mut self.inner
    }

    fn inner_ref(&self) -> &CustomBondForce {
        &self.inner
    }
}

#[pymethods]
impl PyCustomBondForce {
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

    #[new]
    fn new(energy: &str) -> PyResult<Self> {
        let inner = CustomBondForce::new(energy).map_err(|e| err(&e.to_string()))?;
        Ok(Self { inner })
    }

    fn add_per_bond_parameter(&mut self, name: &str) -> usize {
        self.inner_mut().add_per_bond_parameter(name)
    }

    fn num_per_bond_parameters(&mut self) -> usize {
        self.inner_mut().num_per_bond_parameters()
    }

    fn get_per_bond_parameter_name(&mut self, index: usize) -> String {
        self.inner_mut().per_bond_parameter_name(index).to_string()
    }

    fn add_global_parameter(&mut self, name: &str, value: f64) {
        self.inner_mut().add_global_parameter(name, value);
    }

    fn num_global_parameters(&mut self) -> usize {
        self.inner_mut().num_global_parameters()
    }

    fn get_global_parameter_name(&mut self, index: usize) -> String {
        self.inner_mut().global_parameter_name(index).to_string()
    }

    fn get_global_parameter_default_value(&mut self, index: usize) -> f64 {
        self.inner_mut().global_parameter_default_value(index)
    }

    fn set_global_parameter_default_value(&mut self, index: usize, value: f64) {
        self.inner_mut()
            .set_global_parameter_default_value(index, value);
    }

    fn add_energy_parameter_derivative(&mut self, name: &str) {
        self.inner_mut().add_energy_parameter_derivative(name);
    }

    fn num_energy_parameter_derivatives(&mut self) -> usize {
        self.inner_mut().num_energy_parameter_derivatives()
    }

    fn get_energy_parameter_derivative_name(&mut self, index: usize) -> String {
        self.inner_mut()
            .energy_parameter_derivative_name(index)
            .to_string()
    }

    fn add_bond(&mut self, particle1: usize, particle2: usize, params: Vec<f64>) -> usize {
        self.inner_mut().add_bond(particle1, particle2, params)
    }

    fn num_bonds(&mut self) -> usize {
        self.inner_mut().num_bonds()
    }

    fn get_bond_parameters(&mut self, index: usize) -> (usize, usize, Vec<f64>) {
        let (particle1, particle2, params) = self.inner_mut().bond_parameters(index);
        (particle1, particle2, params.to_vec())
    }

    fn set_bond_parameters(
        &mut self,
        index: usize,
        particle1: usize,
        particle2: usize,
        params: Vec<f64>,
    ) {
        self.inner_mut()
            .set_bond_parameters(index, particle1, particle2, params);
    }

    fn set_uses_periodic_boundary_conditions(&mut self, use_pbc: bool) {
        self.inner_mut()
            .set_uses_periodic_boundary_conditions(use_pbc);
    }

    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.inner_ref().uses_periodic_boundary_conditions()
    }
}

#[pyclass(unsendable, name = "CustomAngleForce")]
struct PyCustomAngleForce {
    inner: CustomAngleForce,
}

impl PyCustomAngleForce {
    fn inner_mut(&mut self) -> &mut CustomAngleForce {
        &mut self.inner
    }

    fn inner_ref(&self) -> &CustomAngleForce {
        &self.inner
    }
}

#[pymethods]
impl PyCustomAngleForce {
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

    #[new]
    fn new(energy: &str) -> PyResult<Self> {
        let inner = CustomAngleForce::new(energy).map_err(|e| err(&e.to_string()))?;
        Ok(Self { inner })
    }

    fn add_per_angle_parameter(&mut self, name: &str) -> usize {
        self.inner_mut().add_per_angle_parameter(name)
    }

    fn num_per_angle_parameters(&mut self) -> usize {
        self.inner_mut().num_per_angle_parameters()
    }

    fn get_per_angle_parameter_name(&mut self, index: usize) -> String {
        self.inner_mut().per_angle_parameter_name(index).to_string()
    }

    fn add_global_parameter(&mut self, name: &str, value: f64) {
        self.inner_mut().add_global_parameter(name, value);
    }

    fn num_global_parameters(&mut self) -> usize {
        self.inner_mut().num_global_parameters()
    }

    fn get_global_parameter_name(&mut self, index: usize) -> String {
        self.inner_mut().global_parameter_name(index).to_string()
    }

    fn get_global_parameter_default_value(&mut self, index: usize) -> f64 {
        self.inner_mut().global_parameter_default_value(index)
    }

    fn set_global_parameter_default_value(&mut self, index: usize, value: f64) {
        self.inner_mut()
            .set_global_parameter_default_value(index, value);
    }

    fn add_energy_parameter_derivative(&mut self, name: &str) {
        self.inner_mut().add_energy_parameter_derivative(name);
    }

    fn num_energy_parameter_derivatives(&mut self) -> usize {
        self.inner_mut().num_energy_parameter_derivatives()
    }

    fn get_energy_parameter_derivative_name(&mut self, index: usize) -> String {
        self.inner_mut()
            .energy_parameter_derivative_name(index)
            .to_string()
    }

    fn add_angle(
        &mut self,
        particle1: usize,
        particle2: usize,
        particle3: usize,
        params: Vec<f64>,
    ) -> usize {
        self.inner_mut()
            .add_angle(particle1, particle2, particle3, params)
    }

    fn num_angles(&mut self) -> usize {
        self.inner_mut().num_angles()
    }

    fn get_angle_parameters(&mut self, index: usize) -> (usize, usize, usize, Vec<f64>) {
        let (particle1, particle2, particle3, params) = self.inner_mut().angle_parameters(index);
        (particle1, particle2, particle3, params.to_vec())
    }

    fn set_angle_parameters(
        &mut self,
        index: usize,
        particle1: usize,
        particle2: usize,
        particle3: usize,
        params: Vec<f64>,
    ) {
        self.inner_mut()
            .set_angle_parameters(index, particle1, particle2, particle3, params);
    }

    fn set_uses_periodic_boundary_conditions(&mut self, use_pbc: bool) {
        self.inner_mut()
            .set_uses_periodic_boundary_conditions(use_pbc);
    }

    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.inner_ref().uses_periodic_boundary_conditions()
    }
}

#[pyclass(unsendable, name = "CustomNonbondedForce")]
struct PyCustomNonbondedForce {
    inner: CustomNonbondedForce,
    tables: Vec<tabulated::SharedTable>,
}

impl PyCustomNonbondedForce {
    fn snapshot(&self) -> CustomNonbondedForce {
        let mut force = self.inner.clone();
        for (index, table) in self.tables.iter().enumerate() {
            force
                .set_tabulated_function(index, table.borrow().clone())
                .unwrap();
        }
        force
    }
    fn inner_mut(&mut self) -> &mut CustomNonbondedForce {
        &mut self.inner
    }

    fn inner_ref(&self) -> &CustomNonbondedForce {
        &self.inner
    }
}

#[pymethods]
impl PyCustomNonbondedForce {
    #[classattr]
    #[pyo3(name = "NoCutoff")]
    const NO_CUTOFF: i32 = 0;
    #[classattr]
    #[pyo3(name = "CutoffNonPeriodic")]
    const CUTOFF_NON_PERIODIC: i32 = 1;
    #[classattr]
    #[pyo3(name = "CutoffPeriodic")]
    const CUTOFF_PERIODIC: i32 = 2;

    fn set_nonbonded_method(&mut self, method: i32) -> PyResult<()> {
        let method = match method {
            0 => CustomNonbondedMethod::NoCutoff,
            1 => CustomNonbondedMethod::CutoffNonPeriodic,
            2 => CustomNonbondedMethod::CutoffPeriodic,
            _ => return Err(err("invalid custom nonbonded method")),
        };
        self.inner.set_nonbonded_method(method);
        Ok(())
    }
    fn get_nonbonded_method(&self) -> i32 {
        match self.inner.nonbonded_method() {
            CustomNonbondedMethod::NoCutoff => 0,
            CustomNonbondedMethod::CutoffNonPeriodic => 1,
            CustomNonbondedMethod::CutoffPeriodic => 2,
        }
    }
    fn get_energy_function(&self) -> &str {
        self.inner.energy_function()
    }
    fn set_energy_function(&mut self, expression: &str) -> PyResult<()> {
        self.inner
            .set_energy_function(expression)
            .map_err(|error| err(&error))
    }
    fn add_computed_value(&mut self, name: &str, expression: &str) -> PyResult<usize> {
        self.inner
            .add_computed_value(name, expression)
            .map_err(|error| err(&error))
    }
    fn get_num_computed_values(&self) -> usize {
        self.inner.num_computed_values()
    }
    fn get_computed_value_parameters(&self, index: usize) -> (&str, &str) {
        self.inner.computed_value_parameters(index)
    }
    fn set_computed_value_parameters(
        &mut self,
        index: usize,
        name: &str,
        expression: &str,
    ) -> PyResult<()> {
        self.inner
            .set_computed_value_parameters(index, name, expression)
            .map_err(|error| err(&error))
    }
    fn add_interaction_group(
        &mut self,
        set1: std::collections::HashSet<usize>,
        set2: std::collections::HashSet<usize>,
    ) -> usize {
        self.inner.add_interaction_group(set1, set2)
    }
    fn get_num_interaction_groups(&self) -> usize {
        self.inner.num_interaction_groups()
    }
    fn get_interaction_group_parameters(
        &self,
        index: usize,
    ) -> (
        std::collections::HashSet<usize>,
        std::collections::HashSet<usize>,
    ) {
        let (a, b) = self.inner.interaction_group_parameters(index);
        (a.iter().copied().collect(), b.iter().copied().collect())
    }
    fn set_interaction_group_parameters(
        &mut self,
        index: usize,
        set1: std::collections::HashSet<usize>,
        set2: std::collections::HashSet<usize>,
    ) {
        self.inner
            .set_interaction_group_parameters(index, set1, set2);
    }
    fn get_use_switching_function(&self) -> bool {
        self.inner.use_switching_function()
    }
    fn set_use_switching_function(&mut self, enabled: bool) {
        self.inner.set_use_switching_function(enabled);
    }
    fn get_switching_distance(&self) -> f64 {
        self.inner.switching_distance()
    }
    fn set_switching_distance(&mut self, distance: f64) {
        self.inner.set_switching_distance(distance);
    }
    fn get_use_long_range_correction(&self) -> bool {
        self.inner.use_long_range_correction()
    }
    fn set_use_long_range_correction(&mut self, enabled: bool) {
        self.inner.set_use_long_range_correction(enabled);
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

    #[new]
    fn new(energy: &str) -> PyResult<Self> {
        let inner = CustomNonbondedForce::new(energy).map_err(|e| err(&e.to_string()))?;
        Ok(Self {
            inner,
            tables: Vec::new(),
        })
    }

    fn add_tabulated_function(
        &mut self,
        name: &str,
        function: &Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        let table = tabulated::extract(function)?;
        let index = self
            .inner
            .add_tabulated_function(name, table.borrow().clone())
            .map_err(|error| err(&error.to_string()))?;
        self.tables.push(table);
        Ok(index)
    }
    fn get_num_tabulated_functions(&self) -> usize {
        self.tables.len()
    }
    fn get_tabulated_function_name(&self, index: usize) -> PyResult<String> {
        self.tables.get(index).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err("tabulated function index out of range")
        })?;
        Ok(self.inner.tabulated_function_name(index).to_owned())
    }
    fn get_tabulated_function(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
        let table = self.tables.get(index).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err("tabulated function index out of range")
        })?;
        tabulated::wrap(py, table.clone())
    }

    fn add_per_particle_parameter(&mut self, name: &str) -> usize {
        self.inner_mut().add_per_particle_parameter(name)
    }

    fn num_per_particle_parameters(&mut self) -> usize {
        self.inner_mut().num_per_particle_parameters()
    }

    fn get_per_particle_parameter_name(&mut self, index: usize) -> String {
        self.inner_mut()
            .per_particle_parameter_name(index)
            .to_string()
    }

    fn add_global_parameter(&mut self, name: &str, value: f64) {
        self.inner_mut().add_global_parameter(name, value);
    }

    fn num_global_parameters(&mut self) -> usize {
        self.inner_mut().num_global_parameters()
    }

    fn get_global_parameter_name(&mut self, index: usize) -> String {
        self.inner_mut().global_parameter_name(index).to_string()
    }

    fn get_global_parameter_default_value(&mut self, index: usize) -> f64 {
        self.inner_mut().global_parameter_default_value(index)
    }

    fn set_global_parameter_default_value(&mut self, index: usize, value: f64) {
        self.inner_mut()
            .set_global_parameter_default_value(index, value);
    }

    fn add_energy_parameter_derivative(&mut self, name: &str) {
        self.inner_mut().add_energy_parameter_derivative(name);
    }

    fn num_energy_parameter_derivatives(&mut self) -> usize {
        self.inner_mut().num_energy_parameter_derivatives()
    }

    fn get_energy_parameter_derivative_name(&mut self, index: usize) -> String {
        self.inner_mut()
            .energy_parameter_derivative_name(index)
            .to_string()
    }

    fn add_particle(&mut self, params: Vec<f64>) -> usize {
        self.inner_mut().add_particle(params)
    }

    fn num_particles(&mut self) -> usize {
        self.inner_mut().num_particles()
    }

    fn get_particle_parameters(&mut self, index: usize) -> Vec<f64> {
        self.inner_mut().particle_parameters(index).to_vec()
    }

    fn set_particle_parameters(&mut self, index: usize, params: Vec<f64>) {
        self.inner_mut().set_particle_parameters(index, params);
    }

    fn create_exclusions_from_bonds(&mut self, bonds: Vec<(usize, usize)>, bond_cutoff: usize) {
        self.inner.create_exclusions_from_bonds(&bonds, bond_cutoff);
    }
    fn add_exclusion(&mut self, particle1: usize, particle2: usize) {
        self.inner_mut().add_exclusion(particle1, particle2);
    }

    fn num_exclusions(&mut self) -> usize {
        self.inner_mut().num_exclusions()
    }

    fn get_exclusion_particles(&mut self, index: usize) -> (usize, usize) {
        self.inner_mut().exclusion_particles(index)
    }

    fn set_exclusion_particles(&mut self, index: usize, particle1: usize, particle2: usize) {
        self.inner_mut()
            .set_exclusion_particles(index, particle1, particle2);
    }

    fn set_uses_periodic_boundary_conditions(&mut self, use_pbc: bool) {
        self.inner_mut()
            .set_uses_periodic_boundary_conditions(use_pbc);
    }

    fn set_cutoff_distance(&mut self, cutoff: f64) {
        self.inner_mut().set_cutoff_distance(cutoff);
    }

    fn cutoff_distance(&self) -> f64 {
        self.inner_ref().cutoff_distance()
    }

    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.inner_ref().uses_periodic_boundary_conditions()
    }
}

#[pyclass(unsendable, name = "CustomTorsionForce")]
struct PyCustomTorsionForce {
    inner: CustomTorsionForce,
}

impl PyCustomTorsionForce {
    fn inner_mut(&mut self) -> &mut CustomTorsionForce {
        &mut self.inner
    }

    fn inner_ref(&self) -> &CustomTorsionForce {
        &self.inner
    }
}

#[pymethods]
impl PyCustomTorsionForce {
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

    #[new]
    fn new(energy: &str) -> PyResult<Self> {
        let inner = CustomTorsionForce::new(energy).map_err(|e| err(&e.to_string()))?;
        Ok(Self { inner })
    }

    fn add_per_torsion_parameter(&mut self, name: &str) -> usize {
        self.inner_mut().add_per_torsion_parameter(name)
    }

    fn num_per_torsion_parameters(&mut self) -> usize {
        self.inner_mut().num_per_torsion_parameters()
    }

    fn get_per_torsion_parameter_name(&mut self, index: usize) -> String {
        self.inner_mut()
            .per_torsion_parameter_name(index)
            .to_string()
    }

    fn add_global_parameter(&mut self, name: &str, value: f64) {
        self.inner_mut().add_global_parameter(name, value);
    }

    fn num_global_parameters(&mut self) -> usize {
        self.inner_mut().num_global_parameters()
    }

    fn get_global_parameter_name(&mut self, index: usize) -> String {
        self.inner_mut().global_parameter_name(index).to_string()
    }

    fn get_global_parameter_default_value(&mut self, index: usize) -> f64 {
        self.inner_mut().global_parameter_default_value(index)
    }

    fn set_global_parameter_default_value(&mut self, index: usize, value: f64) {
        self.inner_mut()
            .set_global_parameter_default_value(index, value);
    }

    fn add_energy_parameter_derivative(&mut self, name: &str) {
        self.inner_mut().add_energy_parameter_derivative(name);
    }

    fn num_energy_parameter_derivatives(&mut self) -> usize {
        self.inner_mut().num_energy_parameter_derivatives()
    }

    fn get_energy_parameter_derivative_name(&mut self, index: usize) -> String {
        self.inner_mut()
            .energy_parameter_derivative_name(index)
            .to_string()
    }

    fn add_torsion(
        &mut self,
        particle1: usize,
        particle2: usize,
        particle3: usize,
        particle4: usize,
        params: Vec<f64>,
    ) -> usize {
        self.inner_mut()
            .add_torsion(particle1, particle2, particle3, particle4, params)
    }

    fn num_torsions(&mut self) -> usize {
        self.inner_mut().num_torsions()
    }

    fn get_torsion_parameters(&mut self, index: usize) -> (usize, usize, usize, usize, Vec<f64>) {
        let (particle1, particle2, particle3, particle4, params) =
            self.inner_mut().torsion_parameters(index);
        (particle1, particle2, particle3, particle4, params.to_vec())
    }

    fn set_torsion_parameters(
        &mut self,
        index: usize,
        particle1: usize,
        particle2: usize,
        particle3: usize,
        particle4: usize,
        params: Vec<f64>,
    ) {
        self.inner_mut()
            .set_torsion_parameters(index, particle1, particle2, particle3, particle4, params);
    }

    fn set_uses_periodic_boundary_conditions(&mut self, use_pbc: bool) {
        self.inner_mut()
            .set_uses_periodic_boundary_conditions(use_pbc);
    }

    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.inner_ref().uses_periodic_boundary_conditions()
    }
}

#[pyclass(unsendable, name = "HarmonicBondForce")]
struct PyHarmonicBondForce {
    inner: HarmonicBondForce,
}

impl PyHarmonicBondForce {
    fn inner_mut(&mut self) -> &mut HarmonicBondForce {
        &mut self.inner
    }

    fn inner_ref(&self) -> &HarmonicBondForce {
        &self.inner
    }
}

#[pymethods]
impl PyHarmonicBondForce {
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

    #[new]
    fn new() -> Self {
        Self {
            inner: HarmonicBondForce::new(),
        }
    }

    fn add_bond(&mut self, particle1: usize, particle2: usize, length: f64, k: f64) -> usize {
        self.inner_mut().add_bond(particle1, particle2, length, k)
    }

    fn set_uses_periodic_boundary_conditions(&mut self, use_pbc: bool) {
        self.inner_mut()
            .set_uses_periodic_boundary_conditions(use_pbc);
    }

    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.inner_ref().uses_periodic_boundary_conditions()
    }

    fn num_bonds(&self) -> usize {
        self.inner_ref().num_bonds()
    }

    fn get_bond_parameters(&self, index: usize) -> (usize, usize, f64, f64) {
        let params = self.inner_ref().bond_parameters(index);
        (params.particle1, params.particle2, params.length, params.k)
    }

    fn set_bond_parameters(
        &mut self,
        index: usize,
        particle1: usize,
        particle2: usize,
        length: f64,
        k: f64,
    ) {
        self.inner_mut()
            .set_bond_parameters(index, particle1, particle2, length, k);
    }
}

#[pyclass(unsendable, name = "HarmonicAngleForce")]
struct PyHarmonicAngleForce {
    inner: HarmonicAngleForce,
}

impl PyHarmonicAngleForce {
    fn inner_mut(&mut self) -> &mut HarmonicAngleForce {
        &mut self.inner
    }

    fn inner_ref(&self) -> &HarmonicAngleForce {
        &self.inner
    }
}

#[pymethods]
impl PyHarmonicAngleForce {
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

    #[new]
    fn new() -> Self {
        Self {
            inner: HarmonicAngleForce::new(),
        }
    }

    fn add_angle(
        &mut self,
        particle1: usize,
        particle2: usize,
        particle3: usize,
        angle: f64,
        k: f64,
    ) -> usize {
        self.inner_mut()
            .add_angle(particle1, particle2, particle3, angle, k)
    }

    fn set_uses_periodic_boundary_conditions(&mut self, use_pbc: bool) {
        self.inner_mut()
            .set_uses_periodic_boundary_conditions(use_pbc);
    }

    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.inner_ref().uses_periodic_boundary_conditions()
    }

    fn num_angles(&self) -> usize {
        self.inner_ref().num_angles()
    }

    fn get_angle_parameters(&self, index: usize) -> (usize, usize, usize, f64, f64) {
        let params = self.inner_ref().angle_parameters(index);
        (
            params.particle1,
            params.particle2,
            params.particle3,
            params.angle,
            params.k,
        )
    }

    fn set_angle_parameters(
        &mut self,
        index: usize,
        particle1: usize,
        particle2: usize,
        particle3: usize,
        angle: f64,
        k: f64,
    ) {
        self.inner_mut()
            .set_angle_parameters(index, particle1, particle2, particle3, angle, k);
    }
}

#[pyclass(unsendable, name = "PeriodicTorsionForce")]
struct PyPeriodicTorsionForce {
    inner: PeriodicTorsionForce,
}

impl PyPeriodicTorsionForce {
    fn inner_mut(&mut self) -> &mut PeriodicTorsionForce {
        &mut self.inner
    }

    fn inner_ref(&self) -> &PeriodicTorsionForce {
        &self.inner
    }
}

#[pymethods]
impl PyPeriodicTorsionForce {
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

    #[new]
    fn new() -> Self {
        Self {
            inner: PeriodicTorsionForce::new(),
        }
    }

    fn add_torsion(
        &mut self,
        particles: (usize, usize, usize, usize),
        periodicity: i32,
        phase: f64,
        k: f64,
    ) -> usize {
        let (particle1, particle2, particle3, particle4) = particles;
        self.inner_mut().add_torsion(TorsionParam::new(
            [particle1, particle2, particle3, particle4],
            periodicity,
            phase,
            k,
        ))
    }

    fn set_uses_periodic_boundary_conditions(&mut self, use_pbc: bool) {
        self.inner_mut()
            .set_uses_periodic_boundary_conditions(use_pbc);
    }

    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.inner_ref().uses_periodic_boundary_conditions()
    }

    fn num_torsions(&self) -> usize {
        self.inner_ref().num_torsions()
    }

    fn get_torsion_parameters(&self, index: usize) -> PeriodicTorsionParameters {
        let params = self.inner_ref().torsion_parameters(index);
        (
            (
                params.particle1,
                params.particle2,
                params.particle3,
                params.particle4,
            ),
            params.periodicity,
            params.phase,
            params.k,
        )
    }

    fn set_torsion_parameters(
        &mut self,
        index: usize,
        particles: (usize, usize, usize, usize),
        periodicity: i32,
        phase: f64,
        k: f64,
    ) {
        let (particle1, particle2, particle3, particle4) = particles;
        self.inner_mut().set_torsion_parameters(
            index,
            TorsionParam::new(
                [particle1, particle2, particle3, particle4],
                periodicity,
                phase,
                k,
            ),
        );
    }
}

#[pyclass(unsendable, name = "RBTorsionForce")]
struct PyRBTorsionForce {
    inner: RBTorsionForce,
}

impl PyRBTorsionForce {
    fn inner_mut(&mut self) -> &mut RBTorsionForce {
        &mut self.inner
    }

    fn inner_ref(&self) -> &RBTorsionForce {
        &self.inner
    }
}

#[pymethods]
impl PyRBTorsionForce {
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

    #[new]
    fn new() -> Self {
        Self {
            inner: RBTorsionForce::new(),
        }
    }

    fn add_torsion(
        &mut self,
        particles: (usize, usize, usize, usize),
        coefficients: RBCoefficients,
    ) -> usize {
        let (particle1, particle2, particle3, particle4) = particles;
        let (c0, c1, c2, c3, c4, c5) = coefficients;
        self.inner_mut().add_torsion(RBTorsionParam::new(
            [particle1, particle2, particle3, particle4],
            [c0, c1, c2, c3, c4, c5],
        ))
    }

    fn set_uses_periodic_boundary_conditions(&mut self, use_pbc: bool) {
        self.inner_mut()
            .set_uses_periodic_boundary_conditions(use_pbc);
    }

    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.inner_ref().uses_periodic_boundary_conditions()
    }

    fn num_torsions(&self) -> usize {
        self.inner_ref().num_torsions()
    }

    fn get_torsion_parameters(&self, index: usize) -> RBTorsionParameters {
        let params = self.inner_ref().torsion_parameters(index);
        (
            (
                params.particle1,
                params.particle2,
                params.particle3,
                params.particle4,
            ),
            (
                params.coefficients[0],
                params.coefficients[1],
                params.coefficients[2],
                params.coefficients[3],
                params.coefficients[4],
                params.coefficients[5],
            ),
        )
    }

    fn set_torsion_parameters(
        &mut self,
        index: usize,
        particles: TorsionParticles,
        coefficients: RBCoefficients,
    ) {
        let (particle1, particle2, particle3, particle4) = particles;
        let (c0, c1, c2, c3, c4, c5) = coefficients;
        self.inner_mut().set_torsion_parameters(
            index,
            RBTorsionParam::new(
                [particle1, particle2, particle3, particle4],
                [c0, c1, c2, c3, c4, c5],
            ),
        );
    }
}

#[pyclass(unsendable, name = "System")]
struct PySystem {
    inner: System<B>,
    force_objects: Vec<Py<PyAny>>,
    force_owner: ownership::ForceOwner,
}

impl PySystem {
    fn snapshot(&self, py: Python<'_>) -> PyResult<System<B>> {
        let mut system = self.inner.clone();
        for force in &self.force_objects {
            system.add_force(extract_force_snapshot(force.bind(py))?);
        }
        Ok(system)
    }

    fn inner_mut(&mut self) -> &mut System<B> {
        &mut self.inner
    }

    fn inner_ref(&self) -> &System<B> {
        &self.inner
    }
}

#[pymethods]
impl PySystem {
    #[new]
    fn new() -> Self {
        Self {
            inner: System::new(),
            force_objects: Vec::new(),
            force_owner: ownership::ForceOwner::default(),
        }
    }

    fn add_particle(&mut self, mass: f64) -> usize {
        self.inner_mut().add_particle(mass)
    }

    fn num_particles(&self) -> usize {
        self.inner_ref().num_particles()
    }

    fn get_particle_mass(&self, index: usize) -> f64 {
        self.inner_ref().particle(index).mass
    }

    fn set_particle_mass(&mut self, index: usize, mass: f64) {
        self.inner_mut().set_particle_mass(index, mass);
    }

    fn set_virtual_site(&mut self, index: usize, site: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner_mut()
            .set_virtual_site(index, virtual_sites::from_python(site)?);
        Ok(())
    }
    fn is_virtual_site(&self, index: usize) -> bool {
        self.inner_ref().is_virtual_site(index)
    }
    fn get_virtual_site(&self, index: usize, py: Python<'_>) -> PyResult<Py<PyAny>> {
        virtual_sites::to_python(self.inner_ref().virtual_site(index), py)
    }

    fn add_constraint(&mut self, particle1: usize, particle2: usize, distance: f64) -> usize {
        self.inner_mut()
            .add_constraint(particle1, particle2, distance)
    }

    fn num_constraints(&self) -> usize {
        self.inner_ref().num_constraints()
    }

    fn get_constraint_parameters(&self, index: usize) -> (usize, usize, f64) {
        let constraint = self.inner_ref().constraint_parameters(index);
        (
            constraint.particle1,
            constraint.particle2,
            constraint.distance,
        )
    }

    fn set_constraint_parameters(
        &mut self,
        index: usize,
        particle1: usize,
        particle2: usize,
        distance: f64,
    ) {
        self.inner_mut()
            .set_constraint_parameters(index, particle1, particle2, distance);
    }

    fn remove_constraint(&mut self, index: usize) {
        self.inner_mut().remove_constraint(index);
    }

    fn num_forces(&self) -> usize {
        self.force_objects.len()
    }

    fn set_default_periodic_box_vectors(&mut self, a: Vec3, b: Vec3, c: Vec3) {
        self.inner_mut()
            .set_default_periodic_box_vectors(to_box_vectors(a, b, c));
    }

    fn get_default_periodic_box_vectors(&self) -> Option<Vec<Vec3>> {
        self.inner_ref()
            .default_periodic_box_vectors()
            .map(|vectors| {
                vectors
                    .iter()
                    .map(|v| (v[0], v[1], v[2]))
                    .collect::<Vec<_>>()
            })
    }

    fn add_force(&mut self, force: &Bound<'_, PyAny>) -> PyResult<usize> {
        extract_force_snapshot(force)?;
        self.force_owner.claim(force)?;
        self.force_objects.push(force.clone().unbind());
        Ok(self.force_objects.len() - 1)
    }

    fn get_force(&self, index: usize, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.force_objects
            .get(index)
            .map(|force| force.clone_ref(py))
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("force index out of range"))
    }

    fn get_forces(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.force_objects
            .iter()
            .map(|force| force.clone_ref(py))
            .collect()
    }

    fn remove_force(&mut self, index: usize, py: Python<'_>) -> PyResult<()> {
        if index >= self.force_objects.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "force index out of range",
            ));
        }
        let force = self.force_objects.remove(index);
        self.force_owner.release(force.bind(py));
        Ok(())
    }

    fn uses_periodic_boundary_conditions(&self, py: Python<'_>) -> PyResult<bool> {
        for force in &self.force_objects {
            if extract_force_snapshot(force.bind(py))?.uses_periodic_boundary_conditions() {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn create_context(slf: PyRef<'_, Self>, integrator: &Bound<'_, PyAny>) -> PyResult<PyContext> {
        integrators::create_context(slf.into(), integrator)
    }
}

fn extract_force_snapshot(force: &Bound<'_, PyAny>) -> PyResult<Box<dyn openmm_core::Force<B>>> {
    if let Ok(force) = force.extract::<PyRef<'_, PyGayBerneForce>>() {
        return Ok(Box::new(force.inner.clone()));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyLCPOForce>>() {
        return Ok(Box::new(force.inner.clone()));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyCustomGBForce>>() {
        return Ok(Box::new(force.snapshot()?));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyGBSAOBCForce>>() {
        return Ok(Box::new(force.inner.clone()));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyMonteCarloAnisotropicBarostat>>() {
        return Ok(Box::new(force.inner.clone()));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyMonteCarloMembraneBarostat>>() {
        return Ok(Box::new(force.inner.clone()));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyMonteCarloFlexibleBarostat>>() {
        return Ok(Box::new(force.inner.clone()));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyMonteCarloBarostat>>() {
        return Ok(Box::new(force.inner.clone()));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyAndersenThermostat>>() {
        return Ok(Box::new(force.inner.clone()));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyCMMotionRemover>>() {
        return Ok(Box::new(force.inner.clone()));
    }
    if let Ok(cv) = force.extract::<PyRef<'_, PyCustomCVForce>>() {
        return Ok(Box::new(cv.snapshot(force.py())?));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyRMSDForce>>() {
        return Ok(Box::new(force.inner.clone()));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyRGForce>>() {
        return Ok(Box::new(force.inner.clone()));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyOrientationRestraintForce>>() {
        return Ok(Box::new(force.inner.clone()));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyCustomVolumeForce>>() {
        return Ok(Box::new(force.inner.clone()));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyCustomHbondForce>>() {
        return Ok(Box::new(force.snapshot()));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyCustomCentroidBondForce>>() {
        return Ok(Box::new(force.snapshot()));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyCustomCompoundBondForce>>() {
        return Ok(Box::new(force.snapshot()));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyCMAPTorsionForce>>() {
        return Ok(Box::new(force.inner.clone()));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyNonbondedForce>>() {
        let inner = &force.inner;
        return Ok(Box::new(inner.clone()));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyCustomExternalForce>>() {
        let inner = &force.inner;
        return Ok(Box::new(inner.clone()));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyCustomBondForce>>() {
        let inner = &force.inner;
        return Ok(Box::new(inner.clone()));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyCustomAngleForce>>() {
        let inner = &force.inner;
        return Ok(Box::new(inner.clone()));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyCustomNonbondedForce>>() {
        return Ok(Box::new(force.snapshot()));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyCustomTorsionForce>>() {
        let inner = &force.inner;
        return Ok(Box::new(inner.clone()));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyHarmonicBondForce>>() {
        let inner = &force.inner;
        return Ok(Box::new(inner.clone()));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyHarmonicAngleForce>>() {
        let inner = &force.inner;
        return Ok(Box::new(inner.clone()));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyPeriodicTorsionForce>>() {
        let inner = &force.inner;
        return Ok(Box::new(inner.clone()));
    }
    if let Ok(force) = force.extract::<PyRef<'_, PyRBTorsionForce>>() {
        let inner = &force.inner;
        return Ok(Box::new(inner.clone()));
    }
    Err(err("unsupported force type"))
}

#[pyclass(unsendable, weakref, name = "Context")]
struct PyContext {
    inner: integrators::SharedContext,
    system: Py<PySystem>,
    integrator: Py<PyAny>,
    integrator_preparation: integrators::ContextLink,
    force_objects: Vec<Py<PyAny>>,
}

#[pyclass(name = "MinimizationResult", get_all)]
struct PyMinimizationResult {
    iterations: usize,
    energy: f64,
    force_rms: f64,
    max_constraint_error: f64,
    termination: String,
}

#[pyclass(name = "LocalEnergyMinimizer")]
struct PyLocalEnergyMinimizer;

#[pymethods]
impl PyLocalEnergyMinimizer {
    #[staticmethod]
    #[pyo3(signature = (context, tolerance=10.0, max_iterations=0))]
    fn minimize(
        context: PyRef<'_, PyContext>,
        tolerance: f64,
        max_iterations: usize,
    ) -> PyResult<()> {
        openmm_core::LocalEnergyMinimizer::minimize(
            &mut context.inner.borrow_mut(),
            tolerance,
            max_iterations,
        )
        .map_err(|error| err(&error.to_string()))?;
        Ok(())
    }
}

#[pymethods]
impl PyContext {
    #[new]
    fn new(system: Py<PySystem>, integrator: &Bound<'_, PyAny>) -> PyResult<Self> {
        integrators::create_context(system, integrator)
    }

    fn get_system(&self, py: Python<'_>) -> Py<PySystem> {
        self.system.clone_ref(py)
    }

    fn get_integrator(&self, py: Python<'_>) -> Py<PyAny> {
        self.integrator.clone_ref(py)
    }

    #[pyo3(signature = (tolerance=10.0, max_iterations=0))]
    fn minimize(
        &mut self,
        tolerance: f64,
        max_iterations: usize,
    ) -> PyResult<PyMinimizationResult> {
        let result = openmm_core::LocalEnergyMinimizer::minimize(
            &mut self.inner.borrow_mut(),
            tolerance,
            max_iterations,
        )
        .map_err(|error| err(&error.to_string()))?;
        Ok(PyMinimizationResult {
            iterations: result.iterations,
            energy: result.energy,
            force_rms: result.force_rms,
            max_constraint_error: result.max_constraint_error,
            termination: format!("{:?}", result.termination),
        })
    }
    fn get_molecules(&self) -> Vec<Vec<usize>> {
        self.inner.borrow().molecules().to_vec()
    }
    fn compute_virtual_sites(&mut self) {
        self.inner.borrow_mut().compute_virtual_sites();
    }

    fn apply_constraints(&mut self, tolerance: f64) -> PyResult<()> {
        self.inner
            .borrow_mut()
            .apply_constraints(tolerance)
            .map_err(|error| err(&error.to_string()))
    }

    fn apply_velocity_constraints(&mut self, tolerance: f64) -> PyResult<()> {
        self.inner
            .borrow_mut()
            .apply_velocity_constraints(tolerance)
            .map_err(|error| err(&error.to_string()))
    }

    fn set_positions(&mut self, positions: Vec<Vec3>) {
        let device: Device = Default::default();
        self.inner
            .borrow_mut()
            .set_positions(to_tensor3(&positions, &device));
    }

    fn set_velocities(&mut self, velocities: Vec<Vec3>) {
        let device: Device = Default::default();
        self.inner
            .borrow_mut()
            .set_velocities(to_tensor3(&velocities, &device));
    }

    fn set_velocities_to_temperature(&mut self, temperature: f64) {
        self.inner
            .borrow_mut()
            .set_velocities_to_temperature(temperature);
    }

    fn set_state(&mut self, state: &PyState) -> PyResult<()> {
        let device: Device = Default::default();
        let state = state.to_core(&device)?;
        self.inner.borrow_mut().set_state(&state);
        Ok(())
    }

    fn set_parameter(&mut self, name: &str, value: f64) {
        self.inner.borrow_mut().set_parameter(name, value);
    }

    fn get_parameter(&self, name: &str) -> f64 {
        self.inner.borrow().get_parameter(name)
    }

    fn get_parameters(&self) -> HashMap<String, f64> {
        self.inner.borrow().get_parameters().clone()
    }

    fn set_time(&mut self, time: f64) {
        self.inner.borrow_mut().set_time(time);
    }

    #[getter]
    fn time(&self) -> f64 {
        self.inner.borrow().get_time()
    }

    fn set_step_count(&mut self, step_count: u64) {
        self.inner.borrow_mut().set_step_count(step_count);
    }

    #[getter]
    fn step_count(&self) -> u64 {
        self.inner.borrow().get_step_count()
    }

    fn set_periodic_box_vectors(&mut self, a: Vec3, b: Vec3, c: Vec3) {
        self.inner
            .borrow_mut()
            .set_periodic_box_vectors(to_box_vectors(a, b, c));
    }

    fn get_periodic_box_vectors(&self) -> Option<Vec<Vec3>> {
        self.inner.borrow().periodic_box_vectors().map(|vectors| {
            vectors
                .iter()
                .map(|v| (v[0], v[1], v[2]))
                .collect::<Vec<_>>()
        })
    }

    fn get_collective_variable_values(&self, force_index: usize) -> Vec<f64> {
        self.inner.borrow().collective_variable_values(force_index)
    }
    fn update_force_parameters_at_path(
        &mut self,
        path: Vec<usize>,
        force: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let force = extract_force_snapshot(force)?;
        self.inner
            .borrow_mut()
            .update_force_parameters_at_path(&path, force.as_ref())
            .map_err(|e| err(&e))
    }
    fn update_parameters_in_context(
        &mut self,
        force_index: usize,
        force: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if let Ok(cv) = force.extract::<PyRef<'_, PyCustomCVForce>>() {
            self.inner
                .borrow_mut()
                .update_force_parameters_at_path(&[force_index], &cv.snapshot(force.py())?)
                .map_err(|e| err(&e))?;
            return Ok(());
        }
        if let Ok(force) = force.extract::<PyRef<'_, PyRMSDForce>>() {
            self.inner
                .borrow_mut()
                .update_rmsd_force_parameters(force_index, &force.inner);
            return Ok(());
        }
        if let Ok(force) = force.extract::<PyRef<'_, PyOrientationRestraintForce>>() {
            self.inner
                .borrow_mut()
                .update_orientation_restraint_force_parameters(force_index, &force.inner);
            return Ok(());
        }
        if let Ok(force) = force.extract::<PyRef<'_, PyCustomHbondForce>>() {
            self.inner
                .borrow_mut()
                .update_custom_hbond_force_parameters(force_index, &force.snapshot());
            return Ok(());
        }
        if let Ok(force) = force.extract::<PyRef<'_, PyCustomCentroidBondForce>>() {
            self.inner
                .borrow_mut()
                .update_custom_centroid_force_parameters(force_index, &force.snapshot());
            return Ok(());
        }
        if let Ok(force) = force.extract::<PyRef<'_, PyCustomCompoundBondForce>>() {
            self.inner
                .borrow_mut()
                .update_custom_compound_force_parameters(force_index, &force.snapshot());
            return Ok(());
        }
        if let Ok(force) = force.extract::<PyRef<'_, PyCMAPTorsionForce>>() {
            self.inner
                .borrow_mut()
                .update_cmap_torsion_force_parameters(force_index, &force.inner);
            return Ok(());
        }
        if let Ok(force) = force.extract::<PyRef<'_, PyCustomExternalForce>>() {
            let inner = &force.inner;
            self.inner
                .borrow_mut()
                .update_custom_external_force_parameters(force_index, inner);
            return Ok(());
        }
        if let Ok(force) = force.extract::<PyRef<'_, PyCustomBondForce>>() {
            let inner = &force.inner;
            self.inner
                .borrow_mut()
                .update_custom_bond_force_parameters(force_index, inner);
            return Ok(());
        }
        if let Ok(force) = force.extract::<PyRef<'_, PyCustomAngleForce>>() {
            let inner = &force.inner;
            self.inner
                .borrow_mut()
                .update_custom_angle_force_parameters(force_index, inner);
            return Ok(());
        }
        if let Ok(force) = force.extract::<PyRef<'_, PyCustomNonbondedForce>>() {
            self.inner
                .borrow_mut()
                .update_custom_nonbonded_force_parameters(force_index, &force.snapshot());
            return Ok(());
        }
        if let Ok(force) = force.extract::<PyRef<'_, PyCustomTorsionForce>>() {
            let inner = &force.inner;
            self.inner
                .borrow_mut()
                .update_custom_torsion_force_parameters(force_index, inner);
            return Ok(());
        }
        if let Ok(force) = force.extract::<PyRef<'_, PyCustomGBForce>>() {
            self.inner
                .borrow_mut()
                .update_custom_gb_force_parameters(force_index, &force.snapshot()?);
            return Ok(());
        }
        if let Ok(force) = force.extract::<PyRef<'_, PyGayBerneForce>>() {
            self.inner
                .borrow_mut()
                .update_gay_berne_force_parameters(force_index, &force.inner);
            return Ok(());
        }
        if let Ok(force) = force.extract::<PyRef<'_, PyLCPOForce>>() {
            self.inner
                .borrow_mut()
                .update_lcpo_force_parameters(force_index, &force.inner);
            return Ok(());
        }
        if let Ok(force) = force.extract::<PyRef<'_, PyGBSAOBCForce>>() {
            self.inner
                .borrow_mut()
                .update_gbsa_obc_force_parameters(force_index, &force.inner);
            return Ok(());
        }
        if let Ok(force) = force.extract::<PyRef<'_, PyNonbondedForce>>() {
            let inner = &force.inner;
            self.inner
                .borrow_mut()
                .update_nonbonded_force_parameters(force_index, inner);
            return Ok(());
        }
        if let Ok(force) = force.extract::<PyRef<'_, PyHarmonicBondForce>>() {
            let inner = &force.inner;
            self.inner
                .borrow_mut()
                .update_harmonic_bond_force_parameters(force_index, inner);
            return Ok(());
        }
        if let Ok(force) = force.extract::<PyRef<'_, PyHarmonicAngleForce>>() {
            let inner = &force.inner;
            self.inner
                .borrow_mut()
                .update_harmonic_angle_force_parameters(force_index, inner);
            return Ok(());
        }
        if let Ok(force) = force.extract::<PyRef<'_, PyPeriodicTorsionForce>>() {
            let inner = &force.inner;
            self.inner
                .borrow_mut()
                .update_periodic_torsion_force_parameters(force_index, inner);
            return Ok(());
        }
        if let Ok(force) = force.extract::<PyRef<'_, PyRBTorsionForce>>() {
            let inner = &force.inner;
            self.inner
                .borrow_mut()
                .update_rb_torsion_force_parameters(force_index, inner);
            return Ok(());
        }
        Err(err(
            "unsupported force type for update_parameters_in_context",
        ))
    }

    fn step(&mut self, steps: usize) -> PyResult<()> {
        self.integrator_preparation.prepare()?;
        self.inner
            .borrow_mut()
            .try_step(steps)
            .map_err(|error| err(&error.to_string()))
    }

    #[pyo3(signature = (positions=false, velocities=false, forces=false, energy=false, parameters=false, parameter_derivatives=false, groups=-1, enforce_periodic_box=false))]
    #[allow(clippy::too_many_arguments)]
    fn get_state(
        &self,
        positions: bool,
        velocities: bool,
        forces: bool,
        energy: bool,
        parameters: bool,
        parameter_derivatives: bool,
        groups: i64,
        enforce_periodic_box: bool,
    ) -> PyResult<PyState> {
        self.integrator_preparation.prepare()?;
        Ok(PyState::from_core(
            self.inner
                .borrow()
                .try_get_state_options(
                    StateFlags {
                        positions,
                        velocities,
                        forces,
                        energy,
                        parameters,
                        parameter_derivatives,
                    },
                    group_mask(groups)?,
                    enforce_periodic_box,
                )
                .map_err(|e| err(&e.to_string()))?,
        ))
    }
}

#[pymodule]
fn openmm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLocalEnergyMinimizer>()?;
    m.add_class::<PyMinimizationResult>()?;
    virtual_sites::register(m)?;
    m.add_class::<PyState>()?;
    integrators::register(m)?;
    gbsa_obc::register(m)?;
    lcpo::register(m)?;
    gay_berne::register(m)?;
    custom_gb::register(m)?;
    m.add_class::<PyNonbondedForce>()?;
    m.add_class::<PyCustomExternalForce>()?;
    m.add_class::<PyCustomBondForce>()?;
    m.add_class::<PyCustomCompoundBondForce>()?;
    m.add_class::<PyCustomCentroidBondForce>()?;
    m.add_class::<PyCustomHbondForce>()?;
    m.add_class::<PyRMSDForce>()?;
    m.add_class::<PyRGForce>()?;
    m.add_class::<PyOrientationRestraintForce>()?;
    m.add_class::<PyCustomVolumeForce>()?;
    m.add_class::<PyCustomCVForce>()?;
    m.add_class::<PyCMMotionRemover>()?;
    m.add_class::<PyAndersenThermostat>()?;
    barostat::register(m)?;
    flexible_barostat::register(m)?;
    membrane_barostat::register(m)?;
    anisotropic_barostat::register(m)?;
    m.add_class::<PyCustomAngleForce>()?;
    m.add_class::<PyCustomNonbondedForce>()?;
    m.add_class::<PyCMAPTorsionForce>()?;
    tabulated::register(m)?;
    m.add_class::<PyCustomTorsionForce>()?;
    m.add_class::<PyHarmonicBondForce>()?;
    m.add_class::<PyHarmonicAngleForce>()?;
    m.add_class::<PyPeriodicTorsionForce>()?;
    m.add_class::<PyRBTorsionForce>()?;
    m.add_class::<PySystem>()?;
    m.add_class::<PyContext>()?;
    Ok(())
}

fn group_mask(groups: i64) -> PyResult<u32> {
    if groups < i32::MIN as i64 || groups > u32::MAX as i64 {
        return Err(err("force groups must be a 32-bit mask"));
    }
    Ok(groups as u32)
}
