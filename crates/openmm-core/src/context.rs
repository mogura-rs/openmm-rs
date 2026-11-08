mod pressure;
use crate::forces::alignment::{OrientationRestraintForce, RMSDForce};
use crate::forces::custom_cv::{update_force_parameters, CustomCVForce};
// Context owns the mutable simulation state and drives the MD loop.
// Generic over burn Backend: NdArray (CPU), Wgpu (GPU), Cuda.
use burn::prelude::*;

use crate::constraints::{ConstraintError, ConstraintSolver};
use crate::forces::cmap_torsion::CMAPTorsionForce;
use crate::forces::custom_angle::CustomAngleForce;
use crate::forces::custom_bond::CustomBondForce;
use crate::forces::custom_centroid::CustomCentroidBondForce;
use crate::forces::custom_compound::CustomCompoundBondForce;
use crate::forces::custom_external::CustomExternalForce;
use crate::forces::custom_hbond::CustomHbondForce;
use crate::forces::custom_nonbonded::CustomNonbondedForce;
use crate::forces::custom_torsion::CustomTorsionForce;
use crate::forces::harmonic_angle::HarmonicAngleForce;
use crate::forces::harmonic_bond::HarmonicBondForce;
use crate::forces::nonbonded::NonbondedForce;
use crate::forces::periodic_torsion::PeriodicTorsionForce;
use crate::forces::rb_torsion::RBTorsionForce;
use crate::integrators::langevin::BOLTZMANN;
use crate::integrators::Integrator;
use crate::state::{State, StateFlags};
use crate::system::{validate_box_vectors, System};
use crate::units::BoxVectors;
use crate::virtual_sites::VirtualSitePlan;
use std::collections::HashMap;
use std::sync::OnceLock;

pub struct Context<B: Backend> {
    system: System<B>,
    integrator: Box<dyn Integrator<B>>,
    positions: Tensor<B, 2>,  // [N, 3] nm
    velocities: Tensor<B, 2>, // [N, 3] nm/ps
    periodic_box_vectors: Option<BoxVectors>,
    parameters: HashMap<String, f64>,
    time: f64,
    step_count: u64,
    device: B::Device,
    constraints: ConstraintSolver,
    masses: Tensor<B, 1>,
    host_masses: Vec<f64>,
    virtual_sites: VirtualSitePlan,
    molecules: OnceLock<Vec<Vec<usize>>>,
    empty_forces: OnceLock<Tensor<B, 2>>,
    box_update_positions: Option<Tensor<B, 2>>,
}

impl<B: Backend> Context<B> {
    pub fn system(&self) -> &System<B> {
        &self.system
    }
    pub fn integrator(&self) -> &dyn Integrator<B> {
        self.integrator.as_ref()
    }
    pub fn integrator_mut(&mut self) -> &mut dyn Integrator<B> {
        self.integrator.as_mut()
    }

    pub fn new(
        mut system: System<B>,
        mut integrator: Box<dyn Integrator<B>>,
        device: B::Device,
    ) -> Self {
        let n = system.num_particles();
        assert!(
            n > 0,
            "cannot create a context for a system with no particles"
        );
        let periodic_box_vectors = system.default_periodic_box_vectors().copied();
        assert!(
            system.uses_periodic_boundary_conditions()
                || !system
                    .forces()
                    .iter()
                    .any(|force| force.requires_periodic_system()),
            "a barostat cannot be used with a non-periodic system"
        );
        let masses = system.masses();
        let constraint_definitions = system.constraints().to_vec();
        for force in system.forces_mut() {
            force.initialize_masses(&masses);
            force.initialize_constraints(n, &constraint_definitions);
            force.initialize(n, periodic_box_vectors.as_ref());
            for (a, b) in force.bonded_particles() {
                assert!(a < n && b < n, "bonded particle index out of range");
            }
        }
        let virtual_sites = VirtualSitePlan::new(system.virtual_sites(), &masses);
        let constraints = ConstraintSolver::new(system.constraints(), &masses)
            .unwrap_or_else(|error| panic!("{error}"));
        integrator.initialize();
        let host_masses = masses;
        let masses = crate::tensor::from_floats::<B, 1>(host_masses.as_slice(), &device);
        let mut parameters = HashMap::new();
        for force in system.forces() {
            for (name, value) in force.global_parameters() {
                if let Some(previous) = parameters.insert(name.clone(), value) {
                    assert!(
                        previous == value,
                        "global parameter '{name}' has conflicting default values"
                    );
                }
            }
        }
        let positions = crate::tensor::from_f64_vec(vec![0.0; 3 * n], [n, 3], &device);
        let velocities = crate::tensor::from_f64_vec(vec![0.0; 3 * n], [n, 3], &device);
        Self {
            system,
            integrator,
            positions,
            velocities,
            periodic_box_vectors,
            parameters,
            time: 0.0,
            step_count: 0,
            device,
            constraints,
            masses,
            host_masses,
            virtual_sites,
            molecules: OnceLock::new(),
            empty_forces: OnceLock::new(),
            box_update_positions: None,
        }
    }

    pub fn set_positions(&mut self, pos: Tensor<B, 2>) {
        assert_eq!(
            pos.dims(),
            [self.system.num_particles(), 3],
            "positions must have shape [N, 3]"
        );
        self.positions = pos;
    }

    pub fn set_velocities(&mut self, vel: Tensor<B, 2>) {
        assert_eq!(
            vel.dims(),
            [self.system.num_particles(), 3],
            "velocities must have shape [N, 3]"
        );
        self.velocities = vel;
    }

    pub fn set_velocities_to_temperature(&mut self, temperature: f64) {
        assert!(
            temperature.is_finite() && temperature >= 0.0,
            "temperature must be nonnegative and finite"
        );
        let n = self.system.num_particles();
        let masses = self.system.masses();
        let scales: Vec<f64> = masses
            .iter()
            .map(|&mass| {
                if mass == 0.0 {
                    0.0
                } else {
                    (BOLTZMANN * temperature / mass).sqrt()
                }
            })
            .collect();
        let scale_tensor =
            crate::tensor::from_floats::<B, 1>(scales.as_slice(), &self.device).reshape([n, 1]);
        let noise = Tensor::<B, 2>::random(
            [n, 3],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &self.device,
        );
        self.velocities = noise * scale_tensor;

        let offset = self.integrator.velocity_time_offset();
        if offset != 0.0 {
            let (forces, _) = self.calc_forces_and_energy();
            let inv_masses: Vec<f64> = masses
                .iter()
                .map(|&mass| if mass == 0.0 { 0.0 } else { 1.0 / mass })
                .collect();
            let correction =
                crate::tensor::from_floats::<B, 1>(inv_masses.as_slice(), &self.device)
                    .reshape([n, 1])
                    * (offset);
            self.velocities = self.velocities.clone() - forces * correction;
        }
        self.apply_velocity_constraints(1e-5)
            .unwrap_or_else(|error| panic!("{error}"));
    }

    pub fn get_time(&self) -> f64 {
        self.time
    }

    pub fn set_time(&mut self, time: f64) {
        self.time = time;
    }

    pub fn get_step_count(&self) -> u64 {
        self.step_count
    }

    pub fn set_step_count(&mut self, step_count: u64) {
        self.step_count = step_count;
    }

    pub fn set_periodic_box_vectors(&mut self, vectors: BoxVectors) {
        validate_box_vectors(&vectors);
        self.periodic_box_vectors = Some(vectors);
    }

    pub fn periodic_box_vectors(&self) -> Option<&BoxVectors> {
        self.periodic_box_vectors.as_ref()
    }

    pub fn set_parameter(&mut self, name: &str, value: f64) {
        assert!(
            self.parameters.contains_key(name),
            "unknown global parameter '{name}'"
        );
        for force in self.system.forces_mut() {
            let _ = force.set_global_parameter(name, value);
        }
        self.parameters.insert(name.to_string(), value);
    }

    pub fn get_parameter(&self, name: &str) -> f64 {
        *self
            .parameters
            .get(name)
            .unwrap_or_else(|| panic!("unknown global parameter '{name}'"))
    }

    pub fn get_parameters(&self) -> &HashMap<String, f64> {
        &self.parameters
    }

    pub fn set_state(&mut self, state: &State<B>) {
        self.time = state.time;
        self.step_count = state.step_count;

        if let Some(parameters) = &state.parameters {
            for (name, &value) in parameters {
                self.set_parameter(name, value);
            }
        }
        if let Some(vectors) = state.periodic_box_vectors {
            self.set_periodic_box_vectors(vectors);
        }
        if let Some(positions) = &state.positions {
            self.positions = positions.clone();
        }
        if let Some(velocities) = &state.velocities {
            self.velocities = velocities.clone();
        }
    }

    pub fn update_rmsd_force_parameters(&mut self, force_index: usize, source: &RMSDForce) {
        let force = self
            .system
            .forces_mut()
            .get_mut(force_index)
            .unwrap_or_else(|| panic!("unknown force index {force_index}"));
        let target = force
            .as_any_mut()
            .downcast_mut::<RMSDForce>()
            .unwrap_or_else(|| panic!("force at index {force_index} is not a RMSDForce"));
        target.update_parameters_from(source);
    }

    pub fn update_orientation_restraint_force_parameters(
        &mut self,
        force_index: usize,
        source: &OrientationRestraintForce,
    ) {
        let force = self
            .system
            .forces_mut()
            .get_mut(force_index)
            .unwrap_or_else(|| panic!("unknown force index {force_index}"));
        let target = force
            .as_any_mut()
            .downcast_mut::<OrientationRestraintForce>()
            .unwrap_or_else(|| {
                panic!("force at index {force_index} is not a OrientationRestraintForce")
            });
        target.update_parameters_from(source);
    }

    pub fn collective_variable_values(&self, force_index: usize) -> Vec<f64> {
        let force = self
            .system
            .forces()
            .get(force_index)
            .expect("unknown force index");
        let force = force
            .as_any()
            .downcast_ref::<CustomCVForce<B>>()
            .expect("force is not a CustomCVForce");
        force.collective_variable_values(&self.positions, self.periodic_box_vectors.as_ref())
    }

    pub fn update_force_parameters_at_path(
        &mut self,
        path: &[usize],
        source: &dyn crate::Force<B>,
    ) -> Result<(), String> {
        let (&index, rest) = path
            .split_first()
            .ok_or_else(|| "force path must not be empty".to_owned())?;
        let mut target = self
            .system
            .forces_mut()
            .get_mut(index)
            .ok_or_else(|| "force index out of range".to_owned())?
            .as_mut();
        for &index in rest {
            let cv = target
                .as_any_mut()
                .downcast_mut::<CustomCVForce<B>>()
                .ok_or_else(|| "force path must traverse CustomCVForce objects".to_owned())?;
            if index >= cv.num_collective_variables() {
                return Err("collective variable index out of range".to_owned());
            }
            target = cv.collective_variable_mut(index);
        }
        update_force_parameters(target, source)
    }

    pub fn update_custom_external_force_parameters(
        &mut self,
        force_index: usize,
        source: &CustomExternalForce,
    ) {
        let force = self
            .system
            .forces_mut()
            .get_mut(force_index)
            .unwrap_or_else(|| panic!("unknown force index {force_index}"));
        let target = force
            .as_any_mut()
            .downcast_mut::<CustomExternalForce>()
            .unwrap_or_else(|| panic!("force at index {force_index} is not a CustomExternalForce"));
        target.update_parameters_from(source);
    }

    pub fn update_custom_bond_force_parameters(
        &mut self,
        force_index: usize,
        source: &CustomBondForce,
    ) {
        let force = self
            .system
            .forces_mut()
            .get_mut(force_index)
            .unwrap_or_else(|| panic!("unknown force index {force_index}"));
        let target = force
            .as_any_mut()
            .downcast_mut::<CustomBondForce>()
            .unwrap_or_else(|| panic!("force at index {force_index} is not a CustomBondForce"));
        target.update_parameters_from(source);
    }

    pub fn update_custom_compound_force_parameters(
        &mut self,
        force_index: usize,
        source: &CustomCompoundBondForce,
    ) {
        let force = self
            .system
            .forces_mut()
            .get_mut(force_index)
            .unwrap_or_else(|| panic!("unknown force index {force_index}"));
        let target = force
            .as_any_mut()
            .downcast_mut::<CustomCompoundBondForce>()
            .unwrap_or_else(|| {
                panic!("force at index {force_index} is not a CustomCompoundBondForce")
            });
        target.update_parameters_from(source);
    }

    pub fn update_custom_centroid_force_parameters(
        &mut self,
        force_index: usize,
        source: &CustomCentroidBondForce,
    ) {
        let force = self
            .system
            .forces_mut()
            .get_mut(force_index)
            .unwrap_or_else(|| panic!("unknown force index {force_index}"));
        let target = force
            .as_any_mut()
            .downcast_mut::<CustomCentroidBondForce>()
            .unwrap_or_else(|| {
                panic!("force at index {force_index} is not a CustomCentroidBondForce")
            });
        target.update_parameters_from(source);
    }

    pub fn update_custom_hbond_force_parameters(
        &mut self,
        force_index: usize,
        source: &CustomHbondForce,
    ) {
        let force = self
            .system
            .forces_mut()
            .get_mut(force_index)
            .unwrap_or_else(|| panic!("unknown force index {force_index}"));
        let target = force
            .as_any_mut()
            .downcast_mut::<CustomHbondForce>()
            .unwrap_or_else(|| panic!("force at index {force_index} is not a CustomHbondForce"));
        target.update_parameters_from(source);
    }

    pub fn update_custom_angle_force_parameters(
        &mut self,
        force_index: usize,
        source: &CustomAngleForce,
    ) {
        let force = self
            .system
            .forces_mut()
            .get_mut(force_index)
            .unwrap_or_else(|| panic!("unknown force index {force_index}"));
        let target = force
            .as_any_mut()
            .downcast_mut::<CustomAngleForce>()
            .unwrap_or_else(|| panic!("force at index {force_index} is not a CustomAngleForce"));
        target.update_parameters_from(source);
    }

    pub fn update_custom_nonbonded_force_parameters(
        &mut self,
        force_index: usize,
        source: &CustomNonbondedForce,
    ) {
        let force = self
            .system
            .forces_mut()
            .get_mut(force_index)
            .unwrap_or_else(|| panic!("unknown force index {force_index}"));
        let target = force
            .as_any_mut()
            .downcast_mut::<CustomNonbondedForce>()
            .unwrap_or_else(|| {
                panic!("force at index {force_index} is not a CustomNonbondedForce")
            });
        target.update_parameters_from(source);
    }

    pub fn update_custom_torsion_force_parameters(
        &mut self,
        force_index: usize,
        source: &CustomTorsionForce,
    ) {
        let force = self
            .system
            .forces_mut()
            .get_mut(force_index)
            .unwrap_or_else(|| panic!("unknown force index {force_index}"));
        let target = force
            .as_any_mut()
            .downcast_mut::<CustomTorsionForce>()
            .unwrap_or_else(|| panic!("force at index {force_index} is not a CustomTorsionForce"));
        target.update_parameters_from(source);
    }

    pub fn update_harmonic_bond_force_parameters(
        &mut self,
        force_index: usize,
        source: &HarmonicBondForce,
    ) {
        let force = self
            .system
            .forces_mut()
            .get_mut(force_index)
            .unwrap_or_else(|| panic!("unknown force index {force_index}"));
        let target = force
            .as_any_mut()
            .downcast_mut::<HarmonicBondForce>()
            .unwrap_or_else(|| panic!("force at index {force_index} is not a HarmonicBondForce"));
        target.update_parameters_from(source);
    }

    pub fn update_harmonic_angle_force_parameters(
        &mut self,
        force_index: usize,
        source: &HarmonicAngleForce,
    ) {
        let force = self
            .system
            .forces_mut()
            .get_mut(force_index)
            .unwrap_or_else(|| panic!("unknown force index {force_index}"));
        let target = force
            .as_any_mut()
            .downcast_mut::<HarmonicAngleForce>()
            .unwrap_or_else(|| panic!("force at index {force_index} is not a HarmonicAngleForce"));
        target.update_parameters_from(source);
    }

    pub fn update_cmap_torsion_force_parameters(
        &mut self,
        force_index: usize,
        source: &CMAPTorsionForce,
    ) {
        let force = self
            .system
            .forces_mut()
            .get_mut(force_index)
            .unwrap_or_else(|| panic!("unknown force index {force_index}"));
        let target = force
            .as_any_mut()
            .downcast_mut::<CMAPTorsionForce>()
            .unwrap_or_else(|| panic!("force at index {force_index} is not a CMAPTorsionForce"));
        target.update_parameters_from(source);
    }

    pub fn update_periodic_torsion_force_parameters(
        &mut self,
        force_index: usize,
        source: &PeriodicTorsionForce,
    ) {
        let force = self
            .system
            .forces_mut()
            .get_mut(force_index)
            .unwrap_or_else(|| panic!("unknown force index {force_index}"));
        let target = force
            .as_any_mut()
            .downcast_mut::<PeriodicTorsionForce>()
            .unwrap_or_else(|| {
                panic!("force at index {force_index} is not a PeriodicTorsionForce")
            });
        target.update_parameters_from(source);
    }

    pub fn update_nonbonded_force_parameters(
        &mut self,
        force_index: usize,
        source: &NonbondedForce,
    ) {
        let force = self
            .system
            .forces_mut()
            .get_mut(force_index)
            .unwrap_or_else(|| panic!("unknown force index {force_index}"));
        let target = force
            .as_any_mut()
            .downcast_mut::<NonbondedForce>()
            .unwrap_or_else(|| panic!("force at index {force_index} is not a NonbondedForce"));
        target.update_parameters_from(source);
    }

    pub fn update_rb_torsion_force_parameters(
        &mut self,
        force_index: usize,
        source: &RBTorsionForce,
    ) {
        let force = self
            .system
            .forces_mut()
            .get_mut(force_index)
            .unwrap_or_else(|| panic!("unknown force index {force_index}"));
        let target = force
            .as_any_mut()
            .downcast_mut::<RBTorsionForce>()
            .unwrap_or_else(|| panic!("force at index {force_index} is not a RBTorsionForce"));
        target.update_parameters_from(source);
    }

    /// Compute forces from all force terms.
    pub fn calc_forces_and_energy(&self) -> (Tensor<B, 2>, f64) {
        self.calc_forces_and_energy_groups(u32::MAX)
    }

    pub fn calc_forces_and_energy_groups(&self, groups: u32) -> (Tensor<B, 2>, f64) {
        let n = self.system.num_particles();
        let box_vecs = self
            .periodic_box_vectors
            .or_else(|| self.system.default_periodic_box_vectors().copied());
        let mut total_forces: Option<Tensor<B, 2>> = None;
        let mut total_energy = 0.0f64;

        for force in self.system.forces() {
            if let Some((f, e)) =
                force.calc_forces_and_energy_groups(&self.positions, box_vecs.as_ref(), groups)
            {
                total_forces = Some(match total_forces {
                    Some(total) => total + f,
                    None => f,
                });
                total_energy += e;
            }
        }

        let Some(mut total_forces) = total_forces else {
            return (
                self.empty_forces
                    .get_or_init(|| crate::tensor::zeros([n, 3], &self.device))
                    .clone(),
                total_energy,
            );
        };
        self.virtual_sites.distribute_forces(
            self.system.virtual_sites(),
            &self.positions,
            &mut total_forces,
            box_vecs.as_ref(),
        );
        (total_forces, total_energy)
    }

    pub fn apply_constraints(&mut self, tolerance: f64) -> Result<(), ConstraintError> {
        self.constraints.apply_positions(
            &self.positions.clone(),
            &mut self.positions,
            tolerance,
        )?;
        self.compute_virtual_sites();
        Ok(())
    }

    pub fn compute_virtual_sites(&mut self) {
        self.virtual_sites.compute_positions(
            self.system.virtual_sites(),
            &mut self.positions,
            self.periodic_box_vectors.as_ref(),
        );
    }

    pub fn apply_velocity_constraints(&mut self, tolerance: f64) -> Result<(), ConstraintError> {
        self.constraints
            .apply_velocities(&self.positions, &mut self.velocities, tolerance)
    }

    /// Advance the simulation, panicking if integration fails.
    pub fn step(&mut self, steps: usize) {
        self.try_step(steps)
            .unwrap_or_else(|error| panic!("{error}"));
    }

    pub fn try_step(&mut self, steps: usize) -> Result<(), ConstraintError> {
        for _ in 0..steps {
            self.try_step_once(f64::INFINITY)?;
        }
        Ok(())
    }

    pub fn step_to(&mut self, time: f64) {
        self.try_step_to(time)
            .unwrap_or_else(|error| panic!("{error}"));
    }

    pub fn try_step_to(&mut self, time: f64) -> Result<(), ConstraintError> {
        if !self.integrator.supports_step_to() {
            return Err(ConstraintError(
                "this integrator does not support step_to".into(),
            ));
        }
        if !time.is_finite() || !self.time.is_finite() {
            return Err(ConstraintError(
                "step_to requires finite current and target times".into(),
            ));
        }
        while time > self.time {
            let previous = self.time;
            let remaining = time - previous;
            self.try_step_once(remaining)?;
            if self.integrator.step_size() == remaining {
                self.time = time;
            }
            if self.time <= previous {
                return Err(ConstraintError(
                    "step size is too small to advance the clock".into(),
                ));
            }
        }
        Ok(())
    }

    fn try_step_once(&mut self, limit: f64) -> Result<(), ConstraintError> {
        for index in 0..self.system.forces().len() {
            self.system.forces_mut()[index].update_context_velocities(
                &mut self.velocities,
                &self.host_masses,
                self.integrator.step_size(),
                self.step_count,
            );
            if let Some(vectors) = self.periodic_box_vectors {
                let proposal = self.system.forces_mut()[index].propose_context_box_update(&vectors);
                if let Some(proposal) = proposal {
                    let groups = self.integrator.integration_force_groups();
                    let old_energy = self.energy_at(&self.positions, &vectors, groups);
                    let positions = self.prepare_scaled_positions(
                        proposal.transform,
                        proposal.scale_molecules_as_rigid,
                    );
                    let new_energy = self.energy_at(&positions, &proposal.box_vectors, groups);
                    let scaled_particles = if proposal.scale_molecules_as_rigid {
                        self.molecules().len()
                    } else {
                        self.system.num_particles()
                    };
                    if self.system.forces_mut()[index].accept_context_box_update(
                        &proposal,
                        new_energy - old_energy,
                        scaled_particles,
                    ) {
                        self.box_update_positions =
                            Some(std::mem::replace(&mut self.positions, positions));
                        self.periodic_box_vectors = Some(proposal.box_vectors);
                    } else {
                        self.box_update_positions = Some(positions);
                    }
                }
            }
        }
        let (forces, _) =
            self.calc_forces_and_energy_groups(self.integrator.integration_force_groups());
        self.integrator.step_with_limit(
            &mut self.positions,
            &mut self.velocities,
            &forces,
            &self.masses,
            &self.constraints,
            limit,
        )?;
        self.compute_virtual_sites();
        self.time += self.integrator.step_size();
        self.step_count += 1;
        Ok(())
    }

    pub fn get_state(&self, flags: StateFlags) -> State<B> {
        self.get_state_groups(flags, u32::MAX)
    }

    pub fn get_state_groups(&self, flags: StateFlags, groups: u32) -> State<B> {
        self.get_state_options(flags, groups, false)
    }

    pub fn molecules(&self) -> &[Vec<usize>] {
        self.molecules.get_or_init(|| self.system.molecules())
    }

    fn energy_at(&self, positions: &Tensor<B, 2>, vectors: &BoxVectors, groups: u32) -> f64 {
        self.system
            .forces()
            .iter()
            .filter_map(|force| {
                force
                    .calc_forces_and_energy_groups(positions, Some(vectors), groups)
                    .map(|(_, energy)| energy)
            })
            .sum()
    }

    fn scaled_positions(&self, transform: [[f64; 3]; 3], rigid: bool) -> Tensor<B, 2> {
        let mut positions = crate::tensor::to_f64_vec(&self.positions);
        self.transform_positions(&mut positions, transform, rigid);
        crate::tensor::from_f64_vec(positions, self.positions.dims(), &self.device)
    }

    fn prepare_scaled_positions(&mut self, transform: [[f64; 3]; 3], rigid: bool) -> Tensor<B, 2> {
        let Some(mut target) = self.box_update_positions.take() else {
            return self.scaled_positions(transform, rigid);
        };
        if crate::tensor::with_mut_ndarray_f64(&mut target, |values| {
            crate::tensor::with_f64_data(&self.positions, |source| values.copy_from_slice(source));
            self.transform_positions(values, transform, rigid);
        })
        .is_some()
        {
            target
        } else {
            self.scaled_positions(transform, rigid)
        }
    }

    fn transform_positions(
        &self,
        positions: &mut [f64],
        mut transform: [[f64; 3]; 3],
        rigid: bool,
    ) {
        if rigid {
            for (axis, row) in transform.iter_mut().enumerate() {
                row[axis] -= 1.0;
            }
        }
        let diagonal = transform[0][1] == 0.0
            && transform[0][2] == 0.0
            && transform[1][0] == 0.0
            && transform[1][2] == 0.0
            && transform[2][0] == 0.0
            && transform[2][1] == 0.0;
        let deform = |point: [f64; 3]| {
            if diagonal {
                [
                    transform[0][0] * point[0],
                    transform[1][1] * point[1],
                    transform[2][2] * point[2],
                ]
            } else {
                transform.map(|row| row.iter().zip(point).map(|(a, b)| a * b).sum::<f64>())
            }
        };
        if rigid {
            for molecule in self.molecules() {
                let mut center = [0.0; 3];
                for &atom in molecule {
                    for axis in 0..3 {
                        center[axis] += positions[3 * atom + axis];
                    }
                }
                let inverse_count = 1.0 / molecule.len() as f64;
                for value in &mut center {
                    *value *= inverse_count;
                }
                let offset = deform(center);
                for &atom in molecule {
                    for axis in 0..3 {
                        positions[3 * atom + axis] += offset[axis];
                    }
                }
            }
        } else {
            for position in positions.chunks_exact_mut(3) {
                let scaled = deform([position[0], position[1], position[2]]);
                position.copy_from_slice(&scaled);
            }
        }
    }

    pub fn get_state_options(
        &self,
        flags: StateFlags,
        groups: u32,
        enforce_periodic_box: bool,
    ) -> State<B> {
        let box_vecs = self
            .periodic_box_vectors
            .or_else(|| self.system.default_periodic_box_vectors().copied());
        let (forces, potential_energy) = if flags.forces || flags.energy {
            let (f, e) = self.calc_forces_and_energy_groups(groups);
            (Some(f), Some(e))
        } else {
            (None, None)
        };
        let energy_parameter_derivatives = if flags.parameter_derivatives {
            let mut total_derivatives = self
                .parameters
                .keys()
                .map(|name| (name.clone(), 0.0))
                .collect::<HashMap<_, _>>();
            for force in self.system.forces() {
                if groups & (1 << force.force_group()) == 0 {
                    continue;
                }
                for (name, derivative) in
                    force.energy_parameter_derivatives(&self.positions, box_vecs.as_ref())
                {
                    *total_derivatives.entry(name).or_insert(0.0) += derivative;
                }
            }
            Some(total_derivatives)
        } else {
            None
        };

        let kinetic_energy = flags
            .energy
            .then(|| self.kinetic_energy(forces.as_ref().unwrap()));

        State {
            time: self.time,
            step_count: self.step_count,
            parameters: flags.parameters.then(|| self.parameters.clone()),
            energy_parameter_derivatives,
            periodic_box_vectors: self.periodic_box_vectors,
            positions: flags.positions.then(|| {
                if enforce_periodic_box {
                    self.wrapped_positions()
                } else {
                    self.positions.clone()
                }
            }),
            velocities: flags.velocities.then(|| self.velocities.clone()),
            forces: if flags.forces { forces } else { None },
            potential_energy: if flags.energy { potential_energy } else { None },
            kinetic_energy,
        }
    }

    fn kinetic_energy(&self, forces: &Tensor<B, 2>) -> f64 {
        let offset = self.integrator.kinetic_energy_time_offset();
        if self.constraints.is_empty() {
            let kinetic = crate::tensor::with_ndarray_f64(&self.velocities, |velocities| {
                if offset == 0.0 {
                    Some(
                        0.5 * velocities
                            .chunks_exact(3)
                            .zip(&self.host_masses)
                            .map(|(v, &mass)| mass * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
                            .sum::<f64>(),
                    )
                } else {
                    crate::tensor::with_ndarray_f64(forces, |forces| {
                        let mut total = 0.0;
                        for ((v, f), &mass) in velocities
                            .chunks_exact(3)
                            .zip(forces.chunks_exact(3))
                            .zip(&self.host_masses)
                        {
                            if mass == 0.0 {
                                continue;
                            }
                            let inverse_mass = 1.0 / mass;
                            let vx = v[0] + f[0] * inverse_mass * offset;
                            let vy = v[1] + f[1] * inverse_mass * offset;
                            let vz = v[2] + f[2] * inverse_mass * offset;
                            total += mass * (vx * vx + vy * vy + vz * vz);
                        }
                        0.5 * total
                    })
                }
            })
            .flatten();
            if let Some(kinetic) = kinetic {
                return kinetic;
            }
        }
        let n = self.system.num_particles();
        let mut velocities = if offset == 0.0 {
            self.velocities.clone()
        } else {
            self.velocities.clone()
                + forces.clone() * crate::integrators::inverse_masses(&self.masses) * offset
        };
        if offset != 0.0 {
            self.constraints
                .apply_velocities(
                    &self.positions,
                    &mut velocities,
                    self.integrator.constraint_tolerance(),
                )
                .unwrap_or_else(|error| panic!("{error}"));
        }
        let v2 = velocities.clone() * velocities;
        0.5 * crate::tensor::scalar((self.masses.clone().reshape([n, 1]) * v2).sum())
    }

    fn wrapped_positions(&self) -> Tensor<B, 2> {
        let Some(vectors) = self.periodic_box_vectors else {
            return self.positions.clone();
        };
        let mut positions = self
            .positions
            .to_data()
            .convert::<f64>()
            .to_vec::<f64>()
            .unwrap();
        for molecule in self.molecules() {
            let mut center = [0.0; 3];
            for &particle in molecule {
                for axis in 0..3 {
                    center[axis] += positions[3 * particle + axis];
                }
            }
            for c in &mut center {
                *c /= molecule.len() as f64;
            }
            let mut shift = [0.0; 3];
            for axis in (0..3).rev() {
                let cells = ((center[axis] - shift[axis]) / vectors[axis][axis]).floor();
                for (d, s) in shift.iter_mut().enumerate() {
                    *s += cells * vectors[axis][d];
                }
            }
            for &particle in molecule {
                for axis in 0..3 {
                    positions[3 * particle + axis] -= shift[axis];
                }
            }
        }
        crate::tensor::from_floats::<B, 1>(positions.as_slice(), &self.device)
            .reshape([self.system.num_particles(), 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forces::custom_bond::CustomBondForce;
    use crate::forces::custom_external::CustomExternalForce;
    use crate::forces::custom_nonbonded::CustomNonbondedForce;
    use crate::forces::harmonic_bond::HarmonicBondForce;
    use crate::forces::nonbonded::{NonbondedForce, NonbondedMethod, ONE_4PI_EPS0};
    use crate::forces::rb_torsion::{RBTorsionForce, RBTorsionParam};
    use crate::system::System;
    use crate::VerletIntegrator;
    use approx::assert_relative_eq;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn test_set_velocities_to_temperature() {
        let mut system = System::<B>::new();
        for _ in 0..64 {
            system.add_particle(12.0);
        }

        let integrator = Box::new(VerletIntegrator::new(0.001));
        let device: <B as Backend>::Device = Default::default();
        let mut context = Context::new(system, integrator, device);
        context.set_velocities_to_temperature(300.0);

        let state = context.get_state(StateFlags {
            positions: false,
            velocities: false,
            forces: false,
            energy: true,
            parameters: false,
            parameter_derivatives: false,
        });
        let kinetic_energy = state.kinetic_energy.unwrap();
        let expected = 0.5 * (64 * 3) as f64 * BOLTZMANN * 300.0;
        assert_relative_eq!(kinetic_energy, expected, epsilon = expected * 0.3);
    }

    #[test]
    fn test_time_and_step_count() {
        let mut system = System::<B>::new();
        system.add_particle(1.0);

        let integrator = Box::new(VerletIntegrator::new(0.002));
        let device: <B as Backend>::Device = Default::default();
        let mut context = Context::new(system, integrator, device);
        context.step(5);

        assert_relative_eq!(context.get_time(), 0.01, epsilon = 1e-12);
        assert_eq!(context.get_step_count(), 5);

        let state = context.get_state(StateFlags {
            positions: false,
            velocities: false,
            forces: false,
            energy: false,
            parameters: false,
            parameter_derivatives: false,
        });
        assert_relative_eq!(state.time, 0.01, epsilon = 1e-12);
        assert_eq!(state.step_count, 5);
    }

    #[test]
    fn test_runtime_periodic_box_vectors_override_system_default() {
        let mut system = System::<B>::new();
        system.add_particle(1.0);
        system.add_particle(1.0);
        system.set_default_periodic_box_vectors([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ]);

        let mut nonbonded = NonbondedForce::new();
        nonbonded.set_nonbonded_method(NonbondedMethod::CutoffPeriodic);
        nonbonded.set_cutoff_distance(1.0);
        nonbonded.add_particle(1.0, 1.0, 0.0);
        nonbonded.add_particle(-1.0, 1.0, 0.0);
        system.add_force(Box::new(nonbonded));

        let integrator = Box::new(VerletIntegrator::new(0.001));
        let device: <B as Backend>::Device = Default::default();
        let mut context = Context::new(system, integrator, device);
        let positions =
            Tensor::<B, 1>::from_floats([0.0_f32, 0.0, 0.0, 1.6, 0.0, 0.0], &Default::default())
                .reshape([2, 3]);
        context.set_positions(positions);

        let (_, default_energy) = context.calc_forces_and_energy();
        context.set_periodic_box_vectors([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]);
        let (_, overridden_energy) = context.calc_forces_and_energy();

        assert!(default_energy.abs() > overridden_energy.abs());

        let state = context.get_state(StateFlags {
            positions: false,
            velocities: false,
            forces: false,
            energy: false,
            parameters: false,
            parameter_derivatives: false,
        });
        assert_eq!(
            state.periodic_box_vectors,
            Some([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
        );
    }

    #[test]
    fn test_set_state_restores_snapshot() {
        let mut system = System::<B>::new();
        system.add_particle(1.0);
        system.set_default_periodic_box_vectors([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ]);

        let integrator = Box::new(VerletIntegrator::new(0.001));
        let device: <B as Backend>::Device = Default::default();
        let mut context = Context::new(system, integrator, device);
        context.set_time(1.25);
        context.set_step_count(17);
        context.set_periodic_box_vectors([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]);
        context.set_positions(
            Tensor::<B, 1>::from_floats([1.0_f32, 2.0, 3.0], &device).reshape([1, 3]),
        );
        context.set_velocities(
            Tensor::<B, 1>::from_floats([0.1_f32, 0.2, 0.3], &device).reshape([1, 3]),
        );

        let snapshot = context.get_state(StateFlags::all());

        let mut system2 = System::<B>::new();
        system2.add_particle(1.0);
        let integrator2 = Box::new(VerletIntegrator::new(0.001));
        let mut restored = Context::new(system2, integrator2, device);
        restored.set_state(&snapshot);

        let state = restored.get_state(StateFlags::all());
        assert_relative_eq!(state.time, 1.25, epsilon = 1e-12);
        assert_eq!(state.step_count, 17);
        assert_eq!(
            state.periodic_box_vectors,
            Some([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
        );

        let pos: Vec<f32> = state
            .positions
            .as_ref()
            .unwrap()
            .clone()
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        let vel: Vec<f32> = state
            .velocities
            .as_ref()
            .unwrap()
            .clone()
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        assert_relative_eq!(pos[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(pos[1], 2.0, epsilon = 1e-6);
        assert_relative_eq!(pos[2], 3.0, epsilon = 1e-6);
        assert_relative_eq!(vel[0], 0.1, epsilon = 1e-6);
        assert_relative_eq!(vel[1], 0.2, epsilon = 1e-6);
        assert_relative_eq!(vel[2], 0.3, epsilon = 1e-6);
    }

    #[test]
    fn test_set_state_preserves_missing_fields() {
        let mut system = System::<B>::new();
        system.add_particle(1.0);

        let integrator = Box::new(VerletIntegrator::new(0.001));
        let device: <B as Backend>::Device = Default::default();
        let mut context = Context::new(system, integrator, device);
        context.set_velocities(
            Tensor::<B, 1>::from_floats([0.4_f32, 0.5, 0.6], &device).reshape([1, 3]),
        );

        let positions_only = State {
            time: 2.0,
            step_count: 9,
            parameters: None,
            energy_parameter_derivatives: None,
            periodic_box_vectors: None,
            positions: Some(
                Tensor::<B, 1>::from_floats([7.0_f32, 8.0, 9.0], &device).reshape([1, 3]),
            ),
            velocities: None,
            forces: None,
            potential_energy: None,
            kinetic_energy: None,
        };
        context.set_state(&positions_only);

        let state = context.get_state(StateFlags {
            positions: true,
            velocities: true,
            forces: false,
            energy: false,
            parameters: false,
            parameter_derivatives: false,
        });
        assert_relative_eq!(state.time, 2.0, epsilon = 1e-12);
        assert_eq!(state.step_count, 9);

        let pos: Vec<f32> = state
            .positions
            .as_ref()
            .unwrap()
            .clone()
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        let vel: Vec<f32> = state
            .velocities
            .as_ref()
            .unwrap()
            .clone()
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        assert_relative_eq!(pos[0], 7.0, epsilon = 1e-6);
        assert_relative_eq!(pos[1], 8.0, epsilon = 1e-6);
        assert_relative_eq!(pos[2], 9.0, epsilon = 1e-6);
        assert_relative_eq!(vel[0], 0.4, epsilon = 1e-6);
        assert_relative_eq!(vel[1], 0.5, epsilon = 1e-6);
        assert_relative_eq!(vel[2], 0.6, epsilon = 1e-6);
    }

    #[test]
    fn test_context_global_parameters_round_trip() {
        let mut system = System::<B>::new();
        system.add_particle(1.0);
        system.add_particle(1.0);

        let mut force = CustomBondForce::new("scale*k*(r-r0)^2").unwrap();
        force.add_global_parameter("scale", 1.0);
        force.add_energy_parameter_derivative("scale");
        force.add_per_bond_parameter("r0");
        force.add_per_bond_parameter("k");
        force.add_bond(0, 1, vec![1.0, 2.0]);
        system.add_force(Box::new(force));

        let integrator = Box::new(VerletIntegrator::new(0.001));
        let device: <B as Backend>::Device = Default::default();
        let mut context = Context::new(system, integrator, device);
        context.set_positions(
            Tensor::<B, 1>::from_floats([0.0_f32, 0.0, 0.0, 1.5, 0.0, 0.0], &Default::default())
                .reshape([2, 3]),
        );

        let (_, baseline_energy) = context.calc_forces_and_energy();
        context.set_parameter("scale", 3.0);
        let (_, scaled_energy) = context.calc_forces_and_energy();
        assert_relative_eq!(baseline_energy, 0.5, epsilon = 1e-6);
        assert_relative_eq!(scaled_energy, 1.5, epsilon = 1e-6);
        assert_relative_eq!(context.get_parameter("scale"), 3.0, epsilon = 1e-12);

        let snapshot = context.get_state(StateFlags {
            positions: false,
            velocities: false,
            forces: false,
            energy: false,
            parameters: true,
            parameter_derivatives: true,
        });
        assert_eq!(
            snapshot
                .parameters
                .as_ref()
                .and_then(|parameters| parameters.get("scale"))
                .copied(),
            Some(3.0)
        );
        let derivative = snapshot
            .energy_parameter_derivatives
            .as_ref()
            .and_then(|derivatives| derivatives.get("scale"))
            .copied()
            .unwrap();
        assert_relative_eq!(derivative, 0.5, epsilon = 1e-5);

        let mut system2 = System::<B>::new();
        system2.add_particle(1.0);
        system2.add_particle(1.0);
        let mut force2 = CustomBondForce::new("scale*k*(r-r0)^2").unwrap();
        force2.add_global_parameter("scale", 1.0);
        force2.add_energy_parameter_derivative("scale");
        force2.add_per_bond_parameter("r0");
        force2.add_per_bond_parameter("k");
        force2.add_bond(0, 1, vec![1.0, 2.0]);
        system2.add_force(Box::new(force2));

        let integrator2 = Box::new(VerletIntegrator::new(0.001));
        let mut restored = Context::new(system2, integrator2, Default::default());
        restored.set_state(&snapshot);
        assert_relative_eq!(restored.get_parameter("scale"), 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_update_custom_external_force_parameters() {
        let mut system = System::<B>::new();
        system.add_particle(1.0);

        let mut force = CustomExternalForce::new("k*(x-x0)^2").unwrap();
        force.add_per_particle_parameter("x0");
        force.add_per_particle_parameter("k");
        force.add_particle(0, vec![0.0, 1.0]);
        let force_index = system.add_force(Box::new(force));

        let integrator = Box::new(VerletIntegrator::new(0.001));
        let device: <B as Backend>::Device = Default::default();
        let mut context = Context::new(system, integrator, device);
        context.set_positions(
            Tensor::<B, 1>::from_floats([1.5_f32, 0.0, 0.0], &Default::default()).reshape([1, 3]),
        );

        let (_, initial_energy) = context.calc_forces_and_energy();

        let mut updated = CustomExternalForce::new("k*(x-x0)^2").unwrap();
        updated.add_per_particle_parameter("x0");
        updated.add_per_particle_parameter("k");
        updated.add_particle(0, vec![1.0, 3.0]);
        context.update_custom_external_force_parameters(force_index, &updated);

        let (_, updated_energy) = context.calc_forces_and_energy();
        assert_relative_eq!(initial_energy, 2.25, epsilon = 1e-6);
        assert_relative_eq!(updated_energy, 0.75, epsilon = 1e-6);
    }

    #[test]
    fn test_update_custom_nonbonded_force_parameters_preserves_context_globals() {
        let mut system = System::<B>::new();
        system.add_particle(1.0);
        system.add_particle(1.0);

        let mut force = CustomNonbondedForce::new("scale*q1*q2/r").unwrap();
        force.add_global_parameter("scale", 1.0);
        force.add_per_particle_parameter("q");
        force.add_particle(vec![1.0]);
        force.add_particle(vec![-1.0]);
        let force_index = system.add_force(Box::new(force));

        let integrator = Box::new(VerletIntegrator::new(0.001));
        let device: <B as Backend>::Device = Default::default();
        let mut context = Context::new(system, integrator, device);
        context.set_positions(
            Tensor::<B, 1>::from_floats([0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0], &Default::default())
                .reshape([2, 3]),
        );
        context.set_parameter("scale", 2.0);

        let (_, initial_energy) = context.calc_forces_and_energy();

        let mut updated = CustomNonbondedForce::new("scale*q1*q2/r").unwrap();
        updated.add_global_parameter("scale", 7.0);
        updated.add_per_particle_parameter("q");
        updated.add_particle(vec![2.0]);
        updated.add_particle(vec![-1.0]);
        context.update_custom_nonbonded_force_parameters(force_index, &updated);

        let (_, updated_energy) = context.calc_forces_and_energy();
        assert_relative_eq!(initial_energy, -2.0, epsilon = 1e-6);
        assert_relative_eq!(updated_energy, -4.0, epsilon = 1e-6);
        assert_relative_eq!(context.get_parameter("scale"), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_update_harmonic_bond_force_parameters() {
        let mut system = System::<B>::new();
        system.add_particle(1.0);
        system.add_particle(1.0);

        let mut force = HarmonicBondForce::new();
        force.add_bond(0, 1, 1.0, 2.0);
        let force_index = system.add_force(Box::new(force));

        let integrator = Box::new(VerletIntegrator::new(0.001));
        let device: <B as Backend>::Device = Default::default();
        let mut context = Context::new(system, integrator, device);
        context.set_positions(
            Tensor::<B, 1>::from_floats([0.0_f32, 0.0, 0.0, 1.5, 0.0, 0.0], &Default::default())
                .reshape([2, 3]),
        );

        let (_, initial_energy) = context.calc_forces_and_energy();

        let mut updated = HarmonicBondForce::new();
        updated.add_bond(0, 1, 1.25, 4.0);
        context.update_harmonic_bond_force_parameters(force_index, &updated);

        let (_, updated_energy) = context.calc_forces_and_energy();
        assert_relative_eq!(initial_energy, 0.25, epsilon = 1e-6);
        assert_relative_eq!(updated_energy, 0.125, epsilon = 1e-6);
    }

    #[test]
    fn test_update_nonbonded_force_parameters() {
        let mut system = System::<B>::new();
        system.add_particle(1.0);
        system.add_particle(1.0);

        let mut force = NonbondedForce::new();
        force.add_particle(1.0, 1.0, 0.0);
        force.add_particle(-1.0, 1.0, 0.0);
        let force_index = system.add_force(Box::new(force));

        let integrator = Box::new(VerletIntegrator::new(0.001));
        let device: <B as Backend>::Device = Default::default();
        let mut context = Context::new(system, integrator, device);
        context.set_positions(
            Tensor::<B, 1>::from_floats([0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0], &Default::default())
                .reshape([2, 3]),
        );

        let (_, initial_energy) = context.calc_forces_and_energy();

        let mut updated = NonbondedForce::new();
        updated.add_particle(2.0, 1.0, 0.0);
        updated.add_particle(-1.0, 1.0, 0.0);
        context.update_nonbonded_force_parameters(force_index, &updated);

        let (_, updated_energy) = context.calc_forces_and_energy();
        assert_relative_eq!(initial_energy, -ONE_4PI_EPS0, epsilon = 1e-6);
        assert_relative_eq!(updated_energy, -2.0 * ONE_4PI_EPS0, epsilon = 1e-6);
    }

    #[test]
    fn test_update_rb_torsion_force_parameters() {
        let mut system = System::<B>::new();
        for _ in 0..4 {
            system.add_particle(1.0);
        }

        let mut force = RBTorsionForce::new();
        force.add_torsion(RBTorsionParam::new(
            [0, 1, 2, 3],
            [1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        ));
        let force_index = system.add_force(Box::new(force));

        let integrator = Box::new(VerletIntegrator::new(0.001));
        let device: <B as Backend>::Device = Default::default();
        let mut context = Context::new(system, integrator, device);
        context.set_positions(
            Tensor::<B, 1>::from_floats(
                [
                    0.0_f32, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                ],
                &Default::default(),
            )
            .reshape([4, 3]),
        );

        let (_, initial_energy) = context.calc_forces_and_energy();

        let mut updated = RBTorsionForce::new();
        updated.add_torsion(RBTorsionParam::new(
            [0, 1, 2, 3],
            [2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ));
        context.update_rb_torsion_force_parameters(force_index, &updated);

        let (_, updated_energy) = context.calc_forces_and_energy();
        assert!(updated_energy > initial_energy);
    }
}
