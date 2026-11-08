use super::{
    harmonic_bond::apply_minimum_image_ortho,
    neighbors::NeighborCells,
    nonbonded::ONE_4PI_EPS0,
    pme::{reciprocal_box, PmeCache},
    Force, HostForceEvaluation,
};
use burn::prelude::*;
use std::collections::{BTreeSet, HashSet, VecDeque};
use std::f64::consts::PI;
use std::sync::{Mutex, MutexGuard};

mod solver;

const NONE: usize = usize::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConstantPotentialMethod {
    #[default]
    CG,
    Matrix,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConstantPotentialElectrode {
    pub particles: BTreeSet<usize>,
    pub potential: f64,
    pub gaussian_width: f64,
    pub thomas_fermi_scale: f64,
}

#[derive(Clone, Debug)]
pub struct ConstantPotentialException {
    pub particle1: usize,
    pub particle2: usize,
    pub charge_prod: f64,
}

#[derive(Clone)]
struct Setup {
    system_to_electrode: Vec<usize>,
    electrode_to_system: Vec<usize>,
    parameters: Vec<[f64; 3]>,
    exclusions: HashSet<(usize, usize)>,
    nonzero_exceptions: Vec<usize>,
    pme: (f64, [usize; 3]),
    target: f64,
}

#[derive(Default)]
struct Work {
    charges: Vec<f64>,
    pairs: Vec<(usize, usize, f64, f64)>,
    background_energy: f64,
    forces: Vec<f64>,
    matrix: Vec<f64>,
    constraint: Vec<f64>,
    preconditioner: Vec<f64>,
    electrode_positions: Vec<f64>,
    box_vectors: Option<[[f64; 3]; 3]>,
    solver_valid: bool,
    q: Vec<f64>,
    gradient: Vec<f64>,
    projected: Vec<f64>,
    preconditioned: Vec<f64>,
    step: Vec<f64>,
    product: Vec<f64>,
    baseline: Vec<f64>,
}

#[derive(Default)]
struct Workspace(Mutex<Work>);
impl Clone for Workspace {
    fn clone(&self) -> Self {
        Self::default()
    }
}

#[derive(Clone)]
pub struct ConstantPotentialForce {
    metadata: super::ForceMetadata,
    particles: Vec<f64>,
    exceptions: Vec<ConstantPotentialException>,
    electrodes: Vec<ConstantPotentialElectrode>,
    method: ConstantPotentialMethod,
    cutoff: f64,
    ewald_tolerance: f64,
    pme_parameters: (f64, [usize; 3]),
    cg_tolerance: f64,
    preconditioner: bool,
    charge_constraint: bool,
    charge_target: f64,
    external_field: [f64; 3],
    periodic_exceptions: bool,
    masses: Vec<f64>,
    setup: Option<Setup>,
    neighbors: NeighborCells,
    pme: PmeCache,
    workspace: Workspace,
}

impl Default for ConstantPotentialForce {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstantPotentialForce {
    super::force_metadata!();

    pub fn new() -> Self {
        Self {
            metadata: super::ForceMetadata::new("ConstantPotentialForce"),
            particles: Vec::new(),
            exceptions: Vec::new(),
            electrodes: Vec::new(),
            method: ConstantPotentialMethod::CG,
            cutoff: 1.,
            ewald_tolerance: 5e-4,
            pme_parameters: (0., [0; 3]),
            cg_tolerance: 1e-4,
            preconditioner: true,
            charge_constraint: false,
            charge_target: 0.,
            external_field: [0.; 3],
            periodic_exceptions: false,
            masses: Vec::new(),
            setup: None,
            neighbors: NeighborCells::default(),
            pme: PmeCache::default(),
            workspace: Workspace::default(),
        }
    }

    pub fn add_particle(&mut self, charge: f64) -> usize {
        self.particles.push(charge);
        self.particles.len() - 1
    }
    pub fn num_particles(&self) -> usize {
        self.particles.len()
    }
    pub fn particle_parameters(&self, index: usize) -> f64 {
        self.particles[index]
    }
    pub fn set_particle_parameters(&mut self, index: usize, charge: f64) {
        self.particles[index] = charge;
    }
    pub fn add_exception(
        &mut self,
        particle1: usize,
        particle2: usize,
        charge_prod: f64,
        replace: bool,
    ) -> usize {
        let pair = ordered(particle1, particle2);
        if let Some(index) = self
            .exceptions
            .iter()
            .position(|e| ordered(e.particle1, e.particle2) == pair)
        {
            assert!(replace, "an exception already exists for this pair");
            self.set_exception_parameters(index, particle1, particle2, charge_prod);
            return index;
        }
        self.exceptions.push(ConstantPotentialException {
            particle1,
            particle2,
            charge_prod,
        });
        self.exceptions.len() - 1
    }
    pub fn num_exceptions(&self) -> usize {
        self.exceptions.len()
    }
    pub fn exception_parameters(&self, index: usize) -> &ConstantPotentialException {
        &self.exceptions[index]
    }
    pub fn set_exception_parameters(
        &mut self,
        index: usize,
        particle1: usize,
        particle2: usize,
        charge_prod: f64,
    ) {
        self.exceptions[index] = ConstantPotentialException {
            particle1,
            particle2,
            charge_prod,
        };
    }
    pub fn create_exceptions_from_bonds(&mut self, bonds: &[(usize, usize)], scale: f64) {
        let n = self.particles.len();
        let mut neighbors = vec![Vec::new(); n];
        for &(a, b) in bonds {
            assert!(a < n && b < n, "bond index out of range");
            neighbors[a].push(b);
            neighbors[b].push(a);
        }
        let mut distance = vec![usize::MAX; n];
        let mut queue = VecDeque::new();
        for a in 0..n {
            distance.fill(usize::MAX);
            distance[a] = 0;
            queue.push_back(a);
            while let Some(i) = queue.pop_front() {
                if distance[i] == 3 {
                    continue;
                }
                for &j in &neighbors[i] {
                    if distance[j] != usize::MAX {
                        continue;
                    }
                    distance[j] = distance[i] + 1;
                    queue.push_back(j);
                }
            }
            for (b, &length) in distance[..a].iter().enumerate() {
                if length <= 3 {
                    let product = if length == 3 {
                        scale * self.particles[a] * self.particles[b]
                    } else {
                        0.
                    };
                    self.add_exception(b, a, product, false);
                }
            }
        }
    }
    pub fn add_electrode(
        &mut self,
        particles: impl IntoIterator<Item = usize>,
        potential: f64,
        gaussian_width: f64,
        thomas_fermi_scale: f64,
    ) -> usize {
        self.electrodes.push(ConstantPotentialElectrode {
            particles: particles.into_iter().collect(),
            potential,
            gaussian_width,
            thomas_fermi_scale,
        });
        self.electrodes.len() - 1
    }
    pub fn num_electrodes(&self) -> usize {
        self.electrodes.len()
    }
    pub fn electrode_parameters(&self, index: usize) -> &ConstantPotentialElectrode {
        &self.electrodes[index]
    }
    pub fn set_electrode_parameters(
        &mut self,
        index: usize,
        particles: impl IntoIterator<Item = usize>,
        potential: f64,
        gaussian_width: f64,
        thomas_fermi_scale: f64,
    ) {
        self.electrodes[index] = ConstantPotentialElectrode {
            particles: particles.into_iter().collect(),
            potential,
            gaussian_width,
            thomas_fermi_scale,
        };
    }
    pub fn constant_potential_method(&self) -> ConstantPotentialMethod {
        self.method
    }
    pub fn set_constant_potential_method(&mut self, value: ConstantPotentialMethod) {
        self.method = value;
    }
    pub fn cutoff_distance(&self) -> f64 {
        self.cutoff
    }
    pub fn set_cutoff_distance(&mut self, value: f64) {
        self.cutoff = value;
    }
    pub fn ewald_error_tolerance(&self) -> f64 {
        self.ewald_tolerance
    }
    pub fn set_ewald_error_tolerance(&mut self, value: f64) {
        self.ewald_tolerance = value;
    }
    pub fn cg_error_tolerance(&self) -> f64 {
        self.cg_tolerance
    }
    pub fn set_cg_error_tolerance(&mut self, value: f64) {
        self.cg_tolerance = value;
    }
    pub fn use_preconditioner(&self) -> bool {
        self.preconditioner
    }
    pub fn set_use_preconditioner(&mut self, value: bool) {
        self.preconditioner = value;
    }
    pub fn use_charge_constraint(&self) -> bool {
        self.charge_constraint
    }
    pub fn set_use_charge_constraint(&mut self, value: bool) {
        self.charge_constraint = value;
    }
    pub fn charge_constraint_target(&self) -> f64 {
        self.charge_target
    }
    pub fn set_charge_constraint_target(&mut self, value: f64) {
        self.charge_target = value;
    }
    pub fn external_field(&self) -> [f64; 3] {
        self.external_field
    }
    pub fn set_external_field(&mut self, value: [f64; 3]) {
        self.external_field = value;
    }
    pub fn exceptions_use_periodic_boundary_conditions(&self) -> bool {
        self.periodic_exceptions
    }
    pub fn set_exceptions_use_periodic_boundary_conditions(&mut self, value: bool) {
        self.periodic_exceptions = value;
    }

    pub fn pme_parameters(&self) -> (f64, [usize; 3]) {
        self.pme_parameters
    }
    pub fn set_pme_parameters(&mut self, alpha: f64, dimensions: [usize; 3]) {
        self.pme_parameters = (alpha, dimensions);
    }
    pub fn pme_parameters_in_context(&self) -> (f64, [usize; 3]) {
        self.setup.as_ref().expect("force must be initialized").pme
    }
    pub fn uses_periodic_boundary_conditions(&self) -> bool {
        true
    }

    fn initialize_context(&mut self, n: usize, vectors: Option<&[[f64; 3]; 3]>) {
        assert_eq!(
            n,
            self.particles.len(),
            "ConstantPotentialForce requires one charge per particle"
        );
        let vectors = vectors.expect("ConstantPotentialForce requires a periodic box");
        assert!(
            self.cutoff > 0. && self.cutoff.is_finite(),
            "cutoff must be positive and finite"
        );
        for (axis, vector) in vectors.iter().enumerate() {
            assert!(
                self.cutoff <= 0.5 * vector[axis],
                "cutoff exceeds half the periodic box"
            );
        }
        let mut system_to_electrode = vec![NONE; n];
        let mut electrode_to_system = Vec::new();
        let mut parameters = Vec::new();
        for electrode in &self.electrodes {
            assert!(
                electrode.gaussian_width >= 0. && electrode.thomas_fermi_scale >= 0.,
                "electrode width and Thomas-Fermi scale must be nonnegative"
            );
            for &i in &electrode.particles {
                assert!(i < n, "electrode particle index out of range");
                assert_eq!(system_to_electrode[i], NONE, "electrodes must not overlap");
                system_to_electrode[i] = parameters.len();
                electrode_to_system.push(i);
                parameters.push([
                    electrode.potential,
                    electrode.gaussian_width,
                    electrode.thomas_fermi_scale,
                ]);
            }
        }
        let mut exclusions = HashSet::new();
        let mut nonzero_exceptions = Vec::new();
        for (index, exception) in self.exceptions.iter().enumerate() {
            let (a, b) = ordered(exception.particle1, exception.particle2);
            assert!(a < n && b < n, "exception particle index out of range");
            assert!(exclusions.insert((a, b)), "duplicate exception pair");
            assert!(
                system_to_electrode[a] == NONE && system_to_electrode[b] == NONE,
                "electrode particles cannot belong to exceptions"
            );
            if exception.charge_prod != 0. {
                nonzero_exceptions.push(index);
            }
        }
        assert!(
            !self.charge_constraint || !electrode_to_system.is_empty(),
            "charge constraints require an electrode particle"
        );
        if self.method == ConstantPotentialMethod::Matrix && !electrode_to_system.is_empty() {
            assert_eq!(
                self.masses.len(),
                n,
                "matrix solver requires initialized masses"
            );
            assert!(
                electrode_to_system.iter().all(|&i| self.masses[i] == 0.),
                "matrix electrode particles must have zero mass"
            );
        }
        let pme = if self.pme_parameters.0 == 0. {
            assert!(
                self.ewald_tolerance > 0. && self.ewald_tolerance < 0.5,
                "invalid Ewald tolerance"
            );
            let alpha = (-f64::ln(2. * self.ewald_tolerance)).sqrt() / self.cutoff;
            let grid = std::array::from_fn(|axis| {
                (2. * alpha * vectors[axis][axis] / (3. * self.ewald_tolerance.powf(0.2)))
                    .ceil()
                    .max(6.) as usize
            });
            (alpha, grid)
        } else {
            self.pme_parameters
        };
        assert!(
            pme.0 > 0. && pme.0.is_finite() && pme.1.iter().all(|&n| n >= 5),
            "invalid PME parameters"
        );
        assert!(
            self.cg_tolerance >= 0. && self.cg_tolerance.is_finite(),
            "invalid CG tolerance"
        );
        let target = self.charge_target
            - self
                .particles
                .iter()
                .enumerate()
                .filter(|(i, _)| system_to_electrode[*i] == NONE)
                .map(|(_, q)| q)
                .sum::<f64>();
        self.setup = Some(Setup {
            system_to_electrode,
            electrode_to_system,
            parameters,
            exclusions,
            nonzero_exceptions,
            pme,
            target,
        });
        self.workspace = Workspace::default();
        self.masses = Vec::new();
    }

    pub fn update_parameters_from(&mut self, source: &Self) {
        assert_eq!(
            self.particles.len(),
            source.particles.len(),
            "particle count cannot change"
        );
        let setup = self.setup.as_mut().expect("force must be initialized");
        for (index, exception) in source.exceptions.iter().enumerate() {
            assert!(
                exception.charge_prod == 0. || setup.nonzero_exceptions.contains(&index),
                "the set of non-excluded exceptions cannot change"
            );
        }
        assert!(
            setup
                .nonzero_exceptions
                .iter()
                .all(|&i| i < source.exceptions.len()),
            "non-excluded exception count cannot change"
        );
        let mut parameters = setup.parameters.clone();
        let mut seen = HashSet::new();
        for electrode in &source.electrodes {
            for &i in &electrode.particles {
                let index = *setup
                    .system_to_electrode
                    .get(i)
                    .expect("electrode particle index out of range");
                assert_ne!(index, NONE, "electrode particle membership cannot change");
                parameters[index] = [
                    electrode.potential,
                    electrode.gaussian_width,
                    electrode.thomas_fermi_scale,
                ];
                seen.insert(i);
            }
        }
        assert_eq!(
            seen.len(),
            setup.electrode_to_system.len(),
            "electrode particle membership cannot change"
        );
        let work = self.workspace.0.get_mut().unwrap();
        if parameters != setup.parameters {
            work.solver_valid = false;
        }
        setup.parameters = parameters;
        self.particles.clone_from(&source.particles);
        self.exceptions.clone_from(&source.exceptions);
        self.electrodes.clone_from(&source.electrodes);
        self.external_field = source.external_field;
        self.charge_target = source.charge_target;
        setup.target = self.charge_target;
        for (i, &q) in self.particles.iter().enumerate() {
            if setup.system_to_electrode[i] == NONE {
                setup.target -= q;
                if !work.charges.is_empty() {
                    work.charges[i] = q;
                }
            }
        }
    }

    fn prepare<'a>(
        &'a self,
        positions: &[f64],
        vectors: &[[f64; 3]; 3],
        want_forces: bool,
    ) -> MutexGuard<'a, Work> {
        let setup = self.setup.as_ref().expect("force must be initialized");
        assert_eq!(positions.len(), 3 * self.particles.len());
        for (axis, vector) in vectors.iter().enumerate() {
            assert!(
                vector[axis] >= 1.999999 * self.cutoff,
                "periodic box is smaller than twice the cutoff"
            );
        }
        let mut work = self.workspace.0.lock().unwrap();
        if work.charges.len() != self.particles.len() {
            work.charges = self
                .particles
                .iter()
                .enumerate()
                .map(|(i, &q)| {
                    if setup.system_to_electrode[i] == NONE {
                        q
                    } else {
                        0.
                    }
                })
                .collect();
        }
        work.pairs.clear();
        work.background_energy = 0.;
        if want_forces {
            work.forces.resize(positions.len(), 0.);
            work.forces.fill(0.);
        }
        if want_forces || !setup.electrode_to_system.is_empty() {
            self.neighbors
                .visit(positions, self.cutoff, Some(vectors), |i, j| {
                    let ii = setup.system_to_electrode[i];
                    let jj = setup.system_to_electrode[j];
                    let fixed = ii == NONE && jj == NONE;
                    if (fixed && !want_forces) || setup.exclusions.contains(&ordered(i, j)) {
                        return;
                    }
                    let d = displacement(positions, i, j, Some(vectors));
                    let r2 = d.iter().map(|v| v * v).sum::<f64>();
                    if r2 >= self.cutoff * self.cutoff {
                        return;
                    }
                    let width2 = (if ii == NONE {
                        0.
                    } else {
                        setup.parameters[ii][1].powi(2)
                    }) + (if jj == NONE {
                        0.
                    } else {
                        setup.parameters[jj][1].powi(2)
                    });
                    let r = r2.sqrt();
                    let ar = setup.pme.0 * r;
                    let (eta_erfc, eta_exp) = if width2 == 0. {
                        (0., 0.)
                    } else {
                        let eta = r / width2.sqrt();
                        (
                            libm::erfc(eta),
                            if want_forces {
                                eta * (-eta * eta).exp()
                            } else {
                                0.
                            },
                        )
                    };
                    let difference = libm::erfc(ar) - eta_erfc;
                    let energy_kernel = ONE_4PI_EPS0 * difference / r;
                    let force_kernel = if want_forces {
                        ONE_4PI_EPS0
                            * (difference + 2. / PI.sqrt() * (ar * (-ar * ar).exp() - eta_exp))
                            / (r * r2)
                    } else {
                        0.
                    };
                    if fixed {
                        let qq = work.charges[i] * work.charges[j];
                        work.background_energy += qq * energy_kernel;
                        pair_force(&mut work.forces, i, j, d, qq * force_kernel);
                    } else {
                        work.pairs.push((i, j, energy_kernel, force_kernel));
                    }
                });
        }
        self.solve(positions, vectors, setup, &mut work);
        work
    }

    pub fn charges<B: Backend>(
        &self,
        positions: &Tensor<B, 2>,
        vectors: Option<&[[f64; 3]; 3]>,
    ) -> Vec<f64> {
        crate::tensor::with_f64_data(positions, |positions| {
            let work = self.prepare(positions, vectors.expect("periodic box is required"), false);
            work.charges.clone()
        })
    }

    fn energy_forces(&self, positions: &[f64], vectors: &[[f64; 3]; 3], work: &mut Work) -> f64 {
        let setup = self.setup.as_ref().unwrap();
        let alpha = setup.pme.0;
        let particles = &work.charges;
        let forces = &mut work.forces;
        let mut energy = work.background_energy;
        for &(i, j, energy_kernel, force_kernel) in &work.pairs {
            let qq = particles[i] * particles[j];
            energy += qq * energy_kernel;
            let d = displacement(positions, i, j, Some(vectors));
            pair_force(forces, i, j, d, qq * force_kernel);
        }
        for &(i, j) in &setup.exclusions {
            if i == j {
                continue;
            }
            let d = displacement(positions, i, j, self.periodic_exceptions.then_some(vectors));
            let r2 = d.iter().map(|v| v * v).sum::<f64>();
            let r = r2.sqrt();
            let ar = alpha * r;
            let qq = ONE_4PI_EPS0 * particles[i] * particles[j];
            if ar > 1e-6 {
                let erf = libm::erf(ar);
                energy -= qq * erf / r;
                let factor = -qq * (erf - 2. / PI.sqrt() * ar * (-ar * ar).exp()) / (r * r2);
                pair_force(forces, i, j, d, factor);
            } else {
                energy -= qq * 2. / PI.sqrt() * alpha;
            }
        }
        energy += self
            .pme
            .evaluate_charges(positions, particles, vectors, setup.pme, forces);
        let mut total_charge = 0.;
        for (i, &q) in particles.iter().enumerate() {
            total_charge += q;
            let field_energy = (0..3)
                .map(|axis| positions[3 * i + axis] * self.external_field[axis])
                .sum::<f64>();
            energy -= q * (ONE_4PI_EPS0 / PI.sqrt() * q * alpha + field_energy);
            for axis in 0..3 {
                forces[3 * i + axis] += q * self.external_field[axis];
            }
        }
        let volume = reciprocal_box(vectors).1;
        energy -= 0.5 * PI * ONE_4PI_EPS0 * total_charge * total_charge / (volume * alpha * alpha);
        for (ii, &i) in setup.electrode_to_system.iter().enumerate() {
            let q = particles[i];
            let [potential, width, tf] = setup.parameters[ii];
            energy += q
                * (q * (ONE_4PI_EPS0 / (2. * PI).sqrt() / width + 2. * PI * ONE_4PI_EPS0 * tf)
                    - potential);
        }
        for &index in &setup.nonzero_exceptions {
            let exception = &self.exceptions[index];
            if exception.charge_prod == 0. {
                continue;
            }
            let (i, j) = (exception.particle1, exception.particle2);
            let d = displacement(positions, i, j, self.periodic_exceptions.then_some(vectors));
            let r2 = d.iter().map(|v| v * v).sum::<f64>();
            let potential = ONE_4PI_EPS0 * exception.charge_prod / r2.sqrt();
            energy += potential;
            pair_force(forces, i, j, d, potential / r2);
        }
        energy
    }
}

fn ordered(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}
fn displacement(
    positions: &[f64],
    i: usize,
    j: usize,
    vectors: Option<&[[f64; 3]; 3]>,
) -> [f64; 3] {
    let mut d = std::array::from_fn(|axis| positions[3 * j + axis] - positions[3 * i + axis]);
    if let Some(vectors) = vectors {
        apply_minimum_image_ortho(&mut d, vectors);
    }
    d
}
fn pair_force(forces: &mut [f64], i: usize, j: usize, d: [f64; 3], factor: f64) {
    for axis in 0..3 {
        let value = factor * d[axis];
        forces[3 * i + axis] -= value;
        forces[3 * j + axis] += value;
    }
}

impl<B: Backend> Force<B> for ConstantPotentialForce {
    super::force_metadata_impl!();
    fn initialize_masses(&mut self, masses: &[f64]) {
        self.masses = masses.to_vec();
    }
    fn initialize(&mut self, n: usize, vectors: Option<&[[f64; 3]; 3]>) {
        self.initialize_context(n, vectors);
    }
    fn uses_periodic_boundary_conditions(&self) -> bool {
        true
    }
    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64) {
        crate::tensor::with_f64_data(positions, |values| {
            let vectors = vectors.expect("periodic box is required");
            let mut work = self.prepare(values, vectors, true);
            let energy = self.energy_forces(values, vectors, &mut work);
            (
                crate::tensor::from_f64_vec(
                    work.forces.clone(),
                    positions.dims(),
                    &positions.device(),
                ),
                energy,
            )
        })
    }
    fn prepare_host_evaluation<'a>(
        &'a self,
        positions: &'a Tensor<B, 2>,
        vectors: Option<&'a [[f64; 3]; 3]>,
    ) -> Option<HostForceEvaluation<'a>> {
        crate::tensor::with_f64_data(positions, |values| {
            let vectors = vectors.expect("periodic box is required");
            let mut work = self.prepare(values, vectors, true);
            let energy = self.energy_forces(values, vectors, &mut work);
            Some(HostForceEvaluation::new(
                energy,
                move |coefficient, forces| {
                    for (output, &value) in forces.iter_mut().zip(&work.forces) {
                        *output += coefficient * value;
                    }
                },
            ))
        })
    }
}
