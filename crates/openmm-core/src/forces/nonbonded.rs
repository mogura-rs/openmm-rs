// NonbondedForce: LJ + Coulomb with exclusions/exceptions.
// Methods: NoCutoff, CutoffNonPeriodic, CutoffPeriodic, Ewald, PME, LJPME.
// Reference test: openmm/tests/TestNonbondedForce.h
// Reference impl: openmm/platforms/reference/src/SimTKReference/ReferenceNonbondedIxn.cpp
use super::ewald::{self, EwaldCache};
use super::neighbors::NeighborCells;
use super::pme::{reciprocal_box, PmeCache};
use super::Force;
use crate::forces::harmonic_bond::apply_minimum_image_ortho;
use burn::prelude::*;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::OnceLock;

/// 1/(4*pi*eps0) in kJ/mol * nm / e^2.
pub const ONE_4PI_EPS0: f64 = 138.93545764438198;

/// Nonbonded calculation method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NonbondedMethod {
    /// No cutoff; all pairs computed (non-periodic).
    NoCutoff,
    /// Cutoff, non-periodic (reaction-field or plain truncation).
    CutoffNonPeriodic,
    /// Cutoff with periodic boundary conditions.
    CutoffPeriodic,
    /// Direct Ewald summation (rectangular periodic boxes).
    Ewald,
    /// Particle Mesh Ewald (periodic).
    PME,
    /// PME electrostatics and dispersion.
    LJPME,
}

/// Per-particle nonbonded parameters.
#[derive(Debug, Clone)]
pub struct ParticleParam {
    /// Partial charge in units of elementary charge.
    pub charge: f64,
    /// LJ sigma in nm (combined via Lorentz-Berthelot: sigma_ij = (si+sj)/2).
    pub sigma: f64,
    /// LJ epsilon in kJ/mol (combined: eps_ij = sqrt(ei*ej)).
    pub epsilon: f64,
}

/// Exception (exclusion or 1-4 scaled interaction).
#[derive(Debug, Clone)]
pub struct ExceptionParam {
    pub particle1: usize,
    pub particle2: usize,
    /// Effective charge product (0 for exclusion, scaled for 1-4).
    pub charge_prod: f64,
    /// Effective sigma for this pair.
    pub sigma: f64,
    /// Effective epsilon for this pair (0 for exclusion, scaled for 1-4).
    pub epsilon: f64,
}

#[derive(Debug, Clone)]
pub struct ParameterOffset {
    pub parameter: String,
    pub index: usize,
    pub charge_scale: f64,
    pub sigma_scale: f64,
    pub epsilon_scale: f64,
}

#[derive(Clone)]
struct EffectiveParameters {
    particles: Vec<ParticleParam>,
    exceptions: Vec<ExceptionParam>,
}

#[derive(Clone)]
pub struct NonbondedForce {
    metadata: super::ForceMetadata,
    reciprocal_force_group: i32,
    include_direct_space: bool,
    particles: Vec<ParticleParam>,
    exceptions: Vec<ExceptionParam>,
    global_parameters: Vec<(String, f64)>,
    parameter_values: HashMap<String, f64>,
    particle_offsets: Vec<ParameterOffset>,
    exception_offsets: Vec<ParameterOffset>,
    effective_parameters: OnceLock<EffectiveParameters>,
    method: NonbondedMethod,
    /// Cutoff distance in nm.
    cutoff_distance: f64,
    /// Reaction-field dielectric (used with CutoffNonPeriodic).
    reaction_field_dielectric: f64,
    /// PME error tolerance (used to derive alpha and grid spacing).
    ewald_error_tolerance: f64,
    exceptions_use_pbc: bool,
    use_dispersion_correction: bool,
    use_switching_function: bool,
    switching_distance: f64,
    pme_parameters: Option<(f64, [usize; 3])>,
    pme_cache: PmeCache,
    ljpme_parameters: Option<(f64, [usize; 3])>,
    ljpme_cache: PmeCache,
    ewald_cache: EwaldCache,
    initialized_reciprocal: Option<(f64, [usize; 3])>,
    initialized_ljpme: Option<(f64, [usize; 3])>,
    dispersion_coefficient: OnceLock<f64>,
    exclusions: OnceLock<std::collections::HashSet<(usize, usize)>>,
    neighbor_cells: NeighborCells,
    use_neighbor_cells: bool,
}

impl NonbondedForce {
    super::force_metadata!();
    pub fn reciprocal_space_force_group(&self) -> i32 {
        self.reciprocal_force_group
    }
    pub fn set_reciprocal_space_force_group(&mut self, group: i32) {
        assert!(
            (-1..32).contains(&group),
            "reciprocal force group must be between -1 and 31"
        );
        self.reciprocal_force_group = group;
    }
    pub fn include_direct_space(&self) -> bool {
        self.include_direct_space
    }
    pub fn set_include_direct_space(&mut self, enabled: bool) {
        self.include_direct_space = enabled;
    }

    pub fn new() -> Self {
        Self {
            metadata: super::ForceMetadata::new("NonbondedForce"),
            reciprocal_force_group: -1,
            include_direct_space: true,

            particles: Vec::new(),
            exceptions: Vec::new(),
            global_parameters: Vec::new(),
            parameter_values: HashMap::new(),
            particle_offsets: Vec::new(),
            exception_offsets: Vec::new(),
            effective_parameters: OnceLock::new(),
            method: NonbondedMethod::NoCutoff,
            cutoff_distance: 1.0,
            reaction_field_dielectric: 78.3,
            ewald_error_tolerance: 5e-4,
            exceptions_use_pbc: false,
            use_dispersion_correction: true,
            use_switching_function: false,
            switching_distance: -1.0,
            pme_parameters: None,
            pme_cache: PmeCache::default(),
            ljpme_parameters: None,
            ljpme_cache: PmeCache::dispersion(),
            ewald_cache: EwaldCache::default(),
            initialized_reciprocal: None,
            initialized_ljpme: None,
            dispersion_coefficient: OnceLock::new(),
            exclusions: OnceLock::new(),
            neighbor_cells: NeighborCells::default(),
            use_neighbor_cells: true,
        }
    }

    /// Add a particle. Returns particle index.
    pub fn add_particle(&mut self, charge: f64, sigma: f64, epsilon: f64) -> usize {
        validate_particle_parameters(charge, sigma, epsilon);
        self.dispersion_coefficient.take();
        self.effective_parameters.take();
        self.particles.push(ParticleParam {
            charge,
            sigma,
            epsilon,
        });
        self.particles.len() - 1
    }

    pub fn set_particle_parameters(&mut self, index: usize, charge: f64, sigma: f64, epsilon: f64) {
        validate_particle_parameters(charge, sigma, epsilon);
        self.dispersion_coefficient.take();
        self.effective_parameters.take();
        self.particles[index] = ParticleParam {
            charge,
            sigma,
            epsilon,
        };
    }

    pub fn particle_parameters(&self, index: usize) -> &ParticleParam {
        &self.particles[index]
    }

    pub fn num_particles(&self) -> usize {
        self.particles.len()
    }

    /// Add an exception (or exclusion). Returns exception index.
    pub fn add_exception(
        &mut self,
        p1: usize,
        p2: usize,
        charge_prod: f64,
        sigma: f64,
        epsilon: f64,
    ) -> usize {
        self.exclusions.take();
        self.effective_parameters.take();
        validate_particle_parameters(charge_prod, sigma, epsilon);
        self.exceptions.push(ExceptionParam {
            particle1: p1,
            particle2: p2,
            charge_prod,
            sigma,
            epsilon,
        });
        self.exceptions.len() - 1
    }

    pub fn set_exception_parameters(
        &mut self,
        index: usize,
        p1: usize,
        p2: usize,
        charge_prod: f64,
        sigma: f64,
        epsilon: f64,
    ) {
        self.exclusions.take();
        self.effective_parameters.take();
        validate_particle_parameters(charge_prod, sigma, epsilon);
        self.exceptions[index] = ExceptionParam {
            particle1: p1,
            particle2: p2,
            charge_prod,
            sigma,
            epsilon,
        };
    }

    pub fn exception_parameters(&self, index: usize) -> &ExceptionParam {
        &self.exceptions[index]
    }

    pub fn num_exceptions(&self) -> usize {
        self.exceptions.len()
    }

    pub fn add_global_parameter(&mut self, name: &str, default: f64) -> usize {
        assert!(
            !name.is_empty() && default.is_finite(),
            "global parameters require a name and a finite default value"
        );
        self.global_parameters.push((name.to_owned(), default));
        self.parameter_values.insert(name.to_owned(), default);
        self.invalidate_parameters();
        self.global_parameters.len() - 1
    }

    pub fn num_global_parameters(&self) -> usize {
        self.global_parameters.len()
    }

    pub fn global_parameter_name(&self, index: usize) -> &str {
        &self.global_parameters[index].0
    }

    pub fn global_parameter_default_value(&self, index: usize) -> f64 {
        self.global_parameters[index].1
    }

    pub fn set_global_parameter_default_value(&mut self, index: usize, value: f64) {
        assert!(value.is_finite(), "global parameter values must be finite");
        let (name, default) = &mut self.global_parameters[index];
        *default = value;
        self.parameter_values.insert(name.clone(), value);
        self.invalidate_parameters();
    }

    pub fn set_global_parameter_name(&mut self, index: usize, name: &str) {
        assert!(!name.is_empty(), "global parameter names must not be empty");
        let (old_name, default) = &mut self.global_parameters[index];
        self.parameter_values.remove(old_name);
        self.parameter_values.insert(name.to_owned(), *default);
        *old_name = name.to_owned();
        self.invalidate_parameters();
    }

    fn offset(&self, parameter: &str, index: usize, scales: [f64; 3]) -> ParameterOffset {
        assert!(
            self.parameter_values.contains_key(parameter),
            "unknown offset parameter '{parameter}'"
        );
        assert!(
            scales.iter().all(|x| x.is_finite()),
            "offset scales must be finite"
        );
        ParameterOffset {
            parameter: parameter.to_owned(),
            index,
            charge_scale: scales[0],
            sigma_scale: scales[1],
            epsilon_scale: scales[2],
        }
    }

    pub fn add_particle_parameter_offset(
        &mut self,
        parameter: &str,
        index: usize,
        charge: f64,
        sigma: f64,
        epsilon: f64,
    ) -> usize {
        assert!(
            index < self.particles.len(),
            "particle offset index out of range"
        );
        let offset = self.offset(parameter, index, [charge, sigma, epsilon]);
        self.particle_offsets.push(offset);
        self.invalidate_parameters();
        self.particle_offsets.len() - 1
    }

    pub fn add_exception_parameter_offset(
        &mut self,
        parameter: &str,
        index: usize,
        charge: f64,
        sigma: f64,
        epsilon: f64,
    ) -> usize {
        assert!(
            index < self.exceptions.len(),
            "exception offset index out of range"
        );
        let offset = self.offset(parameter, index, [charge, sigma, epsilon]);
        self.exception_offsets.push(offset);
        self.invalidate_parameters();
        self.exception_offsets.len() - 1
    }

    pub fn num_particle_parameter_offsets(&self) -> usize {
        self.particle_offsets.len()
    }
    pub fn num_exception_parameter_offsets(&self) -> usize {
        self.exception_offsets.len()
    }
    pub fn particle_parameter_offset(&self, index: usize) -> &ParameterOffset {
        &self.particle_offsets[index]
    }
    pub fn exception_parameter_offset(&self, index: usize) -> &ParameterOffset {
        &self.exception_offsets[index]
    }

    pub fn set_particle_parameter_offset(
        &mut self,
        offset_index: usize,
        parameter: &str,
        index: usize,
        charge: f64,
        sigma: f64,
        epsilon: f64,
    ) {
        assert!(
            index < self.particles.len(),
            "particle offset index out of range"
        );
        self.particle_offsets[offset_index] =
            self.offset(parameter, index, [charge, sigma, epsilon]);
        self.invalidate_parameters();
    }

    pub fn set_exception_parameter_offset(
        &mut self,
        offset_index: usize,
        parameter: &str,
        index: usize,
        charge: f64,
        sigma: f64,
        epsilon: f64,
    ) {
        assert!(
            index < self.exceptions.len(),
            "exception offset index out of range"
        );
        self.exception_offsets[offset_index] =
            self.offset(parameter, index, [charge, sigma, epsilon]);
        self.invalidate_parameters();
    }

    fn invalidate_parameters(&mut self) {
        self.effective_parameters.take();
        self.dispersion_coefficient.take();
    }

    fn parameters(&self) -> (&[ParticleParam], &[ExceptionParam]) {
        if self.particle_offsets.is_empty() && self.exception_offsets.is_empty() {
            return (&self.particles, &self.exceptions);
        }
        let parameters = self.effective_parameters.get_or_init(|| {
            let mut particles = self.particles.clone();
            let mut exceptions = self.exceptions.clone();
            let mut seen = std::collections::HashSet::new();
            for offset in self.particle_offsets.iter().rev() {
                if !seen.insert((&offset.parameter, offset.index)) {
                    continue;
                }
                let value = self.parameter_values[&offset.parameter];
                let p = &mut particles[offset.index];
                p.charge += value * offset.charge_scale;
                p.sigma += value * offset.sigma_scale;
                p.epsilon += value * offset.epsilon_scale;
            }
            seen.clear();
            for offset in self.exception_offsets.iter().rev() {
                if !seen.insert((&offset.parameter, offset.index)) {
                    continue;
                }
                let value = self.parameter_values[&offset.parameter];
                let p = &mut exceptions[offset.index];
                p.charge_prod += value * offset.charge_scale;
                p.sigma += value * offset.sigma_scale;
                p.epsilon += value * offset.epsilon_scale;
            }
            EffectiveParameters {
                particles,
                exceptions,
            }
        });
        (&parameters.particles, &parameters.exceptions)
    }

    /// Auto-generate exclusions (1-2, 1-3) and 1-4 scaled exceptions from a bond list.
    /// charge_scale and lj_scale are the 1-4 scaling factors (e.g., 1/1.2 and 0.5 for AMBER).
    pub fn create_exceptions_from_bonds(
        &mut self,
        bonds: &[(usize, usize)],
        coulomb14scale: f64,
        lj14scale: f64,
    ) {
        let n = self.particles.len();
        // Build adjacency list.
        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(a, b) in bonds {
            neighbors[a].push(b);
            neighbors[b].push(a);
        }

        let mut exclusions: std::collections::HashSet<(usize, usize)> = Default::default();
        let mut pairs_14: std::collections::HashSet<(usize, usize)> = Default::default();

        for i in 0..n {
            // 1-2 exclusions
            for &j in &neighbors[i] {
                if i < j {
                    exclusions.insert((i, j));
                }
            }
            // 1-3 exclusions
            for &j in &neighbors[i] {
                for &k in &neighbors[j] {
                    if k != i {
                        let key = if i < k { (i, k) } else { (k, i) };
                        exclusions.insert(key);
                    }
                }
            }
            // 1-4 pairs
            for &j in &neighbors[i] {
                for &k in &neighbors[j] {
                    if k == i {
                        continue;
                    }
                    for &l in &neighbors[k] {
                        if l == j || l == i {
                            continue;
                        }
                        let key = if i < l { (i, l) } else { (l, i) };
                        if !exclusions.contains(&key) {
                            pairs_14.insert(key);
                        }
                    }
                }
            }
        }

        // Add exclusions (zero interaction).
        for (i, j) in &exclusions {
            let sigma = 0.5 * (self.particles[*i].sigma + self.particles[*j].sigma);
            self.add_exception(*i, *j, 0.0, sigma, 0.0);
        }
        // A shorter path takes precedence in rings and branched molecules.
        for (i, j) in pairs_14.difference(&exclusions) {
            let pi = &self.particles[*i];
            let pj = &self.particles[*j];
            let charge_prod = coulomb14scale * pi.charge * pj.charge;
            let sigma = 0.5 * (pi.sigma + pj.sigma);
            let epsilon = lj14scale * (pi.epsilon * pj.epsilon).sqrt();
            self.add_exception(*i, *j, charge_prod, sigma, epsilon);
        }
    }

    pub fn set_nonbonded_method(&mut self, method: NonbondedMethod) {
        self.method = method;
    }

    pub fn nonbonded_method(&self) -> NonbondedMethod {
        self.method
    }

    pub fn uses_periodic_boundary_conditions(&self) -> bool {
        matches!(
            self.method,
            NonbondedMethod::CutoffPeriodic
                | NonbondedMethod::Ewald
                | NonbondedMethod::PME
                | NonbondedMethod::LJPME
        )
    }

    pub fn set_cutoff_distance(&mut self, cutoff: f64) {
        assert!(
            cutoff.is_finite() && cutoff > 0.0,
            "cutoff must be positive and finite"
        );
        self.dispersion_coefficient.take();
        self.cutoff_distance = cutoff;
    }

    pub fn cutoff_distance(&self) -> f64 {
        self.cutoff_distance
    }

    pub fn set_reaction_field_dielectric(&mut self, eps: f64) {
        assert!(
            !eps.is_nan() && eps >= 1.0,
            "reaction field dielectric must be at least one"
        );
        self.reaction_field_dielectric = eps;
    }

    pub fn reaction_field_dielectric(&self) -> f64 {
        self.reaction_field_dielectric
    }

    pub fn set_ewald_error_tolerance(&mut self, tol: f64) {
        assert!(
            tol.is_finite() && tol > 0.0 && tol < 0.5,
            "Ewald tolerance must be between zero and one half"
        );
        self.ewald_error_tolerance = tol;
    }

    pub fn ewald_error_tolerance(&self) -> f64 {
        self.ewald_error_tolerance
    }

    pub fn set_exceptions_use_periodic_boundary_conditions(&mut self, periodic: bool) {
        self.exceptions_use_pbc = periodic;
    }

    pub fn exceptions_use_periodic_boundary_conditions(&self) -> bool {
        self.exceptions_use_pbc
    }

    /// Select cell traversal or exhaustive enumeration; the physical model is unchanged.
    pub fn set_use_neighbor_cells(&mut self, enabled: bool) {
        self.use_neighbor_cells = enabled;
    }

    pub fn set_use_dispersion_correction(&mut self, enabled: bool) {
        self.use_dispersion_correction = enabled;
    }
    pub fn use_dispersion_correction(&self) -> bool {
        self.use_dispersion_correction
    }
    pub fn set_use_switching_function(&mut self, enabled: bool) {
        self.use_switching_function = enabled;
        self.dispersion_coefficient.take();
    }
    pub fn use_switching_function(&self) -> bool {
        self.use_switching_function
    }
    pub fn set_switching_distance(&mut self, distance: f64) {
        assert!(
            distance.is_finite() && distance >= 0.0,
            "switching distance must be nonnegative and finite"
        );
        self.switching_distance = distance;
        self.dispersion_coefficient.take();
    }
    pub fn switching_distance(&self) -> f64 {
        self.switching_distance
    }

    pub fn set_pme_parameters(&mut self, alpha: f64, dimensions: [usize; 3]) {
        assert!(
            alpha.is_finite() && alpha >= 0.0,
            "PME alpha must be nonnegative and finite"
        );
        assert!(
            alpha == 0.0 || dimensions.iter().all(|&n| n >= 5),
            "PME grid dimensions must be at least five"
        );
        self.pme_parameters = if alpha == 0.0 {
            None
        } else {
            Some((alpha, dimensions))
        };
    }

    pub fn pme_parameters(&self) -> (f64, [usize; 3]) {
        self.pme_parameters.unwrap_or((0.0, [0; 3]))
    }

    pub fn set_ljpme_parameters(&mut self, alpha: f64, dimensions: [usize; 3]) {
        assert!(
            alpha.is_finite() && alpha >= 0.0,
            "LJPME alpha must be nonnegative and finite"
        );
        assert!(
            alpha == 0.0 || dimensions.iter().all(|&n| n >= 5),
            "LJPME grid dimensions must be at least five"
        );
        self.ljpme_parameters = if alpha == 0.0 {
            None
        } else {
            Some((alpha, dimensions))
        };
    }

    pub fn ljpme_parameters(&self) -> (f64, [usize; 3]) {
        self.ljpme_parameters.unwrap_or((0.0, [0; 3]))
    }

    pub fn effective_ljpme_parameters(&self, box_vectors: &[[f64; 3]; 3]) -> (f64, [usize; 3]) {
        self.ljpme_parameters.unwrap_or_else(|| {
            let alpha = (-(2.0 * self.ewald_error_tolerance).ln()).sqrt() / self.cutoff_distance;
            let dimensions = std::array::from_fn(|axis| {
                (alpha * box_vectors[axis][axis] / (3.0 * self.ewald_error_tolerance.powf(0.2)))
                    .ceil()
                    .max(6.0) as usize
            });
            (alpha, dimensions)
        })
    }

    pub fn effective_pme_parameters(&self, box_vectors: &[[f64; 3]; 3]) -> (f64, [usize; 3]) {
        self.pme_parameters.unwrap_or_else(|| {
            let alpha = (-(2.0 * self.ewald_error_tolerance).ln()).sqrt() / self.cutoff_distance;
            let dimensions = std::array::from_fn(|axis| {
                (2.0 * alpha * box_vectors[axis][axis]
                    / (3.0 * self.ewald_error_tolerance.powf(0.2)))
                .ceil()
                .max(6.0) as usize
            });
            (alpha, dimensions)
        })
    }

    fn dispersion_correction(&self, volume: f64) -> f64 {
        *self.dispersion_coefficient.get_or_init(|| {
            // The reference tail coefficient uses parameter defaults, not Context values.
            let mut default_particles = Vec::new();
            let particles = if self.particle_offsets.is_empty() {
                &self.particles
            } else {
                default_particles.extend_from_slice(&self.particles);
                let defaults: HashMap<&str, f64> = self
                    .global_parameters
                    .iter()
                    .map(|(n, v)| (n.as_str(), *v))
                    .collect();
                for offset in &self.particle_offsets {
                    let value = defaults[offset.parameter.as_str()];
                    default_particles[offset.index].sigma += value * offset.sigma_scale;
                    default_particles[offset.index].epsilon += value * offset.epsilon_scale;
                }
                &default_particles
            };
            let mut moments = [0.0; 13];
            let mut diagonal = [0.0; 13];
            for particle in particles {
                let mut power = 1.0;
                for k in 0..=12 {
                    moments[k] += particle.epsilon.sqrt() * power;
                    diagonal[k] += particle.epsilon * power;
                    power *= particle.sigma;
                }
            }
            let pair_moment = |power: usize| {
                let mut coefficient = 1.0;
                let mut sum = 0.0;
                for k in 0..=power {
                    sum += coefficient * moments[k] * moments[power - k];
                    coefficient *= (power - k) as f64 / (k + 1) as f64;
                }
                sum / 2.0_f64.powi(power as i32) + diagonal[power]
            };
            let integral = |power: i32| {
                let tail = self.cutoff_distance.powi(3 - power) / (power - 3) as f64;
                if !self.use_switching_function {
                    return tail;
                }
                assert!(
                    self.switching_distance > 0.0 && self.switching_distance < self.cutoff_distance,
                    "dispersion switching distance must be between zero and cutoff"
                );
                let f = |r: f64| {
                    let t = (r - self.switching_distance)
                        / (self.cutoff_distance - self.switching_distance);
                    t.powi(3) * (10.0 + t * (-15.0 + 6.0 * t)) * r.powi(2 - power)
                };
                tail + integrate(&f, self.switching_distance, self.cutoff_distance)
            };
            let n = self.particles.len() as f64;
            8.0 * PI * n / (n + 1.0)
                * (pair_moment(12) * integral(12) - pair_moment(6) * integral(6))
        }) / volume
    }

    pub fn update_parameters_from(&mut self, source: &Self) {
        assert_eq!(
            self.particles.len(),
            source.particles.len(),
            "particle count must match"
        );
        assert_eq!(
            self.exceptions.len(),
            source.exceptions.len(),
            "exception count must match"
        );
        for (old, new) in [
            (&self.particle_offsets, &source.particle_offsets),
            (&self.exception_offsets, &source.exception_offsets),
        ] {
            assert_eq!(old.len(), new.len(), "parameter offset count must match");
            assert!(
                new.iter().all(|n| old
                    .iter()
                    .any(|o| o.parameter == n.parameter && o.index == n.index)),
                "parameter offset names and target indices must not change"
            );
        }
        for (i, (old, new)) in self.exceptions.iter().zip(&source.exceptions).enumerate() {
            assert_eq!(
                (old.particle1, old.particle2),
                (new.particle1, new.particle2),
                "exception particle indices must not change"
            );
            let has_offset = self.exception_offsets.iter().any(|o| o.index == i);
            assert!(
                old.charge_prod != 0.0
                    || old.epsilon != 0.0
                    || has_offset
                    || (new.charge_prod == 0.0 && new.epsilon == 0.0),
                "excluded exceptions cannot become non-excluded without reinitialization"
            );
        }
        self.particles = source.particles.clone();
        self.exceptions = source.exceptions.clone();
        self.particle_offsets = source.particle_offsets.clone();
        self.exception_offsets = source.exception_offsets.clone();
        self.global_parameters = source.global_parameters.clone();
        self.effective_parameters.take();
        self.exclusions.take();
        self.dispersion_coefficient.take();
    }
}

impl NonbondedForce {
    fn evaluate<B: Backend>(
        &self,
        positions: &Tensor<B, 2>,
        box_vectors: Option<&[[f64; 3]; 3]>,
        include_direct: bool,
        include_reciprocal: bool,
    ) -> (Tensor<B, 2>, f64) {
        let device = positions.device();
        let (particles, exceptions) = self.parameters();
        let n = positions.dims()[0];
        assert_eq!(
            n,
            particles.len(),
            "NonbondedForce must have one parameter record per particle"
        );
        let pos: Vec<f64> = positions
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(|v| v as f64)
            .collect();

        let mut forces = vec![0.0f64; n * 3];
        let mut energy = 0.0f64;

        let excluded = self.exclusions.get_or_init(|| {
            exceptions
                .iter()
                .map(|e| (e.particle1.min(e.particle2), e.particle1.max(e.particle2)))
                .collect()
        });

        // Direct space: all pairs (respecting cutoff and exclusions).
        let use_cutoff = self.method != NonbondedMethod::NoCutoff;
        let use_pbc = self.uses_periodic_boundary_conditions();
        let cutoff2 = self.cutoff_distance * self.cutoff_distance;
        if use_pbc {
            let vectors = box_vectors.expect("periodic nonbonded forces require box vectors");
            assert!(
                (0..3).all(|axis| 2.0 * self.cutoff_distance <= vectors[axis][axis] + 1e-12),
                "cutoff must not exceed half the periodic box size"
            );
        }
        if use_cutoff && self.use_switching_function {
            assert!(
                self.switching_distance >= 0.0 && self.switching_distance < self.cutoff_distance,
                "switching distance must be between zero and cutoff"
            );
        }
        let pme = self.initialized_reciprocal.or_else(|| match self.method {
            NonbondedMethod::Ewald => {
                let vectors = box_vectors.unwrap();
                assert!(
                    vectors[1][0] == 0.0 && vectors[2][0] == 0.0 && vectors[2][1] == 0.0,
                    "Ewald requires a rectangular periodic box; use PME for triclinic boxes"
                );
                Some(ewald::parameters(
                    std::array::from_fn(|axis| vectors[axis][axis]),
                    self.cutoff_distance,
                    self.ewald_error_tolerance,
                ))
            }
            NonbondedMethod::PME | NonbondedMethod::LJPME => {
                Some(self.effective_pme_parameters(box_vectors.unwrap()))
            }
            _ => None,
        });
        let ljpme = (self.method == NonbondedMethod::LJPME).then(|| {
            self.initialized_ljpme
                .unwrap_or_else(|| self.effective_ljpme_parameters(box_vectors.unwrap()))
        });

        let mut evaluate_pair = |i: usize, j: usize| {
            let key = (i, j);
            if excluded.contains(&key) {
                return;
            }

            let mut delta = [
                pos[j * 3] - pos[i * 3],
                pos[j * 3 + 1] - pos[i * 3 + 1],
                pos[j * 3 + 2] - pos[i * 3 + 2],
            ];

            if use_pbc {
                if let Some(bv) = box_vectors {
                    apply_minimum_image_ortho(&mut delta, bv);
                }
            }

            let r2 = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];

            if use_cutoff && r2 >= cutoff2 {
                return;
            }

            let r = r2.sqrt();
            let pi = &particles[i];
            let pj = &particles[j];
            let (mut lj_energy, mut lj_force) = lj_pair(
                0.5 * (pi.sigma + pj.sigma),
                (pi.epsilon * pj.epsilon).sqrt(),
                r,
                r2,
            );
            if let Some((alpha, _)) = ljpme {
                let c6 = 4.0 * (pi.epsilon * pj.epsilon).sqrt() * (pi.sigma * pj.sigma).powi(3);
                let (correction, derivative) = dispersion_reciprocal_pair(c6, r2, alpha);
                let shift = lj_pair(
                    0.5 * (pi.sigma + pj.sigma),
                    (pi.epsilon * pj.epsilon).sqrt(),
                    self.cutoff_distance,
                    cutoff2,
                )
                .0 + dispersion_reciprocal_pair(c6, cutoff2, alpha).0;
                lj_energy += correction - shift;
                lj_force += derivative;
            }
            if use_cutoff
                && self.use_switching_function
                && ljpme.is_none()
                && r > self.switching_distance
            {
                let t = (r - self.switching_distance)
                    / (self.cutoff_distance - self.switching_distance);
                let switch = 1.0 + t.powi(3) * (-10.0 + t * (15.0 - 6.0 * t));
                let derivative = -30.0 * t * t * (1.0 - t).powi(2)
                    / (self.cutoff_distance - self.switching_distance);
                lj_force = switch * lj_force + lj_energy * derivative / r;
                lj_energy *= switch;
            }
            let charge = pi.charge * pj.charge;
            let (electrostatic_energy, electrostatic_force) = if let Some((alpha, _)) = pme {
                screened_coulomb(charge, r, r2, alpha)
            } else if use_cutoff {
                reaction_field(
                    charge,
                    r,
                    r2,
                    self.cutoff_distance,
                    self.reaction_field_dielectric,
                )
            } else {
                coulomb_pair(charge, r, r2)
            };
            let e = lj_energy + electrostatic_energy;
            let f_over_r = lj_force + electrostatic_force;

            energy += e;
            for d in 0..3 {
                let f = f_over_r * delta[d];
                forces[i * 3 + d] += f;
                forces[j * 3 + d] -= f;
            }
        };
        if include_direct {
            if use_cutoff && self.use_neighbor_cells && n > 32 {
                self.neighbor_cells.visit(
                    &pos,
                    self.cutoff_distance,
                    if use_pbc { box_vectors } else { None },
                    &mut evaluate_pair,
                );
            } else {
                for i in 0..n {
                    for j in i + 1..n {
                        evaluate_pair(i, j);
                    }
                }
            }
        }

        if include_reciprocal {
            if let Some((alpha, dimensions)) = pme {
                let vectors = box_vectors.unwrap();
                let (_, volume) = reciprocal_box(vectors);
                energy += if self.method == NonbondedMethod::Ewald {
                    self.ewald_cache.evaluate(
                        &pos,
                        particles,
                        std::array::from_fn(|axis| vectors[axis][axis]),
                        alpha,
                        dimensions,
                        &mut forces,
                    )
                } else {
                    self.pme_cache.evaluate(
                        &pos,
                        particles,
                        vectors,
                        alpha,
                        dimensions,
                        &mut forces,
                    )
                };
                let charge_squared: f64 = particles
                    .iter()
                    .map(|particle| particle.charge.powi(2))
                    .sum();
                let total_charge: f64 = particles.iter().map(|particle| particle.charge).sum();
                energy -= ONE_4PI_EPS0
                    * (alpha / PI.sqrt() * charge_squared
                        + PI * total_charge.powi(2) / (2.0 * alpha * alpha * volume));
            }
            if let Some((alpha, dimensions)) = ljpme {
                energy += self.ljpme_cache.evaluate(
                    &pos,
                    particles,
                    box_vectors.unwrap(),
                    alpha,
                    dimensions,
                    &mut forces,
                );
                energy += particles
                    .iter()
                    .map(|p| p.epsilon * p.sigma.powi(6))
                    .sum::<f64>()
                    * alpha.powi(6)
                    / 3.0;
            }
        }
        if include_direct && use_pbc && self.use_dispersion_correction && ljpme.is_none() {
            energy += self.dispersion_correction(reciprocal_box(box_vectors.unwrap()).1);
        }

        // Exceptions are untruncated and use their own periodicity setting.
        for exc in if include_direct { exceptions } else { &[] } {
            let i = exc.particle1;
            let j = exc.particle2;

            let mut delta = [
                pos[j * 3] - pos[i * 3],
                pos[j * 3 + 1] - pos[i * 3 + 1],
                pos[j * 3 + 2] - pos[i * 3 + 2],
            ];

            if use_pbc && self.exceptions_use_pbc {
                if let Some(bv) = box_vectors {
                    apply_minimum_image_ortho(&mut delta, bv);
                }
            }

            let r2 = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
            let r = r2.sqrt();

            let (mut e, mut f_over_r) = lj_coulomb_exception(exc, r, r2);
            if let Some((alpha, _)) = pme {
                let charge = particles[i].charge * particles[j].charge;
                let (correction, derivative) = reciprocal_pair(charge, r, r2, alpha);
                e -= correction;
                f_over_r -= derivative;
            }
            if let Some((alpha, _)) = ljpme {
                let pi = &particles[i];
                let pj = &particles[j];
                let c6 = 4.0 * (pi.epsilon * pj.epsilon).sqrt() * (pi.sigma * pj.sigma).powi(3);
                let (correction, derivative) = dispersion_reciprocal_pair(c6, r2, alpha);
                e += correction;
                f_over_r += derivative;
            }
            energy += e;
            for d in 0..3 {
                let f = f_over_r * delta[d];
                forces[i * 3 + d] += f;
                forces[j * 3 + d] -= f;
            }
        }

        let forces_f32: Vec<f32> = forces.iter().map(|&v| v as f32).collect();
        let forces_tensor =
            Tensor::<B, 1>::from_floats(forces_f32.as_slice(), &device).reshape([n, 3]);

        (forces_tensor, energy)
    }
}

impl Default for NonbondedForce {
    fn default() -> Self {
        Self::new()
    }
}

// --- Force computation ---

impl<B: Backend> Force<B> for NonbondedForce {
    super::force_metadata_impl!();
    fn initialize(&mut self, num_particles: usize, box_vectors: Option<&[[f64; 3]; 3]>) {
        assert_eq!(
            num_particles,
            self.particles.len(),
            "NonbondedForce must have one parameter record per particle"
        );
        let mut pairs = std::collections::HashSet::new();
        for e in &self.exceptions {
            assert!(
                e.particle1 < num_particles && e.particle2 < num_particles,
                "exception particle index out of range"
            );
            assert!(
                pairs.insert((e.particle1.min(e.particle2), e.particle1.max(e.particle2))),
                "duplicate exception pair"
            );
        }
        if self.use_switching_function {
            assert!(
                self.switching_distance >= 0.0 && self.switching_distance < self.cutoff_distance,
                "switching distance must be between zero and cutoff"
            );
        }
        self.initialized_reciprocal = None;
        self.initialized_ljpme = None;
        if self.uses_periodic_boundary_conditions() {
            let vectors = box_vectors.expect("periodic nonbonded forces require box vectors");
            assert!(
                (0..3).all(|axis| self.cutoff_distance <= 0.5 * vectors[axis][axis]),
                "cutoff must not exceed half the periodic box size"
            );
            self.initialized_reciprocal = match self.method {
                NonbondedMethod::Ewald => {
                    assert!(
                        vectors[1][0] == 0.0 && vectors[2][0] == 0.0 && vectors[2][1] == 0.0,
                        "Ewald requires a rectangular periodic box; use PME for triclinic boxes"
                    );
                    Some(ewald::parameters(
                        std::array::from_fn(|axis| vectors[axis][axis]),
                        self.cutoff_distance,
                        self.ewald_error_tolerance,
                    ))
                }
                NonbondedMethod::PME | NonbondedMethod::LJPME => {
                    Some(self.effective_pme_parameters(vectors))
                }
                _ => None,
            };
            if self.method == NonbondedMethod::LJPME {
                self.initialized_ljpme = Some(self.effective_ljpme_parameters(vectors));
            }
        }
    }

    fn global_parameters(&self) -> HashMap<String, f64> {
        self.global_parameters.iter().cloned().collect()
    }

    fn set_global_parameter(&mut self, name: &str, value: f64) -> bool {
        if let Some(current) = self.parameter_values.get_mut(name) {
            if *current != value {
                *current = value;
                self.effective_parameters.take();
            }
            true
        } else {
            false
        }
    }

    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        box_vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64) {
        self.evaluate(positions, box_vectors, self.include_direct_space, true)
    }

    fn calc_forces_and_energy_groups(
        &self,
        positions: &Tensor<B, 2>,
        box_vectors: Option<&[[f64; 3]; 3]>,
        groups: u32,
    ) -> Option<(Tensor<B, 2>, f64)> {
        let direct = self.include_direct_space && groups & (1 << self.force_group()) != 0;
        let reciprocal_group = if self.reciprocal_force_group < 0 {
            self.force_group()
        } else {
            self.reciprocal_force_group as u32
        };
        let reciprocal = matches!(
            self.method,
            NonbondedMethod::Ewald | NonbondedMethod::PME | NonbondedMethod::LJPME
        ) && groups & (1 << reciprocal_group) != 0;
        (direct || reciprocal).then(|| self.evaluate(positions, box_vectors, direct, reciprocal))
    }

    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.uses_periodic_boundary_conditions()
    }
}

/// Positive complement of the screened attractive dispersion interaction.
fn dispersion_reciprocal_pair(c6: f64, r2: f64, alpha: f64) -> (f64, f64) {
    if c6 == 0.0 {
        return (0.0, 0.0);
    }
    let x = alpha * alpha * r2;
    if x < 0.01 {
        let mut term = 1.0 / 6.0;
        let mut value = term;
        let mut derivative = 0.0;
        // (1-exp(-x)*(1+x+x^2/2))/x^3, avoiding cancellation at small r.
        for k in 1..=12 {
            term *= -x * (k + 2) as f64 / ((k + 3) * k) as f64;
            value += term;
            derivative += if x == 0.0 {
                if k == 1 {
                    -1.0 / 8.0
                } else {
                    0.0
                }
            } else {
                k as f64 * term / x
            };
        }
        return (
            c6 * alpha.powi(6) * value,
            2.0 * c6 * alpha.powi(8) * derivative,
        );
    }
    let exponential = (-x).exp();
    let energy = c6 / r2.powi(3) * (1.0 - exponential * (1.0 + x + 0.5 * x * x));
    let derivative =
        -6.0 * c6 / r2.powi(4) * (1.0 - exponential * (1.0 + x + 0.5 * x * x + x.powi(3) / 6.0));
    (energy, derivative)
}

/// Compute energy and force/r for an exception pair (pre-combined params).
fn lj_coulomb_exception(exc: &ExceptionParam, r: f64, r2: f64) -> (f64, f64) {
    let (e_lj, f_lj) = lj_pair(exc.sigma, exc.epsilon, r, r2);
    let (e_c, f_c) = coulomb_pair(exc.charge_prod, r, r2);
    (e_lj + e_c, f_lj + f_c)
}

/// LJ 12-6: E = 4*eps*[(sigma/r)^12 - (sigma/r)^6]
/// Returns (E, dE/dr / r).
#[inline]
fn lj_pair(sigma: f64, epsilon: f64, _r: f64, r2: f64) -> (f64, f64) {
    if epsilon == 0.0 {
        return (0.0, 0.0);
    }
    let sr2 = (sigma * sigma) / r2;
    let sr6 = sr2 * sr2 * sr2;
    let sr12 = sr6 * sr6;
    let e = 4.0 * epsilon * (sr12 - sr6);
    // dE/dr = 4*eps*(-12*sigma^12/r^13 + 6*sigma^6/r^7) = 4*eps*(- 12*sr12 + 6*sr6)/r
    let de_dr_over_r = 4.0 * epsilon * (-12.0 * sr12 + 6.0 * sr6) / r2;
    (e, de_dr_over_r)
}

/// Bare Coulomb: E = ONE_4PI_EPS0 * q1*q2 / r
/// Returns (E, dE/dr / r).
#[inline]
fn coulomb_pair(charge_prod: f64, r: f64, r2: f64) -> (f64, f64) {
    if charge_prod == 0.0 {
        return (0.0, 0.0);
    }
    let e = ONE_4PI_EPS0 * charge_prod / r;
    let de_dr_over_r = -ONE_4PI_EPS0 * charge_prod / (r2 * r);
    (e, de_dr_over_r)
}

/// Reaction-field Coulomb (Tironi et al.):
/// E = ONE_4PI_EPS0 * q1*q2 * [1/r + k_rf*r^2 - c_rf]
/// k_rf = (eps_rf - 1)/(2*eps_rf + 1) / rc^3
/// c_rf = (3*eps_rf)/(2*eps_rf + 1) / rc
#[inline]
fn reaction_field(charge_prod: f64, r: f64, r2: f64, rc: f64, eps_rf: f64) -> (f64, f64) {
    if charge_prod == 0.0 {
        return (0.0, 0.0);
    }
    let rc3 = rc * rc * rc;
    let (k_rf, c_rf) = if eps_rf.is_infinite() {
        (0.5 / rc3, 1.5 / rc)
    } else {
        (
            (eps_rf - 1.0) / (2.0 * eps_rf + 1.0) / rc3,
            (3.0 * eps_rf) / (2.0 * eps_rf + 1.0) / rc,
        )
    };
    let e = ONE_4PI_EPS0 * charge_prod * (1.0 / r + k_rf * r2 - c_rf);
    // dE/dr = ONE_4PI_EPS0 * q1*q2 * (-1/r^2 + 2*k_rf*r)
    let de_dr_over_r = ONE_4PI_EPS0 * charge_prod * (-1.0 / (r2 * r) + 2.0 * k_rf);
    (e, de_dr_over_r)
}

fn validate_particle_parameters(charge: f64, sigma: f64, epsilon: f64) {
    assert!(
        charge.is_finite()
            && sigma.is_finite()
            && epsilon.is_finite()
            && sigma >= 0.0
            && epsilon >= 0.0,
        "nonbonded parameters must be finite with nonnegative sigma and epsilon"
    );
}

fn screened_coulomb(charge: f64, r: f64, r2: f64, alpha: f64) -> (f64, f64) {
    if charge == 0.0 {
        return (0.0, 0.0);
    }
    let erfc = libm::erfc(alpha * r);
    let coefficient = ONE_4PI_EPS0 * charge;
    (
        coefficient * erfc / r,
        -coefficient * (erfc / r + 2.0 * alpha / PI.sqrt() * (-alpha * alpha * r2).exp()) / r2,
    )
}

fn reciprocal_pair(charge: f64, r: f64, r2: f64, alpha: f64) -> (f64, f64) {
    let coefficient = ONE_4PI_EPS0 * charge;
    if alpha * r < 1e-6 {
        return (
            coefficient * 2.0 * alpha / PI.sqrt() * (1.0 - alpha * alpha * r2 / 3.0),
            -coefficient * 4.0 * alpha.powi(3) / (3.0 * PI.sqrt()),
        );
    }
    let erf = libm::erf(alpha * r);
    (
        coefficient * erf / r,
        coefficient * (2.0 * alpha / PI.sqrt() * (-alpha * alpha * r2).exp() - erf / r) / r2,
    )
}

fn integrate(function: &impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    fn recurse(
        f: &impl Fn(f64) -> f64,
        interval: [f64; 2],
        values: [f64; 3],
        estimate: f64,
        tolerance: f64,
        depth: u32,
    ) -> f64 {
        let [a, b] = interval;
        let [fa, fm, fb] = values;
        let midpoint = 0.5 * (a + b);
        let fl = f(0.5 * (a + midpoint));
        let fr = f(0.5 * (midpoint + b));
        let left = (midpoint - a) / 6.0 * (fa + 4.0 * fl + fm);
        let right = (b - midpoint) / 6.0 * (fm + 4.0 * fr + fb);
        let difference = left + right - estimate;
        if depth == 0 || difference.abs() <= 15.0 * tolerance {
            return left + right + difference / 15.0;
        }
        recurse(
            f,
            [a, midpoint],
            [fa, fl, fm],
            left,
            0.5 * tolerance,
            depth - 1,
        ) + recurse(
            f,
            [midpoint, b],
            [fm, fr, fb],
            right,
            0.5 * tolerance,
            depth - 1,
        )
    }
    let values = [function(a), function(0.5 * (a + b)), function(b)];
    let estimate = (b - a) / 6.0 * (values[0] + 4.0 * values[1] + values[2]);
    recurse(
        function,
        [a, b],
        values,
        estimate,
        1e-11 * estimate.abs().max(1e-12),
        20,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use burn::backend::NdArray;

    type B = NdArray;

    fn pos_tensor(data: &[[f64; 3]]) -> Tensor<B, 2> {
        let flat: Vec<f32> = data
            .iter()
            .flat_map(|p| p.iter().map(|&v| v as f32))
            .collect();
        let n = data.len();
        Tensor::<B, 1>::from_floats(flat.as_slice(), &Default::default()).reshape([n, 3])
    }

    fn get_forces(t: Tensor<B, 2>) -> Vec<f64> {
        t.to_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(|v| v as f64)
            .collect()
    }

    #[test]
    fn reciprocal_methods_match_energy_gradients() {
        let coordinates = [[0.1, 0.2, 0.3], [0.6, 0.4, 0.2], [2.8, 0.7, 0.4]];
        for method in [
            NonbondedMethod::Ewald,
            NonbondedMethod::PME,
            NonbondedMethod::LJPME,
        ] {
            let mut force = NonbondedForce::new();
            force.set_nonbonded_method(method);
            force.set_cutoff_distance(1.2);
            force.set_ewald_error_tolerance(1e-5);
            force.add_particle(0.4, 0.3, 0.5);
            force.add_particle(-0.3, 0.32, 0.6);
            force.add_particle(0.1, 0.34, 0.7);
            force.add_exception(0, 2, -0.02, 0.32, 0.4);
            force.set_exceptions_use_periodic_boundary_conditions(true);
            let skew = if method == NonbondedMethod::Ewald {
                0.0
            } else {
                0.3
            };
            let vectors = [[3.0, 0.0, 0.0], [skew, 3.0, 0.0], [-skew, skew, 3.0]];
            let (forces, _) =
                force.calc_forces_and_energy(&pos_tensor(&coordinates), Some(&vectors));
            let forces = get_forces(forces);
            for particle in 0..3 {
                for axis in 0..3 {
                    let mut plus = coordinates;
                    let mut minus = coordinates;
                    plus[particle][axis] = (plus[particle][axis] as f32 + 0.0005) as f64;
                    minus[particle][axis] = (minus[particle][axis] as f32 - 0.0005) as f64;
                    let e_plus = force
                        .calc_forces_and_energy(&pos_tensor(&plus), Some(&vectors))
                        .1;
                    let e_minus = force
                        .calc_forces_and_energy(&pos_tensor(&minus), Some(&vectors))
                        .1;
                    let difference =
                        -(e_plus - e_minus) / (plus[particle][axis] - minus[particle][axis]);
                    assert_relative_eq!(
                        forces[3 * particle + axis],
                        difference,
                        epsilon = 1e-3,
                        max_relative = 1e-4
                    );
                }
            }
        }
    }

    #[test]
    fn dispersion_complement_is_finite_at_overlap() {
        let alpha: f64 = 2.3;
        let c6 = 0.04;
        let (energy, derivative) = dispersion_reciprocal_pair(c6, 0.0, alpha);
        assert_relative_eq!(energy, c6 * alpha.powi(6) / 6.0, epsilon = 1e-14);
        assert_relative_eq!(derivative, -c6 * alpha.powi(8) / 4.0, epsilon = 1e-14);
        for r in [1e-5_f64, 0.02, 0.1, 0.7] {
            let h = r * 1e-4;
            let plus = dispersion_reciprocal_pair(c6, (r + h).powi(2), alpha).0;
            let minus = dispersion_reciprocal_pair(c6, (r - h).powi(2), alpha).0;
            let derivative = dispersion_reciprocal_pair(c6, r * r, alpha).1;
            assert_relative_eq!(
                (plus - minus) / (2.0 * h),
                derivative * r,
                epsilon = 1e-6,
                max_relative = 1e-4
            );
        }
    }

    #[test]
    fn parameter_offsets_update_and_invalidate_cached_values() {
        let mut force = NonbondedForce::new();
        force.add_particle(0.2, 0.3, 0.4);
        force.add_particle(-0.3, 0.35, 0.5);
        force.add_global_parameter("lambda", 0.25);
        force.add_particle_parameter_offset("lambda", 0, 0.4, 0.02, 0.2);
        let positions = pos_tensor(&[[0.1, 0.2, 0.3], [0.6, 0.4, 0.2]]);
        for value in [0.25, 0.7, -0.2, 0.25] {
            Force::<B>::set_global_parameter(&mut force, "lambda", value);
            let (actual_forces, actual_energy) = force.calc_forces_and_energy(&positions, None);
            let mut explicit = NonbondedForce::new();
            explicit.add_particle(0.2 + 0.4 * value, 0.3 + 0.02 * value, 0.4 + 0.2 * value);
            explicit.add_particle(-0.3, 0.35, 0.5);
            let (expected_forces, expected_energy) =
                explicit.calc_forces_and_energy(&positions, None);
            assert_eq!(actual_energy, expected_energy);
            assert_eq!(get_forces(actual_forces), get_forces(expected_forces));
        }
    }

    // Mirrors TestNonbondedForce.h::testCoulomb()
    #[test]
    fn test_coulomb() {
        let mut force = NonbondedForce::new();
        force.add_particle(0.5, 1.0, 0.0);
        force.add_particle(-1.5, 1.0, 0.0);

        let positions = pos_tensor(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        let (forces_t, energy) = Force::<B>::calc_forces_and_energy(&force, &positions, None);
        let f = get_forces(forces_t);

        let expected_f = -ONE_4PI_EPS0 * (-0.75) / 4.0;
        assert_relative_eq!(f[0], expected_f, epsilon = 1e-4);
        assert_relative_eq!(f[3], -expected_f, epsilon = 1e-4);
        assert_relative_eq!(energy, ONE_4PI_EPS0 * (-0.75) / 2.0, epsilon = 1e-4);
    }

    // Mirrors TestNonbondedForce.h::testLJ()
    #[test]
    fn test_lj() {
        let mut force = NonbondedForce::new();
        force.add_particle(0.0, 1.2, 1.0);
        force.add_particle(0.0, 1.4, 2.0);

        let positions = pos_tensor(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        let (forces_t, energy) = Force::<B>::calc_forces_and_energy(&force, &positions, None);
        let f = get_forces(forces_t);

        let sigma = 1.3_f64; // (1.2+1.4)/2
        let eps = 2_f64.sqrt(); // sqrt(1*2)
        let x = sigma / 2.0;
        let expected_e = 4.0 * eps * (x.powi(12) - x.powi(6));
        let expected_f = 4.0 * eps * (-12.0 * x.powi(12) + 6.0 * x.powi(6)) / 2.0;

        assert_relative_eq!(energy, expected_e, epsilon = 1e-4);
        assert_relative_eq!(f[0], expected_f, epsilon = 1e-4);
        assert_relative_eq!(f[3], -expected_f, epsilon = 1e-4);
    }

    // Verify exclusions: excluded pairs contribute zero.
    #[test]
    fn test_exclusions() {
        let mut force = NonbondedForce::new();
        force.add_particle(1.0, 1.0, 1.0);
        force.add_particle(1.0, 1.0, 1.0);
        force.add_particle(1.0, 1.0, 1.0);
        // Exclude 0-1 and 0-2, keep 1-2.
        force.add_exception(0, 1, 0.0, 1.0, 0.0);
        force.add_exception(0, 2, 0.0, 1.0, 0.0);

        let positions = pos_tensor(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        let (forces_t, _) = Force::<B>::calc_forces_and_energy(&force, &positions, None);
        let f = get_forces(forces_t);

        // Particle 0 has no interactions -> zero force.
        assert_relative_eq!(f[0], 0.0, epsilon = 1e-5);
        assert_relative_eq!(f[1], 0.0, epsilon = 1e-5);
        assert_relative_eq!(f[2], 0.0, epsilon = 1e-5);
    }

    // Newton's third law for a 3-particle system.
    #[test]
    fn test_conservation() {
        let mut force = NonbondedForce::new();
        force.add_particle(0.5, 0.3, 0.5);
        force.add_particle(-0.3, 0.4, 0.8);
        force.add_particle(0.1, 0.35, 0.3);

        let positions = pos_tensor(&[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.7, 0.0]]);
        let (forces_t, _) = Force::<B>::calc_forces_and_energy(&force, &positions, None);
        let f = get_forces(forces_t);

        for d in 0..3 {
            let total: f64 = f[d] + f[3 + d] + f[6 + d];
            assert_relative_eq!(total, 0.0, epsilon = 1e-4);
        }
    }
}
