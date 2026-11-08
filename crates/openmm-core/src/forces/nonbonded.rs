// NonbondedForce: LJ + Coulomb with exclusions/exceptions.
// Methods: NoCutoff, CutoffNonPeriodic, CutoffPeriodic, PME.
// Reference test: openmm/tests/TestNonbondedForce.h
// Reference impl: openmm/platforms/reference/src/SimTKReference/ReferenceNonbondedIxn.cpp
use super::neighbors::NeighborCells;
use super::pme::{reciprocal_box, PmeCache};
use super::Force;
use crate::forces::harmonic_bond::apply_minimum_image_ortho;
use burn::prelude::*;
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
    /// Particle Mesh Ewald (periodic).
    PME,
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

#[derive(Clone)]
pub struct NonbondedForce {
    particles: Vec<ParticleParam>,
    exceptions: Vec<ExceptionParam>,
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
    dispersion_coefficient: OnceLock<f64>,
    exclusions: OnceLock<std::collections::HashSet<(usize, usize)>>,
    neighbor_cells: NeighborCells,
    use_neighbor_cells: bool,
}

impl NonbondedForce {
    pub fn new() -> Self {
        Self {
            particles: Vec::new(),
            exceptions: Vec::new(),
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
            NonbondedMethod::CutoffPeriodic | NonbondedMethod::PME
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
            let mut moments = [0.0; 13];
            let mut diagonal = [0.0; 13];
            for particle in &self.particles {
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
        self.particles = source.particles.clone();
        self.exceptions = source.exceptions.clone();
        self.exclusions.take();
        self.dispersion_coefficient.take();
    }
}

impl Default for NonbondedForce {
    fn default() -> Self {
        Self::new()
    }
}

// --- Force computation ---

impl<B: Backend> Force<B> for NonbondedForce {
    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        box_vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64) {
        let device = positions.device();
        let n = positions.dims()[0];
        assert_eq!(
            n,
            self.particles.len(),
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
            self.exceptions
                .iter()
                .map(|e| (e.particle1.min(e.particle2), e.particle1.max(e.particle2)))
                .collect()
        });

        // Direct space: all pairs (respecting cutoff and exclusions).
        let use_cutoff = matches!(
            self.method,
            NonbondedMethod::CutoffNonPeriodic
                | NonbondedMethod::CutoffPeriodic
                | NonbondedMethod::PME
        );
        let use_pbc = matches!(
            self.method,
            NonbondedMethod::CutoffPeriodic | NonbondedMethod::PME
        );
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
        let pme = if self.method == NonbondedMethod::PME {
            Some(self.effective_pme_parameters(box_vectors.unwrap()))
        } else {
            None
        };

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
            let pi = &self.particles[i];
            let pj = &self.particles[j];
            let (mut lj_energy, mut lj_force) = lj_pair(
                0.5 * (pi.sigma + pj.sigma),
                (pi.epsilon * pj.epsilon).sqrt(),
                r,
                r2,
            );
            if use_cutoff && self.use_switching_function && r > self.switching_distance {
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

        if let Some((alpha, dimensions)) = pme {
            let vectors = box_vectors.unwrap();
            let (_, volume) = reciprocal_box(vectors);
            energy += self.pme_cache.evaluate(
                &pos,
                &self.particles,
                vectors,
                alpha,
                dimensions,
                &mut forces,
            );
            let charge_squared: f64 = self
                .particles
                .iter()
                .map(|particle| particle.charge.powi(2))
                .sum();
            let total_charge: f64 = self.particles.iter().map(|particle| particle.charge).sum();
            energy -= ONE_4PI_EPS0
                * (alpha / PI.sqrt() * charge_squared
                    + PI * total_charge.powi(2) / (2.0 * alpha * alpha * volume));
        }
        if use_pbc && self.use_dispersion_correction {
            energy += self.dispersion_correction(reciprocal_box(box_vectors.unwrap()).1);
        }

        // Exceptions are untruncated and use their own periodicity setting.
        for exc in &self.exceptions {
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
                let charge = self.particles[i].charge * self.particles[j].charge;
                let (correction, derivative) = reciprocal_pair(charge, r, r2, alpha);
                e -= correction;
                f_over_r -= derivative;
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

    fn uses_periodic_boundary_conditions(&self) -> bool {
        matches!(
            self.method,
            NonbondedMethod::CutoffPeriodic | NonbondedMethod::PME
        )
    }
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
