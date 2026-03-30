// NonbondedForce: LJ + Coulomb with exclusions/exceptions.
// Methods: NoCutoff, CutoffNonPeriodic, CutoffPeriodic, PME (stub).
// Reference test: openmm/tests/TestNonbondedForce.h
// Reference impl: openmm/platforms/reference/src/SimTKReference/ReferenceNonbondedIxn.cpp
use super::Force;
use crate::forces::harmonic_bond::apply_minimum_image_ortho;
use burn::prelude::*;

/// 1/(4*pi*eps0) in kJ/mol * nm / e^2.
pub const ONE_4PI_EPS0: f64 = 138.935456;

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
        }
    }

    /// Add a particle. Returns particle index.
    pub fn add_particle(&mut self, charge: f64, sigma: f64, epsilon: f64) -> usize {
        self.particles.push(ParticleParam {
            charge,
            sigma,
            epsilon,
        });
        self.particles.len() - 1
    }

    pub fn set_particle_parameters(&mut self, index: usize, charge: f64, sigma: f64, epsilon: f64) {
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
        // Add 1-4 scaled interactions.
        for (i, j) in &pairs_14 {
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
        self.cutoff_distance = cutoff;
    }

    pub fn cutoff_distance(&self) -> f64 {
        self.cutoff_distance
    }

    pub fn set_reaction_field_dielectric(&mut self, eps: f64) {
        self.reaction_field_dielectric = eps;
    }

    pub fn reaction_field_dielectric(&self) -> f64 {
        self.reaction_field_dielectric
    }

    pub fn set_ewald_error_tolerance(&mut self, tol: f64) {
        self.ewald_error_tolerance = tol;
    }

    pub fn ewald_error_tolerance(&self) -> f64 {
        self.ewald_error_tolerance
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
        let pos: Vec<f64> = positions
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(|v| v as f64)
            .collect();

        let mut forces = vec![0.0f64; n * 3];
        let mut energy = 0.0f64;

        // Build exclusion set for O(1) lookup.
        let excluded: std::collections::HashSet<(usize, usize)> = self
            .exceptions
            .iter()
            .map(|e| {
                let (a, b) = (e.particle1, e.particle2);
                if a < b {
                    (a, b)
                } else {
                    (b, a)
                }
            })
            .collect();

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

        for i in 0..n {
            for j in (i + 1)..n {
                let key = (i, j);
                if excluded.contains(&key) {
                    continue;
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
                    continue;
                }

                let r = r2.sqrt();
                let (e, f_over_r) = lj_coulomb_pair(
                    &self.particles[i],
                    &self.particles[j],
                    r,
                    r2,
                    self.method,
                    self.cutoff_distance,
                    self.reaction_field_dielectric,
                );

                energy += e;
                for d in 0..3 {
                    let f = f_over_r * delta[d];
                    forces[i * 3 + d] -= f;
                    forces[j * 3 + d] += f;
                }
            }
        }

        // Exceptions (exclusions have zero params; 1-4s have scaled params).
        for exc in &self.exceptions {
            let i = exc.particle1;
            let j = exc.particle2;

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
            let r = r2.sqrt();

            let (e, f_over_r) = lj_coulomb_exception(exc, r, r2);
            energy += e;
            for d in 0..3 {
                let f = f_over_r * delta[d];
                forces[i * 3 + d] -= f;
                forces[j * 3 + d] += f;
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

/// Compute LJ + Coulomb energy and force/r for a standard particle pair.
/// Returns (energy, dE/dr / r) so that F_i = -(dE/dr / r) * delta.
fn lj_coulomb_pair(
    pi: &ParticleParam,
    pj: &ParticleParam,
    r: f64,
    r2: f64,
    method: NonbondedMethod,
    cutoff: f64,
    rf_dielectric: f64,
) -> (f64, f64) {
    // Lorentz-Berthelot combining rules.
    let sigma = 0.5 * (pi.sigma + pj.sigma);
    let epsilon = (pi.epsilon * pj.epsilon).sqrt();
    let charge_prod = pi.charge * pj.charge;

    let (e_lj, f_lj_over_r) = lj_pair(sigma, epsilon, r, r2);
    let (e_coul, f_coul_over_r) = match method {
        NonbondedMethod::CutoffNonPeriodic => {
            reaction_field(charge_prod, r, r2, cutoff, rf_dielectric)
        }
        _ => coulomb_pair(charge_prod, r, r2),
    };

    (e_lj + e_coul, f_lj_over_r + f_coul_over_r)
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
    let k_rf = (eps_rf - 1.0) / (2.0 * eps_rf + 1.0) / rc3;
    let c_rf = (3.0 * eps_rf) / (2.0 * eps_rf + 1.0) / rc;
    let e = ONE_4PI_EPS0 * charge_prod * (1.0 / r + k_rf * r2 - c_rf);
    // dE/dr = ONE_4PI_EPS0 * q1*q2 * (-1/r^2 + 2*k_rf*r)
    let de_dr_over_r = ONE_4PI_EPS0 * charge_prod * (-1.0 / (r2 * r) + 2.0 * k_rf / r);
    (e, de_dr_over_r)
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

        let expected_f = ONE_4PI_EPS0 * (-0.75) / 4.0;
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
        let expected_f = 4.0 * eps * (12.0 * x.powi(12) - 6.0 * x.powi(6)) / 2.0;

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
