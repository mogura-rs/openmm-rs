use super::{
    harmonic_bond::apply_minimum_image_ortho, neighbors::NeighborCells, nonbonded::ONE_4PI_EPS0,
    Force, HostForceEvaluation,
};
use burn::prelude::*;
use std::sync::Mutex;

const DIELECTRIC_OFFSET: f64 = 0.009;
const PROBE_RADIUS: f64 = 0.14;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GBSAOBCMethod {
    #[default]
    NoCutoff,
    CutoffNonPeriodic,
    CutoffPeriodic,
}

#[derive(Debug, Clone, Copy)]
pub struct GBSAParticle {
    pub charge: f64,
    pub radius: f64,
    pub scaling_factor: f64,
}

#[derive(Default)]
struct Work {
    born: Vec<f64>,
    chain: Vec<f64>,
    derivative: Vec<f64>,
    forces: Vec<f64>,
}
#[derive(Default)]
struct Workspace(Mutex<Work>);
impl Clone for Workspace {
    fn clone(&self) -> Self {
        Self::default()
    }
}

#[derive(Clone)]
pub struct GBSAOBCForce {
    metadata: super::ForceMetadata,
    particles: Vec<GBSAParticle>,
    method: GBSAOBCMethod,
    cutoff: f64,
    solvent_dielectric: f64,
    solute_dielectric: f64,
    surface_area_energy: f64,
    neighbor_cells: NeighborCells,
    workspace: Workspace,
}

impl Default for GBSAOBCForce {
    fn default() -> Self {
        Self::new()
    }
}
impl GBSAOBCForce {
    super::force_metadata!();
    pub fn new() -> Self {
        Self {
            metadata: super::ForceMetadata::new("GBSAOBCForce"),
            particles: Vec::new(),
            method: GBSAOBCMethod::NoCutoff,
            cutoff: 1.0,
            solvent_dielectric: 78.3,
            solute_dielectric: 1.0,
            surface_area_energy: 2.25936,
            neighbor_cells: NeighborCells::default(),
            workspace: Workspace::default(),
        }
    }
    pub fn add_particle(&mut self, charge: f64, radius: f64, scaling_factor: f64) -> usize {
        self.particles.push(GBSAParticle {
            charge,
            radius,
            scaling_factor,
        });
        self.particles.len() - 1
    }
    pub fn num_particles(&self) -> usize {
        self.particles.len()
    }
    pub fn particle_parameters(&self, index: usize) -> &GBSAParticle {
        &self.particles[index]
    }
    pub fn set_particle_parameters(
        &mut self,
        index: usize,
        charge: f64,
        radius: f64,
        scaling_factor: f64,
    ) {
        self.particles[index] = GBSAParticle {
            charge,
            radius,
            scaling_factor,
        };
    }
    pub fn nonbonded_method(&self) -> GBSAOBCMethod {
        self.method
    }
    pub fn set_nonbonded_method(&mut self, method: GBSAOBCMethod) {
        self.method = method;
    }
    pub fn cutoff_distance(&self) -> f64 {
        self.cutoff
    }
    pub fn set_cutoff_distance(&mut self, value: f64) {
        self.cutoff = value;
    }
    pub fn solvent_dielectric(&self) -> f64 {
        self.solvent_dielectric
    }
    pub fn set_solvent_dielectric(&mut self, value: f64) {
        self.solvent_dielectric = value;
    }
    pub fn solute_dielectric(&self) -> f64 {
        self.solute_dielectric
    }
    pub fn set_solute_dielectric(&mut self, value: f64) {
        self.solute_dielectric = value;
    }
    pub fn surface_area_energy(&self) -> f64 {
        self.surface_area_energy
    }
    pub fn set_surface_area_energy(&mut self, value: f64) {
        self.surface_area_energy = value;
    }
    pub fn uses_periodic_boundary_conditions(&self) -> bool {
        self.method == GBSAOBCMethod::CutoffPeriodic
    }
    pub fn update_parameters_from(&mut self, source: &Self) {
        assert_eq!(
            self.particles.len(),
            source.particles.len(),
            "GBSA particle count cannot change in a Context"
        );
        self.particles.clone_from(&source.particles);
    }
    fn check_box(&self, vectors: Option<&[[f64; 3]; 3]>) {
        if self.uses_periodic_boundary_conditions() {
            let vectors = vectors.expect("periodic GBSA requires box vectors");
            assert!(
                (0..3).all(|i| vectors[i][i] >= 2.0 * self.cutoff),
                "GBSA cutoff exceeds half a periodic box width"
            );
        }
    }
    fn visit_pairs(
        &self,
        positions: &[f64],
        vectors: Option<&[[f64; 3]; 3]>,
        mut pair: impl FnMut(usize, usize, [f64; 3], f64),
    ) {
        let vectors = if self.uses_periodic_boundary_conditions() {
            vectors
        } else {
            None
        };
        let mut visit = |i: usize, j: usize| {
            let mut delta =
                std::array::from_fn(|axis| positions[3 * j + axis] - positions[3 * i + axis]);
            if let Some(vectors) = vectors {
                apply_minimum_image_ortho(&mut delta, vectors);
            }
            let distance = delta.iter().map(|v| v * v).sum::<f64>().sqrt();
            if self.method == GBSAOBCMethod::NoCutoff || distance <= self.cutoff {
                pair(i, j, delta, distance);
            }
        };
        if self.method == GBSAOBCMethod::NoCutoff {
            for i in 0..self.particles.len() {
                for j in i + 1..self.particles.len() {
                    visit(i, j);
                }
            }
        } else {
            self.neighbor_cells
                .visit(positions, self.cutoff, vectors, visit);
        }
    }
    fn evaluate(
        &self,
        positions: &[f64],
        vectors: Option<&[[f64; 3]; 3]>,
        work: &mut Work,
        forces: &mut [f64],
    ) -> f64 {
        self.check_box(vectors);
        let n = self.particles.len();
        assert_eq!(
            positions.len(),
            3 * n,
            "GBSA particle count must match positions"
        );
        let Work {
            born,
            chain,
            derivative,
            ..
        } = work;
        born.resize(n, 0.0);
        born.fill(0.0);
        chain.resize(n, 0.0);
        derivative.resize(n, 0.0);
        forces.fill(0.0);
        self.visit_pairs(positions, vectors, |i, j, _, r| {
            born[i] += descreening(self.particles[i], self.particles[j], r);
            born[j] += descreening(self.particles[j], self.particles[i], r);
        });
        let prefactor = if self.solute_dielectric != 0.0 && self.solvent_dielectric != 0.0 {
            -ONE_4PI_EPS0 * (self.solute_dielectric.recip() - self.solvent_dielectric.recip())
        } else {
            0.0
        };
        let surface = 4.0 * std::f64::consts::PI * self.surface_area_energy;
        let mut energy = 0.0;
        for i in 0..n {
            let particle = self.particles[i];
            let offset = particle.radius - DIELECTRIC_OFFSET;
            let psi = 0.5 * offset * born[i];
            let t = (psi - 0.8 * psi * psi + 4.85 * psi * psi * psi).tanh();
            born[i] = (offset.recip() - t / particle.radius).recip();
            chain[i] =
                offset * (1.0 - 1.6 * psi + 14.55 * psi * psi) * (1.0 - t * t) / particle.radius;
            let self_energy = 0.5 * prefactor * particle.charge * particle.charge / born[i].abs();
            energy += self_energy;
            derivative[i] = -self_energy / born[i];
            if born[i] > 0.0 {
                let area = surface
                    * (particle.radius + PROBE_RADIUS).powi(2)
                    * (particle.radius / born[i]).powi(6);
                energy += area;
                derivative[i] -= 6.0 * area / born[i];
            }
        }
        self.visit_pairs(positions, vectors, |i, j, delta, r| {
            let product = born[i] * born[j];
            let d = r * r / (4.0 * product);
            let exponential = (-d).exp();
            let denominator2 = r * r + product * exponential;
            let coefficient = prefactor * self.particles[i].charge * self.particles[j].charge;
            let pair_energy = coefficient / denominator2.sqrt();
            energy += if self.method == GBSAOBCMethod::NoCutoff {
                pair_energy
            } else {
                pair_energy - coefficient / self.cutoff
            };
            let radial = -pair_energy * (1.0 - 0.25 * exponential) / denominator2;
            let born_derivative = -0.5 * pair_energy * exponential * (1.0 + d) / denominator2;
            derivative[i] += born_derivative * born[j];
            derivative[j] += born_derivative * born[i];
            for axis in 0..3 {
                let f = radial * delta[axis];
                forces[3 * i + axis] += f;
                forces[3 * j + axis] -= f;
            }
        });
        for i in 0..n {
            derivative[i] *= born[i] * born[i] * chain[i];
        }
        self.visit_pairs(positions, vectors, |i, j, delta, r| {
            let radial = -(derivative[i]
                * descreening_gradient(self.particles[i], self.particles[j], r)
                + derivative[j] * descreening_gradient(self.particles[j], self.particles[i], r));
            for axis in 0..3 {
                let f = radial * delta[axis];
                forces[3 * i + axis] += f;
                forces[3 * j + axis] -= f;
            }
        });
        energy
    }
}

fn descreening(i: GBSAParticle, j: GBSAParticle, r: f64) -> f64 {
    let radius = i.radius - DIELECTRIC_OFFSET;
    let scaled = (j.radius - DIELECTRIC_OFFSET) * j.scaling_factor;
    if radius >= r + scaled {
        return 0.0;
    }
    let lower = radius.max((r - scaled).abs()).recip();
    let upper = (r + scaled).recip();
    let inverse = r.recip();
    let mut value = lower - upper
        + 0.25 * r * (upper * upper - lower * lower)
        + 0.5 * inverse * (upper / lower).ln()
        + 0.25 * scaled * scaled * inverse * (lower * lower - upper * upper);
    if radius < scaled - r {
        value += 2.0 * (radius.recip() - lower);
    }
    value
}

fn descreening_gradient(i: GBSAParticle, j: GBSAParticle, r: f64) -> f64 {
    let radius = i.radius - DIELECTRIC_OFFSET;
    let scaled = (j.radius - DIELECTRIC_OFFSET) * j.scaling_factor;
    if radius >= r + scaled {
        return 0.0;
    }
    let lower = radius.max((r - scaled).abs()).recip();
    let upper = (r + scaled).recip();
    let inverse = r.recip();
    (0.125 * (1.0 + scaled * scaled * inverse * inverse) * (lower * lower - upper * upper)
        + 0.25 * (upper / lower).ln() * inverse * inverse)
        * inverse
}

impl<B: Backend> Force<B> for GBSAOBCForce {
    super::force_metadata_impl!();
    fn initialize(&mut self, n: usize, vectors: Option<&[[f64; 3]; 3]>) {
        assert_eq!(
            self.particles.len(),
            n,
            "GBSA particle count must match System"
        );
        for particle in &self.particles {
            assert!(particle.charge.is_finite(), "GBSA charges must be finite");
            assert!(
                particle.radius.is_finite() && particle.radius > 0.0,
                "GBSA radii must be positive and finite"
            );
            assert!(
                particle.scaling_factor.is_finite() && particle.scaling_factor > 0.0,
                "GBSA scaling factors must be positive and finite"
            );
        }
        assert!(
            self.solvent_dielectric.is_finite()
                && self.solute_dielectric.is_finite()
                && self.surface_area_energy.is_finite(),
            "GBSA global settings must be finite"
        );
        if self.method != GBSAOBCMethod::NoCutoff {
            assert!(
                self.cutoff.is_finite() && self.cutoff > 0.0,
                "GBSA cutoff must be positive and finite"
            );
        }
        self.check_box(vectors);
    }
    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.uses_periodic_boundary_conditions()
    }
    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64) {
        let mut work = self.workspace.0.lock().unwrap();
        let mut forces = vec![0.0; 3 * self.particles.len()];
        let energy = crate::tensor::with_f64_data(positions, |p| {
            self.evaluate(p, vectors, &mut work, &mut forces)
        });
        (
            crate::tensor::from_f64_vec(forces, positions.dims(), &positions.device()),
            energy,
        )
    }
    fn prepare_host_evaluation<'a>(
        &'a self,
        positions: &'a Tensor<B, 2>,
        vectors: Option<&'a [[f64; 3]; 3]>,
    ) -> Option<HostForceEvaluation<'a>> {
        let mut work = self.workspace.0.lock().unwrap();
        let mut forces = std::mem::take(&mut work.forces);
        forces.resize(3 * self.particles.len(), 0.0);
        let energy = crate::tensor::with_f64_data(positions, |p| {
            self.evaluate(p, vectors, &mut work, &mut forces)
        });
        work.forces = forces;
        Some(HostForceEvaluation::new(
            energy,
            move |coefficient, forces| {
                for (out, value) in forces.iter_mut().zip(&work.forces) {
                    *out += coefficient * value;
                }
            },
        ))
    }
}
