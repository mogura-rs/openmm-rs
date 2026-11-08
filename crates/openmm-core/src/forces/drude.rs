use super::{
    harmonic_bond::apply_minimum_image_ortho, nonbonded::ONE_4PI_EPS0, Force, HostForceEvaluation,
};
use burn::prelude::*;
use std::collections::HashSet;
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc, Mutex,
};

#[derive(Clone, Debug, PartialEq)]
pub struct DrudeParticle {
    pub particle: usize,
    pub parent: usize,
    pub axis2: Option<usize>,
    pub axis3: Option<usize>,
    pub axis4: Option<usize>,
    pub charge: f64,
    pub polarizability: f64,
    pub aniso12: f64,
    pub aniso34: f64,
}
impl DrudeParticle {
    fn validate_parameters(&self) -> Result<(), String> {
        if self.polarizability <= 0. {
            return Err("polarizability must be positive".into());
        }
        if (self.axis2.is_some() && self.aniso12 <= 0.)
            || (self.axis3.is_some() && self.axis4.is_some() && self.aniso34 <= 0.)
        {
            return Err("active anisotropy factors must be positive".into());
        }
        Ok(())
    }
    fn indices(&self) -> (usize, usize, Option<usize>, Option<usize>, Option<usize>) {
        (
            self.particle,
            self.parent,
            self.axis2,
            self.axis3,
            self.axis4,
        )
    }
    fn constants(&self) -> [f64; 3] {
        let a1 = if self.axis2.is_some() {
            self.aniso12
        } else {
            1.
        };
        let a2 = if self.axis3.is_some() && self.axis4.is_some() {
            self.aniso34
        } else {
            1.
        };
        let k = ONE_4PI_EPS0 * self.charge * self.charge / self.polarizability;
        let k3 = k / (3. - a1 - a2);
        [k / a1 - k3, k / a2 - k3, k3]
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct DrudeScreenedPair {
    pub particle1: usize,
    pub particle2: usize,
    pub thole: f64,
}
#[derive(Default)]
struct Work {
    forces: Vec<f64>,
}
pub struct DrudeForce {
    metadata: super::ForceMetadata,
    group: Arc<AtomicU32>,
    particles: Vec<DrudeParticle>,
    pairs: Vec<DrudeScreenedPair>,
    periodic: bool,
    springs: Vec<[f64; 3]>,
    screened: Vec<[f64; 2]>,
    initialized: bool,
    workspace: Mutex<Work>,
}
impl Clone for DrudeForce {
    fn clone(&self) -> Self {
        Self {
            metadata: self.metadata.clone(),
            group: Arc::new(AtomicU32::new(self.force_group())),
            particles: self.particles.clone(),
            pairs: self.pairs.clone(),
            periodic: self.periodic,
            springs: self.springs.clone(),
            screened: self.screened.clone(),
            initialized: self.initialized,
            workspace: Mutex::default(),
        }
    }
}
impl Default for DrudeForce {
    fn default() -> Self {
        Self::new()
    }
}
impl DrudeForce {
    pub fn new() -> Self {
        Self {
            metadata: super::ForceMetadata::new("DrudeForce"),
            group: Arc::new(AtomicU32::new(0)),
            particles: Vec::new(),
            pairs: Vec::new(),
            periodic: false,
            springs: Vec::new(),
            screened: Vec::new(),
            initialized: false,
            workspace: Mutex::default(),
        }
    }
    pub fn snapshot_for_context(&self) -> Self {
        let mut force = self.clone();
        force.group = self.group.clone();
        force
    }
    pub fn force_group(&self) -> u32 {
        self.group.load(Ordering::Relaxed)
    }
    pub fn set_force_group(&mut self, group: u32) {
        assert!(group < 32, "force group must be between zero and 31");
        self.group.store(group, Ordering::Relaxed);
    }
    pub fn name(&self) -> &str {
        &self.metadata.name
    }
    pub fn set_name(&mut self, name: &str) {
        self.metadata.name = name.to_owned();
    }
    pub fn add_particle(&mut self, particle: DrudeParticle) -> Result<usize, String> {
        particle.validate_parameters()?;
        self.particles.push(particle);
        Ok(self.particles.len() - 1)
    }
    pub fn set_particle_parameters(
        &mut self,
        index: usize,
        particle: DrudeParticle,
    ) -> Result<(), String> {
        particle.validate_parameters()?;
        *self
            .particles
            .get_mut(index)
            .ok_or("Drude particle index out of range")? = particle;
        Ok(())
    }
    pub fn particle_parameters(&self, index: usize) -> &DrudeParticle {
        &self.particles[index]
    }
    pub fn num_particles(&self) -> usize {
        self.particles.len()
    }
    pub fn add_screened_pair(&mut self, particle1: usize, particle2: usize, thole: f64) -> usize {
        self.pairs.push(DrudeScreenedPair {
            particle1,
            particle2,
            thole,
        });
        self.pairs.len() - 1
    }
    pub fn set_screened_pair_parameters(
        &mut self,
        index: usize,
        particle1: usize,
        particle2: usize,
        thole: f64,
    ) {
        self.pairs[index] = DrudeScreenedPair {
            particle1,
            particle2,
            thole,
        };
    }
    pub fn screened_pair_parameters(&self, index: usize) -> &DrudeScreenedPair {
        &self.pairs[index]
    }
    pub fn num_screened_pairs(&self) -> usize {
        self.pairs.len()
    }
    pub fn set_uses_periodic_boundary_conditions(&mut self, periodic: bool) {
        self.periodic = periodic;
    }
    pub fn uses_periodic_boundary_conditions(&self) -> bool {
        self.periodic
    }
    fn rebuild(&mut self) {
        self.springs = self
            .particles
            .iter()
            .map(DrudeParticle::constants)
            .collect();
        self.screened = self
            .pairs
            .iter()
            .map(|pair| {
                let a = &self.particles[pair.particle1];
                let b = &self.particles[pair.particle2];
                [
                    ONE_4PI_EPS0 * a.charge * b.charge,
                    pair.thole / (a.polarizability * b.polarizability).powf(1. / 6.),
                ]
            })
            .collect();
    }
    pub fn update_parameters_from(&mut self, source: &Self) {
        assert_eq!(
            self.particles.len(),
            source.particles.len(),
            "Drude particle count cannot change"
        );
        assert_eq!(
            self.pairs.len(),
            source.pairs.len(),
            "screened pair count cannot change"
        );
        for (a, b) in self.particles.iter().zip(&source.particles) {
            assert_eq!(
                a.indices(),
                b.indices(),
                "Drude particle indices cannot change"
            );
        }
        for (a, b) in self.pairs.iter().zip(&source.pairs) {
            assert_eq!(
                (a.particle1, a.particle2),
                (b.particle1, b.particle2),
                "screened pair indices cannot change"
            );
        }
        self.particles.clone_from(&source.particles);
        self.pairs.clone_from(&source.pairs);
        self.rebuild();
    }
    fn evaluate(&self, pos: &[f64], vectors: Option<&[[f64; 3]; 3]>, forces: &mut [f64]) -> f64 {
        assert!(self.initialized, "force must be initialized");
        forces.fill(0.);
        let mut energy = 0.;
        for (p, &[k1, k2, k3]) in self.particles.iter().zip(&self.springs) {
            let delta = displacement(pos, p.particle, p.parent);
            energy += 0.5 * k3 * dot(delta, delta);
            add(forces, p.particle, delta, -k3);
            add(forces, p.parent, delta, k3);
            for (axis, coefficient) in [
                p.axis2.map(|j| (p.parent, j)).map(|v| (v, k1)),
                p.axis3.zip(p.axis4).map(|v| (v, k2)),
            ]
            .into_iter()
            .flatten()
            {
                let mut direction = displacement(pos, axis.0, axis.1);
                let length = dot(direction, direction).sqrt();
                assert!(length > 0., "Drude anisotropy axis has zero length");
                for v in &mut direction {
                    *v /= length;
                }
                let projection = dot(direction, delta);
                energy += 0.5 * coefficient * projection * projection;
                let scale = coefficient * projection;
                add(forces, p.particle, direction, -scale);
                add(forces, p.parent, direction, scale);
                let transverse = std::array::from_fn(|i| delta[i] - direction[i] * projection);
                add(forces, axis.0, transverse, -scale / length);
                add(forces, axis.1, transverse, scale / length);
            }
        }
        for (pair, &[charge_product, scale]) in self.pairs.iter().zip(&self.screened) {
            let a = &self.particles[pair.particle1];
            let b = &self.particles[pair.particle2];
            let particles = [[a.particle, a.parent], [b.particle, b.parent]];
            for j in 0..2 {
                for k in 0..2 {
                    let (a, b) = (particles[0][j], particles[1][k]);
                    let mut delta = displacement(pos, a, b);
                    if self.periodic {
                        apply_minimum_image_ortho(
                            &mut delta,
                            vectors.expect("periodic box is required"),
                        );
                    }
                    let r2 = dot(delta, delta);
                    let r = r2.sqrt();
                    assert!(r > 0., "screened Drude particles coincide");
                    let u = r * scale;
                    let exponential = (-u).exp();
                    let screening = 1. - (1. + 0.5 * u) * exponential;
                    let charge = charge_product * if j == k { 1. } else { -1. };
                    energy += charge * screening / r;
                    let factor =
                        charge / r2 * (screening / r - 0.5 * (1. + u) * exponential * scale);
                    add(forces, a, delta, factor);
                    add(forces, b, delta, -factor);
                }
            }
        }
        energy
    }
}
fn displacement(pos: &[f64], a: usize, b: usize) -> [f64; 3] {
    std::array::from_fn(|i| pos[3 * a + i] - pos[3 * b + i])
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter().zip(b).map(|(a, b)| a * b).sum()
}
fn add(force: &mut [f64], particle: usize, value: [f64; 3], scale: f64) {
    for i in 0..3 {
        force[3 * particle + i] += scale * value[i];
    }
}
impl<B: Backend> Force<B> for DrudeForce {
    fn force_group(&self) -> u32 {
        self.force_group()
    }
    fn has_dynamic_force_group(&self) -> bool {
        true
    }
    fn name(&self) -> &str {
        self.name()
    }
    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.periodic
    }
    fn bonded_particles(&self) -> Vec<(usize, usize)> {
        self.particles
            .iter()
            .map(|p| (p.particle, p.parent))
            .collect()
    }
    fn prepare_for_inner_context(&mut self) {
        self.periodic = false;
        self.group = Arc::new(AtomicU32::new(self.force_group()));
    }
    fn initialize(&mut self, n: usize, _vectors: Option<&[[f64; 3]; 3]>) {
        let mut used = HashSet::new();
        for p in &self.particles {
            for i in [p.particle, p.parent] {
                assert!(i < n, "Drude particle index out of range");
                assert!(
                    used.insert(i),
                    "particle is used by multiple Drude oscillators"
                );
            }
            for i in [p.axis2, p.axis3, p.axis4].into_iter().flatten() {
                assert!(i < n, "Drude orientation particle index out of range");
            }
        }
        let mut pairs = HashSet::new();
        for p in &self.pairs {
            assert!(
                p.particle1 < self.particles.len() && p.particle2 < self.particles.len(),
                "screened pair index out of range"
            );
            assert!(
                pairs.insert((p.particle1.min(p.particle2), p.particle1.max(p.particle2))),
                "duplicate screened pair"
            );
        }
        self.rebuild();
        self.initialized = true;
    }
    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64) {
        crate::tensor::with_f64_data(positions, |pos| {
            let mut force = vec![0.; pos.len()];
            let energy = self.evaluate(pos, vectors, &mut force);
            (
                crate::tensor::from_f64_vec(force, positions.dims(), &positions.device()),
                energy,
            )
        })
    }
    fn prepare_host_evaluation<'a>(
        &'a self,
        positions: &'a Tensor<B, 2>,
        vectors: Option<&'a [[f64; 3]; 3]>,
    ) -> Option<HostForceEvaluation<'a>> {
        crate::tensor::with_f64_data(positions, |pos| {
            let mut work = self.workspace.lock().unwrap();
            work.forces.resize(pos.len(), 0.);
            let energy = self.evaluate(pos, vectors, &mut work.forces);
            Some(HostForceEvaluation::new(energy, move |scale, forces| {
                for (out, &value) in forces.iter_mut().zip(&work.forces) {
                    *out += scale * value;
                }
            }))
        })
    }
}
