use super::{
    harmonic_bond::apply_minimum_image_ortho, neighbors::NeighborCells, Force, HostForceEvaluation,
};
use burn::prelude::*;
use std::sync::Mutex;

#[derive(Debug, Clone, Copy)]
pub struct LCPOParticle {
    pub radius: f64,
    pub p1: f64,
    pub p2: f64,
    pub p3: f64,
    pub p4: f64,
}

struct Contact {
    particles: [usize; 2],
    area: [f64; 2],
    slope: [f64; 2],
}
impl Contact {
    fn other(&self, particle: usize) -> usize {
        if self.particles[0] == particle {
            self.particles[1]
        } else {
            self.particles[0]
        }
    }
}
#[derive(Default)]
struct Work {
    contacts: Vec<Contact>,
    offsets: Vec<usize>,
    cursors: Vec<usize>,
    neighbors: Vec<usize>,
    adjoints: Vec<[f64; 2]>,
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
pub struct LCPOForce {
    metadata: super::ForceMetadata,
    particles: Vec<LCPOParticle>,
    surface_tension: f64,
    periodic: bool,
    cells: NeighborCells,
    workspace: Workspace,
}
impl Default for LCPOForce {
    fn default() -> Self {
        Self::new()
    }
}
impl LCPOForce {
    super::force_metadata!();
    pub fn new() -> Self {
        Self {
            metadata: super::ForceMetadata::new("LCPOForce"),
            particles: Vec::new(),
            surface_tension: 1.0,
            periodic: false,
            cells: NeighborCells::default(),
            workspace: Workspace::default(),
        }
    }
    pub fn add_particle(&mut self, radius: f64, p1: f64, p2: f64, p3: f64, p4: f64) -> usize {
        self.particles.push(LCPOParticle {
            radius,
            p1,
            p2,
            p3,
            p4,
        });
        self.particles.len() - 1
    }
    pub fn num_particles(&self) -> usize {
        self.particles.len()
    }
    pub fn particle_parameters(&self, index: usize) -> &LCPOParticle {
        &self.particles[index]
    }
    pub fn set_particle_parameters(
        &mut self,
        index: usize,
        radius: f64,
        p1: f64,
        p2: f64,
        p3: f64,
        p4: f64,
    ) {
        self.particles[index] = LCPOParticle {
            radius,
            p1,
            p2,
            p3,
            p4,
        };
    }
    pub fn surface_tension(&self) -> f64 {
        self.surface_tension
    }
    pub fn set_surface_tension(&mut self, value: f64) {
        self.surface_tension = value;
    }
    pub fn uses_periodic_boundary_conditions(&self) -> bool {
        self.periodic
    }
    pub fn set_uses_periodic_boundary_conditions(&mut self, periodic: bool) {
        self.periodic = periodic;
    }
    pub fn update_parameters_from(&mut self, source: &Self) {
        assert_eq!(
            self.particles.len(),
            source.particles.len(),
            "LCPO particle count cannot change in a Context"
        );
        source.validate();
        self.particles.clone_from(&source.particles);
        self.surface_tension = source.surface_tension;
        self.periodic = source.periodic;
    }
    fn validate(&self) {
        assert!(
            self.surface_tension.is_finite(),
            "LCPO surface tension must be finite"
        );
        for p in &self.particles {
            assert!(
                p.radius >= 0.0
                    && [p.radius, p.p1, p.p2, p.p3, p.p4]
                        .iter()
                        .all(|v| v.is_finite()),
                "LCPO radii must be nonnegative and all parameters finite"
            );
        }
    }
    fn delta(
        &self,
        positions: &[f64],
        i: usize,
        j: usize,
        vectors: Option<&[[f64; 3]; 3]>,
    ) -> [f64; 3] {
        let mut delta =
            std::array::from_fn(|axis| positions[3 * j + axis] - positions[3 * i + axis]);
        if let Some(vectors) = vectors {
            apply_minimum_image_ortho(&mut delta, vectors);
        }
        delta
    }
    fn evaluate(
        &self,
        positions: &[f64],
        vectors: Option<&[[f64; 3]; 3]>,
        work: &mut Work,
        forces: &mut [f64],
    ) -> f64 {
        let n = self.particles.len();
        assert_eq!(
            positions.len(),
            3 * n,
            "LCPO particle count must match positions"
        );
        let cutoff = 2.0 * self.particles.iter().map(|p| p.radius).fold(0.0, f64::max);
        let vectors = if self.periodic {
            let vectors = vectors.expect("periodic LCPO requires box vectors");
            assert!(
                (0..3).all(|i| vectors[i][i] >= 1.999999 * cutoff),
                "LCPO cutoff exceeds half a periodic box width"
            );
            Some(vectors)
        } else {
            None
        };
        forces.fill(0.0);
        let mut energy = self
            .particles
            .iter()
            .map(|p| 4.0 * std::f64::consts::PI * p.p1 * p.radius * p.radius)
            .sum::<f64>();
        if cutoff == 0.0 {
            return energy * self.surface_tension;
        }
        let Work {
            contacts,
            offsets,
            cursors,
            neighbors,
            adjoints,
            ..
        } = work;
        contacts.clear();
        offsets.resize(n + 1, 0);
        offsets.fill(0);
        self.cells.visit(positions, cutoff, vectors, |i, j| {
            let ri = self.particles[i].radius;
            let rj = self.particles[j].radius;
            if ri == 0.0 || rj == 0.0 {
                return;
            }
            let delta = self.delta(positions, i, j, vectors);
            let r = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
            if r >= ri + rj {
                return;
            }
            assert!(
                r > 0.0,
                "LCPO requires distinct positions for active overlapping particles"
            );
            let diff = (ri * ri - rj * rj) / r;
            contacts.push(Contact {
                particles: [i, j],
                area: [
                    std::f64::consts::PI * ri * (2.0 * ri - r - diff),
                    std::f64::consts::PI * rj * (2.0 * rj - r + diff),
                ],
                slope: [
                    std::f64::consts::PI * ri * (diff / r - 1.0),
                    std::f64::consts::PI * rj * (-diff / r - 1.0),
                ],
            });
            offsets[i + 1] += 1;
            offsets[j + 1] += 1;
        });
        for i in 1..=n {
            offsets[i] += offsets[i - 1];
        }
        cursors.clone_from(offsets);
        neighbors.resize(2 * contacts.len(), 0);
        adjoints.resize(contacts.len(), [0.; 2]);
        adjoints.fill([0.; 2]);
        for (edge, c) in contacts.iter().enumerate() {
            for &i in &c.particles {
                neighbors[cursors[i]] = edge;
                cursors[i] += 1;
            }
        }
        for i in 0..n {
            neighbors[offsets[i]..offsets[i + 1]]
                .sort_unstable_by_key(|&edge| contacts[edge].other(i));
        }
        for (ij, c) in contacts.iter().enumerate() {
            let [i, j] = c.particles;
            let pi = self.particles[i];
            let pj = self.particles[j];
            energy += pi.p2 * c.area[0] + pj.p2 * c.area[1];
            adjoints[ij][0] += pi.p2;
            adjoints[ij][1] += pj.p2;
            let ni = &neighbors[offsets[i]..offsets[i + 1]];
            let nj = &neighbors[offsets[j]..offsets[j + 1]];
            let mut a = ni.partition_point(|&edge| contacts[edge].other(i) <= j);
            let mut b = nj.partition_point(|&edge| contacts[edge].other(j) <= j);
            while a < ni.len() && b < nj.len() {
                let ki = contacts[ni[a]].other(i);
                let kj = contacts[nj[b]].other(j);
                if ki < kj {
                    a += 1;
                    continue;
                }
                if kj < ki {
                    b += 1;
                    continue;
                }
                let ik = ni[a];
                let jk = nj[b];
                let pk = self.particles[ki];
                let [aij, aji] = c.area;
                let [aik, aki] = contacts[ik].area;
                let [ajk, akj] = contacts[jk].area;
                energy += pi.p3 * (ajk + akj)
                    + pi.p4 * (aij * ajk + aik * akj)
                    + pj.p3 * (aik + aki)
                    + pj.p4 * (aji * aik + ajk * aki)
                    + pk.p3 * (aij + aji)
                    + pk.p4 * (aki * aij + akj * aji);
                adjoints[ij][0] += pi.p4 * ajk + pk.p3 + pk.p4 * aki;
                adjoints[ij][1] += pj.p4 * aik + pk.p3 + pk.p4 * akj;
                adjoints[ik][0] += pi.p4 * akj + pj.p3 + pj.p4 * aji;
                adjoints[ik][1] += pj.p3 + pj.p4 * ajk + pk.p4 * aij;
                adjoints[jk][0] += pi.p3 + pi.p4 * aij + pj.p4 * aki;
                adjoints[jk][1] += pi.p3 + pi.p4 * aik + pk.p4 * aji;
                a += 1;
                b += 1;
            }
        }
        for (c, adjoint) in contacts.iter().zip(adjoints) {
            let [i, j] = c.particles;
            let delta = self.delta(positions, i, j, vectors);
            let r = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
            let factor =
                self.surface_tension * (adjoint[0] * c.slope[0] + adjoint[1] * c.slope[1]) / r;
            for axis in 0..3 {
                let f = factor * delta[axis];
                forces[3 * i + axis] += f;
                forces[3 * j + axis] -= f;
            }
        }
        energy * self.surface_tension
    }
}
impl<B: Backend> Force<B> for LCPOForce {
    super::force_metadata_impl!();
    fn initialize(&mut self, n: usize, _vectors: Option<&[[f64; 3]; 3]>) {
        assert_eq!(
            self.particles.len(),
            n,
            "LCPO particle count must match System"
        );
        self.validate();
    }
    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.periodic
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
