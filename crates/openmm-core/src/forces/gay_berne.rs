use super::{
    harmonic_bond::apply_minimum_image_ortho, neighbors::NeighborCells, Force, HostForceEvaluation,
};
use burn::prelude::*;
use nalgebra::{Matrix3, Vector3};
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
type V = Vector3<f64>;
type M = Matrix3<f64>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GayBerneMethod {
    #[default]
    NoCutoff,
    CutoffNonPeriodic,
    CutoffPeriodic,
}
#[derive(Debug, Clone, Copy)]
pub struct GayBerneParticle {
    pub sigma: f64,
    pub epsilon: f64,
    pub xparticle: Option<usize>,
    pub yparticle: Option<usize>,
    pub widths: [f64; 3],
    pub scales: [f64; 3],
}
impl GayBerneParticle {
    fn validate_shape(&self) -> Result<(), String> {
        if self.yparticle.is_none()
            && (self.widths[1] != self.widths[2] || self.scales[1] != self.scales[2])
        {
            return Err("Gay-Berne without a y axis requires axial symmetry".into());
        }
        if self.xparticle.is_none()
            && (self.yparticle.is_some()
                || self.widths[0] != self.widths[2]
                || self.scales[0] != self.scales[2])
        {
            return Err("Gay-Berne without an x axis requires spherical symmetry".into());
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy)]
pub struct GayBerneException {
    pub particles: [usize; 2],
    pub sigma: f64,
    pub epsilon: f64,
}
struct Frame {
    g: M,
    b: M,
    shape: f64,
    isotropic_b: f64,
}
#[derive(Default)]
struct Work {
    frames: Vec<Frame>,
    torques: Vec<V>,
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
pub struct GayBerneForce {
    metadata: super::ForceMetadata,
    particles: Vec<GayBerneParticle>,
    exceptions: Vec<GayBerneException>,
    exception_indices: HashMap<(usize, usize), usize>,
    exclusions: HashSet<(usize, usize)>,
    method: GayBerneMethod,
    cutoff: f64,
    use_switching: bool,
    switching_distance: f64,
    cells: NeighborCells,
    workspace: Workspace,
}
impl Default for GayBerneForce {
    fn default() -> Self {
        Self::new()
    }
}
impl GayBerneForce {
    super::force_metadata!();
    pub fn new() -> Self {
        Self {
            metadata: super::ForceMetadata::new("GayBerneForce"),
            particles: Vec::new(),
            exceptions: Vec::new(),
            exception_indices: HashMap::new(),
            exclusions: HashSet::new(),
            method: GayBerneMethod::NoCutoff,
            cutoff: 1.0,
            use_switching: false,
            switching_distance: -1.0,
            cells: NeighborCells::default(),
            workspace: Workspace::default(),
        }
    }
    pub fn add_particle(&mut self, particle: GayBerneParticle) -> Result<usize, String> {
        particle.validate_shape()?;
        self.particles.push(particle);
        Ok(self.particles.len() - 1)
    }
    pub fn num_particles(&self) -> usize {
        self.particles.len()
    }
    pub fn particle_parameters(&self, index: usize) -> &GayBerneParticle {
        &self.particles[index]
    }
    pub fn set_particle_parameters(
        &mut self,
        index: usize,
        particle: GayBerneParticle,
    ) -> Result<(), String> {
        particle.validate_shape()?;
        self.particles[index] = particle;
        Ok(())
    }
    pub fn num_exceptions(&self) -> usize {
        self.exceptions.len()
    }
    pub fn exception_parameters(&self, index: usize) -> &GayBerneException {
        &self.exceptions[index]
    }
    pub fn add_exception(
        &mut self,
        first: usize,
        second: usize,
        sigma: f64,
        epsilon: f64,
        replace: bool,
    ) -> Result<usize, String> {
        let key = (first.min(second), first.max(second));
        let value = GayBerneException {
            particles: [first, second],
            sigma,
            epsilon,
        };
        let index = if let Some(&index) = self.exception_indices.get(&key) {
            if !replace {
                return Err("Gay-Berne exception already exists for this pair".into());
            }
            self.exceptions[index] = value;
            index
        } else {
            self.exceptions.push(value);
            self.exceptions.len() - 1
        };
        self.exception_indices.insert(key, index);
        self.exclusions = self
            .exceptions
            .iter()
            .map(|e| {
                (
                    e.particles[0].min(e.particles[1]),
                    e.particles[0].max(e.particles[1]),
                )
            })
            .collect();
        Ok(index)
    }
    pub fn set_exception_parameters(
        &mut self,
        index: usize,
        first: usize,
        second: usize,
        sigma: f64,
        epsilon: f64,
    ) {
        self.exceptions[index] = GayBerneException {
            particles: [first, second],
            sigma,
            epsilon,
        };
        self.exclusions = self
            .exceptions
            .iter()
            .map(|e| {
                (
                    e.particles[0].min(e.particles[1]),
                    e.particles[0].max(e.particles[1]),
                )
            })
            .collect();
    }
    pub fn nonbonded_method(&self) -> GayBerneMethod {
        self.method
    }
    pub fn set_nonbonded_method(&mut self, method: GayBerneMethod) {
        self.method = method;
    }
    pub fn cutoff_distance(&self) -> f64 {
        self.cutoff
    }
    pub fn set_cutoff_distance(&mut self, value: f64) {
        self.cutoff = value;
    }
    pub fn use_switching_function(&self) -> bool {
        self.use_switching
    }
    pub fn set_use_switching_function(&mut self, value: bool) {
        self.use_switching = value;
    }
    pub fn switching_distance(&self) -> f64 {
        self.switching_distance
    }
    pub fn set_switching_distance(&mut self, value: f64) {
        self.switching_distance = value;
    }
    pub fn uses_periodic_boundary_conditions(&self) -> bool {
        self.method == GayBerneMethod::CutoffPeriodic
    }
    pub fn update_parameters_from(&mut self, source: &Self) {
        assert_eq!(
            self.particles.len(),
            source.particles.len(),
            "Gay-Berne particle count cannot change in a Context"
        );
        source.validate();
        let metadata = self.metadata.clone();
        *self = source.clone();
        self.metadata = metadata;
    }
    fn validate(&self) {
        let n = self.particles.len();
        for (i, p) in self.particles.iter().enumerate() {
            p.validate_shape()
                .expect("invalid Gay-Berne particle shape");
            assert!(
                p.sigma.is_finite() && p.sigma >= 0.0 && p.epsilon.is_finite() && p.epsilon >= 0.0,
                "Gay-Berne sigma and epsilon must be nonnegative and finite"
            );
            assert!(
                p.widths
                    .iter()
                    .chain(&p.scales)
                    .all(|v| v.is_finite() && *v > 0.0),
                "Gay-Berne widths and energy scales must be positive and finite"
            );
            for axis in [p.xparticle, p.yparticle].into_iter().flatten() {
                assert!(
                    axis < n && axis != i,
                    "invalid Gay-Berne orientation particle"
                );
            }
        }
        let mut seen = HashSet::new();
        for e in &self.exceptions {
            let [i, j] = e.particles;
            assert!(
                i < n && j < n && i != j,
                "invalid Gay-Berne exception particles"
            );
            assert!(
                seen.insert((i.min(j), i.max(j))),
                "duplicate Gay-Berne exception pair"
            );
            assert!(
                e.sigma.is_finite() && e.sigma >= 0.0 && e.epsilon.is_finite() && e.epsilon >= 0.0,
                "Gay-Berne exception sigma and epsilon must be nonnegative and finite"
            );
        }
        if self.method != GayBerneMethod::NoCutoff {
            assert!(
                self.cutoff.is_finite() && self.cutoff > 0.0,
                "Gay-Berne cutoff must be positive and finite"
            );
        }
        if self.use_switching {
            assert!(
                self.switching_distance >= 0.0 && self.switching_distance < self.cutoff,
                "Gay-Berne switching distance must lie between zero and cutoff"
            );
        }
    }
    fn box_vectors<'a>(&self, vectors: Option<&'a [[f64; 3]; 3]>) -> Option<&'a [[f64; 3]; 3]> {
        if !self.uses_periodic_boundary_conditions() {
            return None;
        }
        let vectors = vectors.expect("periodic Gay-Berne requires box vectors");
        assert!(
            (0..3).all(|i| vectors[i][i] >= 1.999999 * self.cutoff),
            "Gay-Berne cutoff exceeds half a periodic box width"
        );
        Some(vectors)
    }
    fn position(positions: &[f64], index: usize) -> V {
        V::from_column_slice(&positions[3 * index..3 * index + 3])
    }
    fn frames(&self, positions: &[f64], work: &mut Work) {
        work.frames.clear();
        work.torques.resize(self.particles.len(), V::zeros());
        work.torques.fill(V::zeros());
        for (i, p) in self.particles.iter().enumerate() {
            let (x, y) = if let Some(axis) = p.xparticle {
                let dx = Self::position(positions, i) - Self::position(positions, axis);
                let norm = dx.norm();
                assert!(norm > 0.0, "Gay-Berne x axis has zero length");
                let x = dx / norm;
                let y = if let Some(axis) = p.yparticle {
                    Self::position(positions, i) - Self::position(positions, axis)
                } else if x[1] > -0.5 && x[1] < 0.5 {
                    V::y()
                } else {
                    V::x()
                };
                let y = y - x * x.dot(&y);
                let norm = y.norm();
                assert!(norm > 0.0, "Gay-Berne axes must not be collinear");
                (x, y / norm)
            } else {
                (V::x(), V::y())
            };
            let axes = [x, y, x.cross(&y)];
            let mut g = M::zeros();
            let mut b = M::zeros();
            let r = p.widths.map(|v| 0.5 * v);
            for k in 0..3 {
                let outer = axes[k] * axes[k].transpose();
                g += outer * (r[k] * r[k]);
                b += outer / p.scales[k].sqrt();
            }
            work.frames.push(Frame {
                g,
                b,
                shape: (r[0] * r[1] + r[2] * r[2]) * (r[0] * r[1]).sqrt(),
                isotropic_b: if p.scales[0] == p.scales[1] && p.scales[0] == p.scales[2] {
                    1.0 / p.scales[0].sqrt()
                } else {
                    0.0
                },
            });
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn interaction(
        &self,
        i: usize,
        j: usize,
        sigma: f64,
        epsilon: f64,
        positions: &[f64],
        vectors: Option<&[[f64; 3]; 3]>,
        work: &mut Work,
        forces: &mut [f64],
    ) -> f64 {
        if epsilon == 0.0 || sigma == 0.0 {
            return 0.0;
        }
        let mut delta: [f64; 3] =
            std::array::from_fn(|axis| positions[3 * i + axis] - positions[3 * j + axis]);
        if let Some(vectors) = vectors {
            apply_minimum_image_ortho(&mut delta, vectors);
        }
        let dr = V::from_column_slice(&delta);
        let r = dr.norm();
        if self.method != GayBerneMethod::NoCutoff && r >= self.cutoff {
            return 0.0;
        }
        assert!(r > 0.0, "Gay-Berne interacting centers must not coincide");
        let (switch, dswitch) = if self.method != GayBerneMethod::NoCutoff
            && self.use_switching
            && r > self.switching_distance
        {
            let t = (r - self.switching_distance) / (self.cutoff - self.switching_distance);
            (
                1.0 + t * t * t * (-10.0 + t * (15.0 - 6.0 * t)),
                t * t * (-30.0 + t * (60.0 - 30.0 * t)) / (self.cutoff - self.switching_distance),
            )
        } else {
            (1.0, 0.0)
        };
        if self.particles[i].xparticle.is_none() && self.particles[j].xparticle.is_none() {
            let g = work.frames[i].g[(0, 0)] + work.frames[j].g[(0, 0)];
            let b = work.frames[i].b[(0, 0)] + work.frames[j].b[(0, 0)];
            let contact = (2.0 * g).sqrt();
            let denominator = r - contact + sigma;
            let rho = sigma / denominator;
            let rho2 = rho * rho;
            let rho6 = rho2 * rho2 * rho2;
            let u = 4.0 * epsilon * (rho6 * rho6 - rho6);
            let du = -24.0 * epsilon * (2.0 * rho6 - 1.0) * rho6 / denominator;
            let eta = (2.0 * work.frames[i].shape * work.frames[j].shape / (g * g * g)).sqrt();
            let scale = 4.0 * eta / (b * b);
            let coefficient = scale * (du * switch + u * dswitch) / r;
            for axis in 0..3 {
                let gradient = coefficient * delta[axis];
                forces[3 * i + axis] -= gradient;
                forces[3 * j + axis] += gradient;
            }
            return u * scale * switch;
        }
        let unit = dr / r;
        let g = work.frames[i].g + work.frames[j].g;
        let gi = g.try_inverse().expect("singular Gay-Berne shape matrix");
        let vg = gi * unit;
        let qg = unit.dot(&vg);
        let isotropic_energy = work.frames[i].isotropic_b > 0.0 && work.frames[j].isotropic_b > 0.0;
        let (vb, qb) = if isotropic_energy {
            let qb = 1.0 / (work.frames[i].isotropic_b + work.frames[j].isotropic_b);
            (unit * qb, qb)
        } else {
            let b = work.frames[i].b + work.frames[j].b;
            let vb = b.try_inverse().expect("singular Gay-Berne energy matrix") * unit;
            (vb, unit.dot(&vb))
        };
        let contact = (2.0 / qg).sqrt();
        let denominator = r - contact + sigma;
        let rho = sigma / denominator;
        let rho2 = rho * rho;
        let rho6 = rho2 * rho2 * rho2;
        let u = 4.0 * epsilon * (rho6 * rho6 - rho6);
        let du = -24.0 * epsilon * (2.0 * rho6 - 1.0) * rho6 / denominator;
        let eta = (2.0 * work.frames[i].shape * work.frames[j].shape / g.determinant()).sqrt();
        let chi = 4.0 * qb * qb;
        let energy = u * eta * chi * switch;
        let dg = 0.25 * contact.powi(3) * du * eta * chi * switch;
        let db = 8.0 * qb * u * eta * switch;
        let mut gradient = unit * (du * eta * chi * switch + u * eta * chi * dswitch)
            + (vg - unit * qg) * (2.0 * dg / r);
        if !isotropic_energy {
            gradient += (vb - unit * qb) * (2.0 * db / r);
        }
        for axis in 0..3 {
            forces[3 * i + axis] -= gradient[axis];
            forces[3 * j + axis] += gradient[axis];
        }
        for particle in [i, j] {
            let f = &work.frames[particle];
            let product = gi * f.g;
            let determinant_torque = V::new(
                product[(2, 1)] - product[(1, 2)],
                product[(0, 2)] - product[(2, 0)],
                product[(1, 0)] - product[(0, 1)],
            );
            work.torques[particle] +=
                (f.g * vg).cross(&vg) * (2.0 * dg) + determinant_torque * energy;
            if !isotropic_energy {
                work.torques[particle] += (f.b * vb).cross(&vb) * (2.0 * db);
            }
        }
        energy
    }
    fn apply_torques(&self, positions: &[f64], work: &Work, forces: &mut [f64]) {
        for (i, p) in self.particles.iter().enumerate() {
            let Some(xindex) = p.xparticle else { continue };
            let a = Self::position(positions, i) - Self::position(positions, xindex);
            let length = a.norm();
            let x = a / length;
            let torque = work.torques[i];
            let mut fx = -torque.cross(&x) / length;
            let mut fy = V::zeros();
            if let Some(yindex) = p.yparticle {
                let b = Self::position(positions, i) - Self::position(positions, yindex);
                let parallel = x.dot(&b);
                let transverse = b - x * parallel;
                let norm = transverse.norm();
                let z = x.cross(&(transverse / norm));
                let twist = torque.dot(&x);
                fx += z * (twist * parallel / (norm * length));
                fy = -z * (twist / norm);
                for axis in 0..3 {
                    forces[3 * yindex + axis] += fy[axis];
                }
            }
            for axis in 0..3 {
                forces[3 * xindex + axis] += fx[axis];
                forces[3 * i + axis] -= fx[axis] + fy[axis];
            }
        }
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
            "Gay-Berne particle count must match positions"
        );
        let vectors = self.box_vectors(vectors);
        self.frames(positions, work);
        forces.fill(0.0);
        let mut energy = 0.0;
        let mut visit = |i: usize, j: usize| {
            let pi = self.particles[i];
            let pj = self.particles[j];
            if pi.epsilon == 0.0
                || pj.epsilon == 0.0
                || self.exclusions.contains(&(i.min(j), i.max(j)))
            {
                return;
            }
            energy += self.interaction(
                i,
                j,
                0.5 * (pi.sigma + pj.sigma),
                (pi.epsilon * pj.epsilon).sqrt(),
                positions,
                vectors,
                work,
                forces,
            );
        };
        if self.method == GayBerneMethod::NoCutoff {
            for i in 0..n {
                for j in i + 1..n {
                    visit(i, j);
                }
            }
        } else {
            self.cells.visit(positions, self.cutoff, vectors, visit);
        }
        for e in &self.exceptions {
            energy += self.interaction(
                e.particles[0],
                e.particles[1],
                e.sigma,
                e.epsilon,
                positions,
                vectors,
                work,
                forces,
            );
        }
        self.apply_torques(positions, work, forces);
        energy
    }
}
impl<B: Backend> Force<B> for GayBerneForce {
    super::force_metadata_impl!();
    fn initialize(&mut self, n: usize, vectors: Option<&[[f64; 3]; 3]>) {
        assert_eq!(
            n,
            self.particles.len(),
            "Gay-Berne particle count must match System"
        );
        self.validate();
        if self.uses_periodic_boundary_conditions() {
            let v = vectors.expect("periodic Gay-Berne requires box vectors");
            assert!(
                (0..3).all(|i| v[i][i] >= 2.0 * self.cutoff),
                "Gay-Berne cutoff exceeds half a periodic box width"
            );
        }
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
