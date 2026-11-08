use super::Force;
use burn::prelude::*;
use nalgebra::{Matrix3, Matrix4, SymmetricEigen, Vector3, Vector4};
use std::sync::Arc;

fn select_particles(requested: &[usize], n: usize) -> Vec<usize> {
    let particles = if requested.is_empty() {
        (0..n).collect::<Vec<_>>()
    } else {
        requested.to_vec()
    };
    assert!(
        !particles.is_empty(),
        "particle selection must not be empty"
    );
    let mut sorted = particles.clone();
    sorted.sort_unstable();
    assert!(
        sorted.last().copied().unwrap() < n,
        "particle index out of range"
    );
    assert!(
        sorted.windows(2).all(|p| p[0] != p[1]),
        "duplicate particle index"
    );
    particles
}

fn center(positions: &[f64], particles: &[usize]) -> Vector3<f64> {
    let mut result = Vector3::zeros();
    for &p in particles {
        result += Vector3::from_column_slice(&positions[3 * p..3 * p + 3]);
    }
    result / particles.len() as f64
}

fn scale_forces(forces: &mut [f64], particles: &[usize], scale: f64) {
    if particles.len() * 3 == forces.len() {
        for value in forces {
            *value *= scale;
        }
    } else {
        for &particle in particles {
            for value in &mut forces[3 * particle..3 * particle + 3] {
                *value *= scale;
            }
        }
    }
}

fn quaternion_matrix(r: &Matrix3<f64>) -> Matrix4<f64> {
    Matrix4::new(
        r[(0, 0)] + r[(1, 1)] + r[(2, 2)],
        r[(1, 2)] - r[(2, 1)],
        r[(2, 0)] - r[(0, 2)],
        r[(0, 1)] - r[(1, 0)],
        r[(1, 2)] - r[(2, 1)],
        r[(0, 0)] - r[(1, 1)] - r[(2, 2)],
        r[(0, 1)] + r[(1, 0)],
        r[(0, 2)] + r[(2, 0)],
        r[(2, 0)] - r[(0, 2)],
        r[(0, 1)] + r[(1, 0)],
        -r[(0, 0)] + r[(1, 1)] - r[(2, 2)],
        r[(1, 2)] + r[(2, 1)],
        r[(0, 1)] - r[(1, 0)],
        r[(0, 2)] + r[(2, 0)],
        r[(1, 2)] + r[(2, 1)],
        -r[(0, 0)] - r[(1, 1)] + r[(2, 2)],
    )
}

fn rotation(q: &Vector4<f64>) -> Matrix3<f64> {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    Matrix3::new(
        w * w + x * x - y * y - z * z,
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),
        w * w - x * x + y * y - z * z,
        2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        w * w - x * x - y * y + z * z,
    )
}

#[derive(Clone)]
struct AlignmentReference {
    positions: Arc<[[f64; 3]]>,
    requested: Vec<usize>,
    selected: Arc<[usize]>,
    centered: Arc<[Vector3<f64>]>,
}

impl AlignmentReference {
    fn new(positions: Vec<[f64; 3]>, particles: Vec<usize>) -> Self {
        Self {
            positions: positions.into(),
            requested: particles,
            selected: Arc::from([]),
            centered: Arc::from([]),
        }
    }
    fn initialize(&mut self, n: usize) {
        assert_eq!(
            self.positions.len(),
            n,
            "reference position count must match the System"
        );
        let particles = select_particles(&self.requested, n);
        let mut center = Vector3::zeros();
        for &p in &particles {
            center += Vector3::from(self.positions[p]);
        }
        center /= particles.len() as f64;
        self.centered = particles
            .iter()
            .map(|&p| Vector3::from(self.positions[p]) - center)
            .collect::<Vec<_>>()
            .into();
        self.selected = particles.into();
    }
    fn alignment(
        &self,
        positions: &[f64],
    ) -> (Vector3<f64>, SymmetricEigen<f64, nalgebra::Const<4>>, usize) {
        assert!(
            !self.selected.is_empty(),
            "alignment force has not been initialized"
        );
        let center = center(positions, &self.selected);
        let mut correlation = Matrix3::zeros();
        for (&p, reference) in self.selected.iter().zip(self.centered.iter()) {
            let current = Vector3::from_column_slice(&positions[3 * p..3 * p + 3]) - center;
            correlation += reference * current.transpose();
        }
        let eig = SymmetricEigen::new(quaternion_matrix(&correlation));
        let index = (0..4)
            .max_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]))
            .unwrap();
        (center, eig, index)
    }
    fn update_from(&mut self, source: &Self) {
        let n = self.positions.len();
        assert_eq!(
            n,
            source.positions.len(),
            "reference position count must not change"
        );
        let mut updated = source.clone();
        updated.initialize(n);
        *self = updated;
    }
}

macro_rules! reference_api {
    () => {
        super::force_metadata!();
        pub fn reference_positions(&self) -> &[[f64; 3]] {
            &self.reference.positions
        }
        pub fn set_reference_positions(&mut self, positions: Vec<[f64; 3]>) {
            self.reference.positions = positions.into();
            self.reference.selected = Arc::from([]);
        }
        pub fn particles(&self) -> &[usize] {
            &self.reference.requested
        }
        pub fn set_particles(&mut self, particles: Vec<usize>) {
            self.reference.requested = particles;
            self.reference.selected = Arc::from([]);
        }
        pub fn uses_periodic_boundary_conditions(&self) -> bool {
            false
        }
    };
}

#[derive(Clone)]
pub struct RMSDForce {
    metadata: super::ForceMetadata,
    reference: AlignmentReference,
}
impl RMSDForce {
    reference_api!();
    pub fn new(reference_positions: Vec<[f64; 3]>, particles: Vec<usize>) -> Self {
        Self {
            metadata: super::ForceMetadata::new("RMSDForce"),
            reference: AlignmentReference::new(reference_positions, particles),
        }
    }
    pub fn update_parameters_from(&mut self, source: &Self) {
        self.reference.update_from(&source.reference);
    }
}
impl<B: Backend> Force<B> for RMSDForce {
    super::force_metadata_impl!();
    fn initialize(&mut self, n: usize, _vectors: Option<&[[f64; 3]; 3]>) {
        self.reference.initialize(n);
    }
    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        _vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64) {
        let n = positions.dims()[0];
        crate::tensor::with_f64_data(positions, |pos| {
            let (center, eig, index) = self.reference.alignment(pos);
            let q = eig.eigenvectors.column(index).into_owned();
            let rotation = rotation(&q);
            let mut forces = vec![0.0; 3 * n];
            let mut residual = 0.0;
            for (&p, reference) in self
                .reference
                .selected
                .iter()
                .zip(self.reference.centered.iter())
            {
                let delta = Vector3::from_column_slice(&pos[3 * p..3 * p + 3])
                    - center
                    - rotation * reference;
                residual += delta.norm_squared();
                forces[3 * p..3 * p + 3].copy_from_slice(delta.as_slice());
            }
            let count = self.reference.selected.len() as f64;
            let msd = residual / count;
            let rmsd = if msd < 1e-20 { 0.0 } else { msd.sqrt() };
            if rmsd == 0.0 {
                forces.fill(0.0);
            } else {
                let scale = -1.0 / (count * rmsd);
                scale_forces(&mut forces, &self.reference.selected, scale);
            }
            (
                crate::tensor::from_f64_vec(forces, [n, 3], &positions.device()),
                rmsd,
            )
        })
    }
}

#[derive(Clone)]
pub struct OrientationRestraintForce {
    metadata: super::ForceMetadata,
    reference: AlignmentReference,
    k: f64,
}
impl OrientationRestraintForce {
    reference_api!();
    pub fn new(k: f64, reference_positions: Vec<[f64; 3]>, particles: Vec<usize>) -> Self {
        Self {
            metadata: super::ForceMetadata::new("OrientationRestraintForce"),
            reference: AlignmentReference::new(reference_positions, particles),
            k,
        }
    }
    pub fn k(&self) -> f64 {
        self.k
    }
    pub fn set_k(&mut self, k: f64) {
        self.k = k;
    }
    pub fn update_parameters_from(&mut self, source: &Self) {
        self.reference.update_from(&source.reference);
        self.k = source.k;
    }
}
impl<B: Backend> Force<B> for OrientationRestraintForce {
    super::force_metadata_impl!();
    fn initialize(&mut self, n: usize, _vectors: Option<&[[f64; 3]; 3]>) {
        self.reference.initialize(n);
    }
    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        _vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64) {
        let n = positions.dims()[0];
        crate::tensor::with_f64_data(positions, |pos| {
            let (_, eig, index) = self.reference.alignment(pos);
            let q = eig.eigenvectors.column(index).into_owned();
            let energy = 2.0 * self.k * (1.0 - q[0] * q[0]);
            let mut forces = vec![0.0; 3 * n];
            if q[0] * q[0] < 1.0 {
                let mut sensitivity = Vector4::zeros();
                for other in 0..4 {
                    if other != index {
                        sensitivity += eig.eigenvectors.column(other)
                            * (eig.eigenvectors[(0, other)]
                                / (eig.eigenvalues[index] - eig.eigenvalues[other]));
                    }
                }
                let mut gradient = Matrix3::zeros();
                for ref_axis in 0..3 {
                    for axis in 0..3 {
                        let mut basis = Matrix3::zeros();
                        basis[(ref_axis, axis)] = 1.0;
                        gradient[(axis, ref_axis)] =
                            4.0 * self.k * q[0] * sensitivity.dot(&(quaternion_matrix(&basis) * q));
                    }
                }
                for (&p, reference) in self
                    .reference
                    .selected
                    .iter()
                    .zip(self.reference.centered.iter())
                {
                    let force = gradient * reference;
                    forces[3 * p..3 * p + 3].copy_from_slice(force.as_slice());
                }
            }
            (
                crate::tensor::from_f64_vec(forces, [n, 3], &positions.device()),
                energy,
            )
        })
    }
}

#[derive(Clone)]
pub struct RGForce {
    metadata: super::ForceMetadata,
    particles: Vec<usize>,
    selected: Arc<[usize]>,
}
impl RGForce {
    super::force_metadata!();
    pub fn new(particles: Vec<usize>) -> Self {
        Self {
            metadata: super::ForceMetadata::new("RGForce"),
            particles,
            selected: Arc::from([]),
        }
    }
    pub fn particles(&self) -> &[usize] {
        &self.particles
    }
    pub fn set_particles(&mut self, particles: Vec<usize>) {
        self.particles = particles;
        self.selected = Arc::from([]);
    }
    pub fn uses_periodic_boundary_conditions(&self) -> bool {
        false
    }

    fn prepare_host(&self, positions: &[f64]) -> (Vector3<f64>, f64) {
        assert!(
            !self.selected.is_empty(),
            "RG force has not been initialized"
        );
        let center = center(positions, &self.selected);
        let mut squared = 0.0;
        for &p in self.selected.iter() {
            let delta = Vector3::from_column_slice(&positions[3 * p..3 * p + 3]) - center;
            squared += delta.norm_squared();
        }
        let count = self.selected.len() as f64;
        let rg = (squared / count).sqrt();
        (center, rg)
    }

    fn accumulate_host(
        &self,
        positions: &[f64],
        center: Vector3<f64>,
        rg: f64,
        coefficient: f64,
        total: &mut [f64],
    ) {
        assert_eq!(total.len(), positions.len(), "force buffer shape mismatch");
        if rg > 0.0 {
            let scale = -coefficient / (self.selected.len() as f64 * rg);
            for &p in self.selected.iter() {
                let delta = Vector3::from_column_slice(&positions[3 * p..3 * p + 3]) - center;
                for axis in 0..3 {
                    total[3 * p + axis] += scale * delta[axis];
                }
            }
        }
    }
}
impl<B: Backend> Force<B> for RGForce {
    super::force_metadata_impl!();
    fn initialize(&mut self, n: usize, _vectors: Option<&[[f64; 3]; 3]>) {
        self.selected = select_particles(&self.particles, n).into();
    }
    fn prepare_host_evaluation<'a>(
        &'a self,
        positions: &'a Tensor<B, 2>,
        _vectors: Option<&'a [[f64; 3]; 3]>,
    ) -> Option<super::HostForceEvaluation<'a>> {
        let (center, rg) =
            crate::tensor::with_f64_data(positions, |positions| self.prepare_host(positions));
        Some(super::HostForceEvaluation::new(
            rg,
            move |coefficient, total| {
                crate::tensor::with_f64_data(positions, |positions| {
                    self.accumulate_host(positions, center, rg, coefficient, total);
                });
            },
        ))
    }
    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        _vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64) {
        assert!(
            !self.selected.is_empty(),
            "RG force has not been initialized"
        );
        let n = positions.dims()[0];
        crate::tensor::with_f64_data(positions, |pos| {
            let center = center(pos, &self.selected);
            let mut forces = vec![0.0; 3 * n];
            let mut squared = 0.0;
            for &p in self.selected.iter() {
                let delta = Vector3::from_column_slice(&pos[3 * p..3 * p + 3]) - center;
                squared += delta.norm_squared();
                forces[3 * p..3 * p + 3].copy_from_slice(delta.as_slice());
            }
            let count = self.selected.len() as f64;
            let rg = (squared / count).sqrt();
            if rg > 0.0 {
                let scale = -1.0 / (count * rg);
                scale_forces(&mut forces, &self.selected, scale);
            }
            (
                crate::tensor::from_f64_vec(forces, [n, 3], &positions.device()),
                rg,
            )
        })
    }
}
