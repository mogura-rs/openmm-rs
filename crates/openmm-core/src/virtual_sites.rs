use burn::prelude::*;
use nalgebra::{Matrix3, Vector3};

type Vector = Vector3<f64>;

#[derive(Debug, Clone)]
pub enum VirtualSite {
    TwoParticleAverage {
        particles: [usize; 2],
        weights: [f64; 2],
    },
    ThreeParticleAverage {
        particles: [usize; 3],
        weights: [f64; 3],
    },
    OutOfPlane {
        particles: [usize; 3],
        weights: [f64; 3],
    },
    LocalCoordinates {
        particles: Vec<usize>,
        origin_weights: Vec<f64>,
        x_weights: Vec<f64>,
        y_weights: Vec<f64>,
        local_position: [f64; 3],
    },
    Symmetry {
        particle: usize,
        rotation: [[f64; 3]; 3],
        offset: [f64; 3],
        use_box_vectors: bool,
    },
}

impl VirtualSite {
    pub fn particles(&self) -> &[usize] {
        match self {
            Self::TwoParticleAverage { particles, .. } => particles,
            Self::ThreeParticleAverage { particles, .. } | Self::OutOfPlane { particles, .. } => {
                particles
            }
            Self::LocalCoordinates { particles, .. } => particles,
            Self::Symmetry { particle, .. } => std::slice::from_ref(particle),
        }
    }

    fn validate(&self) {
        if let Self::LocalCoordinates {
            particles,
            origin_weights,
            x_weights,
            y_weights,
            ..
        } = self
        {
            assert!(
                particles.len() >= 2,
                "local coordinate sites must depend on at least two particles"
            );
            for weights in [origin_weights, x_weights, y_weights] {
                assert_eq!(
                    weights.len(),
                    particles.len(),
                    "local coordinate weight count must match particle count"
                );
                assert!(
                    weights.iter().all(|v| v.is_finite()),
                    "virtual site weights must be finite"
                );
            }
            assert!(
                (origin_weights.iter().sum::<f64>() - 1.0).abs() <= 1e-6,
                "origin weights must sum to one"
            );
            assert!(
                x_weights.iter().sum::<f64>().abs() <= 1e-6
                    && y_weights.iter().sum::<f64>().abs() <= 1e-6,
                "axis weights must sum to zero"
            );
        }
        if let Self::Symmetry { rotation, .. } = self {
            let matrix = matrix(rotation);
            assert!(
                (matrix.transpose() * matrix - Matrix3::identity()).norm() < 1e-6,
                "symmetry rotation matrix must be orthogonal"
            );
        }
    }

    fn position(&self, positions: &[Vector], box_vectors: Option<&[[f64; 3]; 3]>) -> Vector {
        match self {
            Self::TwoParticleAverage { particles, weights } => {
                weighted(positions, particles, weights)
            }
            Self::ThreeParticleAverage { particles, weights } => {
                weighted(positions, particles, weights)
            }
            Self::OutOfPlane {
                particles: [a, b, c],
                weights: [wx, wy, wz],
            } => {
                let x = positions[*b] - positions[*a];
                let y = positions[*c] - positions[*a];
                positions[*a] + x * (*wx) + y * (*wy) + x.cross(&y) * (*wz)
            }
            Self::LocalCoordinates {
                particles,
                origin_weights,
                x_weights,
                y_weights,
                local_position,
            } => {
                let origin = weighted(positions, particles, origin_weights);
                let x = weighted(positions, particles, x_weights);
                let y = weighted(positions, particles, y_weights);
                let x_unit = normalized(x);
                let z_unit = normalized(x.cross(&y));
                origin
                    + x_unit * local_position[0]
                    + z_unit.cross(&x_unit) * local_position[1]
                    + z_unit * local_position[2]
            }
            Self::Symmetry {
                particle,
                rotation,
                offset,
                use_box_vectors,
            } => {
                let (transform, translation) =
                    symmetry_transform(rotation, offset, *use_box_vectors, box_vectors);
                transform * positions[*particle] + translation
            }
        }
    }

    fn distribute(
        &self,
        positions: &[Vector],
        forces: &mut [Vector],
        force: Vector,
        box_vectors: Option<&[[f64; 3]; 3]>,
    ) {
        match self {
            Self::TwoParticleAverage { particles, weights } => {
                for (particle, weight) in particles.iter().zip(weights) {
                    forces[*particle] += force * (*weight);
                }
            }
            Self::ThreeParticleAverage { particles, weights } => {
                for (particle, weight) in particles.iter().zip(weights) {
                    forces[*particle] += force * (*weight);
                }
            }
            Self::OutOfPlane {
                particles: [a, b, c],
                weights: [wx, wy, wz],
            } => {
                let x = positions[*b] - positions[*a];
                let y = positions[*c] - positions[*a];
                let fx = force * (*wx) + y.cross(&force) * (*wz);
                let fy = force * (*wy) + force.cross(&x) * (*wz);
                forces[*a] += force - fx - fy;
                forces[*b] += fx;
                forces[*c] += fy;
            }
            Self::LocalCoordinates {
                particles,
                origin_weights,
                x_weights,
                y_weights,
                local_position,
            } => {
                let x = weighted(positions, particles, x_weights);
                let y = weighted(positions, particles, y_weights);
                let z = x.cross(&y);
                let x_unit = normalized(x);
                let z_unit = normalized(z);
                let gy_unit = force * local_position[1];
                let gx_unit = force * local_position[0] + gy_unit.cross(&z_unit);
                let gz_unit = force * local_position[2] + x_unit.cross(&gy_unit);
                let gz = normalization_adjoint(z, gz_unit);
                let gx = normalization_adjoint(x, gx_unit) + y.cross(&gz);
                let gy = gz.cross(&x);
                for (j, &particle) in particles.iter().enumerate() {
                    forces[particle] +=
                        force * origin_weights[j] + gx * x_weights[j] + gy * y_weights[j];
                }
            }
            Self::Symmetry {
                particle,
                rotation,
                offset,
                use_box_vectors,
            } => {
                let (transform, _) =
                    symmetry_transform(rotation, offset, *use_box_vectors, box_vectors);
                forces[*particle] += transform.transpose() * force;
            }
        }
    }
}

fn weighted(positions: &[Vector], particles: &[usize], weights: &[f64]) -> Vector {
    particles
        .iter()
        .zip(weights)
        .fold(Vector::zeros(), |sum, (&particle, &weight)| {
            sum + positions[particle] * weight
        })
}

fn normalized(value: Vector) -> Vector {
    let norm = value.norm();
    if norm > 0.0 {
        value / norm
    } else {
        Vector::zeros()
    }
}

fn normalization_adjoint(value: Vector, gradient: Vector) -> Vector {
    let norm = value.norm();
    if norm > 0.0 {
        let unit = value / norm;
        (gradient - unit * gradient.dot(&unit)) / norm
    } else {
        Vector::zeros()
    }
}

fn matrix(rows: &[[f64; 3]; 3]) -> Matrix3<f64> {
    Matrix3::from_rows(&rows.map(|row| Vector::from(row).transpose()))
}

fn symmetry_transform(
    rotation: &[[f64; 3]; 3],
    offset: &[f64; 3],
    use_box_vectors: bool,
    box_vectors: Option<&[[f64; 3]; 3]>,
) -> (Matrix3<f64>, Vector) {
    let rotation = matrix(rotation);
    let offset = Vector::from(*offset);
    if use_box_vectors {
        let basis =
            matrix(box_vectors.expect("fractional symmetry sites require periodic box vectors"))
                .transpose();
        (
            basis
                * rotation
                * basis
                    .try_inverse()
                    .expect("periodic box must be nonsingular"),
            basis * offset,
        )
    } else {
        (rotation, offset)
    }
}

pub(crate) struct VirtualSitePlan {
    order: Vec<usize>,
}

impl VirtualSitePlan {
    pub(crate) fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
    pub fn new(sites: &[Option<VirtualSite>], masses: &[f64]) -> Self {
        for (index, site) in sites.iter().enumerate() {
            if let Some(site) = site {
                site.validate();
                assert_eq!(masses[index], 0.0, "virtual sites must have zero mass");
                assert!(
                    site.particles()
                        .iter()
                        .all(|&i| i < sites.len() && i != index),
                    "virtual site dependencies must be valid and cannot refer to the site itself"
                );
            }
        }
        let mut status = vec![0_u8; sites.len()];
        let mut order = Vec::new();
        let mut stack = Vec::new();
        for root in 0..sites.len() {
            if sites[root].is_none() || status[root] != 0 {
                continue;
            }
            stack.push((root, 0));
            status[root] = 1;
            while let Some((index, next)) = stack.last_mut() {
                let dependencies = sites[*index].as_ref().unwrap().particles();
                if *next == dependencies.len() {
                    status[*index] = 2;
                    order.push(*index);
                    stack.pop();
                    continue;
                }
                let child = dependencies[*next];
                *next += 1;
                if sites[child].is_some() {
                    assert_ne!(status[child], 1, "virtual site definitions are circular");
                    if status[child] == 0 {
                        status[child] = 1;
                        stack.push((child, 0));
                    }
                }
            }
        }
        Self { order }
    }

    pub fn compute_positions<B: Backend>(
        &self,
        sites: &[Option<VirtualSite>],
        positions: &mut Tensor<B, 2>,
        box_vectors: Option<&[[f64; 3]; 3]>,
    ) {
        if self.order.is_empty() {
            return;
        }
        let mut data = read_vectors(positions);
        for &index in &self.order {
            data[index] = sites[index].as_ref().unwrap().position(&data, box_vectors);
        }
        *positions = write_vectors(&data, &positions.device());
    }

    pub fn distribute_forces<B: Backend>(
        &self,
        sites: &[Option<VirtualSite>],
        positions: &Tensor<B, 2>,
        forces: &mut Tensor<B, 2>,
        box_vectors: Option<&[[f64; 3]; 3]>,
    ) {
        if self.order.is_empty() {
            return;
        }
        let positions = read_vectors(positions);
        let mut data = read_vectors(forces);
        for &index in self.order.iter().rev() {
            let force = data[index];
            sites[index]
                .as_ref()
                .unwrap()
                .distribute(&positions, &mut data, force, box_vectors);
        }
        *forces = write_vectors(&data, &forces.device());
    }
}

fn read_vectors<B: Backend>(tensor: &Tensor<B, 2>) -> Vec<Vector> {
    tensor
        .to_data()
        .convert::<f64>()
        .to_vec::<f64>()
        .unwrap()
        .chunks_exact(3)
        .map(|row| Vector::new(row[0], row[1], row[2]))
        .collect()
}

fn write_vectors<B: Backend>(values: &[Vector], device: &B::Device) -> Tensor<B, 2> {
    let data: Vec<f64> = values.iter().flat_map(|v| v.iter().copied()).collect();
    crate::tensor::from_floats::<B, 1>(data.as_slice(), device).reshape([values.len(), 3])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn virtual_site_force_jacobians_match_energy_gradients() {
        let sites = [
            VirtualSite::TwoParticleAverage {
                particles: [0, 1],
                weights: [0.3, 0.7],
            },
            VirtualSite::ThreeParticleAverage {
                particles: [0, 1, 2],
                weights: [0.2, 0.3, 0.5],
            },
            VirtualSite::OutOfPlane {
                particles: [0, 1, 2],
                weights: [0.2, 0.3, 0.4],
            },
            VirtualSite::LocalCoordinates {
                particles: vec![0, 1, 2],
                origin_weights: vec![0.2, 0.3, 0.5],
                x_weights: vec![-1.0, 1.0, 0.0],
                y_weights: vec![-1.0, 0.0, 1.0],
                local_position: [0.1, 0.2, 0.3],
            },
            VirtualSite::Symmetry {
                particle: 0,
                rotation: [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                offset: [0.1, 0.2, 0.3],
                use_box_vectors: false,
            },
            VirtualSite::Symmetry {
                particle: 0,
                rotation: [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                offset: [0.1, 0.2, 0.3],
                use_box_vectors: true,
            },
        ];
        let positions = [
            Vector::new(0.1, 0.2, 0.3),
            Vector::new(0.5, 0.1, 0.4),
            Vector::new(0.2, 0.7, 0.8),
        ];
        let box_vectors = [[2.0, 0.0, 0.0], [0.3, 2.4, 0.0], [-0.2, 0.4, 2.8]];
        for site in sites {
            let position = site.position(&positions, Some(&box_vectors));
            let mut forces = [Vector::zeros(); 3];
            site.distribute(&positions, &mut forces, -position, Some(&box_vectors));
            for particle in 0..3 {
                for axis in 0..3 {
                    let mut plus = positions;
                    let mut minus = positions;
                    plus[particle][axis] += 1e-6;
                    minus[particle][axis] -= 1e-6;
                    let numeric = -(site.position(&plus, Some(&box_vectors)).norm_squared()
                        - site.position(&minus, Some(&box_vectors)).norm_squared())
                        / 4e-6;
                    assert!(
                        (numeric - forces[particle][axis]).abs() < 1e-8,
                        "{site:?}, particle {particle}, axis {axis}"
                    );
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "circular")]
    fn cycles_are_rejected() {
        let sites = [
            Some(VirtualSite::TwoParticleAverage {
                particles: [1, 2],
                weights: [0.5, 0.5],
            }),
            Some(VirtualSite::TwoParticleAverage {
                particles: [0, 2],
                weights: [0.5, 0.5],
            }),
            None,
        ];
        let _ = VirtualSitePlan::new(&sites, &[0.0, 0.0, 1.0]);
    }
}
