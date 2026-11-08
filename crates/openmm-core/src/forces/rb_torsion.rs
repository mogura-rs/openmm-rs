// RBTorsionForce: E = sum_i c_i * cos(phi)^i
// OpenMM convention differs from PeriodicTorsionForce by a phase shift of pi:
// if theta is the periodic torsion angle, then phi = theta - pi.
// Reference test: openmm/tests/TestRBTorsionForce.h
use super::Force;
use crate::forces::harmonic_bond::apply_minimum_image_ortho;
use crate::forces::periodic_torsion::{sub3_pos, torsion_geometry};
use burn::prelude::*;

#[derive(Debug, Clone)]
pub struct RBTorsionParam {
    pub particle1: usize,
    pub particle2: usize,
    pub particle3: usize,
    pub particle4: usize,
    pub coefficients: [f64; 6],
}

impl RBTorsionParam {
    pub fn new(particles: [usize; 4], coefficients: [f64; 6]) -> Self {
        let [particle1, particle2, particle3, particle4] = particles;
        Self {
            particle1,
            particle2,
            particle3,
            particle4,
            coefficients,
        }
    }
}

#[derive(Clone)]
pub struct RBTorsionForce {
    metadata: super::ForceMetadata,
    torsions: Vec<RBTorsionParam>,
    use_pbc: bool,
}

impl RBTorsionForce {
    super::force_metadata!();
    pub fn new() -> Self {
        Self {
            metadata: super::ForceMetadata::new("RBTorsionForce"),

            torsions: Vec::new(),
            use_pbc: false,
        }
    }

    pub fn add_torsion(&mut self, torsion: RBTorsionParam) -> usize {
        self.torsions.push(torsion);
        self.torsions.len() - 1
    }

    pub fn set_torsion_parameters(&mut self, index: usize, torsion: RBTorsionParam) {
        self.torsions[index] = torsion;
    }

    pub fn torsion_parameters(&self, index: usize) -> &RBTorsionParam {
        &self.torsions[index]
    }

    pub fn num_torsions(&self) -> usize {
        self.torsions.len()
    }

    pub fn set_uses_periodic_boundary_conditions(&mut self, use_pbc: bool) {
        self.use_pbc = use_pbc;
    }

    pub fn uses_periodic_boundary_conditions(&self) -> bool {
        self.use_pbc
    }

    pub fn update_parameters_from(&mut self, source: &Self) {
        assert_eq!(
            self.torsions.len(),
            source.torsions.len(),
            "torsion count must match"
        );
        self.torsions = source.torsions.clone();
    }
}

impl Default for RBTorsionForce {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Force<B> for RBTorsionForce {
    super::force_metadata_impl!();
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

        for tor in &self.torsions {
            let indices = [tor.particle1, tor.particle2, tor.particle3, tor.particle4];
            let mut bonds = [
                sub3_pos(&pos, indices[1], indices[0]),
                sub3_pos(&pos, indices[2], indices[1]),
                sub3_pos(&pos, indices[3], indices[2]),
            ];
            if self.use_pbc {
                if let Some(vectors) = box_vectors {
                    for bond in &mut bonds {
                        apply_minimum_image_ortho(bond, vectors);
                    }
                }
            }
            let Some((phi, directions)) = torsion_geometry(&bonds[0], &bonds[1], &bonds[2]) else {
                continue;
            };
            let cosine = -phi.cos();
            let mut value = tor.coefficients[5];
            let mut derivative = 0.0;
            for i in (0..5).rev() {
                derivative = derivative * cosine + value;
                value = value * cosine + tor.coefficients[i];
            }
            energy += value;
            derivative *= phi.sin();
            for (particle, direction) in indices.into_iter().zip(directions) {
                for axis in 0..3 {
                    forces[particle * 3 + axis] += derivative * direction[axis];
                }
            }
        }

        let forces_f32: Vec<f32> = forces.iter().map(|&v| v as f32).collect();
        let forces_tensor =
            Tensor::<B, 1>::from_floats(forces_f32.as_slice(), &device).reshape([n, 3]);
        (forces_tensor, energy)
    }

    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.use_pbc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use burn::backend::NdArray;
    use std::f64::consts::PI;

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
    fn test_rb_torsions() {
        let mut force = RBTorsionForce::new();
        force.add_torsion(RBTorsionParam::new(
            [0, 1, 2, 3],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ));

        let positions = pos_tensor(&[
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ]);
        let (forces_t, energy) = Force::<B>::calc_forces_and_energy(&force, &positions, None);
        let f = get_forces(forces_t);

        let psi = PI / 4.0 - PI;
        let coeffs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let expected_energy: f64 = coeffs
            .iter()
            .enumerate()
            .map(|(i, &c)| c * psi.cos().powi(i as i32))
            .sum();
        let expected_torque: f64 = coeffs
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, &c)| -c * i as f64 * psi.cos().powi(i as i32 - 1) * psi.sin())
            .sum();

        assert_relative_eq!(energy, expected_energy, epsilon = 1e-5);
        assert_relative_eq!(f[2], expected_torque, epsilon = 1e-5);
        assert_relative_eq!(f[10], 0.5 * expected_torque, epsilon = 1e-5);
        assert_relative_eq!(f[11], -0.5 * expected_torque, epsilon = 1e-5);

        for d in 0..3 {
            let total = f[d] + f[3 + d] + f[6 + d] + f[9 + d];
            assert_relative_eq!(total, 0.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_periodic_boundary_conditions() {
        let mut force = RBTorsionForce::new();
        force.set_uses_periodic_boundary_conditions(true);
        force.add_torsion(RBTorsionParam::new(
            [0, 1, 2, 3],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ));

        let box_vectors = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]];
        let positions = pos_tensor(&[
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
        ]);
        let (forces_t, energy) =
            Force::<B>::calc_forces_and_energy(&force, &positions, Some(&box_vectors));
        let f = get_forces(forces_t);

        let psi = PI / 2.0;
        let coeffs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let expected_energy: f64 = coeffs
            .iter()
            .enumerate()
            .map(|(i, &c)| c * psi.cos().powi(i as i32))
            .sum();
        let expected_torque: f64 = coeffs
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, &c)| -c * i as f64 * psi.cos().powi(i as i32 - 1) * psi.sin())
            .sum();

        assert_relative_eq!(energy, expected_energy, epsilon = 1e-5);
        assert_relative_eq!(f[2], expected_torque, epsilon = 1e-5);
        assert_relative_eq!(f[10], -expected_torque, epsilon = 1e-5);

        for d in 0..3 {
            let total = f[d] + f[3 + d] + f[6 + d] + f[9 + d];
            assert_relative_eq!(total, 0.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_set_torsion_parameters() {
        let mut force = RBTorsionForce::new();
        let index = force.add_torsion(RBTorsionParam::new([0, 1, 2, 3], [1.0; 6]));
        force.set_torsion_parameters(index, RBTorsionParam::new([0, 1, 2, 3], [2.0; 6]));

        assert_eq!(force.num_torsions(), 1);
        assert_eq!(force.torsion_parameters(0).coefficients, [2.0; 6]);
    }
}
