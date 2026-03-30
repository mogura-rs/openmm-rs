// HarmonicAngleForce: E = 0.5 * k * (theta - theta0)^2
// theta is the angle at particle2 (vertex), formed by particles 1-2-3.
// Reference test: openmm/tests/TestHarmonicAngleForce.h
use super::Force;
use crate::forces::harmonic_bond::apply_minimum_image_ortho;
use burn::prelude::*;

#[derive(Debug, Clone)]
pub struct AngleParam {
    pub particle1: usize,
    pub particle2: usize, // vertex
    pub particle3: usize,
    /// Equilibrium angle in radians.
    pub angle: f64,
    /// Force constant in kJ/mol/rad^2.
    pub k: f64,
}

#[derive(Clone)]
pub struct HarmonicAngleForce {
    angles: Vec<AngleParam>,
    use_pbc: bool,
}

impl HarmonicAngleForce {
    pub fn new() -> Self {
        Self {
            angles: Vec::new(),
            use_pbc: false,
        }
    }

    /// Returns angle index.
    pub fn add_angle(&mut self, p1: usize, p2: usize, p3: usize, angle: f64, k: f64) -> usize {
        self.angles.push(AngleParam {
            particle1: p1,
            particle2: p2,
            particle3: p3,
            angle,
            k,
        });
        self.angles.len() - 1
    }

    pub fn set_angle_parameters(
        &mut self,
        index: usize,
        p1: usize,
        p2: usize,
        p3: usize,
        angle: f64,
        k: f64,
    ) {
        self.angles[index] = AngleParam {
            particle1: p1,
            particle2: p2,
            particle3: p3,
            angle,
            k,
        };
    }

    pub fn angle_parameters(&self, index: usize) -> &AngleParam {
        &self.angles[index]
    }

    pub fn num_angles(&self) -> usize {
        self.angles.len()
    }

    pub fn set_uses_periodic_boundary_conditions(&mut self, use_pbc: bool) {
        self.use_pbc = use_pbc;
    }

    pub fn uses_periodic_boundary_conditions(&self) -> bool {
        self.use_pbc
    }

    pub fn update_parameters_from(&mut self, source: &Self) {
        assert_eq!(
            self.angles.len(),
            source.angles.len(),
            "angle count must match"
        );
        self.angles = source.angles.clone();
    }
}

impl Default for HarmonicAngleForce {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Force<B> for HarmonicAngleForce {
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

        for ang in &self.angles {
            let i = ang.particle1;
            let j = ang.particle2; // vertex
            let k = ang.particle3;

            // Vectors from vertex j to endpoints.
            let mut rji = [
                pos[i * 3] - pos[j * 3],
                pos[i * 3 + 1] - pos[j * 3 + 1],
                pos[i * 3 + 2] - pos[j * 3 + 2],
            ];
            let mut rjk = [
                pos[k * 3] - pos[j * 3],
                pos[k * 3 + 1] - pos[j * 3 + 1],
                pos[k * 3 + 2] - pos[j * 3 + 2],
            ];

            if self.use_pbc {
                if let Some(bv) = box_vectors {
                    apply_minimum_image_ortho(&mut rji, bv);
                    apply_minimum_image_ortho(&mut rjk, bv);
                }
            }

            let rji_len = dot3(&rji, &rji).sqrt();
            let rjk_len = dot3(&rjk, &rjk).sqrt();

            // Clamp cos to [-1, 1] to avoid NaN from acos.
            let cos_theta = (dot3(&rji, &rjk) / (rji_len * rjk_len)).clamp(-1.0, 1.0);
            let theta = cos_theta.acos();
            let d_theta = theta - ang.angle;

            energy += 0.5 * ang.k * d_theta * d_theta;

            // Force on particle i:
            // dE/dr_i = (dE/dtheta) * (dtheta/dr_i)
            // dtheta/dr_i = (cos_theta * rji_hat - rjk_hat) / (|rji| * sin_theta)
            let sin_theta = (1.0 - cos_theta * cos_theta).sqrt().max(1e-8);
            let de_dtheta = ang.k * d_theta;

            for d in 0..3 {
                let fi = de_dtheta * (cos_theta * rji[d] / rji_len - rjk[d] / rjk_len)
                    / (rji_len * sin_theta);
                let fk = de_dtheta * (cos_theta * rjk[d] / rjk_len - rji[d] / rji_len)
                    / (rjk_len * sin_theta);

                forces[i * 3 + d] -= fi;
                forces[k * 3 + d] -= fk;
                forces[j * 3 + d] += fi + fk; // Newton's third law
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

#[inline]
pub fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use burn::backend::NdArray;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

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

    // Equilibrium angle: energy and forces should be zero.
    #[test]
    fn test_equilibrium() {
        let mut force = HarmonicAngleForce::new();
        force.add_angle(0, 1, 2, FRAC_PI_2, 1.0);

        let positions = pos_tensor(&[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let (forces_t, energy) = Force::<B>::calc_forces_and_energy(&force, &positions, None);
        let f = get_forces(forces_t);

        assert_relative_eq!(energy, 0.0, epsilon = 1e-5);
        for &fi in &f {
            assert_relative_eq!(fi, 0.0, epsilon = 1e-4);
        }
    }

    // Displaced angle: verify energy formula.
    #[test]
    fn test_displaced_angle() {
        let mut force = HarmonicAngleForce::new();
        force.add_angle(0, 1, 2, FRAC_PI_2, 2.0);

        // p0=(1,0,0), p1=(0,0,0), p2=(1,1,0) => theta = pi/4
        let positions = pos_tensor(&[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]);
        let (_, energy) = Force::<B>::calc_forces_and_energy(&force, &positions, None);

        let expected = 0.5 * 2.0 * (FRAC_PI_4 - FRAC_PI_2).powi(2);
        assert_relative_eq!(energy, expected, epsilon = 1e-5);
    }

    // Verify Newton's third law: sum of forces = 0.
    #[test]
    fn test_conservation() {
        let mut force = HarmonicAngleForce::new();
        force.add_angle(0, 1, 2, FRAC_PI_2, 1.5);

        let positions = pos_tensor(&[[1.2, 0.3, 0.0], [0.0, 0.0, 0.0], [0.4, 1.1, 0.0]]);
        let (forces_t, _) = Force::<B>::calc_forces_and_energy(&force, &positions, None);
        let f = get_forces(forces_t);

        for d in 0..3 {
            let total: f64 = f[d] + f[3 + d] + f[6 + d];
            assert_relative_eq!(total, 0.0, epsilon = 1e-4);
        }
    }
}
