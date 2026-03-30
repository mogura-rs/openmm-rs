// PeriodicTorsionForce: E = k * (1 + cos(n*phi - phase))
// phi is the dihedral angle between planes (i,j,k) and (j,k,l).
// Multiple terms per torsion are supported (same bond, different periodicity).
// Reference test: openmm/tests/TestPeriodicTorsionForce.h
use super::Force;
use crate::forces::harmonic_bond::apply_minimum_image_ortho;
use burn::prelude::*;

#[derive(Debug, Clone)]
pub struct TorsionParam {
    pub particle1: usize,
    pub particle2: usize,
    pub particle3: usize,
    pub particle4: usize,
    /// Periodicity (n).
    pub periodicity: i32,
    /// Phase offset in radians.
    pub phase: f64,
    /// Force constant in kJ/mol.
    pub k: f64,
}

impl TorsionParam {
    pub fn new(particles: [usize; 4], periodicity: i32, phase: f64, k: f64) -> Self {
        let [particle1, particle2, particle3, particle4] = particles;
        Self {
            particle1,
            particle2,
            particle3,
            particle4,
            periodicity,
            phase,
            k,
        }
    }
}

#[derive(Clone)]
pub struct PeriodicTorsionForce {
    torsions: Vec<TorsionParam>,
    use_pbc: bool,
}

impl PeriodicTorsionForce {
    pub fn new() -> Self {
        Self {
            torsions: Vec::new(),
            use_pbc: false,
        }
    }

    /// Returns torsion index.
    pub fn add_torsion(&mut self, torsion: TorsionParam) -> usize {
        self.torsions.push(torsion);
        self.torsions.len() - 1
    }

    pub fn set_torsion_parameters(&mut self, index: usize, torsion: TorsionParam) {
        self.torsions[index] = torsion;
    }

    pub fn torsion_parameters(&self, index: usize) -> &TorsionParam {
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

impl Default for PeriodicTorsionForce {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Force<B> for PeriodicTorsionForce {
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
            let i = tor.particle1;
            let j = tor.particle2;
            let k = tor.particle3;
            let l = tor.particle4;

            // Bond vectors.
            let mut b1 = sub3_pos(&pos, j, i); // r_i - r_j
            let mut b2 = sub3_pos(&pos, k, j); // r_j - r_k  (central bond)
            let mut b3 = sub3_pos(&pos, l, k); // r_k - r_l

            if self.use_pbc {
                if let Some(bv) = box_vectors {
                    apply_minimum_image_ortho(&mut b1, bv);
                    apply_minimum_image_ortho(&mut b2, bv);
                    apply_minimum_image_ortho(&mut b3, bv);
                }
            }

            // Normal vectors to the two planes.
            let n1 = cross3(&b1, &b2);
            let n2 = cross3(&b2, &b3);

            let n1_len2 = dot3(&n1, &n1);
            let n2_len2 = dot3(&n2, &n2);
            let n1_len = n1_len2.sqrt();
            let n2_len = n2_len2.sqrt();

            if n1_len < 1e-8 || n2_len < 1e-8 {
                continue; // degenerate torsion
            }

            let cos_phi = dot3(&n1, &n2) / (n1_len * n2_len);
            let cos_phi = cos_phi.clamp(-1.0, 1.0);

            // Sign of phi: positive when b1, b2, b3 form a right-handed system.
            let sign = if dot3(&cross3(&n1, &n2), &b2) < 0.0 {
                -1.0
            } else {
                1.0
            };
            let phi = sign * cos_phi.acos();

            let n = tor.periodicity as f64;
            let arg = n * phi - tor.phase;
            energy += tor.k * (1.0 + arg.cos());

            // dE/dphi = -k * n * sin(n*phi - phase)
            let de_dphi = -tor.k * n * arg.sin();

            // Force on each atom via chain rule through n1, n2.
            // See: Blondel & Karplus, J. Comput. Chem. 17, 1132 (1996).
            let b2_len = dot3(&b2, &b2).sqrt();

            let f1 = scale3(&cross3(&b1, &b2), -de_dphi * b2_len / n1_len2);
            let f4 = scale3(&cross3(&b2, &b3), de_dphi * b2_len / n2_len2);

            // f2 and f3 preserve momentum.
            let b1_dot_b2 = dot3(&b1, &b2);
            let b3_dot_b2 = dot3(&b3, &b2);
            let b2_len2 = b2_len * b2_len;

            let f2 = add3(
                &scale3(&f1, -1.0 + b1_dot_b2 / b2_len2),
                &scale3(&f4, b3_dot_b2 / b2_len2),
            );
            let f3 = add3(
                &scale3(&f4, -1.0 + b3_dot_b2 / b2_len2),
                &scale3(&f1, b1_dot_b2 / b2_len2),
            );

            for d in 0..3 {
                forces[i * 3 + d] += f1[d];
                forces[j * 3 + d] -= f2[d];
                forces[k * 3 + d] -= f3[d];
                forces[l * 3 + d] += f4[d];
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

// --- Vector helpers ---

#[inline]
pub fn sub3_pos(pos: &[f64], a: usize, b: usize) -> [f64; 3] {
    [
        pos[a * 3] - pos[b * 3],
        pos[a * 3 + 1] - pos[b * 3 + 1],
        pos[a * 3 + 2] - pos[b * 3 + 2],
    ]
}

#[inline]
pub fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub fn scale3(a: &[f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub fn add3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
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

    // Mirrors TestPeriodicTorsionForce.h::testTorsions()
    // phi=0 => E = k*(1 + cos(n*0 - phase))
    #[test]
    fn test_energy_at_zero_dihedral() {
        let mut force = PeriodicTorsionForce::new();
        // fully extended: phi = pi
        force.add_torsion(TorsionParam::new([0, 1, 2, 3], 2, 0.0, 3.0));

        let positions = pos_tensor(&[
            [-1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, -1.0],
        ]);
        let (_, energy) = Force::<B>::calc_forces_and_energy(&force, &positions, None);

        // phi = pi, n=2, phase=0 => E = 3*(1 + cos(2*pi)) = 3*2 = 6
        assert_relative_eq!(energy, 3.0 * (1.0 + (2.0 * PI).cos()), epsilon = 1e-4);
    }

    // Verify Newton's third law.
    #[test]
    fn test_conservation() {
        let mut force = PeriodicTorsionForce::new();
        force.add_torsion(TorsionParam::new([0, 1, 2, 3], 3, PI / 4.0, 1.5));

        let positions = pos_tensor(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 1.0],
        ]);
        let (forces_t, _) = Force::<B>::calc_forces_and_energy(&force, &positions, None);
        let f: Vec<f64> = forces_t
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(|v| v as f64)
            .collect();

        for d in 0..3 {
            let total: f64 = (0..4).map(|p| f[p * 3 + d]).sum();
            assert_relative_eq!(total, 0.0, epsilon = 1e-4);
        }
    }
}
