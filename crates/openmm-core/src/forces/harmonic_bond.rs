// HarmonicBondForce: E = 0.5 * k * (r - r0)^2
// Reference test: openmm/tests/TestHarmonicBondForce.h
use super::Force;
use burn::prelude::*;

#[derive(Debug, Clone)]
pub struct BondParam {
    pub particle1: usize,
    pub particle2: usize,
    /// Equilibrium distance in nm.
    pub length: f64,
    /// Force constant in kJ/mol/nm^2.
    pub k: f64,
}

#[derive(Clone)]
pub struct HarmonicBondForce {
    metadata: super::ForceMetadata,
    bonds: Vec<BondParam>,
    use_pbc: bool,
}

impl HarmonicBondForce {
    super::force_metadata!();
    pub fn new() -> Self {
        Self {
            metadata: super::ForceMetadata::new("HarmonicBondForce"),

            bonds: Vec::new(),
            use_pbc: false,
        }
    }

    /// Returns bond index.
    pub fn add_bond(&mut self, p1: usize, p2: usize, length: f64, k: f64) -> usize {
        self.bonds.push(BondParam {
            particle1: p1,
            particle2: p2,
            length,
            k,
        });
        self.bonds.len() - 1
    }

    pub fn set_bond_parameters(&mut self, index: usize, p1: usize, p2: usize, length: f64, k: f64) {
        self.bonds[index] = BondParam {
            particle1: p1,
            particle2: p2,
            length,
            k,
        };
    }

    pub fn bond_parameters(&self, index: usize) -> &BondParam {
        &self.bonds[index]
    }

    pub fn num_bonds(&self) -> usize {
        self.bonds.len()
    }

    pub fn set_uses_periodic_boundary_conditions(&mut self, use_pbc: bool) {
        self.use_pbc = use_pbc;
    }

    pub fn uses_periodic_boundary_conditions(&self) -> bool {
        self.use_pbc
    }

    pub fn update_parameters_from(&mut self, source: &Self) {
        assert_eq!(
            self.bonds.len(),
            source.bonds.len(),
            "bond count must match"
        );
        for (bond, other) in self.bonds.iter().zip(&source.bonds) {
            assert_eq!(
                (bond.particle1, bond.particle2),
                (other.particle1, other.particle2),
                "bond particles cannot change in a Context"
            );
        }
        self.bonds.clone_from(&source.bonds);
    }
}

impl Default for HarmonicBondForce {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Force<B> for HarmonicBondForce {
    super::force_metadata_impl!();
    fn bonded_particles(&self) -> Vec<(usize, usize)> {
        self.bonds
            .iter()
            .flat_map(|p| [(p.particle1, p.particle2)])
            .collect()
    }

    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        box_vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64) {
        let device = positions.device();
        let n = positions.dims()[0];
        let pos: Vec<f64> = positions
            .to_data()
            .convert::<f64>()
            .to_vec::<f64>()
            .unwrap();

        let mut forces = vec![0.0f64; n * 3];
        let mut energy = 0.0f64;

        for bond in &self.bonds {
            let i = bond.particle1;
            let j = bond.particle2;

            let mut delta = [
                pos[j * 3] - pos[i * 3],
                pos[j * 3 + 1] - pos[i * 3 + 1],
                pos[j * 3 + 2] - pos[i * 3 + 2],
            ];

            if self.use_pbc {
                if let Some(bv) = box_vectors {
                    apply_minimum_image_ortho(&mut delta, bv);
                }
            }

            let r = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
            let dr = r - bond.length;

            energy += 0.5 * bond.k * dr * dr;

            let f_mag = bond.k * dr / r;
            for d in 0..3 {
                let f = f_mag * delta[d];
                forces[i * 3 + d] += f;
                forces[j * 3 + d] -= f;
            }
        }

        let forces_tensor =
            crate::tensor::from_floats::<B, 1>(forces.as_slice(), &device).reshape([n, 3]);

        (forces_tensor, energy)
    }

    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.use_pbc
    }
}

/// Minimum image for a reduced triclinic box. delta is r_j - r_i.
pub fn apply_minimum_image_ortho(delta: &mut [f64; 3], bv: &[[f64; 3]; 3]) {
    for d in (0..3).rev() {
        let shift = (delta[d] / bv[d][d] + 0.5).floor();
        for axis in 0..=d {
            delta[axis] -= shift * bv[d][axis];
        }
    }
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

    // Mirrors TestHarmonicBondForce.h::testBonds()
    #[test]
    fn test_bonds() {
        let mut force = HarmonicBondForce::new();
        force.add_bond(0, 1, 1.5, 0.8);
        force.add_bond(1, 2, 1.2, 0.7);

        // positions: p0=(0,2,0), p1=(0,0,0), p2=(1,0,0)
        let positions = pos_tensor(&[[0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let (forces_t, energy) = Force::<B>::calc_forces_and_energy(&force, &positions, None);
        let f = get_forces(forces_t);

        const TOL: f64 = 1e-5;
        // p0: r=2, dr=0.5, F_y = -k*dr = -0.4
        assert_relative_eq!(f[1], -0.8 * 0.5, epsilon = TOL);
        // p2: r=1, dr=-0.2, F_x = +k*|dr| = 0.14
        assert_relative_eq!(f[6], 0.7 * 0.2, epsilon = TOL);
        // p1: Newton's third law
        assert_relative_eq!(f[3], -f[0] - f[6], epsilon = TOL);
        // E = 0.5*0.8*0.5^2 + 0.5*0.7*0.2^2
        assert_relative_eq!(energy, 0.5 * 0.8 * 0.25 + 0.5 * 0.7 * 0.04, epsilon = TOL);
    }

    // Mirrors TestHarmonicBondForce.h::testPeriodic()
    #[test]
    fn test_periodic() {
        let mut force = HarmonicBondForce::new();
        force.add_bond(0, 1, 1.2, 0.8);
        force.set_uses_periodic_boundary_conditions(true);

        let bv: [[f64; 3]; 3] = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]];
        // p0=(0,2,0), p1=(0,0,0): delta=(0,-2,0), min-image=(0,1,0), r=1
        let positions = pos_tensor(&[[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]);
        let (_, energy) = Force::<B>::calc_forces_and_energy(&force, &positions, Some(&bv));
        // r=1, dr=-0.2, E=0.5*0.8*0.04
        assert_relative_eq!(energy, 0.5 * 0.8 * 0.04, epsilon = 1e-5);
    }
}
