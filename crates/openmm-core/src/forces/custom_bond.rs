// CustomBondForce: user-defined energy as a function of bond distance r.
// Energy expression may reference r and per-bond parameters.
// Reference test: openmm/tests/TestCustomBondForce.h
use super::custom_evaluator::CustomEvaluator;
use super::Force;
use crate::expression::Expression;
use crate::forces::harmonic_bond::apply_minimum_image_ortho;
use burn::prelude::*;
use std::collections::HashMap;

#[derive(Clone)]
pub struct CustomBondForce {
    metadata: super::ForceMetadata,
    /// Energy expression as a function of r and per-bond params.
    energy_expr: Expression,
    /// Names of per-bond parameters, in order.
    per_bond_params: Vec<String>,
    bonds: Vec<BondRecord>,
    global_param_names: Vec<String>,
    global_params: HashMap<String, f64>,
    energy_param_deriv_names: Vec<String>,
    use_pbc: bool,
}

#[derive(Clone)]
struct BondRecord {
    p1: usize,
    p2: usize,
    params: Vec<f64>,
}

impl CustomBondForce {
    super::force_metadata!();
    pub fn new(energy: &str) -> Result<Self, crate::expression::ParseError> {
        let expr = Expression::parse(energy)?;
        Ok(Self {
            metadata: super::ForceMetadata::new("CustomBondForce"),
            energy_expr: expr,
            per_bond_params: Vec::new(),
            bonds: Vec::new(),
            global_param_names: Vec::new(),
            global_params: HashMap::new(),
            energy_param_deriv_names: Vec::new(),
            use_pbc: false,
        })
    }

    /// Declare a per-bond parameter by name. Returns parameter index.
    pub fn add_per_bond_parameter(&mut self, name: &str) -> usize {
        self.per_bond_params.push(name.to_string());
        self.per_bond_params.len() - 1
    }

    pub fn num_per_bond_parameters(&self) -> usize {
        self.per_bond_params.len()
    }

    pub fn per_bond_parameter_name(&self, index: usize) -> &str {
        &self.per_bond_params[index]
    }

    /// Add a global (constant) parameter.
    pub fn add_global_parameter(&mut self, name: &str, value: f64) {
        if !self.global_params.contains_key(name) {
            self.global_param_names.push(name.to_string());
        }
        self.global_params.insert(name.to_string(), value);
    }

    pub fn num_global_parameters(&self) -> usize {
        self.global_param_names.len()
    }

    pub fn global_parameter_name(&self, index: usize) -> &str {
        &self.global_param_names[index]
    }

    pub fn global_parameter_default_value(&self, index: usize) -> f64 {
        let name = &self.global_param_names[index];
        self.global_params[name]
    }

    pub fn set_global_parameter_default_value(&mut self, index: usize, value: f64) {
        let name = self.global_param_names[index].clone();
        *self
            .global_params
            .get_mut(&name)
            .expect("unknown global parameter") = value;
    }

    pub fn add_energy_parameter_derivative(&mut self, name: &str) {
        assert!(
            self.global_params.contains_key(name),
            "unknown global parameter"
        );
        if !self
            .energy_param_deriv_names
            .iter()
            .any(|existing| existing == name)
        {
            self.energy_param_deriv_names.push(name.to_string());
        }
    }

    pub fn num_energy_parameter_derivatives(&self) -> usize {
        self.energy_param_deriv_names.len()
    }

    pub fn energy_parameter_derivative_name(&self, index: usize) -> &str {
        &self.energy_param_deriv_names[index]
    }

    pub fn set_global_parameter_value(&mut self, name: &str, value: f64) {
        *self
            .global_params
            .get_mut(name)
            .expect("unknown global parameter") = value;
    }

    /// Add a bond. `params` must match the declared per-bond parameters.
    pub fn add_bond(&mut self, p1: usize, p2: usize, params: Vec<f64>) -> usize {
        assert_eq!(params.len(), self.per_bond_params.len());
        self.bonds.push(BondRecord { p1, p2, params });
        self.bonds.len() - 1
    }

    pub fn bond_parameters(&self, index: usize) -> (usize, usize, &[f64]) {
        let bond = &self.bonds[index];
        (bond.p1, bond.p2, &bond.params)
    }

    pub fn set_bond_parameters(&mut self, index: usize, p1: usize, p2: usize, params: Vec<f64>) {
        assert_eq!(params.len(), self.per_bond_params.len());
        let b = &mut self.bonds[index];
        b.p1 = p1;
        b.p2 = p2;
        b.params = params;
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
            self.per_bond_params, source.per_bond_params,
            "per-bond parameter definitions must match"
        );
        assert_eq!(
            self.bonds.len(),
            source.bonds.len(),
            "bond count must match"
        );
        self.bonds = source.bonds.clone();
    }
}

impl CustomBondForce {
    fn evaluate<B: Backend>(
        &self,
        positions: &Tensor<B, 2>,
        box_vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64, HashMap<String, f64>) {
        let device = positions.device();
        let n = positions.dims()[0];
        let pos: Vec<f64> = positions
            .to_data()
            .convert::<f64>()
            .to_vec::<f64>()
            .unwrap();

        let mut forces = vec![0.0f64; n * 3];
        let mut energy = 0.0f64;
        let mut evaluator = CustomEvaluator::new(
            &self.energy_expr,
            &self.global_params,
            self.per_bond_params.iter().map(String::as_str),
            &["r"],
            &self.energy_param_deriv_names,
        );

        for bond in &self.bonds {
            let i = bond.p1;
            let j = bond.p2;

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

            evaluator.set_coordinates(&[r]);
            evaluator.set_parameters(bond.params.iter().copied());
            energy += evaluator.evaluate();
            let de_dr = evaluator.coordinate_derivative(0);

            let f_over_r = de_dr / r;
            for d in 0..3 {
                let f = f_over_r * delta[d];
                forces[i * 3 + d] += f;
                forces[j * 3 + d] -= f;
            }
        }

        let forces_tensor =
            crate::tensor::from_floats::<B, 1>(forces.as_slice(), &device).reshape([n, 3]);

        (forces_tensor, energy, evaluator.finish())
    }
}

impl<B: Backend> Force<B> for CustomBondForce {
    fn initialize(&mut self, num_particles: usize, _box_vectors: Option<&[[f64; 3]; 3]>) {
        let _ = CustomEvaluator::new(
            &self.energy_expr,
            &self.global_params,
            self.per_bond_params.iter().map(String::as_str),
            &["r"],
            &self.energy_param_deriv_names,
        );
        for record in &self.bonds {
            assert!(
                [record.p1, record.p2].iter().all(|&p| p < num_particles),
                "custom force particle index out of range"
            );
        }
    }
    super::force_metadata_impl!();
    fn bonded_particles(&self) -> Vec<(usize, usize)> {
        self.bonds.iter().flat_map(|p| [(p.p1, p.p2)]).collect()
    }

    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        box_vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64) {
        let (forces, energy, _) = self.evaluate::<B>(positions, box_vectors);
        (forces, energy)
    }

    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.use_pbc
    }

    fn global_parameters(&self) -> HashMap<String, f64> {
        self.global_params.clone()
    }

    fn set_global_parameter(&mut self, name: &str, value: f64) -> bool {
        match self.global_params.get_mut(name) {
            Some(parameter) => {
                *parameter = value;
                true
            }
            None => false,
        }
    }

    fn energy_parameter_derivatives(
        &self,
        positions: &Tensor<B, 2>,
        box_vectors: Option<&[[f64; 3]; 3]>,
    ) -> HashMap<String, f64> {
        self.evaluate::<B>(positions, box_vectors).2
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

    // CustomBondForce with E=k*(r-r0)^2 should give same result as HarmonicBondForce.
    #[test]
    fn test_harmonic_equivalent() {
        let mut force = CustomBondForce::new("k*(r-r0)^2").unwrap();
        force.add_per_bond_parameter("r0");
        force.add_per_bond_parameter("k");
        force.add_bond(0, 1, vec![1.5, 0.8]);

        let positions = pos_tensor(&[[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]);
        let (_, energy) = Force::<B>::calc_forces_and_energy(&force, &positions, None);

        // r=2, r0=1.5, k=0.8 => E = 0.8*(0.5)^2 = 0.2
        assert_relative_eq!(energy, 0.8 * 0.25, epsilon = 1e-4);
    }

    #[test]
    fn test_energy_parameter_derivative_registration() {
        let mut force = CustomBondForce::new("scale*k*(r-r0)^2").unwrap();
        force.add_global_parameter("scale", 1.0);
        force.add_energy_parameter_derivative("scale");
        force.add_energy_parameter_derivative("scale");

        assert_eq!(force.num_energy_parameter_derivatives(), 1);
        assert_eq!(force.energy_parameter_derivative_name(0), "scale");
    }

    #[test]
    fn test_global_parameter_metadata() {
        let mut force = CustomBondForce::new("scale*k*(r-r0)^2").unwrap();
        force.add_global_parameter("scale", 1.0);
        force.add_global_parameter("bias", 2.0);

        assert_eq!(force.num_global_parameters(), 2);
        assert_eq!(force.global_parameter_name(0), "scale");
        assert_eq!(force.global_parameter_name(1), "bias");
        assert_relative_eq!(
            force.global_parameter_default_value(1),
            2.0,
            epsilon = 1e-12
        );

        force.set_global_parameter_default_value(0, 3.5);
        assert_relative_eq!(
            force.global_parameter_default_value(0),
            3.5,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_per_bond_metadata() {
        let mut force = CustomBondForce::new("k*(r-r0)^2").unwrap();
        force.add_per_bond_parameter("r0");
        force.add_per_bond_parameter("k");
        force.add_bond(0, 1, vec![1.0, 2.0]);

        assert_eq!(force.num_per_bond_parameters(), 2);
        assert_eq!(force.per_bond_parameter_name(0), "r0");
        assert_eq!(force.per_bond_parameter_name(1), "k");
        assert_eq!(force.num_bonds(), 1);
    }
}
