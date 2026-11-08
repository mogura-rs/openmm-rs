// CustomTorsionForce: user-defined energy as a function of dihedral angle theta (radians).
// Reference test: openmm/tests/TestCustomTorsionForce.h
use super::Force;
use crate::expression::Expression;
use crate::forces::harmonic_bond::apply_minimum_image_ortho;
use crate::forces::periodic_torsion::{sub3_pos, torsion_geometry};
use burn::prelude::*;
use std::collections::HashMap;

#[derive(Clone)]
pub struct CustomTorsionForce {
    metadata: super::ForceMetadata,
    energy_expr: Expression,
    per_torsion_params: Vec<String>,
    torsions: Vec<TorsionRecord>,
    global_param_names: Vec<String>,
    global_params: HashMap<String, f64>,
    energy_param_deriv_names: Vec<String>,
    use_pbc: bool,
}

#[derive(Clone)]
struct TorsionRecord {
    p1: usize,
    p2: usize,
    p3: usize,
    p4: usize,
    params: Vec<f64>,
}

impl CustomTorsionForce {
    super::force_metadata!();
    pub fn new(energy: &str) -> Result<Self, crate::expression::ParseError> {
        Ok(Self {
            metadata: super::ForceMetadata::new("CustomTorsionForce"),
            energy_expr: Expression::parse(energy)?,
            per_torsion_params: Vec::new(),
            torsions: Vec::new(),
            global_param_names: Vec::new(),
            global_params: HashMap::new(),
            energy_param_deriv_names: Vec::new(),
            use_pbc: false,
        })
    }

    pub fn add_per_torsion_parameter(&mut self, name: &str) -> usize {
        self.per_torsion_params.push(name.to_string());
        self.per_torsion_params.len() - 1
    }

    pub fn num_per_torsion_parameters(&self) -> usize {
        self.per_torsion_params.len()
    }

    pub fn per_torsion_parameter_name(&self, index: usize) -> &str {
        &self.per_torsion_params[index]
    }

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

    pub fn add_torsion(
        &mut self,
        p1: usize,
        p2: usize,
        p3: usize,
        p4: usize,
        params: Vec<f64>,
    ) -> usize {
        assert_eq!(params.len(), self.per_torsion_params.len());
        self.torsions.push(TorsionRecord {
            p1,
            p2,
            p3,
            p4,
            params,
        });
        self.torsions.len() - 1
    }

    pub fn num_torsions(&self) -> usize {
        self.torsions.len()
    }

    pub fn torsion_parameters(&self, index: usize) -> (usize, usize, usize, usize, &[f64]) {
        let torsion = &self.torsions[index];
        (
            torsion.p1,
            torsion.p2,
            torsion.p3,
            torsion.p4,
            &torsion.params,
        )
    }

    pub fn set_torsion_parameters(
        &mut self,
        index: usize,
        p1: usize,
        p2: usize,
        p3: usize,
        p4: usize,
        params: Vec<f64>,
    ) {
        assert_eq!(params.len(), self.per_torsion_params.len());
        let torsion = &mut self.torsions[index];
        torsion.p1 = p1;
        torsion.p2 = p2;
        torsion.p3 = p3;
        torsion.p4 = p4;
        torsion.params = params;
    }

    pub fn set_uses_periodic_boundary_conditions(&mut self, use_pbc: bool) {
        self.use_pbc = use_pbc;
    }

    pub fn uses_periodic_boundary_conditions(&self) -> bool {
        self.use_pbc
    }

    pub fn update_parameters_from(&mut self, source: &Self) {
        assert_eq!(
            self.per_torsion_params, source.per_torsion_params,
            "per-torsion parameter definitions must match"
        );
        assert_eq!(
            self.torsions.len(),
            source.torsions.len(),
            "torsion count must match"
        );
        self.torsions = source.torsions.clone();
    }
}

impl<B: Backend> Force<B> for CustomTorsionForce {
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
            .convert::<f64>()
            .to_vec::<f64>()
            .unwrap();

        let mut forces = vec![0.0f64; n * 3];
        let mut energy = 0.0f64;

        for tor in &self.torsions {
            let i = tor.p1;
            let j = tor.p2;
            let k = tor.p3;
            let l = tor.p4;

            let mut b1 = sub3_pos(&pos, j, i);
            let mut b2 = sub3_pos(&pos, k, j);
            let mut b3 = sub3_pos(&pos, l, k);

            if self.use_pbc {
                if let Some(bv) = box_vectors {
                    apply_minimum_image_ortho(&mut b1, bv);
                    apply_minimum_image_ortho(&mut b2, bv);
                    apply_minimum_image_ortho(&mut b3, bv);
                }
            }

            let Some((phi, directions)) = torsion_geometry(&b1, &b2, &b3) else {
                continue;
            };

            let mut vars: HashMap<&str, f64> = HashMap::new();
            vars.insert("theta", phi);
            for (name, &val) in &self.global_params {
                vars.insert(name.as_str(), val);
            }
            for (name, &val) in self.per_torsion_params.iter().zip(&tor.params) {
                vars.insert(name.as_str(), val);
            }

            energy += self.energy_expr.compute(&vars).unwrap_or(0.0);

            // Numerical dE/dphi
            let dph = 1e-5_f64;
            vars.insert("theta", phi + dph);
            let ep = self.energy_expr.compute(&vars).unwrap_or(0.0);
            vars.insert("theta", phi - dph);
            let em = self.energy_expr.compute(&vars).unwrap_or(0.0);
            let de_dphi = (ep - em) / (2.0 * dph);

            for (particle, direction) in [i, j, k, l].into_iter().zip(directions) {
                for axis in 0..3 {
                    forces[particle * 3 + axis] += de_dphi * direction[axis];
                }
            }
        }

        (
            crate::tensor::from_floats::<B, 1>(forces.as_slice(), &device).reshape([n, 3]),
            energy,
        )
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
        let mut derivatives = HashMap::new();
        let step = 1e-6;
        for name in &self.energy_param_deriv_names {
            let value = *self
                .global_params
                .get(name)
                .expect("unknown global parameter");
            let mut plus = self.clone();
            plus.global_params.insert(name.clone(), value + step);
            let energy_plus = Force::<B>::calc_forces_and_energy(&plus, positions, box_vectors).1;

            let mut minus = self.clone();
            minus.global_params.insert(name.clone(), value - step);
            let energy_minus = Force::<B>::calc_forces_and_energy(&minus, positions, box_vectors).1;

            derivatives.insert(name.clone(), (energy_plus - energy_minus) / (2.0 * step));
        }
        derivatives
    }
}
