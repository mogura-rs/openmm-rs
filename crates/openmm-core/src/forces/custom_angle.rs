// CustomAngleForce: user-defined energy as a function of angle theta (radians).
// Reference test: openmm/tests/TestCustomAngleForce.h
use super::Force;
use crate::expression::Expression;
use crate::forces::harmonic_angle::dot3 as dot;
use crate::forces::harmonic_bond::apply_minimum_image_ortho;
use burn::prelude::*;
use std::collections::HashMap;

#[derive(Clone)]
pub struct CustomAngleForce {
    metadata: super::ForceMetadata,
    energy_expr: Expression,
    per_angle_params: Vec<String>,
    angles: Vec<AngleRecord>,
    global_param_names: Vec<String>,
    global_params: HashMap<String, f64>,
    energy_param_deriv_names: Vec<String>,
    use_pbc: bool,
}

#[derive(Clone)]
struct AngleRecord {
    p1: usize,
    p2: usize, // vertex
    p3: usize,
    params: Vec<f64>,
}

impl CustomAngleForce {
    super::force_metadata!();
    pub fn new(energy: &str) -> Result<Self, crate::expression::ParseError> {
        Ok(Self {
            metadata: super::ForceMetadata::new("CustomAngleForce"),
            energy_expr: Expression::parse(energy)?,
            per_angle_params: Vec::new(),
            angles: Vec::new(),
            global_param_names: Vec::new(),
            global_params: HashMap::new(),
            energy_param_deriv_names: Vec::new(),
            use_pbc: false,
        })
    }

    pub fn add_per_angle_parameter(&mut self, name: &str) -> usize {
        self.per_angle_params.push(name.to_string());
        self.per_angle_params.len() - 1
    }

    pub fn num_per_angle_parameters(&self) -> usize {
        self.per_angle_params.len()
    }

    pub fn per_angle_parameter_name(&self, index: usize) -> &str {
        &self.per_angle_params[index]
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

    pub fn add_angle(&mut self, p1: usize, p2: usize, p3: usize, params: Vec<f64>) -> usize {
        assert_eq!(params.len(), self.per_angle_params.len());
        self.angles.push(AngleRecord { p1, p2, p3, params });
        self.angles.len() - 1
    }

    pub fn num_angles(&self) -> usize {
        self.angles.len()
    }

    pub fn angle_parameters(&self, index: usize) -> (usize, usize, usize, &[f64]) {
        let angle = &self.angles[index];
        (angle.p1, angle.p2, angle.p3, &angle.params)
    }

    pub fn set_angle_parameters(
        &mut self,
        index: usize,
        p1: usize,
        p2: usize,
        p3: usize,
        params: Vec<f64>,
    ) {
        assert_eq!(params.len(), self.per_angle_params.len());
        let angle = &mut self.angles[index];
        angle.p1 = p1;
        angle.p2 = p2;
        angle.p3 = p3;
        angle.params = params;
    }

    pub fn set_uses_periodic_boundary_conditions(&mut self, use_pbc: bool) {
        self.use_pbc = use_pbc;
    }

    pub fn uses_periodic_boundary_conditions(&self) -> bool {
        self.use_pbc
    }

    pub fn update_parameters_from(&mut self, source: &Self) {
        assert_eq!(
            self.per_angle_params, source.per_angle_params,
            "per-angle parameter definitions must match"
        );
        assert_eq!(
            self.angles.len(),
            source.angles.len(),
            "angle count must match"
        );
        self.angles = source.angles.clone();
    }
}

impl<B: Backend> Force<B> for CustomAngleForce {
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

        for ang in &self.angles {
            let i = ang.p1;
            let j = ang.p2;
            let k = ang.p3;

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

            let rji_len = dot(&rji, &rji).sqrt();
            let rjk_len = dot(&rjk, &rjk).sqrt();
            let cos_theta = (dot(&rji, &rjk) / (rji_len * rjk_len)).clamp(-1.0, 1.0);
            let theta = cos_theta.acos();

            let mut vars: HashMap<&str, f64> = HashMap::new();
            vars.insert("theta", theta);
            for (name, &val) in &self.global_params {
                vars.insert(name.as_str(), val);
            }
            for (name, &val) in self.per_angle_params.iter().zip(&ang.params) {
                vars.insert(name.as_str(), val);
            }

            energy += self.energy_expr.compute(&vars).unwrap_or(0.0);

            // Numerical dE/dtheta.
            let dth = 1e-5_f64;
            vars.insert("theta", theta + dth);
            let e_plus = self.energy_expr.compute(&vars).unwrap_or(0.0);
            vars.insert("theta", theta - dth);
            let e_minus = self.energy_expr.compute(&vars).unwrap_or(0.0);
            let de_dtheta = (e_plus - e_minus) / (2.0 * dth);

            let sin_theta = (1.0_f64 - cos_theta * cos_theta).sqrt().max(1e-8);
            for d in 0..3 {
                let fi = de_dtheta * (cos_theta * rji[d] / rji_len - rjk[d] / rjk_len)
                    / (rji_len * sin_theta);
                let fk = de_dtheta * (cos_theta * rjk[d] / rjk_len - rji[d] / rji_len)
                    / (rjk_len * sin_theta);
                forces[i * 3 + d] -= fi;
                forces[k * 3 + d] -= fk;
                forces[j * 3 + d] += fi + fk;
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
