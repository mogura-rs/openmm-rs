// CustomExternalForce: user-defined energy per particle as a function of (x, y, z).
// Reference test: openmm/tests/TestCustomExternalForce.h
use super::Force;
use crate::expression::Expression;
use burn::prelude::*;
use std::collections::HashMap;

#[derive(Clone)]
pub struct CustomExternalForce {
    energy_expr: Expression,
    per_particle_params: Vec<String>,
    particles: Vec<ParticleRecord>,
    global_param_names: Vec<String>,
    global_params: HashMap<String, f64>,
    energy_param_deriv_names: Vec<String>,
}

#[derive(Clone)]
struct ParticleRecord {
    index: usize,
    params: Vec<f64>,
}

impl CustomExternalForce {
    pub fn new(energy: &str) -> Result<Self, crate::expression::ParseError> {
        Ok(Self {
            energy_expr: Expression::parse(energy)?,
            per_particle_params: Vec::new(),
            particles: Vec::new(),
            global_param_names: Vec::new(),
            global_params: HashMap::new(),
            energy_param_deriv_names: Vec::new(),
        })
    }

    pub fn add_per_particle_parameter(&mut self, name: &str) -> usize {
        self.per_particle_params.push(name.to_string());
        self.per_particle_params.len() - 1
    }

    pub fn num_per_particle_parameters(&self) -> usize {
        self.per_particle_params.len()
    }

    pub fn per_particle_parameter_name(&self, index: usize) -> &str {
        &self.per_particle_params[index]
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

    pub fn add_particle(&mut self, index: usize, params: Vec<f64>) -> usize {
        assert_eq!(params.len(), self.per_particle_params.len());
        self.particles.push(ParticleRecord { index, params });
        self.particles.len() - 1
    }

    pub fn num_particles(&self) -> usize {
        self.particles.len()
    }

    pub fn particle_parameters(&self, index: usize) -> (usize, &[f64]) {
        let particle = &self.particles[index];
        (particle.index, &particle.params)
    }

    pub fn set_particle_parameters(&mut self, record_index: usize, index: usize, params: Vec<f64>) {
        assert_eq!(params.len(), self.per_particle_params.len());
        let r = &mut self.particles[record_index];
        r.index = index;
        r.params = params;
    }

    pub fn uses_periodic_boundary_conditions(&self) -> bool {
        false
    }

    pub fn update_parameters_from(&mut self, source: &Self) {
        assert_eq!(
            self.per_particle_params, source.per_particle_params,
            "per-particle parameter definitions must match"
        );
        assert_eq!(
            self.particles.len(),
            source.particles.len(),
            "particle count must match"
        );
        self.particles = source.particles.clone();
    }
}

impl<B: Backend> Force<B> for CustomExternalForce {
    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        _box_vectors: Option<&[[f64; 3]; 3]>,
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

        for rec in &self.particles {
            let i = rec.index;
            let x = pos[i * 3];
            let y = pos[i * 3 + 1];
            let z = pos[i * 3 + 2];

            let mut vars: HashMap<&str, f64> = HashMap::new();
            vars.insert("x", x);
            vars.insert("y", y);
            vars.insert("z", z);
            for (name, &val) in &self.global_params {
                vars.insert(name.as_str(), val);
            }
            for (name, &val) in self.per_particle_params.iter().zip(&rec.params) {
                vars.insert(name.as_str(), val);
            }

            energy += self.energy_expr.compute(&vars).unwrap_or(0.0);

            // Numerical gradient dE/dx, dE/dy, dE/dz.
            let dr = 1e-5_f64;
            for (d, coord) in ["x", "y", "z"].iter().enumerate() {
                vars.insert(coord, [x, y, z][d] + dr);
                let ep = self.energy_expr.compute(&vars).unwrap_or(0.0);
                vars.insert(coord, [x, y, z][d] - dr);
                let em = self.energy_expr.compute(&vars).unwrap_or(0.0);
                vars.insert(coord, [x, y, z][d]); // restore
                forces[i * 3 + d] -= (ep - em) / (2.0 * dr);
            }
        }

        let f32v: Vec<f32> = forces.iter().map(|&v| v as f32).collect();
        (
            Tensor::<B, 1>::from_floats(f32v.as_slice(), &device).reshape([n, 3]),
            energy,
        )
    }

    fn uses_periodic_boundary_conditions(&self) -> bool {
        false
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use burn::backend::NdArray;

    type B = NdArray;

    // E = k*x^2 => F_x = -2*k*x
    #[test]
    fn test_harmonic_potential() {
        let mut force = CustomExternalForce::new("k*x^2").unwrap();
        force.add_global_parameter("k", 1.0);
        force.add_particle(0, vec![]);

        let flat = vec![2.0f32, 0.0, 0.0];
        let pos =
            Tensor::<NdArray, 1>::from_floats(flat.as_slice(), &Default::default()).reshape([1, 3]);
        let (forces_t, energy) = Force::<B>::calc_forces_and_energy(&force, &pos, None);

        assert_relative_eq!(energy, 4.0, epsilon = 1e-4);
        let f: Vec<f64> = forces_t
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(|v| v as f64)
            .collect();
        assert_relative_eq!(f[0], -4.0, epsilon = 1e-3); // -dE/dx = -2*k*x = -4
    }

    #[test]
    fn test_energy_parameter_derivative_registration() {
        let mut force = CustomExternalForce::new("k*x^2").unwrap();
        force.add_global_parameter("k", 1.0);
        force.add_energy_parameter_derivative("k");
        force.add_energy_parameter_derivative("k");

        assert_eq!(force.num_energy_parameter_derivatives(), 1);
        assert_eq!(force.energy_parameter_derivative_name(0), "k");
    }

    #[test]
    fn test_global_parameter_metadata() {
        let mut force = CustomExternalForce::new("k*x^2").unwrap();
        force.add_global_parameter("k", 1.0);
        force.add_global_parameter("offset", 2.0);

        assert_eq!(force.num_global_parameters(), 2);
        assert_eq!(force.global_parameter_name(0), "k");
        assert_eq!(force.global_parameter_name(1), "offset");
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
    fn test_per_particle_metadata() {
        let mut force = CustomExternalForce::new("k*x^2").unwrap();
        force.add_per_particle_parameter("k");
        force.add_particle(0, vec![1.0]);

        assert_eq!(force.num_per_particle_parameters(), 1);
        assert_eq!(force.per_particle_parameter_name(0), "k");
        assert_eq!(force.num_particles(), 1);
    }
}
