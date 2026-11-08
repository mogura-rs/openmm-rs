use super::{
    custom_evaluator::CustomEvaluator, harmonic_bond::apply_minimum_image_ortho,
    neighbors::NeighborCells, Force,
};
use crate::expression::{Expression, ParseError};
use burn::prelude::*;
use std::collections::{HashMap, HashSet};

pub use super::custom_nonbonded::CustomNonbondedMethod as CustomHbondMethod;
const COORDINATES: [&str; 18] = [
    "\0x1", "\0y1", "\0z1", "\0x2", "\0y2", "\0z2", "\0x3", "\0y3", "\0z3", "\0x4", "\0y4", "\0z4",
    "\0x5", "\0y5", "\0z5", "\0x6", "\0y6", "\0z6",
];

#[derive(Clone)]
struct Group {
    atoms: [Option<usize>; 3],
    params: Vec<f64>,
}

#[derive(Clone)]
pub struct CustomHbondForce {
    metadata: super::ForceMetadata,
    energy_expr: Expression,
    energy_source: String,
    donor_params: Vec<String>,
    acceptor_params: Vec<String>,
    donors: Vec<Group>,
    acceptors: Vec<Group>,
    global_param_names: Vec<String>,
    global_params: HashMap<String, f64>,
    exclusions: Vec<(usize, usize)>,
    exclusion_set: HashSet<(usize, usize)>,
    method: CustomHbondMethod,
    cutoff: f64,
    neighbor_cells: NeighborCells,
    use_neighbor_cells: bool,
}

impl CustomHbondForce {
    super::force_metadata!();
    pub fn new(energy: &str) -> Result<Self, ParseError> {
        Ok(Self {
            metadata: super::ForceMetadata::new("CustomHbondForce"),
            energy_expr: Expression::parse_hbond(energy)?,
            energy_source: energy.to_owned(),
            donor_params: Vec::new(),
            acceptor_params: Vec::new(),
            donors: Vec::new(),
            acceptors: Vec::new(),
            global_param_names: Vec::new(),
            global_params: HashMap::new(),
            exclusions: Vec::new(),
            exclusion_set: HashSet::new(),
            method: CustomHbondMethod::NoCutoff,
            cutoff: 1.0,
            neighbor_cells: NeighborCells::default(),
            use_neighbor_cells: true,
        })
    }
    pub fn energy_function(&self) -> &str {
        &self.energy_source
    }
    pub fn set_energy_function(&mut self, energy: &str) -> Result<(), String> {
        let mut expression = Expression::parse_hbond(energy).map_err(|e| e.to_string())?;
        for index in 0..self.num_tabulated_functions() {
            expression
                .add_tabulated_function(
                    self.tabulated_function_name(index),
                    self.tabulated_function(index).clone(),
                )
                .map_err(|e| e.to_string())?;
        }
        self.energy_expr = expression;
        self.energy_source = energy.to_owned();
        Ok(())
    }
    pub fn nonbonded_method(&self) -> CustomHbondMethod {
        self.method
    }
    pub fn set_nonbonded_method(&mut self, method: CustomHbondMethod) {
        self.method = method;
    }
    pub fn cutoff_distance(&self) -> f64 {
        self.cutoff
    }
    pub fn set_cutoff_distance(&mut self, cutoff: f64) {
        self.cutoff = cutoff;
    }
    pub fn uses_periodic_boundary_conditions(&self) -> bool {
        self.method == CustomHbondMethod::CutoffPeriodic
    }
    pub fn set_use_neighbor_cells(&mut self, enabled: bool) {
        self.use_neighbor_cells = enabled;
    }
    pub fn add_tabulated_function(
        &mut self,
        name: &str,
        function: impl Into<crate::TabulatedFunction>,
    ) -> Result<usize, crate::expression::EvalError> {
        self.energy_expr.add_tabulated_function(name, function)
    }
    pub fn num_tabulated_functions(&self) -> usize {
        self.energy_expr.num_tabulated_functions()
    }
    pub fn tabulated_function_name(&self, index: usize) -> &str {
        self.energy_expr.tabulated_function_name(index)
    }
    pub fn tabulated_function(&self, index: usize) -> &crate::TabulatedFunction {
        self.energy_expr.tabulated_function(index)
    }
    pub fn set_tabulated_function(
        &mut self,
        index: usize,
        function: impl Into<crate::TabulatedFunction>,
    ) -> Result<(), crate::expression::EvalError> {
        self.energy_expr.set_tabulated_function(index, function)
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

    pub fn add_per_donor_parameter(&mut self, name: &str) -> usize {
        self.donor_params.push(name.to_owned());
        self.donor_params.len() - 1
    }
    pub fn num_per_donor_parameters(&self) -> usize {
        self.donor_params.len()
    }
    pub fn per_donor_parameter_name(&self, index: usize) -> &str {
        &self.donor_params[index]
    }
    pub fn add_donor(
        &mut self,
        first: usize,
        second: Option<usize>,
        third: Option<usize>,
        params: Vec<f64>,
    ) -> usize {
        assert_eq!(
            params.len(),
            self.donor_params.len(),
            "incorrect donor parameter count"
        );
        self.donors.push(Group {
            atoms: [Some(first), second, third],
            params,
        });
        self.donors.len() - 1
    }
    pub fn num_donors(&self) -> usize {
        self.donors.len()
    }
    pub fn donor_parameters(&self, index: usize) -> ([Option<usize>; 3], &[f64]) {
        (self.donors[index].atoms, &self.donors[index].params)
    }
    pub fn set_donor_parameters(
        &mut self,
        index: usize,
        first: usize,
        second: Option<usize>,
        third: Option<usize>,
        params: Vec<f64>,
    ) {
        assert_eq!(
            params.len(),
            self.donor_params.len(),
            "incorrect donor parameter count"
        );
        self.donors[index] = Group {
            atoms: [Some(first), second, third],
            params,
        };
    }

    pub fn add_per_acceptor_parameter(&mut self, name: &str) -> usize {
        self.acceptor_params.push(name.to_owned());
        self.acceptor_params.len() - 1
    }
    pub fn num_per_acceptor_parameters(&self) -> usize {
        self.acceptor_params.len()
    }
    pub fn per_acceptor_parameter_name(&self, index: usize) -> &str {
        &self.acceptor_params[index]
    }
    pub fn add_acceptor(
        &mut self,
        first: usize,
        second: Option<usize>,
        third: Option<usize>,
        params: Vec<f64>,
    ) -> usize {
        assert_eq!(
            params.len(),
            self.acceptor_params.len(),
            "incorrect acceptor parameter count"
        );
        self.acceptors.push(Group {
            atoms: [Some(first), second, third],
            params,
        });
        self.acceptors.len() - 1
    }
    pub fn num_acceptors(&self) -> usize {
        self.acceptors.len()
    }
    pub fn acceptor_parameters(&self, index: usize) -> ([Option<usize>; 3], &[f64]) {
        (self.acceptors[index].atoms, &self.acceptors[index].params)
    }
    pub fn set_acceptor_parameters(
        &mut self,
        index: usize,
        first: usize,
        second: Option<usize>,
        third: Option<usize>,
        params: Vec<f64>,
    ) {
        assert_eq!(
            params.len(),
            self.acceptor_params.len(),
            "incorrect acceptor parameter count"
        );
        self.acceptors[index] = Group {
            atoms: [Some(first), second, third],
            params,
        };
    }

    pub fn add_exclusion(&mut self, donor: usize, acceptor: usize) -> usize {
        self.exclusions.push((donor, acceptor));
        self.exclusion_set.insert((donor, acceptor));
        self.exclusions.len() - 1
    }
    pub fn num_exclusions(&self) -> usize {
        self.exclusions.len()
    }
    pub fn exclusion_particles(&self, index: usize) -> (usize, usize) {
        self.exclusions[index]
    }
    pub fn set_exclusion_particles(&mut self, index: usize, donor: usize, acceptor: usize) {
        self.exclusions[index] = (donor, acceptor);
        self.exclusion_set = self.exclusions.iter().copied().collect();
    }
    fn check_box(&self, vectors: Option<&[[f64; 3]; 3]>) {
        if self.method != CustomHbondMethod::NoCutoff {
            assert!(self.cutoff > 0.0, "cutoff distance must be positive");
        }
        if self.uses_periodic_boundary_conditions() {
            let vectors = vectors.expect("periodic force requires box vectors");
            assert!(
                (0..3).all(|axis| vectors[axis][axis] >= 2.0 * self.cutoff),
                "cutoff exceeds half the periodic box width"
            );
        }
    }
    pub fn update_parameters_from(&mut self, source: &Self) {
        assert_eq!(
            self.donor_params, source.donor_params,
            "donor parameter definitions must match"
        );
        assert_eq!(
            self.acceptor_params, source.acceptor_params,
            "acceptor parameter definitions must match"
        );
        assert_eq!(
            self.donors.len(),
            source.donors.len(),
            "donor count must match"
        );
        assert_eq!(
            self.acceptors.len(),
            source.acceptors.len(),
            "acceptor count must match"
        );
        for (group, other) in self
            .donors
            .iter()
            .chain(&self.acceptors)
            .zip(source.donors.iter().chain(&source.acceptors))
        {
            assert_eq!(
                group.atoms, other.atoms,
                "hydrogen bond group particles cannot change in a Context"
            );
        }
        self.energy_expr
            .update_tabulated_functions_from(&source.energy_expr);
        for (group, other) in self
            .donors
            .iter_mut()
            .chain(&mut self.acceptors)
            .zip(source.donors.iter().chain(&source.acceptors))
        {
            group.params.clone_from(&other.params);
        }
    }
    fn evaluate<B: Backend>(
        &self,
        positions: &Tensor<B, 2>,
        vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64) {
        self.check_box(vectors);
        let n = positions.dims()[0];
        let pos = positions
            .to_data()
            .convert::<f64>()
            .to_vec::<f64>()
            .unwrap();
        let vectors = vectors.filter(|_| self.uses_periodic_boundary_conditions());
        let mut evaluator = CustomEvaluator::new(
            &self.energy_expr,
            &self.global_params,
            self.donor_params
                .iter()
                .chain(&self.acceptor_params)
                .map(String::as_str),
            &COORDINATES,
            &[],
        );
        evaluator.set_periodic_box(vectors);
        let mut forces = vec![0.0; 3 * n];
        let mut energy = 0.0;
        let cutoff =
            (self.method != CustomHbondMethod::NoCutoff).then_some(self.cutoff * self.cutoff);
        let mut pair = |donor_index: usize, acceptor_index: usize| {
            if self.exclusion_set.contains(&(donor_index, acceptor_index)) {
                return;
            }
            let donor = &self.donors[donor_index];
            let acceptor = &self.acceptors[acceptor_index];
            if let Some(cutoff) = cutoff {
                let d = donor.atoms[0].unwrap();
                let a = acceptor.atoms[0].unwrap();
                let mut delta = std::array::from_fn(|axis| pos[3 * d + axis] - pos[3 * a + axis]);
                if let Some(vectors) = vectors {
                    apply_minimum_image_ortho(&mut delta, vectors);
                }
                if delta.iter().map(|d| d * d).sum::<f64>() >= cutoff {
                    return;
                }
            }
            let atoms = [
                acceptor.atoms[0],
                acceptor.atoms[1],
                acceptor.atoms[2],
                donor.atoms[0],
                donor.atoms[1],
                donor.atoms[2],
            ];
            let mut coordinates = [0.0; 18];
            for (index, particle) in atoms.iter().enumerate() {
                if let Some(particle) = particle {
                    coordinates[3 * index..3 * index + 3]
                        .copy_from_slice(&pos[3 * particle..3 * particle + 3]);
                }
            }
            evaluator.set_coordinates(&coordinates);
            evaluator.set_parameters(donor.params.iter().chain(&acceptor.params).copied());
            energy += evaluator.evaluate();
            for (index, particle) in atoms.iter().enumerate() {
                if let Some(particle) = particle {
                    for axis in 0..3 {
                        forces[3 * particle + axis] -=
                            evaluator.coordinate_derivative(3 * index + axis);
                    }
                }
            }
        };
        if cutoff.is_some() && self.use_neighbor_cells {
            let primary: Vec<f64> = self
                .donors
                .iter()
                .chain(&self.acceptors)
                .flat_map(|group| {
                    let index = group.atoms[0].unwrap();
                    pos[3 * index..3 * index + 3].iter().copied()
                })
                .collect();
            self.neighbor_cells
                .visit(&primary, self.cutoff, vectors, |i, j| {
                    if i < self.donors.len() && j >= self.donors.len() {
                        pair(i, j - self.donors.len());
                    }
                });
        } else {
            for donor in 0..self.donors.len() {
                for acceptor in 0..self.acceptors.len() {
                    pair(donor, acceptor);
                }
            }
        }
        (
            crate::tensor::from_floats::<B, 1>(forces.as_slice(), &positions.device())
                .reshape([n, 3]),
            energy,
        )
    }
}

impl<B: Backend> Force<B> for CustomHbondForce {
    super::force_metadata_impl!();
    fn initialize(&mut self, num_particles: usize, vectors: Option<&[[f64; 3]; 3]>) {
        self.check_box(vectors);
        let _ = CustomEvaluator::new(
            &self.energy_expr,
            &self.global_params,
            self.donor_params
                .iter()
                .chain(&self.acceptor_params)
                .map(String::as_str),
            &COORDINATES,
            &[],
        );
        for (groups, offset) in [(&self.acceptors, 0), (&self.donors, 3)] {
            for group in groups {
                for (index, particle) in group.atoms.iter().enumerate() {
                    if let Some(particle) = particle {
                        assert!(
                            *particle < num_particles,
                            "hydrogen bond particle index out of range"
                        );
                    } else {
                        assert!(
                            (0..3).all(|axis| self
                                .energy_expr
                                .variable_index(COORDINATES[3 * (index + offset) + axis])
                                .is_none()),
                            "energy uses an omitted hydrogen bond particle"
                        );
                    }
                }
            }
        }
        assert_eq!(
            self.exclusions.len(),
            self.exclusion_set.len(),
            "duplicate hydrogen bond exclusion"
        );
        for &(donor, acceptor) in &self.exclusions {
            assert!(
                donor < self.donors.len() && acceptor < self.acceptors.len(),
                "hydrogen bond exclusion index out of range"
            );
        }
    }
    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.uses_periodic_boundary_conditions()
    }
    fn global_parameters(&self) -> HashMap<String, f64> {
        self.global_params.clone()
    }
    fn set_global_parameter(&mut self, name: &str, value: f64) -> bool {
        if let Some(parameter) = self.global_params.get_mut(name) {
            *parameter = value;
            true
        } else {
            false
        }
    }
    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64) {
        self.evaluate(positions, vectors)
    }
}
