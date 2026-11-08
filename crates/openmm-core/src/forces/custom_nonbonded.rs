// CustomNonbondedForce: user-defined pairwise energy as a function of r
// and per-particle parameters.
// Reference test: openmm/tests/TestCustomNonbondedForce.h
use super::custom_evaluator::CustomEvaluator;
use super::neighbors::NeighborCells;
use super::Force;
use crate::expression::Expression;
use crate::forces::harmonic_bond::apply_minimum_image_ortho;
use burn::prelude::*;
use std::collections::{BTreeSet, HashMap, HashSet};
mod long_range;
use long_range::CorrectionCache;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CustomNonbondedMethod {
    #[default]
    NoCutoff,
    CutoffNonPeriodic,
    CutoffPeriodic,
}

#[derive(Clone)]
struct ComputedValue {
    name: String,
    source: String,
    program: Expression,
    paired_names: [String; 2],
}

#[derive(Clone)]
pub struct CustomNonbondedForce {
    metadata: super::ForceMetadata,
    energy_expr: Expression,
    energy_source: String,
    computed_values: Vec<ComputedValue>,
    per_particle_params: Vec<String>,
    paired_parameter_names: Vec<[String; 2]>,
    particles: Vec<Vec<f64>>,
    exclusions: Vec<(usize, usize)>,
    exclusion_set: HashSet<(usize, usize)>,
    interaction_groups: Vec<(BTreeSet<usize>, BTreeSet<usize>)>,
    global_param_names: Vec<String>,
    global_params: HashMap<String, f64>,
    energy_param_deriv_names: Vec<String>,
    method: CustomNonbondedMethod,
    cutoff_distance: f64,
    switching_distance: f64,
    use_switching_function: bool,
    use_long_range_correction: bool,
    long_range_cache: CorrectionCache,
    use_neighbor_cells: bool,
    neighbor_cells: NeighborCells,
}

impl CustomNonbondedForce {
    super::force_metadata!();
    pub fn new(energy: &str) -> Result<Self, crate::expression::ParseError> {
        Ok(Self {
            metadata: super::ForceMetadata::new("CustomNonbondedForce"),
            energy_expr: Expression::parse_unbound(energy)?,
            energy_source: energy.to_owned(),
            computed_values: Vec::new(),
            per_particle_params: Vec::new(),
            paired_parameter_names: Vec::new(),
            particles: Vec::new(),
            exclusions: Vec::new(),
            exclusion_set: HashSet::new(),
            interaction_groups: Vec::new(),
            global_param_names: Vec::new(),
            global_params: HashMap::new(),
            energy_param_deriv_names: Vec::new(),
            method: CustomNonbondedMethod::NoCutoff,
            cutoff_distance: 1.0,
            switching_distance: -1.0,
            use_switching_function: false,
            use_long_range_correction: false,
            long_range_cache: CorrectionCache::default(),
            use_neighbor_cells: true,
            neighbor_cells: NeighborCells::default(),
        })
    }

    pub fn add_tabulated_function(
        &mut self,
        name: &str,
        function: impl Into<crate::TabulatedFunction>,
    ) -> Result<usize, crate::expression::EvalError> {
        let function = function.into();
        let mut expression = self.energy_expr.clone();
        let mut computed_values = self.computed_values.clone();
        let index = expression.add_tabulated_function(name, function.clone())?;
        for value in &mut computed_values {
            value
                .program
                .add_tabulated_function(name, function.clone())?;
        }
        self.energy_expr = expression;
        self.computed_values = computed_values;
        self.long_range_cache.clear();
        Ok(index)
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
        let function = function.into();
        self.energy_expr
            .set_tabulated_function(index, function.clone())?;
        for value in &mut self.computed_values {
            value
                .program
                .set_tabulated_function(index, function.clone())?;
        }
        self.long_range_cache.clear();
        Ok(())
    }

    fn bind_tables(&self, program: &mut Expression) -> Result<(), crate::expression::EvalError> {
        for index in 0..self.num_tabulated_functions() {
            program.add_tabulated_function(
                self.tabulated_function_name(index),
                self.tabulated_function(index).clone(),
            )?;
        }
        Ok(())
    }
    pub fn energy_function(&self) -> &str {
        &self.energy_source
    }
    pub fn set_energy_function(&mut self, source: &str) -> Result<(), String> {
        let mut program = Expression::parse_unbound(source).map_err(|error| error.to_string())?;
        self.bind_tables(&mut program)
            .map_err(|error| error.to_string())?;
        self.energy_expr = program;
        self.energy_source = source.to_owned();
        self.long_range_cache.clear();
        Ok(())
    }
    fn computed_value(&self, name: &str, source: &str) -> Result<ComputedValue, String> {
        let mut program = Expression::parse_unbound(source).map_err(|error| error.to_string())?;
        self.bind_tables(&mut program)
            .map_err(|error| error.to_string())?;
        Ok(ComputedValue {
            name: name.to_owned(),
            source: source.to_owned(),
            program,
            paired_names: [format!("{name}1"), format!("{name}2")],
        })
    }
    pub fn add_computed_value(&mut self, name: &str, expression: &str) -> Result<usize, String> {
        self.computed_values
            .push(self.computed_value(name, expression)?);
        self.long_range_cache.clear();
        Ok(self.computed_values.len() - 1)
    }
    pub fn num_computed_values(&self) -> usize {
        self.computed_values.len()
    }
    pub fn computed_value_parameters(&self, index: usize) -> (&str, &str) {
        (
            &self.computed_values[index].name,
            &self.computed_values[index].source,
        )
    }
    pub fn set_computed_value_parameters(
        &mut self,
        index: usize,
        name: &str,
        expression: &str,
    ) -> Result<(), String> {
        self.computed_values[index] = self.computed_value(name, expression)?;
        self.long_range_cache.clear();
        Ok(())
    }
    pub fn add_interaction_group(
        &mut self,
        set1: impl IntoIterator<Item = usize>,
        set2: impl IntoIterator<Item = usize>,
    ) -> usize {
        self.interaction_groups
            .push((set1.into_iter().collect(), set2.into_iter().collect()));
        self.long_range_cache.clear();
        self.interaction_groups.len() - 1
    }
    pub fn num_interaction_groups(&self) -> usize {
        self.interaction_groups.len()
    }
    pub fn interaction_group_parameters(
        &self,
        index: usize,
    ) -> (&BTreeSet<usize>, &BTreeSet<usize>) {
        (
            &self.interaction_groups[index].0,
            &self.interaction_groups[index].1,
        )
    }
    pub fn set_interaction_group_parameters(
        &mut self,
        index: usize,
        set1: impl IntoIterator<Item = usize>,
        set2: impl IntoIterator<Item = usize>,
    ) {
        self.interaction_groups[index] = (set1.into_iter().collect(), set2.into_iter().collect());
        self.long_range_cache.clear();
    }
    pub fn nonbonded_method(&self) -> CustomNonbondedMethod {
        self.method
    }
    pub fn set_nonbonded_method(&mut self, method: CustomNonbondedMethod) {
        self.method = method;
        self.long_range_cache.clear();
    }
    pub fn use_switching_function(&self) -> bool {
        self.use_switching_function
    }
    pub fn set_use_switching_function(&mut self, enabled: bool) {
        self.use_switching_function = enabled;
        self.long_range_cache.clear();
    }
    pub fn switching_distance(&self) -> f64 {
        self.switching_distance
    }
    pub fn set_switching_distance(&mut self, distance: f64) {
        self.switching_distance = distance;
        self.long_range_cache.clear();
    }
    pub fn use_long_range_correction(&self) -> bool {
        self.use_long_range_correction
    }
    pub fn set_use_long_range_correction(&mut self, enabled: bool) {
        self.use_long_range_correction = enabled;
        self.long_range_cache.clear();
    }
    fn active_cutoff(&self) -> Option<f64> {
        (self.method != CustomNonbondedMethod::NoCutoff).then_some(self.cutoff_distance)
    }

    pub fn add_per_particle_parameter(&mut self, name: &str) -> usize {
        self.long_range_cache.clear();
        self.per_particle_params.push(name.to_string());
        self.paired_parameter_names
            .push([format!("{name}1"), format!("{name}2")]);
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
        self.long_range_cache.clear();
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
        self.long_range_cache.clear();
    }

    pub fn add_energy_parameter_derivative(&mut self, name: &str) {
        self.long_range_cache.clear();
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

    /// Add a particle. Returns particle index.
    pub fn add_particle(&mut self, params: Vec<f64>) -> usize {
        assert_eq!(params.len(), self.per_particle_params.len());
        self.particles.push(params);
        self.long_range_cache.clear();
        self.particles.len() - 1
    }

    pub fn num_particles(&self) -> usize {
        self.particles.len()
    }

    pub fn particle_parameters(&self, index: usize) -> &[f64] {
        &self.particles[index]
    }

    pub fn set_particle_parameters(&mut self, index: usize, params: Vec<f64>) {
        assert_eq!(params.len(), self.per_particle_params.len());
        self.particles[index] = params;
        self.long_range_cache.clear();
    }

    pub fn add_exclusion(&mut self, p1: usize, p2: usize) {
        let key = if p1 < p2 { (p1, p2) } else { (p2, p1) };
        self.exclusions.push(key);
        self.exclusion_set.insert(key);
    }

    pub fn create_exclusions_from_bonds(&mut self, bonds: &[(usize, usize)], bond_cutoff: usize) {
        let n = self.particles.len();
        let mut neighbors = vec![Vec::new(); n];
        for &(a, b) in bonds {
            assert!(a < n && b < n, "bond particle index out of range");
            neighbors[a].push(b);
            neighbors[b].push(a);
        }
        for root in 0..n {
            let mut visited = BTreeSet::from([root]);
            let mut frontier = vec![root];
            for _ in 0..bond_cutoff {
                let mut next = Vec::new();
                for particle in frontier {
                    for &neighbor in &neighbors[particle] {
                        if visited.insert(neighbor) {
                            next.push(neighbor);
                        }
                    }
                }
                frontier = next;
                if frontier.is_empty() {
                    break;
                }
            }
            for neighbor in visited.into_iter().filter(|&i| i < root) {
                self.add_exclusion(neighbor, root);
            }
        }
    }

    pub fn num_exclusions(&self) -> usize {
        self.exclusions.len()
    }

    pub fn exclusion_particles(&self, index: usize) -> (usize, usize) {
        self.exclusions[index]
    }

    pub fn set_exclusion_particles(&mut self, index: usize, p1: usize, p2: usize) {
        self.exclusions[index] = if p1 < p2 { (p1, p2) } else { (p2, p1) };
        self.exclusion_set = self.exclusions.iter().copied().collect();
    }

    pub fn set_uses_periodic_boundary_conditions(&mut self, use_pbc: bool) {
        if use_pbc {
            self.method = CustomNonbondedMethod::CutoffPeriodic;
        } else if self.method == CustomNonbondedMethod::CutoffPeriodic {
            self.method = CustomNonbondedMethod::CutoffNonPeriodic;
        }
        self.long_range_cache.clear();
    }

    pub fn uses_periodic_boundary_conditions(&self) -> bool {
        self.method == CustomNonbondedMethod::CutoffPeriodic
    }

    pub fn set_use_neighbor_cells(&mut self, enabled: bool) {
        self.use_neighbor_cells = enabled;
    }

    pub fn set_cutoff_distance(&mut self, cutoff: f64) {
        assert!(
            cutoff.is_finite() && cutoff > 0.0,
            "cutoff distance must be positive and finite"
        );
        self.cutoff_distance = cutoff;
        self.long_range_cache.clear();
    }

    pub fn cutoff_distance(&self) -> f64 {
        self.cutoff_distance
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

        self.energy_expr
            .update_tabulated_functions_from(&source.energy_expr);
        for value in &mut self.computed_values {
            value
                .program
                .update_tabulated_functions_from(&source.energy_expr);
        }
        self.particles = source.particles.clone();
        self.long_range_cache.clear();
    }
}

impl CustomNonbondedForce {
    fn parameter_names(&self) -> impl Iterator<Item = &str> {
        self.paired_parameter_names
            .iter()
            .flatten()
            .map(String::as_str)
            .chain(
                self.computed_values
                    .iter()
                    .flat_map(|value| value.paired_names.iter().map(String::as_str)),
            )
    }
    fn compute_values(&self, particles: &[Vec<f64>]) -> Vec<f64> {
        let count = self.computed_values.len();
        let mut result = vec![0.0; particles.len() * count];
        for (index, value) in self.computed_values.iter().enumerate() {
            let mut evaluator = CustomEvaluator::new(
                &value.program,
                &self.global_params,
                self.per_particle_params.iter().map(String::as_str),
                &["r"],
                &[],
            );
            for (particle, parameters) in particles.iter().enumerate() {
                evaluator.set_parameters(parameters.iter().copied());
                result[particle * count + index] = evaluator.evaluate();
            }
        }
        result
    }
    fn check_periodic_box(&self, vectors: Option<&[[f64; 3]; 3]>) {
        if self.uses_periodic_boundary_conditions() {
            let vectors = vectors.expect("periodic nonbonded force requires box vectors");
            for (axis, vector) in vectors.iter().enumerate() {
                assert!(
                    vector[axis] >= 2.0 * self.cutoff_distance,
                    "periodic box width must be at least twice the cutoff"
                );
            }
        }
    }
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
            self.parameter_names(),
            &["r"],
            &self.energy_param_deriv_names,
        );

        self.check_periodic_box(box_vectors);
        let computed = self.compute_values(&self.particles);
        let count = self.computed_values.len();
        let periodic = self.uses_periodic_boundary_conditions();
        let cutoff2 = self.active_cutoff().map(|c| c * c);
        let mut pair = |i: usize, j: usize| {
            let key = (i.min(j), i.max(j));
            if self.exclusion_set.contains(&key) {
                return;
            }

            let mut delta = [
                pos[j * 3] - pos[i * 3],
                pos[j * 3 + 1] - pos[i * 3 + 1],
                pos[j * 3 + 2] - pos[i * 3 + 2],
            ];
            if periodic {
                if let Some(bv) = box_vectors {
                    apply_minimum_image_ortho(&mut delta, bv);
                }
            }

            let r2 = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
            if cutoff2.map(|c2| r2 >= c2).unwrap_or(false) {
                return;
            }
            let r = r2.sqrt();

            evaluator.set_coordinates(&[r]);
            evaluator.set_parameter_pairs(0, &self.particles[i], &self.particles[j]);
            if count > 0 {
                evaluator.set_parameter_pairs(
                    self.per_particle_params.len(),
                    &computed[i * count..(i + 1) * count],
                    &computed[j * count..(j + 1) * count],
                );
            }
            let de_dr = if self.use_switching_function
                && cutoff2.is_some()
                && r > self.switching_distance
            {
                let width = self.cutoff_distance - self.switching_distance;
                let t = (r - self.switching_distance) / width;
                let switch = 1.0 + t * t * t * (-10.0 + t * (15.0 - 6.0 * t));
                let derivative = t * t * (-30.0 + t * (60.0 - 30.0 * t)) / width;
                let value = evaluator.evaluate_scaled(switch);
                energy += switch * value;
                switch * evaluator.coordinate_derivative(0) + derivative * value
            } else {
                energy += evaluator.evaluate();
                evaluator.coordinate_derivative(0)
            };

            let f_over_r = de_dr / r;
            for d in 0..3 {
                let f = f_over_r * delta[d];
                forces[i * 3 + d] += f;
                forces[j * 3 + d] -= f;
            }
        };
        if !self.interaction_groups.is_empty() {
            for (set1, set2) in &self.interaction_groups {
                for &i in set1 {
                    for &j in set2 {
                        if i == j || (i > j && set1.contains(&j) && set2.contains(&i)) {
                            continue;
                        }
                        pair(i, j);
                    }
                }
            }
        } else if let Some(cutoff) = self.active_cutoff().filter(|_| self.use_neighbor_cells) {
            self.neighbor_cells.visit(
                &pos,
                cutoff,
                box_vectors.filter(|_| self.uses_periodic_boundary_conditions()),
                pair,
            );
        } else {
            for i in 0..n {
                for j in i + 1..n {
                    pair(i, j);
                }
            }
        }

        let mut derivatives = evaluator.finish();
        if self.use_long_range_correction && self.uses_periodic_boundary_conditions() {
            let vectors = box_vectors.unwrap();
            let volume = vectors[0][0] * vectors[1][1] * vectors[2][2];
            self.long_range_cache
                .add(self, volume, &mut energy, &mut derivatives);
        }
        (
            crate::tensor::from_floats::<B, 1>(forces.as_slice(), &device).reshape([n, 3]),
            energy,
            derivatives,
        )
    }
}

impl<B: Backend> Force<B> for CustomNonbondedForce {
    fn initialize(&mut self, num_particles: usize, box_vectors: Option<&[[f64; 3]; 3]>) {
        self.check_periodic_box(box_vectors);
        if self.use_switching_function {
            assert!(
                self.switching_distance >= 0.0 && self.switching_distance < self.cutoff_distance,
                "switching distance must satisfy 0 <= switch < cutoff"
            );
        }
        assert!(
            self.computed_values.is_empty() || self.energy_param_deriv_names.is_empty(),
            "parameter derivatives cannot be combined with computed values"
        );
        for value in &self.computed_values {
            value
                .program
                .validate_variables(
                    self.per_particle_params
                        .iter()
                        .map(String::as_str)
                        .chain(self.global_params.keys().map(String::as_str))
                        .chain(["r"]),
                )
                .unwrap();
        }
        assert_eq!(
            self.exclusion_set.len(),
            self.exclusions.len(),
            "duplicate nonbonded exclusions"
        );
        for &(i, j) in &self.exclusions {
            assert!(
                i < num_particles && j < num_particles,
                "exclusion particle index out of range"
            );
        }
        for (set1, set2) in &self.interaction_groups {
            assert!(
                set1.iter().chain(set2.iter()).all(|&i| i < num_particles),
                "interaction group particle index out of range"
            );
        }
        let _ = CustomEvaluator::new(
            &self.energy_expr,
            &self.global_params,
            self.parameter_names(),
            &["r"],
            &self.energy_param_deriv_names,
        );
        assert_eq!(
            self.particles.len(),
            num_particles,
            "custom nonbonded particle count must match system"
        );
    }
    super::force_metadata_impl!();
    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        box_vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64) {
        let (forces, energy, _) = self.evaluate::<B>(positions, box_vectors);
        (forces, energy)
    }

    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.method == CustomNonbondedMethod::CutoffPeriodic
    }

    fn global_parameters(&self) -> HashMap<String, f64> {
        self.global_params.clone()
    }

    fn set_global_parameter(&mut self, name: &str, value: f64) -> bool {
        match self.global_params.get_mut(name) {
            Some(parameter) => {
                *parameter = value;
                self.long_range_cache.clear();
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
