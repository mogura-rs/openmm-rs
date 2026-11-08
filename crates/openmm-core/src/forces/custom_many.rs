use super::{
    harmonic_bond::apply_minimum_image_ortho, neighbors::NeighborCells,
    particle_parameters::ParticleParameters, Force, HostForceEvaluation,
};
use crate::expression::{DerivativeProgram, Expression};
use burn::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CustomManyParticleMethod {
    #[default]
    NoCutoff,
    CutoffNonPeriodic,
    CutoffPeriodic,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PermutationMode {
    #[default]
    SinglePermutation,
    UniqueCentralParticle,
}
#[derive(Clone, Copy)]
enum Input {
    Coordinate(usize, usize),
    Parameter(usize, usize),
    Global(usize),
}
struct Evaluator {
    program: DerivativeProgram,
    inputs: Vec<(usize, Input)>,
    globals: Vec<(usize, usize)>,
    gradients: Vec<(usize, usize, usize)>,
}
impl Evaluator {
    fn new(force: &CustomManyParticleForce) -> Self {
        let mut names = HashMap::new();
        for (i, (name, _)) in force.globals.iter().enumerate() {
            names.insert(name.clone(), Input::Global(i));
        }
        for particle in 0..force.count {
            for (axis, name) in ['x', 'y', 'z'].into_iter().enumerate() {
                names.insert(
                    format!("{name}{}", particle + 1),
                    Input::Coordinate(particle, axis),
                );
            }
            for (index, name) in force.parameters.iter().enumerate() {
                names.insert(
                    format!("{name}{}", particle + 1),
                    Input::Parameter(particle, index),
                );
            }
        }
        force
            .expression
            .validate_variables(names.keys().map(String::as_str))
            .unwrap();
        let inputs: Vec<_> = force
            .expression
            .variables
            .iter()
            .map(|name| names[name])
            .collect();
        let gradients: Vec<_> = inputs
            .iter()
            .enumerate()
            .filter_map(|(index, input)| {
                if let Input::Coordinate(particle, axis) = input {
                    Some((index, *particle, *axis))
                } else {
                    None
                }
            })
            .collect();
        let varying: Vec<_> = inputs
            .iter()
            .map(|v| !matches!(v, Input::Global(_)))
            .collect();
        let program = DerivativeProgram::new(
            force.expression.clone(),
            &varying,
            gradients.iter().map(|v| v.0),
        );
        let globals = inputs
            .iter()
            .enumerate()
            .filter_map(|(index, input)| {
                if let Input::Global(i) = input {
                    Some((index, *i))
                } else {
                    None
                }
            })
            .collect();
        let inputs = inputs
            .into_iter()
            .enumerate()
            .filter(|(_, input)| !matches!(input, Input::Global(_)))
            .collect();
        Self {
            program,
            inputs,
            globals,
            gradients,
        }
    }
    fn prepare(&mut self, force: &CustomManyParticleForce, vectors: Option<&[[f64; 3]; 3]>) {
        self.program.set_periodic_box(vectors);
        for &(index, global) in &self.globals {
            self.program.set(index, force.globals[global].1);
        }
        self.program.prepare();
    }
    fn evaluate(
        &mut self,
        force: &CustomManyParticleForce,
        particles: &[usize],
        positions: &[f64],
        forces: &mut [f64],
    ) -> f64 {
        for &(index, input) in &self.inputs {
            let value = match input {
                Input::Coordinate(p, a) => positions[3 * particles[p] + a],
                Input::Parameter(p, i) => force.particles[particles[p]][i],
                Input::Global(_) => unreachable!(),
            };
            self.program.set(index, value);
        }
        let energy = self.program.evaluate();
        for &(index, p, a) in &self.gradients {
            forces[3 * particles[p] + a] -= self.program.derivative(index);
        }
        energy
    }
}
#[derive(Default)]
struct Graph {
    edges: Vec<[usize; 2]>,
    offsets: Vec<usize>,
    cursors: Vec<usize>,
    adjacent: Vec<usize>,
}
impl Graph {
    fn neighbors(&self, particle: usize) -> &[usize] {
        &self.adjacent[self.offsets[particle]..self.offsets[particle + 1]]
    }
    fn build(
        &mut self,
        force: &CustomManyParticleForce,
        positions: &[f64],
        vectors: Option<&[[f64; 3]; 3]>,
    ) {
        let n = force.particles.len();
        self.edges.clear();
        self.offsets.resize(n + 1, 0);
        self.offsets.fill(0);
        force.cells.visit(positions, force.cutoff, vectors, |i, j| {
            if !force.excluded(i, j) && force.within_cutoff(positions, i, j, vectors) {
                self.edges.push([i, j]);
                self.offsets[i + 1] += 1;
                self.offsets[j + 1] += 1;
            }
        });
        for i in 0..n {
            self.offsets[i + 1] += self.offsets[i];
        }
        self.cursors.clone_from(&self.offsets);
        self.adjacent.resize(2 * self.edges.len(), 0);
        for &[i, j] in &self.edges {
            self.adjacent[self.cursors[i]] = j;
            self.cursors[i] += 1;
            self.adjacent[self.cursors[j]] = i;
            self.cursors[j] += 1;
        }
        for i in 0..n {
            self.adjacent[self.offsets[i]..self.offsets[i + 1]].sort_unstable();
        }
    }
}
#[derive(Default)]
struct Work {
    evaluator: Option<Evaluator>,
    graph: Graph,
    selected: Vec<usize>,
    permuted: Vec<usize>,
    candidates: Vec<usize>,
    forces: Vec<f64>,
}
#[derive(Default)]
struct Cache(Mutex<Work>);
impl Clone for Cache {
    fn clone(&self) -> Self {
        Self::default()
    }
}

#[derive(Clone)]
pub struct CustomManyParticleForce {
    metadata: super::ForceMetadata,
    count: usize,
    source: String,
    expression: Expression,
    parameters: Vec<String>,
    globals: Vec<(String, f64)>,
    particles: ParticleParameters,
    types: Vec<i32>,
    exclusions: Vec<(usize, usize)>,
    exclusion_set: HashSet<(usize, usize)>,
    filters: Vec<Vec<i32>>,
    method: CustomManyParticleMethod,
    mode: PermutationMode,
    cutoff: f64,
    kernel_box: Option<(bool, f64)>,
    cells: NeighborCells,
    cache: Cache,
}
impl CustomManyParticleForce {
    super::force_metadata!();
    pub fn new(count: usize, energy: &str) -> Result<Self, String> {
        if count == 0 {
            return Err("CustomManyParticleForce requires at least one particle per set".into());
        }
        Ok(Self {
            metadata: super::ForceMetadata::new("CustomManyParticleForce"),
            count,
            source: energy.to_owned(),
            expression: Expression::parse_geometric(energy, Some(('p', count)))
                .map_err(|e| e.to_string())?,
            parameters: Vec::new(),
            globals: Vec::new(),
            particles: ParticleParameters::default(),
            types: Vec::new(),
            exclusions: Vec::new(),
            exclusion_set: HashSet::new(),
            filters: vec![Vec::new(); count],
            method: CustomManyParticleMethod::NoCutoff,
            mode: PermutationMode::SinglePermutation,
            cutoff: 1.,
            kernel_box: None,
            cells: NeighborCells::default(),
            cache: Cache::default(),
        })
    }
    fn invalidate(&mut self) {
        self.cache = Cache::default();
    }
    pub fn num_particles_per_set(&self) -> usize {
        self.count
    }
    pub fn energy_function(&self) -> &str {
        &self.source
    }
    pub fn set_energy_function(&mut self, source: &str) -> Result<(), String> {
        let mut expression = Expression::parse_geometric(source, Some(('p', self.count)))
            .map_err(|e| e.to_string())?;
        for i in 0..self.num_tabulated_functions() {
            expression
                .add_tabulated_function(
                    self.tabulated_function_name(i),
                    self.tabulated_function(i).clone(),
                )
                .map_err(|e| e.to_string())?;
        }
        self.expression = expression;
        self.source = source.to_owned();
        self.invalidate();
        Ok(())
    }
    pub fn add_per_particle_parameter(&mut self, name: &str) -> usize {
        self.parameters.push(name.to_owned());
        self.invalidate();
        self.parameters.len() - 1
    }
    pub fn num_per_particle_parameters(&self) -> usize {
        self.parameters.len()
    }
    pub fn per_particle_parameter_name(&self, i: usize) -> &str {
        &self.parameters[i]
    }
    pub fn set_per_particle_parameter_name(&mut self, i: usize, name: &str) {
        self.parameters[i] = name.to_owned();
        self.invalidate();
    }
    pub fn add_global_parameter(&mut self, name: &str, value: f64) -> usize {
        self.globals.push((name.to_owned(), value));
        self.invalidate();
        self.globals.len() - 1
    }
    pub fn num_global_parameters(&self) -> usize {
        self.globals.len()
    }
    pub fn global_parameter_name(&self, i: usize) -> &str {
        &self.globals[i].0
    }
    pub fn set_global_parameter_name(&mut self, i: usize, name: &str) {
        self.globals[i].0 = name.to_owned();
        self.invalidate();
    }
    pub fn global_parameter_default_value(&self, i: usize) -> f64 {
        self.globals[i].1
    }
    pub fn set_global_parameter_default_value(&mut self, i: usize, value: f64) {
        self.globals[i].1 = value;
    }
    pub fn add_particle(&mut self, parameters: Vec<f64>, kind: i32) -> usize {
        self.particles.push(parameters);
        self.types.push(kind);
        self.particles.len() - 1
    }
    pub fn num_particles(&self) -> usize {
        self.particles.len()
    }
    pub fn particle_parameters(&self, i: usize) -> (&[f64], i32) {
        (&self.particles[i], self.types[i])
    }
    pub fn set_particle_parameters(&mut self, i: usize, parameters: Vec<f64>, kind: i32) {
        self.particles.set(i, parameters);
        self.types[i] = kind;
    }
    pub fn add_exclusion(&mut self, i: usize, j: usize) -> usize {
        self.exclusions.push((i, j));
        self.exclusion_set.insert((i.min(j), i.max(j)));
        self.exclusions.len() - 1
    }
    pub fn num_exclusions(&self) -> usize {
        self.exclusions.len()
    }
    pub fn exclusion_particles(&self, i: usize) -> (usize, usize) {
        self.exclusions[i]
    }
    pub fn set_exclusion_particles(&mut self, index: usize, i: usize, j: usize) {
        self.exclusions[index] = (i, j);
        self.exclusion_set = self
            .exclusions
            .iter()
            .map(|&(i, j)| (i.min(j), i.max(j)))
            .collect();
    }
    pub fn create_exclusions_from_bonds(&mut self, bonds: &[(usize, usize)], cutoff: usize) {
        if cutoff == 0 {
            return;
        }
        let n = self.particles.len();
        let mut adjacent = vec![Vec::new(); n];
        for &(i, j) in bonds {
            assert!(i < n && j < n, "bond particle index out of range");
            adjacent[i].push(j);
            adjacent[j].push(i);
        }
        let mut distances = vec![usize::MAX; n];
        let mut queue = VecDeque::new();
        for i in 0..n {
            distances.fill(usize::MAX);
            distances[i] = 0;
            queue.clear();
            queue.push_back(i);
            while let Some(j) = queue.pop_front() {
                if distances[j] >= cutoff {
                    continue;
                }
                for &k in &adjacent[j] {
                    if distances[k] == usize::MAX {
                        distances[k] = distances[j] + 1;
                        queue.push_back(k);
                    }
                }
            }
            for (j, &distance) in distances.iter().enumerate().take(i) {
                if distance <= cutoff {
                    self.add_exclusion(j, i);
                }
            }
        }
    }
    pub fn type_filter(&self, index: usize) -> &[i32] {
        &self.filters[index]
    }
    pub fn set_type_filter(&mut self, index: usize, mut types: Vec<i32>) {
        types.sort_unstable();
        types.dedup();
        self.filters[index] = types;
    }
    pub fn nonbonded_method(&self) -> CustomManyParticleMethod {
        self.method
    }
    pub fn set_nonbonded_method(&mut self, method: CustomManyParticleMethod) {
        self.method = method;
    }
    pub fn permutation_mode(&self) -> PermutationMode {
        self.mode
    }
    pub fn set_permutation_mode(&mut self, mode: PermutationMode) {
        self.mode = mode;
    }
    pub fn cutoff_distance(&self) -> f64 {
        self.cutoff
    }
    pub fn set_cutoff_distance(&mut self, value: f64) {
        self.cutoff = value;
    }
    pub fn uses_periodic_boundary_conditions(&self) -> bool {
        self.method == CustomManyParticleMethod::CutoffPeriodic
    }
    pub fn add_tabulated_function(
        &mut self,
        name: &str,
        function: impl Into<crate::TabulatedFunction>,
    ) -> Result<usize, String> {
        let i = self
            .expression
            .add_tabulated_function(name, function)
            .map_err(|e| e.to_string())?;
        self.invalidate();
        Ok(i)
    }
    pub fn num_tabulated_functions(&self) -> usize {
        self.expression.num_tabulated_functions()
    }
    pub fn tabulated_function_name(&self, i: usize) -> &str {
        self.expression.tabulated_function_name(i)
    }
    pub fn tabulated_function(&self, i: usize) -> &crate::TabulatedFunction {
        self.expression.tabulated_function(i)
    }
    pub fn set_tabulated_function(
        &mut self,
        i: usize,
        function: impl Into<crate::TabulatedFunction>,
    ) -> Result<(), String> {
        self.expression
            .set_tabulated_function(i, function)
            .map_err(|e| e.to_string())?;
        self.invalidate();
        Ok(())
    }
    pub fn update_parameters_from(&mut self, source: &Self) {
        assert_eq!(
            self.particles.len(),
            source.particles.len(),
            "CustomManyParticle particle count cannot change in a Context"
        );
        assert_eq!(
            self.parameters.len(),
            source.parameters.len(),
            "CustomManyParticle parameter count cannot change in a Context"
        );
        assert!(
            source
                .particles
                .iter()
                .all(|p| p.len() == self.parameters.len()),
            "incorrect CustomManyParticle parameter count"
        );
        let changed = (0..source.num_tabulated_functions()).any(|i| {
            let name = source.tabulated_function_name(i);
            let old = (0..self.num_tabulated_functions())
                .find(|&j| self.tabulated_function_name(j) == name)
                .map_or(0, |j| self.tabulated_function(j).update_count());
            old != source.tabulated_function(i).update_count()
        });
        self.particles.copy_from(&source.particles);
        if changed {
            assert_eq!(
                self.globals
                    .iter()
                    .map(|(name, _)| name)
                    .collect::<Vec<_>>(),
                source
                    .globals
                    .iter()
                    .map(|(name, _)| name)
                    .collect::<Vec<_>>(),
                "CustomManyParticle global definitions cannot change in a Context"
            );
            self.source.clone_from(&source.source);
            self.expression = source.expression.clone();
            self.parameters.clone_from(&source.parameters);
            self.types.clone_from(&source.types);
            self.filters.clone_from(&source.filters);
            self.exclusions.clone_from(&source.exclusions);
            self.exclusion_set.clone_from(&source.exclusion_set);
            self.mode = source.mode;
            self.method = source.method;
            self.cutoff = source.cutoff;
            self.invalidate();
        }
    }
    fn excluded(&self, i: usize, j: usize) -> bool {
        self.exclusion_set.contains(&(i.min(j), i.max(j)))
    }
    fn within_cutoff(
        &self,
        positions: &[f64],
        i: usize,
        j: usize,
        vectors: Option<&[[f64; 3]; 3]>,
    ) -> bool {
        let mut delta =
            std::array::from_fn(|axis| positions[3 * j + axis] - positions[3 * i + axis]);
        if let Some(v) = vectors {
            apply_minimum_image_ortho(&mut delta, v);
        }
        delta.iter().map(|v| v * v).sum::<f64>().sqrt() < self.cutoff
    }
    fn accepts(&self, index: usize, particle: usize) -> bool {
        self.filters[index].is_empty()
            || self.filters[index]
                .binary_search(&self.types[particle])
                .is_ok()
    }
    fn reorder(&self, particles: &mut [usize], fixed: usize) -> bool {
        if fixed == particles.len() {
            return true;
        }
        for index in fixed..particles.len() {
            particles.swap(fixed, index);
            let accepted =
                self.accepts(fixed, particles[fixed]) && self.reorder(particles, fixed + 1);
            if accepted {
                return true;
            }
            particles.swap(fixed, index);
        }
        false
    }
    fn interaction(
        &self,
        selected: &[usize],
        permuted: &mut [usize],
        positions: &[f64],
        evaluator: &mut Evaluator,
        forces: &mut [f64],
    ) -> f64 {
        permuted.copy_from_slice(selected);
        // Reference bypasses filters when only one ordering exists.
        let single_order = self.count == 1
            || (self.count == 2 && self.mode == PermutationMode::UniqueCentralParticle);
        if !single_order && self.filters.iter().any(|f| !f.is_empty()) {
            let first = usize::from(self.mode == PermutationMode::UniqueCentralParticle);
            if (first == 1 && !self.accepts(0, permuted[0])) || !self.reorder(permuted, first) {
                return 0.;
            }
        }
        evaluator.evaluate(self, permuted, positions, forces)
    }
    #[allow(clippy::too_many_arguments)]
    fn extend(
        &self,
        selected: &mut Vec<usize>,
        permuted: &mut [usize],
        candidates: &[usize],
        start: usize,
        graph: Option<&Graph>,
        positions: &[f64],
        evaluator: &mut Evaluator,
        forces: &mut [f64],
    ) -> f64 {
        if selected.len() == self.count {
            return self.interaction(selected, permuted, positions, evaluator, forces);
        }
        let mut energy = 0.;
        for index in start..candidates.len() {
            if candidates.len() - index < self.count - selected.len() {
                break;
            }
            let candidate = candidates[index];
            if candidate == selected[0] {
                continue;
            }
            let valid = selected.iter().all(|&p| {
                if self.excluded(p, candidate) {
                    return false;
                }
                self.mode == PermutationMode::UniqueCentralParticle
                    || graph.is_none_or(|g| g.neighbors(p).binary_search(&candidate).is_ok())
            });
            if !valid {
                continue;
            }
            selected.push(candidate);
            energy += self.extend(
                selected,
                permuted,
                candidates,
                index + 1,
                graph,
                positions,
                evaluator,
                forces,
            );
            selected.pop();
        }
        energy
    }
    fn evaluate(
        &self,
        positions: &[f64],
        vectors: Option<&[[f64; 3]; 3]>,
        work: &mut Work,
        forces: &mut [f64],
    ) -> f64 {
        let n = self.particles.len();
        assert_eq!(
            positions.len(),
            3 * n,
            "CustomManyParticle particle count must match positions"
        );
        let (periodic, box_cutoff) = self
            .kernel_box
            .unwrap_or((self.uses_periodic_boundary_conditions(), self.cutoff));
        let vectors = if periodic {
            assert!(
                self.method != CustomManyParticleMethod::NoCutoff,
                "a periodic CustomManyParticle Context requires a cutoff interaction"
            );
            let vectors = vectors.expect("periodic CustomManyParticle requires box vectors");
            assert!(
                (0..3).all(|i| vectors[i][i] >= 2. * box_cutoff.max(self.cutoff)),
                "CustomManyParticle cutoff exceeds half a periodic box width"
            );
            Some(vectors)
        } else {
            None
        };
        assert!(
            !self.uses_periodic_boundary_conditions() || periodic,
            "periodic mode cannot be introduced by a CustomManyParticle table update"
        );
        if work.evaluator.is_none() {
            work.evaluator = Some(Evaluator::new(self));
        }
        let evaluator = work.evaluator.as_mut().unwrap();
        evaluator.prepare(
            self,
            vectors.filter(|_| self.uses_periodic_boundary_conditions()),
        );
        forces.fill(0.);
        if n < self.count {
            return 0.;
        }
        work.permuted.resize(self.count, 0);
        work.selected.clear();
        let mut energy = 0.;
        if self.count == 1 {
            for i in 0..n {
                energy += self.interaction(&[i], &mut work.permuted, positions, evaluator, forces);
            }
        } else if self.count == 2 && self.method != CustomManyParticleMethod::NoCutoff {
            self.cells.visit(positions, self.cutoff, vectors, |i, j| {
                if self.excluded(i, j) || !self.within_cutoff(positions, i, j, vectors) {
                    return;
                }
                energy +=
                    self.interaction(&[i, j], &mut work.permuted, positions, evaluator, forces);
                if self.mode == PermutationMode::UniqueCentralParticle {
                    energy +=
                        self.interaction(&[j, i], &mut work.permuted, positions, evaluator, forces);
                }
            });
        } else {
            let cutoff = self.method != CustomManyParticleMethod::NoCutoff;
            if cutoff {
                work.graph.build(self, positions, vectors);
            } else {
                work.candidates.clear();
                work.candidates.extend(0..n);
            }
            for center in 0..n {
                work.selected.push(center);
                let candidates = if cutoff {
                    work.graph.neighbors(center)
                } else {
                    &work.candidates
                };
                let start = if self.mode == PermutationMode::SinglePermutation {
                    candidates.partition_point(|&p| p <= center)
                } else {
                    0
                };
                energy += self.extend(
                    &mut work.selected,
                    &mut work.permuted,
                    candidates,
                    start,
                    cutoff.then_some(&work.graph),
                    positions,
                    evaluator,
                    forces,
                );
                work.selected.clear();
            }
        }
        energy
    }
}
impl<B: Backend> Force<B> for CustomManyParticleForce {
    super::force_metadata_impl!();
    fn initialize(&mut self, n: usize, vectors: Option<&[[f64; 3]; 3]>) {
        assert_eq!(
            n,
            self.particles.len(),
            "CustomManyParticle particle count must match System"
        );
        assert!(
            self.particles
                .iter()
                .all(|p| p.len() == self.parameters.len()),
            "incorrect CustomManyParticle parameter count"
        );
        let mut seen = HashSet::new();
        for &(i, j) in &self.exclusions {
            assert!(
                i < n && j < n,
                "CustomManyParticle exclusion index out of range"
            );
            assert!(
                seen.insert((i.min(j), i.max(j))),
                "duplicate CustomManyParticle exclusion"
            );
        }
        if self.method != CustomManyParticleMethod::NoCutoff {
            assert!(
                self.cutoff.is_finite() && self.cutoff > 0.,
                "CustomManyParticle cutoff must be positive and finite"
            );
        }
        if self.uses_periodic_boundary_conditions() {
            let v = vectors.expect("periodic CustomManyParticle requires box vectors");
            assert!(
                (0..3).all(|i| v[i][i] >= 2. * self.cutoff),
                "CustomManyParticle cutoff exceeds half a periodic box width"
            );
        }
        self.kernel_box = Some((self.uses_periodic_boundary_conditions(), self.cutoff));
        self.cache.0.lock().unwrap().evaluator = Some(Evaluator::new(self));
    }
    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.uses_periodic_boundary_conditions()
    }
    fn global_parameters(&self) -> HashMap<String, f64> {
        self.globals.iter().cloned().collect()
    }
    fn set_global_parameter(&mut self, name: &str, value: f64) -> bool {
        if let Some((_, v)) = self.globals.iter_mut().find(|(n, _)| n == name) {
            *v = value;
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
        let mut work = self.cache.0.lock().unwrap();
        let mut forces = vec![0.; 3 * self.particles.len()];
        let energy = crate::tensor::with_f64_data(positions, |p| {
            self.evaluate(p, vectors, &mut work, &mut forces)
        });
        (
            crate::tensor::from_f64_vec(forces, positions.dims(), &positions.device()),
            energy,
        )
    }
    fn prepare_host_evaluation<'a>(
        &'a self,
        positions: &'a Tensor<B, 2>,
        vectors: Option<&'a [[f64; 3]; 3]>,
    ) -> Option<HostForceEvaluation<'a>> {
        let mut work = self.cache.0.lock().unwrap();
        let mut forces = std::mem::take(&mut work.forces);
        forces.resize(3 * self.particles.len(), 0.);
        let energy = crate::tensor::with_f64_data(positions, |p| {
            self.evaluate(p, vectors, &mut work, &mut forces)
        });
        work.forces = forces;
        Some(HostForceEvaluation::new(
            energy,
            move |coefficient, forces| {
                for (out, value) in forces.iter_mut().zip(&work.forces) {
                    *out += coefficient * value;
                }
            },
        ))
    }
}
