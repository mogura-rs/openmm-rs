use super::{
    harmonic_bond::apply_minimum_image_ortho, neighbors::NeighborCells, Force, HostForceEvaluation,
};
use crate::expression::{DerivativeProgram, Expression};
use burn::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CustomGBMethod {
    #[default]
    NoCutoff,
    CutoffNonPeriodic,
    CutoffPeriodic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GBComputationType {
    SingleParticle,
    ParticlePair,
    ParticlePairNoExclusions,
}

#[derive(Clone)]
struct Computation {
    name: String,
    source: String,
    kind: GBComputationType,
    expression: Expression,
}

#[derive(Clone)]
struct ParticleParameters {
    data: Vec<f64>,
    offsets: Vec<usize>,
}
impl Default for ParticleParameters {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            offsets: vec![0],
        }
    }
}
impl ParticleParameters {
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }
    fn push(&mut self, values: Vec<f64>) {
        self.data.extend(values);
        self.offsets.push(self.data.len());
    }
    fn set(&mut self, index: usize, values: Vec<f64>) {
        let start = self.offsets[index];
        let end = self.offsets[index + 1];
        if values.len() == end - start {
            self.data[start..end].copy_from_slice(&values);
        } else {
            let count = values.len();
            self.data.splice(start..end, values);
            for offset in &mut self.offsets[index + 1..] {
                *offset = *offset - (end - start) + count;
            }
        }
    }
    fn iter(&self) -> impl Iterator<Item = &[f64]> {
        self.offsets
            .windows(2)
            .map(|range| &self.data[range[0]..range[1]])
    }
    fn copy_from(&mut self, source: &Self) {
        self.data.clone_from(&source.data);
        self.offsets.clone_from(&source.offsets);
    }
}
impl std::ops::Index<usize> for ParticleParameters {
    type Output = [f64];
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[self.offsets[index]..self.offsets[index + 1]]
    }
}

#[derive(Clone, Copy)]
enum Input {
    Distance,
    Coordinate(usize),
    Parameter(usize, usize),
    Value(usize, usize),
    Global(usize),
}

struct Evaluator {
    program: DerivativeProgram,
    inputs: Vec<(usize, Input)>,
    globals: Vec<(usize, usize)>,
    gradients: Vec<(usize, Input)>,
}

impl Evaluator {
    fn new(force: &CustomGBForce, computation: &Computation, preceding: usize) -> Self {
        let mut names = HashMap::new();
        for (i, (name, _)) in force.globals.iter().enumerate() {
            names.insert(name.clone(), Input::Global(i));
        }
        if computation.kind == GBComputationType::SingleParticle {
            for (i, name) in ["x", "y", "z"].iter().enumerate() {
                names.insert((*name).to_owned(), Input::Coordinate(i));
            }
            for (i, name) in force.parameters.iter().enumerate() {
                names.insert(name.clone(), Input::Parameter(i, 0));
            }
            for (i, value) in force.values[..preceding].iter().enumerate() {
                names.insert(value.name.clone(), Input::Value(i, 0));
            }
        } else {
            for (i, name) in force.parameters.iter().enumerate() {
                for side in 0..2 {
                    names.insert(format!("{name}{}", side + 1), Input::Parameter(i, side));
                }
            }
            names.insert("r".to_owned(), Input::Distance);
            for (i, value) in force.values[..preceding].iter().enumerate() {
                for side in 0..2 {
                    names.insert(format!("{}{}", value.name, side + 1), Input::Value(i, side));
                }
            }
        }
        computation
            .expression
            .validate_variables(names.keys().map(String::as_str))
            .unwrap();
        let inputs: Vec<_> = computation
            .expression
            .variables
            .iter()
            .map(|name| names[name])
            .collect();
        let gradients: Vec<_> = inputs
            .iter()
            .enumerate()
            .filter_map(|(i, input)| match input {
                Input::Parameter(_, _) => None,
                Input::Global(index) if !force.derivatives.contains(index) => None,
                _ => Some((i, *input)),
            })
            .collect();
        let varying: Vec<_> = inputs
            .iter()
            .map(|input| !matches!(input, Input::Global(_)))
            .collect();
        let program = DerivativeProgram::new(
            computation.expression.clone(),
            &varying,
            gradients.iter().map(|(i, _)| *i),
        );
        let globals = inputs
            .iter()
            .enumerate()
            .filter_map(|(i, input)| {
                if let Input::Global(index) = input {
                    Some((i, *index))
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

    fn prepare(&mut self, force: &CustomGBForce) {
        for &(index, global) in &self.globals {
            self.program.set(index, force.globals[global].1);
        }
        self.program.prepare();
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate<const DERIVATIVES: bool>(
        &mut self,
        force: &CustomGBForce,
        positions: &[f64],
        values: &[f64],
        particles: [usize; 2],
        distance: f64,
    ) -> f64 {
        let n = force.particles.len();
        for &(index, input) in &self.inputs {
            let value = match input {
                Input::Distance => distance,
                Input::Coordinate(axis) => positions[3 * particles[0] + axis],
                Input::Parameter(index, side) => force.particles[particles[side]][index],
                Input::Value(index, side) => values[n * index + particles[side]],
                Input::Global(_) => unreachable!(),
            };
            self.program.set(index, value);
        }
        if DERIVATIVES {
            self.program.evaluate()
        } else {
            self.program.value()
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn accumulate(
        &self,
        n: usize,
        particles: [usize; 2],
        delta: [f64; 3],
        distance: f64,
        scale: f64,
        forces: &mut [f64],
        adjoints: &mut [f64],
        derivatives: &mut [f64],
    ) {
        for &(index, input) in &self.gradients {
            let derivative = scale * self.program.derivative(index);
            match input {
                Input::Distance => {
                    if derivative != 0.0 {
                        let factor = derivative / distance;
                        for axis in 0..3 {
                            let f = factor * delta[axis];
                            forces[3 * particles[0] + axis] += f;
                            forces[3 * particles[1] + axis] -= f;
                        }
                    }
                }
                Input::Coordinate(axis) => forces[3 * particles[0] + axis] -= derivative,
                Input::Value(index, side) => adjoints[n * index + particles[side]] += derivative,
                Input::Global(index) => derivatives[index] += derivative,
                Input::Parameter(_, _) => unreachable!(),
            }
        }
    }
}

#[derive(Default)]
struct Work {
    evaluators: Vec<Evaluator>,
    values: Vec<f64>,
    adjoints: Vec<f64>,
    derivatives: Vec<f64>,
    forces: Vec<f64>,
    prepared: bool,
}
#[derive(Default)]
struct Cache(Mutex<Work>);
impl Clone for Cache {
    fn clone(&self) -> Self {
        Self::default()
    }
}

#[derive(Clone)]
pub struct CustomGBForce {
    metadata: super::ForceMetadata,
    parameters: Vec<String>,
    globals: Vec<(String, f64)>,
    derivatives: Vec<usize>,
    particles: ParticleParameters,
    exclusions: Vec<(usize, usize)>,
    exclusion_set: HashSet<(usize, usize)>,
    values: Vec<Computation>,
    energies: Vec<Computation>,
    tables: Vec<(String, crate::TabulatedFunction)>,
    method: CustomGBMethod,
    cutoff: f64,
    neighbors: NeighborCells,
    cache: Cache,
}
impl Default for CustomGBForce {
    fn default() -> Self {
        Self::new()
    }
}
impl CustomGBForce {
    super::force_metadata!();
    pub fn new() -> Self {
        Self {
            metadata: super::ForceMetadata::new("CustomGBForce"),
            parameters: Vec::new(),
            globals: Vec::new(),
            derivatives: Vec::new(),
            particles: ParticleParameters::default(),
            exclusions: Vec::new(),
            exclusion_set: HashSet::new(),
            values: Vec::new(),
            energies: Vec::new(),
            tables: Vec::new(),
            method: CustomGBMethod::NoCutoff,
            cutoff: 1.0,
            neighbors: NeighborCells::default(),
            cache: Cache::default(),
        }
    }
    fn invalidate(&mut self) {
        self.cache = Cache::default();
    }
    fn computation(
        &self,
        name: &str,
        source: &str,
        kind: GBComputationType,
    ) -> Result<Computation, String> {
        let mut expression = Expression::parse_unbound(source).map_err(|e| e.to_string())?;
        for (name, table) in &self.tables {
            expression
                .add_tabulated_function(name, table.clone())
                .map_err(|e| e.to_string())?;
        }
        Ok(Computation {
            name: name.to_owned(),
            source: source.to_owned(),
            kind,
            expression,
        })
    }
    pub fn add_computed_value(
        &mut self,
        name: &str,
        expression: &str,
        kind: GBComputationType,
    ) -> Result<usize, String> {
        self.values.push(self.computation(name, expression, kind)?);
        self.invalidate();
        Ok(self.values.len() - 1)
    }
    pub fn num_computed_values(&self) -> usize {
        self.values.len()
    }
    pub fn computed_value_parameters(&self, index: usize) -> (&str, &str, GBComputationType) {
        let value = &self.values[index];
        (&value.name, &value.source, value.kind)
    }
    pub fn set_computed_value_parameters(
        &mut self,
        index: usize,
        name: &str,
        expression: &str,
        kind: GBComputationType,
    ) -> Result<(), String> {
        self.values[index] = self.computation(name, expression, kind)?;
        self.invalidate();
        Ok(())
    }
    pub fn add_energy_term(
        &mut self,
        expression: &str,
        kind: GBComputationType,
    ) -> Result<usize, String> {
        self.energies.push(self.computation("", expression, kind)?);
        self.invalidate();
        Ok(self.energies.len() - 1)
    }
    pub fn num_energy_terms(&self) -> usize {
        self.energies.len()
    }
    pub fn energy_term_parameters(&self, index: usize) -> (&str, GBComputationType) {
        let term = &self.energies[index];
        (&term.source, term.kind)
    }
    pub fn set_energy_term_parameters(
        &mut self,
        index: usize,
        expression: &str,
        kind: GBComputationType,
    ) -> Result<(), String> {
        self.energies[index] = self.computation("", expression, kind)?;
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
    pub fn per_particle_parameter_name(&self, index: usize) -> &str {
        &self.parameters[index]
    }
    pub fn set_per_particle_parameter_name(&mut self, index: usize, name: &str) {
        self.parameters[index] = name.to_owned();
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
    pub fn global_parameter_name(&self, index: usize) -> &str {
        &self.globals[index].0
    }
    pub fn set_global_parameter_name(&mut self, index: usize, name: &str) {
        self.globals[index].0 = name.to_owned();
        self.invalidate();
    }
    pub fn global_parameter_default_value(&self, index: usize) -> f64 {
        self.globals[index].1
    }
    pub fn set_global_parameter_default_value(&mut self, index: usize, value: f64) {
        self.globals[index].1 = value;
    }
    pub fn add_energy_parameter_derivative(&mut self, name: &str) {
        let index = self
            .globals
            .iter()
            .position(|(key, _)| key == name)
            .expect("unknown global parameter");
        if !self.derivatives.contains(&index) {
            self.derivatives.push(index);
            self.invalidate();
        }
    }
    pub fn num_energy_parameter_derivatives(&self) -> usize {
        self.derivatives.len()
    }
    pub fn energy_parameter_derivative_name(&self, index: usize) -> &str {
        &self.globals[self.derivatives[index]].0
    }
    pub fn add_particle(&mut self, parameters: Vec<f64>) -> usize {
        self.particles.push(parameters);
        self.particles.len() - 1
    }
    pub fn num_particles(&self) -> usize {
        self.particles.len()
    }
    pub fn particle_parameters(&self, index: usize) -> &[f64] {
        &self.particles[index]
    }
    pub fn set_particle_parameters(&mut self, index: usize, parameters: Vec<f64>) {
        self.particles.set(index, parameters);
    }
    pub fn add_exclusion(&mut self, first: usize, second: usize) -> usize {
        self.exclusions.push((first, second));
        self.exclusion_set
            .insert((first.min(second), first.max(second)));
        self.exclusions.len() - 1
    }
    pub fn num_exclusions(&self) -> usize {
        self.exclusions.len()
    }
    pub fn exclusion_particles(&self, index: usize) -> (usize, usize) {
        self.exclusions[index]
    }
    pub fn set_exclusion_particles(&mut self, index: usize, first: usize, second: usize) {
        self.exclusions[index] = (first, second);
        self.exclusion_set = self
            .exclusions
            .iter()
            .map(|&(i, j)| (i.min(j), i.max(j)))
            .collect();
    }
    pub fn nonbonded_method(&self) -> CustomGBMethod {
        self.method
    }
    pub fn set_nonbonded_method(&mut self, method: CustomGBMethod) {
        self.method = method;
    }
    pub fn cutoff_distance(&self) -> f64 {
        self.cutoff
    }
    pub fn set_cutoff_distance(&mut self, value: f64) {
        self.cutoff = value;
    }
    pub fn uses_periodic_boundary_conditions(&self) -> bool {
        self.method == CustomGBMethod::CutoffPeriodic
    }
    pub fn add_tabulated_function(
        &mut self,
        name: &str,
        function: impl Into<crate::TabulatedFunction>,
    ) -> Result<usize, String> {
        let function = function.into();
        let mut values = self.values.clone();
        let mut energies = self.energies.clone();
        for c in values.iter_mut().chain(&mut energies) {
            c.expression
                .add_tabulated_function(name, function.clone())
                .map_err(|e| e.to_string())?;
        }
        if self.tables.iter().any(|(key, _)| key == name) {
            return Err("duplicate tabulated function name".into());
        }
        self.values = values;
        self.energies = energies;
        self.tables.push((name.to_owned(), function));
        self.invalidate();
        Ok(self.tables.len() - 1)
    }
    pub fn add_function(
        &mut self,
        name: &str,
        values: Vec<f64>,
        min: f64,
        max: f64,
    ) -> Result<usize, String> {
        let function =
            crate::Continuous1DFunction::new([values.len()], values, [[min, max]], false)
                .map_err(|e| e.to_string())?;
        self.add_tabulated_function(name, function)
    }
    pub fn num_functions(&self) -> usize {
        self.tables.len()
    }
    pub fn function_parameters(&self, index: usize) -> Result<(&str, &[f64], f64, f64), String> {
        let (name, function) = self
            .tables
            .get(index)
            .ok_or("CustomGB table index out of range")?;
        let crate::TabulatedFunction::Continuous1D(function) = function else {
            return Err("function is not a Continuous1DFunction".into());
        };
        Ok((
            name,
            function.values(),
            function.bounds()[0][0],
            function.bounds()[0][1],
        ))
    }
    pub fn set_function_parameters(
        &mut self,
        index: usize,
        name: &str,
        values: Vec<f64>,
        min: f64,
        max: f64,
    ) -> Result<(), String> {
        let mut tables = self.tables.clone();
        let (old_name, table) = tables
            .get_mut(index)
            .ok_or("CustomGB table index out of range")?;
        let crate::TabulatedFunction::Continuous1D(function) = table else {
            return Err("function is not a Continuous1DFunction".into());
        };
        function
            .set_function_parameters([values.len()], values, [[min, max]])
            .map_err(|e| e.to_string())?;
        *old_name = name.to_owned();
        let old = std::mem::replace(&mut self.tables, tables);
        let rebuilt = (|| {
            let values = self
                .values
                .iter()
                .map(|c| self.computation(&c.name, &c.source, c.kind))
                .collect::<Result<Vec<_>, _>>()?;
            let energies = self
                .energies
                .iter()
                .map(|c| self.computation(&c.name, &c.source, c.kind))
                .collect::<Result<Vec<_>, _>>()?;
            Ok::<_, String>((values, energies))
        })();
        match rebuilt {
            Ok((values, energies)) => {
                self.values = values;
                self.energies = energies;
                self.invalidate();
                Ok(())
            }
            Err(e) => {
                self.tables = old;
                Err(e)
            }
        }
    }
    pub fn num_tabulated_functions(&self) -> usize {
        self.tables.len()
    }
    pub fn tabulated_function_name(&self, index: usize) -> &str {
        &self.tables[index].0
    }
    pub fn tabulated_function(&self, index: usize) -> &crate::TabulatedFunction {
        &self.tables[index].1
    }
    pub fn set_tabulated_function(
        &mut self,
        index: usize,
        function: impl Into<crate::TabulatedFunction>,
    ) -> Result<(), String> {
        let function = function.into();
        let mut values = self.values.clone();
        let mut energies = self.energies.clone();
        for c in values.iter_mut().chain(&mut energies) {
            c.expression
                .set_tabulated_function(index, function.clone())
                .map_err(|e| e.to_string())?;
        }
        self.values = values;
        self.energies = energies;
        self.tables[index].1 = function;
        self.invalidate();
        Ok(())
    }
    pub fn update_parameters_from(&mut self, source: &Self) {
        assert_eq!(
            self.particles.len(),
            source.particles.len(),
            "CustomGB particle count cannot change in a Context"
        );
        assert_eq!(
            self.parameters, source.parameters,
            "CustomGB parameter definitions cannot change in a Context"
        );
        for parameters in source.particles.iter() {
            assert_eq!(
                parameters.len(),
                self.parameters.len(),
                "wrong CustomGB particle parameter count"
            );
        }
        let changed = source.tables.iter().any(|(name, table)| {
            let previous = self
                .tables
                .iter()
                .find(|(key, _)| key == name)
                .map_or(0, |(_, table)| table.update_count());
            previous != table.update_count()
        });
        self.particles.copy_from(&source.particles);
        if changed {
            assert_eq!(
                self.globals.iter().map(|v| &v.0).collect::<Vec<_>>(),
                source.globals.iter().map(|v| &v.0).collect::<Vec<_>>(),
                "CustomGB global definitions cannot change in a Context"
            );
            self.values.clone_from(&source.values);
            self.energies.clone_from(&source.energies);
            self.tables.clone_from(&source.tables);
            self.derivatives.clone_from(&source.derivatives);
            self.invalidate();
        }
    }
    fn check_box(&self, vectors: Option<&[[f64; 3]; 3]>) {
        if self.uses_periodic_boundary_conditions() {
            let vectors = vectors.expect("periodic CustomGB requires box vectors");
            assert!(
                (0..3).all(|i| vectors[i][i] >= 2.0 * self.cutoff),
                "CustomGB cutoff exceeds half a periodic box width"
            );
        }
    }
    fn prepare(&self, work: &mut Work) {
        if work.prepared {
            return;
        }
        assert!(
            !self.values.is_empty(),
            "CustomGB requires at least one computed value"
        );
        for (i, c) in self.values.iter().enumerate() {
            assert_eq!(
                c.kind == GBComputationType::SingleParticle,
                i != 0,
                "CustomGB requires a first pair value followed by single-particle values"
            );
            work.evaluators.push(Evaluator::new(self, c, i));
        }
        for c in &self.energies {
            work.evaluators
                .push(Evaluator::new(self, c, self.values.len()));
        }
        work.prepared = true;
    }
    fn visit_pairs(
        &self,
        positions: &[f64],
        vectors: Option<&[[f64; 3]; 3]>,
        kind: GBComputationType,
        mut pair: impl FnMut(usize, usize, [f64; 3], f64),
    ) {
        let vectors = vectors.filter(|_| self.uses_periodic_boundary_conditions());
        let mut visit = |i: usize, j: usize| {
            if kind == GBComputationType::ParticlePair
                && self.exclusion_set.contains(&(i.min(j), i.max(j)))
            {
                return;
            }
            let mut delta =
                std::array::from_fn(|axis| positions[3 * j + axis] - positions[3 * i + axis]);
            if let Some(vectors) = vectors {
                apply_minimum_image_ortho(&mut delta, vectors);
            }
            let r = delta.iter().map(|v| v * v).sum::<f64>().sqrt();
            if self.method == CustomGBMethod::NoCutoff || r < self.cutoff {
                pair(i, j, delta, r);
            }
        };
        if self.method == CustomGBMethod::NoCutoff {
            for i in 0..self.particles.len() {
                for j in i + 1..self.particles.len() {
                    visit(i, j);
                }
            }
        } else {
            self.neighbors.visit(positions, self.cutoff, vectors, visit);
        }
    }
    fn evaluate(
        &self,
        positions: &[f64],
        vectors: Option<&[[f64; 3]; 3]>,
        work: &mut Work,
        forces: &mut [f64],
    ) -> f64 {
        let n = self.particles.len();
        let m = self.values.len();
        assert_eq!(
            positions.len(),
            3 * n,
            "CustomGB particle count must match positions"
        );
        self.check_box(vectors);
        self.prepare(work);
        let Work {
            evaluators,
            values,
            adjoints,
            derivatives,
            ..
        } = work;
        for evaluator in &mut *evaluators {
            evaluator.prepare(self);
        }
        values.resize(n * m, 0.0);
        values.fill(0.0);
        adjoints.resize(n * m, 0.0);
        adjoints.fill(0.0);
        derivatives.resize(self.globals.len(), 0.0);
        derivatives.fill(0.0);
        forces.fill(0.0);
        for (k, c) in self.values.iter().enumerate() {
            let evaluator = &mut evaluators[k];
            if c.kind == GBComputationType::SingleParticle {
                for i in 0..n {
                    values[k * n + i] =
                        evaluator.evaluate::<false>(self, positions, values, [i, i], 0.0);
                }
            } else {
                self.visit_pairs(positions, vectors, c.kind, |i, j, _, r| {
                    values[k * n + i] +=
                        evaluator.evaluate::<false>(self, positions, values, [i, j], r);
                    values[k * n + j] +=
                        evaluator.evaluate::<false>(self, positions, values, [j, i], r);
                });
            }
        }
        let mut energy = 0.0;
        for (k, c) in self.energies.iter().enumerate() {
            let evaluator = &mut evaluators[m + k];
            if c.kind == GBComputationType::SingleParticle {
                for i in 0..n {
                    energy += evaluator.evaluate::<true>(self, positions, values, [i, i], 0.0);
                    evaluator.accumulate(
                        n,
                        [i, i],
                        [0.0; 3],
                        0.0,
                        1.0,
                        forces,
                        adjoints,
                        derivatives,
                    );
                }
            } else {
                self.visit_pairs(positions, vectors, c.kind, |i, j, delta, r| {
                    energy += evaluator.evaluate::<true>(self, positions, values, [i, j], r);
                    evaluator.accumulate(n, [i, j], delta, r, 1.0, forces, adjoints, derivatives);
                });
            }
        }
        for (k, c) in self.values.iter().enumerate().rev() {
            let evaluator = &mut evaluators[k];
            if c.kind == GBComputationType::SingleParticle {
                for i in 0..n {
                    let scale = adjoints[k * n + i];
                    if scale == 0.0 {
                        continue;
                    }
                    evaluator.evaluate::<true>(self, positions, values, [i, i], 0.0);
                    evaluator.accumulate(
                        n,
                        [i, i],
                        [0.0; 3],
                        0.0,
                        scale,
                        forces,
                        adjoints,
                        derivatives,
                    );
                }
            } else {
                self.visit_pairs(positions, vectors, c.kind, |i, j, delta, r| {
                    for (particles, direction) in [([i, j], delta), ([j, i], delta.map(|v| -v))] {
                        let scale = adjoints[k * n + particles[0]];
                        if scale == 0.0 {
                            continue;
                        }
                        evaluator.evaluate::<true>(self, positions, values, particles, r);
                        evaluator.accumulate(
                            n,
                            particles,
                            direction,
                            r,
                            scale,
                            forces,
                            adjoints,
                            derivatives,
                        );
                    }
                });
            }
        }
        energy
    }
}

impl<B: Backend> Force<B> for CustomGBForce {
    super::force_metadata_impl!();
    fn initialize(&mut self, n: usize, vectors: Option<&[[f64; 3]; 3]>) {
        assert_eq!(
            self.particles.len(),
            n,
            "CustomGB particle count must match System"
        );
        for p in self.particles.iter() {
            assert_eq!(
                p.len(),
                self.parameters.len(),
                "wrong CustomGB particle parameter count"
            );
        }
        assert_eq!(
            self.exclusions.len(),
            self.exclusion_set.len(),
            "duplicate CustomGB exclusion"
        );
        for &(i, j) in &self.exclusions {
            assert!(i < n && j < n, "CustomGB exclusion index out of range");
        }
        if self.method != CustomGBMethod::NoCutoff {
            assert!(
                self.cutoff.is_finite() && self.cutoff > 0.0,
                "CustomGB cutoff must be positive and finite"
            );
        }
        self.check_box(vectors);
        self.prepare(&mut self.cache.0.lock().unwrap());
    }
    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.uses_periodic_boundary_conditions()
    }
    fn global_parameters(&self) -> HashMap<String, f64> {
        self.globals.iter().cloned().collect()
    }
    fn set_global_parameter(&mut self, name: &str, value: f64) -> bool {
        let mut found = false;
        for (key, current) in &mut self.globals {
            if key == name {
                *current = value;
                found = true;
            }
        }
        found
    }
    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64) {
        let mut work = self.cache.0.lock().unwrap();
        let mut forces = vec![0.0; 3 * self.particles.len()];
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
        forces.resize(3 * self.particles.len(), 0.0);
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
    fn energy_parameter_derivatives(
        &self,
        positions: &Tensor<B, 2>,
        vectors: Option<&[[f64; 3]; 3]>,
    ) -> HashMap<String, f64> {
        let mut work = self.cache.0.lock().unwrap();
        let mut forces = std::mem::take(&mut work.forces);
        forces.resize(3 * self.particles.len(), 0.0);
        crate::tensor::with_f64_data(positions, |p| {
            self.evaluate(p, vectors, &mut work, &mut forces)
        });
        work.forces = forces;
        self.derivatives
            .iter()
            .map(|&i| (self.globals[i].0.clone(), work.derivatives[i]))
            .collect()
    }
}
