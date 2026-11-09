use super::{Force, HostForceEvaluation};
use crate::expression::{DerivativeProgram, Expression, ParseError};
use burn::prelude::*;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc, Mutex, MutexGuard,
};

#[derive(Clone, Debug, PartialEq)]
pub enum ATMTransformation {
    FixedDisplacement {
        displacement1: [f64; 3],
        displacement0: [f64; 3],
    },
    ParticleOffsetDisplacement {
        destination1: usize,
        origin1: usize,
        state0: Option<[usize; 2]>,
    },
}
impl Default for ATMTransformation {
    fn default() -> Self {
        Self::FixedDisplacement {
            displacement1: [0.; 3],
            displacement0: [0.; 3],
        }
    }
}
impl ATMTransformation {
    pub fn fixed(displacement1: [f64; 3], displacement0: [f64; 3]) -> Self {
        Self::FixedDisplacement {
            displacement1,
            displacement0,
        }
    }
    pub fn particle_offset(
        destination1: usize,
        origin1: usize,
        state0: Option<[usize; 2]>,
    ) -> Self {
        Self::ParticleOffsetDisplacement {
            destination1,
            origin1,
            state0,
        }
    }
    fn offset(&self, state: usize) -> Option<[usize; 2]> {
        match self {
            Self::ParticleOffsetDisplacement {
                destination1,
                origin1,
                state0,
            } => {
                if state == 0 {
                    *state0
                } else {
                    Some([*destination1, *origin1])
                }
            }
            _ => None,
        }
    }
    fn displacement(&self, state: usize) -> [f64; 3] {
        match self {
            Self::FixedDisplacement {
                displacement1,
                displacement0,
            } => {
                if state == 0 {
                    *displacement0
                } else {
                    *displacement1
                }
            }
            _ => [0.; 3],
        }
    }
    fn validate(&self, n: usize) {
        for state in 0..2 {
            if let Some([destination, origin]) = self.offset(state) {
                assert!(
                    destination < n && origin < n,
                    "ATM displacement particle index is out of range"
                );
            }
            assert!(
                self.displacement(state).iter().all(|x| x.is_finite()),
                "ATM displacements must be finite"
            );
        }
    }
}

struct Program {
    evaluator: DerivativeProgram,
    energies: [Option<usize>; 2],
    globals: Vec<(String, usize)>,
    derivatives: Vec<(String, Option<usize>)>,
}
struct Work<B: Backend> {
    positions: Option<Tensor<B, 2>>,
    forces: [Vec<f64>; 2],
    program: Option<Program>,
    energies: [f64; 2],
    coefficients: [f64; 2],
    combined: f64,
}
impl<B: Backend> Default for Work<B> {
    fn default() -> Self {
        Self {
            positions: None,
            forces: [Vec::new(), Vec::new()],
            program: None,
            energies: [0.; 2],
            coefficients: [0.; 2],
            combined: 0.,
        }
    }
}

pub struct ATMForce<B: Backend> {
    metadata: super::ForceMetadata,
    group: Arc<AtomicU32>,
    energy: String,
    expression: Expression,
    forces: Vec<Box<dyn Force<B>>>,
    particles: Vec<ATMTransformation>,
    global_names: Vec<String>,
    globals: HashMap<String, f64>,
    parameters: Option<HashMap<String, f64>>,
    derivative_names: Vec<String>,
    captured_periodic: Option<bool>,
    workspace: Mutex<Work<B>>,
}
impl<B: Backend> Clone for ATMForce<B> {
    fn clone(&self) -> Self {
        Self {
            metadata: self.metadata.clone(),
            group: Arc::new(AtomicU32::new(self.force_group())),
            energy: self.energy.clone(),
            expression: self.expression.clone(),
            forces: self.forces.clone(),
            particles: self.particles.clone(),
            global_names: self.global_names.clone(),
            globals: self.globals.clone(),
            parameters: self.parameters.clone(),
            derivative_names: self.derivative_names.clone(),
            captured_periodic: self.captured_periodic,
            workspace: Mutex::default(),
        }
    }
}
impl<B: Backend> ATMForce<B> {
    pub fn new(energy: &str) -> Result<Self, ParseError> {
        Ok(Self {
            metadata: super::ForceMetadata::new("ATMForce"),
            group: Arc::new(AtomicU32::new(0)),
            energy: energy.to_owned(),
            expression: Expression::parse_unbound(energy)?,
            forces: Vec::new(),
            particles: Vec::new(),
            global_names: Vec::new(),
            globals: HashMap::new(),
            parameters: None,
            derivative_names: Vec::new(),
            captured_periodic: None,
            workspace: Mutex::default(),
        })
    }
    pub fn with_default_parameters(values: [f64; 9]) -> Result<Self, String> {
        let [lambda1, lambda2, alpha, _, _, umax, ubcore, acore, direction] = values;
        if alpha < 0. || (lambda1 != lambda2 && alpha == 0.) {
            return Err("ATM alpha must be nonnegative and positive for unequal lambdas".into());
        }
        if umax < ubcore || acore < 0. || (direction != 1. && direction != -1.) {
            return Err("invalid ATM soft-core parameters or direction".into());
        }
        let expression = "select(step(Direction),u0,u1)+select(Lambda2-Lambda1,((Lambda2-Lambda1)/Alpha)*log(1+exp(-Alpha*(usc-Uh)))+Lambda2*usc+W0,Lambda2*usc+W0);usc=select(Acore,select(step(u-Ubcore),(Umax-Ubcore)*fsc+Ubcore,u),u);fsc=(z^Acore-1)/(z^Acore+1);z=1+2*(y/Acore)+2*(y/Acore)^2;y=(u-Ubcore)/(Umax-Ubcore);u=select(step(Direction),1,-1)*(u1-u0)";
        let mut force = Self::new(expression).map_err(|e| e.to_string())?;
        for (name, value) in [
            "Lambda1",
            "Lambda2",
            "Alpha",
            "Uh",
            "W0",
            "Umax",
            "Ubcore",
            "Acore",
            "Direction",
        ]
        .into_iter()
        .zip(values)
        {
            force.add_global_parameter(name, value);
        }
        Ok(force)
    }
    pub fn snapshot_for_context(&self) -> Self {
        let mut force = self.clone();
        force.group = self.group.clone();
        force
    }
    pub fn force_group(&self) -> u32 {
        self.group.load(Ordering::Relaxed)
    }
    pub fn set_force_group(&mut self, group: u32) {
        assert!(group < 32, "force group must be between zero and 31");
        self.group.store(group, Ordering::Relaxed);
    }
    pub fn name(&self) -> &str {
        &self.metadata.name
    }
    pub fn set_name(&mut self, name: &str) {
        self.metadata.name = name.to_owned();
    }
    pub fn energy_function(&self) -> &str {
        &self.energy
    }
    pub fn set_energy_function(&mut self, energy: &str) -> Result<(), String> {
        self.expression = Expression::parse_unbound(energy).map_err(|e| e.to_string())?;
        self.energy = energy.to_owned();
        self.workspace.get_mut().unwrap().program = None;
        Ok(())
    }
    pub fn add_particle(&mut self, transformation: ATMTransformation) -> usize {
        self.particles.push(transformation);
        self.particles.len() - 1
    }
    pub fn num_particles(&self) -> usize {
        self.particles.len()
    }
    pub fn particle_transformation(&self, index: usize) -> &ATMTransformation {
        &self.particles[index]
    }
    pub fn set_particle_transformation(&mut self, index: usize, transformation: ATMTransformation) {
        self.particles[index] = transformation;
    }
    pub fn add_force(&mut self, force: Box<dyn Force<B>>) -> usize {
        self.forces.push(force);
        self.parameters = None;
        self.forces.len() - 1
    }
    pub fn num_forces(&self) -> usize {
        self.forces.len()
    }
    pub fn force(&self, index: usize) -> &(dyn Force<B> + 'static) {
        self.forces[index].as_ref()
    }
    pub fn force_mut(&mut self, index: usize) -> &mut (dyn Force<B> + 'static) {
        self.forces[index].as_mut()
    }
    pub fn replace_force(&mut self, index: usize, force: Box<dyn Force<B>>) {
        self.forces[index] = force;
        self.parameters = None;
    }
    pub fn add_global_parameter(&mut self, name: &str, value: f64) -> usize {
        if !self.globals.contains_key(name) {
            self.global_names.push(name.to_owned());
        }
        self.globals.insert(name.to_owned(), value);
        self.parameters = None;
        self.workspace.get_mut().unwrap().program = None;
        self.global_names.iter().position(|n| n == name).unwrap()
    }
    pub fn num_global_parameters(&self) -> usize {
        self.global_names.len()
    }
    pub fn global_parameter_name(&self, index: usize) -> &str {
        &self.global_names[index]
    }
    pub fn set_global_parameter_name(&mut self, index: usize, name: &str) {
        let old = std::mem::replace(&mut self.global_names[index], name.to_owned());
        let value = self.globals.remove(&old).unwrap();
        self.globals.insert(name.to_owned(), value);
        for derivative in &mut self.derivative_names {
            if *derivative == old {
                *derivative = name.to_owned();
            }
        }
        self.parameters = None;
        self.workspace.get_mut().unwrap().program = None;
    }
    pub fn global_parameter_default_value(&self, index: usize) -> f64 {
        self.globals[&self.global_names[index]]
    }
    pub fn set_global_parameter_default_value(&mut self, index: usize, value: f64) {
        self.globals.insert(self.global_names[index].clone(), value);
        self.parameters = None;
    }
    pub fn add_energy_parameter_derivative(&mut self, name: &str) {
        assert!(
            self.globals.contains_key(name),
            "unknown ATM global parameter"
        );
        self.derivative_names.push(name.to_owned());
        self.workspace.get_mut().unwrap().program = None;
    }
    pub fn num_energy_parameter_derivatives(&self) -> usize {
        self.derivative_names.len()
    }
    pub fn energy_parameter_derivative_name(&self, index: usize) -> &str {
        &self.derivative_names[index]
    }
    pub fn update_parameters_from(&mut self, source: &Self) {
        assert_eq!(
            self.particles.len(),
            source.particles.len(),
            "ATM particle count cannot change"
        );
        for transformation in &source.particles {
            transformation.validate(self.particles.len());
        }
        self.particles.clone_from(&source.particles);
    }
    pub fn uses_periodic_boundary_conditions(&self) -> bool {
        self.captured_periodic
            .unwrap_or_else(|| self.children_use_periodic_boundary_conditions())
    }
    fn children_use_periodic_boundary_conditions(&self) -> bool {
        self.forces
            .iter()
            .any(|force| force.uses_periodic_boundary_conditions())
    }
    fn parameter_values(&self) -> HashMap<String, f64> {
        let mut values = HashMap::new();
        for force in &self.forces {
            for (name, value) in force.global_parameters() {
                if let Some(previous) = values.insert(name.clone(), value) {
                    assert!(
                        previous == value,
                        "ATM child parameter '{name}' has conflicting defaults"
                    );
                }
            }
        }
        values.extend(
            self.globals
                .iter()
                .map(|(name, &value)| (name.clone(), value)),
        );
        values
    }
    fn program(&self) -> Program {
        let expression = self.expression.clone();
        expression
            .validate_functions()
            .unwrap_or_else(|e| panic!("{e}"));
        expression
            .validate_variables(
                ["u0", "u1"]
                    .into_iter()
                    .chain(self.global_names.iter().map(String::as_str)),
            )
            .unwrap_or_else(|e| panic!("{e}"));
        let energies = [
            expression.variable_index("u0"),
            expression.variable_index("u1"),
        ];
        let globals = self
            .global_names
            .iter()
            .filter_map(|name| expression.variable_index(name).map(|i| (name.clone(), i)))
            .collect();
        let derivatives = self
            .derivative_names
            .iter()
            .map(|name| (name.clone(), expression.variable_index(name)))
            .collect();
        let count = expression.variables.len();
        Program {
            evaluator: DerivativeProgram::new(expression, &vec![true; count], 0..count),
            energies,
            globals,
            derivatives,
        }
    }
    fn evaluate<'a>(
        &'a self,
        positions: &Tensor<B, 2>,
        vectors: Option<&[[f64; 3]; 3]>,
        want_forces: bool,
    ) -> MutexGuard<'a, Work<B>> {
        let n = positions.dims()[0];
        assert_eq!(
            n,
            self.particles.len(),
            "ATM requires one transformation per System particle"
        );
        let mut work = self.workspace.lock().unwrap();
        if work
            .positions
            .as_ref()
            .is_none_or(|p| p.dims() != positions.dims())
        {
            work.positions = Some(crate::tensor::zeros([n, 3], &positions.device()));
        }
        for state in 0..2 {
            crate::tensor::with_f64_data(positions, |input| {
                let transform = |output: &mut [f64]| {
                    for (i, transformation) in self.particles.iter().enumerate() {
                        let offset = transformation.offset(state);
                        let fixed = transformation.displacement(state);
                        for axis in 0..3 {
                            output[3 * i + axis] = input[3 * i + axis]
                                + offset.map_or(fixed[axis], |[destination, origin]| {
                                    input[3 * destination + axis] - input[3 * origin + axis]
                                });
                        }
                    }
                };
                if crate::tensor::with_mut_ndarray_f64(work.positions.as_mut().unwrap(), transform)
                    .is_none()
                {
                    let mut values = vec![0.; 3 * n];
                    transform(&mut values);
                    work.positions = Some(crate::tensor::from_f64_vec(
                        values,
                        [n, 3],
                        &positions.device(),
                    ));
                }
            });
            if want_forces {
                work.forces[state].resize(3 * n, 0.);
                work.forces[state].fill(0.);
            }
            let Work {
                positions: displaced,
                forces,
                ..
            } = &mut *work;
            let displaced = displaced.as_ref().unwrap();
            let mut energy = 0.;
            for force in &self.forces {
                if let Some(prepared) = force.prepare_host_evaluation(displaced, vectors) {
                    energy += prepared.energy;
                    if want_forces {
                        prepared.accumulate(1., &mut forces[state]);
                    }
                } else {
                    let (values, potential) = force.calc_forces_and_energy(displaced, vectors);
                    energy += potential;
                    if want_forces {
                        crate::tensor::with_f64_data(&values, |values| {
                            for (total, value) in forces[state].iter_mut().zip(values) {
                                *total += value;
                            }
                        });
                    }
                }
            }
            work.energies[state] = energy;
        }
        if work.program.is_none() {
            work.program = Some(self.program());
        }
        let Work {
            program,
            energies,
            coefficients,
            combined,
            ..
        } = &mut *work;
        let program = program.as_mut().unwrap();
        for (name, index) in &program.globals {
            program.evaluator.set(*index, self.globals[name]);
        }
        for (index, energy) in program.energies.into_iter().zip(*energies) {
            if let Some(index) = index {
                program.evaluator.set(index, energy);
            }
        }
        program.evaluator.prepare();
        let preliminary = program.evaluator.evaluate();
        let mut clamped = false;
        for state in 0..2 {
            coefficients[state] =
                program.energies[state].map_or(0., |index| program.evaluator.derivative(index));
            if coefficients[state].abs() < f32::MIN_POSITIVE as f64 && !energies[state].is_finite()
            {
                clamped = true;
                energies[state] = f32::MAX as f64;
                if let Some(index) = program.energies[state] {
                    program.evaluator.set(index, energies[state]);
                }
            }
        }
        *combined = if clamped {
            program.evaluator.evaluate()
        } else {
            preliminary
        };
        work
    }
    fn accumulate(&self, work: &Work<B>, scale: f64, output: &mut [f64]) {
        for state in 0..2 {
            let magnitude = work.coefficients[state].abs();
            if magnitude.is_nan() || magnitude <= f32::MIN_POSITIVE as f64 {
                continue;
            }
            let coefficient = scale * work.coefficients[state];
            for (i, transformation) in self.particles.iter().enumerate() {
                for axis in 0..3 {
                    let value = coefficient * work.forces[state][3 * i + axis];
                    output[3 * i + axis] += value;
                    if let Some([destination, origin]) = transformation.offset(state) {
                        output[3 * destination + axis] += value;
                        output[3 * origin + axis] -= value;
                    }
                }
            }
        }
    }
    pub fn perturbation_energy(
        &self,
        positions: &Tensor<B, 2>,
        vectors: Option<&[[f64; 3]; 3]>,
    ) -> (f64, f64, f64) {
        let work = self.evaluate(positions, vectors, false);
        (work.energies[1], work.energies[0], work.combined)
    }
}
impl<B: Backend> Force<B> for ATMForce<B> {
    fn force_group(&self) -> u32 {
        self.force_group()
    }
    fn has_dynamic_force_group(&self) -> bool {
        true
    }
    fn name(&self) -> &str {
        self.name()
    }
    fn prepare_for_inner_context(&mut self) {
        self.captured_periodic = None;
        self.group = Arc::new(AtomicU32::new(self.force_group()));
        for force in &mut self.forces {
            force.prepare_for_inner_context();
        }
    }
    fn initialize_masses(&mut self, masses: &[f64]) {
        for force in &mut self.forces {
            force.initialize_masses(masses);
        }
    }
    fn initialize(&mut self, n: usize, vectors: Option<&[[f64; 3]; 3]>) {
        assert_eq!(
            n,
            self.particles.len(),
            "ATM requires one transformation per System particle"
        );
        for transformation in &self.particles {
            transformation.validate(n);
        }
        self.captured_periodic = Some(self.uses_periodic_boundary_conditions());
        for force in &mut self.forces {
            force.prepare_for_inner_context();
        }
        assert!(
            self.children_use_periodic_boundary_conditions()
                || !self
                    .forces
                    .iter()
                    .any(|force| force.requires_periodic_system()),
            "an ATM child barostat requires a periodic child force"
        );
        for force in &mut self.forces {
            force.initialize_constraints(n, &[]);
            force.initialize(n, vectors);
        }
        let parameters = self.parameter_values();
        self.parameters = Some(parameters.clone());
        for (name, value) in parameters {
            self.set_global_parameter(&name, value);
        }
        let program = self.program();
        self.workspace.get_mut().unwrap().program = Some(program);
    }
    fn uses_periodic_boundary_conditions(&self) -> bool {
        self.uses_periodic_boundary_conditions()
    }
    fn bonded_particles(&self) -> Vec<(usize, usize)> {
        self.forces
            .iter()
            .flat_map(|f| f.bonded_particles())
            .collect()
    }
    fn global_parameters(&self) -> HashMap<String, f64> {
        self.parameters
            .clone()
            .unwrap_or_else(|| self.parameter_values())
    }
    fn set_global_parameter(&mut self, name: &str, value: f64) -> bool {
        let mut known = false;
        if let Some(current) = self
            .parameters
            .as_mut()
            .and_then(|values| values.get_mut(name))
        {
            *current = value;
            known = true;
        }
        if let Some(current) = self.globals.get_mut(name) {
            *current = value;
            known = true;
        }
        for force in &mut self.forces {
            known |= force.set_global_parameter(name, value);
        }
        known
    }
    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64) {
        let work = self.evaluate(positions, vectors, true);
        let mut forces = vec![0.; positions.dims()[0] * 3];
        self.accumulate(&work, 1., &mut forces);
        (
            crate::tensor::from_f64_vec(forces, positions.dims(), &positions.device()),
            work.combined,
        )
    }
    fn prepare_host_evaluation<'a>(
        &'a self,
        positions: &'a Tensor<B, 2>,
        vectors: Option<&'a [[f64; 3]; 3]>,
    ) -> Option<HostForceEvaluation<'a>> {
        let work = self.evaluate(positions, vectors, true);
        Some(HostForceEvaluation::new(
            work.combined,
            move |coefficient, forces| self.accumulate(&work, coefficient, forces),
        ))
    }
    fn energy_parameter_derivatives(
        &self,
        positions: &Tensor<B, 2>,
        vectors: Option<&[[f64; 3]; 3]>,
    ) -> HashMap<String, f64> {
        let work = self.evaluate(positions, vectors, false);
        let program = work.program.as_ref().unwrap();
        let mut derivatives = HashMap::new();
        for (name, index) in &program.derivatives {
            *derivatives.entry(name.clone()).or_insert(0.) +=
                index.map_or(0., |i| program.evaluator.derivative(i));
        }
        derivatives
    }
}
