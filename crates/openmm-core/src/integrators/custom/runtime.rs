use super::{compiler::*, failure, ComputationType, Data, Variables};
use crate::{
    constraints::ConstraintError,
    integrators::{ContextProgram, ContextUpdateSettings},
    Context,
};
use burn::prelude::{Backend, Tensor};
use rand::Rng;
use rand_chacha::ChaCha12Rng;
use rand_distr::StandardNormal;
use std::sync::Mutex;

struct Inputs<'a> {
    variables: &'a Variables,
    parameters: &'a [f64],
    cache: Option<&'a ForceCache>,
    external_forces: Option<&'a [f64]>,
    position: [f64; 3],
    velocity: [f64; 3],
    mass: f64,
    atom: usize,
    random: [[f64; 3]; 2],
}

impl Inputs<'_> {
    fn value(&self, binding: Binding) -> [f64; 3] {
        match binding {
            Binding::Dt => [self.variables.step_size; 3],
            Binding::Global(i) => [self.variables.globals[i].1; 3],
            Binding::Parameter(i) => [self.parameters[i]; 3],
            Binding::PerDof(i) => self.variables.per_dof[i].1[self.atom],
            Binding::Position => self.position,
            Binding::Velocity => self.velocity,
            Binding::Mass => [self.mass; 3],
            Binding::Force => self
                .external_forces
                .or_else(|| self.cache.map(|cache| cache.forces.as_slice()))
                .map_or([0.0; 3], |forces| {
                    std::array::from_fn(|axis| forces[3 * self.atom + axis])
                }),
            Binding::Energy => [self.cache.map_or(0.0, |cache| cache.energy); 3],
            Binding::Derivative(i) => [self.cache.map_or(0.0, |cache| cache.derivatives[i]); 3],
            Binding::Uniform => self.random[0],
            Binding::Gaussian => self.random[1],
        }
    }
}

impl CompiledExpression {
    fn prepare(&mut self, inputs: &Inputs<'_>) {
        match &mut self.evaluator {
            Evaluator::Scalar(program) => {
                for &(index, binding) in &self.constants {
                    program.set(index, inputs.value(binding)[0]);
                }
                program.prepare();
            }
            Evaluator::Vector(program) => {
                for &(index, binding) in &self.constants {
                    program.set(index, inputs.value(binding));
                }
                program.prepare();
            }
        }
    }
    fn evaluate(&mut self, inputs: &Inputs<'_>, global: bool) -> [f64; 3] {
        match &mut self.evaluator {
            Evaluator::Vector(program) => {
                for &(index, binding) in &self.varying {
                    program.set(index, inputs.value(binding));
                }
                program.evaluate()
            }
            Evaluator::Scalar(program) => {
                let mut result = [0.0; 3];
                for (axis, output) in result
                    .iter_mut()
                    .enumerate()
                    .take(if global { 1 } else { 3 })
                {
                    for &(index, binding) in &self.varying {
                        program.set(index, inputs.value(binding)[axis]);
                    }
                    *output = program.evaluate();
                }
                result
            }
        }
    }
}

fn noise(random: &mut ChaCha12Rng, enabled: bool, global: bool) -> [[f64; 3]; 2] {
    let mut result = [[0.0; 3]; 2];
    if enabled {
        let [uniform, gaussian] = &mut result;
        for (uniform, gaussian) in uniform
            .iter_mut()
            .zip(gaussian)
            .take(if global { 1 } else { 3 })
        {
            *uniform = random.random::<f64>();
            *gaussian = random.sample(StandardNormal);
        }
    }
    result
}

fn set_global<B: Backend>(
    target: Target,
    value: f64,
    variables: &mut Variables,
    compiled: &mut Compiled,
    context: &mut Context<B>,
) -> Result<(), ConstraintError> {
    match target {
        Target::Dt => {
            if !value.is_finite() || value < 0.0 {
                return Err(failure("step size must be nonnegative and finite"));
            }
            variables.step_size = value;
        }
        Target::Global(i) => variables.globals[i].1 = value,
        Target::Parameter(i) => {
            compiled.parameters[i] = value;
            context.set_parameter(&compiled.parameter_names[i], value);
        }
        _ => return Err(failure("illegal global output target")),
    }
    Ok(())
}

impl Compiled {
    fn refresh_parameters<B: Backend>(&mut self, context: &Context<B>) {
        for (value, name) in self.parameters.iter_mut().zip(&self.parameter_names) {
            *value = context.get_parameter(name);
        }
    }
    fn fill_cache<B: Backend>(&mut self, index: usize, context: &Context<B>) {
        let cache = &mut self.caches[index];
        if cache.valid {
            return;
        }
        let (forces, energy) = context.calc_forces_and_energy_groups(cache.groups);
        crate::tensor::with_f64_data(&forces, |forces| {
            cache.forces.resize(forces.len(), 0.0);
            cache.forces.copy_from_slice(forces);
        });
        cache.energy = energy;
        if cache.needs_derivatives {
            let derivatives = context.program_parameter_derivatives(cache.groups);
            for (value, name) in cache.derivatives.iter_mut().zip(&self.parameter_names) {
                *value = derivatives.get(name).copied().unwrap_or(0.0);
            }
        }
        cache.valid = true;
    }
    fn execute<B: Backend>(
        &mut self,
        data: &mut Data,
        context: &mut Context<B>,
        updates: ContextUpdateSettings,
    ) -> Result<f64, ConstraintError> {
        self.refresh_parameters(context);
        if self.revision != Some(context.force_state_revision()) {
            for cache in &mut self.caches {
                cache.valid = false;
            }
        }
        if self.steps.iter().any(|step| {
            matches!(
                step.kind,
                ComputationType::ConstrainPositions | ComputationType::ConstrainVelocities
            )
        }) {
            context.read_program_particles(|p, _, _| {
                self.old_positions.resize(p.len(), 0.0);
                self.old_positions.copy_from_slice(p);
            });
        }
        let mut index = 0;
        while index < self.steps.len() {
            let cache_index = self.steps[index].cache;
            if let Some(i) = cache_index {
                self.fill_cache(i, context);
            }
            let mut next = index + 1;
            let mut invalidates = self.steps[index].invalidates;
            use ComputationType::*;
            match self.steps[index].kind {
                ComputeGlobal | IfBlockStart | WhileBlockStart => {
                    let step = &mut self.steps[index];
                    let random = data.random.with_random(|random| {
                        noise(
                            random,
                            step.expressions.iter().any(|expr| expr.random),
                            true,
                        )
                    });
                    let inputs = Inputs {
                        variables: &data.variables,
                        parameters: &self.parameters,
                        cache: cache_index.map(|i| &self.caches[i]),
                        external_forces: None,
                        position: [0.0; 3],
                        velocity: [0.0; 3],
                        mass: 0.0,
                        atom: 0,
                        random,
                    };
                    for expr in &mut step.expressions {
                        expr.prepare(&inputs);
                    }
                    let lhs = step.expressions[0].evaluate(&inputs, true)[0];
                    if step.kind == ComputeGlobal {
                        let target = step.target;
                        set_global(target, lhs, &mut data.variables, self, context)?;
                    } else {
                        let rhs = step.expressions[1].evaluate(&inputs, true)[0];
                        if !step.comparison.unwrap().test(lhs, rhs) {
                            next = step.jump.unwrap();
                        }
                    }
                }
                ComputePerDof | ComputeSum => {
                    let step = &mut self.steps[index];
                    let is_sum = step.kind == ComputeSum;
                    let target = step.target;
                    let expr = &mut step.expressions[0];
                    let cache = cache_index.map(|i| &self.caches[i]);
                    expr.prepare(&Inputs {
                        variables: &data.variables,
                        parameters: &self.parameters,
                        cache,
                        external_forces: None,
                        position: [0.0; 3],
                        velocity: [0.0; 3],
                        mass: 0.0,
                        atom: 0,
                        random: [[0.0; 3]; 2],
                    });

                    let sum = data.random.with_random(|random| {
                        let mut evaluate =
                            |variables: &Variables, atom, position, velocity, mass| {
                                let inputs = Inputs {
                                    variables,
                                    parameters: &self.parameters,
                                    cache,
                                    external_forces: None,
                                    position,
                                    velocity,
                                    mass,
                                    atom,
                                    random: noise(random, expr.random, false),
                                };
                                expr.evaluate(&inputs, false)
                            };
                        if matches!(target, Target::Position | Target::Velocity) {
                            context.update_program_values(
                                matches!(target, Target::Position),
                                |atom, p, v, mass| evaluate(&data.variables, atom, p, v, mass),
                            );
                            0.0
                        } else {
                            context.read_program_particles(|p, v, masses| {
                                let mut sum = 0.0;
                                for (atom, &mass) in masses.iter().enumerate() {
                                    if mass == 0.0 {
                                        continue;
                                    }
                                    let result = evaluate(
                                        &data.variables,
                                        atom,
                                        std::array::from_fn(|a| p[3 * atom + a]),
                                        std::array::from_fn(|a| v[3 * atom + a]),
                                        mass,
                                    );
                                    if is_sum {
                                        sum += (result[0] + result[1]) + result[2];
                                    } else if let Target::PerDof(i) = target {
                                        data.variables.per_dof[i].1[atom] = result;
                                    }
                                }
                                sum
                            })
                        }
                    });
                    if is_sum {
                        set_global(target, sum, &mut data.variables, self, context)?;
                    }
                }
                ConstrainPositions => {
                    context.constrain_program_particles(
                        &self.old_positions,
                        false,
                        data.tolerance,
                    )?;
                    context.read_program_particles(|p, _, _| self.old_positions.copy_from_slice(p));
                }
                ConstrainVelocities => context.constrain_program_particles(
                    &self.old_positions,
                    true,
                    data.tolerance,
                )?,
                UpdateContextState => {
                    let mut current = updates;
                    if current.uses_program_step_size {
                        current.step_size = data.variables.step_size;
                    }
                    invalidates = context.update_context_state(current);
                    self.refresh_parameters(context);
                }
                BlockEnd => {
                    if let Some(jump) = self.steps[index].jump {
                        next = jump;
                    }
                }
            }
            if invalidates {
                for cache in &mut self.caches {
                    cache.valid = false;
                }
            }
            index = next;
        }
        self.revision = Some(context.force_state_revision());
        Ok(data.variables.step_size)
    }
}

impl<B: Backend> ContextProgram<B> for Mutex<Data> {
    fn execute(
        &self,
        context: &mut Context<B>,
        updates: ContextUpdateSettings,
    ) -> Result<f64, ConstraintError> {
        let mut data = self.lock().unwrap();
        let mut compiled = match data.compiled.take() {
            Some(compiled) => compiled,
            None => Compiled::new(&data, context.get_parameters())?,
        };
        let result = compiled.execute(&mut data, context, updates);
        data.compiled = Some(compiled);
        result
    }
    fn kinetic_energy(
        &self,
        context: &Context<B>,
        forces: &Tensor<B, 2>,
    ) -> Result<f64, ConstraintError> {
        let mut data = self.lock().unwrap();
        let mut compiled = match data.compiled.take() {
            Some(compiled) => compiled,
            None => Compiled::new(&data, context.get_parameters())?,
        };
        compiled.refresh_parameters(context);
        compiled.kinetic.prepare(&Inputs {
            variables: &data.variables,
            parameters: &compiled.parameters,
            cache: None,
            external_forces: None,
            position: [0.0; 3],
            velocity: [0.0; 3],
            mass: 0.0,
            atom: 0,
            random: [[0.0; 3]; 2],
        });
        let value = crate::tensor::with_f64_data(forces, |forces| {
            context.read_program_particles(|p, v, masses| {
                let mut sum = 0.0;
                for (atom, &mass) in masses.iter().enumerate() {
                    if mass == 0.0 {
                        continue;
                    }
                    let inputs = Inputs {
                        variables: &data.variables,
                        parameters: &compiled.parameters,
                        cache: None,
                        external_forces: Some(forces),
                        position: std::array::from_fn(|a| p[3 * atom + a]),
                        velocity: std::array::from_fn(|a| v[3 * atom + a]),
                        mass,
                        atom,
                        random: [[0.0; 3]; 2],
                    };
                    let value = compiled.kinetic.evaluate(&inputs, false);
                    sum += (value[0] + value[1]) + value[2];
                }
                sum
            })
        });
        data.compiled = Some(compiled);
        Ok(value)
    }
}
