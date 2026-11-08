use super::{failure, ComputationType, Data};
use crate::{
    constraints::ConstraintError,
    expression::{
        EnergyDerivative, Expression, ValueProgram, VectorExpression, VectorValueProgram,
    },
};
use std::collections::HashMap;

pub(super) enum Evaluator {
    Scalar(Box<ValueProgram>),
    Vector(Box<VectorValueProgram>),
}
enum RawEvaluator {
    Scalar(Box<Expression>),
    Vector(Box<VectorExpression>),
}

#[derive(Clone, Copy)]
pub(super) enum Binding {
    Dt,
    Global(usize),
    Parameter(usize),
    PerDof(usize),
    Position,
    Velocity,
    Mass,
    Force,
    Energy,
    Derivative(usize),
    Uniform,
    Gaussian,
}

pub(super) struct CompiledExpression {
    pub evaluator: Evaluator,
    pub constants: Vec<(usize, Binding)>,
    pub varying: Vec<(usize, Binding)>,
    pub random: bool,
}

#[derive(Clone, Copy)]
pub(super) enum Target {
    None,
    Dt,
    Global(usize),
    Parameter(usize),
    Position,
    Velocity,
    PerDof(usize),
}

#[derive(Clone, Copy)]
pub(super) enum Comparison {
    Eq,
    Lt,
    Gt,
    Ne,
    Le,
    Ge,
}
impl Comparison {
    pub fn test(self, left: f64, right: f64) -> bool {
        match self {
            Self::Eq => left == right,
            Self::Lt => left < right,
            Self::Gt => left > right,
            Self::Ne => left != right,
            Self::Le => left <= right,
            Self::Ge => left >= right,
        }
    }
}

pub(super) struct Step {
    pub kind: ComputationType,
    pub target: Target,
    pub expressions: Vec<CompiledExpression>,
    pub comparison: Option<Comparison>,
    pub jump: Option<usize>,
    pub cache: Option<usize>,
    pub invalidates: bool,
}

pub(super) struct ForceCache {
    pub groups: u32,
    pub valid: bool,
    pub forces: Vec<f64>,
    pub energy: f64,
    pub derivatives: Vec<f64>,
    pub needs_derivatives: bool,
}

pub(super) struct Compiled {
    pub steps: Vec<Step>,
    pub revision: Option<u64>,
    pub kinetic: CompiledExpression,
    pub caches: Vec<ForceCache>,
    pub parameter_names: Vec<String>,
    pub parameters: Vec<f64>,
    pub old_positions: Vec<f64>,
}

fn group_variable(name: &str, prefix: &str) -> Option<Option<u8>> {
    if name == prefix {
        return Some(None);
    }
    (0..32).find(|i| name == format!("{prefix}{i}")).map(Some)
}

struct Parsed {
    expression: CompiledExpression,
    group_names: Vec<String>,
    derivatives: bool,
}

fn compile_expression(
    src: &str,
    vector: bool,
    global: bool,
    kinetic: bool,
    data: &Data,
    parameters: &[String],
) -> Result<Parsed, ConstraintError> {
    let convert = |e: &dyn std::fmt::Display| failure(e.to_string());
    let (parsed, scalar_derivatives) =
        Expression::parse_integrator(src).map_err(|e| convert(&e))?;
    let (mut evaluator, derivatives): (RawEvaluator, Vec<EnergyDerivative>) =
        if vector && parsed.has_vector_operations() {
            let (mut expr, derivatives) =
                VectorExpression::parse_integrator(src).map_err(|e| convert(&e))?;
            for (name, function) in &data.tables {
                expr.add_tabulated_function(name, function.clone())
                    .map_err(|e| convert(&e))?;
            }
            expr.validate_functions().map_err(|e| convert(&e))?;
            (RawEvaluator::Vector(Box::new(expr)), derivatives)
        } else {
            let mut expr = parsed;
            let derivatives = scalar_derivatives;
            if !kinetic {
                for (name, function) in &data.tables {
                    expr.add_tabulated_function(name, function.clone())
                        .map_err(|e| convert(&e))?;
                }
            }
            expr.validate_functions().map_err(|e| convert(&e))?;
            (RawEvaluator::Scalar(Box::new(expr)), derivatives)
        };
    let variables = match &mut evaluator {
        RawEvaluator::Scalar(v) => &v.variables,
        RawEvaluator::Vector(v) => &v.variables,
    };
    let mut bindings = Vec::new();
    let mut group_names = Vec::new();
    let mut random = false;
    for name in variables {
        let binding = if let Some(deriv) = derivatives.iter().find(|d| d.variable == *name) {
            if kinetic {
                return Err(failure(
                    "deriv is not available in a kinetic energy expression",
                ));
            }
            group_names.push(deriv.energy.clone());
            Binding::Derivative(
                parameters
                    .iter()
                    .position(|p| p == &deriv.parameter)
                    .ok_or_else(|| {
                        failure("the second deriv argument must be a Context parameter")
                    })?,
            )
        } else if name == "dt" {
            Binding::Dt
        } else if let Some(i) = data.variables.globals.iter().position(|v| v.0 == *name) {
            Binding::Global(i)
        } else if let Some(i) = parameters.iter().position(|v| v == name) {
            Binding::Parameter(i)
        } else if !global && name == "x" {
            Binding::Position
        } else if !global && name == "v" {
            Binding::Velocity
        } else if !global && name == "m" {
            Binding::Mass
        } else if !global && data.variables.per_dof.iter().any(|v| v.0 == *name) {
            Binding::PerDof(
                data.variables
                    .per_dof
                    .iter()
                    .position(|v| v.0 == *name)
                    .unwrap(),
            )
        } else if !global && group_variable(name, "f").is_some() && (!kinetic || name == "f") {
            group_names.push(name.clone());
            Binding::Force
        } else if !kinetic && group_variable(name, "energy").is_some() {
            group_names.push(name.clone());
            Binding::Energy
        } else if !kinetic && name == "uniform" {
            random = true;
            Binding::Uniform
        } else if !kinetic && name == "gaussian" {
            random = true;
            Binding::Gaussian
        } else {
            return Err(failure(format!(
                "unknown variable in integrator expression: {name}"
            )));
        };
        bindings.push(binding);
    }
    group_names.sort();
    group_names.dedup();
    let dynamic: Vec<_> = bindings
        .iter()
        .map(|binding| {
            matches!(
                binding,
                Binding::PerDof(_)
                    | Binding::Position
                    | Binding::Velocity
                    | Binding::Mass
                    | Binding::Force
                    | Binding::Uniform
                    | Binding::Gaussian
            )
        })
        .collect();
    let mut constants = Vec::new();
    let mut varying = Vec::new();
    for (index, binding) in bindings.into_iter().enumerate() {
        if dynamic[index] {
            varying.push((index, binding));
        } else {
            constants.push((index, binding));
        }
    }
    let evaluator = match evaluator {
        RawEvaluator::Scalar(expression) => {
            Evaluator::Scalar(Box::new(ValueProgram::new(*expression, &dynamic)))
        }
        RawEvaluator::Vector(expression) => {
            Evaluator::Vector(Box::new((*expression).value_program(&dynamic)))
        }
    };
    Ok(Parsed {
        expression: CompiledExpression {
            evaluator,
            constants,
            varying,
            random,
        },
        group_names,
        derivatives: !derivatives.is_empty(),
    })
}

fn target(
    name: &str,
    per_dof: bool,
    data: &Data,
    parameters: &[String],
) -> Result<Target, ConstraintError> {
    if per_dof {
        if name == "x" {
            return Ok(Target::Position);
        }
        if name == "v" {
            return Ok(Target::Velocity);
        }
        if let Some(i) = data.variables.per_dof.iter().position(|v| v.0 == name) {
            return Ok(Target::PerDof(i));
        }
    } else {
        if name == "dt" {
            return Ok(Target::Dt);
        }
        if let Some(i) = data.variables.globals.iter().position(|v| v.0 == name) {
            return Ok(Target::Global(i));
        }
        if let Some(i) = parameters.iter().position(|v| v == name) {
            return Ok(Target::Parameter(i));
        }
    }
    Err(failure(format!(
        "illegal integrator output variable: {name}"
    )))
}

impl Compiled {
    pub(super) fn new(
        data: &Data,
        parameters: &HashMap<String, f64>,
    ) -> Result<Self, ConstraintError> {
        let mut parameter_names: Vec<_> = parameters.keys().cloned().collect();
        parameter_names.sort();
        let mut caches: Vec<ForceCache> = Vec::new();
        let mut steps = Vec::new();
        let mut block_stack = Vec::new();
        for definition in &data.computations {
            use ComputationType::*;
            let mut step = Step {
                kind: definition.kind,
                target: Target::None,
                expressions: Vec::new(),
                comparison: None,
                jump: None,
                cache: None,
                invalidates: false,
            };
            let mut expressions = Vec::new();
            match definition.kind {
                ComputeGlobal | ComputePerDof | ComputeSum => {
                    step.target = target(
                        &definition.variable,
                        definition.kind == ComputePerDof,
                        data,
                        &parameter_names,
                    )?;
                    expressions.push(definition.expression.as_str());
                    step.invalidates =
                        matches!(step.target, Target::Position | Target::Parameter(_));
                }
                IfBlockStart | WhileBlockStart => {
                    let (offset, op, comparison) = [
                        (">=", Comparison::Ge),
                        ("<=", Comparison::Le),
                        ("!=", Comparison::Ne),
                        (">", Comparison::Gt),
                        ("<", Comparison::Lt),
                        ("=", Comparison::Eq),
                    ]
                    .into_iter()
                    .find_map(|(op, c)| definition.expression.find(op).map(|i| (i, op, c)))
                    .ok_or_else(|| failure("no comparison operator found in condition"))?;
                    expressions.push(&definition.expression[..offset]);
                    expressions.push(&definition.expression[offset + op.len()..]);
                    step.comparison = Some(comparison);
                    block_stack.push(steps.len());
                }
                BlockEnd => {
                    let start: usize = block_stack
                        .pop()
                        .ok_or_else(|| failure("unexpected end of block"))?;
                    let end = steps.len();
                    let first: &mut Step = &mut steps[start];
                    first.jump = Some(end + 1);
                    if first.kind == WhileBlockStart {
                        step.jump = Some(start);
                    }
                }
                ConstrainPositions => step.invalidates = true,
                UpdateContextState => step.invalidates = true,
                ConstrainVelocities => (),
            }
            let mut group: Option<Option<u8>> = None;
            let mut needs_derivatives = false;
            for expression in expressions {
                let parsed = compile_expression(
                    expression,
                    matches!(definition.kind, ComputePerDof | ComputeSum),
                    !matches!(definition.kind, ComputePerDof | ComputeSum),
                    false,
                    data,
                    &parameter_names,
                )?;
                for name in parsed.group_names {
                    let selection = group_variable(&name, "f")
                        .or_else(|| group_variable(&name, "energy"))
                        .unwrap();
                    if group.is_some() && (group != Some(None) || selection.is_some()) {
                        return Err(failure(
                            "a single computation cannot depend on multiple force groups",
                        ));
                    }
                    group = Some(selection);
                }
                needs_derivatives |= parsed.derivatives;
                step.expressions.push(parsed.expression);
            }
            if let Some(group) = group {
                let groups = group.map_or(data.groups, |g| 1u32 << g);
                let index = if let Some(i) = caches.iter().position(|v| v.groups == groups) {
                    i
                } else {
                    caches.push(ForceCache {
                        groups,
                        valid: false,
                        forces: Vec::new(),
                        energy: 0.0,
                        derivatives: vec![0.0; parameter_names.len()],
                        needs_derivatives: false,
                    });
                    caches.len() - 1
                };
                caches[index].needs_derivatives |= needs_derivatives;
                step.cache = Some(index);
            }
            steps.push(step);
        }
        if !block_stack.is_empty() {
            return Err(failure("missing EndBlock"));
        }
        let kinetic =
            compile_expression(&data.kinetic, false, false, true, data, &parameter_names)?
                .expression;
        Ok(Self {
            steps,
            revision: None,
            kinetic,
            caches,
            parameters: vec![0.0; parameter_names.len()],
            parameter_names,
            old_positions: Vec::new(),
        })
    }
}
