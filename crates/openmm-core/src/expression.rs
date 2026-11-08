//! Compiled scalar expressions with reusable reverse-mode differentiation.

use crate::tabulated::TabulatedFunction;
use std::collections::HashMap;
use std::fmt;
mod geometry;
use geometry::GeometryFunction;

#[derive(Debug, Clone, Copy)]
enum FunctionBinding {
    Table(usize),
    Geometry(GeometryFunction),
}

#[derive(Clone, Copy, Default)]
enum GeometryMode {
    #[default]
    None,
    External,
    Particles(char, usize),
}

impl GeometryMode {
    fn function(self, name: &str) -> Option<GeometryFunction> {
        match self {
            Self::External if name == "periodicdistance" => Some(GeometryFunction::Distance),
            Self::Particles(_, _) if name != "periodicdistance" => {
                GeometryFunction::from_name(name)
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Expression {
    instructions: Vec<Instruction>,
    pub variables: Vec<String>,
    function_names: Vec<String>,
    function_calls: Vec<FunctionCall>,
    function_bindings: Vec<Option<FunctionBinding>>,
    tables: Vec<(String, TabulatedFunction)>,
}

#[derive(Debug, Clone)]
pub struct Workspace {
    inputs: Vec<f64>,
    values: Vec<f64>,
    adjoints: Vec<f64>,
    gradients: Vec<f64>,
    differentiable: Vec<bool>,
    function_derivatives: Vec<f64>,
    periodic_box: Option<[[f64; 3]; 3]>,
}

impl Workspace {
    pub fn set_periodic_box(&mut self, vectors: Option<&[[f64; 3]; 3]>) {
        self.periodic_box = vectors.copied();
    }
    #[inline]
    pub fn set(&mut self, index: Option<usize>, value: f64) {
        if let Some(index) = index {
            self.inputs[index] = value;
        }
    }

    #[inline]
    pub fn derivative(&self, index: Option<usize>) -> f64 {
        index.map_or(0.0, |index| self.gradients[index])
    }
}

impl Expression {
    pub fn parse(src: &str) -> Result<Self, ParseError> {
        Self::parse_impl(src, false, GeometryMode::None)
    }

    pub fn parse_unbound(src: &str) -> Result<Self, ParseError> {
        Self::parse_impl(src, true, GeometryMode::None)
    }

    pub fn parse_geometric(
        src: &str,
        particles: Option<(char, usize)>,
    ) -> Result<Self, ParseError> {
        let mode = particles.map_or(GeometryMode::External, |(prefix, count)| {
            GeometryMode::Particles(prefix, count)
        });
        Self::parse_impl(src, particles.is_some(), mode)
    }

    fn parse_impl(
        src: &str,
        allow_unbound: bool,
        geometry: GeometryMode,
    ) -> Result<Self, ParseError> {
        let parts: Vec<_> = src.split(';').collect();
        let mut definitions = HashMap::new();
        for definition in parts[1..].iter().rev().filter(|s| !s.trim().is_empty()) {
            let (name, value) = definition
                .split_once('=')
                .ok_or_else(|| ParseError::new("subexpression does not specify a name"))?;
            let name = name.trim();
            if name.is_empty() {
                return Err(ParseError::new("subexpression does not specify a name"));
            }
            definitions.insert(
                name.to_owned(),
                parse_tree(value, &definitions, allow_unbound, geometry)?,
            );
        }
        let root = optimize(parse_tree(parts[0], &definitions, allow_unbound, geometry)?);
        let mut variables = Vec::new();
        collect_vars(&root, &mut variables);
        variables.sort();
        variables.dedup();
        let mut instructions = Vec::new();
        let mut function_names = Vec::new();
        let mut function_calls = Vec::new();
        compile(
            &root,
            &variables,
            &mut function_names,
            &mut function_calls,
            &mut instructions,
            &mut HashMap::new(),
        );
        let mut offset = 0;
        for call in &mut function_calls {
            call.derivative_offset = offset;
            offset += call.arguments.len();
        }
        Ok(Self {
            instructions,
            variables,
            function_bindings: function_names
                .iter()
                .map(|name| geometry.function(name).map(FunctionBinding::Geometry))
                .collect(),
            function_names,
            function_calls,
            tables: Vec::new(),
        })
    }

    pub fn add_tabulated_function(
        &mut self,
        name: &str,
        function: impl Into<TabulatedFunction>,
    ) -> Result<usize, EvalError> {
        let function = function.into();
        if self.tables.iter().any(|(existing, _)| existing == name) {
            return Err(EvalError(format!("duplicate tabulated function: {name}")));
        }
        if Op::function(name).is_ok() {
            return Err(EvalError(format!("reserved function name: {name}")));
        }
        if let Some(binding) = self
            .function_names
            .iter()
            .position(|existing| existing == name)
        {
            for call in &self.function_calls {
                if call.binding == binding && call.arguments.len() != function.arity() {
                    return Err(EvalError(format!(
                        "{name} requires {} arguments",
                        function.arity()
                    )));
                }
            }
            self.function_bindings[binding] = Some(FunctionBinding::Table(self.tables.len()));
        }
        self.tables.push((name.to_owned(), function));
        Ok(self.tables.len() - 1)
    }

    pub fn num_tabulated_functions(&self) -> usize {
        self.tables.len()
    }
    pub fn tabulated_function_name(&self, index: usize) -> &str {
        &self.tables[index].0
    }
    pub fn tabulated_function(&self, index: usize) -> &TabulatedFunction {
        &self.tables[index].1
    }
    pub fn set_tabulated_function(
        &mut self,
        index: usize,
        function: impl Into<TabulatedFunction>,
    ) -> Result<(), EvalError> {
        let function = function.into();
        if std::mem::discriminant(&self.tables[index].1) != std::mem::discriminant(&function) {
            return Err(EvalError("tabulated function type cannot change".into()));
        }
        self.tables[index].1 = function;
        Ok(())
    }
    pub fn update_tabulated_functions_from(&mut self, source: &Self) {
        assert_eq!(
            self.tables.len(),
            source.tables.len(),
            "tabulated function count must match"
        );
        for index in 0..self.tables.len() {
            assert_eq!(
                self.tables[index].0, source.tables[index].0,
                "tabulated function names must match"
            );
            self.set_tabulated_function(index, source.tables[index].1.clone())
                .unwrap();
        }
    }
    pub fn validate_functions(&self) -> Result<(), EvalError> {
        for (name, binding) in self.function_names.iter().zip(&self.function_bindings) {
            if binding.is_none() {
                return Err(EvalError(format!("undefined function: {name}")));
            }
        }
        Ok(())
    }

    pub fn variable_index(&self, name: &str) -> Option<usize> {
        self.variables
            .binary_search_by(|v| v.as_str().cmp(name))
            .ok()
    }

    pub fn validate_variables<'a>(
        &self,
        names: impl IntoIterator<Item = &'a str>,
    ) -> Result<(), EvalError> {
        self.validate_functions()?;
        let names: Vec<_> = names.into_iter().collect();
        for name in &self.variables {
            if !names.contains(&name.as_str()) {
                return Err(EvalError(format!("undefined variable: {name}")));
            }
        }
        Ok(())
    }

    pub fn workspace(&self) -> Workspace {
        self.workspace_with_derivatives(0..self.variables.len())
    }

    pub fn workspace_with_derivatives(&self, inputs: impl IntoIterator<Item = usize>) -> Workspace {
        let mut active_inputs = vec![false; self.variables.len()];
        for input in inputs {
            active_inputs[input] = true;
        }
        let mut differentiable = vec![false; self.instructions.len()];
        for (index, instruction) in self.instructions.iter().enumerate() {
            differentiable[index] = match instruction {
                Instruction::Function(call) => self.function_calls[*call]
                    .arguments
                    .iter()
                    .any(|&arg| differentiable[arg]),
                Instruction::Literal(_) => false,
                Instruction::Variable(input) => active_inputs[*input],
                Instruction::Operation(op, args) => match op {
                    Op::Step
                    | Op::Delta
                    | Op::Floor
                    | Op::Ceil
                    | Op::Eq
                    | Op::Ne
                    | Op::Lt
                    | Op::Le
                    | Op::Gt
                    | Op::Ge => false,
                    Op::Select => differentiable[args[1]] || differentiable[args[2]],
                    _ => args[..op.arity()].iter().any(|&arg| differentiable[arg]),
                },
            };
        }
        Workspace {
            inputs: vec![0.0; self.variables.len()],
            values: vec![0.0; self.instructions.len()],
            adjoints: vec![0.0; self.instructions.len()],
            gradients: vec![0.0; self.variables.len()],
            differentiable,
            function_derivatives: vec![
                0.0;
                self.function_calls
                    .iter()
                    .map(|call| call.arguments.len())
                    .sum()
            ],
            periodic_box: None,
        }
    }

    pub fn compute(&self, vars: &HashMap<&str, f64>) -> Result<f64, EvalError> {
        self.validate_functions()?;
        let mut workspace = self.workspace_with_derivatives([]);
        for (index, name) in self.variables.iter().enumerate() {
            workspace.inputs[index] = *vars
                .get(name.as_str())
                .ok_or_else(|| EvalError(format!("undefined variable: {name}")))?;
        }
        Ok(self.evaluate(&mut workspace))
    }

    /// Evaluate energy and the requested input derivatives without allocating.
    pub fn evaluate(&self, workspace: &mut Workspace) -> f64 {
        assert_eq!(workspace.values.len(), self.instructions.len());
        for (index, instruction) in self.instructions.iter().enumerate() {
            workspace.values[index] = match *instruction {
                Instruction::Function(call) => {
                    let function_call = &self.function_calls[call];
                    let mut arguments = [0.0; 12];
                    for (i, &arg) in function_call.arguments.iter().enumerate() {
                        arguments[i] = workspace.values[arg];
                    }
                    let offset = function_call.derivative_offset;
                    let count = function_call.arguments.len();
                    let derivatives = &mut workspace.function_derivatives[offset..offset + count];
                    match self.function_bindings[function_call.binding].expect("unbound function") {
                        FunctionBinding::Table(index) => {
                            let (value, gradient) = self.tables[index].1.evaluate([
                                arguments[0],
                                arguments[1],
                                arguments[2],
                            ]);
                            derivatives.copy_from_slice(&gradient[..count]);
                            value
                        }
                        FunctionBinding::Geometry(function) => {
                            let (value, gradient) = function
                                .evaluate(&arguments[..count], workspace.periodic_box.as_ref());
                            derivatives.copy_from_slice(&gradient[..count]);
                            value
                        }
                    }
                }
                Instruction::Literal(bits) => f64::from_bits(bits),
                Instruction::Variable(input) => workspace.inputs[input],
                Instruction::Operation(op, args) => op.value(args.map(|a| workspace.values[a])),
            };
        }
        workspace.adjoints.fill(0.0);
        workspace.gradients.fill(0.0);
        let last = self.instructions.len() - 1;
        workspace.adjoints[last] = 1.0;
        for index in (0..self.instructions.len()).rev() {
            let adjoint = workspace.adjoints[index];
            if adjoint == 0.0 || !workspace.differentiable[index] {
                continue;
            }
            match self.instructions[index] {
                Instruction::Variable(input) => workspace.gradients[input] += adjoint,
                Instruction::Operation(op, args) => {
                    let values = args.map(|a| workspace.values[a]);
                    let active = args.map(|arg| workspace.differentiable[arg]);
                    let derivatives = op.derivatives(values, workspace.values[index], active);
                    for i in 0..op.arity() {
                        if active[i] && derivatives[i] != 0.0 {
                            workspace.adjoints[args[i]] += adjoint * derivatives[i];
                        }
                    }
                }
                Instruction::Function(call) => {
                    for (i, &arg) in self.function_calls[call].arguments.iter().enumerate() {
                        let derivative = workspace.function_derivatives
                            [self.function_calls[call].derivative_offset + i];
                        if workspace.differentiable[arg] && derivative != 0.0 {
                            workspace.adjoints[arg] += adjoint * derivative;
                        }
                    }
                }
                Instruction::Literal(_) => {}
            }
        }
        workspace.values[last]
    }
}

#[derive(Debug, Clone)]
pub struct ParseError(String);
impl ParseError {
    fn new(message: &str) -> Self {
        Self(message.to_owned())
    }
}
impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parse error: {}", self.0)
    }
}
impl std::error::Error for ParseError {}

#[derive(Debug, Clone)]
pub struct EvalError(pub String);
impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "eval error: {}", self.0)
    }
}
impl std::error::Error for EvalError {}

#[derive(Debug, Clone)]
enum Expr {
    Literal(f64),
    Variable(String),
    Operation(Op, Vec<Expr>),
    Function(String, Vec<Expr>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Instruction {
    Literal(u64),
    Variable(usize),
    Operation(Op, [usize; 3]),
    Function(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FunctionCall {
    binding: usize,
    arguments: Vec<usize>,
    derivative_offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Neg,
    Sqrt,
    Exp,
    Log,
    Log10,
    Sin,
    Cos,
    Sec,
    Csc,
    Tan,
    Cot,
    Asin,
    Acos,
    Atan,
    Atan2,
    Sinh,
    Cosh,
    Tanh,
    Erf,
    Erfc,
    Step,
    Delta,
    Square,
    Cube,
    Recip,
    Min,
    Max,
    Abs,
    Floor,
    Ceil,
    Select,
    Eq,
    Ne,
    Le,
    Ge,
    Lt,
    Gt,
}

impl Op {
    fn function(name: &str) -> Result<Self, ParseError> {
        Ok(match name {
            "sqrt" => Self::Sqrt,
            "exp" => Self::Exp,
            "log" => Self::Log,
            "log10" => Self::Log10,
            "sin" => Self::Sin,
            "cos" => Self::Cos,
            "sec" => Self::Sec,
            "csc" => Self::Csc,
            "tan" => Self::Tan,
            "cot" => Self::Cot,
            "asin" => Self::Asin,
            "acos" => Self::Acos,
            "atan" => Self::Atan,
            "atan2" => Self::Atan2,
            "sinh" => Self::Sinh,
            "cosh" => Self::Cosh,
            "tanh" => Self::Tanh,
            "erf" => Self::Erf,
            "erfc" => Self::Erfc,
            "step" => Self::Step,
            "delta" => Self::Delta,
            "square" => Self::Square,
            "cube" => Self::Cube,
            "recip" => Self::Recip,
            "min" => Self::Min,
            "max" => Self::Max,
            "abs" => Self::Abs,
            "floor" => Self::Floor,
            "ceil" => Self::Ceil,
            "select" => Self::Select,
            _ => return Err(ParseError::new(&format!("unknown function: {name}"))),
        })
    }

    fn arity(self) -> usize {
        match self {
            Self::Select => 3,
            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Div
            | Self::Pow
            | Self::Atan2
            | Self::Min
            | Self::Max
            | Self::Eq
            | Self::Ne
            | Self::Le
            | Self::Ge
            | Self::Lt
            | Self::Gt => 2,
            _ => 1,
        }
    }

    fn value(self, [a, b, c]: [f64; 3]) -> f64 {
        match self {
            Self::Add => a + b,
            Self::Sub => a - b,
            Self::Mul => a * b,
            Self::Div => a / b,
            Self::Pow => a.powf(b),
            Self::Neg => -a,
            Self::Sqrt => a.sqrt(),
            Self::Exp => a.exp(),
            Self::Log => a.ln(),
            Self::Log10 => a.log10(),
            Self::Sin => a.sin(),
            Self::Cos => a.cos(),
            Self::Sec => 1.0 / a.cos(),
            Self::Csc => 1.0 / a.sin(),
            Self::Tan => a.tan(),
            Self::Cot => 1.0 / a.tan(),
            Self::Asin => a.asin(),
            Self::Acos => a.acos(),
            Self::Atan => a.atan(),
            Self::Atan2 => a.atan2(b),
            Self::Sinh => a.sinh(),
            Self::Cosh => a.cosh(),
            Self::Tanh => a.tanh(),
            Self::Erf => libm::erf(a),
            Self::Erfc => libm::erfc(a),
            Self::Step => f64::from(a >= 0.0),
            Self::Delta => f64::from(a == 0.0),
            Self::Square => a * a,
            Self::Cube => a * a * a,
            Self::Recip => 1.0 / a,
            Self::Min => {
                if b < a {
                    b
                } else {
                    a
                }
            }
            Self::Max => {
                if a < b {
                    b
                } else {
                    a
                }
            }
            Self::Abs => a.abs(),
            Self::Floor => a.floor(),
            Self::Ceil => a.ceil(),
            Self::Select => {
                if a != 0.0 {
                    b
                } else {
                    c
                }
            }
            Self::Eq => f64::from(a - b == 0.0),
            Self::Ne => 1.0 - f64::from(a - b == 0.0),
            Self::Le => f64::from(b - a >= 0.0),
            Self::Ge => f64::from(a - b >= 0.0),
            Self::Lt => 1.0 - f64::from(a - b >= 0.0),
            Self::Gt => 1.0 - f64::from(b - a >= 0.0),
        }
    }

    fn derivatives(self, [a, b, _]: [f64; 3], value: f64, active: [bool; 3]) -> [f64; 3] {
        let mut result = [0.0; 3];
        result[0] = match self {
            Self::Add => {
                result[1] = 1.0;
                1.0
            }
            Self::Sub => {
                result[1] = -1.0;
                1.0
            }
            Self::Mul => {
                result[1] = a;
                b
            }
            Self::Div => {
                if active[1] {
                    result[1] = -value / b;
                }
                if active[0] {
                    1.0 / b
                } else {
                    0.0
                }
            }
            Self::Pow => {
                if active[1] {
                    result[1] = value * a.ln();
                }
                if b == 0.0 || !active[0] {
                    0.0
                } else {
                    b * a.powf(b - 1.0)
                }
            }
            Self::Neg => -1.0,
            Self::Sqrt => 0.5 / value,
            Self::Exp => value,
            Self::Log => 1.0 / a,
            Self::Log10 => 1.0 / (a * std::f64::consts::LN_10),
            Self::Sin => a.cos(),
            Self::Cos => -a.sin(),
            Self::Sec => value * a.tan(),
            Self::Csc => -value / a.tan(),
            Self::Tan => 1.0 + value * value,
            Self::Cot => -(1.0 + value * value),
            Self::Asin => 1.0 / (1.0 - a * a).sqrt(),
            Self::Acos => -1.0 / (1.0 - a * a).sqrt(),
            Self::Atan => 1.0 / (1.0 + a * a),
            Self::Atan2 => {
                result[1] = -a / (a * a + b * b);
                b / (a * a + b * b)
            }
            Self::Sinh => a.cosh(),
            Self::Cosh => a.sinh(),
            Self::Tanh => 1.0 - value * value,
            Self::Erf => 2.0 / std::f64::consts::PI.sqrt() * (-a * a).exp(),
            Self::Erfc => -2.0 / std::f64::consts::PI.sqrt() * (-a * a).exp(),
            Self::Square => 2.0 * a,
            Self::Cube => 3.0 * a * a,
            Self::Recip => -value * value,
            Self::Min => {
                if a >= b {
                    result[1] = 1.0;
                    0.0
                } else {
                    1.0
                }
            }
            Self::Max => {
                if a >= b {
                    1.0
                } else {
                    result[1] = 1.0;
                    0.0
                }
            }
            Self::Abs => {
                if a >= 0.0 {
                    1.0
                } else {
                    -1.0
                }
            }
            Self::Select => {
                result[if a != 0.0 { 1 } else { 2 }] = 1.0;
                0.0
            }
            _ => 0.0,
        };
        result
    }
}

fn optimize(expr: Expr) -> Expr {
    if let Expr::Function(name, args) = expr {
        return Expr::Function(name, args.into_iter().map(optimize).collect());
    }
    let Expr::Operation(op, args) = expr else {
        return expr;
    };
    let args: Vec<_> = args.into_iter().map(optimize).collect();
    if args.iter().all(|arg| matches!(arg, Expr::Literal(_))) {
        let mut values = [0.0; 3];
        for (index, arg) in args.iter().enumerate() {
            if let Expr::Literal(value) = arg {
                values[index] = *value;
            }
        }
        return Expr::Literal(op.value(values));
    }
    if op == Op::Div {
        if let Expr::Literal(denominator) = args[1] {
            let inverse = denominator.recip();
            if inverse.is_normal() {
                return Expr::Operation(Op::Mul, vec![args[0].clone(), Expr::Literal(inverse)]);
            }
        }
    }
    if op == Op::Pow {
        if let Expr::Literal(power) = args[1] {
            if power.fract() == 0.0 && (0.0..=16.0).contains(&power) {
                return integer_power(&args[0], power as usize);
            }
        }
    }
    Expr::Operation(op, args)
}

fn integer_power(base: &Expr, power: usize) -> Expr {
    match power {
        0 => Expr::Literal(1.0),
        1 => base.clone(),
        _ => {
            let half = integer_power(base, power / 2);
            let square = Expr::Operation(Op::Mul, vec![half.clone(), half]);
            if power.is_multiple_of(2) {
                square
            } else {
                Expr::Operation(Op::Mul, vec![square, base.clone()])
            }
        }
    }
}

fn collect_vars(expr: &Expr, variables: &mut Vec<String>) {
    match expr {
        Expr::Variable(name) => variables.push(name.clone()),
        Expr::Operation(_, args) | Expr::Function(_, args) => {
            args.iter().for_each(|arg| collect_vars(arg, variables))
        }
        _ => {}
    }
}

fn compile(
    expr: &Expr,
    variables: &[String],
    function_names: &mut Vec<String>,
    function_calls: &mut Vec<FunctionCall>,
    instructions: &mut Vec<Instruction>,
    cache: &mut HashMap<Instruction, usize>,
) -> usize {
    let instruction = match expr {
        Expr::Literal(value) => Instruction::Literal(value.to_bits()),
        Expr::Variable(name) => Instruction::Variable(variables.binary_search(name).unwrap()),
        Expr::Function(name, args) => {
            let binding = match function_names.iter().position(|existing| existing == name) {
                Some(index) => index,
                None => {
                    function_names.push(name.clone());
                    function_names.len() - 1
                }
            };
            let arguments = args
                .iter()
                .map(|arg| {
                    compile(
                        arg,
                        variables,
                        function_names,
                        function_calls,
                        instructions,
                        cache,
                    )
                })
                .collect();
            let call = FunctionCall {
                binding,
                arguments,
                derivative_offset: 0,
            };
            let index = function_calls
                .iter()
                .position(|existing| existing == &call)
                .unwrap_or_else(|| {
                    function_calls.push(call);
                    function_calls.len() - 1
                });
            Instruction::Function(index)
        }

        Expr::Operation(op, args) => {
            let mut indexes = [0; 3];
            for (index, arg) in args.iter().enumerate() {
                indexes[index] = compile(
                    arg,
                    variables,
                    function_names,
                    function_calls,
                    instructions,
                    cache,
                );
            }
            Instruction::Operation(*op, indexes)
        }
    };
    if let Some(index) = cache.get(&instruction) {
        return *index;
    }
    let index = instructions.len();
    instructions.push(instruction);
    cache.insert(instruction, index);
    index
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Name(String),
    Operator(String),
    Left,
    Right,
    Comma,
}

fn tokenize(src: &str) -> Result<Vec<Token>, ParseError> {
    let mut chars = src.chars().peekable();
    let mut tokens = Vec::new();
    while let Some(c) = chars.next() {
        match c {
            c if c.is_whitespace() => {}
            '(' => tokens.push(Token::Left),
            ')' => tokens.push(Token::Right),
            ',' => tokens.push(Token::Comma),
            '+' | '-' | '*' | '/' | '^' | '=' | '!' | '<' | '>' => {
                let mut op = c.to_string();
                if matches!(c, '=' | '!' | '<' | '>') && chars.peek() == Some(&'=') {
                    op.push(chars.next().unwrap());
                }
                if op == "=" || op == "!" {
                    return Err(ParseError::new("incomplete comparison operator"));
                }
                tokens.push(Token::Operator(op));
            }
            c if c.is_ascii_digit() || c == '.' => {
                let mut number = c.to_string();
                while let Some(&next) = chars.peek() {
                    if next.is_ascii_digit()
                        || matches!(next, '.' | 'e' | 'E')
                        || (matches!(next, '+' | '-') && number.ends_with(['e', 'E']))
                    {
                        number.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }
                tokens.push(Token::Number(
                    number
                        .parse()
                        .map_err(|_| ParseError::new(&format!("bad number: {number}")))?,
                ));
            }
            c => {
                let mut name = c.to_string();
                while let Some(&next) = chars.peek() {
                    if next.is_whitespace() || "+-*/^=!<>() ,".contains(next) {
                        break;
                    }
                    name.push(chars.next().unwrap());
                }
                tokens.push(Token::Name(name));
            }
        }
    }
    Ok(tokens)
}

fn parse_tree(
    src: &str,
    definitions: &HashMap<String, Expr>,
    allow_unbound: bool,
    geometry: GeometryMode,
) -> Result<Expr, ParseError> {
    let mut parser = Parser {
        tokens: tokenize(src)?,
        position: 0,
        definitions,
        allow_unbound,
        geometry,
    };
    let result = parser.expression(0)?;
    if parser.position != parser.tokens.len() {
        return Err(ParseError::new("unexpected token after expression"));
    }
    Ok(result)
}

struct Parser<'a> {
    tokens: Vec<Token>,
    position: usize,
    definitions: &'a HashMap<String, Expr>,
    allow_unbound: bool,
    geometry: GeometryMode,
}
impl Parser<'_> {
    fn expression(&mut self, precedence: u8) -> Result<Expr, ParseError> {
        let token = self
            .tokens
            .get(self.position)
            .cloned()
            .ok_or_else(|| ParseError::new("unexpected end of expression"))?;
        self.position += 1;
        let mut result = match token {
            Token::Number(value) => Expr::Literal(value),
            Token::Operator(op) if op == "-" => Expr::Operation(Op::Neg, vec![self.expression(2)?]),
            Token::Left => {
                let result = self.expression(0)?;
                self.expect(Token::Right)?;
                result
            }
            Token::Name(name) => {
                if self.tokens.get(self.position) == Some(&Token::Left) {
                    self.position += 1;
                    let op = Op::function(&name);
                    let mut args = Vec::new();
                    if self.tokens.get(self.position) != Some(&Token::Right) {
                        loop {
                            args.push(self.expression(0)?);
                            if self.tokens.get(self.position) != Some(&Token::Comma) {
                                break;
                            }
                            self.position += 1;
                        }
                    }
                    self.expect(Token::Right)?;
                    match op {
                        Ok(op) => {
                            if args.len() != op.arity() {
                                return Err(ParseError::new(&format!(
                                    "{name} requires {} arguments",
                                    op.arity()
                                )));
                            }
                            Expr::Operation(op, args)
                        }
                        Err(error) => {
                            if let Some(function) = self.geometry.function(&name) {
                                if args.len() != function.arity() {
                                    return Err(ParseError::new(&format!(
                                        "{name} requires {} arguments",
                                        function.arity()
                                    )));
                                }
                                Expr::Function(name, args)
                            } else if let GeometryMode::Particles(prefix, count) = self.geometry {
                                if matches!(name.as_str(), "distance" | "angle" | "dihedral") {
                                    let expected = match name.as_str() {
                                        "distance" => 2,
                                        "angle" => 3,
                                        _ => 4,
                                    };
                                    if args.len() != expected {
                                        return Err(ParseError::new(&format!(
                                            "{name} requires {expected} particles"
                                        )));
                                    }
                                    let mut coordinates = Vec::with_capacity(3 * expected);
                                    for arg in args {
                                        let Expr::Variable(symbol) = arg else {
                                            return Err(ParseError::new(
                                                "geometric arguments must be particle identifiers",
                                            ));
                                        };
                                        let index = symbol
                                            .strip_prefix(prefix)
                                            .and_then(|s| s.parse::<usize>().ok())
                                            .filter(|&index| {
                                                index >= 1
                                                    && index <= count
                                                    && symbol == format!("{prefix}{index}")
                                            })
                                            .ok_or_else(|| {
                                                ParseError::new(&format!(
                                                    "unknown particle: {symbol}"
                                                ))
                                            })?;
                                        coordinates.extend(
                                            ['x', 'y', 'z'].map(|axis| {
                                                Expr::Variable(format!("{axis}{index}"))
                                            }),
                                        );
                                    }
                                    Expr::Function(format!("point{name}"), coordinates)
                                } else {
                                    if !(1..=3).contains(&args.len()) {
                                        return Err(ParseError::new(
                                            "tabulated functions require 1-3 arguments",
                                        ));
                                    }
                                    Expr::Function(name, args)
                                }
                            } else {
                                if !self.allow_unbound {
                                    return Err(error);
                                }
                                if !(1..=3).contains(&args.len()) {
                                    return Err(ParseError::new(
                                        "tabulated functions require 1-3 arguments",
                                    ));
                                }
                                Expr::Function(name, args)
                            }
                        }
                    }
                } else {
                    self.definitions
                        .get(&name)
                        .cloned()
                        .unwrap_or(Expr::Variable(name))
                }
            }
            _ => return Err(ParseError::new("unexpected token")),
        };
        while let Some(Token::Operator(operator)) = self.tokens.get(self.position) {
            let (op, level) = match operator.as_str() {
                "+" => (Op::Add, 0),
                "-" => (Op::Sub, 0),
                "*" => (Op::Mul, 1),
                "/" => (Op::Div, 1),
                "^" => (Op::Pow, 3),
                "==" => (Op::Eq, 4),
                "!=" => (Op::Ne, 4),
                "<=" => (Op::Le, 4),
                ">=" => (Op::Ge, 4),
                "<" => (Op::Lt, 4),
                ">" => (Op::Gt, 4),
                _ => unreachable!(),
            };
            if level < precedence {
                break;
            }
            self.position += 1;
            let rhs = self.expression(level + u8::from(op != Op::Pow))?;
            result = Expr::Operation(op, vec![result, rhs]);
            // Lepton comparisons return immediately from their precedence level.
            if level == 4 {
                return Ok(result);
            }
        }
        Ok(result)
    }

    fn expect(&mut self, token: Token) -> Result<(), ParseError> {
        if self.tokens.get(self.position) != Some(&token) {
            return Err(ParseError::new(&format!("expected {token:?}")));
        }
        self.position += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluate(src: &str, bindings: &[(&str, f64)]) -> (f64, Vec<f64>) {
        let expression = Expression::parse(src).unwrap();
        let mut workspace = expression.workspace();
        for &(name, value) in bindings {
            workspace.set(expression.variable_index(name), value);
        }
        let value = expression.evaluate(&mut workspace);
        (
            value,
            bindings
                .iter()
                .map(|(name, _)| workspace.derivative(expression.variable_index(name)))
                .collect(),
        )
    }

    #[test]
    fn bound_tables_propagate_chain_rule_and_reject_invalid_calls() {
        use crate::tabulated::Continuous1DFunction;
        let table = Continuous1DFunction::new([2], vec![1.0, 5.0], [[0.0, 2.0]], false).unwrap();
        let mut expression = Expression::parse_unbound("g*f(x^2)+f(f(x))").unwrap();
        assert!(expression.validate_variables(["g", "x"]).is_err());
        expression
            .add_tabulated_function("f", table.clone())
            .unwrap();
        assert!(expression
            .add_tabulated_function("f", table.clone())
            .is_err());
        expression.validate_variables(["g", "x"]).unwrap();
        let mut workspace = expression.workspace();
        workspace.set(expression.variable_index("x"), 0.3);
        workspace.set(expression.variable_index("g"), 1.7);
        assert!((expression.evaluate(&mut workspace) - 6.206).abs() < 1e-14);
        assert!((workspace.derivative(expression.variable_index("x")) - 6.04).abs() < 1e-14);
        assert!((workspace.derivative(expression.variable_index("g")) - 1.18).abs() < 1e-14);
        let mut invalid = Expression::parse_unbound("f(x,y)").unwrap();
        assert!(invalid.add_tabulated_function("f", table).is_err());
    }

    #[test]
    fn precedence_and_definitions() {
        for (source, expected) in [
            ("2^3^2", 512.0),
            ("-2^2", -4.0),
            ("10-4*2", 2.0),
            ("2+3<4", 3.0),
            ("1e2 + .25", 100.25),
        ] {
            assert_eq!(evaluate(source, &[]).0, expected, "{source}");
        }
        assert_eq!(
            evaluate("a+a; a=b*b; b=x+1;", &[("x", 2.0)]),
            (18.0, vec![12.0])
        );
        assert_eq!(
            Expression::parse("a; b=x; a=b").unwrap().variables,
            vec!["b"]
        );
        assert!(Expression::parse("1<2<3").is_err());
    }

    #[test]
    fn errors_are_explicit() {
        for source in [
            "",
            "foo(1)",
            "sin(1,2)",
            "min(1)",
            "select(1,2)",
            "sin(1 2)",
            "sqrt(",
            "a; bad",
        ] {
            assert!(Expression::parse(source).is_err(), "{source}");
        }
        let expression = Expression::parse("x+y").unwrap();
        assert!(expression.compute(&HashMap::from([("x", 1.0)])).is_err());
        assert!(expression.validate_variables(["x"]).is_err());
    }

    #[test]
    fn smooth_function_gradients() {
        for source in [
            "sin(x)",
            "cos(x)",
            "tan(x)",
            "sec(x)",
            "csc(x)",
            "cot(x)",
            "asin(x)",
            "acos(x)",
            "atan(x)",
            "atan2(x,y)",
            "sinh(x)",
            "cosh(x)",
            "tanh(x)",
            "exp(x)",
            "log(x)",
            "log10(x)",
            "sqrt(x)",
            "erf(x)",
            "erfc(x)",
            "square(x)",
            "cube(x)",
            "recip(x)",
            "x^y",
            "x/y",
            "x*y+x-y",
            "min(x,y)",
            "max(x,y)",
        ] {
            let (_, derivatives) = evaluate(source, &[("x", 0.4), ("y", 0.7)]);
            for (i, derivative) in derivatives.into_iter().enumerate() {
                let mut plus = [("x", 0.4), ("y", 0.7)];
                let mut minus = plus;
                plus[i].1 += 1e-6;
                minus[i].1 -= 1e-6;
                let numerical = (evaluate(source, &plus).0 - evaluate(source, &minus).0) / 2e-6;
                assert!(
                    (derivative - numerical).abs() < 2e-8,
                    "{source} input {i}: {derivative} vs {numerical}"
                );
            }
        }
    }

    #[test]
    fn branch_derivatives_and_negative_powers() {
        assert_eq!(
            evaluate("min(x,y)", &[("x", 1.0), ("y", 1.0)]).1,
            vec![0.0, 1.0]
        );
        assert_eq!(
            evaluate("max(x,y)", &[("x", 1.0), ("y", 1.0)]).1,
            vec![1.0, 0.0]
        );
        assert_eq!(evaluate("abs(x)", &[("x", 0.0)]).1, vec![1.0]);
        assert_eq!(
            evaluate("select(x,sqrt(-1),y^2)", &[("x", 0.0), ("y", 2.0)]),
            (4.0, vec![0.0, 4.0])
        );
        assert_eq!(evaluate("x^3", &[("x", -2.0)]), (-8.0, vec![12.0]));
        assert_eq!(evaluate("x^0", &[("x", 0.0)]), (1.0, vec![0.0]));
        for source in [
            "step(x)", "delta(x)", "floor(x)", "ceil(x)", "x==1", "x!=1", "x<1", "x<=1", "x>1",
            "x>=1",
        ] {
            assert_eq!(evaluate(source, &[("x", 1.0)]).1, vec![0.0], "{source}");
        }
    }

    #[test]
    fn shared_subexpressions_accumulate_gradients() {
        let expression = Expression::parse("a*a+a; a=x*y").unwrap();
        assert_eq!(expression.instructions.len(), 5);
        assert_eq!(
            evaluate("a*a+a; a=x*y", &[("x", 2.0), ("y", 3.0)]),
            (42.0, vec![39.0, 26.0])
        );
    }

    #[test]
    fn restricted_derivatives_preserve_the_selected_inputs() {
        let expression = Expression::parse("x^y+z*sqrt(t)").unwrap();
        let mut workspace = expression.workspace_with_derivatives(
            ["x", "z"].map(|name| expression.variable_index(name).unwrap()),
        );
        for (name, value) in [("x", -2.0), ("y", 3.0), ("z", 2.0), ("t", 4.0)] {
            workspace.set(expression.variable_index(name), value);
        }
        assert_eq!(expression.evaluate(&mut workspace), -4.0);
        assert_eq!(workspace.derivative(expression.variable_index("x")), 12.0);
        assert_eq!(workspace.derivative(expression.variable_index("z")), 2.0);
        assert_eq!(workspace.derivative(expression.variable_index("y")), 0.0);
        assert_eq!(workspace.derivative(expression.variable_index("t")), 0.0);
        assert_eq!(evaluate("x^2", &[("x", 1e-180)]).1, vec![2e-180]);
    }
}
