//! Compiled scalar expressions with reusable reverse-mode differentiation.

use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone)]
pub struct Expression {
    instructions: Vec<Instruction>,
    pub variables: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Workspace {
    inputs: Vec<f64>,
    values: Vec<f64>,
    adjoints: Vec<f64>,
    gradients: Vec<f64>,
}

impl Workspace {
    pub fn set(&mut self, index: Option<usize>, value: f64) {
        if let Some(index) = index {
            self.inputs[index] = value;
        }
    }

    pub fn derivative(&self, index: Option<usize>) -> f64 {
        index.map_or(0.0, |index| self.gradients[index])
    }
}

impl Expression {
    pub fn parse(src: &str) -> Result<Self, ParseError> {
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
            definitions.insert(name.to_owned(), parse_tree(value, &definitions)?);
        }
        let root = parse_tree(parts[0], &definitions)?;
        let mut variables = Vec::new();
        collect_vars(&root, &mut variables);
        variables.sort();
        variables.dedup();
        let mut instructions = Vec::new();
        compile(&root, &variables, &mut instructions, &mut HashMap::new());
        Ok(Self {
            instructions,
            variables,
        })
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
        let names: Vec<_> = names.into_iter().collect();
        for name in &self.variables {
            if !names.contains(&name.as_str()) {
                return Err(EvalError(format!("undefined variable: {name}")));
            }
        }
        Ok(())
    }

    pub fn workspace(&self) -> Workspace {
        Workspace {
            inputs: vec![0.0; self.variables.len()],
            values: vec![0.0; self.instructions.len()],
            adjoints: vec![0.0; self.instructions.len()],
            gradients: vec![0.0; self.variables.len()],
        }
    }

    pub fn compute(&self, vars: &HashMap<&str, f64>) -> Result<f64, EvalError> {
        let mut workspace = self.workspace();
        for (index, name) in self.variables.iter().enumerate() {
            workspace.inputs[index] = *vars
                .get(name.as_str())
                .ok_or_else(|| EvalError(format!("undefined variable: {name}")))?;
        }
        Ok(self.evaluate(&mut workspace))
    }

    /// Evaluate energy and all input derivatives without allocating.
    pub fn evaluate(&self, workspace: &mut Workspace) -> f64 {
        assert_eq!(workspace.values.len(), self.instructions.len());
        for (index, instruction) in self.instructions.iter().enumerate() {
            workspace.values[index] = match *instruction {
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
            if adjoint == 0.0 {
                continue;
            }
            match self.instructions[index] {
                Instruction::Variable(input) => workspace.gradients[input] += adjoint,
                Instruction::Operation(op, args) => {
                    let values = args.map(|a| workspace.values[a]);
                    let derivatives = op.derivatives(values, workspace.values[index]);
                    for i in 0..op.arity() {
                        if derivatives[i] != 0.0 {
                            workspace.adjoints[args[i]] += adjoint * derivatives[i];
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Instruction {
    Literal(u64),
    Variable(usize),
    Operation(Op, [usize; 3]),
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

    fn derivatives(self, [a, b, _]: [f64; 3], value: f64) -> [f64; 3] {
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
                result[1] = -a / (b * b);
                1.0 / b
            }
            Self::Pow => {
                result[1] = value * a.ln();
                if b == 0.0 {
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
            Self::Recip => -1.0 / (a * a),
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

fn collect_vars(expr: &Expr, variables: &mut Vec<String>) {
    match expr {
        Expr::Variable(name) => variables.push(name.clone()),
        Expr::Operation(_, args) => args.iter().for_each(|arg| collect_vars(arg, variables)),
        _ => {}
    }
}

fn compile(
    expr: &Expr,
    variables: &[String],
    instructions: &mut Vec<Instruction>,
    cache: &mut HashMap<Instruction, usize>,
) -> usize {
    let instruction = match expr {
        Expr::Literal(value) => Instruction::Literal(value.to_bits()),
        Expr::Variable(name) => Instruction::Variable(variables.binary_search(name).unwrap()),
        Expr::Operation(op, args) => {
            let mut indexes = [0; 3];
            for (index, arg) in args.iter().enumerate() {
                indexes[index] = compile(arg, variables, instructions, cache);
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

fn parse_tree(src: &str, definitions: &HashMap<String, Expr>) -> Result<Expr, ParseError> {
    let mut parser = Parser {
        tokens: tokenize(src)?,
        position: 0,
        definitions,
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
                    let op = Op::function(&name)?;
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
                    if args.len() != op.arity() {
                        return Err(ParseError::new(&format!(
                            "{name} requires {} arguments",
                            op.arity()
                        )));
                    }
                    Expr::Operation(op, args)
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
}
