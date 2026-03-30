// Expression parser and evaluator for Custom*Force.
// Supports: +, -, *, /, ^ (right-associative), unary -, parentheses,
// numeric literals, named variables, and built-in functions.
//
// Built-in functions: sin, cos, tan, asin, acos, atan, sqrt, exp, log,
//                     log10, abs, floor, ceil, step (Heaviside), delta, select

use std::collections::HashMap;
use std::fmt;

// ─── Public API ──────────────────────────────────────────────────────────────

/// Compiled expression ready for repeated evaluation.
#[derive(Debug, Clone)]
pub struct Expression {
    root: Expr,
    /// Variables referenced by this expression (for caller validation).
    pub variables: Vec<String>,
}

impl Expression {
    /// Parse a mathematical expression string.
    pub fn parse(src: &str) -> Result<Self, ParseError> {
        let tokens = tokenize(src)?;
        let mut parser = Parser { tokens, pos: 0 };
        let root = parser.parse_expr(0)?;
        if parser.pos != parser.tokens.len() {
            return Err(ParseError::new("unexpected token after expression"));
        }
        let mut variables = Vec::new();
        collect_vars(&root, &mut variables);
        variables.sort();
        variables.dedup();
        Ok(Expression { root, variables })
    }

    /// Evaluate the expression with a variable binding map.
    pub fn compute(&self, vars: &HashMap<&str, f64>) -> Result<f64, EvalError> {
        compute_node(&self.root, vars)
    }
}

// ─── Error types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ParseError(String);

impl ParseError {
    fn new(msg: &str) -> Self {
        Self(msg.to_string())
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parse error: {}", self.0)
    }
}

#[derive(Debug, Clone)]
pub struct EvalError(pub String);

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "eval error: {}", self.0)
    }
}

// ─── AST ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum Expr {
    Lit(f64),
    Var(String),
    Neg(Box<Expr>),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>),
}

#[derive(Debug, Clone, Copy)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

// ─── Tokenizer ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Num(f64),
    Ident(String),
    LParen,
    RParen,
    Comma,
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
}

fn tokenize(src: &str) -> Result<Vec<Token>, ParseError> {
    let mut tokens = Vec::new();
    let mut chars = src.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            ' ' | '\t' | '\n' => {
                chars.next();
            }
            '(' => {
                chars.next();
                tokens.push(Token::LParen);
            }
            ')' => {
                chars.next();
                tokens.push(Token::RParen);
            }
            ',' => {
                chars.next();
                tokens.push(Token::Comma);
            }
            '+' => {
                chars.next();
                tokens.push(Token::Plus);
            }
            '-' => {
                chars.next();
                tokens.push(Token::Minus);
            }
            '*' => {
                chars.next();
                tokens.push(Token::Star);
            }
            '/' => {
                chars.next();
                tokens.push(Token::Slash);
            }
            '^' => {
                chars.next();
                tokens.push(Token::Caret);
            }
            '0'..='9' | '.' => {
                let mut s = String::new();
                while let Some(&d) = chars.peek() {
                    if d.is_ascii_digit()
                        || d == '.'
                        || d == 'e'
                        || d == 'E'
                        || ((d == '+' || d == '-') && s.ends_with(['e', 'E']))
                    {
                        s.push(d);
                        chars.next();
                    } else {
                        break;
                    }
                }
                let v: f64 = s
                    .parse()
                    .map_err(|_| ParseError::new(&format!("bad number: {s}")))?;
                tokens.push(Token::Num(v));
            }
            'a'..='z' | 'A'..='Z' | '_' => {
                let mut s = String::new();
                while let Some(&d) = chars.peek() {
                    if d.is_alphanumeric() || d == '_' {
                        s.push(d);
                        chars.next();
                    } else {
                        break;
                    }
                }
                tokens.push(Token::Ident(s));
            }
            other => return Err(ParseError::new(&format!("unexpected char: {other}"))),
        }
    }
    Ok(tokens)
}

// ─── Pratt parser ────────────────────────────────────────────────────────────

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

/// Returns (left_bp, right_bp) for infix operators.
fn infix_bp(tok: &Token) -> Option<(u8, u8)> {
    match tok {
        Token::Plus | Token::Minus => Some((1, 2)),
        Token::Star | Token::Slash => Some((3, 4)),
        Token::Caret => Some((6, 5)), // right-associative
        _ => None,
    }
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next_tok(&mut self) -> Option<&Token> {
        let t = self.tokens.get(self.pos);
        self.pos += 1;
        t
    }

    fn expect(&mut self, expected: &Token) -> Result<(), ParseError> {
        match self.next_tok() {
            Some(t) if t == expected => Ok(()),
            other => Err(ParseError::new(&format!(
                "expected {expected:?}, got {other:?}"
            ))),
        }
    }

    fn parse_expr(&mut self, min_bp: u8) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_atom()?;
        while let Some((l_bp, r_bp)) = self.peek().and_then(infix_bp) {
            if l_bp <= min_bp {
                break;
            }

            let op = match self.next_tok().unwrap() {
                Token::Plus => BinOp::Add,
                Token::Minus => BinOp::Sub,
                Token::Star => BinOp::Mul,
                Token::Slash => BinOp::Div,
                Token::Caret => BinOp::Pow,
                _ => unreachable!(),
            };
            let rhs = self.parse_expr(r_bp)?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_atom(&mut self) -> Result<Expr, ParseError> {
        match self.next_tok() {
            Some(Token::Num(v)) => Ok(Expr::Lit(*v)),
            Some(Token::Minus) => {
                let e = self.parse_expr(5)?;
                Ok(Expr::Neg(Box::new(e)))
            }
            Some(Token::LParen) => {
                let e = self.parse_expr(0)?;
                self.expect(&Token::RParen)?;
                Ok(e)
            }
            Some(Token::Ident(name)) => {
                let name = name.clone();
                if self.peek() == Some(&Token::LParen) {
                    self.next_tok(); // consume '('
                    let mut args = Vec::new();
                    while self.peek() != Some(&Token::RParen) {
                        args.push(self.parse_expr(0)?);
                        if self.peek() == Some(&Token::Comma) {
                            self.next_tok();
                        }
                    }
                    self.expect(&Token::RParen)?;
                    Ok(Expr::Call(name, args))
                } else {
                    Ok(Expr::Var(name))
                }
            }
            other => Err(ParseError::new(&format!("unexpected token: {other:?}"))),
        }
    }
}

// ─── Evaluator ───────────────────────────────────────────────────────────────

fn compute_node(expr: &Expr, vars: &HashMap<&str, f64>) -> Result<f64, EvalError> {
    match expr {
        Expr::Lit(v) => Ok(*v),
        Expr::Var(name) => vars
            .get(name.as_str())
            .copied()
            .ok_or_else(|| EvalError(format!("undefined variable: {name}"))),
        Expr::Neg(e) => Ok(-compute_node(e, vars)?),
        Expr::Bin(op, l, r) => {
            let lv = compute_node(l, vars)?;
            let rv = compute_node(r, vars)?;
            Ok(match op {
                BinOp::Add => lv + rv,
                BinOp::Sub => lv - rv,
                BinOp::Mul => lv * rv,
                BinOp::Div => lv / rv,
                BinOp::Pow => lv.powf(rv),
            })
        }
        Expr::Call(name, args) => {
            let argv: Result<Vec<f64>, _> = args.iter().map(|a| compute_node(a, vars)).collect();
            builtin(name, &argv?)
        }
    }
}

fn builtin(name: &str, args: &[f64]) -> Result<f64, EvalError> {
    let arg1 = || {
        args.first()
            .copied()
            .ok_or_else(|| EvalError(format!("{name}: requires 1 argument")))
    };
    let arg2 = || {
        if args.len() >= 2 {
            Ok((args[0], args[1]))
        } else {
            Err(EvalError(format!("{name}: requires 2 arguments")))
        }
    };

    match name {
        "sin" => Ok(arg1()?.sin()),
        "cos" => Ok(arg1()?.cos()),
        "tan" => Ok(arg1()?.tan()),
        "asin" => Ok(arg1()?.asin()),
        "acos" => Ok(arg1()?.acos()),
        "atan" => Ok(arg1()?.atan()),
        "atan2" => {
            let (y, x) = arg2()?;
            Ok(y.atan2(x))
        }
        "sqrt" => Ok(arg1()?.sqrt()),
        "exp" => Ok(arg1()?.exp()),
        "log" => Ok(arg1()?.ln()),
        "log10" => Ok(arg1()?.log10()),
        "abs" => Ok(arg1()?.abs()),
        "floor" => Ok(arg1()?.floor()),
        "ceil" => Ok(arg1()?.ceil()),
        "min" => {
            let (a, b) = arg2()?;
            Ok(a.min(b))
        }
        "max" => {
            let (a, b) = arg2()?;
            Ok(a.max(b))
        }
        // Heaviside: 1 if x >= 0, else 0
        "step" => Ok(if arg1()? >= 0.0 { 1.0 } else { 0.0 }),
        // Kronecker delta: 1 if x == 0
        "delta" => Ok(if arg1()? == 0.0 { 1.0 } else { 0.0 }),
        // select(cond, if_true, if_false)
        "select" => {
            if args.len() < 3 {
                return Err(EvalError("select: needs 3 arguments".into()));
            }
            Ok(if args[0] != 0.0 { args[1] } else { args[2] })
        }
        _ => Err(EvalError(format!("unknown function: {name}"))),
    }
}

fn collect_vars(expr: &Expr, out: &mut Vec<String>) {
    match expr {
        Expr::Var(name) => out.push(name.clone()),
        Expr::Neg(e) => collect_vars(e, out),
        Expr::Bin(_, l, r) => {
            collect_vars(l, out);
            collect_vars(r, out);
        }
        Expr::Call(_, args) => args.iter().for_each(|a| collect_vars(a, out)),
        Expr::Lit(_) => {}
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn compute(src: &str, vars: &[(&str, f64)]) -> f64 {
        let expr = Expression::parse(src).unwrap();
        let map: HashMap<&str, f64> = vars.iter().cloned().collect();
        expr.compute(&map).unwrap()
    }

    #[test]
    fn test_literals() {
        assert_eq!(compute("3.14", &[]), 314.0 / 100.0);
        assert_eq!(compute("1e2", &[]), 100.0);
    }

    #[test]
    fn test_arithmetic() {
        assert_eq!(compute("2 + 3", &[]), 5.0);
        assert_eq!(compute("10 - 4 * 2", &[]), 2.0);
        assert_eq!(compute("2^10", &[]), 1024.0);
        assert!((compute("1/3", &[]) - 1.0 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_variables() {
        // k*(r-r0)^2 = 2.0*(1.5-1.0)^2 = 0.5
        assert_eq!(
            compute("k*(r-r0)^2", &[("k", 2.0), ("r", 1.5), ("r0", 1.0)]),
            0.5
        );
    }

    #[test]
    fn test_functions() {
        assert!((compute("sqrt(4)", &[]) - 2.0).abs() < 1e-12);
        assert!((compute("sin(0)", &[]) - 0.0).abs() < 1e-12);
        assert!((compute("cos(0)", &[]) - 1.0).abs() < 1e-12);
        assert!((compute("exp(0)", &[]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_unary_minus() {
        assert_eq!(compute("-3", &[]), -3.0);
        assert_eq!(compute("-k", &[("k", 2.0)]), -2.0);
    }

    #[test]
    fn test_step() {
        assert_eq!(compute("step(-1)", &[]), 0.0);
        assert_eq!(compute("step(0)", &[]), 1.0);
        assert_eq!(compute("step(5)", &[]), 1.0);
    }

    #[test]
    fn test_variable_list() {
        let expr = Expression::parse("k*(r-r0)^2").unwrap();
        assert_eq!(expr.variables, vec!["k", "r", "r0"]);
    }
}
