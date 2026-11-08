use super::{parse_tree, EvalError, Expr, Expression, GeometryMode, Op, ParseError, Workspace};
use crate::tabulated::TabulatedFunction;
use std::collections::HashMap;

/// A compiled three-component expression with OpenMM vector operations.
#[derive(Clone, Debug)]
pub struct VectorExpression {
    components: [Expression; 3],
    bindings: [Vec<(usize, usize, usize)>; 3],
    pub variables: Vec<String>,
}

pub struct VectorWorkspace {
    inputs: Vec<[f64; 3]>,
    components: [Workspace; 3],
}

impl VectorWorkspace {
    pub fn set(&mut self, index: Option<usize>, value: [f64; 3]) {
        if let Some(index) = index {
            self.inputs[index] = value;
        }
    }
}

impl VectorExpression {
    pub fn parse_unbound(src: &str) -> Result<Self, ParseError> {
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
                parse_tree(value, &definitions, true, GeometryMode::None)?,
            );
        }
        Self::from_root(parse_tree(
            parts[0],
            &definitions,
            true,
            GeometryMode::None,
        )?)
    }

    pub(super) fn from_root(root: Expr) -> Result<Self, ParseError> {
        let components = lower(&root)?.map(|node| Expression::from_root(node, GeometryMode::None));
        let mut variables: Vec<_> = components
            .iter()
            .flat_map(|expr| {
                expr.variables
                    .iter()
                    .map(|name| name.split_once(':').unwrap().1.to_owned())
            })
            .collect();
        variables.sort();
        variables.dedup();
        let bindings = std::array::from_fn(|axis| {
            components[axis]
                .variables
                .iter()
                .enumerate()
                .map(|(input, name)| {
                    let (axis, name) = name.split_once(':').unwrap();
                    (
                        input,
                        variables.binary_search(&name.to_owned()).unwrap(),
                        axis.parse().unwrap(),
                    )
                })
                .collect()
        });
        Ok(Self {
            components,
            bindings,
            variables,
        })
    }

    pub fn variable_index(&self, name: &str) -> Option<usize> {
        self.variables
            .binary_search_by(|value| value.as_str().cmp(name))
            .ok()
    }

    pub fn add_tabulated_function(
        &mut self,
        name: &str,
        function: impl Into<TabulatedFunction>,
    ) -> Result<usize, EvalError> {
        if matches!(name, "dot" | "cross" | "vector" | "_x" | "_y" | "_z") {
            return Err(EvalError(format!("reserved vector function name: {name}")));
        }
        let function = function.into();
        let mut index = 0;
        for component in &mut self.components {
            index = component.add_tabulated_function(name, function.clone())?;
        }
        Ok(index)
    }

    pub fn validate_functions(&self) -> Result<(), EvalError> {
        for component in &self.components {
            component.validate_functions()?;
        }
        Ok(())
    }

    pub fn workspace(&self) -> VectorWorkspace {
        VectorWorkspace {
            inputs: vec![[0.0; 3]; self.variables.len()],
            components: std::array::from_fn(|axis| {
                self.components[axis].workspace_with_derivatives(std::iter::empty())
            }),
        }
    }

    pub fn evaluate(&self, workspace: &mut VectorWorkspace) -> [f64; 3] {
        std::array::from_fn(|axis| {
            for &(input, variable, source_axis) in &self.bindings[axis] {
                workspace.components[axis]
                    .set(Some(input), workspace.inputs[variable][source_axis]);
            }
            self.components[axis].evaluate(&mut workspace.components[axis])
        })
    }
}

fn lower(node: &Expr) -> Result<[Expr; 3], ParseError> {
    let result = match node {
        Expr::Literal(value) => std::array::from_fn(|_| Expr::Literal(*value)),
        Expr::Variable(name) => {
            std::array::from_fn(|axis| Expr::Variable(format!("{axis}:{name}")))
        }
        Expr::Operation(op, args) => {
            let args = args.iter().map(lower).collect::<Result<Vec<_>, _>>()?;
            std::array::from_fn(|axis| {
                Expr::Operation(*op, args.iter().map(|value| value[axis].clone()).collect())
            })
        }
        Expr::Function(name, args) => {
            let expected = match name.as_str() {
                "dot" | "cross" => Some(2),
                "vector" => Some(3),
                "_x" | "_y" | "_z" => Some(1),
                _ => None,
            };
            if expected.is_some_and(|n| n != args.len()) {
                return Err(ParseError::new(&format!(
                    "{name} requires {} arguments",
                    expected.unwrap()
                )));
            }
            let args = args.iter().map(lower).collect::<Result<Vec<_>, _>>()?;
            match name.as_str() {
                "dot" => {
                    let terms: Vec<_> = (0..3)
                        .map(|k| {
                            Expr::Operation(Op::Mul, vec![args[0][k].clone(), args[1][k].clone()])
                        })
                        .collect();
                    let sum = Expr::Operation(
                        Op::Add,
                        vec![
                            Expr::Operation(Op::Add, terms[..2].to_vec()),
                            terms[2].clone(),
                        ],
                    );
                    std::array::from_fn(|_| sum.clone())
                }
                "cross" => std::array::from_fn(|k| {
                    let a = (k + 1) % 3;
                    let b = (k + 2) % 3;
                    Expr::Operation(
                        Op::Sub,
                        vec![
                            Expr::Operation(Op::Mul, vec![args[0][a].clone(), args[1][b].clone()]),
                            Expr::Operation(Op::Mul, vec![args[0][b].clone(), args[1][a].clone()]),
                        ],
                    )
                }),
                "vector" => std::array::from_fn(|axis| args[axis][axis].clone()),
                "_x" | "_y" | "_z" => {
                    let axis = match name.as_str() {
                        "_x" => 0,
                        "_y" => 1,
                        _ => 2,
                    };
                    std::array::from_fn(|_| args[0][axis].clone())
                }
                _ => std::array::from_fn(|axis| {
                    Expr::Function(
                        name.clone(),
                        args.iter().map(|value| value[axis].clone()).collect(),
                    )
                }),
            }
        }
    };
    Ok(result)
}

pub(crate) struct VectorValueProgram {
    components: Vec<super::ValueProgram>,
    inputs: Vec<Vec<(usize, usize, usize)>>,
    outputs: [usize; 3],
}

impl VectorExpression {
    pub(crate) fn value_program(self, varying_inputs: &[bool]) -> VectorValueProgram {
        let mut unique = Vec::new();
        let mut sources = [0; 3];
        for (axis, source) in sources.iter_mut().enumerate() {
            let expression = &self.components[axis];
            if let Some(previous) = unique.iter().position(|&i: &usize| {
                let other = &self.components[i];
                expression.variables == other.variables
                    && expression.instructions == other.instructions
                    && expression.function_calls == other.function_calls
                    && expression.function_names == other.function_names
            }) {
                *source = previous;
            } else {
                *source = unique.len();
                unique.push(axis);
            }
        }
        let mut inputs = vec![Vec::new(); self.variables.len()];
        let mut components = Vec::new();
        for (index, &axis) in unique.iter().enumerate() {
            let mut varying = vec![false; self.components[axis].variables.len()];
            for &(input, variable, source_axis) in &self.bindings[axis] {
                varying[input] = varying_inputs[variable];
                inputs[variable].push((index, input, source_axis));
            }
            components.push(super::ValueProgram::new(
                self.components[axis].clone(),
                &varying,
            ));
        }
        VectorValueProgram {
            components,
            inputs,
            outputs: sources,
        }
    }
}

impl VectorValueProgram {
    pub(crate) fn set(&mut self, input: usize, value: [f64; 3]) {
        for &(component, input, axis) in &self.inputs[input] {
            self.components[component].set(input, value[axis]);
        }
    }
    pub(crate) fn prepare(&mut self) {
        for component in &mut self.components {
            component.prepare();
        }
    }
    pub(crate) fn evaluate(&mut self) -> [f64; 3] {
        let mut values = [0.0; 3];
        for (value, component) in values.iter_mut().zip(&mut self.components) {
            *value = component.evaluate();
        }
        self.outputs.map(|i| values[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn vector_operations_preserve_component_semantics() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, -5.0, 6.0];
        for (source, expected) in [
            ("cross(a,b)", [27.0, 6.0, -13.0]),
            ("dot(a,b)", [12.0; 3]),
            ("vector(_z(a),b,_x(a)+2)", [3.0, -5.0, 3.0]),
            ("sqrt(dot(a,a))*a", a.map(|v| v * 14.0_f64.sqrt())),
            ("cross(c,b)+a; c=2*a", [55.0, 14.0, -23.0]),
            ("vector(a,b,a+b)", [1.0, -5.0, 9.0]),
        ] {
            let expression = VectorExpression::parse_unbound(source).unwrap();
            expression.validate_functions().unwrap();
            let mut workspace = expression.workspace();
            workspace.set(expression.variable_index("a"), a);
            workspace.set(expression.variable_index("b"), b);
            for (actual, expected) in expression
                .evaluate(&mut workspace)
                .into_iter()
                .zip(expected)
            {
                assert_relative_eq!(actual, expected, epsilon = 1e-12);
            }
        }
        for source in ["dot(a)", "cross(a,b,c)", "_z(a,b)", "vector(a,b)"] {
            assert!(VectorExpression::parse_unbound(source).is_err());
        }
    }
}
