use super::*;

pub(crate) struct DerivativeProgram {
    expression: Expression,
    values: Vec<f64>,
    adjoints: Vec<f64>,
    inputs: Vec<usize>,
    constants: Vec<(usize, Instruction)>,
    varying: Vec<(usize, Instruction)>,
    reverse: Vec<ReverseStep>,
    function_derivatives: Vec<f64>,
    periodic_box: Option<[[f64; 3]; 3]>,
}

enum ReverseStep {
    Operation {
        target: usize,
        op: Op,
        args: [usize; 3],
        active: [bool; 3],
    },
    Function {
        target: usize,
        call: usize,
        active: Vec<bool>,
    },
}

impl DerivativeProgram {
    pub(crate) fn new(
        expression: Expression,
        varying_inputs: &[bool],
        derivative_inputs: impl IntoIterator<Item = usize>,
    ) -> Self {
        assert_eq!(expression.variables.len(), varying_inputs.len());
        let mut values = vec![0.0; expression.instructions.len()];
        let mut inputs = vec![0; expression.variables.len()];
        let mut varies = vec![false; values.len()];
        let differentiable = expression
            .workspace_with_derivatives(derivative_inputs)
            .differentiable;
        let mut constants = Vec::new();
        let mut varying = Vec::new();
        let mut reverse = Vec::new();
        for (index, &instruction) in expression.instructions.iter().enumerate() {
            match instruction {
                Instruction::Literal(bits) => {
                    values[index] = f64::from_bits(bits);
                    continue;
                }
                Instruction::Variable(input) => {
                    inputs[input] = index;
                    varies[index] = varying_inputs[input];
                    continue;
                }
                Instruction::Operation(op, args) => {
                    varies[index] = args[..op.arity()].iter().any(|&i| varies[i]);
                    if differentiable[index] {
                        reverse.push(ReverseStep::Operation {
                            target: index,
                            op,
                            args,
                            active: args.map(|i| differentiable[i]),
                        });
                    }
                }
                Instruction::Function(call) => {
                    let arguments = &expression.function_calls[call].arguments;
                    varies[index] = arguments.iter().any(|&i| varies[i]);
                    if differentiable[index] {
                        reverse.push(ReverseStep::Function {
                            target: index,
                            call,
                            active: arguments.iter().map(|&i| differentiable[i]).collect(),
                        });
                    }
                }
            }
            if varies[index] {
                varying.push((index, instruction));
            } else {
                constants.push((index, instruction));
            }
        }
        reverse.reverse();
        let adjoints = vec![0.; values.len()];
        let function_derivatives = vec![
            0.;
            expression
                .function_calls
                .iter()
                .map(|c| c.arguments.len())
                .sum()
        ];
        Self {
            expression,
            values,
            adjoints,
            inputs,
            constants,
            varying,
            reverse,
            function_derivatives,
            periodic_box: None,
        }
    }
    pub(crate) fn set_periodic_box(&mut self, vectors: Option<&[[f64; 3]; 3]>) {
        self.periodic_box = vectors.copied();
    }
    pub(crate) fn set(&mut self, index: usize, value: f64) {
        self.values[self.inputs[index]] = value;
    }
    pub(crate) fn prepare(&mut self) {
        Self::run(
            &self.expression,
            &mut self.values,
            &mut self.function_derivatives,
            &self.constants,
            self.periodic_box.as_ref(),
        );
    }
    pub(crate) fn value(&mut self) -> f64 {
        Self::run(
            &self.expression,
            &mut self.values,
            &mut self.function_derivatives,
            &self.varying,
            self.periodic_box.as_ref(),
        );
        *self.values.last().unwrap()
    }
    pub(crate) fn evaluate(&mut self) -> f64 {
        let value = self.value();
        self.adjoints.fill(0.);
        let root = self.adjoints.len() - 1;
        self.adjoints[root] = 1.;
        for step in &self.reverse {
            match step {
                ReverseStep::Operation {
                    target,
                    op,
                    args,
                    active,
                } => {
                    let adjoint = self.adjoints[*target];
                    if adjoint == 0. {
                        continue;
                    }
                    let derivatives =
                        op.derivatives(args.map(|i| self.values[i]), self.values[*target], *active);
                    for i in 0..op.arity() {
                        if active[i] && derivatives[i] != 0. {
                            self.adjoints[args[i]] += adjoint * derivatives[i];
                        }
                    }
                }
                ReverseStep::Function {
                    target,
                    call,
                    active,
                } => {
                    let adjoint = self.adjoints[*target];
                    if adjoint == 0. {
                        continue;
                    }
                    let call = &self.expression.function_calls[*call];
                    for (i, &arg) in call.arguments.iter().enumerate() {
                        let derivative = self.function_derivatives[call.derivative_offset + i];
                        if active[i] && derivative != 0. {
                            self.adjoints[arg] += adjoint * derivative;
                        }
                    }
                }
            }
        }
        value
    }
    pub(crate) fn derivative(&self, index: usize) -> f64 {
        self.adjoints[self.inputs[index]]
    }
    fn run(
        expression: &Expression,
        values: &mut [f64],
        derivatives: &mut [f64],
        instructions: &[(usize, Instruction)],
        periodic_box: Option<&[[f64; 3]; 3]>,
    ) {
        for &(index, instruction) in instructions {
            values[index] = match instruction {
                Instruction::Operation(op, args) => op.value(args.map(|i| values[i])),
                Instruction::Function(call) => {
                    let call = &expression.function_calls[call];
                    let mut arguments = [0.; 12];
                    for (i, &arg) in call.arguments.iter().enumerate() {
                        arguments[i] = values[arg];
                    }
                    let count = call.arguments.len();
                    let partials =
                        &mut derivatives[call.derivative_offset..call.derivative_offset + count];
                    match expression.function_bindings[call.binding].expect("unbound function") {
                        FunctionBinding::Table(index) => {
                            let (value, gradient) = expression.tables[index].1.evaluate([
                                arguments[0],
                                arguments[1],
                                arguments[2],
                            ]);
                            partials.copy_from_slice(&gradient[..count]);
                            value
                        }
                        FunctionBinding::Geometry(function) => {
                            let (value, gradient) =
                                function.evaluate(&arguments[..count], periodic_box);
                            partials.copy_from_slice(&gradient[..count]);
                            value
                        }
                    }
                }
                _ => unreachable!(),
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scheduled_geometry_tracks_periodic_boxes_and_constant_subexpressions() {
        let vectors = [[2.0, 0.0, 0.0], [0.4, 2.2, 0.0], [-0.3, 0.2, 2.4]];
        for source in [
            "a*distance(p1,p2)+angle(p1,p2,p3)+cos(dihedral(p1,p2,p3,p4))",
            "pointdistance(x1,y1,z1,(x2+x3)/2,(y2+y3)/2,(z2+z3)/2)",
            "pointdistance(a,0,0,0,0,0)+distance(p1,p4)",
        ] {
            let expression = Expression::parse_geometric(source, Some(('p', 4))).unwrap();
            let varying: Vec<_> = expression
                .variables
                .iter()
                .map(|name| name != "a")
                .collect();
            let mut program =
                DerivativeProgram::new(expression.clone(), &varying, 0..varying.len());
            let mut ordinary = expression.workspace();
            let coordinates = [0.2, 0.3, 0.1, 0.6, 0.1, 0.4, 1.4, 1.2, 0.7, 1.1, 1.8, 0.3];
            for periodic in [true, false, true] {
                let vectors = periodic.then_some(&vectors);
                program.set_periodic_box(vectors);
                ordinary.set_periodic_box(vectors);
                for (index, name) in expression.variables.iter().enumerate() {
                    let value = if name == "a" {
                        1.7
                    } else {
                        let particle: usize = name[1..].parse().unwrap();
                        let axis = match name.as_bytes()[0] {
                            b'x' => 0,
                            b'y' => 1,
                            _ => 2,
                        };
                        coordinates[3 * (particle - 1) + axis]
                    };
                    program.set(index, value);
                    ordinary.set(Some(index), value);
                }
                program.prepare();
                let expected = expression.evaluate(&mut ordinary);
                assert!((program.evaluate() - expected).abs() < 1e-13);
                for index in 0..varying.len() {
                    assert!(
                        (program.derivative(index) - ordinary.derivative(Some(index))).abs()
                            < 1e-12
                    );
                }
            }
        }
    }
    #[test]
    fn scheduled_derivatives_preserve_dynamic_globals_and_shared_subexpressions() {
        for source in [
            "exp(a)*x+sqrt(b)*y+sin(a+b)",
            "select(x,log(abs(y)+1),cos(a))+erf(x)+erfc(y)+atan2(x,y)",
            "floor(x)+ceil(y)+step(x)+delta(y)+min(a,x)+max(b,y)",
            "abs(a*x)^2.3+recip(b)+square(x)+cube(y)",
            "sinh(x)+cosh(y)+tanh(a)+asin(x)+acos(y)+atan(b)",
            "sec(x)+csc(y)+tan(a)+cot(b)+log10(abs(x)+1)",
            "a",
            "x",
            "0",
            "a+b",
            "(x+y)^3+(x+y)^2+sin(x+y)",
            "t(x)*t(a)+t(b+x)+d(y)",
        ] {
            let mut expression = Expression::parse_unbound(source).unwrap();
            expression
                .add_tabulated_function(
                    "t",
                    crate::Continuous1DFunction::new(
                        [4],
                        vec![0., 1., -0.2, 0.5],
                        [[-1., 1.]],
                        false,
                    )
                    .unwrap(),
                )
                .unwrap();
            expression
                .add_tabulated_function(
                    "d",
                    crate::Discrete1DFunction::new([3], vec![0., 1., 2.]).unwrap(),
                )
                .unwrap();
            let varying: Vec<_> = expression
                .variables
                .iter()
                .map(|s| s == "x" || s == "y")
                .collect();
            let mut program =
                DerivativeProgram::new(expression.clone(), &varying, 0..varying.len());
            let mut workspace = expression.workspace();
            for a in [0.1, 0.2, 0.7] {
                for (i, name) in expression.variables.iter().enumerate() {
                    if !varying[i] {
                        let v = if name == "a" { a } else { 0.4 };
                        program.set(i, v);
                        workspace.set(Some(i), v);
                    }
                }
                program.prepare();
                for x in [-0.7, -0.1, 0.3, 0.8] {
                    for (i, name) in expression.variables.iter().enumerate() {
                        if varying[i] {
                            let v = if name == "x" { x } else { 0.5 };
                            program.set(i, v);
                            workspace.set(Some(i), v);
                        }
                    }
                    let expected = expression.evaluate(&mut workspace);
                    assert_eq!(program.value(), expected, "{source}");
                    assert_eq!(program.evaluate(), expected, "{source}");
                    for i in 0..varying.len() {
                        assert_eq!(
                            program.derivative(i),
                            workspace.derivative(Some(i)),
                            "{source}: {}",
                            expression.variables[i]
                        );
                    }
                }
            }
        }
    }
}
