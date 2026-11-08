use super::*;

/// A value-only schedule that hoists operations independent of particle inputs.
pub(crate) struct ValueProgram {
    expression: Expression,
    values: Vec<f64>,
    inputs: Vec<usize>,
    constants: Vec<(usize, Instruction)>,
    varying: Vec<(usize, Instruction)>,
}

impl ValueProgram {
    pub(crate) fn new(expression: Expression, varying_inputs: &[bool]) -> Self {
        assert_eq!(expression.variables.len(), varying_inputs.len());
        let mut values = vec![0.0; expression.instructions.len()];
        let mut inputs = vec![0; expression.variables.len()];
        let mut varies = vec![false; values.len()];
        let mut constants = Vec::new();
        let mut varying = Vec::new();
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
                    varies[index] = args[..op.arity()].iter().any(|&i| varies[i])
                }
                Instruction::Function(call) => {
                    varies[index] = expression.function_calls[call]
                        .arguments
                        .iter()
                        .any(|&i| varies[i])
                }
            }
            if varies[index] {
                varying.push((index, instruction));
            } else {
                constants.push((index, instruction));
            }
        }
        Self {
            expression,
            values,
            inputs,
            constants,
            varying,
        }
    }
    pub(crate) fn set(&mut self, input: usize, value: f64) {
        self.values[self.inputs[input]] = value;
    }
    pub(crate) fn prepare(&mut self) {
        Self::run(&self.expression, &mut self.values, &self.constants);
    }
    pub(crate) fn evaluate(&mut self) -> f64 {
        Self::run(&self.expression, &mut self.values, &self.varying);
        *self.values.last().unwrap()
    }
    fn run(expression: &Expression, values: &mut [f64], instructions: &[(usize, Instruction)]) {
        for &(index, instruction) in instructions {
            values[index] = match instruction {
                Instruction::Operation(op, args) => op.value(args.map(|a| values[a])),
                Instruction::Function(call) => {
                    let call = &expression.function_calls[call];
                    let mut arguments = [0.0; 12];
                    for (i, &arg) in call.arguments.iter().enumerate() {
                        arguments[i] = values[arg];
                    }
                    match expression.function_bindings[call.binding].expect("unbound function") {
                        FunctionBinding::Table(index) => {
                            expression.tables[index]
                                .1
                                .evaluate([arguments[0], arguments[1], arguments[2]])
                                .0
                        }
                        FunctionBinding::Geometry(function) => {
                            function
                                .evaluate(&arguments[..call.arguments.len()], None)
                                .0
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
    fn value_schedule_matches_full_evaluation_after_constant_inputs_change() {
        for source in [
            "exp(a)*x+sqrt(b)*y+sin(a+b)",
            "select(x,log(abs(y)+1),cos(a))+erf(x)+erfc(y)+atan2(x,y)",
            "floor(x)+ceil(y)+step(x)+delta(y)+min(a,x)+max(b,y)",
            "abs(a*x)^2.3+recip(b)+square(x)+cube(y)",
            "sinh(x)+cosh(y)+tanh(a)+asin(x)+acos(y)+atan(b)",
            "sec(x)+csc(y)+tan(a)+cot(b)+log10(abs(x)+1)",
            "a",
            "0",
            "x",
            "a+b",
        ] {
            let expression = Expression::parse(source).unwrap();
            let varying: Vec<_> = expression
                .variables
                .iter()
                .map(|name| name == "x" || name == "y")
                .collect();
            let mut program = ValueProgram::new(expression.clone(), &varying);
            let mut workspace = expression.workspace_with_derivatives([]);
            for a in [0.1, 0.2, 0.7] {
                for (i, name) in expression.variables.iter().enumerate() {
                    if !varying[i] {
                        let value = if name == "a" { a } else { 0.4 };
                        program.set(i, value);
                        workspace.set(Some(i), value);
                    }
                }
                program.prepare();
                for x in [-0.7, -0.1, 0.3, 0.8] {
                    for (i, name) in expression.variables.iter().enumerate() {
                        if varying[i] {
                            let value = if name == "x" { x } else { 0.5 };
                            program.set(i, value);
                            workspace.set(Some(i), value);
                        }
                    }
                    assert_eq!(
                        program.evaluate(),
                        expression.evaluate(&mut workspace),
                        "{source}"
                    );
                }
            }
        }
    }
}
