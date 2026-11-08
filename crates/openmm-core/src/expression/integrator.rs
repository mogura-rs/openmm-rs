use super::*;

#[derive(Clone)]
pub(crate) struct EnergyDerivative {
    pub variable: String,
    pub energy: String,
    pub parameter: String,
}

fn parse(src: &str) -> Result<(Expr, Vec<EnergyDerivative>), ParseError> {
    let parts: Vec<_> = src.split(';').collect();
    let mut definitions = HashMap::new();
    for definition in parts[1..].iter().rev().filter(|s| !s.trim().is_empty()) {
        let (name, value) = definition
            .split_once('=')
            .ok_or_else(|| ParseError::new("subexpression does not specify a name"))?;
        if name.trim().is_empty() {
            return Err(ParseError::new("empty subexpression name"));
        }
        definitions.insert(
            name.trim().to_owned(),
            parse_tree(value, &definitions, true, GeometryMode::None)?,
        );
    }
    let root = optimize(parse_tree(
        parts[0],
        &definitions,
        true,
        GeometryMode::None,
    )?);
    let mut derivatives = Vec::new();
    let root = replace(root, &mut derivatives)?;
    Ok((root, derivatives))
}

fn replace(node: Expr, derivatives: &mut Vec<EnergyDerivative>) -> Result<Expr, ParseError> {
    Ok(match node {
        Expr::Function(ref name, ref args) if name == "deriv" => {
            let [Expr::Variable(energy), Expr::Variable(parameter)] = args.as_slice() else {
                return Err(ParseError::new(
                    "deriv requires energy and a Context parameter",
                ));
            };
            if energy != "energy" && !(0..32).any(|i| *energy == format!("energy{i}")) {
                return Err(ParseError::new(
                    "the first deriv argument must be energy or energy0 through energy31",
                ));
            }
            let variable = format!("\0deriv:{}", derivatives.len());
            derivatives.push(EnergyDerivative {
                variable: variable.clone(),
                energy: energy.clone(),
                parameter: parameter.clone(),
            });
            Expr::Variable(variable)
        }
        Expr::Function(name, args) => Expr::Function(
            name,
            args.into_iter()
                .map(|arg| replace(arg, derivatives))
                .collect::<Result<_, _>>()?,
        ),
        Expr::Operation(op, args) => Expr::Operation(
            op,
            args.into_iter()
                .map(|arg| replace(arg, derivatives))
                .collect::<Result<_, _>>()?,
        ),
        value => value,
    })
}

impl Expression {
    pub(crate) fn parse_integrator(src: &str) -> Result<(Self, Vec<EnergyDerivative>), ParseError> {
        let (root, derivatives) = parse(src)?;
        Ok((Self::from_root(root, GeometryMode::None), derivatives))
    }
}

impl VectorExpression {
    pub(crate) fn parse_integrator(src: &str) -> Result<(Self, Vec<EnergyDerivative>), ParseError> {
        let (root, derivatives) = parse(src)?;
        Ok((Self::from_root(root)?, derivatives))
    }
}
