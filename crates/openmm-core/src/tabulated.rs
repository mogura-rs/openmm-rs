//! Uniform-grid natural and periodic splines, and nearest-index tables.

use std::{fmt, sync::Arc};

#[derive(Debug, Clone)]
pub struct TableError(pub String);

impl fmt::Display for TableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for TableError {}

#[derive(Debug, Clone)]
pub struct ContinuousFunction<const D: usize> {
    shape: [usize; D],
    bounds: [[f64; 2]; D],
    periodic: bool,
    nodes: Arc<Vec<Vec<f64>>>,
    update_count: u64,
}

pub type Continuous1DFunction = ContinuousFunction<1>;
pub type Continuous2DFunction = ContinuousFunction<2>;
pub type Continuous3DFunction = ContinuousFunction<3>;

impl<const D: usize> ContinuousFunction<D> {
    pub fn new(
        shape: [usize; D],
        values: Vec<f64>,
        bounds: [[f64; 2]; D],
        periodic: bool,
    ) -> Result<Self, TableError> {
        check_shape(&shape, values.len(), if periodic { 3 } else { 2 })?;
        if bounds
            .iter()
            .any(|b| !b[0].is_finite() || !b[1].is_finite() || b[1] <= b[0])
        {
            return Err(TableError(
                "table bounds must be finite and increasing".into(),
            ));
        }
        if periodic {
            for axis in 0..D {
                let stride: usize = shape[..axis].iter().product();
                for index in 0..values.len() {
                    if (index / stride).is_multiple_of(shape[axis]) {
                        let first = values[index];
                        let last = values[index + (shape[axis] - 1) * stride];
                        if (first - last).abs() > 1e-15 + 1e-15 * last.abs() {
                            return Err(TableError("periodic table endpoints must agree".into()));
                        }
                    }
                }
            }
        }
        // Store nodal mixed derivatives, avoiding 4^D coefficients per cell.
        let mut nodes = Vec::with_capacity(1 << D);
        nodes.push(values);
        for mask in 1usize..1 << D {
            let axis = mask.trailing_zeros() as usize;
            let source = &nodes[mask ^ (1 << axis)];
            let stride: usize = shape[..axis].iter().product();
            let width = shape[axis];
            let mut output = vec![0.0; source.len()];
            let mut line = vec![0.0; width];
            for block in (0..source.len()).step_by(stride * width) {
                for offset in 0..stride {
                    for i in 0..width {
                        line[i] = source[block + offset + i * stride];
                    }
                    let slopes = spline_slopes(&line, periodic);
                    for i in 0..width {
                        output[block + offset + i * stride] = slopes[i];
                    }
                }
            }
            nodes.push(output);
        }
        Ok(Self {
            shape,
            bounds,
            periodic,
            nodes: Arc::new(nodes),
            update_count: 1,
        })
    }

    pub fn set_function_parameters(
        &mut self,
        shape: [usize; D],
        values: Vec<f64>,
        bounds: [[f64; 2]; D],
    ) -> Result<(), TableError> {
        let mut replacement = Self::new(shape, values, bounds, self.periodic)?;
        replacement.update_count = self.update_count + 1;
        *self = replacement;
        Ok(())
    }

    pub fn shape(&self) -> &[usize; D] {
        &self.shape
    }
    pub fn values(&self) -> &[f64] {
        &self.nodes[0]
    }
    pub fn bounds(&self) -> &[[f64; 2]; D] {
        &self.bounds
    }
    pub fn periodic(&self) -> bool {
        self.periodic
    }
    pub fn update_count(&self) -> u64 {
        self.update_count
    }

    pub fn evaluate(&self, arguments: [f64; D]) -> (f64, [f64; D]) {
        let mut lower = [0usize; D];
        let mut basis = [[0.0; 4]; D];
        let mut derivatives = [[0.0; 4]; D];
        for axis in 0..D {
            let [min, max] = self.bounds[axis];
            let length = max - min;
            let mut x = arguments[axis];
            if self.periodic {
                let s = (x - min) / length;
                x = min + length * (s - s.floor());
            }
            if x < min || x > max {
                return (0.0, [0.0; D]);
            }
            let inv_step = (self.shape[axis] - 1) as f64 / length;
            let position = (x - min) * inv_step;
            lower[axis] = (position.floor() as usize).min(self.shape[axis] - 2);
            let t = position - lower[axis] as f64;
            let t2 = t * t;
            let t3 = t2 * t;
            basis[axis] = [
                2.0 * t3 - 3.0 * t2 + 1.0,
                -2.0 * t3 + 3.0 * t2,
                t3 - 2.0 * t2 + t,
                t3 - t2,
            ];
            derivatives[axis] = [
                6.0 * t2 - 6.0 * t,
                -6.0 * t2 + 6.0 * t,
                3.0 * t2 - 4.0 * t + 1.0,
                3.0 * t2 - 2.0 * t,
            ]
            .map(|v| v * inv_step);
        }
        // Contract one axis at a time, reusing the same stack buffer for all partials.
        let mut partials = [[0.0; 4]; 16];
        let mut count = 1usize << (2 * (D - 1));
        for (term, output) in partials[..count].iter_mut().enumerate() {
            let mut node = lower[0];
            let mut mask = 0;
            let mut stride = self.shape[0];
            for (axis, &start) in lower.iter().enumerate().skip(1) {
                let choice = (term >> (2 * (axis - 1))) & 3;
                node += (start + (choice & 1)) * stride;
                stride *= self.shape[axis];
                mask |= (choice >> 1) << axis;
            }
            for choice in 0..4 {
                let coefficient = self.nodes[mask | (choice >> 1)][node + (choice & 1)];
                output[0] += coefficient * basis[0][choice];
                output[1] += coefficient * derivatives[0][choice];
            }
        }
        for axis in 1..D {
            count /= 4;
            for i in 0..count {
                let mut output = [0.0; 4];
                for choice in 0..4 {
                    let input = partials[4 * i + choice];
                    for component in 0..=axis {
                        output[component] += input[component] * basis[axis][choice];
                    }
                    output[axis + 1] += input[0] * derivatives[axis][choice];
                }
                partials[i] = output;
            }
        }
        (partials[0][0], std::array::from_fn(|i| partials[0][i + 1]))
    }
}

fn check_shape<const D: usize>(
    shape: &[usize; D],
    count: usize,
    minimum: usize,
) -> Result<(), TableError> {
    if !(1..=3).contains(&D) || shape.iter().any(|&size| size < minimum) {
        return Err(TableError(format!(
            "table requires 1-3 dimensions and at least {minimum} points per axis"
        )));
    }
    if shape
        .iter()
        .try_fold(1usize, |n, &size| n.checked_mul(size))
        != Some(count)
    {
        return Err(TableError(
            "table value count does not match its dimensions".into(),
        ));
    }
    Ok(())
}

fn solve_tridiagonal(diagonal: &[f64], rhs: &[f64]) -> Vec<f64> {
    let mut result = rhs.to_vec();
    let mut reciprocal = vec![0.0; rhs.len()];
    reciprocal[0] = 1.0 / diagonal[0];
    result[0] *= reciprocal[0];
    for i in 1..rhs.len() {
        reciprocal[i] = 1.0 / (diagonal[i] - reciprocal[i - 1]);
        result[i] = (result[i] - result[i - 1]) * reciprocal[i];
    }
    for i in (0..rhs.len() - 1).rev() {
        result[i] -= reciprocal[i] * result[i + 1];
    }
    result
}

fn spline_slopes(values: &[f64], periodic: bool) -> Vec<f64> {
    let n = values.len();
    if !periodic {
        let mut diagonal = vec![4.0; n];
        diagonal[0] = 2.0;
        diagonal[n - 1] = 2.0;
        let mut rhs = vec![0.0; n];
        rhs[0] = 3.0 * (values[1] - values[0]);
        rhs[n - 1] = 3.0 * (values[n - 1] - values[n - 2]);
        for i in 1..n - 1 {
            rhs[i] = 3.0 * (values[i + 1] - values[i - 1]);
        }
        return solve_tridiagonal(&diagonal, &rhs);
    }
    let count = n - 1;
    if count == 2 {
        return vec![0.0; n];
    }
    let mut diagonal = vec![4.0; count];
    diagonal[0] = 8.0;
    diagonal[count - 1] = 4.25;
    let rhs: Vec<_> = (0..count)
        .map(|i| 3.0 * (values[(i + 1) % count] - values[(i + count - 1) % count]))
        .collect();
    let mut result = solve_tridiagonal(&diagonal, &rhs);
    let mut update = vec![0.0; count];
    update[0] = -4.0;
    update[count - 1] = 1.0;
    let correction = solve_tridiagonal(&diagonal, &update);
    let scale = (result[0] - 0.25 * result[count - 1])
        / (1.0 + correction[0] - 0.25 * correction[count - 1]);
    for i in 0..count {
        result[i] -= scale * correction[i];
    }
    result.push(result[0]);
    result
}

#[derive(Debug, Clone)]
pub struct DiscreteFunction<const D: usize> {
    shape: [usize; D],
    values: Arc<Vec<f64>>,
    update_count: u64,
}
pub type Discrete1DFunction = DiscreteFunction<1>;
pub type Discrete2DFunction = DiscreteFunction<2>;
pub type Discrete3DFunction = DiscreteFunction<3>;

impl<const D: usize> DiscreteFunction<D> {
    pub fn new(shape: [usize; D], values: Vec<f64>) -> Result<Self, TableError> {
        check_shape(&shape, values.len(), 1)?;
        Ok(Self {
            shape,
            values: Arc::new(values),
            update_count: 0,
        })
    }
    pub fn set_function_parameters(
        &mut self,
        shape: [usize; D],
        values: Vec<f64>,
    ) -> Result<(), TableError> {
        let mut replacement = Self::new(shape, values)?;
        replacement.update_count = self.update_count + 1;
        *self = replacement;
        Ok(())
    }
    pub fn shape(&self) -> &[usize; D] {
        &self.shape
    }
    pub fn values(&self) -> &[f64] {
        &self.values
    }
    pub fn update_count(&self) -> u64 {
        self.update_count
    }
    pub fn evaluate(&self, arguments: [f64; D]) -> (f64, [f64; D]) {
        let mut index = 0;
        let mut stride = 1;
        for (axis, value) in arguments.into_iter().enumerate() {
            index += (value.round().max(0.0) as usize).min(self.shape[axis] - 1) * stride;
            stride *= self.shape[axis];
        }
        (self.values[index], [0.0; D])
    }
}

#[derive(Debug, Clone)]
pub enum TabulatedFunction {
    Continuous1D(Continuous1DFunction),
    Continuous2D(Continuous2DFunction),
    Continuous3D(Continuous3DFunction),
    Discrete1D(Discrete1DFunction),
    Discrete2D(Discrete2DFunction),
    Discrete3D(Discrete3DFunction),
}

impl TabulatedFunction {
    pub fn arity(&self) -> usize {
        match self {
            Self::Continuous1D(_) | Self::Discrete1D(_) => 1,
            Self::Continuous2D(_) | Self::Discrete2D(_) => 2,
            Self::Continuous3D(_) | Self::Discrete3D(_) => 3,
        }
    }
    pub fn evaluate(&self, arguments: [f64; 3]) -> (f64, [f64; 3]) {
        macro_rules! evaluate {
            ($f:expr, $d:literal) => {{
                let (value, gradient) = $f.evaluate(std::array::from_fn(|i| arguments[i]));
                let mut result = [0.0; 3];
                result[..$d].copy_from_slice(&gradient);
                (value, result)
            }};
        }
        match self {
            Self::Continuous1D(f) => evaluate!(f, 1),
            Self::Continuous2D(f) => evaluate!(f, 2),
            Self::Continuous3D(f) => evaluate!(f, 3),
            Self::Discrete1D(f) => evaluate!(f, 1),
            Self::Discrete2D(f) => evaluate!(f, 2),
            Self::Discrete3D(f) => evaluate!(f, 3),
        }
    }
}

macro_rules! conversion {
    ($type:ty, $variant:ident) => {
        impl From<$type> for TabulatedFunction {
            fn from(value: $type) -> Self {
                Self::$variant(value)
            }
        }
    };
}
conversion!(Continuous1DFunction, Continuous1D);
conversion!(Continuous2DFunction, Continuous2D);
conversion!(Continuous3DFunction, Continuous3D);
conversion!(Discrete1DFunction, Discrete1D);
conversion!(Discrete2DFunction, Discrete2D);
conversion!(Discrete3DFunction, Discrete3D);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn natural_line_and_outside_domain() {
        let f = Continuous1DFunction::new([2], vec![2.0, 6.0], [[-1.0, 1.0]], false).unwrap();
        for x in [-1.0, -0.7, 0.1, 1.0] {
            let (value, gradient) = f.evaluate([x]);
            assert!((value - (4.0 + 2.0 * x)).abs() < 1e-14);
            assert!((gradient[0] - 2.0).abs() < 1e-14);
        }
        assert_eq!(f.evaluate([-1.01]), (0.0, [0.0]));
        assert_eq!(f.evaluate([1.01]), (0.0, [0.0]));
    }

    #[test]
    fn periodic_wrapping_and_chain_rule() {
        let f = Continuous2DFunction::new(
            [4, 3],
            vec![1.0, 2.0, 3.0, 1.0, 4.0, 5.0, 2.0, 4.0, 1.0, 2.0, 3.0, 1.0],
            [[-2.0, 2.0], [-1.0, 1.0]],
            true,
        )
        .unwrap();
        for point in [[-2.0, -1.0], [-0.7, 0.4], [1.2, -0.6], [2.0, 1.0]] {
            let (value, gradient) = f.evaluate(point);
            assert!((f.evaluate([point[0] + 8.0, point[1] - 4.0]).0 - value).abs() < 1e-13);
            for axis in 0..2 {
                let mut plus = point;
                let mut minus = point;
                plus[axis] += 1e-6;
                minus[axis] -= 1e-6;
                let numerical = (f.evaluate(plus).0 - f.evaluate(minus).0) / 2e-6;
                assert!((numerical - gradient[axis]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn discrete_rounding_clamping_and_updates() {
        let mut f = Discrete2DFunction::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let snapshot = f.clone();
        assert_eq!(f.evaluate([-0.5, 0.5]), (3.0, [0.0, 0.0]));
        assert_eq!(f.evaluate([0.5, -100.0]), (2.0, [0.0, 0.0]));
        assert_eq!(f.evaluate([100.0, 100.0]), (4.0, [0.0, 0.0]));
        f.set_function_parameters([1, 1], vec![9.0]).unwrap();
        assert_eq!(f.update_count(), 1);
        assert_eq!(f.evaluate([0.0, 0.0]).0, 9.0);
        assert_eq!(snapshot.evaluate([0.0, 0.0]).0, 1.0);
    }
}
