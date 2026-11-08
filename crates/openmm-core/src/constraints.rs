use burn::prelude::*;
use std::fmt;

/// A fixed distance in nm. Constraint distances do not use periodic wrapping.
#[derive(Debug, Clone, Copy)]
pub struct Constraint {
    pub particle1: usize,
    pub particle2: usize,
    pub distance: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConstraintError(pub String);

impl fmt::Display for ConstraintError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConstraintError {}

struct PreparedConstraint {
    indices: [usize; 2],
    inverse_masses: [f64; 2],
    distance_squared: f64,
}

/// Sparse SHAKE/RATTLE projection with storage linear in the constraint count.
pub struct ConstraintSolver {
    constraints: Vec<PreparedConstraint>,
}

impl ConstraintSolver {
    pub fn new(constraints: &[Constraint], masses: &[f64]) -> Result<Self, ConstraintError> {
        let mut prepared = Vec::with_capacity(constraints.len());
        let mut pairs = std::collections::HashSet::with_capacity(constraints.len());
        for constraint in constraints {
            let Constraint {
                particle1: i,
                particle2: j,
                distance,
            } = *constraint;
            if i >= masses.len() || j >= masses.len() || i == j {
                return Err(ConstraintError(
                    "constraint particle indices must be distinct and valid".into(),
                ));
            }
            if !distance.is_finite() || distance <= 0.0 {
                return Err(ConstraintError(
                    "constraint distance must be positive and finite".into(),
                ));
            }
            if !pairs.insert((i.min(j), i.max(j))) {
                return Err(ConstraintError(
                    "duplicate constraint between the same particles".into(),
                ));
            }
            if (masses[i] == 0.0) != (masses[j] == 0.0) {
                return Err(ConstraintError(
                    "a constraint cannot connect a massive and massless particle".into(),
                ));
            }
            if masses[i] != 0.0 {
                prepared.push(PreparedConstraint {
                    indices: [i, j],
                    inverse_masses: [masses[i].recip(), masses[j].recip()],
                    distance_squared: distance * distance,
                });
            }
        }
        Ok(Self {
            constraints: prepared,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    pub(crate) fn solve(
        &self,
        reference: &[f64],
        values: &mut [f64],
        velocities: bool,
        tolerance: f64,
    ) -> Result<(), ConstraintError> {
        if !tolerance.is_finite() || tolerance <= 0.0 || tolerance >= 1.0 {
            return Err(ConstraintError(
                "constraint tolerance must be finite and between zero and one".into(),
            ));
        }
        let directions: Vec<[f64; 3]> = self
            .constraints
            .iter()
            .map(|constraint| {
                let [i, j] = constraint.indices;
                std::array::from_fn(|axis| reference[3 * i + axis] - reference[3 * j + axis])
            })
            .collect();
        for _ in 0..1000 {
            let mut converged = true;
            for (constraint, direction) in self.constraints.iter().zip(&directions) {
                let [i, j] = constraint.indices;
                let [wi, wj] = constraint.inverse_masses;
                let delta: [f64; 3] =
                    std::array::from_fn(|axis| values[3 * i + axis] - values[3 * j + axis]);
                let dot: f64 = delta.iter().zip(direction).map(|(a, b)| a * b).sum();
                let multiplier = if velocities {
                    let norm: f64 = direction.iter().map(|v| v * v).sum();
                    if norm == 0.0 || !norm.is_finite() {
                        return Err(ConstraintError(
                            "constraint has coincident or nonfinite reference positions".into(),
                        ));
                    }
                    let correction = -dot / ((wi + wj) * norm);
                    if correction.abs() <= tolerance {
                        continue;
                    }
                    correction
                } else {
                    let norm: f64 = delta.iter().map(|v| v * v).sum();
                    if (norm / constraint.distance_squared).sqrt().is_finite()
                        && ((norm / constraint.distance_squared).sqrt() - 1.0).abs() <= tolerance
                    {
                        continue;
                    }
                    let denominator = 2.0 * (wi + wj) * dot;
                    if denominator.abs() < f64::MIN_POSITIVE || !denominator.is_finite() {
                        return Err(ConstraintError(
                            "constraint projection is singular; check positions and step size"
                                .into(),
                        ));
                    }
                    (constraint.distance_squared - norm) / denominator
                };
                if !multiplier.is_finite() {
                    return Err(ConstraintError(
                        "constraint projection produced a nonfinite correction".into(),
                    ));
                }
                converged = false;
                for axis in 0..3 {
                    let correction = multiplier * direction[axis];
                    values[3 * i + axis] += wi * correction;
                    values[3 * j + axis] -= wj * correction;
                }
            }
            if converged {
                return Ok(());
            }
        }
        Err(ConstraintError(
            "constraints failed to converge in 1000 iterations".into(),
        ))
    }

    fn project<B: Backend>(
        &self,
        reference: &Tensor<B, 2>,
        values: &mut Tensor<B, 2>,
        velocities: bool,
        tolerance: f64,
    ) -> Result<(), ConstraintError> {
        if self.is_empty() {
            return Ok(());
        }
        let read = |tensor: &Tensor<B, 2>| -> Vec<f64> {
            tensor.to_data().convert::<f64>().to_vec::<f64>().unwrap()
        };
        let reference = read(reference);
        let mut data = read(values);
        self.solve(&reference, &mut data, velocities, tolerance)?;

        *values = crate::tensor::from_floats::<B, 1>(data.as_slice(), &values.device())
            .reshape(values.dims());
        Ok(())
    }

    pub fn apply_positions<B: Backend>(
        &self,
        reference: &Tensor<B, 2>,
        positions: &mut Tensor<B, 2>,
        tolerance: f64,
    ) -> Result<(), ConstraintError> {
        self.project(reference, positions, false, tolerance)
    }

    pub fn apply_velocities<B: Backend>(
        &self,
        positions: &Tensor<B, 2>,
        velocities: &mut Tensor<B, 2>,
        tolerance: f64,
    ) -> Result<(), ConstraintError> {
        self.project(positions, velocities, true, tolerance)
    }
}
