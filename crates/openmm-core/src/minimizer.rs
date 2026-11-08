use crate::{Constraint, ConstraintError, Context, StateFlags};
use burn::prelude::*;
use std::collections::VecDeque;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinimizationTermination {
    Converged,
    IterationLimit,
    LineSearchFailed,
    ReporterStopped,
    ConstraintStalled,
}

#[derive(Debug)]
pub struct MinimizationResult {
    pub iterations: usize,
    pub energy: f64,
    pub force_rms: f64,
    pub max_constraint_error: f64,
    pub termination: MinimizationTermination,
}

#[derive(Debug)]
pub enum MinimizationError {
    InvalidTolerance,
    NonFiniteInitialEnergy,
    Constraint(ConstraintError),
}

impl fmt::Display for MinimizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTolerance => write!(
                formatter,
                "minimization tolerance must be positive and finite"
            ),
            Self::NonFiniteInitialEnergy => write!(
                formatter,
                "initial minimization energy or gradient is not finite"
            ),
            Self::Constraint(error) => write!(formatter, "{error}"),
        }
    }
}
impl std::error::Error for MinimizationError {}

#[derive(Debug)]
pub struct MinimizationProgress {
    pub system_energy: f64,
    pub restraint_energy: f64,
    pub restraint_strength: f64,
    pub max_constraint_error: f64,
}

pub trait MinimizationReporter {
    /// Return true to end the current optimization stage.
    fn report(
        &mut self,
        iteration: usize,
        positions: &[f64],
        gradient: &[f64],
        progress: &MinimizationProgress,
    ) -> bool;
}

pub struct LocalEnergyMinimizer;

struct Evaluation {
    energy: f64,
    gradient: Vec<f64>,
    progress: MinimizationProgress,
}

struct Objective<'a, B: Backend> {
    context: &'a mut Context<B>,
    masses: Vec<f64>,
    constraints: Vec<Constraint>,
    strength: f64,
    device: B::Device,
}

impl<B: Backend> Objective<'_, B> {
    fn evaluate(&mut self, positions: &[f64]) -> Evaluation {
        let data = positions;
        self.context.set_positions(
            crate::tensor::from_floats::<B, 1>(data, &self.device).reshape([self.masses.len(), 3]),
        );
        self.context.compute_virtual_sites();
        let groups = self.context.integrator().integration_force_groups();
        let (forces, system_energy) = self.context.calc_forces_and_energy_groups(groups);
        let mut gradient: Vec<f64> = forces
            .to_data()
            .convert::<f64>()
            .to_vec::<f64>()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, &force)| {
                if self.masses[i / 3] == 0.0 {
                    0.0
                } else {
                    -force
                }
            })
            .collect();
        let mut restraint_energy = 0.0;
        let mut max_constraint_error: f64 = 0.0;
        for constraint in &self.constraints {
            let (a, b) = (constraint.particle1, constraint.particle2);
            let delta: [f64; 3] =
                std::array::from_fn(|axis| positions[3 * b + axis] - positions[3 * a + axis]);
            let distance = dot(&delta, &delta).sqrt();
            let error = distance - constraint.distance;
            max_constraint_error = max_constraint_error.max(error.abs() / constraint.distance);
            restraint_energy += 0.5 * self.strength * error * error;
            let scale = self.strength * error / distance;
            for axis in 0..3 {
                if self.masses[a] != 0.0 {
                    gradient[3 * a + axis] -= scale * delta[axis];
                }
                if self.masses[b] != 0.0 {
                    gradient[3 * b + axis] += scale * delta[axis];
                }
            }
        }
        Evaluation {
            energy: system_energy + restraint_energy,
            gradient,
            progress: MinimizationProgress {
                system_energy,
                restraint_energy,
                restraint_strength: self.strength,
                max_constraint_error,
            },
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

struct History {
    displacement: Vec<f64>,
    gradient_change: Vec<f64>,
    inverse_curvature: f64,
}

impl LocalEnergyMinimizer {
    pub fn minimize<B: Backend>(
        context: &mut Context<B>,
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<MinimizationResult, MinimizationError> {
        Self::minimize_with_reporter(context, tolerance, max_iterations, None)
    }

    pub fn minimize_with_reporter<B: Backend>(
        context: &mut Context<B>,
        tolerance: f64,
        max_iterations: usize,
        mut reporter: Option<&mut dyn MinimizationReporter>,
    ) -> Result<MinimizationResult, MinimizationError> {
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(MinimizationError::InvalidTolerance);
        }
        let constraint_tolerance = context.integrator().constraint_tolerance();
        let working_tolerance = constraint_tolerance.max(1e-4);
        context
            .apply_constraints(working_tolerance)
            .map_err(MinimizationError::Constraint)?;
        let initial_tensor = context
            .get_state(StateFlags {
                positions: true,
                ..Default::default()
            })
            .positions
            .unwrap();
        let initial: Vec<f64> = initial_tensor
            .to_data()
            .convert::<f64>()
            .to_vec::<f64>()
            .unwrap();
        let mut positions = initial.clone();
        let norm = (dot(&initial, &initial) / (initial.len() / 3) as f64)
            .sqrt()
            .max(1.0);
        let epsilon = tolerance / norm;
        let mut objective = Objective {
            masses: context.system().masses(),
            constraints: context.system().constraints().to_vec(),
            strength: 100.0 / working_tolerance,
            device: initial_tensor.device(),
            context,
        };
        let mut total_iterations = 0;
        let mut previous_errors = [f64::INFINITY; 2];
        let termination = loop {
            let mut evaluation = objective.evaluate(&positions);
            if !evaluation.energy.is_finite() || evaluation.gradient.iter().any(|x| !x.is_finite())
            {
                return Err(MinimizationError::NonFiniteInitialEnergy);
            }
            let mut history: VecDeque<History> = VecDeque::new();
            let mut iteration = 0;
            let stage_termination = loop {
                let gradient_norm = dot(&evaluation.gradient, &evaluation.gradient).sqrt();
                if gradient_norm <= epsilon * dot(&positions, &positions).sqrt().max(1.0) {
                    break MinimizationTermination::Converged;
                }
                if max_iterations > 0 && iteration >= max_iterations {
                    break MinimizationTermination::IterationLimit;
                }
                let mut direction = evaluation.gradient.clone();
                let mut coefficients = Vec::with_capacity(history.len());
                for entry in history.iter().rev() {
                    let coefficient =
                        entry.inverse_curvature * dot(&entry.displacement, &direction);
                    coefficients.push(coefficient);
                    for (d, y) in direction.iter_mut().zip(&entry.gradient_change) {
                        *d -= coefficient * y;
                    }
                }
                if let Some(last) = history.back() {
                    let scale = 1.0
                        / (last.inverse_curvature
                            * dot(&last.gradient_change, &last.gradient_change));
                    for d in &mut direction {
                        *d *= scale;
                    }
                }
                for (entry, coefficient) in history.iter().zip(coefficients.into_iter().rev()) {
                    let beta = entry.inverse_curvature * dot(&entry.gradient_change, &direction);
                    for (d, s) in direction.iter_mut().zip(&entry.displacement) {
                        *d += (coefficient - beta) * s;
                    }
                }
                for d in &mut direction {
                    *d = -*d;
                }
                let mut slope = dot(&direction, &evaluation.gradient);
                if slope >= 0.0 || !slope.is_finite() {
                    history.clear();
                    direction.clone_from(&evaluation.gradient);
                    for d in &mut direction {
                        *d = -*d;
                    }
                    slope = -gradient_norm * gradient_norm;
                }
                let mut step = if iteration == 0 {
                    1.0 / dot(&direction, &direction).sqrt()
                } else {
                    1.0
                };
                let mut accepted = None;
                for _ in 0..40 {
                    let trial: Vec<f64> = positions
                        .iter()
                        .zip(&direction)
                        .map(|(x, d)| x + step * d)
                        .collect();
                    let candidate = objective.evaluate(&trial);
                    if !candidate.energy.is_finite()
                        || candidate.gradient.iter().any(|x| !x.is_finite())
                        || candidate.energy > evaluation.energy + 1e-4 * step * slope
                    {
                        step *= 0.5;
                    } else {
                        let trial_slope = dot(&candidate.gradient, &direction);
                        if trial_slope < 0.9 * slope {
                            step *= 2.1;
                        } else if trial_slope > -0.9 * slope {
                            step *= 0.5;
                        } else {
                            accepted = Some((trial, candidate));
                            break;
                        }
                    }
                    if !(1e-20..=1e20).contains(&step) {
                        break;
                    }
                }
                let Some((trial, candidate)) = accepted else {
                    objective.evaluate(&positions);
                    break MinimizationTermination::LineSearchFailed;
                };
                let displacement: Vec<f64> =
                    trial.iter().zip(&positions).map(|(a, b)| a - b).collect();
                let gradient_change: Vec<f64> = candidate
                    .gradient
                    .iter()
                    .zip(&evaluation.gradient)
                    .map(|(a, b)| a - b)
                    .collect();
                let curvature = dot(&displacement, &gradient_change);
                if curvature > 1e-16 && curvature.is_finite() {
                    if history.len() == 6 {
                        history.pop_front();
                    }
                    history.push_back(History {
                        displacement,
                        gradient_change,
                        inverse_curvature: 1.0 / curvature,
                    });
                }
                positions = trial;
                evaluation = candidate;
                total_iterations += 1;
                let stopped = reporter.as_mut().is_some_and(|reporter| {
                    reporter.report(
                        iteration,
                        &positions,
                        &evaluation.gradient,
                        &evaluation.progress,
                    )
                });
                iteration += 1;
                if stopped {
                    break MinimizationTermination::ReporterStopped;
                }
            };
            let error = evaluation.progress.max_constraint_error;
            if error <= working_tolerance {
                break stage_termination;
            }
            if error >= previous_errors[1] || !objective.strength.is_finite() {
                positions.clone_from(&initial);
                objective.evaluate(&positions);
                break MinimizationTermination::ConstraintStalled;
            }
            previous_errors = [error, previous_errors[0]];
            objective.strength *= 10.0;
            if error > 100.0 * working_tolerance {
                positions.clone_from(&initial);
            }
        };
        if constraint_tolerance < working_tolerance {
            objective
                .context
                .apply_constraints(constraint_tolerance)
                .map_err(MinimizationError::Constraint)?;
        }
        let groups = objective.context.integrator().integration_force_groups();
        let state = objective.context.get_state_groups(
            StateFlags {
                positions: true,
                energy: true,
                forces: true,
                ..Default::default()
            },
            groups,
        );
        let forces = state
            .forces
            .unwrap()
            .to_data()
            .convert::<f64>()
            .to_vec::<f64>()
            .unwrap();
        let force_squared: f64 = forces
            .iter()
            .enumerate()
            .filter(|(i, _)| objective.masses[i / 3] != 0.0)
            .map(|(_, &x)| x.powi(2))
            .sum();
        let mobile = objective.masses.iter().filter(|&&mass| mass != 0.0).count();
        let final_positions = state
            .positions
            .unwrap()
            .to_data()
            .convert::<f64>()
            .to_vec::<f64>()
            .unwrap();
        let max_constraint_error = objective
            .constraints
            .iter()
            .map(|c| {
                let distance = (0..3)
                    .map(|d| {
                        (final_positions[3 * c.particle1 + d]
                            - final_positions[3 * c.particle2 + d])
                            .powi(2)
                    })
                    .sum::<f64>()
                    .sqrt();
                (distance - c.distance).abs() / c.distance
            })
            .fold(0.0, f64::max);
        Ok(MinimizationResult {
            iterations: total_iterations,
            energy: state.potential_energy.unwrap(),
            force_rms: (force_squared / (3 * mobile).max(1) as f64).sqrt(),
            max_constraint_error,
            termination,
        })
    }
}
