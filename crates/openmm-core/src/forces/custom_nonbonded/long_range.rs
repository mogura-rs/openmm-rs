use super::CustomNonbondedForce;
use crate::forces::custom_evaluator::CustomEvaluator;
use std::{collections::HashMap, f64::consts::TAU, sync::Mutex};

#[derive(Default)]
pub(super) struct CorrectionCache(Mutex<Option<Vec<f64>>>);

impl Clone for CorrectionCache {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl CorrectionCache {
    pub fn clear(&mut self) {
        *self.0.get_mut().unwrap_or_else(|error| error.into_inner()) = None;
    }

    pub fn add(
        &self,
        force: &CustomNonbondedForce,
        volume: f64,
        energy: &mut f64,
        derivatives: &mut HashMap<String, f64>,
    ) {
        let mut cache = self.0.lock().unwrap_or_else(|error| error.into_inner());
        let coefficient = cache.get_or_insert_with(|| coefficients(force));
        *energy += coefficient[0] / volume;
        for (i, name) in force.energy_param_deriv_names.iter().enumerate() {
            *derivatives.entry(name.clone()).or_default() += coefficient[i + 1] / volume;
        }
    }
}

fn coefficients(force: &CustomNonbondedForce) -> Vec<f64> {
    let components = 1 + force.energy_param_deriv_names.len();
    let mut result = vec![0.0; components];
    let n = force.particles.len();
    if n == 0 {
        return result;
    }
    assert!(
        force.energy_expr.variable_index("r").is_some(),
        "long range correction requires an energy that depends on r"
    );
    let mut classes = Vec::new();
    let mut counts = Vec::new();
    let mut indexes = HashMap::new();
    let mut particle_classes = Vec::with_capacity(n);
    for parameters in &force.particles {
        let key: Vec<u64> = parameters
            .iter()
            .map(|&v| if v == 0.0 { 0 } else { v.to_bits() })
            .collect();
        let index = *indexes.entry(key).or_insert_with(|| {
            counts.push(0u64);
            classes.push(parameters.clone());
            classes.len() - 1
        });
        counts[index] += 1;
        particle_classes.push(index);
    }
    let mut group_counts = HashMap::<(usize, usize), u64>::new();
    for (set1, set2) in &force.interaction_groups {
        for &i in set1 {
            for &j in set2 {
                if i >= j && set1.contains(&j) && set2.contains(&i) {
                    continue;
                }
                let a = particle_classes[i];
                let b = particle_classes[j];
                *group_counts.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
    }
    let computed = force.compute_values(&classes);
    let count = force.computed_values.len();
    let mut evaluator = CustomEvaluator::new(
        &force.energy_expr,
        &force.global_params,
        force.parameter_names(),
        &["r"],
        &force.energy_param_deriv_names,
    );
    for i in 0..classes.len() {
        for j in i..classes.len() {
            let interactions = if force.interaction_groups.is_empty() {
                if i == j {
                    counts[i] * (counts[i] + 1) / 2
                } else {
                    counts[i] * counts[j]
                }
            } else {
                *group_counts.get(&(i, j)).unwrap_or(&0)
            };
            if interactions == 0 {
                continue;
            }
            evaluator.set_parameters(
                classes[i]
                    .iter()
                    .zip(&classes[j])
                    .flat_map(|(&a, &b)| [a, b])
                    .chain(
                        (0..count).flat_map(|k| [computed[i * count + k], computed[j * count + k]]),
                    ),
            );
            let mut integral = integrate(components, |x, values| {
                let r = force.cutoff_distance / x;
                evaluator.set_coordinates(&[r]);
                values[0] = evaluator.evaluate_scaled(0.0);
                for (i, value) in values.iter_mut().enumerate().skip(1) {
                    *value = evaluator.parameter_derivative(i - 1);
                }
                let factor = r.powi(4) / force.cutoff_distance;
                for value in values {
                    *value *= factor;
                }
            });
            if force.use_switching_function {
                let width = force.cutoff_distance - force.switching_distance;
                let correction = integrate(components, |x, values| {
                    let r = force.switching_distance + x * width;
                    evaluator.set_coordinates(&[r]);
                    values[0] = evaluator.evaluate_scaled(0.0);
                    for (i, value) in values.iter_mut().enumerate().skip(1) {
                        *value = evaluator.parameter_derivative(i - 1);
                    }
                    let factor = width * x * x * x * (10.0 + x * (-15.0 + 6.0 * x)) * r * r;
                    for value in values {
                        *value *= factor;
                    }
                });
                for (value, correction) in integral.iter_mut().zip(correction) {
                    *value += correction;
                }
            }
            for (total, value) in result.iter_mut().zip(integral) {
                *total += interactions as f64 * value;
            }
        }
    }
    let n = n as f64;
    let scale = TAU * n * n / (n * (n + 1.0) / 2.0);
    for value in &mut result {
        *value *= scale;
    }
    result
}

fn integrate(components: usize, mut evaluate: impl FnMut(f64, &mut [f64])) -> Vec<f64> {
    let mut sum = vec![0.0; components];
    let mut additions = vec![0.0; components];
    let mut values = vec![0.0; components];
    let mut active = vec![true; components];
    let mut points = 1usize;
    for iteration in 0..=10 {
        additions.fill(0.0);
        for i in 0..points {
            if i % 3 == 1 {
                continue;
            }
            evaluate((i as f64 + 0.5) / points as f64, &mut values);
            for k in 0..components {
                if active[k] {
                    additions[k] += values[k];
                }
            }
        }
        for k in 0..components {
            if !active[k] {
                continue;
            }
            let previous = sum[k];
            sum[k] = additions[k] / points as f64 + previous / 3.0;
            let change = ((sum[k] - previous) / sum[k]).abs();
            if iteration > 2 && (change < 1e-5 || sum[k] == 0.0) {
                active[k] = false;
            } else {
                assert!(sum[k].is_finite() && iteration < 10 && !(iteration > 7 && change > 1e-3), "long range correction did not converge: the energy must be finite in the switching interval and decay fast enough for a finite integral");
            }
        }
        if active.iter().all(|&value| !value) {
            return sum;
        }
        points *= 3;
    }
    unreachable!()
}
