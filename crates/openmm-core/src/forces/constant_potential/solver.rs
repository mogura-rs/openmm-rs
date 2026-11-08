use super::*;

struct Operator<'a> {
    force: &'a ConstantPotentialForce,
    setup: &'a Setup,
    positions: &'a [f64],
    vectors: &'a [[f64; 3]; 3],
    pairs: &'a [(usize, usize, f64, f64)],
}

impl Operator<'_> {
    fn derivatives(&self, particles: &[f64], output: &mut [f64]) {
        output.fill(0.);
        for &(i, j, factor, _) in self.pairs {
            let ii = self.setup.system_to_electrode[i];
            let jj = self.setup.system_to_electrode[j];
            if ii != NONE {
                output[ii] += particles[j] * factor;
            }
            if jj != NONE {
                output[jj] += particles[i] * factor;
            }
        }
        self.force.pme.charge_derivatives(
            self.positions,
            particles,
            self.vectors,
            self.setup.pme,
            &self.setup.electrode_to_system,
            output,
        );
        let volume = reciprocal_box(self.vectors).1;
        let alpha = self.setup.pme.0;
        let total_charge = particles.iter().sum::<f64>();
        let plasma = PI * ONE_4PI_EPS0 * total_charge / (volume * alpha * alpha);
        for (ii, &i) in self.setup.electrode_to_system.iter().enumerate() {
            let [potential, width, tf] = self.setup.parameters[ii];
            let diagonal = 2. * ONE_4PI_EPS0 / (2. * PI).sqrt() / width
                - 2. * ONE_4PI_EPS0 / PI.sqrt() * alpha
                + 4. * PI * ONE_4PI_EPS0 * tf;
            let field = (0..3)
                .map(|axis| self.positions[3 * i + axis] * self.force.external_field[axis])
                .sum::<f64>();
            output[ii] += particles[i] * diagonal - plasma - potential - field;
        }
    }
}

fn packed(i: usize, j: usize) -> usize {
    i * (i + 1) / 2 + j
}
fn cholesky(matrix: &mut [f64], n: usize) {
    for i in 0..n {
        for j in 0..=i {
            let mut value = matrix[packed(i, j)];
            for k in 0..j {
                value -= matrix[packed(i, k)] * matrix[packed(j, k)];
            }
            if i == j {
                assert!(
                    value > 0. && value.is_finite(),
                    "electrode matrix is not positive definite"
                );
                matrix[packed(i, j)] = value.sqrt();
            } else {
                matrix[packed(i, j)] = value / matrix[packed(j, j)];
            }
        }
    }
}
fn solve_cholesky(matrix: &[f64], values: &mut [f64]) {
    let n = values.len();
    for i in 0..n {
        let mut value = values[i];
        for j in 0..i {
            value -= matrix[packed(i, j)] * values[j];
        }
        values[i] = value / matrix[packed(i, i)];
    }
    for i in (0..n).rev() {
        let mut value = values[i];
        for j in i + 1..n {
            value -= matrix[packed(j, i)] * values[j];
        }
        values[i] = value / matrix[packed(i, i)];
    }
}
fn norm2(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum()
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn project(input: &[f64], output: &mut [f64], constrained: bool) {
    let mean = if constrained {
        input.iter().sum::<f64>() / input.len() as f64
    } else {
        0.
    };
    for (output, &input) in output.iter_mut().zip(input) {
        *output = input - mean;
    }
}
fn precondition(
    gradient: &[f64],
    projected: &[f64],
    diagonal: Option<&[f64]>,
    output: &mut [f64],
    constrained: bool,
) {
    if let Some(diagonal) = diagonal {
        let offset = if constrained {
            dot(diagonal, gradient)
        } else {
            0.
        };
        for ((output, &gradient), &diagonal) in output.iter_mut().zip(gradient).zip(diagonal) {
            *output = diagonal * (gradient - offset);
        }
    } else {
        output.copy_from_slice(projected);
    }
}

impl ConstantPotentialForce {
    pub(super) fn solve(
        &self,
        positions: &[f64],
        vectors: &[[f64; 3]; 3],
        setup: &Setup,
        work: &mut Work,
    ) {
        let count = setup.electrode_to_system.len();
        if count == 0 {
            return;
        }
        let Work {
            charges: particles,
            pairs,
            matrix,
            constraint,
            preconditioner,
            electrode_positions,
            box_vectors,
            solver_valid,
            q,
            gradient,
            projected,
            preconditioned,
            step,
            product,
            baseline,
            ..
        } = work;
        for buffer in [
            &mut *q,
            &mut *gradient,
            &mut *projected,
            &mut *preconditioned,
            &mut *step,
            &mut *product,
            &mut *baseline,
        ] {
            buffer.resize(count, 0.);
        }
        let operator = Operator {
            force: self,
            setup,
            positions,
            vectors,
            pairs,
        };
        let electrode_moved = self.method == ConstantPotentialMethod::Matrix
            && (electrode_positions.len() != 3 * count
                || setup
                    .electrode_to_system
                    .iter()
                    .enumerate()
                    .any(|(ii, &i)| {
                        electrode_positions[3 * ii..3 * ii + 3] != positions[3 * i..3 * i + 3]
                    }));
        if !*solver_valid || box_vectors.as_ref() != Some(vectors) || electrode_moved {
            if self.method == ConstantPotentialMethod::Matrix {
                matrix.resize(
                    count
                        .checked_mul(count + 1)
                        .expect("electrode matrix size overflow")
                        / 2,
                    0.,
                );
                electrode_positions.resize(3 * count, 0.);
                for (ii, &i) in setup.electrode_to_system.iter().enumerate() {
                    electrode_positions[3 * ii..3 * ii + 3]
                        .copy_from_slice(&positions[3 * i..3 * i + 3]);
                    particles[i] = 0.;
                }
                operator.derivatives(particles, baseline);
                for (ii, &i) in setup.electrode_to_system.iter().enumerate() {
                    particles[i] = 1.;
                    operator.derivatives(particles, gradient);
                    particles[i] = 0.;
                    for jj in 0..=ii {
                        matrix[packed(ii, jj)] = gradient[jj] - baseline[jj];
                    }
                }
                cholesky(matrix, count);
                if self.charge_constraint {
                    constraint.resize(count, 1.);
                    constraint.fill(1.);
                    solve_cholesky(matrix, constraint);
                    let sum = constraint.iter().sum::<f64>();
                    for value in constraint.iter_mut() {
                        *value /= sum;
                    }
                }
            } else if self.preconditioner {
                preconditioner.resize(count, 0.);
                let mut pme_diagonal = [0.];
                self.pme.charge_derivatives(
                    &[0.; 3],
                    &[1.],
                    vectors,
                    setup.pme,
                    &[0],
                    &mut pme_diagonal,
                );
                let alpha = setup.pme.0;
                let volume = reciprocal_box(vectors).1;
                let ewald = pme_diagonal[0]
                    - 2. * ONE_4PI_EPS0 / PI.sqrt() * alpha
                    - PI * ONE_4PI_EPS0 / (volume * alpha * alpha);
                for (value, &[_, width, tf]) in preconditioner.iter_mut().zip(&setup.parameters) {
                    *value = (2. * ONE_4PI_EPS0 / (2. * PI).sqrt() / width
                        + 4. * PI * ONE_4PI_EPS0 * tf
                        + ewald)
                        .recip();
                }
                let sum = preconditioner.iter().sum::<f64>();
                for value in preconditioner.iter_mut() {
                    *value /= sum;
                }
            }
            *box_vectors = Some(*vectors);
            *solver_valid = true;
        }
        if self.method == ConstantPotentialMethod::Matrix {
            for &i in &setup.electrode_to_system {
                particles[i] = 0.;
            }
            operator.derivatives(particles, q);
            for value in q.iter_mut() {
                *value = -*value;
            }
            solve_cholesky(matrix, q);
            let offset = if self.charge_constraint {
                setup.target - q.iter().sum::<f64>()
            } else {
                0.
            };
            for (ii, &i) in setup.electrode_to_system.iter().enumerate() {
                particles[i] = q[ii]
                    + if self.charge_constraint {
                        offset * constraint[ii]
                    } else {
                        0.
                    };
            }
            return;
        }
        let target = self.cg_tolerance * self.cg_tolerance * count as f64;
        if self.charge_constraint {
            let offset = (setup.target
                - setup
                    .electrode_to_system
                    .iter()
                    .map(|&i| particles[i])
                    .sum::<f64>())
                / count as f64;
            for &i in &setup.electrode_to_system {
                particles[i] += offset;
            }
        }
        operator.derivatives(particles, gradient);
        project(gradient, projected, self.charge_constraint);
        if norm2(projected) <= target {
            return;
        }
        for (ii, &i) in setup.electrode_to_system.iter().enumerate() {
            q[ii] = particles[i];
            particles[i] = 0.;
        }
        operator.derivatives(particles, baseline);
        let diagonal = self.preconditioner.then_some(preconditioner.as_slice());
        precondition(
            gradient,
            projected,
            diagonal,
            preconditioned,
            self.charge_constraint,
        );
        for (step, &gradient) in step.iter_mut().zip(preconditioned.iter()) {
            *step = -gradient;
        }
        let mut converged = false;
        for iteration in 0..=count {
            for (&i, &step) in setup.electrode_to_system.iter().zip(step.iter()) {
                particles[i] = step;
            }
            operator.derivatives(particles, product);
            for (value, &base) in product.iter_mut().zip(baseline.iter()) {
                *value -= base;
            }
            if norm2(product) <= target {
                converged = true;
                break;
            }
            let scale = dot(step, product).recip();
            let alpha = -dot(step, gradient) * scale;
            for (q, &step) in q.iter_mut().zip(step.iter()) {
                *q += alpha * step;
            }
            if self.charge_constraint {
                let offset = (setup.target - q.iter().sum::<f64>()) / count as f64;
                for q in q.iter_mut() {
                    *q += offset;
                }
            }
            if iteration != 0 && iteration % 32 == 0 {
                for (&i, &q) in setup.electrode_to_system.iter().zip(q.iter()) {
                    particles[i] = q;
                }
                operator.derivatives(particles, gradient);
            } else {
                for (gradient, &product) in gradient.iter_mut().zip(product.iter()) {
                    *gradient += alpha * product;
                }
            }
            project(gradient, projected, self.charge_constraint);
            if norm2(projected) <= target {
                converged = true;
                break;
            }
            precondition(
                gradient,
                projected,
                diagonal,
                preconditioned,
                self.charge_constraint,
            );
            let beta = dot(preconditioned, product) * scale;
            for (step, &gradient) in step.iter_mut().zip(preconditioned.iter()) {
                *step = beta * *step - gradient;
            }
            if self.charge_constraint {
                let offset = step.iter().sum::<f64>() / count as f64;
                for value in step.iter_mut() {
                    *value -= offset;
                }
            }
        }
        assert!(
            converged,
            "constant-potential conjugate gradient iterations did not converge"
        );
        for (&i, &q) in setup.electrode_to_system.iter().zip(q.iter()) {
            particles[i] = q;
        }
    }
}
