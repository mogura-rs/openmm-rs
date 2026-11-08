use super::nonbonded::{ParticleParam, ONE_4PI_EPS0};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

type Complex64 = Complex<f64>;

#[derive(Default)]
pub(crate) struct PmeCache {
    workspace: Mutex<Option<Workspace>>,
    dispersion: bool,
}

impl Clone for PmeCache {
    fn clone(&self) -> Self {
        Self {
            workspace: Mutex::default(),
            dispersion: self.dispersion,
        }
    }
}

impl PmeCache {
    pub(crate) fn dispersion() -> Self {
        Self {
            workspace: Mutex::default(),
            dispersion: true,
        }
    }

    pub fn evaluate(
        &self,
        positions: &[f64],
        particles: &[ParticleParam],
        box_vectors: &[[f64; 3]; 3],
        alpha: f64,
        dimensions: [usize; 3],
        forces: &mut [f64],
    ) -> f64 {
        let mut guard = self.workspace.lock().unwrap();
        if guard
            .as_ref()
            .is_none_or(|workspace| workspace.dimensions != dimensions)
        {
            *guard = Some(Workspace::new(dimensions));
        }
        guard.as_mut().unwrap().evaluate(
            positions,
            particles.iter().map(|particle| {
                if self.dispersion {
                    2.0 * particle.epsilon.sqrt() * particle.sigma.powi(3)
                } else {
                    particle.charge
                }
            }),
            box_vectors,
            alpha,
            forces,
            self.dispersion,
        )
    }

    pub(crate) fn evaluate_charges(
        &self,
        positions: &[f64],
        charges: &[f64],
        box_vectors: &[[f64; 3]; 3],
        parameters: (f64, [usize; 3]),
        forces: &mut [f64],
    ) -> f64 {
        let (alpha, dimensions) = parameters;
        let mut guard = self.workspace.lock().unwrap();
        if guard
            .as_ref()
            .is_none_or(|work| work.dimensions != dimensions)
        {
            *guard = Some(Workspace::new(dimensions));
        }
        guard.as_mut().unwrap().evaluate(
            positions,
            charges.iter().copied(),
            box_vectors,
            alpha,
            forces,
            false,
        )
    }

    pub(crate) fn charge_derivatives(
        &self,
        positions: &[f64],
        charges: &[f64],
        box_vectors: &[[f64; 3]; 3],
        parameters: (f64, [usize; 3]),
        indices: &[usize],
        derivatives: &mut [f64],
    ) {
        assert!(
            !self.dispersion,
            "charge derivatives require electrostatic PME"
        );
        assert_eq!(indices.len(), derivatives.len());
        let (alpha, dimensions) = parameters;
        let mut guard = self.workspace.lock().unwrap();
        if guard
            .as_ref()
            .is_none_or(|work| work.dimensions != dimensions)
        {
            *guard = Some(Workspace::new(dimensions));
        }
        let work = guard.as_mut().unwrap();
        work.potential(
            positions,
            charges.iter().copied(),
            box_vectors,
            alpha,
            false,
        );
        let reciprocal = reciprocal_box(box_vectors).0;
        let [nx, ny, nz] = dimensions;
        for (&particle, derivative) in indices.iter().zip(derivatives) {
            let (base, weights, _) = stencil(
                &positions[3 * particle..3 * particle + 3],
                &reciprocal,
                dimensions,
            );
            for ix in 0..5 {
                for iy in 0..5 {
                    for iz in 0..5 {
                        let index = (((base[0] + ix) % nx) * ny + (base[1] + iy) % ny) * nz
                            + (base[2] + iz) % nz;
                        *derivative +=
                            work.grid[index].re * weights[0][ix] * weights[1][iy] * weights[2][iz];
                    }
                }
            }
        }
    }
}

struct Workspace {
    dimensions: [usize; 3],
    grid: Vec<Complex64>,
    line: Vec<Complex64>,
    scratch: Vec<Complex64>,
    forward: [Arc<dyn Fft<f64>>; 3],
    inverse: [Arc<dyn Fft<f64>>; 3],
    moduli: [Vec<f64>; 3],
}

fn spline(fraction: f64) -> ([f64; 5], [f64; 5]) {
    // Cardinal splines and their analytic derivatives; see LICENSE-PME.
    let mut weights = [0.0; 5];
    let mut derivatives = [0.0; 5];
    weights[0] = 1.0 - fraction;
    weights[1] = fraction;
    for order in 3..=5 {
        if order == 5 {
            derivatives[0] = -weights[0];
            for i in 1..5 {
                derivatives[i] = weights[i - 1] - weights[i];
            }
        }
        let denominator = (order - 1) as f64;
        weights[order - 1] = fraction * weights[order - 2] / denominator;
        for i in (1..order - 1).rev() {
            weights[i] = ((fraction + (order - 1 - i) as f64) * weights[i - 1]
                + ((i + 1) as f64 - fraction) * weights[i])
                / denominator;
        }
        weights[0] *= (1.0 - fraction) / denominator;
    }
    (weights, derivatives)
}

pub(crate) fn reciprocal_box(box_vectors: &[[f64; 3]; 3]) -> ([[f64; 3]; 3], f64) {
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let [a, b, c] = *box_vectors;
    let bc = cross(b, c);
    let volume = a.iter().zip(bc).map(|(x, y)| x * y).sum::<f64>();
    assert!(
        volume.is_finite() && volume > 0.0,
        "periodic box volume must be positive and finite"
    );
    (
        [
            bc.map(|v| v / volume),
            cross(c, a).map(|v| v / volume),
            cross(a, b).map(|v| v / volume),
        ],
        volume,
    )
}

fn stencil(
    position: &[f64],
    reciprocal: &[[f64; 3]; 3],
    dimensions: [usize; 3],
) -> ([usize; 3], [[f64; 5]; 3], [[f64; 5]; 3]) {
    let mut indices = [0; 3];
    let mut weights = [[0.0; 5]; 3];
    let mut derivatives = [[0.0; 5]; 3];
    for axis in 0..3 {
        let fractional = (0..3)
            .map(|j| position[j] * reciprocal[axis][j])
            .sum::<f64>();
        let coordinate = (fractional - fractional.floor()) * dimensions[axis] as f64;
        indices[axis] = coordinate.floor() as usize % dimensions[axis];
        (weights[axis], derivatives[axis]) = spline(coordinate - coordinate.floor());
    }
    (indices, weights, derivatives)
}

impl Workspace {
    fn new(dimensions: [usize; 3]) -> Self {
        assert!(
            dimensions.iter().all(|&n| n >= 5),
            "PME grid dimensions must be at least five"
        );
        let count = dimensions
            .iter()
            .try_fold(1_usize, |count, &n| count.checked_mul(n))
            .expect("PME grid size overflow");
        let mut planner = FftPlanner::<f64>::new();
        let forward = dimensions.map(|n| planner.plan_fft_forward(n));
        let inverse = dimensions.map(|n| planner.plan_fft_inverse(n));
        let scratch_size = forward
            .iter()
            .chain(&inverse)
            .map(|fft| fft.get_inplace_scratch_len())
            .max()
            .unwrap();
        let (weights, _) = spline(0.0);
        let moduli = dimensions.map(|n| {
            let mut values: Vec<f64> = (0..n)
                .map(|k| {
                    let sum: Complex64 = weights
                        .iter()
                        .enumerate()
                        .map(|(i, &w)| {
                            Complex64::from_polar(w, 2.0 * PI * k as f64 * i as f64 / n as f64)
                        })
                        .sum();
                    sum.norm_sqr()
                })
                .collect();
            for k in 0..n {
                if values[k] < 1e-7 {
                    values[k] = 0.5 * (values[(k + n - 1) % n] + values[(k + 1) % n]);
                }
            }
            values
        });
        Self {
            dimensions,
            grid: vec![Complex64::default(); count],
            line: vec![Complex64::default(); *dimensions.iter().max().unwrap()],
            scratch: vec![Complex64::default(); scratch_size],
            forward,
            inverse,
            moduli,
        }
    }

    fn transform(&mut self, inverse: bool) {
        let [nx, ny, nz] = self.dimensions;
        let plans = if inverse {
            &self.inverse
        } else {
            &self.forward
        };
        for row in self.grid.chunks_exact_mut(nz) {
            plans[2].process_with_scratch(row, &mut self.scratch);
        }
        for x in 0..nx {
            for z in 0..nz {
                for y in 0..ny {
                    self.line[y] = self.grid[(x * ny + y) * nz + z];
                }
                plans[1].process_with_scratch(&mut self.line[..ny], &mut self.scratch);
                for y in 0..ny {
                    self.grid[(x * ny + y) * nz + z] = self.line[y];
                }
            }
        }
        for y in 0..ny {
            for z in 0..nz {
                for x in 0..nx {
                    self.line[x] = self.grid[(x * ny + y) * nz + z];
                }
                plans[0].process_with_scratch(&mut self.line[..nx], &mut self.scratch);
                for x in 0..nx {
                    self.grid[(x * ny + y) * nz + z] = self.line[x];
                }
            }
        }
    }

    fn potential(
        &mut self,
        positions: &[f64],
        amplitudes: impl Iterator<Item = f64> + Clone,
        box_vectors: &[[f64; 3]; 3],
        alpha: f64,
        dispersion: bool,
    ) -> f64 {
        let (reciprocal, volume) = reciprocal_box(box_vectors);
        let [nx, ny, nz] = self.dimensions;
        self.grid.fill(Complex64::default());
        for (position, charge) in positions.chunks_exact(3).zip(amplitudes) {
            if charge == 0.0 {
                continue;
            }
            let (indices, w, _) = stencil(position, &reciprocal, self.dimensions);
            for ix in 0..5 {
                for iy in 0..5 {
                    for iz in 0..5 {
                        let index = (((indices[0] + ix) % nx) * ny + (indices[1] + iy) % ny) * nz
                            + (indices[2] + iz) % nz;
                        self.grid[index].re += charge * w[0][ix] * w[1][iy] * w[2][iz];
                    }
                }
            }
        }
        self.transform(false);
        let frequency = |k: usize, n: usize| {
            if k < n.div_ceil(2) {
                k as f64
            } else {
                k as f64 - n as f64
            }
        };
        let mut energy = 0.0;
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    let index = (x * ny + y) * nz + z;
                    if index == 0 && !dispersion {
                        self.grid[index] = Complex64::default();
                        continue;
                    }
                    let modes = [frequency(x, nx), frequency(y, ny), frequency(z, nz)];
                    let k2 = (0..3)
                        .map(|axis| {
                            (0..3)
                                .map(|d| modes[d] * reciprocal[d][axis])
                                .sum::<f64>()
                                .powi(2)
                        })
                        .sum::<f64>();
                    let moduli = self.moduli[0][x] * self.moduli[1][y] * self.moduli[2][z];
                    let influence = if dispersion {
                        let k = k2.sqrt();
                        let b = PI * k / alpha;
                        -PI * PI.sqrt() / (3.0 * volume * moduli)
                            * (2.0 * PI.powi(3) * PI.sqrt() * libm::erfc(b) * k * k2
                                + (-b * b).exp() * (alpha.powi(3) - 2.0 * alpha * PI * PI * k2))
                    } else {
                        ONE_4PI_EPS0 * (-PI * PI * k2 / (alpha * alpha)).exp()
                            / (PI * volume * k2 * moduli)
                    };
                    energy += 0.5 * influence * self.grid[index].norm_sqr();
                    self.grid[index] *= influence;
                }
            }
        }
        self.transform(true);
        energy
    }

    fn evaluate(
        &mut self,
        positions: &[f64],
        amplitudes: impl Iterator<Item = f64> + Clone,
        box_vectors: &[[f64; 3]; 3],
        alpha: f64,
        forces: &mut [f64],
        dispersion: bool,
    ) -> f64 {
        let energy = self.potential(
            positions,
            amplitudes.clone(),
            box_vectors,
            alpha,
            dispersion,
        );
        let reciprocal = reciprocal_box(box_vectors).0;
        let [nx, ny, nz] = self.dimensions;
        for ((position, charge), force) in positions
            .chunks_exact(3)
            .zip(amplitudes)
            .zip(forces.chunks_exact_mut(3))
        {
            if charge == 0.0 {
                continue;
            }
            let (indices, w, dw) = stencil(position, &reciprocal, self.dimensions);
            let mut gradient = [0.0; 3];
            for ix in 0..5 {
                for iy in 0..5 {
                    for iz in 0..5 {
                        let index = (((indices[0] + ix) % nx) * ny + (indices[1] + iy) % ny) * nz
                            + (indices[2] + iz) % nz;
                        let potential = self.grid[index].re;
                        gradient[0] += potential * dw[0][ix] * w[1][iy] * w[2][iz];
                        gradient[1] += potential * w[0][ix] * dw[1][iy] * w[2][iz];
                        gradient[2] += potential * w[0][ix] * w[1][iy] * dw[2][iz];
                    }
                }
            }
            for (axis, value) in force.iter_mut().enumerate() {
                *value -= charge
                    * (0..3)
                        .map(|d| gradient[d] * self.dimensions[d] as f64 * reciprocal[d][axis])
                        .sum::<f64>();
            }
        }
        energy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn charge_derivatives_match_reciprocal_energy_differences() {
        let cache = PmeCache::default();
        let positions = [0.1, 0.2, 0.3, 0.7, 0.4, 1.2, 1.4, 0.3, 0.8];
        for tilt in [0., 0.35] {
            let vectors = [[3., 0., 0.], [tilt, 3.2, 0.], [-tilt, 0.1, 3.4]];
            let parameters = (2.4, [12, 14, 16]);
            let mut charges = [0., 0.3, -0.4];
            let mut derivatives = [0.; 3];
            cache.charge_derivatives(
                &positions,
                &charges,
                &vectors,
                parameters,
                &[0, 1, 2],
                &mut derivatives,
            );
            let mut forces = [0.; 9];
            let energy =
                cache.evaluate_charges(&positions, &charges, &vectors, parameters, &mut forces);
            assert!(
                (energy
                    - 0.5
                        * charges
                            .iter()
                            .zip(derivatives)
                            .map(|(q, d)| q * d)
                            .sum::<f64>())
                .abs()
                    < 1e-12
            );
            for i in 0..3 {
                let original = charges[i];
                charges[i] = original + 1e-5;
                let plus = cache.evaluate_charges(
                    &positions,
                    &charges,
                    &vectors,
                    parameters,
                    &mut [0.; 9],
                );
                charges[i] = original - 1e-5;
                let minus = cache.evaluate_charges(
                    &positions,
                    &charges,
                    &vectors,
                    parameters,
                    &mut [0.; 9],
                );
                charges[i] = original;
                assert!(((plus - minus) / 2e-5 - derivatives[i]).abs() < 2e-8);
            }
        }
    }
    #[test]
    fn spline_partition_and_derivatives() {
        for fraction in [0.0, 0.13, 0.5, 0.999] {
            let (w, dw) = spline(fraction);
            assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-14);
            assert!(dw.iter().sum::<f64>().abs() < 1e-14);
            let (plus, _) = spline(fraction + 1e-6);
            let (minus, _) = spline(fraction - 1e-6);
            for i in 0..5 {
                assert!(((plus[i] - minus[i]) / 2e-6 - dw[i]).abs() < 1e-9);
            }
        }
    }
}
