// Deconvolution adapted from OpenMM QTBIntegratorUtilities.
// Copyright (c) 2025 Stanford University and the Authors.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
use crate::constraints::ConstraintError;
use crate::integrators::langevin::BOLTZMANN;
use std::f64::consts::PI;

/// Standard and deconvolved QTB energies on the OpenMM reference frequency grid.
/// Temperature is in K, friction in 1/ps, step size in ps, and energies in kJ/mol.
/// The computation uses O(n^2) time and O(n) auxiliary memory.
pub fn quantum_thermal_bath_spectrum(
    temperature: f64,
    friction: f64,
    step_size: f64,
    frequencies: usize,
) -> Result<(Vec<f64>, Vec<f64>), ConstraintError> {
    if !step_size.is_finite() || step_size <= 0.0 {
        return Err(ConstraintError(
            "QTB step size must be finite and positive".into(),
        ));
    }
    spectrum(
        temperature,
        friction,
        PI / (frequencies as f64 * step_size),
        frequencies,
    )
}

pub(super) fn spectrum(
    temperature: f64,
    friction: f64,
    dw: f64,
    n: usize,
) -> Result<(Vec<f64>, Vec<f64>), ConstraintError> {
    if !temperature.is_finite()
        || temperature < 0.0
        || !friction.is_finite()
        || friction <= 0.0
        || !dw.is_finite()
        || dw <= 0.0
        || n < 2
    {
        return Err(ConstraintError("QTB spectrum requires nonnegative finite temperature, positive finite friction and frequency spacing, and at least two frequencies".into()));
    }
    let hbar = 1.054571628e-34 * 6.02214076e23 / (1000.0 * 1e-12);
    let kt = BOLTZMANN * temperature;
    let theta: Vec<_> = (0..n)
        .map(|i| {
            if i == 0 {
                kt
            } else {
                let energy = hbar * dw * i as f64;
                energy * (0.5 + 1.0 / (energy / kt).exp_m1())
            }
        })
        .collect();
    let omega2: Vec<_> = (0..n).map(|i| ((i as f64 + 0.5) * dw).powi(2)).collect();
    let damping: Vec<_> = omega2.iter().map(|w2| friction * friction * w2).collect();
    let mut row = vec![0.0; n];
    let mut normalization = vec![0.0; n];
    let mut h = vec![0.0; n];
    let mut current: Vec<_> = theta.iter().map(|v| 0.5 * v).collect();
    let fill_row = |i: usize, row: &mut [f64]| {
        for j in 0..n {
            let difference = omega2[j] - omega2[i];
            row[j] = 1.0 / (difference * difference + damping[j]);
        }
    };
    for i in 0..n {
        fill_row(i, &mut row);
        normalization[i] = 0.5 / row.iter().sum::<f64>();
        for j in 0..n {
            h[j] += row[j] * normalization[i] * current[i];
        }
    }
    let mut denominator = vec![0.0; n];
    // With M_ij = dw*C_ij*scale_i, dw*D = M^T*M. Apply both factors
    // without constructing either quadratic matrix.
    for _ in 0..20 {
        denominator.fill(0.0);
        for (i, &scale) in normalization.iter().enumerate() {
            fill_row(i, &mut row);
            let projection =
                row.iter().zip(&current).map(|(a, b)| a * b).sum::<f64>() * scale * scale;
            for j in 0..n {
                denominator[j] += row[j] * projection;
            }
        }
        for j in 0..n {
            current[j] *= h[j] / denominator[j];
        }
    }
    if theta
        .iter()
        .chain(&current)
        .any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(ConstraintError(
            "QTB spectrum is not finite for these parameters".into(),
        ));
    }
    Ok((theta, current))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn matches_reference_deconvolution() {
        // OpenMM QTBIntegratorUtilities, Reference, dt=0.001 ps.
        for (n, temperature, friction, expected) in [
            (
                6,
                300.0,
                10.0,
                vec![
                    (0, 2.4938859588004267),
                    (1, 16.627931782380685),
                    (2, 33.25527918921168),
                    (5, 83.13734610505983),
                ],
            ),
            (
                128,
                70.0,
                1.0,
                vec![
                    (0, 0.5819554905174056),
                    (1, 0.894327712297645),
                    (2, 1.5737806189721284),
                    (64, 49.8796824145325),
                    (127, 98.98074170971641),
                ],
            ),
        ] {
            let (_, actual) =
                quantum_thermal_bath_spectrum(temperature, friction, 0.001, n).unwrap();
            for (i, expected) in expected {
                assert_relative_eq!(actual[i], expected, epsilon = 1e-12, max_relative = 2e-12);
            }
        }
        let (theta, actual) = quantum_thermal_bath_spectrum(0.0, 10.0, 0.001, 2).unwrap();
        assert_eq!(theta[0], 0.0);
        assert_eq!(actual[0], 0.0);
        assert_relative_eq!(actual[1], 49.880_037_040_690_48, epsilon = 1e-12);
    }

    #[test]
    fn rejects_invalid_spectrum_parameters() {
        for (temperature, friction, dt, n) in [
            (-1.0, 1.0, 0.001, 6),
            (300.0, 0.0, 0.001, 6),
            (300.0, 1.0, 0.0, 6),
            (300.0, 1.0, 0.001, 1),
            (f64::NAN, 1.0, 0.001, 6),
            (300.0, f64::INFINITY, 0.001, 6),
        ] {
            assert!(quantum_thermal_bath_spectrum(temperature, friction, dt, n).is_err());
        }
    }
}
