// Splitting and adaptation adapted from OpenMM ReferenceQTBDynamics.
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
use super::spectrum::spectrum;
use crate::constraints::ConstraintError;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use rand_distr::StandardNormal;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

pub(super) struct Configuration {
    pub dt: f64,
    pub friction: f64,
    pub cutoff: f64,
    pub steps: usize,
    pub particle_types: Vec<usize>,
    pub particles: Vec<Vec<usize>>,
    pub rates: Vec<f64>,
    pub masses: Vec<f64>,
}

pub(super) struct Bath {
    pub config: Configuration,
    frequency_count: usize,
    frequencies: Vec<f64>,
    base_filter: Vec<f64>,
    gamma: Vec<Arc<Vec<f64>>>,
    temperature: Option<f64>,
    step: usize,
    first_segment: bool,
    seed: [u8; 32],
    random_positions: Vec<u128>,
    force: Vec<f64>,
    velocity: Vec<f64>,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
    first: Vec<Complex<f64>>,
    second: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
    filter: Vec<f64>,
    derivative: Vec<f64>,
}

impl Bath {
    pub fn new(config: Configuration, random: &mut ChaCha12Rng) -> Self {
        let nfft = 3 * config.steps;
        let count = nfft / 2 + 1;
        let n = config.particle_types.len();
        let gamma = Arc::new(vec![config.friction; count]);
        let mut planner = FftPlanner::new();
        let forward = planner.plan_fft_forward(nfft);
        let inverse = planner.plan_fft_inverse(nfft);
        let scratch = vec![
            Complex::default();
            forward
                .get_inplace_scratch_len()
                .max(inverse.get_inplace_scratch_len())
        ];
        let adapt = config.rates.iter().any(|&rate| rate > 0.0);
        Self {
            frequency_count: count,
            frequencies: (0..count)
                .map(|i| 2.0 * PI * i as f64 / (nfft as f64 * config.dt))
                .collect(),
            base_filter: vec![0.0; count],
            gamma: vec![gamma; config.particles.len()],
            temperature: None,
            step: 0,
            first_segment: true,
            seed: random.random(),
            random_positions: vec![0; 3 * n],
            force: vec![0.0; 3 * n * config.steps],
            velocity: if adapt {
                vec![0.0; 3 * n * config.steps]
            } else {
                Vec::new()
            },
            forward,
            inverse,
            first: vec![Complex::default(); nfft],
            second: vec![Complex::default(); nfft],
            scratch,
            filter: vec![0.0; count],
            derivative: vec![0.0; count],
            config,
        }
    }

    pub fn adapted_friction(&self, particle: usize) -> Result<Vec<f64>, ConstraintError> {
        let kind = self
            .config
            .particle_types
            .get(particle)
            .ok_or_else(|| ConstraintError("QTB particle index is out of range".into()))?;
        Ok(self.gamma[*kind].as_ref().clone())
    }

    pub fn prepare(&mut self, temperature: f64) -> Result<(), ConstraintError> {
        if self.step != 0 {
            return Ok(());
        }
        if self.temperature != Some(temperature) {
            let (_, target) = spectrum(
                temperature,
                self.config.friction,
                self.frequencies[1],
                self.frequency_count,
            )?;
            let dt = self.config.dt;
            let friction = self.config.friction;
            let damping = (-dt * friction).exp();
            for (i, &omega) in self.frequencies.iter().enumerate() {
                let cutoff = 1.0
                    / (1.0 + ((omega - self.config.cutoff) / (self.config.cutoff / 100.0)).exp());
                let numerator = (-dt * friction).exp_m1().powi(2)
                    + 4.0 * damping * (0.5 * omega * dt).sin().powi(2);
                let correction = numerator / ((friction * friction + omega * omega) * dt * dt);
                self.base_filter[i] = cutoff * target[i] * correction / friction;
            }
            self.temperature = Some(temperature);
        }
        if !self.first_segment {
            self.adapt()?;
        }
        self.generate();
        self.first_segment = false;
        Ok(())
    }

    fn adapt(&mut self) -> Result<(), ConstraintError> {
        let length = self.config.steps;
        for (kind, particles) in self.config.particles.iter().enumerate() {
            let rate = self.config.rates[kind];
            let mass = self.config.masses[kind];
            if rate == 0.0 || mass == 0.0 {
                continue;
            }
            self.derivative.fill(0.0);
            for &particle in particles {
                for axis in 0..3 {
                    let base = (3 * particle + axis) * length;
                    self.first.fill(Complex::default());
                    self.second.fill(Complex::default());
                    for i in 0..length {
                        self.first[i].re = self.velocity[base + i];
                        self.second[i].re = self.force[base + i];
                    }
                    self.forward
                        .process_with_scratch(&mut self.first, &mut self.scratch);
                    self.forward
                        .process_with_scratch(&mut self.second, &mut self.scratch);
                    for i in 0..self.frequency_count {
                        self.derivative[i] += mass * self.gamma[kind][i] * self.first[i].norm_sqr()
                            - (self.first[i] * self.second[i].conj()).re;
                    }
                }
            }
            let scale = self.config.dt * rate / (3 * particles.len() * length) as f64;
            let gamma = Arc::make_mut(&mut self.gamma[kind]);
            for (i, value) in gamma.iter_mut().enumerate() {
                let proposed = *value - scale * self.derivative[i];
                if !proposed.is_finite() {
                    return Err(ConstraintError(
                        "QTB friction adaptation became nonfinite".into(),
                    ));
                }
                *value = proposed.max(0.0);
            }
        }
        Ok(())
    }

    fn generate(&mut self) {
        let length = self.config.steps;
        let nfft = self.first.len();
        let mut random = ChaCha12Rng::from_seed(self.seed);
        for (kind, particles) in self.config.particles.iter().enumerate() {
            let mass = self.config.masses[kind];
            if mass == 0.0 {
                continue;
            }
            for i in 0..self.frequency_count {
                self.filter[i] = (self.base_filter[i] * self.gamma[kind][i]).sqrt();
            }
            let amplitude =
                (2.0 * mass * self.config.friction / self.config.dt).sqrt() / nfft as f64;
            for &particle in particles {
                for axis in 0..3 {
                    let stream = 3 * particle + axis;
                    random.set_stream(stream as u64);
                    random.set_word_pos(self.random_positions[stream]);
                    // Regenerate the overlapping white-noise window from its RNG position.
                    // Retaining counters avoids storing 2*segment_length values per DOF.
                    for i in 0..nfft {
                        self.first[i] = Complex::new(random.sample(StandardNormal), 0.0);
                        if i + 1 == length {
                            self.random_positions[stream] = random.get_word_pos();
                        }
                    }
                    self.forward
                        .process_with_scratch(&mut self.first, &mut self.scratch);
                    for i in 0..self.frequency_count {
                        self.first[i] *= self.filter[i];
                        if i != 0 && nfft - i != i {
                            self.first[nfft - i] *= self.filter[i];
                        }
                    }
                    self.inverse
                        .process_with_scratch(&mut self.first, &mut self.scratch);
                    for i in 0..length {
                        self.force[stream * length + i] = amplitude * self.first[length + i].re;
                    }
                }
            }
        }
    }

    pub fn update(&mut self, p: &mut [f64], v: &mut [f64], f: &[f64], m: &[f64], phase: u8) {
        let dt = self.config.dt;
        let damping = (-dt * self.config.friction).exp();
        let half_damping = (-0.5 * dt * self.config.friction).exp();
        for (particle, &mass) in m.iter().enumerate() {
            if mass == 0.0 {
                continue;
            }
            for axis in 0..3 {
                let index = 3 * particle + axis;
                if phase != 2 {
                    v[index] += dt / mass * f[index];
                }
                if phase != 1 {
                    let noise_index = index * self.config.steps + self.step;
                    let dv = dt / mass * self.force[noise_index];
                    if !self.velocity.is_empty() {
                        self.velocity[noise_index] = half_damping * v[index] + 0.5 * dv;
                    }
                    p[index] += 0.5 * dt * v[index];
                    v[index] = damping * v[index] + dv;
                    p[index] += 0.5 * dt * v[index];
                }
            }
        }
    }

    pub fn finish_step(&mut self) {
        self.step = (self.step + 1) % self.config.steps;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn regenerated_noise_matches_direct_fourier_windows() {
        for length in [1, 2, 3, 4, 7] {
            let config = Configuration {
                dt: 0.001,
                friction: 10.0,
                cutoff: 500.0,
                steps: length,
                particle_types: vec![0],
                particles: vec![vec![0]],
                rates: vec![0.0],
                masses: vec![2.0],
            };
            let mut bath = Bath::new(config, &mut ChaCha12Rng::seed_from_u64(71));
            let nfft = 3 * length;
            let mut sequences = Vec::new();
            for axis in 0..3 {
                let mut random = ChaCha12Rng::from_seed(bath.seed);
                random.set_stream(axis);
                sequences.push(
                    (0..8 * length)
                        .map(|_| random.sample::<f64, _>(StandardNormal))
                        .collect::<Vec<_>>(),
                );
            }
            for segment in 0..5 {
                bath.prepare(300.0).unwrap();
                for (axis, sequence) in sequences.iter().enumerate() {
                    let mut frequency = vec![Complex::default(); nfft];
                    for (k, value) in frequency.iter_mut().enumerate() {
                        for j in 0..nfft {
                            let angle = -2.0 * PI * (j * k) as f64 / nfft as f64;
                            *value += Complex::from_polar(sequence[segment * length + j], angle);
                        }
                        let bin = k.min(nfft - k);
                        *value *= (bath.base_filter[bin] * 10.0).sqrt();
                    }
                    for time in 0..length {
                        let mut expected = Complex::<f64>::default();
                        for (k, &value) in frequency.iter().enumerate() {
                            let angle = 2.0 * PI * ((length + time) * k) as f64 / nfft as f64;
                            expected += value * Complex::from_polar(1.0, angle);
                        }
                        let expected = expected.re / nfft as f64 * (40.0_f64 / 0.001).sqrt();
                        assert_relative_eq!(
                            bath.force[axis * length + time],
                            expected,
                            epsilon = 2e-10,
                            max_relative = 2e-12
                        );
                    }
                }
                for _ in 0..length {
                    bath.finish_step();
                }
            }
            assert!(bath.velocity.is_empty());
            assert_eq!(bath.random_positions.len(), 3);
        }
    }
}
