use super::nonbonded::{ParticleParam, ONE_4PI_EPS0};
use rustfft::num_complex::Complex64;
use std::f64::consts::PI;
use std::sync::Mutex;

#[derive(Default)]
pub(crate) struct EwaldCache(Mutex<Workspace>);

impl Clone for EwaldCache {
    fn clone(&self) -> Self {
        Self::default()
    }
}

#[derive(Default)]
struct Workspace {
    phases: Vec<Complex64>,
    xy: Vec<Complex64>,
    xyz: Vec<Complex64>,
}

pub(crate) fn parameters(widths: [f64; 3], cutoff: f64, tolerance: f64) -> (f64, [usize; 3]) {
    let alpha = (-(2.0 * tolerance).ln()).sqrt() / cutoff;
    let limits = widths.map(|width| {
        let error = |k: usize| {
            tolerance
                - 0.05
                    * (width * alpha).sqrt()
                    * k as f64
                    * (-(k as f64 * PI / (width * alpha)).powi(2)).exp()
        };
        let mut k = 10;
        if error(k) > 0.0 {
            while k > 0 && error(k) > 0.0 {
                k -= 1;
            }
            k += 1;
        } else {
            while error(k) < 0.0 {
                k += 1;
            }
        }
        k | 1
    });
    (alpha, limits)
}

impl EwaldCache {
    pub(crate) fn evaluate(
        &self,
        positions: &[f64],
        particles: &[ParticleParam],
        widths: [f64; 3],
        alpha: f64,
        limits: [usize; 3],
        forces: &mut [f64],
    ) -> f64 {
        let mut guard = self.0.lock().unwrap();
        let Workspace { phases, xy, xyz } = &mut *guard;
        let n = particles.len();
        let maximum = *limits.iter().max().unwrap();
        phases.resize(maximum * n * 3, Complex64::default());
        xy.resize(n, Complex64::default());
        xyz.resize(n, Complex64::default());
        let waves = widths.map(|width| 2.0 * PI / width);
        for (i, position) in positions.chunks_exact(3).enumerate() {
            for axis in 0..3 {
                phases[i * 3 + axis] = Complex64::new(1.0, 0.0);
                let base = Complex64::from_polar(1.0, position[axis] * waves[axis]);
                for k in 1..maximum {
                    phases[(k * n + i) * 3 + axis] = phases[((k - 1) * n + i) * 3 + axis] * base;
                }
            }
        }
        let phase = |k: isize, i: usize, axis: usize| {
            let value = phases[(k.unsigned_abs() * n + i) * 3 + axis];
            if k < 0 {
                value.conj()
            } else {
                value
            }
        };
        let coefficient = ONE_4PI_EPS0 * 4.0 * PI / widths.iter().product::<f64>();
        let mut energy = 0.0;
        for x in 0..limits[0] as isize {
            let low_y = if x == 0 { 0 } else { 1 - limits[1] as isize };
            for y in low_y..limits[1] as isize {
                for (i, value) in xy.iter_mut().enumerate() {
                    *value = phase(x, i, 0) * phase(y, i, 1);
                }
                let low_z = if x == 0 && y == 0 {
                    1
                } else {
                    1 - limits[2] as isize
                };
                for z in low_z..limits[2] as isize {
                    let mut structure = Complex64::default();
                    for (i, value) in xyz.iter_mut().enumerate() {
                        *value = xy[i] * phase(z, i, 2) * particles[i].charge;
                        structure += *value;
                    }
                    let wave = [
                        x as f64 * waves[0],
                        y as f64 * waves[1],
                        z as f64 * waves[2],
                    ];
                    let k2: f64 = wave.iter().map(|v| v * v).sum();
                    let influence = coefficient * (-k2 / (4.0 * alpha * alpha)).exp() / k2;
                    energy += influence * structure.norm_sqr();
                    for (force, value) in forces.chunks_exact_mut(3).zip(xyz.iter()) {
                        let scale =
                            2.0 * influence * (structure.re * value.im - structure.im * value.re);
                        for axis in 0..3 {
                            force[axis] += scale * wave[axis];
                        }
                    }
                }
            }
        }
        energy
    }
}
