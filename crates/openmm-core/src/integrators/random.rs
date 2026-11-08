use burn::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use rand_distr::StandardNormal;
use std::sync::Mutex;

pub(crate) struct GaussianStream {
    seed: i32,
    random: Mutex<Option<ChaCha12Rng>>,
}

impl Default for GaussianStream {
    fn default() -> Self {
        Self {
            seed: 0,
            random: Mutex::new(None),
        }
    }
}

impl GaussianStream {
    pub(crate) fn seed(&self) -> i32 {
        self.seed
    }

    pub(crate) fn set_seed(&mut self, seed: i32) {
        self.seed = seed;
    }

    pub(crate) fn initialize(&mut self) {
        *self.random.get_mut().unwrap() = Some(Self::generator(self.seed));
    }

    pub(crate) fn with_random<R>(&self, generate: impl FnOnce(&mut ChaCha12Rng) -> R) -> R {
        let mut random = self.random.lock().unwrap();
        let random = random.get_or_insert_with(|| Self::generator(self.seed));
        generate(random)
    }

    fn generator(seed: i32) -> ChaCha12Rng {
        if seed == 0 {
            ChaCha12Rng::from_os_rng()
        } else {
            ChaCha12Rng::seed_from_u64(seed as u32 as u64)
        }
    }

    pub(crate) fn tensor<B: Backend>(&self, masses: &Tensor<B, 1>) -> Tensor<B, 2> {
        let n = masses.dims()[0];
        let mut values = vec![0.0; 3 * n];
        crate::tensor::with_f64_data(masses, |masses| {
            self.with_random(|random| {
                for (atom, &mass) in masses.iter().enumerate() {
                    if mass != 0.0 {
                        for axis in 0..3 {
                            values[3 * atom + axis] = random.sample::<f64, _>(StandardNormal);
                        }
                    }
                }
            });
        });
        crate::tensor::from_f64_vec(values, [n, 3], &masses.device())
    }
}
