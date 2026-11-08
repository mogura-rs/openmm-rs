use super::Force;
use burn::prelude::*;

#[derive(Clone)]
pub struct CMMotionRemover {
    metadata: super::ForceMetadata,
    frequency: u64,
}

impl CMMotionRemover {
    super::force_metadata!();

    pub fn new(frequency: u64) -> Self {
        assert!(frequency > 0, "frequency must be positive");
        Self {
            metadata: super::ForceMetadata::new("CMMotionRemover"),
            frequency,
        }
    }

    pub fn frequency(&self) -> u64 {
        self.frequency
    }

    pub fn set_frequency(&mut self, frequency: u64) {
        assert!(frequency > 0, "frequency must be positive");
        self.frequency = frequency;
    }

    pub fn uses_periodic_boundary_conditions(&self) -> bool {
        false
    }
}

impl Default for CMMotionRemover {
    fn default() -> Self {
        Self::new(1)
    }
}

impl<B: Backend> Force<B> for CMMotionRemover {
    super::force_metadata_impl!();

    fn update_context_velocities(
        &mut self,
        velocities: &mut Tensor<B, 2>,
        masses: &[f64],
        _step_size: f64,
        step_count: u64,
    ) {
        if !step_count.is_multiple_of(self.frequency) {
            return;
        }
        let mass: f64 = masses.iter().sum();
        if mass == 0.0 {
            return;
        }
        let mut values = crate::tensor::to_f64_vec(velocities);
        let mut momentum = [0.0; 3];
        for (velocity, mass) in values.chunks_exact(3).zip(masses) {
            for axis in 0..3 {
                momentum[axis] += mass * velocity[axis];
            }
        }
        for component in &mut momentum {
            *component /= mass;
        }
        for (velocity, &mass) in values.chunks_exact_mut(3).zip(masses) {
            if mass != 0.0 {
                for axis in 0..3 {
                    velocity[axis] -= momentum[axis];
                }
            }
        }
        *velocities = crate::tensor::from_f64_vec(values, [masses.len(), 3], &velocities.device());
    }

    fn calc_forces_and_energy_groups(
        &self,
        _positions: &Tensor<B, 2>,
        _box_vectors: Option<&[[f64; 3]; 3]>,
        _groups: u32,
    ) -> Option<(Tensor<B, 2>, f64)> {
        None
    }

    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,
        _box_vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64) {
        (Tensor::zeros(positions.shape(), &positions.device()), 0.0)
    }

    fn prepare_host_evaluation<'a>(
        &'a self,
        _positions: &'a Tensor<B, 2>,
        _box_vectors: Option<&'a [[f64; 3]; 3]>,
    ) -> Option<super::HostForceEvaluation<'a>> {
        Some(super::HostForceEvaluation::new(0.0, |_, _| {}))
    }
}
