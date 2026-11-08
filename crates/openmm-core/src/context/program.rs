use super::*;
use crate::integrators::ContextUpdateSettings;

impl<B: Backend> Context<B> {
    pub(crate) fn update_context_state(&mut self, settings: ContextUpdateSettings) -> bool {
        let mut changed = false;
        for index in 0..self.system.forces().len() {
            self.system.forces_mut()[index].update_context_velocities(
                &mut self.velocities,
                &self.host_masses,
                settings.step_size,
                self.step_count,
            );
            if let Some(vectors) = self.periodic_box_vectors {
                let proposal = self.system.forces_mut()[index].propose_context_box_update(&vectors);
                if let Some(proposal) = proposal {
                    let groups = settings.force_groups;
                    let old_energy = self.energy_at(&self.positions, &vectors, groups);
                    let positions = self.prepare_scaled_positions(
                        proposal.transform,
                        proposal.scale_molecules_as_rigid,
                    );
                    let new_energy = self.energy_at(&positions, &proposal.box_vectors, groups);
                    let scaled_particles = if proposal.scale_molecules_as_rigid {
                        self.molecules().len()
                    } else {
                        self.system.num_particles()
                    };
                    if self.system.forces_mut()[index].accept_context_box_update(
                        &proposal,
                        new_energy - old_energy,
                        scaled_particles,
                    ) {
                        self.box_update_positions =
                            Some(std::mem::replace(&mut self.positions, positions));
                        self.periodic_box_vectors = Some(proposal.box_vectors);
                        changed = true;
                    } else {
                        self.box_update_positions = Some(positions);
                    }
                }
            }
        }
        if changed {
            self.invalidate_force_cache();
        }
        changed
    }

    pub(crate) fn read_program_particles<R>(
        &self,
        read: impl FnOnce(&[f64], &[f64], &[f64]) -> R,
    ) -> R {
        crate::tensor::with_f64_data(&self.positions, |p| {
            crate::tensor::with_f64_data(&self.velocities, |v| read(p, v, &self.host_masses))
        })
    }

    pub(crate) fn update_program_values(
        &mut self,
        positions: bool,
        mut update: impl FnMut(usize, [f64; 3], [f64; 3], f64) -> [f64; 3],
    ) {
        if positions {
            self.invalidate_force_cache();
        }
        let (values, other) = if positions {
            (&mut self.positions, &self.velocities)
        } else {
            (&mut self.velocities, &self.positions)
        };
        crate::tensor::with_f64_data(other, |other| {
            let mut run = |values: &mut [f64]| {
                for (atom, &mass) in self.host_masses.iter().enumerate() {
                    if mass == 0.0 {
                        continue;
                    }
                    let a = std::array::from_fn(|i| values[3 * atom + i]);
                    let b = std::array::from_fn(|i| other[3 * atom + i]);
                    let result = if positions {
                        update(atom, a, b, mass)
                    } else {
                        update(atom, b, a, mass)
                    };
                    values[3 * atom..3 * atom + 3].copy_from_slice(&result);
                }
            };
            if crate::tensor::with_mut_ndarray_f64(values, &mut run).is_none() {
                let mut data = crate::tensor::to_f64_vec(values);
                run(&mut data);
                *values = crate::tensor::from_f64_vec(data, values.dims(), &self.device);
            }
        });
    }

    pub(crate) fn constrain_program_particles(
        &mut self,
        reference: &[f64],
        velocities: bool,
        tolerance: f64,
    ) -> Result<(), ConstraintError> {
        if self.constraints.is_empty() {
            return Ok(());
        }
        if !velocities {
            self.invalidate_force_cache();
        }
        let values = if velocities {
            &mut self.velocities
        } else {
            &mut self.positions
        };
        if let Some(result) = crate::tensor::with_mut_ndarray_f64(values, |values| {
            self.constraints
                .solve(reference, values, velocities, tolerance)
        }) {
            return result;
        }
        let mut data = crate::tensor::to_f64_vec(values);
        self.constraints
            .solve(reference, &mut data, velocities, tolerance)?;
        *values = crate::tensor::from_f64_vec(data, values.dims(), &self.device);
        Ok(())
    }

    pub(crate) fn program_parameter_derivatives(&self, groups: u32) -> HashMap<String, f64> {
        let mut result = HashMap::new();
        for force in self.system.forces() {
            if groups & (1 << force.force_group()) != 0 {
                for (name, value) in force.energy_parameter_derivatives(
                    &self.positions,
                    self.periodic_box_vectors.as_ref(),
                ) {
                    *result.entry(name).or_insert(0.0) += value;
                }
            }
        }
        result
    }
}
