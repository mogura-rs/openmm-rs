use super::Context;
use crate::forces::anisotropic_barostat::MonteCarloAnisotropicBarostat;
use crate::forces::barostat::{
    scaled_box, scaled_box_axes, volume, MonteCarloBarostat, BAR_NM3_TO_KJ_PER_MOL,
};
use crate::forces::flexible_barostat::{reduced_box, MonteCarloFlexibleBarostat};
use crate::forces::membrane_barostat::MonteCarloMembraneBarostat;
use burn::prelude::*;

const COMPONENTS: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];
const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

impl<B: Backend> Context<B> {
    pub fn compute_current_pressure(&self, force_index: usize) -> Result<f64, String> {
        let barostat = self
            .system
            .forces()
            .get(force_index)
            .and_then(|force| force.as_any().downcast_ref::<MonteCarloBarostat>())
            .ok_or_else(|| "force index does not identify a MonteCarloBarostat".to_owned())?;
        let vectors = self
            .periodic_box_vectors
            .as_ref()
            .ok_or_else(|| "a barostat requires a periodic box".to_owned())?;
        let rigid = barostat.scale_molecules_as_rigid();
        let energy = |scale: f64| {
            let transform = [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, scale]];
            let positions = self.scaled_positions(transform, rigid);
            self.energy_at(
                &positions,
                &scaled_box(vectors, scale),
                self.integrator.integration_force_groups(),
            )
        };
        let difference = energy(1.001) - energy(0.999);
        let kinetic: f64 = self.pressure_kinetic_tensor(rigid).iter().take(3).sum();
        let current_volume = volume(vectors);
        let delta_volume = current_volume * (1.001_f64.powi(3) - 0.999_f64.powi(3));
        Ok(
            ((2.0 / 3.0) * kinetic / current_volume - difference / delta_volume)
                / BAR_NM3_TO_KJ_PER_MOL,
        )
    }

    pub fn compute_current_anisotropic_pressure(
        &self,
        force_index: usize,
    ) -> Result<[f64; 3], String> {
        let barostat = self
            .system
            .forces()
            .get(force_index)
            .and_then(|force| {
                force
                    .as_any()
                    .downcast_ref::<MonteCarloAnisotropicBarostat>()
            })
            .ok_or_else(|| {
                "force index does not identify a MonteCarloAnisotropicBarostat".to_owned()
            })?;
        self.axis_pressures(barostat.scale_molecules_as_rigid())
    }

    pub fn compute_current_membrane_pressure(
        &self,
        force_index: usize,
    ) -> Result<[f64; 3], String> {
        let barostat = self
            .system
            .forces()
            .get(force_index)
            .and_then(|force| force.as_any().downcast_ref::<MonteCarloMembraneBarostat>())
            .ok_or_else(|| {
                "force index does not identify a MonteCarloMembraneBarostat".to_owned()
            })?;
        self.axis_pressures(barostat.scale_molecules_as_rigid())
    }

    fn axis_pressures(&self, rigid: bool) -> Result<[f64; 3], String> {
        let vectors = self
            .periodic_box_vectors
            .as_ref()
            .ok_or_else(|| "a barostat requires a periodic box".to_owned())?;
        let kinetic = self.pressure_kinetic_tensor(rigid);
        let current_volume = volume(vectors);
        Ok(std::array::from_fn(|axis| {
            let energy = |factor: f64| {
                let mut scale = [1.0; 3];
                scale[axis] = factor;
                let transform = [
                    [scale[0], 0.0, 0.0],
                    [0.0, scale[1], 0.0],
                    [0.0, 0.0, scale[2]],
                ];
                self.energy_at(
                    &self.scaled_positions(transform, rigid),
                    &scaled_box_axes(vectors, scale),
                    self.integrator.integration_force_groups(),
                )
            };
            (2.0 * kinetic[axis] / current_volume
                - (energy(1.001) - energy(0.999)) / (0.002 * current_volume))
                / BAR_NM3_TO_KJ_PER_MOL
        }))
    }

    fn pressure_kinetic_tensor(&self, rigid: bool) -> [f64; 6] {
        crate::tensor::with_f64_data(&self.velocities, |velocities| {
            let mut kinetic = [0.0; 6];
            if rigid {
                for molecule in self.molecules() {
                    let mut momentum = [0.0; 3];
                    let mut mass = 0.0;
                    for &atom in molecule {
                        mass += self.host_masses[atom];
                        for axis in 0..3 {
                            momentum[axis] += self.host_masses[atom] * velocities[3 * atom + axis];
                        }
                    }
                    if mass > 0.0 {
                        for (component, (i, j)) in COMPONENTS.iter().copied().enumerate() {
                            kinetic[component] += 0.5 * momentum[i] * momentum[j] / mass;
                        }
                    }
                }
            } else {
                for (mass, velocity) in self.host_masses.iter().zip(velocities.chunks_exact(3)) {
                    for (component, (i, j)) in COMPONENTS.iter().copied().enumerate() {
                        kinetic[component] += 0.5 * mass * velocity[i] * velocity[j];
                    }
                }
            }
            kinetic
        })
    }

    pub fn compute_current_flexible_pressure(
        &self,
        force_index: usize,
    ) -> Result<[f64; 6], String> {
        let rigid = self.flexible_rigid_mode(force_index)?;
        let vectors = self
            .periodic_box_vectors
            .as_ref()
            .ok_or_else(|| "a barostat requires a periodic box".to_owned())?;
        let kinetic = self.pressure_kinetic_tensor(rigid);
        let current_volume = volume(vectors);
        let groups = self.integrator.integration_force_groups();
        let mut pressure = [0.0; 6];
        for (component, (column, row)) in COMPONENTS.iter().copied().enumerate() {
            let mut energies = [0.0; 2];
            if component < 3 || vectors[row][column] != 0.0 {
                for (sign, delta) in [0.001, -0.001].into_iter().enumerate() {
                    let mut trial = *vectors;
                    trial[row][column] *= 1.0 + delta;
                    let trial = reduced_box(trial)
                        .ok_or_else(|| "invalid pressure trial box".to_owned())?;
                    energies[sign] = if component < 3 {
                        let mut transform = IDENTITY;
                        transform[component][component] += delta;
                        self.energy_at(&self.scaled_positions(transform, rigid), &trial, groups)
                    } else {
                        self.energy_at(&self.positions, &trial, groups)
                    };
                }
            }
            pressure[component] = (2.0 * kinetic[component] - (energies[0] - energies[1]) / 0.002)
                / (current_volume * BAR_NM3_TO_KJ_PER_MOL);
        }
        Ok(pressure)
    }

    pub fn compute_stress_tensor(
        &self,
        force_index: usize,
        include_kinetic: bool,
    ) -> Result<[f64; 6], String> {
        let rigid = self.flexible_rigid_mode(force_index)?;
        let vectors = self
            .periodic_box_vectors
            .as_ref()
            .ok_or_else(|| "a barostat requires a periodic box".to_owned())?;
        let kinetic = if include_kinetic {
            self.pressure_kinetic_tensor(rigid)
        } else {
            [0.0; 6]
        };
        let current_volume = volume(vectors);
        let groups = self.integrator.integration_force_groups();
        let mut stress = [0.0; 6];
        for (component, (i, j)) in COMPONENTS.iter().copied().enumerate() {
            let mut energies = [0.0; 2];
            for (sign, delta) in [0.001, -0.001].into_iter().enumerate() {
                let mut trial = *vectors;
                for vector in &mut trial {
                    vector[i] += delta * vector[j];
                }
                let trial =
                    reduced_box(trial).ok_or_else(|| "invalid stress trial box".to_owned())?;
                let mut transform = IDENTITY;
                transform[i][j] += delta;
                energies[sign] =
                    self.energy_at(&self.scaled_positions(transform, rigid), &trial, groups);
            }
            stress[component] = ((energies[0] - energies[1]) / 0.002 - 2.0 * kinetic[component])
                / (current_volume * BAR_NM3_TO_KJ_PER_MOL);
        }
        Ok(stress)
    }

    fn flexible_rigid_mode(&self, force_index: usize) -> Result<bool, String> {
        self.system
            .forces()
            .get(force_index)
            .and_then(|force| force.as_any().downcast_ref::<MonteCarloFlexibleBarostat>())
            .map(MonteCarloFlexibleBarostat::scale_molecules_as_rigid)
            .ok_or_else(|| "force index does not identify a MonteCarloFlexibleBarostat".to_owned())
    }
}
