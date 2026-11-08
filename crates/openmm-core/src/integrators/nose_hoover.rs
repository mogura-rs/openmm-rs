// Algorithms adapted from OpenMM ReferenceNoseHooverChain and ReferenceNoseHooverDynamics.
// Copyright (c) 2006-2024 Stanford University and Simbios.
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
use super::{langevin::BOLTZMANN, Integrator};
use crate::constraints::{ConstraintError, ConstraintSolver};
use crate::{CMMotionRemover, System};
use burn::prelude::*;
use std::sync::Mutex;

#[derive(Clone, Debug)]
pub struct NoseHooverChain {
    pub temperature: f64,
    pub relative_temperature: f64,
    pub collision_frequency: f64,
    pub relative_collision_frequency: f64,
    pub num_degrees_of_freedom: i32,
    pub chain_length: usize,
    pub num_mts: usize,
    pub num_ys: usize,
    pub chain_id: usize,
    pub atoms: Vec<usize>,
    pub pairs: Vec<(usize, usize)>,
}

impl NoseHooverChain {
    pub fn validate(&self) -> Result<(), ConstraintError> {
        for value in [
            self.temperature,
            self.relative_temperature,
            self.collision_frequency,
            self.relative_collision_frequency,
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(ConstraintError(
                    "Nose-Hoover temperatures and frequencies must be positive and finite".into(),
                ));
            }
        }
        if self.chain_length == 0 || self.num_mts == 0 || ![1, 3, 5, 7].contains(&self.num_ys) {
            return Err(ConstraintError("chain length and MTS count must be positive; Yoshida-Suzuki count must be 1, 3, 5 or 7".into()));
        }
        Ok(())
    }

    pub fn yoshida_suzuki_weights(&self) -> &'static [f64] {
        match self.num_ys {
            1 => &[1.0],
            3 => &[0.828981543588751, -0.657963087177502, 0.828981543588751],
            5 => &[
                0.2967324292201065,
                0.2967324292201065,
                -0.186929716880426,
                0.2967324292201065,
                0.2967324292201065,
            ],
            7 => &[
                0.784513610477560,
                0.235573213359357,
                -1.17767998417887,
                1.31518632068391,
                -1.17767998417887,
                0.235573213359357,
                0.784513610477560,
            ],
            _ => panic!("invalid Yoshida-Suzuki count"),
        }
    }
}

#[derive(Default)]
struct ChainState {
    positions: Vec<f64>,
    velocities: Vec<f64>,
    forces: Vec<f64>,
    masses: Vec<f64>,
}

impl ChainState {
    #[allow(clippy::too_many_arguments)]
    fn propagate(
        &mut self,
        kinetic: f64,
        dof: i32,
        temperature: f64,
        frequency: f64,
        dt: f64,
        chain: &NoseHooverChain,
    ) -> f64 {
        let n = chain.chain_length;
        self.positions.resize(n, 0.0);
        self.velocities.resize(n, 0.0);
        self.forces.resize(n, 0.0);
        self.masses.resize(n, 0.0);
        let kt = BOLTZMANN * temperature;
        self.masses.fill(kt / (frequency * frequency));
        self.masses[0] *= dof as f64;
        let ke2 = 2.0 * kinetic;
        self.forces[0] = (ke2 - dof as f64 * kt) / self.masses[0];
        for i in 0..n - 1 {
            self.forces[i + 1] = (self.masses[i] * self.velocities[i] * self.velocities[i] - kt)
                / self.masses[i + 1];
        }
        let mut scale = 1.0;
        for _ in 0..chain.num_mts {
            for &weight in chain.yoshida_suzuki_weights() {
                let h = weight * dt / chain.num_mts as f64;
                self.velocities[n - 1] += 0.5 * h * self.forces[n - 1];
                for i in (0..n - 1).rev() {
                    let a = (-0.25 * h * self.velocities[i + 1]).exp();
                    self.velocities[i] = a * (self.velocities[i] * a + 0.5 * h * self.forces[i]);
                }
                scale *= (-h * self.velocities[0]).exp();
                for i in 0..n {
                    self.positions[i] += self.velocities[i] * h;
                }
                self.forces[0] = (scale * scale * ke2 - dof as f64 * kt) / self.masses[0];
                for i in 0..n - 1 {
                    let a = (-0.25 * h * self.velocities[i + 1]).exp();
                    self.velocities[i] = a * (a * self.velocities[i] + 0.5 * h * self.forces[i]);
                    self.forces[i + 1] = (self.masses[i] * self.velocities[i] * self.velocities[i]
                        - kt)
                        / self.masses[i + 1];
                }
                self.velocities[n - 1] += 0.5 * h * self.forces[n - 1];
            }
        }
        scale
    }

    fn energy(&self, dof: i32, temperature: f64, frequency: f64) -> f64 {
        let kt = BOLTZMANN * temperature;
        let mut kinetic = 0.0;
        let mut potential = 0.0;
        for (i, (&x, &v)) in self.positions.iter().zip(&self.velocities).enumerate() {
            let prefactor = if i == 0 { dof as f64 } else { 1.0 };
            kinetic += 0.5 * (prefactor * kt / (frequency * frequency)) * v * v;
            potential += prefactor * kt * x;
        }
        kinetic + potential
    }
}

/// Middle-splitting Nose-Hoover chains with separate pair COM and relative baths.
pub struct NoseHooverIntegrator {
    dt: f64,
    tolerance: f64,
    groups: u32,
    maximum_pair_distance: f64,
    subsystem: bool,
    initialized: bool,
    chains: Vec<NoseHooverChain>,
    all_atoms: Vec<usize>,
    all_pairs: Vec<(usize, usize, f64)>,
    states: Mutex<Vec<(ChainState, ChainState)>>,
}

impl NoseHooverIntegrator {
    pub fn new(step_size: f64) -> Self {
        assert!(step_size.is_finite() && step_size > 0.0);
        Self {
            dt: step_size,
            tolerance: 1e-5,
            groups: u32::MAX,
            maximum_pair_distance: 0.0,
            subsystem: true,
            initialized: false,
            chains: Vec::new(),
            all_atoms: Vec::new(),
            all_pairs: Vec::new(),
            states: Mutex::new(Vec::new()),
        }
    }
    pub fn thermostated(
        temperature: f64,
        frequency: f64,
        step_size: f64,
        chain_length: usize,
        num_mts: usize,
        num_ys: usize,
    ) -> Result<Self, ConstraintError> {
        let mut result = Self::new(step_size);
        result.add_thermostat(temperature, frequency, chain_length, num_mts, num_ys)?;
        Ok(result)
    }
    pub fn add_thermostat(
        &mut self,
        temperature: f64,
        frequency: f64,
        chain_length: usize,
        num_mts: usize,
        num_ys: usize,
    ) -> Result<usize, ConstraintError> {
        let index = self.add_subsystem_thermostat(
            Vec::new(),
            Vec::new(),
            temperature,
            frequency,
            temperature,
            frequency,
            chain_length,
            num_mts,
            num_ys,
        )?;
        self.subsystem = false;
        Ok(index)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn add_subsystem_thermostat(
        &mut self,
        atoms: Vec<usize>,
        pairs: Vec<(usize, usize)>,
        temperature: f64,
        frequency: f64,
        relative_temperature: f64,
        relative_frequency: f64,
        chain_length: usize,
        num_mts: usize,
        num_ys: usize,
    ) -> Result<usize, ConstraintError> {
        if self.initialized {
            return Err(ConstraintError(
                "cannot add a thermostat after initialization".into(),
            ));
        }
        if self
            .chains
            .iter()
            .any(|c| c.atoms.is_empty() && c.pairs.is_empty())
        {
            return Err(ConstraintError(
                "an existing thermostat already applies to all particles".into(),
            ));
        }
        let index = self.chains.len();
        let chain = NoseHooverChain {
            temperature,
            relative_temperature,
            collision_frequency: frequency,
            relative_collision_frequency: relative_frequency,
            num_degrees_of_freedom: 0,
            chain_length,
            num_mts,
            num_ys,
            chain_id: index,
            atoms,
            pairs,
        };
        chain.validate()?;
        self.chains.push(chain);
        Ok(index)
    }
    pub fn num_thermostats(&self) -> usize {
        self.chains.len()
    }
    pub fn has_subsystem_thermostats(&self) -> bool {
        self.subsystem
    }
    pub fn thermostat(&self, index: usize) -> &NoseHooverChain {
        &self.chains[index]
    }
    pub fn set_thermostat(
        &mut self,
        index: usize,
        chain: NoseHooverChain,
    ) -> Result<(), ConstraintError> {
        chain.validate()?;
        let old = self
            .chains
            .get(index)
            .ok_or_else(|| ConstraintError("thermostat index out of range".into()))?;
        if chain.chain_length != old.chain_length
            || chain.num_mts != old.num_mts
            || chain.num_ys != old.num_ys
            || chain.chain_id != old.chain_id
        {
            return Err(ConstraintError("chain structure is immutable".into()));
        }
        if self.initialized
            && (chain.atoms != old.atoms
                || chain.pairs != old.pairs
                || (chain.num_degrees_of_freedom <= 0
                    && (!chain.atoms.is_empty() || !chain.pairs.is_empty())))
        {
            return Err(ConstraintError("cannot change thermostat particles after initialization; degrees of freedom must be positive".into()));
        }
        self.chains[index] = chain;
        Ok(())
    }
    pub fn maximum_pair_distance(&self) -> f64 {
        self.maximum_pair_distance
    }
    pub fn set_maximum_pair_distance(&mut self, value: f64) {
        assert!(value.is_finite() && value >= 0.0);
        self.maximum_pair_distance = value;
    }
    pub fn all_thermostated_individual_particles(&self) -> &[usize] {
        &self.all_atoms
    }
    pub fn all_thermostated_pairs(&self) -> &[(usize, usize, f64)] {
        &self.all_pairs
    }
    pub fn integration_force_groups(&self) -> u32 {
        self.groups
    }
    pub fn set_integration_force_groups(&mut self, groups: u32) {
        self.groups = groups;
    }
    pub fn set_step_size(&mut self, value: f64) {
        assert!(value.is_finite() && value > 0.0);
        self.dt = value;
    }
    pub fn set_constraint_tolerance(&mut self, value: f64) {
        assert!(value.is_finite() && value > 0.0 && value < 1.0);
        self.tolerance = value;
    }
    pub fn compute_heat_bath_energy(&self) -> f64 {
        let states = self.states.lock().unwrap();
        self.chains
            .iter()
            .zip(states.iter())
            .map(|(c, (absolute, relative))| {
                if c.num_degrees_of_freedom <= 0 {
                    return 0.0;
                }
                absolute.energy(
                    c.num_degrees_of_freedom,
                    c.temperature,
                    c.collision_frequency,
                ) + relative.energy(
                    3 * c.pairs.len() as i32,
                    c.relative_temperature,
                    c.relative_collision_frequency,
                )
            })
            .sum()
    }

    fn configured<B: Backend>(
        &self,
        system: &System<B>,
    ) -> Result<Vec<NoseHooverChain>, ConstraintError> {
        let n = system.num_particles();
        let mut chains = self.chains.clone();
        let mut owners = vec![None; n];
        for (index, chain) in chains.iter_mut().enumerate() {
            chain.validate()?;
            if chain.atoms.is_empty() && chain.pairs.is_empty() {
                chain.atoms = (0..n).filter(|&a| system.particle(a).mass > 0.0).collect();
            }
            for &atom in &chain.atoms {
                if atom >= n || system.particle(atom).mass == 0.0 {
                    return Err(ConstraintError(
                        "thermostat atoms must exist and have positive mass".into(),
                    ));
                }
            }
            for &(a, b) in &chain.pairs {
                if a >= n || b >= n || a == b || system.particle(a).mass <= 0.0 {
                    return Err(ConstraintError(
                        "thermostat pairs require distinct particles and a massive first particle"
                            .into(),
                    ));
                }
            }
            for atom in chain
                .atoms
                .iter()
                .copied()
                .chain(chain.pairs.iter().flat_map(|&(a, b)| [a, b]))
            {
                if owners[atom].replace(index).is_some() {
                    return Err(ConstraintError(
                        "each particle can occur in only one thermostat entry".into(),
                    ));
                }
            }
            chain.num_degrees_of_freedom = 3 * (chain.atoms.len() + chain.pairs.len()) as i32;
        }
        for constraint in system.constraints() {
            let (a, b) = (constraint.particle1, constraint.particle2);
            if system.particle(a).mass > 0.0 && system.particle(b).mass > 0.0 {
                if owners[a] != owners[b] {
                    return Err(ConstraintError(
                        "constrained particles must belong to the same thermostat".into(),
                    ));
                }
                if let Some(index) = owners[a] {
                    chains[index].num_degrees_of_freedom -= 1;
                }
            }
        }
        if !chains.is_empty()
            && system
                .forces()
                .iter()
                .any(|f| f.as_any().is::<CMMotionRemover>())
        {
            let mut largest = 0;
            for i in 1..chains.len() {
                if chains[i].atoms.len() + chains[i].pairs.len()
                    > chains[largest].atoms.len() + chains[largest].pairs.len()
                {
                    largest = i;
                }
            }
            chains[largest].num_degrees_of_freedom -= 3;
        }
        if chains
            .iter()
            .any(|c| c.num_degrees_of_freedom <= 0 && (!c.atoms.is_empty() || !c.pairs.is_empty()))
        {
            return Err(ConstraintError(
                "an active thermostat requires positive degrees of freedom".into(),
            ));
        }
        Ok(chains)
    }

    fn kick(&self, velocity: &mut [f64], force: &[f64], mass: &[f64]) {
        for &a in &self.all_atoms {
            if mass[a] > 0.0 {
                for k in 0..3 {
                    velocity[3 * a + k] += force[3 * a + k] * mass[a].recip() * self.dt;
                }
            }
        }
        for &(a, b, _) in &self.all_pairs {
            let inv_total = (mass[a] + mass[b]).recip();
            let fa = mass[a] * inv_total;
            let fb = mass[b] * inv_total;
            let inv_reduced = if mass[a] * mass[b] != 0.0 {
                (mass[a] + mass[b]) / (mass[a] * mass[b])
            } else {
                0.0
            };
            for k in 0..3 {
                let ia = 3 * a + k;
                let ib = 3 * b + k;
                let com = velocity[ia] * fa
                    + velocity[ib] * fb
                    + (force[ia] + force[ib]) * self.dt * inv_total;
                let relative = velocity[ib] - velocity[ia]
                    + (fa * force[ib] - fb * force[ia]) * self.dt * inv_reduced;
                if mass[a] > 0.0 {
                    velocity[ia] = com - relative * fb;
                }
                if mass[b] > 0.0 {
                    velocity[ib] = com + relative * fa;
                }
            }
        }
    }
    fn drift_thermostat(&self, position: &mut [f64], velocity: &mut [f64], mass: &[f64]) {
        let half_dt = 0.5 * self.dt;
        for (a, &m) in mass.iter().enumerate() {
            if m > 0.0 {
                for k in 0..3 {
                    position[3 * a + k] += velocity[3 * a + k] * half_dt;
                }
            }
        }
        let mut states = self.states.lock().unwrap();
        for (chain, (absolute, relative)) in self.chains.iter().zip(states.iter_mut()) {
            let mut abs_ke = 0.0;
            let mut rel_ke = 0.0;
            for &a in &chain.atoms {
                let norm = (0..3)
                    .map(|k| velocity[3 * a + k] * velocity[3 * a + k])
                    .sum::<f64>();
                abs_ke += 0.5 * mass[a] * norm;
            }
            for &(a, b) in &chain.pairs {
                let inv_mass = (mass[a] + mass[b]).recip();
                let fa = mass[a] * inv_mass;
                let fb = mass[b] * inv_mass;
                let mut com2 = 0.0;
                let mut relative2 = 0.0;
                for k in 0..3 {
                    let com = fa * velocity[3 * a + k] + fb * velocity[3 * b + k];
                    let relative = velocity[3 * b + k] - velocity[3 * a + k];
                    com2 += com * com;
                    relative2 += relative * relative;
                }
                abs_ke += 0.5 * (mass[a] + mass[b]) * com2;
                rel_ke += 0.5 * (mass[a] * mass[b] * inv_mass) * relative2;
            }
            let abs_scale = if abs_ke < 1e-8 {
                1.0
            } else {
                absolute.propagate(
                    abs_ke,
                    chain.num_degrees_of_freedom,
                    chain.temperature,
                    chain.collision_frequency,
                    self.dt,
                    chain,
                )
            };
            let rel_scale = if chain.pairs.is_empty() {
                1.0
            } else {
                relative.propagate(
                    rel_ke,
                    3 * chain.pairs.len() as i32,
                    chain.relative_temperature,
                    chain.relative_collision_frequency,
                    self.dt,
                    chain,
                )
            };
            for &a in &chain.atoms {
                for k in 0..3 {
                    velocity[3 * a + k] *= abs_scale;
                }
            }
            for &(a, b) in &chain.pairs {
                let inv_mass = (mass[a] + mass[b]).recip();
                let fa = mass[a] * inv_mass;
                let fb = mass[b] * inv_mass;
                for k in 0..3 {
                    let com = fa * velocity[3 * a + k] + fb * velocity[3 * b + k];
                    let relative = velocity[3 * b + k] - velocity[3 * a + k];
                    velocity[3 * a + k] = abs_scale * com - rel_scale * relative * fb;
                    velocity[3 * b + k] = abs_scale * com + rel_scale * relative * fa;
                }
            }
        }
        for (a, &m) in mass.iter().enumerate() {
            if m > 0.0 {
                for k in 0..3 {
                    position[3 * a + k] += velocity[3 * a + k] * half_dt;
                }
            }
        }
    }
    fn hard_wall(&self, position: &mut [f64], velocity: &mut [f64], mass: &[f64]) {
        if self.maximum_pair_distance == 0.0 {
            return;
        }
        for &(a, b, temperature) in &self.all_pairs {
            let delta: [f64; 3] =
                std::array::from_fn(|k| position[3 * a + k] - position[3 * b + k]);
            let r = delta.iter().map(|x| x * x).sum::<f64>().sqrt();
            if r <= self.maximum_pair_distance {
                continue;
            }
            let direction = delta.map(|x| x / r);
            let excess = r - self.maximum_pair_distance;
            let scale = (temperature * BOLTZMANN / mass[a]).sqrt();
            let va = (0..3)
                .map(|k| velocity[3 * a + k] * direction[k])
                .sum::<f64>();
            let vb = (0..3)
                .map(|k| velocity[3 * b + k] * direction[k])
                .sum::<f64>();
            if mass[b] == 0.0 {
                let h = if va == 0.0 {
                    self.dt
                } else {
                    (excess / va.abs()).min(self.dt)
                };
                let new_va = if va == 0.0 {
                    -scale
                } else {
                    -va.signum() * scale
                };
                for k in 0..3 {
                    position[3 * a + k] += direction[k] * (-excess + h * new_va);
                    velocity[3 * a + k] += direction[k] * (new_va - va);
                }
            } else {
                let inv_total = (mass[a] + mass[b]).recip();
                let com = (mass[a] * va + mass[b] * vb) * inv_total;
                let ra = va - com;
                let rb = vb - com;
                let h = if ra == rb {
                    self.dt
                } else {
                    (excess / (ra - rb).abs()).min(self.dt)
                };
                let new_ra = if ra == 0.0 && rb == 0.0 {
                    -scale * mass[b] * inv_total
                } else {
                    -ra.signum() * scale * mass[b] * inv_total
                };
                let new_rb = if ra == 0.0 && rb == 0.0 {
                    scale * mass[a] * inv_total
                } else {
                    -rb.signum() * scale * mass[a] * inv_total
                };
                for k in 0..3 {
                    position[3 * a + k] +=
                        direction[k] * (-excess * mass[b] * inv_total + h * new_ra);
                    position[3 * b + k] +=
                        direction[k] * (excess * mass[a] * inv_total + h * new_rb);
                    velocity[3 * a + k] += direction[k] * (new_ra + com - va);
                    velocity[3 * b + k] += direction[k] * (new_rb + com - vb);
                }
            }
        }
    }
    fn update<B: Backend>(
        &self,
        p: &mut Tensor<B, 2>,
        v: &mut Tensor<B, 2>,
        f: &Tensor<B, 2>,
        m: &Tensor<B, 1>,
        phase: u8,
    ) {
        let run = |p: &mut [f64], v: &mut [f64], f: &[f64], m: &[f64]| {
            if phase == 0 || phase == 1 {
                self.kick(v, f, m);
            }
            if phase == 0 || phase == 2 {
                self.drift_thermostat(p, v, m);
            }
            if phase == 0 || phase == 3 {
                self.hard_wall(p, v, m);
            }
        };
        if super::try_host_step(p, v, f, m, run).is_some() {
            return;
        }
        let mut position = crate::tensor::to_f64_vec(p);
        let mut velocity = crate::tensor::to_f64_vec(v);
        run(
            &mut position,
            &mut velocity,
            &crate::tensor::to_f64_vec(f),
            &crate::tensor::to_f64_vec(m),
        );
        *p = crate::tensor::from_f64_vec(position, p.dims(), &p.device());
        *v = crate::tensor::from_f64_vec(velocity, v.dims(), &v.device());
    }
}

impl<B: Backend> Integrator<B> for NoseHooverIntegrator {
    fn validate_system(&self, system: &System<B>) -> Result<(), ConstraintError> {
        self.configured(system).map(|_| ())
    }
    fn initialize_system(&mut self, system: &System<B>) {
        self.chains = self
            .configured(system)
            .expect("invalid Nose-Hoover configuration");
        self.all_pairs = self
            .chains
            .iter()
            .flat_map(|c| c.pairs.iter().map(|&(a, b)| (a, b, c.relative_temperature)))
            .collect();
        let mut paired = vec![false; system.num_particles()];
        for &(a, b, _) in &self.all_pairs {
            paired[a] = true;
            paired[b] = true;
        }
        self.all_atoms = (0..system.num_particles())
            .filter(|&a| !paired[a])
            .collect();
        *self.states.get_mut().unwrap() = self
            .chains
            .iter()
            .map(|_| (ChainState::default(), ChainState::default()))
            .collect();
        self.initialized = true;
    }
    fn step_size(&self) -> f64 {
        self.dt
    }
    fn set_step_size(&mut self, value: f64) {
        Self::set_step_size(self, value);
    }
    fn constraint_tolerance(&self) -> f64 {
        self.tolerance
    }
    fn set_constraint_tolerance(&mut self, value: f64) {
        Self::set_constraint_tolerance(self, value);
    }
    fn integration_force_groups(&self) -> u32 {
        self.groups
    }
    fn set_integration_force_groups(&mut self, value: u32) {
        self.groups = value;
    }
    fn step(&self, p: &mut Tensor<B, 2>, v: &mut Tensor<B, 2>, f: &Tensor<B, 2>, m: &Tensor<B, 1>) {
        assert!(
            self.initialized,
            "NoseHooverIntegrator requires system initialization"
        );
        self.update(p, v, f, m, 0);
    }
    fn step_constrained(
        &self,
        p: &mut Tensor<B, 2>,
        v: &mut Tensor<B, 2>,
        f: &Tensor<B, 2>,
        m: &Tensor<B, 1>,
        c: &ConstraintSolver,
    ) -> Result<(), ConstraintError> {
        if !self.initialized {
            return Err(ConstraintError(
                "NoseHooverIntegrator requires system initialization".into(),
            ));
        }
        if c.is_empty() {
            self.step(p, v, f, m);
            return Ok(());
        }
        let previous = p.clone();
        self.update(p, v, f, m, 1);
        c.apply_velocities(p, v, self.tolerance)?;
        self.update(p, v, f, m, 2);
        let proposed = p.clone();
        c.apply_positions(&previous, p, self.tolerance)?;
        super::velocities_from_positions(p, &proposed, v, m, self.dt, true);
        self.update(p, v, f, m, 3);
        Ok(())
    }
}
