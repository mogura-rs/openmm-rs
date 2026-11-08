// DPD splitting and voxel traversal adapted from OpenMM ReferenceDPDDynamics
// and ReferenceNeighborList. Copyright (c) 2025 Stanford University and Simbios.
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
use super::{langevin::BOLTZMANN, random::GaussianStream, Integrator, StepEnvironment};
use crate::constraints::{ConstraintError, ConstraintSolver};
use crate::System;
use burn::prelude::*;
use rand::Rng;
use rand_distr::StandardNormal;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Mutex;

type BoxVectors = [[f64; 3]; 3];
type TypePair = (i32, i32, f64, f64);

fn pair_key(a: i32, b: i32) -> (i32, i32) {
    (a.min(b), a.max(b))
}
fn delta(p: &[f64], i: usize, j: usize, vectors: Option<&BoxVectors>) -> [f64; 3] {
    let mut d = std::array::from_fn(|k| p[3 * j + k] - p[3 * i + k]);
    if let Some(b) = vectors {
        for axis in (0..3).rev() {
            let shift = (d[axis] / b[axis][axis] + 0.5).floor();
            for k in 0..3 {
                d[k] -= shift * b[axis][k];
            }
        }
    }
    d
}
fn norm2(d: [f64; 3]) -> f64 {
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

struct Grid<'a> {
    sizes: [f64; 3],
    counts: [i64; 3],
    radius: [i64; 3],
    vectors: Option<&'a BoxVectors>,
}
impl<'a> Grid<'a> {
    fn new(cutoff: f64, vectors: Option<&'a BoxVectors>) -> Result<Self, ConstraintError> {
        let mut counts = [1; 3];
        let mut sizes = [cutoff; 3];
        if let Some(b) = vectors {
            for k in 0..3 {
                let cells = 2.0 * (b[k][k] / cutoff).floor();
                if !cells.is_finite() || cells < 2.0 || cells > (i32::MAX / 4) as f64 {
                    return Err(ConstraintError("DPD cutoff must not exceed a periodic box diagonal, and the voxel count must be representable".into()));
                }
                counts[k] = cells as i64;
                sizes[k] = b[k][k] / counts[k] as f64;
            }
        }
        Ok(Self {
            counts,
            sizes,
            radius: std::array::from_fn(|k| (cutoff / sizes[k]) as i64 + 1),
            vectors,
        })
    }
    fn traversal_rank(&self, center: [i64; 3], neighbor: [i64; 3]) -> Option<[i64; 3]> {
        let [nx, ny, nz] = self.counts;
        let [dx, dy, dz] = self.radius;
        let choose = |key: i64, low: i64, high: i64, count: i64| {
            let index = if self.vectors.is_some() {
                low + (key - low).rem_euclid(count)
            } else {
                key
            };
            if index < low
                || index > high
                || (self.vectors.is_some() && (index + count) % count != key)
            {
                None
            } else {
                Some(index)
            }
        };
        let z = choose(neighbor[2], center[2] - dz, center[2] + dz, nz)?;
        let boxz = (z as f32 / nz as f32).floor() as i64;
        let mut miny = center[1] - dy;
        let mut maxy = center[1] + dy;
        if let Some(b) = self.vectors {
            let offset = boxz as f64 * b[2][1] / self.sizes[1];
            miny -= offset.ceil() as i64;
            maxy -= offset.floor() as i64;
        }
        let y = choose(neighbor[1], miny, maxy, ny)?;
        let boxy = (y as f32 / ny as f32).floor() as i64;
        let mut minx = center[0] - dx;
        let mut maxx = center[0] + dx;
        if let Some(b) = self.vectors {
            let offset = (boxy as f64 * b[1][0] + boxz as f64 * b[2][0]) / self.sizes[0];
            minx -= offset.ceil() as i64;
            maxx -= offset.floor() as i64;
        }
        let x = choose(neighbor[0], minx, maxx, nx)?;
        Some([z, y, x])
    }
    fn index(&self, p: &[f64], atom: usize) -> [i64; 3] {
        let mut r = [p[3 * atom], p[3 * atom + 1], p[3 * atom + 2]];
        if let Some(b) = self.vectors {
            for axis in (0..3).rev() {
                let shift = (r[axis] / b[axis][axis]).floor();
                for k in 0..3 {
                    r[k] -= shift * b[axis][k];
                }
            }
        }
        std::array::from_fn(|k| {
            let value = (r[k] / self.sizes[k]).floor() as i64;
            if self.vectors.is_some() {
                value.min(self.counts[k] - 1)
            } else {
                value
            }
        })
    }
}

#[derive(Default)]
struct Neighbors {
    cells: HashMap<[i64; 3], (usize, usize)>,
    dense: Vec<(usize, usize)>,
    small: Vec<([i64; 3], usize)>,
    voxels: Vec<[i64; 3]>,
    next: Vec<usize>,
}
impl Neighbors {
    fn dense_index(key: [i64; 3], counts: [i64; 3]) -> Option<usize> {
        if (0..3).any(|k| key[k] < 0 || key[k] >= counts[k]) {
            return None;
        }
        Some((key[0] + counts[0] * (key[1] + counts[1] * key[2])) as usize)
    }
    fn visit(
        &mut self,
        p: &[f64],
        cutoff: f64,
        vectors: Option<&BoxVectors>,
        mut pair: impl FnMut(usize, usize) -> Result<(), ConstraintError>,
    ) -> Result<(), ConstraintError> {
        let grid = Grid::new(cutoff, vectors)?;
        if p.len() / 3 <= 32 {
            self.voxels.clear();
            self.voxels
                .extend((0..p.len() / 3).map(|i| grid.index(p, i)));
            for i in 0..self.voxels.len() {
                self.small.clear();
                for j in 0..i {
                    if norm2(delta(p, i, j, vectors)) <= cutoff * cutoff {
                        if let Some(rank) = grid.traversal_rank(self.voxels[i], self.voxels[j]) {
                            self.small.push((rank, j));
                        }
                    }
                }
                self.small.sort_unstable();
                for &(_, j) in &self.small {
                    pair(i, j)?;
                }
            }
            return Ok(());
        }
        self.cells.clear();
        let dense_size = if vectors.is_some() {
            grid.counts
                .iter()
                .try_fold(1_usize, |size, &count| size.checked_mul(count as usize))
                .filter(|&size| size <= p.len() && size <= 16 * 1024 * 1024)
        } else {
            None
        };
        if let Some(size) = dense_size {
            self.dense.resize(size, (usize::MAX, usize::MAX));
            self.dense.fill((usize::MAX, usize::MAX));
        }
        self.next.resize(p.len() / 3, usize::MAX);
        self.next.fill(usize::MAX);
        let [nx, ny, nz] = grid.counts;
        let [dx, dy, dz] = grid.radius;
        let periodic = vectors.is_some();
        for i in 0..p.len() / 3 {
            let center = grid.index(p, i);
            let minz = center[2] - dz;
            let maxz = if periodic {
                (center[2] + dz).min(minz + nz - 1)
            } else {
                center[2] + dz
            };
            for z in minz..=maxz {
                let boxz = (z as f32 / nz as f32).floor() as i64;
                let mut miny = center[1] - dy;
                let mut maxy = center[1] + dy;
                if let Some(b) = vectors {
                    let offset = boxz as f64 * b[2][1] / grid.sizes[1];
                    miny -= offset.ceil() as i64;
                    maxy = (maxy - offset.floor() as i64).min(miny + ny - 1);
                }
                for y in miny..=maxy {
                    let boxy = (y as f32 / ny as f32).floor() as i64;
                    let mut minx = center[0] - dx;
                    let mut maxx = center[0] + dx;
                    if let Some(b) = vectors {
                        let offset =
                            (boxy as f64 * b[1][0] + boxz as f64 * b[2][0]) / grid.sizes[0];
                        minx -= offset.ceil() as i64;
                        maxx = (maxx - offset.floor() as i64).min(minx + nx - 1);
                    }
                    for x in minx..=maxx {
                        let key = if periodic {
                            [(x + nx) % nx, (y + ny) % ny, (z + nz) % nz]
                        } else {
                            [x, y, z]
                        };
                        let cell = if dense_size.is_some() {
                            Self::dense_index(key, grid.counts)
                                .map(|index| self.dense[index])
                                .or_else(|| self.cells.get(&key).copied())
                        } else {
                            self.cells.get(&key).copied()
                        };
                        if let Some((mut j, _)) = cell {
                            while j != usize::MAX {
                                if norm2(delta(p, i, j, vectors)) <= cutoff * cutoff {
                                    pair(i, j)?;
                                }
                                j = self.next[j];
                            }
                        }
                    }
                }
            }
            let cell = if let Some(index) =
                dense_size.and_then(|_| Self::dense_index(center, grid.counts))
            {
                &mut self.dense[index]
            } else {
                self.cells.entry(center).or_insert((usize::MAX, usize::MAX))
            };
            if cell.1 == usize::MAX {
                cell.0 = i;
            } else {
                self.next[cell.1] = i;
            }
            cell.1 = i;
        }
        Ok(())
    }
}

struct Configuration {
    types: Vec<i32>,
    pairs: HashMap<(i32, i32), (f64, f64)>,
    defaults: (f64, f64),
    max_cutoff: f64,
    periodic: bool,
}
#[derive(Default)]
struct Scratch {
    midpoint: Vec<f64>,
    neighbors: Neighbors,
}

pub struct DPDIntegrator {
    dt: f64,
    temperature: f64,
    friction: f64,
    cutoff: f64,
    tolerance: f64,
    groups: u32,
    types: BTreeMap<i32, i32>,
    pairs: Vec<TypePair>,
    config: Option<Configuration>,
    random: GaussianStream,
    scratch: Mutex<Scratch>,
}
impl DPDIntegrator {
    pub fn new(
        temperature: f64,
        default_friction: f64,
        default_cutoff: f64,
        step_size: f64,
    ) -> Self {
        assert!(temperature.is_finite() && temperature >= 0.0);
        assert!(default_friction.is_finite() && default_friction >= 0.0);
        assert!(default_cutoff.is_finite() && default_cutoff > 0.0);
        assert!(step_size.is_finite() && step_size > 0.0);
        Self {
            dt: step_size,
            temperature,
            friction: default_friction,
            cutoff: default_cutoff,
            tolerance: 1e-5,
            groups: u32::MAX,
            types: BTreeMap::new(),
            pairs: Vec::new(),
            config: None,
            random: GaussianStream::default(),
            scratch: Mutex::new(Scratch::default()),
        }
    }
    pub fn temperature(&self) -> f64 {
        self.temperature
    }
    pub fn set_temperature(&mut self, value: f64) {
        assert!(value.is_finite() && value >= 0.0);
        self.temperature = value;
    }
    pub fn default_friction(&self) -> f64 {
        self.friction
    }
    pub fn set_default_friction(&mut self, value: f64) {
        assert!(value.is_finite() && value >= 0.0);
        self.friction = value;
    }
    pub fn default_cutoff(&self) -> f64 {
        self.cutoff
    }
    pub fn set_default_cutoff(&mut self, value: f64) {
        assert!(value.is_finite() && value > 0.0);
        self.cutoff = value;
    }
    pub fn set_step_size(&mut self, value: f64) {
        assert!(value.is_finite() && value > 0.0);
        self.dt = value;
    }
    pub fn set_constraint_tolerance(&mut self, value: f64) {
        assert!(value.is_finite() && value > 0.0 && value < 1.0);
        self.tolerance = value;
    }
    pub fn integration_force_groups(&self) -> u32 {
        self.groups
    }
    pub fn set_integration_force_groups(&mut self, value: u32) {
        self.groups = value;
    }
    pub fn random_number_seed(&self) -> i32 {
        self.random.seed()
    }
    pub fn set_random_number_seed(&mut self, value: i32) {
        self.random.set_seed(value);
    }
    pub fn particle_type(&self, index: i32) -> i32 {
        *self.types.get(&index).unwrap_or(&0)
    }
    pub fn set_particle_type(&mut self, index: i32, value: i32) {
        self.types.insert(index, value);
    }
    pub fn particle_types(&self) -> &BTreeMap<i32, i32> {
        &self.types
    }
    pub fn num_type_pairs(&self) -> usize {
        self.pairs.len()
    }
    pub fn add_type_pair(&mut self, a: i32, b: i32, friction: f64, cutoff: f64) -> usize {
        let index = self.pairs.len();
        self.pairs.push((a, b, friction, cutoff));
        index
    }
    pub fn type_pair_parameters(&self, index: usize) -> Result<TypePair, ConstraintError> {
        self.pairs
            .get(index)
            .copied()
            .ok_or_else(|| ConstraintError("DPD type pair index is out of range".into()))
    }
    pub fn set_type_pair_parameters(
        &mut self,
        index: usize,
        a: i32,
        b: i32,
        friction: f64,
        cutoff: f64,
    ) -> Result<(), ConstraintError> {
        *self
            .pairs
            .get_mut(index)
            .ok_or_else(|| ConstraintError("DPD type pair index is out of range".into()))? =
            (a, b, friction, cutoff);
        Ok(())
    }
    fn configure<B: Backend>(&self, system: &System<B>) -> Result<Configuration, ConstraintError> {
        let types: Vec<_> = (0..system.num_particles())
            .map(|i| self.particle_type(i as i32))
            .collect();
        let present: HashSet<_> = types.iter().copied().collect();
        let mut pairs = HashMap::new();
        let mut max_cutoff = self.cutoff;
        for &(a, b, friction, cutoff) in &self.pairs {
            if !friction.is_finite() || friction < 0.0 || !cutoff.is_finite() || cutoff <= 0.0 {
                return Err(ConstraintError(
                    "DPD type pair friction must be nonnegative and cutoff positive, both finite"
                        .into(),
                ));
            }
            if present.contains(&a) && present.contains(&b) {
                pairs.insert(pair_key(a, b), (friction, cutoff));
                max_cutoff = max_cutoff.max(cutoff);
            }
        }
        Ok(Configuration {
            types,
            pairs,
            defaults: (self.friction, self.cutoff),
            max_cutoff,
            periodic: system.uses_periodic_boundary_conditions(),
        })
    }
    fn kick(&self, v: &mut [f64], f: &[f64], m: &[f64]) {
        for (i, &mass) in m.iter().enumerate() {
            if mass != 0.0 {
                for k in 0..3 {
                    v[3 * i + k] += self.dt / mass * f[3 * i + k];
                }
            }
        }
    }
    fn drift_thermostat(
        &self,
        p: &mut [f64],
        v: &mut [f64],
        m: &[f64],
        vectors: Option<&BoxVectors>,
    ) -> Result<(), ConstraintError> {
        let config = self.config.as_ref().ok_or_else(|| {
            ConstraintError("DPDIntegrator must be initialized with a System".into())
        })?;
        let vectors = if config.periodic {
            Some(vectors.ok_or_else(|| {
                ConstraintError("periodic DPD requires current box vectors".into())
            })?)
        } else {
            None
        };
        let mut scratch = self.scratch.lock().unwrap();
        let Scratch {
            midpoint,
            neighbors,
        } = &mut *scratch;
        midpoint.resize(p.len(), 0.0);
        for (i, &mass) in m.iter().enumerate() {
            for k in 0..3 {
                midpoint[3 * i + k] = p[3 * i + k]
                    + if mass != 0.0 {
                        0.5 * self.dt * v[3 * i + k]
                    } else {
                        0.0
                    };
            }
        }
        self.random.with_random(|random| {
            neighbors.visit(p, config.max_cutoff, vectors, |i, j| {
                if m[i] == 0.0 && m[j] == 0.0 {
                    return Ok(());
                }
                let (friction, cutoff) = config
                    .pairs
                    .get(&pair_key(config.types[i], config.types[j]))
                    .copied()
                    .unwrap_or(config.defaults);
                let d = delta(midpoint, i, j, vectors);
                let r = norm2(d).sqrt();
                if r >= cutoff {
                    return Ok(());
                }
                if r == 0.0 {
                    return Err(ConstraintError(
                        "DPD radial direction is undefined for coincident particles".into(),
                    ));
                }
                let inverse_i = if m[i] == 0.0 { 0.0 } else { m[i].recip() };
                let inverse_j = if m[j] == 0.0 { 0.0 } else { m[j].recip() };
                let reduced = (inverse_i + inverse_j).recip();
                let omega = 1.0 - r / cutoff;
                let scale = (-self.dt * 2.0 * friction * omega * omega).exp();
                let direction = d.map(|value| value / r);
                let radial = (0..3)
                    .map(|k| (v[3 * j + k] - v[3 * i + k]) * direction[k])
                    .sum::<f64>();
                let dv = (1.0 - scale) * radial
                    + (1.0 - scale * scale).sqrt()
                        * (BOLTZMANN * self.temperature / reduced).sqrt()
                        * random.sample::<f64, _>(StandardNormal);
                for k in 0..3 {
                    if m[i] != 0.0 {
                        v[3 * i + k] += (reduced / m[i]) * dv * direction[k];
                    }
                    if m[j] != 0.0 {
                        v[3 * j + k] -= (reduced / m[j]) * dv * direction[k];
                    }
                }
                Ok(())
            })
        })?;
        for (i, &mass) in m.iter().enumerate() {
            if mass != 0.0 {
                for k in 0..3 {
                    p[3 * i + k] = midpoint[3 * i + k] + 0.5 * self.dt * v[3 * i + k];
                }
            }
        }
        Ok(())
    }
    fn update<B: Backend>(
        &self,
        p: &mut Tensor<B, 2>,
        v: &mut Tensor<B, 2>,
        f: &Tensor<B, 2>,
        m: &Tensor<B, 1>,
        vectors: Option<&BoxVectors>,
        phase: u8,
    ) -> Result<(), ConstraintError> {
        let run = |p: &mut [f64], v: &mut [f64], f: &[f64], m: &[f64]| {
            if phase != 2 {
                self.kick(v, f, m);
            }
            if phase != 1 {
                self.drift_thermostat(p, v, m, vectors)?;
            }
            Ok(())
        };
        if let Some(result) = super::try_host_step(p, v, f, m, run) {
            return result;
        }
        let mut position = crate::tensor::to_f64_vec(p);
        let mut velocity = crate::tensor::to_f64_vec(v);
        run(
            &mut position,
            &mut velocity,
            &crate::tensor::to_f64_vec(f),
            &crate::tensor::to_f64_vec(m),
        )?;
        *p = crate::tensor::from_f64_vec(position, p.dims(), &p.device());
        *v = crate::tensor::from_f64_vec(velocity, v.dims(), &v.device());
        Ok(())
    }
    fn advance<B: Backend>(
        &self,
        p: &mut Tensor<B, 2>,
        v: &mut Tensor<B, 2>,
        f: &Tensor<B, 2>,
        m: &Tensor<B, 1>,
        c: &ConstraintSolver,
        vectors: Option<&BoxVectors>,
    ) -> Result<(), ConstraintError> {
        if c.is_empty() {
            return self.update(p, v, f, m, vectors, 0);
        }
        let previous = p.clone();
        self.update(p, v, f, m, vectors, 1)?;
        c.apply_velocities(p, v, self.tolerance)?;
        self.update(p, v, f, m, vectors, 2)?;
        let proposed = p.clone();
        c.apply_positions(&previous, p, self.tolerance)?;
        super::velocities_from_positions(p, &proposed, v, m, self.dt, true);
        Ok(())
    }
}
impl<B: Backend> Integrator<B> for DPDIntegrator {
    fn validate_system(&self, system: &System<B>) -> Result<(), ConstraintError> {
        self.configure(system).map(|_| ())
    }
    fn initialize_system(&mut self, system: &System<B>) {
        self.config = Some(self.configure(system).expect("invalid DPD configuration"));
        self.random.initialize();
        *self.scratch.get_mut().unwrap() = Scratch::default();
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
    fn velocity_time_offset(&self) -> f64 {
        0.5 * self.dt
    }
    fn kinetic_energy_time_offset(&self) -> f64 {
        0.0
    }
    fn step(&self, p: &mut Tensor<B, 2>, v: &mut Tensor<B, 2>, f: &Tensor<B, 2>, m: &Tensor<B, 1>) {
        self.update(p, v, f, m, None, 0).expect("DPD step failed");
    }
    fn step_constrained(
        &self,
        p: &mut Tensor<B, 2>,
        v: &mut Tensor<B, 2>,
        f: &Tensor<B, 2>,
        m: &Tensor<B, 1>,
        c: &ConstraintSolver,
    ) -> Result<(), ConstraintError> {
        self.advance(p, v, f, m, c, None)
    }
    fn step_in_environment(
        &self,
        p: &mut Tensor<B, 2>,
        v: &mut Tensor<B, 2>,
        f: &Tensor<B, 2>,
        m: &Tensor<B, 1>,
        c: &ConstraintSolver,
        environment: StepEnvironment,
    ) -> Result<(), ConstraintError> {
        self.advance(p, v, f, m, c, environment.box_vectors.as_ref())
    }
}
