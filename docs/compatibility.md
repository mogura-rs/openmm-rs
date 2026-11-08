# OpenMM compatibility and performance

The target is functional compatibility with the supplied OpenMM source checkout,
including the Python interface and bundled plugins. Correctness, execution speed,
and peak memory are acceptance criteria. This is an active implementation; the
current crate is not yet a compatible replacement for OpenMM.

Reference paths below are relative to the OpenMM checkout. Use `OPENMM_SOURCE` to
select that checkout; no machine-specific location is required. Record its Git
revision in generated validation results.

## Acceptance checklist

- [ ] Core API: System, Context, State, units, force groups, runtime parameters,
      parameter derivatives, parameter updates, periodic boxes, checkpoints.
- [ ] Bonded forces: harmonic bonds/angles, periodic and RB torsions, CMAP.
- [ ] Nonbonded forces: no cutoff, reaction field, Ewald, PME, LJPME, exceptions,
      switching, dispersion correction, parameter offsets, reciprocal groups.
- [ ] Constraints: positions, velocities, coupled constraints, rigid water,
      massless particles, integration and minimization.
- [ ] Virtual sites: two/three-particle averages, out-of-plane, local coordinates,
      nested sites, force redistribution, periodic geometry.
- [ ] Integrators: Verlet, Langevin, LangevinMiddle, Brownian, variable step,
      Nose-Hoover, custom, compound, DPD, QTB.
- [ ] Thermostats and pressure: Andersen, center-of-mass motion removal,
      isotropic/anisotropic/membrane/flexible Monte Carlo barostats.
- [ ] Custom forces: bond, angle, torsion, external, nonbonded, compound bond,
      centroid bond, hydrogen bond, GB, CV, many-particle, volume; expressions,
      tabulated functions, global parameters and energy derivatives.
- [ ] Other forces: GBSAOBC, Gay-Berne, RMSD, RG, orientation restraint, LCPO,
      ATM, constant potential, Python callbacks.
- [ ] Local energy minimization, serialization, restart and checkpoint behavior.
- [ ] Python: API names/signatures, errors, units, object ownership and updates.
- [ ] Application layer: topology, force fields, structure/trajectory files,
      simulation, modeller, reporters and simulation preparation.
- [ ] Bundled plugins: AMOEBA, Drude, RPMD and PME execution behavior.
- [ ] CPU and GPU: supported precision modes, numerical parity, reproducibility
      where promised, backend selection, properties and device handling.
- [ ] End-to-end reference tests, real molecular systems and long-run stability.
- [ ] Matched benchmarks: initialization, force evaluation, integration throughput,
      peak host/device memory, and scaling with particle count.

## Current implementation

| Area | State | Outstanding work |
| --- | --- | --- |
| Core state and parameters | Partial | Force groups, integration masks and state filtering are implemented; checkpointing and broader validation remain |
| Virtual sites | Partial | Five site types, nested dependencies and force redistribution are implemented; see the fractional symmetry exception below |
| Bonded forces | Partial | Reference coverage, CMAP, GPU execution |
| Nonbonded | Partial | Six methods, particle/exception offsets, runtime updates, reciprocal groups, switching and dispersion correction are implemented; complete API behavior and broader coverage remain |
| Constraints | Partial | Sparse SHAKE/RATTLE positions and velocities are implemented; specialized solvers and broader reference coverage remain |
| Custom forces | Partial | Five force types and a scalar expression evaluator exist; broader API and numerical validation remain |
| Integrators | Partial | Verlet, LFMiddle and Brownian have constrained/unconstrained reference comparisons; other integrators and wider statistical validation remain |
| Python | Partial | Thin CPU wrappers exist, but the interface is not yet OpenMM-compatible |
| GPU | Unverified | Backend features exist; force implementations currently copy positions to the host |
| Performance | In progress | Finite-cutoff pairs use reusable cell storage, PME reuses FFT plans/workspace, and dispersion uses linear-time moments; force tensors still allocate and transfer host data |

`scripts/validate-reference.sh` builds the external OpenMM Reference library and
runs C++ fixtures against the Rust Python extension. Current fixtures cover seven
integrator configurations at four time points, forty-one nonbonded configurations,
seven virtual site configurations at three time points, fifteen bonded/custom
force configurations, and 120 group/integration configurations.
Nonbonded fixtures include Ewald, electrostatic/dispersion PME, neutral and charged
systems, exceptions, runtime offsets, parameter updates and periodic box changes.
Mesh parameters are fixed when a Context is initialized, as in the reference.
The Reference kernel ignores switching for LJPME and uses global parameter defaults
for the analytical dispersion coefficient. Runtime offsets affect pair and
reciprocal terms but do not recompute that coefficient; parameter updates do.
Group fixtures compare masks including the high bit, direct/reciprocal splitting,
disabled direct-space terms, restricted integration, forces and kinetic energy.
Use `scripts/benchmark-cpu.sh` after reference validation to compare exhaustive
and cell traversal with the OpenMM Reference platform. It records process peak
RSS and timing; optimized OpenMM CPU/GPU performance is a separate requirement.

## Validation and optimization rules

Numerical comparisons use the supplied reference implementation, matching system,
parameters, positions, precision, platform and requested state fields. Stochastic
integrators require deterministic zero-temperature cases and statistical tests;
matching seeds across different random generators does not imply matching paths.

Optimize measured bottlenecks while retaining reference comparisons. Priorities:

1. Reuse prepared force data, neighbor lists and scratch storage.
2. Use cell/Verlet lists for finite cutoffs and FFT meshes for reciprocal terms.
3. Keep GPU state and force accumulation on the device across integration steps.
4. Avoid materializing dense pair matrices or unnecessary state snapshots.
5. Measure both throughput and peak host/device memory over several system sizes.

GPU jobs must run through Slurm. Use at most two nodes concurrently across all
jobs for this project. Generated builds, logs, trajectories and benchmark results
belong under ignored directories.

## Documentation discrepancies found during the initial audit

- `CLAUDE.md` lists NonbondedForce, Custom forces and Python bindings as future
  work, although partial implementations already exist.
- Its force-trait sketch omits clone/downcast and parameter-related methods.
- It describes compile-time unit enforcement, but public simulation methods use
  scalar floating-point values in OpenMM units.
- Its GPU build commands enable a backend but do not establish GPU force execution.
- Its original Langevin description said BAOAB; the reference implements LFMiddle with a
  full force kick before the two drifts and thermostat.
- The original nonbonded force tests claimed to mirror OpenMM but expected
  inverted Coulomb and Lennard-Jones forces. They now use the physical sign, with
  independent energy-gradient and compiled-reference comparisons.
- The original periodic-box test used a cutoff larger than half the box width;
  it now uses a valid cutoff, matching OpenMM's validation.
- The original periodic/custom torsion comments described analytic forces and
  momentum conservation, but the force signs and middle-particle terms did not
  match the energy gradient. Both now share a validated analytic geometry helper.

Implementation and reference tests take precedence over outdated descriptions.

## Fractional symmetry force exception

For a triclinic box, the supplied reference snapshot's fractional `SymmetrySite`
force redistribution differs from the negative energy gradient. The default Rust
implementation uses the Jacobian transpose of the position transformation and
is checked against finite differences of the reference energy.

`Context::set_reference_symmetry_forces(true)` (also available in Python) enables
the reference snapshot's force convention for explicit regression compatibility.
That option can produce nonconservative forces for fractional rotations in
triclinic boxes. Cartesian symmetry sites and the other site types are unaffected.

The `SymmetrySite` formulas in the reference header also repeat the x-coordinate
label and contain index typos in the y/z rows. Position evaluation follows the
reference implementation's matrix multiplication.
