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
| Core state and parameters | Partial | Force groups, integration masks, state filtering, molecular connectivity and molecule-preserving periodic wrapping are implemented; checkpointing and broader validation remain |
| Virtual sites | Partial | Five site types, nested dependencies and force redistribution are implemented; see the fractional symmetry exception below |
| Bonded forces | Partial | Harmonic bonds/angles, periodic/RB torsions and CMAP have analytic forces and reference comparisons; broader API validation and GPU execution remain |
| Nonbonded | Partial | Six methods, particle/exception offsets, runtime updates, reciprocal groups, switching and dispersion correction are implemented; complete API behavior and broader coverage remain |
| Constraints | Partial | Sparse SHAKE/RATTLE positions and velocities are implemented; specialized solvers and broader reference coverage remain |
| Custom forces | Partial | Five force types use compiled scalar expressions, shared subexpressions and analytic coordinate/parameter derivatives; six tabulated function types are available for CustomNonbondedForce; geometric functions, other force types and broader API coverage remain |
| Integrators | Partial | Verlet, LFMiddle and Brownian have constrained/unconstrained reference comparisons; other integrators and wider statistical validation remain |
| Minimization | Partial | L-BFGS with strong-Wolfe search, constraint penalties/projection, fixed particles, virtual sites, integration masks and a Rust reporter are implemented; Python reporters and broader molecular validation remain |
| Python | Partial | Thin CPU wrappers exist, but the interface is not yet OpenMM-compatible |
| Precision | Partial | CPU core supports f32 and f64; Python uses f64. GPU precision modes remain unverified |
| GPU | Unverified | Backend features exist; force implementations currently copy positions to the host |
| Performance | In progress | Finite-cutoff pairs use reusable cell storage, PME reuses FFT plans/workspace, and dispersion uses linear-time moments; force tensors still allocate and transfer host data |

`scripts/validate-reference.sh` builds the external OpenMM Reference library and
runs C++ fixtures against the Rust Python extension. Current fixtures cover seven
integrator configurations at four time points, forty-one nonbonded configurations,
seven virtual site configurations at three time points, fifteen bonded/custom
force configurations, 120 group/integration configurations, three molecular
wrapping configurations, six minimization configurations and 184 custom-expression
states with forces and parameter derivatives, plus 1,134 tabulated-function states, 480 CMAP states and 384 custom-nonbonded states.
Custom nonbonded fixtures cover all three methods, switching, overlapping and
empty interaction groups, computed values, table updates, global changes, box
changes and long-range corrections. Pair forces and energies use strict double
precision tolerances. Tail corrections use a separate `2e-7` comparison tolerance:
the supplied Reference implementation evaluates its quadrature integrands in
single precision, while Rust retains f64. A separate analytic power-law integral
checks the correction and its parameter derivative.
CMAP fixtures cover five map sizes, shared atoms, angle boundaries, triclinic
periodicity, runtime map changes, force groups and four-step Verlet trajectories.
Nonbonded fixtures include Ewald, electrostatic/dispersion PME, neutral and charged
systems, exceptions, runtime offsets, parameter updates and periodic box changes.
Mesh parameters are fixed when a Context is initialized, as in the reference.
The Reference kernel ignores switching for LJPME and uses global parameter defaults
for the analytical dispersion coefficient. Runtime offsets affect pair and
reciprocal terms but do not recompute that coefficient; parameter updates do.
Group fixtures compare masks including the high bit, direct/reciprocal splitting,
disabled direct-space terms, restricted integration, forces and kinetic energy.
Molecular connectivity follows explicit bond forces, constraints and virtual site
dependencies. Angles, torsions and nonbonded exceptions do not independently join
molecules in the supplied reference implementation.
The minimizer returns its stopping reason, iteration count, physical force RMS
and constraint error. A constrained stationary point can have nonzero physical
forces, so that RMS alone is not a constrained convergence criterion. Finite
precision can end a line search before the requested gradient tolerance is reached.
For double-precision Rust input, use `openmm_core::tensor::from_floats` or explicit
`TensorData` with f64 storage. Burn 0.20.1's `Tensor::from_floats` converts input
through f32 regardless of the backend type; the crate helper preserves the selected
backend precision. Existing f32 backend code remains supported.

Scalar custom expressions support all standard Lepton scalar functions, comparisons
with the supplied parser's precedence, and semicolon definitions evaluated from
right to left. Function arity and unknown variables are rejected explicitly.
Constant subexpressions are folded, small nonnegative integer powers use products,
and reverse-mode evaluation skips derivatives that the caller does not request.
Derivative buffers are reused within each force evaluation; interactions
do not allocate variable maps or use numerical differences. `log10` remains a Rust
extension. Continuous and discrete 1D/2D/3D tables support natural/periodic
splines, analytical chain-rule derivatives, clamped nearest-index lookup and
runtime parameter updates. Continuous tables store nodal mixed derivatives and
contract one axis at a time; immutable table data is shared by cloned forces.
Python table objects retain shared ownership when registered with a force;
Context updates copy a snapshot.
CustomNonbondedForce also supports explicit cutoff methods, switching functions,
interaction groups, per-particle computed values, graph-based exclusions and a
cached long-range correction. The cache is invalidated by parameter/table updates;
box changes rescale the cached coefficient by the current volume. Interaction
groups avoid duplicates within one group and retain repeated pairs across groups.
Computed values cannot be combined with registered parameter derivatives, matching
the reference. Runtime parameter updates retain the Context's exclusions, groups,
method and cutoff settings. Setting the cutoff distance alone no longer enables a
cutoff; select `CutoffNonPeriodic` or `CutoffPeriodic` explicitly.
Geometry functions such as `periodicdistance`
are not yet supported. A constant NaN in an unselected branch can prevent the
supplied Lepton optimizer from terminating because NaN is unequal to itself.
Reference fixtures therefore use defined scalar domains; Rust separately checks
that unselected branches do not contaminate derivatives.

Pass fixture names to select a focused reference run, for example
`scripts/validate-reference.sh virtual_sites minimizer`. With no arguments, the
script runs all fixtures. Minimized endpoints are compared after convergence;
the iteration-limited fixture separately verifies stopping after one iteration.
Use `scripts/benchmark-cpu.sh` and `scripts/benchmark-custom.sh` after reference validation to compare exhaustive
and cell traversal with the OpenMM Reference platform. They record process peak
RSS and timing; optimized OpenMM CPU/GPU performance is a separate requirement.
For the 13,824-particle custom Lennard-Jones workload, a same-node comparison
after adding switching, tables, computed values and interaction groups measured
about 5.12 evaluations/s, versus 5.42 for the earlier implementation and 5.35
for OpenMM Reference. Most of the initial regression was recovered, but an
approximately 6% gap to the earlier implementation remains. This benchmark
does not establish performance against optimized OpenMM CPU or GPU platforms.

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

- The original expression type claimed to be compiled, but recursively evaluated
  its syntax tree, matched function names at runtime and allocated argument lists.
  It now executes indexed instructions with reusable reverse-mode buffers.

- The validation and benchmark documentation invoked shell scripts directly, but
  those scripts lacked executable permissions. Both entry points are now executable.

- The Reference long-range correction failure message asks for decay faster than
  `1/r^2`. Its actual three-dimensional tail integral contains an `r^2` measure,
  so an unmodified power-law energy `r^-p` needs `p > 3` to converge. Rust reports
  failure in terms of convergence of the integral instead.

Implementation and reference tests take precedence over outdated descriptions.

## Fractional symmetry force exception

For a triclinic box, the supplied reference snapshot's fractional `SymmetrySite`
force redistribution differs from the negative energy gradient. The Rust
implementation uses the Jacobian transpose of the position transformation and
is checked against finite differences of the reference energy.

Only the conservative force is implemented. This is an intentional compatibility
exception: the snapshot's fractional-symmetry force and trajectories are not
reproduced. Reference energies and positions are compared at matching configurations,
and finite differences of the reference energy validate the redistributed force.
Cartesian symmetry sites and the other site types are unaffected.

The `SymmetrySite` formulas in the reference header also repeat the x-coordinate
label and contain index typos in the y/z rows. Position evaluation follows the
reference implementation's matrix multiplication.

## Table endpoint roundoff

The supplied Reference implementation can reject a declared upper endpoint when
constructing its final grid knot rounds downward. A six-point continuous table
on `[-1.1, 2.2]` reproduces this: evaluation at `2.2` throws an out-of-range error.
Rust treats the declared bounds as authoritative and evaluates the last cell.
The reference fixture records this exception separately from the 1,134 matching
states; it checks that Rust returns the final table value and finite forces.
Rust also validates periodic endpoint agreement when creating a table, whereas
the reference defers that check until spline construction in a Context.
