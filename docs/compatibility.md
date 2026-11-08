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
| Custom forces | Partial | Ten force types use compiled expressions and analytic derivatives, including nested collective variables; nonbonded, compound, centroid, hydrogen bond and CV forces support six table types; periodic external distances, geometric functions and volume energies are implemented; other force types and broader API coverage remain |
| Collective geometry | Partial | RMSD, radius of gyration and orientation restraint use analytic forces; reference/selection updates are implemented for RMSD and orientation restraint |
| Integrators | Partial | Verlet, LFMiddle and Brownian have constrained/unconstrained reference comparisons; other integrators and wider statistical validation remain |
| State updates | Partial | CMMotionRemover applies scheduled mass-weighted velocity corrections; AndersenThermostat uses constraint-group collisions and runtime temperature/frequency parameters; barostats remain |
| Minimization | Partial | L-BFGS with strong-Wolfe search, constraint penalties/projection, fixed particles, virtual sites, integration masks and a Rust reporter are implemented; Python reporters and broader molecular validation remain |
| Python | Partial | CPU wrappers retain force definitions in System and nested CV objects, snapshot current definitions at Context creation and reject duplicate ownership; integrator settings are shared with the bound Context; broader names/signatures, units and platform selection remain incomplete |
| Precision | Partial | CPU core supports f32 and f64; Python uses f64. GPU precision modes remain unverified |
| GPU | Unverified | Backend features exist; force implementations currently copy positions to the host |
| Performance | In progress | Finite-cutoff pairs use reusable cell storage, PME reuses FFT plans/workspace, and dispersion uses linear-time moments; force tensors still allocate and transfer host data |

`scripts/validate-reference.sh` builds the external OpenMM Reference library and
runs C++ fixtures against the Rust Python extension. Current fixtures cover seven
integrator configurations at four time points, forty-one nonbonded configurations,
seven virtual site configurations at three time points, fifteen bonded/custom
force configurations, 120 group/integration configurations, three molecular
wrapping configurations, six minimization configurations and 184 custom-expression
states with forces and parameter derivatives, plus 1,134 tabulated-function states, 480 CMAP states, 384 custom-nonbonded states, 864 geometric-expression states, 864 centroid states, 1,296 hydrogen bond states, 336 collective-geometry/volume states, 360 composite-CV states in two force-cache modes and 576 center-of-mass-removal states. Andersen fixtures add 216 deterministic states and independent collision/velocity-distribution checks on 36,000 particles.
Geometric fixtures cover external periodic distances, compound distance/angle/dihedral
functions, explicit point functions, shared particles, triclinic boxes, nested
functions, table and bond updates, global parameters, groups and short trajectories.
Centroid fixtures cover mass/default, explicit and mixed weights, repeated particles,
overlapping groups, zero weights, massless particles, updates and periodic geometry.
Hydrogen bond fixtures cover all three cutoff methods, donor/acceptor parameters,
omitted unused atoms, repeated primary atoms, exclusions, table/global updates,
triclinic boxes and trajectories. Dihedral fixtures use nondegenerate coordinates;
the angle and its force are undefined when either defining plane has zero area.
Collective fixtures cover selected/all particles, proper rotations, reflections,
deformations, reference/selection changes, restraint strength, massless particles,
global and box changes, force groups and four-step trajectories. Independent tests
check energy gradients and known rigid-rotation energies.
Composite-CV fixtures combine bonds, RG, RMSD, orientation, PME, volume and nested
CV forces. They compare inherited and outer parameters, registered derivatives,
table updates, explicit child-force updates, periodic boxes and short trajectories.
Center-of-mass-removal fixtures cover Verlet, LFMiddle and Brownian integration,
constraints, massless particles, three update frequencies, four starting step
counts and changed definition objects after Context creation. Velocity updates
run before force evaluation, in System order, independently of force-group masks.
State queries do not trigger them. CMMotionRemover adjusts only massive particles
and keeps the frequency captured when the Context was created, matching Reference.
AndersenThermostat groups particles by constraint connectivity, draws one collision
per group with probability `1-exp(-frequency*dt)` and resamples each massive
particle from the Maxwell distribution. Frequency is in inverse ps and temperature
is in kelvin. Context parameters change the running values; force setters change the definition
defaults. Rust and Python capture defaults when constructing a Context.
Fixed particles retain their velocities. The Rust implementation uses a per-thermostat ChaCha12 generator:
nonzero seeds reproduce Rust runs, while seed zero initializes from OS entropy.
Cross-implementation random paths are not identical. Deterministic tests cover
zero collision frequency, zero temperature, certain collisions, all three
integrators, constraints, force groups, massless particles and parameter changes.
Statistical tests separately check the collision probability, transitive groups,
normalized first/second/fourth velocity moments and mass scaling.

Python System objects retain the original force objects and expose `get_force`,
`get_forces` and `remove_force`. Context creation snapshots the current definitions,
including nested CV children and shared tables. Definition edits affect new
Contexts; existing Contexts retain their captured settings until an explicit
parameter update. A force can belong to only one System or CustomCVForce, and
failed additions leave ownership and force lists unchanged. Reference fixtures
compare 288 states across repeated Context creation, bond membership, groups,
periodicity, masses, tables, nested CV parameters and thermostat defaults.
Additional Python probes cover ownership conflicts, cycles, object retention and
removal. Retained Python force references remain valid after their owner is removed
or deleted and may be attached again; this is a safe lifetime extension over the
reference wrapper, whose underlying C++ object is deleted.

Python Verlet, LFMiddle and Brownian integrators share their settings with their
bound Context. Step size, tolerance, force-group masks, temperature and friction
setters affect subsequent execution and state queries. `Context(system, integrator)`
and `system.create_context(integrator)` retain the original System and Integrator;
`get_system()` and `get_integrator()` return those objects. Both `integrator.step()`
and `context.step()` advance the same state. A live integrator cannot bind to a
second Context. The integrator uses a weak link to the Context, so keeping the
integrator does not create a reference cycle. Dropping the Context releases the
binding; stepping an unbound integrator raises an error, and rebinding after
release is a safe lifetime extension.
LFMiddle and Brownian capture a per-integrator ChaCha12 random stream at Context
initialization. Nonzero seeds reproduce Rust runs independently of how other
Contexts are stepped; seed zero uses OS entropy. Changing the seed afterward,
including before the first step, changes the initialization setting and leaves the
active stream unchanged. A new binding initializes a new stream. Both precisions
consume Gaussian samples in the same massive-particle order, while floating-point
results retain their backend precision. Other backends receive host-generated
noise tensors; resident GPU random-number generation remains unimplemented.

On the f64 CPU backend, both stochastic integrators update unique position and
velocity storage directly and preserve shared State snapshots through copy-on-write.
LFMiddle retains velocity projection before its drift/thermostat/drift sequence,
then applies position projection and velocity correction. Brownian velocities are
reconstructed from the final constrained displacement. Single precision and other
backends retain tensor operations. When no selected force contributes, Context
reuses an immutable zero-force tensor. It initializes that cache only when needed.
Allocation tests verify that warmed, unconstrained f64 integration uses no new dense
arrays, both directly and through a free-particle Context. Constrained projection
still uses temporary state storage.

Integrator getters/setters provide both snake_case and OpenMM camelCase names;
the existing scalar properties remain available. Other classes still need API
naming/signature coverage. Inputs retain the core's finite-value checks: timestep
and Brownian friction must be positive, and constraint tolerance must be between
zero and one. The source accepts some boundary values outside this supported
domain; error and boundary compatibility remain incomplete.
Reference fixtures cover 72 states with changing settings, all three integrators,
constraints, fixed particles and kinetic energy. Temperature changes are checked
independently for LFMiddle and Brownian using 36,000 normalized velocity components
at each of three temperatures. Both implementations must satisfy six-standard-error
bounds for the first, second and fourth Gaussian moments. Random paths are not
compared across implementations. Python probes check double binding, retained
objects, collection, rebinding, invalid settings and both stepping interfaces.
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
Molecular connectivity follows explicit bond forces, centroid bonds, constraints
and virtual site dependencies. Angles, torsions, compound bonds and nonbonded exceptions do not independently join
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
CustomCompoundBondForce supports particle coordinates, distance/angle/dihedral
functions, explicit point functions, tables and registered parameter derivatives.
CustomExternalForce supports `periodicdistance` with coordinate chain rules.
CustomCentroidBondForce normalizes explicit weights or particle masses, evaluates
the raw weighted centers and redistributes forces with the same normalized weights.
Periodic geometry applies between centers; it does not unwrap particles within a
group. Parameter updates retain the Context's group membership and weights while
updating bond parameters and tables, matching the supplied Reference kernel.
CustomHbondForce supports donor/acceptor distance, angle and dihedral functions,
tables, exclusions, all three cutoff methods and parameter updates. Cutoff traversal
uses primary donor/acceptor atoms and cell storage without a dense pair matrix.
An exhaustive traversal option supports numerical checks. Unused secondary atoms
may be omitted; expressions that reference an omitted atom are rejected explicitly.
Context updates preserve particle membership, exclusions, methods and cutoffs.
CustomCVForce composes up to 32 force energies with analytic chain-rule forces,
tabulated functions and registered parameter derivatives. Child parameters are
shared with the outer Context, and outer defaults override consistent child
defaults. Child parameter derivatives are included only when the parent has
registered at least one derivative, matching the supplied Reference implementation.
Updating the parent copies tables only. Explicit Rust/Python Context updates can
address a child through a sequence of force indices; nested paths are supported.
The OpenMM `getInnerContext()` API and a compatible Python Context view remain
unimplemented. Python retains the original child objects and snapshots them when
creating or updating the Context; changing a child object alone does not update a
running Context.

CV force tensors use a configurable cache with a default 16 MiB payload limit.
The cache retains all child tensors when their estimated payload fits, otherwise
it recomputes them after evaluating the outer expression. This is an upper bound
on cached force payload, not total process memory. Zero disables caching; the
largest `usize` value removes the payload limit. On the f64 CPU backend, RG child
forces add directly into the parent buffer and bypass dense child tensors in all
cache modes. Other children retain tensor evaluation, and other backends retain
tensor force accumulation. Energy-only RG requests do not allocate force arrays.
RG preparation retains the center and radius until the outer expression supplies
its chain-rule coefficient, avoiding a second geometry calculation. The tensor
construction helper promotes native CPU input bytes to shared array storage;
otherwise Burn's byte storage can copy the full tensor when it is cloned for a
read. An allocation regression test bounds large allocations independently of
the number of RG variables and checks all three cache settings.
Geometric angle derivatives retain the reference cross-product norm floor of
`1e-6`; this regularizes nearly collinear angles and differs from the exact energy
gradient below that threshold. Coincident-point distance derivatives are zero.
A constant NaN in an unselected branch can prevent the
supplied Lepton optimizer from terminating because NaN is unequal to itself.
Reference fixtures therefore use defined scalar domains; Rust separately checks
that unselected branches do not contaminate derivatives.

RMSD and orientation restraint share a quaternion alignment calculation. The
reference coordinates are centered once per initialization or parameter update;
clones share immutable coordinate storage. Orientation derivatives use the
eigenvector sensitivity of the 4-by-4 alignment matrix, contracted before the
particle loop. These forces and RG use raw coordinates without periodic wrapping.
CustomVolumeForce evaluates the volume and six independent box components with
zero particle forces. Its box variables override global parameters with the same
names, matching Reference.

RMSD computes squared residuals after optimal rotation to avoid cancellation
between coordinate norms and the largest eigenvalue. It retains the Reference
zero-force threshold of mean squared deviation below `1e-20`. Exactly rigid
transformations therefore return zero despite roundoff in the eigensolver.
The twelve rigid-transform probes return up to `3.97e-8` for Reference RMSD;
Rust returns zero in each probe. Nonrigid fixtures retain strict tolerances.
For a collapsed RG selection, Reference returns zero energy and nonfinite forces.
Rust returns the zero subgradient at that nondifferentiable point; this is a
compatibility exception. Orientation derivatives require a unique optimal
rotation; degenerate alignments are outside the differentiable domain.

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
A subsequent ten-evaluation run with compound and centroid support measured
5.22 evaluations/s for Rust and 5.32 for Reference at the same particle count.
Process peak RSS was 6,616 KiB and 20,300 KiB, respectively. Short timing runs
vary; these results do not remove the earlier same-node regression finding.

Use `scripts/benchmark-hbond.sh` to measure donor/acceptor cutoff traversal.
On its 20,736-particle periodic workload, ten force/energy evaluations measured
5.16 evaluations/s with Rust cells, 0.80 with Rust exhaustive traversal and 0.46
with OpenMM Reference. Peak process RSS was 7,896, 8,000 and 13,704 KiB,
respectively. At 768 particles, exhaustive Rust traversal was faster than cells
(164 versus 150 evaluations/s); cell construction has a cost for small systems.
These measurements cover the supplied expression and Reference platform only.

`scripts/benchmark-collective.sh` compares Context state requests for both forces
and energy, including kinetic energy, for RMSD, orientation restraint and RG.
It runs three process repetitions per implementation and particle count, with
50,000, 10,000 and 5,000 requests at 512, 4,096 and 13,824 particles.
Generated summaries include median/range throughput and peak RSS.
The initial Rust implementation was 4 to 16 times slower than Reference in this
test. The measurements exposed repeated NdArray copies and reshapes, redundant
zero-force accumulation and separate temporary tensors for kinetic energy.
Collective kernels now read CPU storage through views and construct their output
with its final shape. Native f64 output vectors transfer ownership into shared
arrays, and the initial Context coordinates and velocities use shared storage.
For unconstrained contiguous f64 CPU states, the shifted
kinetic energy uses a single reduction over borrowed velocity/force slices.
Other backends and constrained states retain the tensor calculation.

At 13,824 particles, the three-repetition measurements were:

| Force | Rust median requests/s | Reference median requests/s | Rust peak RSS, KiB | Reference peak RSS, KiB |
| --- | ---: | ---: | ---: | ---: |
| RMSD | 4,787 | 1,927 | 8,228 | 10,460 |
| Orientation restraint | 6,093 | 1,123 | 8,336 | 10,592 |
| RG | 7,626 | 3,990 | 7,500 | 9,872 |

RSS is the maximum across the three processes. These results use the scripted
coordinate workload and the Reference platform; they do not establish GPU or
optimized OpenMM CPU performance. The current code passed all 20 reference
fixtures with 513,719 scalar comparisons, 126 Rust tests and a build with the
default backend feature disabled.

`scripts/benchmark-cv.sh` measures a weighted mean of disjoint RG selections,
squared by a CustomCVForce. Both implementations request Context forces and
energy, including kinetic energy. It runs 5,000 requests at 4,096 particles and
300 at 65,536 particles, with three process repetitions for every configuration.
The default bounded cache measurements were:

| Particles | CVs | Rust median requests/s | Reference median requests/s | Rust peak RSS, KiB | Reference peak RSS, KiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 2 | 20,787 | 8,440 | 4,900 | 8,824 |
| 4,096 | 8 | 17,208 | 2,216 | 4,892 | 9,328 |
| 4,096 | 32 | 15,196 | 565.8 | 4,848 | 12,368 |
| 65,536 | 2 | 1,271 | 465.3 | 17,360 | 27,720 |
| 65,536 | 8 | 757.9 | 105.9 | 17,196 | 36,960 |
| 65,536 | 32 | 768.8 | 25.0 | 17,336 | 75,040 |

Unbounded and disabled caches were also measured. RG uses direct accumulation
in every mode, so these cases do not retain dense child force arrays. Generic
CV children still use the configured tensor cache. These measurements establish
performance for the scripted RG composition and Reference platform only;
optimized CPU/GPU platforms, other CV workloads and integration remain separate
requirements.

`scripts/benchmark-andersen.sh` measures complete Verlet integration with an
Andersen thermostat at 300 K, collision frequency 37/ps and timestep 0.001 ps,
without potential forces. It uses 10 warmup steps, then 1,000 steps at 4,096
particles or 300 at 65,536, with three process repetitions. Random paths differ;
each final kinetic energy must lie within six equilibrium standard deviations.

| Particles | Rust median steps/s | Reference median steps/s | Rust peak RSS, KiB | Reference peak RSS, KiB |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 18,110 | 7,891 | 4,864 | 7,860 |
| 65,536 | 1,195 | 492 | 17,564 | 22,216 |

The initial implementation reached 164 steps/s and 28,476 KiB at 65,536 particles.
The CPU f64 Verlet update now reads forces and masses through borrowed views and
updates coordinates/velocities in one pass. Andersen and center-of-mass updates
reuse mutable velocity storage. Shared State snapshots trigger copy-on-write,
and noncontiguous tensors retain logical element order. Other backends retain
their tensor paths. Molecular connectivity is constructed lazily when requested
or needed for periodic wrapping; invalid bonded indices are still rejected at
Context initialization. These measurements cover this thermostat workload on the
Reference platform, not optimized CPU/GPU platforms or molecular force fields.

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

## CPU integration performance

`scripts/benchmark-stochastic.sh` compares native Context stepping against OpenMM
Reference for free particles, 0.001 ps steps, friction 0.7/ps and initial velocities
(0.1, 0.2, 0.3) nm/ps. It uses ten warmup steps and three process repetitions per
case. The timed step counts are 30,000, 1,500 and 200 at 8, 4,096 and 65,536
particles. Langevin and Brownian run at both 0 K and 300 K; Verlet has no thermostat.
Final kinetic energies are checked against deterministic expectations or six
standard deviations of the finite-time Gaussian distribution. Random generators
and paths differ across implementations. The table shows median throughput and
maximum process RSS across the three repetitions, using the 300 K stochastic cases.

| Integrator | Particles | Rust steps/s | Reference steps/s | Rust peak RSS, KiB | Reference peak RSS, KiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Verlet | 8 | 2,100,968 | 3,383,615 | 3,784 | 7,004 |
| Verlet | 4,096 | 40,255 | 10,990 | 4,804 | 7,432 |
| Verlet | 65,536 | 2,844 | 675 | 15,944 | 17,748 |
| LFMiddle | 8 | 1,104,659 | 1,347,205 | 4,000 | 7,268 |
| LFMiddle | 4,096 | 6,129 | 3,530 | 4,948 | 7,696 |
| LFMiddle | 65,536 | 388 | 226 | 16,256 | 20,932 |
| Brownian | 8 | 1,160,727 | 1,478,042 | 4,004 | 7,080 |
| Brownian | 4,096 | 6,809 | 3,696 | 4,988 | 7,456 |
| Brownian | 65,536 | 429 | 231 | 16,192 | 19,608 |

Rust is faster for the two larger sizes and uses less peak process memory in these
cases. At eight particles, Reference remains faster for all three integrators.
These measurements cover free particles on the Reference platform; molecular
force fields, constrained workloads and optimized CPU/GPU platforms remain
separate performance requirements.

`scripts/benchmark-python-integrator.py` measures Python stepping against the
preceding implementation with shared Context bindings and tensor-based stochastic
updates. The script accepts an extension, integrator, particle count and step
count. Both versions use release builds, free particles, zero temperature,
0.001 ps steps, ten warmup steps and three repetitions. It checks final kinetic
energy. This isolates the change in CPU integration and empty-force caching;
it is not a comparison with the OpenMM Python wrapper.

| Integrator | Particles | Previous steps/s | Current steps/s | Previous/current peak RSS, KiB |
| --- | ---: | ---: | ---: | ---: |
| Verlet | 8 | 1,617,572.5 | 1,523,627.0 | 23,976 / 23,972 |
| Verlet | 4,096 | 36,193.3 | 40,741.6 | 24,084 / 24,160 |
| Verlet | 65,536 | 2,364.3 | 2,752.8 | 43,760 / 44,288 |
| LFMiddle | 8 | 130,669.4 | 1,094,259.2 | 23,996 / 23,824 |
| LFMiddle | 4,096 | 1,572.6 | 6,260.3 | 23,968 / 23,996 |
| LFMiddle | 65,536 | 89.8 | 391.9 | 47,184 / 44,292 |
| Brownian | 8 | 163,687.7 | 1,136,759.9 | 24,092 / 24,088 |
| Brownian | 4,096 | 1,696.1 | 6,953.8 | 24,052 / 23,988 |
| Brownian | 65,536 | 97.8 | 431.9 | 47,192 / 44,404 |

Throughput is the median and RSS the maximum across three processes, including
Python. Stochastic integration improves by about 4 to 8 times. Eight-particle
Verlet is about 6% slower in this measurement, while the larger cases improve.
Small-system overhead remains optimization work.

## Documentation discrepancies found during implementation

- The original Verlet header called its update velocity Verlet, but the stored
  velocity and update equations implement leapfrog. The header now names leapfrog.
- The binding guide named `NdArrayBackend` and placed the concrete type in Cargo
  configuration. The actual alias is `NdArray<f64>` in the binding source; the
  guide now identifies that location.

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

- Burn's NdArray storage clone comments describe borrowed-byte clones as cheap
  shared copies. Native `TensorData` byte allocations can instead copy the whole
  buffer. CPU constructors use shared arrays, including the initial Context state,
  and force output vectors transfer ownership directly into those arrays.

- `ReferenceAndersenThermostat.cpp` labels its step size and collision frequency
  as fs and inverse fs. Its caller passes ps and inverse ps, matching the public
  `AndersenThermostat` header. The internal comment is stale; the reference code
  uses the public units.

- `CLAUDE.md` omitted the direct optional ndarray dependency from its CPU feature
  example and prescribed Burn autodiff for custom forces. The implementation uses
  compiled scalar expressions with reverse-mode derivatives and analytic geometry;
  the guide now describes that implementation. Its mdtraj-rs entry also described
  file I/O as available even though the dependency is still disabled.

- HarmonicBondForce and CustomBondForce parameter updates previously copied changed
  particle indices, although Reference rejects changes to bond membership. They
  now reject such updates before modifying the Context. This also preserves the
  connectivity used by lazy molecule construction.

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

## Centroid connectivity exception

The supplied `CustomCentroidBondForceImpl::getBondedParticles()` passes local bond
positions to `addBondsBetweenGroups()` instead of the bond's group indices. A
two-group bond using groups 2 and 3 therefore joins groups 0 and 1 in the reference.
This affects molecular connectivity and molecule-based periodic wrapping.
Rust joins the groups actually named by the bond. The centroid reference fixture
records both connectivity results explicitly, while forces, energies and ordinary
trajectories retain numerical comparisons with the reference. One-group bonds do
not add molecular connectivity. Group connections use a spanning tree to avoid
materializing a Cartesian product of particle pairs.
