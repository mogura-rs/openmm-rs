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
| Custom forces | Partial | Twelve force types use compiled expressions and analytic derivatives, including nested collective variables; nonbonded, compound, centroid, hydrogen bond, GB, many-particle and CV forces support six table types; periodic external distances, geometric functions and volume energies are implemented; other force types and broader API coverage remain |
| Implicit solvation | Partial | GBSA-OBC and programmable CustomGB support polar/surface forces, three cutoff methods, particle updates, cell traversal and nested CV use; application parameter assignment and resident GPU execution remain |
| Anisotropic particles | Partial | Gay-Berne energies, conservative orientation forces, exceptions and live updates are implemented; source-force exceptions are explicit; resident GPU execution remains |
| Conducting electrodes | Partial | Constant-potential PME, CG and packed-matrix charge solvers, external fields, total-charge constraints, parameter updates and CPU comparisons are implemented; molecular preparation and resident GPU execution remain |
| Surface area | Partial | LCPO overlap energies, analytic forces, periodic distances and parameter updates are implemented; application parameter assignment and resident GPU execution remain |
| Collective geometry | Partial | RMSD, radius of gyration and orientation restraint use analytic forces; reference/selection updates are implemented for RMSD and orientation restraint |
| Integrators | Partial | Fixed and variable Verlet/Langevin, LFMiddle, Brownian, nested CompoundIntegrator, Nose-Hoover chains, DPD, QTB and programmable CustomIntegrator have reference comparisons; checkpointing, resident GPU integration and wider molecular validation remain |
| State updates | Partial | CMMotionRemover applies scheduled mass-weighted velocity corrections; AndersenThermostat uses constraint-group collisions and runtime temperature/frequency parameters; all four Monte Carlo barostats have CPU implementations and reference fixtures; broader molecular NPT stability and GPU execution remain |
| Minimization | Partial | L-BFGS with strong-Wolfe search, constraint penalties/projection, fixed particles, virtual sites, integration masks and a Rust reporter are implemented; Python reporters and broader molecular validation remain |
| Python | Partial | CPU wrappers retain force definitions in System and nested CV objects, snapshot current definitions at Context creation and reject duplicate ownership; integrator settings are shared with the bound Context; broader names/signatures, units and platform selection remain incomplete |
| Precision | Partial | CPU core supports f32 and f64; Python uses f64. GPU precision modes remain unverified |
| GPU | Unverified | Backend features exist; force implementations currently copy positions to the host |
| Performance | In progress | Finite-cutoff pairs use reusable cell storage, PME reuses FFT plans/workspace, and dispersion uses linear-time moments; force tensors still allocate and transfer host data |

`scripts/validate-reference.sh` builds the external OpenMM Reference library and
runs C++ fixtures against the Rust Python extension. Both libraries use release
builds by default; set `OPENMM_RS_PROFILE=debug` for a debug Rust extension. The
selected Rust profile is recorded with the generated results. Current fixtures cover seven
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

MonteCarloBarostat proposes uniform changes in volume, scales all box vectors
isotropically and accepts using the energy difference, pressure work and the
Jacobian for coordinate scaling. Rigid scaling translates each molecule by the change
in its unweighted coordinate centroid; atomic scaling moves every particle,
including massless particles. Coordinates are used as stored, without molecule
unwrapping. Accepted volume moves leave velocities unchanged. Proposed coordinates
are evaluated separately, so rejection preserves the current positions and box.
The f64 CPU path reuses one proposal buffer, swaps it with the active positions
on acceptance and preserves retained State snapshots through copy-on-write. The
buffer is allocated only when a volume move first needs it. Single precision and
other backends retain tensor construction. Instantaneous-pressure queries use
temporary coordinate storage and do not change the Context cache.
The proposal width adapts to the observed acceptance fraction, matching Reference.

MonteCarloAnisotropicBarostat selects one enabled Cartesian axis per attempt and
scales that component of every box vector and coordinate. Each axis has its own
pressure parameter, proposal width and acceptance counters. Axis permissions are
fixed at construction; all three disabled axes leave the box unchanged. Molecular
and atomic scaling use the same coordinate-buffer reuse as the isotropic barostat.
Its pressure query returns three components. Coordinates and box vectors are
perturbed along the same Cartesian axis, including off-diagonal box entries; see
the triclinic-pressure compatibility exception below.

MonteCarloMembraneBarostat adds surface-tension work `-gamma*deltaArea` in the
XY plane, with gamma in bar nm. Positive tension favors larger area. XY can scale
isotropically or one axis at a time, and Z can vary freely, stay fixed or compensate
the XY scale to preserve volume. In constant-volume mode the pressure work and
volume Jacobian are zero. XY and Z mode changes take effect in existing Contexts,
like frequency changes. The pressure query reports three Cartesian components,
independently of these shape permissions, and uses the same triclinic correction.

MonteCarloFlexibleBarostat perturbs the six independent box-matrix entries,
flips negative diagonals and reduces the resulting lattice by integer basis
changes. Coordinate scaling uses the three diagonal length ratios, matching the
reference proposal. Its acceptance work includes the reference measure terms
`-(Nscaled-2)*kT*log(Vnew/Vold)` and
`-kT*log(ax_new^2*by_new/(ax_old^2*by_old))`, in addition to energy and pressure work.
The common proposal length starts at one percent of the cube root of volume.
The adaptive upper limit uses the proposed volume, including rejected moves.

The flexible class exposes two different six-component queries, ordered
`XX, YY, ZZ, XY, XZ, YZ`. `computeCurrentPressure` preserves the source API's
fractional change to one box-matrix entry at a time; diagonal probes also scale
that coordinate component, while off-diagonal probes leave coordinates fixed.
A zero off-diagonal entry therefore gives zero potential contribution to its
corresponding pressure component. `computeStressTensor` instead applies the same
Cartesian strain to coordinates and all box vectors. It returns tension-positive
stress and optionally includes the raw molecular/atomic kinetic tensor. The
kinetic option defaults to false. Both queries reduce trial box vectors and
preserve the original state. Shared pressure helpers are in `context/pressure.rs`.

State updates run in System order, independently of force-group masks. Barostat
energies use the integrator's selected force groups, including separate reciprocal
PME groups. Its schedule uses an internal counter rather than the Context step
count. Frequency changes take effect in existing Contexts; zero disables moves.
Default temperature and pressure affect new Contexts; the named Context parameters
change a running simulation. Random streams capture their seed at initialization
and are independent between Rust Contexts. Nonzero seeds reproduce Rust paths;
seed zero uses OS entropy. Cross-implementation paths differ.

Instantaneous pressure uses molecular or atomic kinetic energy from the stored
velocities and a central finite difference of potential energy at length scales
1.001 and 0.999. Molecular kinetic energy uses mass-weighted momentum, while
coordinate scaling uses unweighted centroids. It does not apply the integrator's
kinetic-energy time shift or recompute virtual sites during the volume probe,
matching Reference. The query preserves coordinates, velocities, box, clock and
step count. Python pressure queries identify the original force retained when the
Context was created; later System edits do not change that lookup.

The isotropic pressure fixtures cover 324 states with unequal and zero masses, constraints,
virtual sites, triclinic boxes, changed globals, volume energies, PME direct and
reciprocal groups, and short trajectories. Another 60 states compare disabled and
changed frequencies independently of Context step counts. Statistical checks use
128 free atoms, connected into 64 diatomic molecules, and a volume potential
`q*V` with `q = 0.2 kJ/mol/nm^3`. Both implementations sample the first and second
volume moments in rigid and atomic modes, separately testing the gamma law with
shape `Nscaled + 1` and scale `kT/(P + q)`, with pressure expressed as energy per
volume. Each run discards 10,000 steps, then uses forty blocks of 5,000 steps,
recording a volume every twenty steps. Each mean must lie within six block standard
errors of its analytic expectation. These checks test configurational sampling;
they do not establish long-run molecular NPT stability or thermostat equilibration.

The anisotropic fixture adds 648 pressure states across orthorhombic and triclinic
cells with the same force-group, mass, constraint and virtual-site coverage.
Orthorhombic pressures are checked against the source pressure API and an independent
finite difference. Triclinic pressures are checked against Cartesian finite
differences of source energies. The largest pressure difference was 2.02e-11 bar.
Another 480 states cover all eight axis permissions and live frequency changes.
Six statistical runs independently check each moving axis, in rigid and atomic
modes, using the same volume distribution and block protocol. These runs change
Context pressures and temperature and then alter force defaults, verifying that
sampling uses the runtime parameters. Rust tests also check directional pressure
work, rejected-coordinate preservation, seeded Context independence, f32 scaling,
invalid-volume rejection and unchanged snapshots. Warmed isotropic and anisotropic
moves allocate no additional dense position arrays without retained snapshots.

The membrane fixture adds 648 pressure states, 360 schedules with changes to XY/Z
modes and frequency, and twelve statistical runs across all mode pairs and both
coordinate-scaling modes. Eight runs with variable volume use a box potential
that cancels the imposed pressure/surface work and confines the remaining box
coordinates. This gives independent gamma distributions for area and Z in
XY-isotropic mode, or for each permitted box length in XY-anisotropic mode, with
shape `Nscaled + 1`. First and second moments must each lie within six block
standard errors of their analytic values. The four constant-volume runs use
harmonic area or length potentials and compare Rust and Reference moments within
six combined standard errors; these are comparisons of the implemented algorithms,
not an independent thermodynamic validation of constant-volume shape sampling.
All runs discard 10,000 steps and use forty blocks of 5,000 steps, with samples
every twenty steps. The pressure difference stays below 2.02e-11 bar. Rust tests
also check the sign of surface-tension work, pressure-independent constant-volume
paths, selected energy groups, f32 geometry and dense-storage reuse.

Flexible-cell fixtures compare 972 states for both pressure/stress APIs and the
kinetic-stress option, including orthorhombic cells, triclinic cells and probes
that cross a reduced-cell boundary. They also cover 60 frequency schedules and
two confined-box sampling runs. A box potential cancels the imposed pressure
and adds linear diagonal terms plus harmonic tilt terms. In the regime where
tilts stay far from reduction boundaries, the diagonal lengths have gamma shapes
`Nscaled+1`, `Nscaled` and `Nscaled-1`, while the three tilts have zero-mean Gaussian
distributions. First and second moments are checked within six block standard
errors using the same 10,000-step warmup and forty-block protocol. The largest
pressure, configurational-stress and total-stress differences were respectively
2.08e-11, 2.56e-12 and 2.02e-11 bar. Rust regressions independently check a tilted
harmonic bond's analytic virial, raw kinetic cross terms, lattice equivalence after
reduction, one/two-molecule Jacobians, f32 scaling and dense-buffer reuse. The
six-atom Lennard-Jones stress is also compared with the LAMMPS numbers stored in
the supplied OpenMM tests; LAMMPS itself is not run by this suite.




Python System objects retain the original force objects and expose `get_force`,
`get_forces` and `remove_force`. Context creation snapshots the current definitions,
including nested CV children and shared tables. Definition edits affect new
Contexts; existing Contexts retain their captured settings until an explicit
parameter update. The barostat frequency is a live setting, as described above. A force can belong to only one System or CustomCVForce, and
failed additions leave ownership and force lists unchanged. Reference fixtures
compare 288 states across repeated Context creation, bond membership, groups,
periodicity, masses, tables, nested CV parameters and thermostat defaults.
Additional Python probes cover ownership conflicts, cycles, object retention and
removal. Retained Python force references remain valid after their owner is removed
or deleted and may be attached again; this is a safe lifetime extension over the
reference wrapper, whose underlying C++ object is deleted.

Python Verlet, Langevin, LFMiddle and Brownian integrators share their settings with their
bound Context. Step size, tolerance, force-group masks, temperature and friction
setters affect subsequent execution and state queries. `Context(system, integrator)`
and `system.create_context(integrator)` retain the original System and Integrator;
`get_system()` and `get_integrator()` return those objects. Both `integrator.step()`
and `context.step()` advance the same state. A live integrator cannot bind to a
second Context. The integrator uses a weak link to the Context, so keeping the
integrator does not create a reference cycle. Dropping the Context releases the
binding; stepping an unbound integrator raises an error, and rebinding after
release is a safe lifetime extension.
`LangevinIntegrator` inherits the LFMiddle implementation, matching the supplied
source. Rust exposes a type alias; Python exposes a subclass that shares all live
settings and binding behavior. Seeded Rust trajectories are also compared between
the two names, with and without constraints.
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
Reference fixtures cover 96 states with changing settings, all four integrators,
constraints, fixed particles and kinetic energy. Temperature changes are checked
independently for Langevin, LFMiddle and Brownian using 36,000 normalized velocity components
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
optimized OpenMM CPU performance. The current code passed all 21 reference
fixtures with 532,247 scalar comparisons, 135 Rust tests and a build with the
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

`scripts/benchmark-barostat.sh` measures complete Verlet steps with an isotropic
volume move on every step, at 20 bar and 300 K. Zero-strength bonds define diatomic
molecules, and the integration mask excludes potential forces. This isolates
coordinate scaling and Context overhead. The script starts at the ideal-gas mean
volume, uses 100 warmup steps and checks the final volume against a six-standard-
deviation bound. It measures 20,000, 2,000 and 300 steps at 512, 4,096 and 65,536
atoms, with three process repetitions for every case.

| Scaling | Atoms | Rust median steps/s | Reference median steps/s | Rust peak RSS, KiB | Reference peak RSS, KiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rigid | 512 | 138,020 | 42,349 | 3,976 | 7,232 |
| Rigid | 4,096 | 18,973 | 5,011 | 5,132 | 8,548 |
| Rigid | 65,536 | 1,335 | 313 | 23,892 | 29,800 |
| Atomic | 512 | 194,941 | 41,900 | 4,048 | 7,264 |
| Atomic | 4,096 | 24,494 | 5,012 | 4,880 | 8,620 |
| Atomic | 65,536 | 1,887 | 312 | 21,000 | 31,244 |

RSS is the maximum across the three processes. The first implementation reached
1,001 rigid and 1,346 atomic steps/s at 65,536 atoms. Reusing proposal storage and
specializing diagonal coordinate transforms improved throughput. Keeping the
buffer increased peak RSS by about 1.6 MiB in the rigid case, which exposed
additional copies in the final harmonic-bond energy/force query. Harmonic bonds
now borrow f64 CPU coordinates and transfer their output vector directly into
shared tensor storage. An allocation regression permits only the output force
array. A separate check verifies zero new dense arrays during warmed volume moves;
a retained State snapshot requires at most one copy-on-write array.

This benchmark covers coordinate updates and the Reference platform. Molecular
force-field throughput, optimized CPU/GPU platforms and equilibrium convergence
require separate measurements. Barostat trial energies still evaluate force
kernels that return both energy and forces; dedicated energy-only kernels remain
performance work.

Passing `anisotropic` to `scripts/benchmark-barostat.sh` runs the same workload
with only X allowed to change. It uses the same sizes, temperatures, pressures,
iteration counts and repetitions as the isotropic measurement.

| Scaling | Atoms | Rust median steps/s | Reference median steps/s | Rust peak RSS, KiB | Reference peak RSS, KiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rigid | 512 | 149,847 | 43,150 | 4,152 | 7,232 |
| Rigid | 4,096 | 20,727 | 5,000 | 5,160 | 8,636 |
| Rigid | 65,536 | 1,340 | 312 | 23,828 | 29,848 |
| Atomic | 512 | 162,513 | 41,663 | 4,200 | 7,280 |
| Atomic | 4,096 | 27,095 | 5,047 | 4,928 | 8,412 |
| Atomic | 65,536 | 1,884 | 308 | 20,940 | 31,244 |

The 65,536-atom cases run about 4.3 and 6.1 times faster and use less peak process
memory than Reference. This remains a coordinate-update benchmark with excluded
potential forces, so the molecular force-field performance target remains open.

Passing `membrane` runs XY-isotropic moves with Z fixed at 3 nm, pressure 20 bar
and surface tension 7 bar nm. Its initial volume and check use effective pressure
`P - gamma/Lz`; the remaining benchmark protocol is unchanged.

| Scaling | Atoms | Rust median steps/s | Reference median steps/s | Rust peak RSS, KiB | Reference peak RSS, KiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rigid | 512 | 154,046 | 42,195 | 4,036 | 7,272 |
| Rigid | 4,096 | 18,701 | 4,988 | 5,216 | 8,620 |
| Rigid | 65,536 | 1,337 | 310 | 23,860 | 29,856 |
| Atomic | 512 | 196,684 | 41,607 | 4,132 | 7,492 |
| Atomic | 4,096 | 27,689 | 5,058 | 5,028 | 8,768 |
| Atomic | 65,536 | 1,897 | 312 | 20,968 | 31,268 |

These cases retain the reused proposal storage. The benchmark measures coordinate
and Context overhead with potential forces excluded; it does not establish lipid
bilayer equilibration or molecular force-field throughput.

Passing `flexible` to the same benchmark permits all six box entries to change.
It uses the existing pressure, temperature, warmup, sizes and step counts. The
output also records box diagonals and tilts and verifies that the cell stays
finite and reduced.

| Scaling | Atoms | Rust median steps/s | Reference median steps/s | Rust peak RSS, KiB | Reference peak RSS, KiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rigid | 512 | 152,121 | 42,315 | 4,196 | 7,264 |
| Rigid | 4,096 | 20,961 | 5,039 | 5,128 | 8,644 |
| Rigid | 65,536 | 1,342 | 312 | 23,872 | 29,908 |
| Atomic | 512 | 194,623 | 41,466 | 4,176 | 7,480 |
| Atomic | 4,096 | 27,381 | 5,095 | 5,096 | 8,512 |
| Atomic | 65,536 | 1,887 | 314 | 21,016 | 31,248 |

These unrestrained fluid cells can develop large aspect ratios: at 512 atoms,
the final maximum/minimum diagonal ratios were about 11 and 22 in Rust and
7 and 5 in Reference for rigid and atomic scaling. Their random paths differ.
The finite timing runs verify coordinate-update throughput and a broad final-volume
bound, not equilibrium cell shape. Confined-shape statistics are tested separately.

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

## Variable-step integration

`VariableVerletIntegrator` and `VariableLangevinIntegrator` select a step from the
RMS acceleration, limit increases to twice the preceding internal step and retain
the old step when a proposed increase is less than 20 percent. A positive maximum
step size adds a cap; zero disables that cap. `stepTo` also caps the step at the
remaining time and finishes at the requested time. Past targets perform no steps.
The initial reported step size is zero. State updates run before selecting the
next step, so they observe the preceding reported step size, matching Reference.

Variable Verlet advances velocities by half the sum of the old and new steps,
then reconstructs the stored velocity from the constrained position displacement.
Variable Langevin reuses LFMiddle's kick, velocity projection, drift, thermostat,
drift and final position correction. Its random stream remains independent of
other Contexts. Both classes preserve fixed particles and recompute virtual sites.
Their kinetic-energy queries use a half-step force correction, including variable
Langevin, matching the supplied Reference kernel. Thermal velocity initialization
uses zero force correction for both variable-step classes, as their reference
classes inherit the base velocity-time offset. This offset is distinct from the
kinetic-energy correction.

Changing error tolerance resets the internal step history on the next execution.
Changing variable Langevin temperature or friction also resets that history, while
preserving its random stream. Setters that restore the preceding settings before
execution do not reset history. Setting the public step size changes the reported
value and kinetic-energy correction, but does not overwrite internal step history.
A newly bound Context starts fresh internal history. Python exposes live settings,
seeds, force masks and `stepTo` through both naming conventions.

The reference fixture compares 336 states, including constraints, unequal and zero
masses, virtual sites, force masks, global changes, maximum/error/friction changes,
reported-step overrides and bounded final steps. Six thermal cases independently
check position and velocity Gaussian moments over 36,000 components at three
temperatures and two frictions. The expected variances include covariance between
position and velocity across the two drifts. Separate tests check constant
acceleration, step growth and hysteresis, single precision, seed behavior,
ownership, invalid settings and reuse of dense f64 state storage.

`scripts/benchmark-stochastic.sh variable` uses the same free-particle protocol as
the fixed-step benchmark, with error tolerance `1e-5` and maximum step 0.001 ps.
Free particles always select the maximum. Both final simulation time and step
count are checked, along with kinetic energy. Reference timings include the
per-step clock repair described below. Throughput is the median and RSS the maximum
of three processes; the Langevin rows use 300 K.

| Integrator | Particles | Rust steps/s | Reference steps/s | Rust peak RSS, KiB | Reference peak RSS, KiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Variable Verlet | 8 | 1,253,734 | 2,697,447 | 3,740 | 6,784 |
| Variable Verlet | 4,096 | 24,602 | 11,948 | 4,812 | 7,432 |
| Variable Verlet | 65,536 | 1,510 | 740 | 16,016 | 17,984 |
| Variable Langevin | 8 | 924,600 | 1,260,466 | 4,104 | 7,028 |
| Variable Langevin | 4,096 | 5,656 | 3,544 | 5,088 | 7,368 |
| Variable Langevin | 65,536 | 364 | 221 | 16,400 | 19,544 |

The larger cases run faster and use less process memory in Rust. At eight particles,
Reference remains faster despite the added clock repair. This measures integration
and adaptive-estimator overhead with zero forces; changing-step molecular systems,
long-run stability and optimized CPU/GPU platforms remain separate requirements.

### Adaptive-clock compatibility exception

The supplied Reference kernels compare the chosen step with the already capped
maximum, then replace the clock with the requested final time when they are equal.
Consequently, `step(1)` with a positive maximum can set time to infinity; `stepTo`
can jump to a distant target after only one capped step. Both classes reproduce
this source failure with a 0.001 ps maximum. Rust instead adds the actual step,
and only snaps the clock to the target when the remaining interval was completed.

Reference trajectory fixtures repair only its clock after each ordinary step.
Their bounded-step helper temporarily lowers the maximum to the remaining time,
then restores it. Coordinates, velocities and selected steps still come from the
unmodified Reference kernel. The fixture records the original clock failure as a
separate exception. This correction is intentional and is not hidden by tolerance.

Zero forces and no finite step bound produce an infinite proposed first step.
Rust returns an integration error instead of producing nonfinite coordinates;
choose a positive maximum or a finite `stepTo` target for this case. Nonfinite targets are rejected.
These input-domain checks extend the source's behavior and do not establish full
error compatibility.

## Compound integration

`CompoundIntegrator` owns multiple child integrators and selects one for ordinary
Context steps. Compounds can contain other compounds. Step size, constraint
tolerance and kinetic energy use the selected child; setting integration force
groups applies recursively to all children. Selecting a different child leaves
stored velocities unchanged. When algorithms interpret those velocities at
different time offsets, the caller must adjust them, as required by OpenMM.
LFMiddle thermal initialization applies a half-step force shift, while its
kinetic energy uses the stored velocities without a shift.
Thermal velocity initialization in a compound uses the first child's offset, recursively,
matching the supplied implementation even when a different child is selected.

A direct `child.step()` integrates that child, including when it is inactive.
Its force mask and actual step size control the integration. Context state-update
forces still consult the top-level selected child's settings, and State kinetic
energy still uses that selection. Adaptive children retain their own step history
across switches and support `stepTo()` directly. The adaptive clock exception above
also applies within compounds. Each stochastic child retains its Rust random
stream; interleaving another child does not consume it. Cross-implementation
random paths are not promised.

Rust callers use `context.integrator_mut().select_integrator(index)` to change
the active child, `component(&[parent, child])` to access nested settings, and
`context.try_step_component(&path, steps)` or `try_step_to_component(&path, time)`
for direct child execution. Python exposes snake_case and OpenMM camelCase names
for the compound methods and returns the original child objects. Children have
one live owner; duplicate ownership, cycles and binding a child independently
are rejected. Binding checks the whole tree before creating a Context. Weak
Context links permit collection and rebinding without a reference cycle.

The comparison fixture covers 1,152 states across flat/nested compounds, three
child orderings, six integrator types, constraints, virtual sites, massless
particles, force masks, live settings, zero-temperature initialization and direct
inactive-child execution. Separate tests check state-update routing, bounded
adaptive clocks, f32 execution, ownership errors and retained random streams.
The warmed f64 CPU path reuses dense position/velocity storage; compound dispatch
does not allocate additional particle-sized buffers. Checkpoint serialization,
unsupported child integrators, full molecular switching protocols and GPU
execution remain outstanding.

`scripts/benchmark-stochastic.sh compound` alternates between two children every
step, including selection overhead. Conditions match the free-particle benchmark
above: f64, dt=0.001 ps, 10 warmup steps, three repetitions, and friction 0.7/ps
for LFMiddle. The table uses zero-temperature Verlet and 300 K LFMiddle. Throughput
is the median and RSS the maximum measured process peak, in KiB.

| Child type | Particles | Rust steps/s | Reference steps/s | Rust RSS | Reference RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Verlet | 8 | 1,855,475 | 3,265,720 | 3,836 | 6,920 |
| Verlet | 4,096 | 43,032 | 10,964 | 4,896 | 7,412 |
| Verlet | 65,536 | 2,866 | 675 | 16,128 | 20,248 |
| LFMiddle | 8 | 1,124,090 | 1,365,817 | 4,016 | 7,252 |
| LFMiddle | 4,096 | 6,121 | 3,601 | 5,192 | 7,732 |
| LFMiddle | 65,536 | 390 | 226 | 16,408 | 23,628 |

Final time and step count are checked, and kinetic energy must satisfy the
finite-time Gaussian expectation within six standard deviations. Rust is faster
and uses less process memory for the two larger sizes. Reference is faster at
eight particles. This measures free integration and switching overhead; force
fields, useful switching protocols, long-run stability and optimized OpenMM
CPU/GPU platforms are outside this benchmark.

### Compound ownership boundary

The supplied header requires adding every child before creating the Context.
Its `initialize()` does not set the compound's base owner field, so the source's
`addIntegrator()` guard can incorrectly accept a child afterward, leaving that
child uninitialized. Rust enforces the documented lifetime and rejects additions
after the first successful Context binding, including after Context collection.
This does not reproduce the source's invalid-lifetime behavior. Empty compounds
raise Python errors when queried or bound instead of dereferencing a missing child.

## Nose-Hoover chains

`NoseHooverIntegrator` implements the Reference middle splitting: full force
kick, velocity constraint projection, half drift, chain propagation, half drift,
position constraint projection with velocity correction, and optional pair hard
walls. Virtual sites are updated afterward. The constructor taking only a step
size starts with no thermostats; a full-system thermostat or disjoint subsystem
thermostats can be added before Context creation. Rust provides `new(step_size)`
and `thermostated(temperature, frequency, step_size, length, mts, ys)`;
Python accepts the corresponding OpenMM constructor forms.

Individual particles share the absolute bath with the center-of-mass motion of
registered pairs. Pair relative motion has a separate temperature and collision
frequency. Chain lengths are configurable, MTS counts are positive, and
Yoshida-Suzuki decompositions with 1, 3, 5 or 7 stages use the supplied weights.
Constraints reduce the absolute degrees of freedom; a CMMotionRemover subtracts
three more from the largest thermostat, resolving ties by insertion order.
Cross-thermostat constraints, overlapping entries, invalid indices and massless
individual thermostat atoms are rejected. A pair can have a massless second
particle, as in the Reference hard-wall branch.

Temperature, frequency, step size, constraint tolerance, force groups and maximum
pair distance are live settings. Python `getThermostat()` returns a retained view
of its chain, with live scalar setters and degrees-of-freedom updates. Particle
membership is frozen after initialization so the integration lists and masks
cannot disagree. Hard-wall temperature remains the relative temperature captured
when the Context was created, matching the supplied implementation even after a
live relative-temperature change. Individual integration lists include every
non-pair atom, including unthermostated and massless particles. Thermal velocity
initialization and kinetic energy use stored velocities without a force shift.

Heat-bath energy includes the chain kinetic and potential terms. Before a chain
has propagated it contributes zero, avoiding an uninitialized-vector access in
the Reference kernel and matching the Common backend's safe behavior. Chain
state persists across step-size and thermostat-parameter changes, and resets
when the integrator is initialized for a new Context. Nested compound binding
configures each child from the System. Checkpoint and XML serialization are still
outstanding, as are the Drude plugin, molecular-system validation and resident
GPU execution. Non-finite or nonpositive temperatures/frequencies, zero MTS
counts, and nonpositive active degrees of freedom are rejected before execution;
these produce singular masses or undefined updates in the supplied code.

The reference fixture compares 3,584 states spanning two chain lengths, two MTS
counts, all four Yoshida-Suzuki decompositions, full/subsystem/pair-only baths,
constraints, virtual sites, force masks, zero-temperature velocity initialization,
live chain settings, and hard-wall reflections. Four additional states check
massive/massless pair parents and the captured hard-wall temperature. Rust tests
also check f32 execution, degree counting, scalar one-bead propagation, constrained
NVE behavior, zero-velocity initialization and ownership validation.

Six oscillator cases use 24 massive particles at 70, 300 and 600 K, each with
collision frequencies 1 and 10/ps. Each run uses dt=0.001 ps, 20 ps warmup and
100 ps measurement, sampled every 0.01 ps. Both implementations must agree on
kinetic block means and remain within 10% of the equipartition mean, 30% of the
canonical kinetic variance, and 1% of the target mean kinetic energy for maximum
extended-energy drift. The measured maximum relative deviations were 0.84% for the kinetic mean and
6.06% for its variance; maximum extended-energy drift was 0.130% of the target
mean kinetic energy. Reference and Rust gave the same oscillator summaries.
These are finite oscillator checks, not a claim of convergence or stability for
general molecular systems.

### Approved pair thermostat and kinetic-energy exceptions

The supplied Reference and Common chain kernels gate absolute propagation on the
number of individual atoms. With pairs only and nonzero COM kinetic energy,
Reference leaves the absolute scale at zero and erases COM velocity at the first
step. Rust propagates the absolute chain whenever its thermostat contains atoms
or pairs. The fixture keeps an extra stationary, force-free Reference atom and
subtracts its three degrees of freedom to activate the missing branch without
changing the pair dynamics. An independent one-bead recurrence verifies the
corrected pair velocities and energy; an unmodified-source probe records the
original zero-COM result.

The source integrator reports only each thermostat's absolute masked kinetic
energy, omitting pair relative motion and unthermostated particles. Rust reports
one half of the mass-weighted squared velocity over all particles. Reference
comparisons compute this same total directly from its saved velocities and record
the source API's omitted energy separately. These differences are intentional
compatibility exceptions approved for physical correctness.

The Reference guard for absolute masked kinetic energy below `1e-8` skips both
chains even when the pair has relative motion. Rust retains the absolute-chain
guard but propagates each registered relative chain independently of absolute
kinetic energy. This third correction is also approved. A source probe records
the unmodified behavior, while a scalar one-bead recurrence checks the corrected
relative velocities, heat-bath energy and conserved COM momentum.

A hard-wall violation with exactly zero radial velocity makes Reference divide
by zero. Rust instead chooses finite inward radial velocities, conserving pair
COM momentum for two massive particles; the massless-parent case moves only the
massive particle. The zero-radial rule is a defensive extension to this singular
source branch and has an independent invariant test.

### Nose-Hoover performance

`scripts/benchmark-stochastic.sh nose` measures free-particle Context stepping in
f64 at 300 K, frequency 0.7/ps and dt=0.001 ps, using length 3, 3 MTS and 7
Yoshida-Suzuki stages. Pair cases contain two individual atoms plus disjoint
pairs, with relative temperature 30 K and frequency 2/ps. Alternating atoms start
with velocities (0.1,0.2,0.3) and (0.2,0.4,0.6) nm/ps, exercising both baths.
After 10 warmup steps, the three sizes run 30,000, 1,500 and 200 measured steps.
Three process repetitions alternate implementation order. The table shows median
throughput and maximum process RSS in KiB. Final full-particle kinetic energies,
clock and step count are checked across implementations.

| Bath layout | Particles | Rust steps/s | Reference steps/s | Rust RSS | Reference RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Individual atoms | 8 | 390,090 | 409,712 | 4,032 | 7,184 |
| Individual atoms | 4,096 | 17,528 | 8,604 | 5,132 | 7,888 |
| Individual atoms | 65,536 | 1,162 | 543 | 18,492 | 22,492 |
| Atoms and pairs | 8 | 229,087 | 222,013 | 4,028 | 7,052 |
| Atoms and pairs | 4,096 | 12,110 | 6,701 | 5,052 | 8,120 |
| Atoms and pairs | 65,536 | 802 | 420 | 18,376 | 25,444 |

The warmed f64 CPU integration path reuses dense particle buffers and small
chain-mass/force arrays; an allocation regression checks that unconstrained steps
do not allocate particle-sized temporaries. Rust uses less process memory in
these cases and is faster at the larger sizes. Reference remains slightly faster
for eight individual atoms. These timings exclude potential-force costs and do
not establish equilibrium sampling or optimized OpenMM CPU/GPU performance.

## Dissipative particle dynamics

`DPDIntegrator` implements the Reference middle splitting: full force kick,
velocity constraints, half drift, sequential radial pair thermostat updates,
second half drift, position constraints, velocity reconstruction and virtual
sites. Kinetic energy uses stored velocities, while thermal initialization uses
a half-step force offset. Temperature, step size, constraint tolerance and force
groups remain live. Particle types, pair overrides, default friction/cutoff and
periodic-system status are captured when a Context is created. Pair overrides
are symmetric, later entries win, and invalid entries are rejected even when
neither type occurs in the System. Python exposes both naming conventions,
retains original integrator objects and supports nested CompoundIntegrator
selection and direct execution of an inactive child.

The pair thermostat uses the midpoint distance `r`, weight `w=1-r/cutoff`, decay
`a=exp(-2*friction*w*w*dt)`, and reduced mass `mu`. Its radial relative velocity
has conditional mean `a*v_radial` and variance `(1-a*a)*kT/mu`. Each update
preserves the two massive particles' total momentum and leaves their tangential
relative velocity unchanged. Particle velocities are updated in place, so pair
order matters. The CPU implementation preserves Reference voxel traversal and
particle insertion order. Neighbor membership is evaluated at the old positions
with the maximum captured cutoff, then each pair is tested against its own
cutoff at the midpoint. A pair that enters the cutoff during the first half drift
can therefore be absent from that step, matching the supplied implementation.

The current box is passed to the integrator after Context state updates, including
barostat changes. Reference periodic voxel wrapping and triclinic minimum images
are retained. This target is specifically the Reference sequential algorithm;
the supplied Common kernel accumulates pair increments from pre-thermostat
velocities. Reference agreement does not establish GPU trajectory agreement.

### Storage and traversal

On the f64 CPU path, positions and velocities are updated directly and one
midpoint buffer is reused. A small-system path visits candidate pairs directly
and sorts them in Reference traversal order. Larger periodic systems use a dense
cell array when the cell count is at most three times the particle count and at
most 16 million cells. Other grids use a reusable hash map and linked particle
indices. No full pair list or per-cell particle vectors are allocated each step.
A warmed unconstrained Context has a regression test that rejects new dense
particle-buffer allocations. The f32 path also works, but currently converts
state to host f64 scratch rather than using the direct f64 update path.

Captured type overrides use sparse symmetric entries instead of a quadratic
matrix over all distinct types. Storage scales with particles, occupied cells,
and configured overrides; unused type pairs do not enlarge the search cutoff.
The default cutoff and every applicable override contribute to the captured
maximum, including an earlier override superseded by a later entry, as in
Reference.

### Numerical and statistical checks

The deterministic fixture checks 1,728 states and 722,304 scalar comparisons across
non-periodic, orthorhombic
and triclinic systems; default and explicit types; constraints and virtual sites;
and standalone, selected nested and directly called inactive integrators. It
changes live settings and box vectors, checks captured defaults and overrides,
and exercises both small-system and dense-cell traversal. Positions and forces
matched exactly; maximum velocity error was `2.8e-17` nm/ps and maximum kinetic
energy error was `9.7e-10` kJ/mol. Core tests also check
f32 execution, exact two-particle damping, cutoff crossings, sparse configuration,
fixed particles, RNG reproducibility and unchanged tangential motion.

Six one-step ensembles use 16,000 samples each at 50, 300 and 600 K with friction
1 and 20/ps. Reference and Rust are independently checked against the conditional
radial Gaussian mean and variance; matching random paths is not required. Three
periodic ideal-fluid checks use 64 particles in a 3 nm cube, cutoff 1.2 nm,
friction 10/ps and dt=0.002 ps at 70, 300 and 600 K. Each has 20 ps warmup and
100 ps measurement, sampled every 0.02 ps. Total momentum starts at zero, so the
kinetic distribution is checked with `3*(64-1)` degrees of freedom. Mean kinetic
energy must be within 6% of equipartition, variance within 20% of the canonical
value, and momentum error below `3e-10` Da nm/ps. The measured maximum relative
deviations were 0.993% for the mean and 9.46% for the variance, with maximum
momentum error `1.01e-12` Da nm/ps across both implementations. These finite ideal-fluid checks
do not establish molecular force-field stability or general sampling convergence.

### Approved fixed-particle exception and singular inputs

For a massive particle interacting with a massless particle, the supplied
Reference and Common formulas use `m1*m2/(m1+m2)=0` and then divide thermal energy
by that value. Reference produces NaN positions and velocities even at zero
temperature. The approved correction sets a massless particle's inverse mass to
zero and uses `mu=1/(inverse_mass1+inverse_mass2)`. The massive particle then
receives the radial OU update, while the fixed particle's state is unchanged.
Its momentum exchange is with the fixed external support, so two-particle momentum
conservation is not asserted in this case. Two massless particles are skipped.
Independent tests cover zero-temperature damping and finite-temperature variance
for both index orders; source probes retain the original NaN results.

Coincident interacting particles have no defined radial direction. Rust reports
an explicit error instead of propagating the source NaNs. Periodic cutoffs larger
than a box diagonal and unrepresentable voxel grids also produce explicit errors.
These input checks do not define new collision physics. Checkpoints,
serialization, resident GPU execution and broader molecular DPD validation remain
outside the implemented subset.

### DPD performance

`scripts/benchmark-stochastic.sh dpd` compares single-process f64 integration
against OpenMM Reference at 0 and 300 K, friction 0.7/ps, cutoff 0.7 nm and
dt=0.001 ps. Particles start on a 0.3 nm cubic lattice in a periodic cube whose
side is `max(5, ceil(cuberoot(N)))*0.3` nm. Alternating velocities are
`(0.1,0.2,0.3)` and `(0.2,0.4,0.6)` nm/ps. An empty periodic force marks the
System as periodic, and the integration group mask excludes potential forces.
After ten warmup steps, 8/4096/65536-particle cases time 3000/100/10 steps.
The table reports the median of three runs and maximum process RSS.

| Temperature, K | Particles | Rust steps/s | Reference steps/s | Rust peak RSS, KiB | Reference peak RSS, KiB |
| --- | --- | --- | --- | --- | --- |
| 0 | 8 | 210097 | 90677 | 4092 | 7448 |
| 0 | 4096 | 31.16 | 19.52 | 5372 | 9544 |
| 0 | 65536 | 2.18 | 1.05 | 24496 | 44676 |
| 300 | 8 | 221439 | 90723 | 4168 | 7276 |
| 300 | 4096 | 29.85 | 18.64 | 5352 | 9400 |
| 300 | 65536 | 2.13 | 1.08 | 24436 | 44660 |

The initial hash-only traversal reached about 42,000 steps/s for eight particles
and 1.43 steps/s for 65,536 particles. Direct pair traversal for small systems
and dense periodic cells improve both cases without changing the pair-update
order. The final clock, finite energy and total momentum are checked in every
run. Zero-temperature kinetic energy is also compared between implementations;
positive-temperature paths differ. These short timing runs do not test
equilibrium or molecular force-field throughput. Optimized OpenMM CPU and GPU
platforms are not included.

## Adaptive quantum thermal bath

`QTBIntegrator` implements the Reference middle splitting with a full force kick,
velocity constraints, two half drifts around a colored-noise velocity update,
position constraints and velocity correction. Virtual sites are updated after
integration. Temperature changes take effect at the next segment boundary;
constraint tolerance and integration groups remain live. Step size, friction,
segment length, cutoff, particle types, adaptation rates and the random stream
are captured when the Context is initialized. Their getters still expose newly
assigned settings, which take effect on reinitialization.

Explicit particle types are arbitrary integers. Particles of one type must have
identical masses and share adapted friction coefficients. Each particle without
an explicit type receives its own type. Fixed particles remain fixed. Both
Python naming styles and nested CompoundIntegrator binding are supported.

The physical step size is stored separately from its requested setting. After a
late `setStepSize()`, the Context clock, thermostat state updates and thermal
velocity offset use the captured physical step size. This is an approved
compatibility exception: the supplied Reference kernel integrates with the old
step size while advancing its clock with the new one. Its thermal velocity
offset also reads the new value. Reference fixtures repair only the recorded
clock; their position and velocity propagation is unchanged.

### Frequency filtering and adaptation

A segment contains `L = ceil(segment_length/dt)` steps. Its noise is generated
with an FFT of length `M = 3*L`, retaining the central `L` samples. Rust uses all
`floor(M/2)+1` real-FFT frequency bins at angular frequencies
`omega[k] = 2*pi*k/(M*dt)`, including the Nyquist component when `M` is even.
This is the second approved exception. The supplied kernel uses `(M+1)/2` bins,
leaving the even-length Nyquist component as unfiltered white noise. With odd
`M`, its frequency spacing also differs from the actual FFT spacing. A zero-K
bath with a cutoff below every nonzero frequency therefore still moves particles
in the supplied even-length implementation; the corrected implementation does
not. The adapted-friction array gains one element for even `M`, and its frequency
interpretation follows the actual FFT grid for both parities.

The target quantum-oscillator spectrum, deconvolution, logistic cutoff and
finite-step Langevin compensation follow `QTBIntegratorUtilities` and
`ReferenceQTBDynamics`. The deconvolution uses the corrected frequency spacing
and the complete bin count. After each segment, zero-padded velocity and random
force transforms estimate the fluctuation-dissipation residual for each type.
The rate-scaled residual updates that type's friction spectrum, with coefficients
clamped at zero. Temperature changes recompute the target spectrum at the next
boundary; adaptation still uses the preceding segment's recorded data.

Friction must be positive when creating a Context because the supplied
spectral equations are singular at zero friction. Segment length and cutoff
must be positive and finite, and adaptation rates must be nonnegative and
finite. Invalid particle indices and unequal masses within a type produce
errors. These checks replace undefined or nonfinite source behavior for invalid
configurations.

### Spectrum and memory optimization

The source constructs a dense deconvolution matrix using cubic work. With
`M_ij = dw*C_ij*scale_i`, that matrix action is `M^T*(M*f)`. Rust evaluates the
two factors directly and reuses a single row buffer. The same 20 iterations now
require quadratic work and linear auxiliary memory in the frequency count.
`scripts/benchmark-qtb-spectrum.sh` compares the original frequency grid against
the unmodified source utility in 58 conditions: 7,608 values agree with maximum
absolute error `2.42e-12` and scaled relative error `3.68e-14`.

| Frequencies | Rust median seconds | Reference median seconds | Speedup |
| --- | --- | --- | --- |
| 512 | 0.01388 | 0.83537 | 60.2 |
| 1500 | 0.11615 | 20.55003 | 177.0 |

Both utilities run on one CPU thread. These are medians of three runs; median
peak process RSS for the complete 58-case process is 3,052 KiB for Rust and
24,044 KiB for Reference. These timings measure spectrum construction alone.

The trajectory implementation caches FFT plans and scratch buffers. It uses each
degree-of-freedom index as a ChaCha stream identifier, retains its word position
and regenerates each overlapping white-noise window. This preserves the shared
samples between adjacent windows without storing the white-noise history.
Random forces require `24*N*L` bytes; segment velocities require another
`24*N*L` bytes only when adaptation is enabled. The source stores `120*N*L`
bytes for these three histories. Adapted spectra initially share storage and
copy only when a type is updated. FFT scratch and type metadata are additional.
The tradeoff is regenerating three windows' worth of Gaussian samples per new
segment. A direct DFT test checks the reconstructed windows across odd and even
segment lengths. Warm f64 CPU steps, including segment boundaries, allocate no
new dense particle buffers.

### Numerical coverage and performance

The reference fixture checks 864 deterministic states and 53,136 values across
constraints, virtual sites, fixed particles, types, captured settings, force
groups, parameter changes and nested/direct-child execution. Maximum position,
velocity, force and kinetic-energy differences are respectively zero,
`4.34e-19`, `1.11e-16` and `6.71e-10` in OpenMM units. Clock values agree after
repairing the source clock discrepancy.

Eight positive-temperature or zero-point-noise conditions test every within-
segment covariance against the analytic filtered Gaussian spectrum. Each uses
9,216 independent coordinate streams; the largest mean and covariance residuals
are 1.90 and 2.54 standard errors. A separate direct-DFT calculation checks 128
adapted coefficients over four segments, including shared types, fixed particles
and 15 zero clamps; its largest error is `2.19e-11`. API tests check delayed
temperature changes, seed capture, object ownership, reinitialization and invalid
inputs. The CPU fallback also covers f32 damping and captured clock behavior.

Four harmonic-oscillator cases additionally use 32 independent particles at
70/300 K and angular frequencies 20/100 per ps. Friction is 10/ps, the step is
0.001 ps, the segment contains 501 steps, and one shared type adapts at rate
0.01. Each run equilibrates for 10 ps and measures another 100 ps, recording
10,000 samples. Across Rust and Reference, the mean potential energy differs
from the quantum harmonic target by at most 1.27%, and its variance by at most
5.00%. The corresponding Rust maxima are 0.76% and 3.77%. Raw kinetic energies
are recorded separately and are not compared to the potential-energy target.
These checks cover harmonic systems; they do not establish anharmonic leakage
control or general molecular convergence.

`scripts/benchmark-stochastic.sh qtb` measures single-thread free-particle
integration at 300 K, friction 0.7/ps, `dt=0.001 ps`, 33 steps per segment and
cutoff 500/ps. All particles share a type. After 10 warmup steps, the
8/4096/65536-particle cases time 9900/330/66 steps. Every timed run includes
colored-noise generation. Initial spectrum construction is excluded. The table
reports medians of three runs and maximum process RSS.

| Adaptation rate | Particles | Rust steps/s | Reference steps/s | Rust peak RSS, KiB | Reference peak RSS, KiB |
| --- | --- | --- | --- | --- | --- |
| 0 | 8 | 484769 | 97154 | 4816 | 7368 |
| 0 | 4096 | 1242.97 | 237.19 | 9256 | 24480 |
| 0 | 65536 | 74.18 | 15.79 | 74320 | 281928 |
| 0.01 | 8 | 340090 | 93154 | 4748 | 7552 |
| 0.01 | 4096 | 751.46 | 237.23 | 12412 | 24328 |
| 0.01 | 65536 | 29.71 | 15.80 | 124880 | 281972 |

The Reference benchmark explicitly selects one worker through a small
`ReferencePlatform` subclass; its default pool otherwise uses the number of
online CPUs. Source and Rust frequency grids differ by the approved correction,
and their random paths differ. These timing cases check finite raw kinetic
energy, the clock and step count. Noise statistics and adaptation are validated
separately. The tests do not establish equilibrium for a molecular force field,
zero-point leakage control in anharmonic systems or GPU performance. Raw kinetic
energy is reported as `sum(m*v^2/2)`; classical temperature and pressure formulas
must not be interpreted as quantum thermodynamic estimates. Checkpoint and state
serialization of QTB histories remain part of the unfinished serialization work.

## Programmable integration

`CustomIntegrator` executes global assignments, per-DOF assignments, sums,
position/velocity constraints, explicit Context state updates, and nested `if`
and `while` blocks. Rust and Python expose the computation list, named global
and per-DOF variables, tabulated functions, random seed, kinetic-energy expression,
step size, constraint tolerance and integration force groups. The Python wrapper
provides snake_case and OpenMM camelCase methods and the nine computation constants.
CompoundIntegrator supports selected, nested and directly invoked inactive children.

Expressions bind `dt`, masses, positions, velocities, force-group forces and
energies, Context parameters, integrator variables and registered energy parameter
derivatives. Per-DOF operations include `dot`, `cross`, `vector`, `_x`, `_y` and
`_z`, and tabulated functions evaluate component by component. Sums include all
three components of each massive particle; fixed particles retain their stored
values and contribute nothing. Repeated uses of `uniform` or `gaussian` in one
expression share one draw per component. A condition shares its scalar draws
between both sides of the comparison. Each integrator owns its random stream;
nonzero seeds reproduce Rust trajectories, but Reference trajectories use a
different generator and consume unused random draws as well.

Context updates run only when the program reaches `UpdateContextState`. The
ordinary integrator's automatic update is skipped. Position constraints use the
step-start positions, then replace that reference after each position projection;
velocity constraints use the most recent such reference, matching Reference.
Virtual sites are refreshed at the end of a step. The custom kinetic-energy
expression is summed over massive degrees of freedom, and its `f` refers to the
force groups requested by the energy query. Thermal velocity initialization has
no half-step force offset for this integrator.

The computation list and variable declarations freeze at Context binding.
Reference prepares expressions lazily, on the first step or kinetic-energy query.
Rust follows that timing for the default force-group mask, kinetic-energy
expression and table contents. Later table or kinetic-expression setters update
the definition returned by getters; a new Context is required to compile those
changes. Global and per-DOF values, step size and tolerance remain live. Invalid
variables, targets, block structure and function calls produce errors before the
program executes. The supplied analyzer also rejects `fN` together with `energyN`
in one computation, even for the same N; this rejection is currently preserved.

### Program storage and evaluation

CPU f64 programs update only the destination array. Read-only sums and auxiliary
variable assignments borrow positions and velocities, preserving shared State
snapshots without copying them. Per-DOF arrays, constraint reference positions and
per-group force buffers are retained. The f32 backend uses an explicit host
conversion path. There are no resident GPU program kernels yet.

The value evaluator omits reverse-mode differentiation and partitions operations
by their input dependencies. Expressions that depend only on step size, globals,
Context parameters, energy or energy derivatives are evaluated before the particle
loop. Scalar per-DOF expressions reuse one compiled program for three components;
vector expressions reuse identical component programs, so a dot product is
calculated once. Tabulated data remains shared. No compilation or particle-sized
scratch allocation occurs during warmed unconstrained execution.

Force, energy and derivative caches belong to their force-group masks. Position,
box, Context parameter and force-definition changes invalidate them. Valid caches
survive successive steps; velocity-only updates preserve them. An explicit
barostat move invalidates caches when its proposal is accepted. The regression
suite checks cross-step force counts, external changes and energy before/after
volume moves, in addition to the existing allocation checks.

### Reference exceptions

The supplied Reference kernel captures the initial step size for its clock while
its expressions read the updated value. Changing an initial 0.001 ps step to
0.002 ps advances coordinates by 0.002 ps but records only 0.001 ps. Assigning
`dt=0.003` in the program has the same defect. Rust advances time by the updated
step size, as the Common implementation does. Explicit Context updates also see
the current step size of the active integrator; a directly invoked inactive child
does not replace the root integrator's settings for those updates.

Reference's vector evaluator binds `f` but omits numbered forces and energy
variables. Consequently `cross(x,f0)` and `vector(energy,energy,energy)` throw
missing-variable errors. Rust binds them to the selected force/energy cache,
consistent with the Common implementation. Original Reference clock and variable
errors remain in the fixture alongside the corrected Rust assertions. GPU behavior
has only been inspected in the source and is not claimed as validated execution.

The source keeps one shared energy-derivative map while caching multiple group
energies. After evaluating `energy1` and `energy2`, `deriv(energy1,lambda)` can
therefore return group 2's derivative. With `E1=lambda*r`, `E2=5*lambda*r` and
`r=2`, Reference returns 10 instead of 2. An independent Reference derivative
query and finite difference both give 2. Rust retains derivatives with each cached
group, so the result corresponds to the energy named in `deriv`.

### Numerical coverage

The fixture compares 600 states across five programs, three direct/compound
routing modes, constraints, fixed particles, virtual sites, group selection,
parameter changes, table evaluation, nested control flow, vector operations and
kinetic-energy expressions. Python checks cover definition access, ownership,
Context release/rebinding, late table changes and invalid programs. Gaussian and
uniform moments and component covariances are checked with 8,192 massive particles;
additional checks cover repeated random references, conditional draws and seed
behavior. The original Source defects are reported separately rather than hidden
inside comparison tolerances.

Across those 600 states, positions, velocities, forces, potential energies and
times agree exactly with the supplied Reference build; the largest kinetic-energy
difference is 4.19e-9 kJ/mol. Global and auxiliary-variable differences remain
below 6.7e-16 and 5.3e-18, respectively.

A conservative harmonic program runs for 100 ps (100,000 steps). Its maximum
relative energy drift is 5.83e-6 in both implementations, and final coordinates
and velocities agree within 3e-10. A 256-particle BAOAB program thermalizes for
20 ps and then measures harmonic position/velocity moments for 60 ps. The four
block-mean deviations are below 0.88 standard errors for Rust and 0.58 for
Reference. These tests exercise force-cache reuse over long step sequences; they
do not establish stability for arbitrary programs or complete molecular force fields.

### Program evaluation performance

`scripts/benchmark-stochastic.sh custom` measures three free-particle programs
with ten warmup steps and three repetitions, reversing process order for the
second repetition. Both implementations update velocity and position, store an
auxiliary per-DOF value and compute a global sum. The vector variant includes
cross products, component extraction, vector construction and a dot product. The
Langevin variant applies an exact OU velocity update with friction 0.7/ps; its
kinetic energy is checked against the finite-time Gaussian distribution. All
programs use a 0.001 ps step. The scalar/vector variants use deterministic
velocities. Force evaluation, molecular interactions and GPU execution are absent.

The table gives median steps/s and the maximum process peak RSS of the three
runs. RSS includes runtime and library overhead. Measurements use 3,000, 300 and
30 timed steps at 8, 4,096 and 65,536 particles, respectively. The eight-particle
runs are short and particularly sensitive to process scheduling. Rust generates
random values only for expressions that reference them, whereas Reference also
generates unused draws; this contributes to the measured difference.

| Program | T (K) | Particles | Rust steps/s | Reference steps/s | Rust RSS (KiB) | Reference RSS (KiB) |
|---|---:|---:|---:|---:|---:|---:|
| Scalar | 0 | 8 | 292,458 | 272,498 | 4,376 | 8,024 |
| Scalar | 0 | 4,096 | 840.09 | 652.31 | 5,312 | 8,948 |
| Scalar | 0 | 65,536 | 50.67 | 40.90 | 19,816 | 26,320 |
| Vector | 0 | 8 | 253,210 | 109,341 | 4,476 | 8,064 |
| Vector | 0 | 4,096 | 704.21 | 261.86 | 5,352 | 9,120 |
| Vector | 0 | 65,536 | 42.36 | 16.45 | 19,832 | 26,244 |
| Langevin | 0 | 8 | 264,479 | 256,338 | 4,524 | 7,852 |
| Langevin | 0 | 4,096 | 699.29 | 582.69 | 5,404 | 9,144 |
| Langevin | 0 | 65,536 | 42.97 | 36.75 | 18,512 | 26,264 |
| Langevin | 300 | 8 | 260,919 | 252,203 | 4,448 | 8,004 |
| Langevin | 300 | 4,096 | 680.39 | 580.58 | 5,560 | 8,948 |
| Langevin | 300 | 65,536 | 42.64 | 36.37 | 18,300 | 26,248 |

## OBC implicit solvation

`GBSAOBCForce` implements OBC-II Born radii, polar solvation and the ACE surface
term. Rust and Python expose particle charge/radius/scaling parameters, all three
nonbonded methods, cutoff distance, solute/solvent dielectrics, surface energy,
force group and name. Python provides both snake_case and OpenMM camelCase names.
Particle updates affect a bound Context only after an explicit update call;
method, cutoff, dielectric and surface settings remain those captured at Context
creation. The force also supports nested CustomCVForce evaluation and updates.

The implementation uses the Reference dielectric offset of 0.009 nm, probe
radius of 0.14 nm and OBC-II coefficients (1, 0.8, 4.85). Polar self terms use the
square-root denominator even for negative Born radii, preserving the supplied
Reference behavior for positive input radii below the offset. This is API/numeric
coverage, not a recommendation to use those radii. A radius equal to the offset
makes the model singular. Nonfinite parameters and nonpositive active cutoffs
are rejected; finite positive radii and scale factors are checked at binding.
As in Reference, a zero solute or solvent dielectric disables the polar term.

Cutoff methods include pairs exactly at the cutoff, truncate the descreening
integrals and subtract the constant polar pair energy at the cutoff. The burial
contribution can still make energy discontinuous there; a switching function is
not introduced. Periodic evaluation uses the current reduced triclinic box and
checks its half-width cutoff restriction. Other methods ignore box vectors.

Three pair passes compute descreening sums, polar interactions and the Born-radius
chain-rule force. Each unordered pair supplies both directed descreening terms.
Cutoff traversal uses the reusable cell storage, without a stored pair matrix.
NoCutoff retains quadratic work. The native f64 path borrows positions and reuses
three scalar arrays, totaling 24 bytes per particle, excluding parameters,
neighbor storage and the returned force array. The force output transfers its
allocation directly into the CPU tensor. Nested CV evaluation can instead retain
one reusable force buffer and accumulate it with the CV coefficient. A warmed
allocation test checks one particle-sized allocation for a returned force tensor
and none for direct host accumulation. The f32 backend converts explicitly;
resident GPU force kernels remain unimplemented.

### Solvation validation

The fixture compares 1,536 states and 474,624 scalar values across methods,
dielectric/surface settings, unequal and nested radii, fixed particles, direct
and CV use, force groups, live boxes, parameter updates and short trajectories.
Maximum absolute differences are 1.57e-11 kJ/mol in energy and 1.71e-12 kJ/mol/nm
in force. Another 792 finite-difference components agree within 8.30e-8 kJ/mol/nm.
Six cases check the exact cutoff and its two sides, and two isolated-particle
cases cover positive radii below the dielectric offset. Native tests also cover
single-precision input, isolated analytic energies and zero total force.

Two 16-particle systems combine GBSA, Coulomb/Lennard-Jones interactions and
harmonic bonds for 20,000 steps at 0.0005 ps. At 1 ps, coordinate and velocity
differences are below 1.51e-12 nm and 4.28e-12 nm/ps. Over 10 ps, Rust's maximum
energy drift is 1.32e-4 kJ/mol (4.04e-6 relative), and total momentum changes by
less than 7.6e-14 Da nm/ps. Reference has the same energy-drift scale and momentum
changes below 3.9e-10 Da nm/ps.

Pointwise trajectory equality is not asserted at 10 ps. The maximum differences
reach 4.89e-6 nm and 1.80e-5 nm/ps. An independent Reference run perturbs one
initial coordinate by 1e-12 nm; it subsequently differs by up to 2.36e-6 nm and
8.47e-6 nm/ps. The short-time parity and sensitivity control are consistent with
amplification of floating-point differences. Both final differences and controls
are retained in the validation output. These synthetic systems do not establish
biomolecular ensemble convergence or production force-field readiness.

### Solvation performance

`scripts/benchmark-gbsa.sh` compares native release builds on a 0.3 nm lattice,
with varying charge/radius/scale parameters and a 0.8 nm cutoff. Each process
warms once and times 100, 10 or 1 complete force-and-energy requests for 512,
2,048 or 8,192 particles. Three independent processes alternate implementation
order; the table reports median throughput and the largest process peak RSS.
The summary checks energy and force-norm agreement before accepting timings.
These measurements include Context overhead and returned force arrays, but do
not measure integration, solvated biomolecules or GPU execution. One periodic
2,048-particle Rust sample was slower than the other two (10.00 versus about
13.27 evaluations/s); raw samples preserve that variation.

| Method | Particles | Rust evaluations/s | Reference evaluations/s | Rust peak RSS KiB | Reference peak RSS KiB |
|---|---:|---:|---:|---:|---:|
| NoCutoff | 512 | 55.705 | 44.207 | 4,160 | 7,216 |
| NoCutoff | 2,048 | 3.381 | 2.660 | 4,644 | 7,780 |
| NoCutoff | 8,192 | 0.206 | 0.141 | 6,664 | 8,868 |
| Nonperiodic cutoff | 512 | 229.599 | 132.161 | 4,332 | 7,444 |
| Nonperiodic cutoff | 2,048 | 50.519 | 10.593 | 4,596 | 7,736 |
| Nonperiodic cutoff | 8,192 | 10.922 | 0.718 | 6,556 | 8,740 |
| Periodic cutoff | 512 | 81.276 | 39.449 | 4,200 | 7,268 |
| Periodic cutoff | 2,048 | 13.266 | 2.932 | 4,692 | 7,716 |
| Periodic cutoff | 8,192 | 4.223 | 0.191 | 6,704 | 8,720 |

## Programmable generalized Born forces

`CustomGBForce` supports the three nonbonded methods, directed pair sums for the
first computed value, subsequent single-particle values, single/pair energy
terms, exclusions, six table types, global parameters and requested energy
parameter derivatives. Pair energies visit each unordered pair once; the first
computed value visits both directions. Single-particle expressions can depend
on Cartesian coordinates. Particle parameters use contiguous storage.

Computed values and reverse adjoints each occupy `N*M` doubles for `N` particles
and `M` computed values. The reverse pass propagates force and global-parameter
derivatives without allocating an `N*M*P` tensor for `P` derivative parameters.
Expression inputs are bound once, constant operations are prepared before a
particle pass, and derivative instructions omit inputs that cannot contribute
to a requested derivative. The expression evaluator remains an interpreter;
these schedules do not generate native machine code. Cutoff evaluations reuse
cell storage. Nested CV evaluations release prepared force storage before a
separate derivative evaluation to avoid retaining its workspace lock.

Particle updates are copied into an existing Context. A changed table update
count also reloads the current computed-value and energy expressions, matching
the supplied Reference implementation. Without such a table change, modified
expressions remain inactive until a new Context is constructed. Runtime global
values, exclusions and cutoff settings retain their existing Context values.
The deprecated continuous-1D function API and table resizing/renaming are also
covered. An empty computed-value list is rejected before evaluation; the supplied
Reference implementation dereferences its first value and can crash instead.

The reference fixture spans 2,592 states from three models (a multistage chain,
OBC and all six table types), three cutoff methods, both pair exclusion modes,
direct and nested CV use, orthorhombic/triclinic boxes, force groups, particle/table
updates, virtual sites, massless particles and short trajectories. Independent
finite differences of source energies check every mobile coordinate and both
global derivatives in 72 configurations. Rust regressions also check exact
cutoff inclusion, f32, four-stage derivative propagation and nested CV lock
release. An allocation regression with 4,096 particles, four computed values and
24 requested derivatives checks that working storage does not scale with the
product of particle, computed-value and derivative counts. Warm host force
accumulation reuses its dense buffers; returning an owned force tensor allocates
one output buffer.

The direct comparisons produced maximum absolute errors of `1.82e-12` kJ/mol
in energy, `1.09e-12` kJ/mol/nm in force, and `1.10e-11` in global derivatives.
Across 1,656 finite-difference components, the maximum errors were `3.23e-7`
for forces and `6.51e-7` for parameter derivatives.

A separate five-ps fixture combines custom OBC, Lennard-Jones/electrostatics and
harmonic bonds with a 0.1-fs step. The nonperiodic all-pair system and a periodic
network whose pairs remain inside the cutoff check conservation and momentum.
A third periodic chain deliberately crosses the hard cutoff. Its energy jump
is reproduced by both implementations and is recorded separately; this custom
expression does not smooth the descreening and polar terms at the cutoff.
The largest absolute energy changes were `7.34e-6` and `1.03e-6` kJ/mol in
the two noncrossing cases. The crossing case reached `11.220016` kJ/mol in both
implementations. Trajectory agreement is checked over the first 0.1 ps, and longer differences
are reported without treating them as a universal trajectory tolerance. These
small systems do not establish biomolecular stability or sampling convergence.

`scripts/benchmark-custom-gb.sh` measures 256, 1,024 and 4,096 particles for
all three methods and two expression models, using one warmup and three processes
per condition. The OBC model requests two global derivatives; the six-value
chain requests 24. Timed queries include force and energy evaluation, and the
fixture separately verifies the squared derivative norm. Construction is excluded
from timing; process peak RSS includes Context and runtime overhead. The supplied
Reference build uses Lepton JIT compilation, whereas Rust uses the interpreted
schedule described above. These measurements do not cover an optimized OpenMM
CPU platform, molecular integration or GPU execution.

A shared cell-grid regression was added after a partially filled lattice exposed
a rounding mismatch between grid-size division and cell-key multiplication.
Both calculations now use the same reciprocal-cutoff multiplication. The test
checks candidate completeness and duplicate absence against brute force,
including translated coordinates. This correction applies to every force using
the shared neighbor cells.

The OBC expressions are slower than the Reference JIT in these cases. The
six-value model with 24 requested derivatives is faster in Rust, whose reverse
chain avoids the reference's per-parameter propagation work. Peak process RSS
is lower in Rust for every measured condition. These results establish a
workload-dependent tradeoff, not a universal speed advantage.

| Model | Method | Particles | Rust evaluations/s | Reference evaluations/s | Rust peak RSS KiB | Reference peak RSS KiB |
|---|---|---:|---:|---:|---:|---:|
| OBC | NoCutoff | 256 | 15.2017 | 37.0416 | 4,440 | 8,464 |
| OBC | NoCutoff | 1,024 | 0.9364 | 2.4787 | 4,728 | 8,860 |
| OBC | NoCutoff | 4,096 | 0.0586 | 0.1544 | 5,812 | 9,828 |
| OBC | Nonperiodic cutoff | 256 | 74.9484 | 141.7591 | 4,684 | 8,716 |
| OBC | Nonperiodic cutoff | 1,024 | 14.8415 | 37.6712 | 4,856 | 9,108 |
| OBC | Nonperiodic cutoff | 4,096 | 3.2390 | 7.8700 | 5,756 | 11,456 |
| OBC | Periodic cutoff | 256 | 51.5322 | 91.0751 | 4,504 | 8,708 |
| OBC | Periodic cutoff | 1,024 | 10.8036 | 22.6375 | 4,748 | 8,916 |
| OBC | Periodic cutoff | 4,096 | 2.3739 | 4.8770 | 5,584 | 11,648 |
| Six-value chain | NoCutoff | 256 | 29.8594 | 7.4436 | 4,428 | 12,036 |
| Six-value chain | NoCutoff | 1,024 | 1.8609 | 0.5096 | 4,700 | 12,392 |
| Six-value chain | NoCutoff | 4,096 | 0.1157 | 0.0316 | 6,092 | 17,248 |
| Six-value chain | Nonperiodic cutoff | 256 | 132.7459 | 29.7904 | 4,488 | 12,436 |
| Six-value chain | Nonperiodic cutoff | 1,024 | 26.6716 | 7.6433 | 4,708 | 12,872 |
| Six-value chain | Nonperiodic cutoff | 4,096 | 5.7420 | 1.7850 | 6,044 | 18,440 |
| Six-value chain | Periodic cutoff | 256 | 88.8532 | 22.3346 | 4,520 | 12,308 |
| Six-value chain | Periodic cutoff | 1,024 | 17.7491 | 6.0278 | 4,808 | 12,952 |
| Six-value chain | Periodic cutoff | 4,096 | 3.8952 | 1.3900 | 5,912 | 18,644 |

## LCPO surface potential

`LCPOForce` implements the one-, two- and three-body overlap terms, surface
tension, zero-radius exclusion, optional periodic distances and parameter updates.
Radii include any solvent probe radius chosen by the caller; the force does not
add one. Surface tension has units kJ/mol/nm^2, and the fourth LCPO coefficient
has units nm^-2. A particle with zero coefficients and positive radius still
occludes other particles. The algebraic overlap formula is retained even for
fully contained spheres, matching Reference without additional clipping.

Each active contact stores its two directed overlap areas and radial derivatives.
A compact adjacency array identifies triangles; each unordered triangle is visited
once and contributes all six directed terms. Reverse area adjoints then produce
the pair forces. Storage scales with particles and actual contacts, without a
dense particle-pair matrix or a map per particle. Dense configurations can still
have quadratically many contacts. Warm host force accumulation reuses its buffers;
an owned force tensor allocates one output buffer. Noncoincident active particle
positions are required because the LCPO overlap formula contains inverse distance.

Parameter updates reload particle values, surface tension and the periodic flag,
matching the Reference kernel's complete reinitialization. Force group and name
remain those captured by the Context. The search cutoff is twice the largest
radius, while each actual contact must satisfy `r < radius_i + radius_j`.
The periodic box check follows the Reference evaluation-time tolerance.

Four Rust tests cover coordinate gradients, nonperiodic/triclinic geometry,
periodic images, containment, exact tangency, zero radius versus zero coefficients,
parameter updates and f32. A 4,096-particle allocation regression checks sparse
contact storage and warm force-buffer reuse. The reference fixture adds 768 states
covering four system sizes, positive/negative surface tension, zero coefficients
and radii, force groups, nested CVs, massless particles, virtual sites, updates,
box changes and short trajectories. Twelve configurations also compare forces
with 252 independent finite differences of source energies. Across 237,312
state comparisons, the largest absolute differences were `1.93e-9` kJ/mol in
energy and `2.48e-10` kJ/mol/nm in force. The largest finite-difference force
error was `8.73e-10` kJ/mol/nm.

The five-ps conservation fixture uses eight particles with harmonic bonds and a
0.2-fs step, in both nonperiodic and triclinic periodic conditions. Every pair
remains inside its overlap boundary. This condition matters because changes in
the overlap graph can switch three-body terms abruptly. The largest energy change
over 5 ps was `4.91e-6` kJ/mol in both
implementations; final position and velocity differences were below `1.09e-11`
in nm and nm/ps. These short model systems do not establish conservation for
arbitrary boundary crossings or biomolecular sampling convergence.

`scripts/benchmark-lcpo.sh` compares 512, 2,048 and 8,192 particles on a 0.3-nm
lattice, with radii from 0.20 to 0.28 nm, coefficients `(0.7,-0.2,0.08,0.1)` and
unit surface tension. Nonperiodic and periodic conditions use one warmup and
three processes each. Energy and squared force norm are checked across
implementations. Timings exclude construction, while process RSS includes Context
storage, reusable workspaces and runtime overhead. This is a surface-potential
benchmark, not complete force-field throughput or GPU execution.

| Geometry | Particles | Rust evaluations/s | Reference evaluations/s | Rust peak RSS KiB | Reference peak RSS KiB |
|---|---:|---:|---:|---:|---:|
| Nonperiodic | 512 | 707.007 | 179.084 | 3,756 | 7,500 |
| Nonperiodic | 2,048 | 155.305 | 36.842 | 5,328 | 9,928 |
| Nonperiodic | 8,192 | 38.542 | 8.574 | 11,516 | 20,268 |
| Periodic | 512 | 302.482 | 98.112 | 4,124 | 7,968 |
| Periodic | 2,048 | 74.067 | 25.452 | 5,480 | 10,632 |
| Periodic | 8,192 | 24.481 | 5.913 | 11,524 | 21,092 |

## Gay-Berne anisotropic potential

`GayBerneForce` supports spherical, axially symmetric and biaxial particles,
Lorentz-Berthelot sigma/epsilon mixing, explicit exceptions, three cutoff methods,
quintic switching, triclinic distances and nested collective variables. Particle
frames use the specified x direction and Gram-Schmidt projection of the y direction.
Orientation vectors use the original coordinates, matching the source; periodic
minimum images apply only to separations between interacting centers. Keep each
ellipsoid and its orientation particles in a consistent molecular image.

The core implements f64/f32 force evaluation and Python provides both naming
conventions for the force API. Context creation snapshots the definition. An
explicit parameter update reloads particle orientations, exceptions (including
changed pairs and additional exceptions), cutoff method and switching settings,
matching the supplied Reference reconstruction. The total particle count must
remain unchanged. Metadata and the existing Context's force group remain captured.

Each particle has two symmetric shape/energy matrices and a scalar shape factor.
The CPU workspace stores these frames and accumulated torques in linear memory;
finite cutoffs use reusable cell candidates. Spherical pairs use scalar formulas
without matrix inversion or torque evaluation. Pairs with isotropic energy-axis
scales also omit the energy-matrix inverse and its orientation derivative. A
4,096-particle allocation test
limits the largest workspace allocation to 256 bytes per particle, rejects dense
pair storage, and verifies one output-force allocation for the tensor API and no
new dense allocation for warmed host accumulation. This bound applies to individual
allocations, not the sum of every workspace buffer or process RSS.

### Approved conservative-force exceptions

The supplied Reference force has two orientation errors. Its derivative of the
energy-axis factor omits a factor depending on the pair direction. In addition,
the conversion of torques to forces uses the unprojected y-axis length and omits
the corresponding x-axis correction when the defining vectors are not orthogonal.
Rust differentiates the complete potential, including the Gram-Schmidt frame, and
implements only the conservative result. Source force values and trajectories
for general anisotropic particles are therefore intentional compatibility exceptions.
Reference energy finite differences are the force oracle. Spherical cases also
retain direct comparisons with the original Reference forces.

The Reference kernel applies the switching polynomial even with `NoCutoff`,
although its header says the switch is ignored. With sigma 0.3 nm, epsilon
1 kJ/mol, separation 1.5 nm, cutoff 1 nm and switch distance 0.8 nm, enabling
this flag changes the source energy from -0.000255983616 to 0.33997824 kJ/mol.
Rust ignores the switch with `NoCutoff`, as approved. Initial switch-parameter
validation still follows the source. The source Common GPU implementation also
guards switching by the cutoff mode in its code, but it has not been executed
as part of this validation.

Rust rejects zero-length or collinear orientation axes, nonfinite parameters and
coincident interacting centers. Zero-strength interactions return zero before
singular distance arithmetic. These are explicit input guards; they do not attempt
to reproduce nonfinite source values for degenerate geometries.

For spherical particles whose widths equal their individual sigma values, the
source header's claimed reduction to ordinary Lennard-Jones requires equal
sigmas. When they differ, the contact diameter is the root-mean-square diameter
and the shape factor also differs from one. Rust preserves this source energy;
it does not replace the potential by Lorentz-Berthelot Lennard-Jones. A permanent
two-sphere probe compares both expressions and asserts this distinction. For
sigmas 0.2 and 0.4 nm, epsilon 1 kJ/mol and separation 0.65 nm, the source
Gay-Berne energy is -0.03183537877 kJ/mol while ordinary Lennard-Jones gives
-0.03829032596 kJ/mol.

### Anisotropic validation

The reference fixture contains 576 states covering three methods, three particle
shapes, switching, direct/nested CV use, four update/coordinate phases and four
force-group queries. It includes a fixed particle, a virtual site, changed axis
particles, moved/added exceptions, periodic box changes and method changes within
an existing Context. The 12,096 independent coordinate derivatives use central
finite differences of the supplied source energies, including virtual-site force
redistribution. After the spherical and isotropic-energy optimizations, maximum energy and force
errors are 1.89e-15 kJ/mol and 1.60e-9 kJ/mol/nm. The original source forces differ
from those same derivatives by up to 0.852 kJ/mol/nm; that discrepancy is retained
in the generated report rather than hidden by a tolerance.

Two six-particle biaxial systems run 20,000 steps at 0.0001 ps with harmonic
orientation restraints, for 2 ps total, with and without periodic boundaries.
The maximum energy variation is 1.03e-8 kJ/mol and momentum error is 1.84e-15
Da nm/ps. All 200 sampled Rust configurations are evaluated again by the original
source energy oracle; their maximum energy difference is 1.44e-17 kJ/mol.
Source finite differences at the first and last samples additionally check the
forces. These short deterministic tests establish local force consistency and
integration stability for the specified systems, not equilibrium sampling of a
molecular liquid.

### Anisotropic performance

`scripts/benchmark-gay-berne.sh` compares f64 force/energy queries on a 0.45 nm
cubic lattice. Spherical particles vary in diameter, sigma and epsilon. Axial
systems use one orientation particle with zero epsilon per ellipsoid, so their
active-center count is half the particle count below. Both models have isotropic
energy-axis scales, for which the original Reference forces are conservative.
Cutoff methods use 0.9 nm and begin switching at 0.7 nm. The script checks energy
and squared force norm for every measured process.

Each condition has one warmup, 50/10/3 timed queries for 128/512/2,048 particles,
and three separate processes. The table gives median queries/s and the largest
process peak RSS across those runs. Construction is excluded from timing; RSS
includes runtime overhead, Context storage and returned force snapshots. These
are synthetic force queries against Reference, not optimized OpenMM CPU/GPU
platforms or complete MD throughput.

| Shape | Method | Particles | Rust queries/s | Reference queries/s | Rust peak KiB | Reference peak KiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Spherical | NoCutoff | 128 | 3224.526 | 771.521 | 3,752 | 7,200 |
| Spherical | NoCutoff | 512 | 198.901 | 48.486 | 3,984 | 7,440 |
| Spherical | NoCutoff | 2,048 | 12.724 | 2.997 | 4,460 | 7,904 |
| Spherical | Nonperiodic cutoff | 128 | 5312.437 | 3849.298 | 3,760 | 7,036 |
| Spherical | Nonperiodic cutoff | 512 | 1343.720 | 433.050 | 4,040 | 7,436 |
| Spherical | Nonperiodic cutoff | 2,048 | 249.429 | 37.524 | 4,476 | 7,940 |
| Spherical | Periodic cutoff | 128 | 3699.384 | 1888.996 | 3,784 | 7,076 |
| Spherical | Periodic cutoff | 512 | 419.647 | 169.084 | 3,960 | 7,212 |
| Spherical | Periodic cutoff | 2,048 | 106.395 | 12.507 | 4,612 | 8,072 |
| Axial | NoCutoff | 128 | 4297.021 | 3011.313 | 3,784 | 7,208 |
| Axial | NoCutoff | 512 | 278.271 | 189.597 | 3,976 | 7,252 |
| Axial | NoCutoff | 2,048 | 17.371 | 11.854 | 4,644 | 7,940 |
| Axial | Nonperiodic cutoff | 128 | 11076.977 | 9625.339 | 3,720 | 7,020 |
| Axial | Nonperiodic cutoff | 512 | 1391.831 | 1239.513 | 3,996 | 7,180 |
| Axial | Nonperiodic cutoff | 2,048 | 244.948 | 119.805 | 4,648 | 7,880 |
| Axial | Periodic cutoff | 128 | 6345.835 | 4375.223 | 3,744 | 7,012 |
| Axial | Periodic cutoff | 512 | 756.384 | 560.305 | 3,964 | 7,452 |
| Axial | Periodic cutoff | 2,048 | 131.008 | 45.116 | 4,476 | 7,840 |

The spherical path avoids per-pair matrix work. For finite cutoffs, candidate
cells also avoid Reference's scan of all particle pairs. NoCutoff remains
quadratic. The gains for small systems are sensitive to process startup, CPU
frequency and scheduling; the generated summaries retain each repetition.

## Programmable many-particle interactions

`CustomManyParticleForce` evaluates arbitrary set sizes with compiled coordinate,
distance, angle, dihedral and point-function expressions, global/per-particle
parameters and all six tabulated-function types. The CPU core supports f64/f32;
the Python wrapper provides snake_case and OpenMM-style force method names,
integer particle types, set-valued type filters, exclusions and live updates.
This class has no source API for requesting global energy derivatives. As in
Reference, a nested CV does not acquire derivatives of its child's global
parameters unless that child force provides them.

`SinglePermutation` evaluates each unordered set once. With a cutoff, every pair
in the set must be closer than the cutoff. `UniqueCentralParticle` fixes p1 as
the center and evaluates unordered choices of the remaining particles around it;
only distances from the center determine the cutoff. Exclusions apply to every
pair, including two noncentral particles. The energy must be symmetric under any
remaining allowed relabeling, as required by the source API. Type filters select
the first valid ordering in the source's recursive swap order.

The implementation preserves the supplied Reference's single-order filter
behavior: filters are bypassed for one-particle expressions in either mode, and
for two-particle expressions in `UniqueCentralParticle` mode. The original source
has only one possible ordering in these cases and skips its rejection lookup.
Permanent source probes with three particles of type 0, a filter allowing only
type 1 and constant energy 1 produce energies 3 for one-particle sets and 6 for
central pairs. Two-particle `SinglePermutation` and both three-particle modes
correctly produce zero. This discrepancy from the header is deliberately recorded;
it is not described as successful filtering.

### Captured parameters and table updates

Context creation captures the expression, particle types, filters, exclusions,
permutation mode and cutoff behavior. Ordinary updates copy numeric per-particle
values. A changed table update count reconstructs the supplied Reference's whole
interaction, so Rust also reloads the expression, particle types, filters,
exclusions, permutation mode, cutoff method and distance at that point. Existing
Context global values remain unchanged by modified definition defaults. Changed
table node counts and ranges are accepted, matching the actual Reference code.

The original Reference kernel separately captures whether it must supply periodic
box vectors and the cutoff used for its outer box check. Reconstructing the
interaction does not replace those captured fields. Rust preserves them. Thus an
already nonperiodic Context can reload `NoCutoff` or `CutoffNonPeriodic` settings,
and a periodic Context validates both captured and reloaded cutoff limits. Rust
rejects changes that would introduce periodic evaluation without a captured box,
or remove the cutoff from an already periodic interaction; the source reaches
uninitialized box data or an assertion for those inconsistent transitions.
Create a new Context to change that periodicity contract.

A periodic-to-nonperiodic cutoff reload retains periodic neighbor selection but
rebuilds geometric functions as nonperiodic. This mixed behavior also follows
the source. With a 3 nm box, particles at x=0 and x=2.8 nm, cutoff 0.9 nm and
energy `distance(p1,p2)+t(0)`, changing the method and raising the table from zero
to one changes the energy from 0.2 to 3.8 kJ/mol and the first x force from -1 to
+1 kJ/mol/nm. A permanent source probe records both results.

### Enumeration, expression evaluation and memory

No full set list is constructed. Cutoff pair interactions stream cell candidates
directly. Larger sets use a reusable sparse neighbor graph: single-permutation
sets are cliques, and central-particle sets are combinations of center neighbors.
Excluded pairs prune the search before expression evaluation. NoCutoff retains
combinatorial cost, but its enumeration needs only a particle-index vector and
scratch proportional to set size.

Type matching searches valid permutations as needed, without allocating all
permutations or the source's table of all type combinations. Per-particle values
use contiguous storage. Expression inputs are bound once; a scheduled reverse
pass computes coordinate derivatives, and global-only terms are prepared once
per force evaluation. Periodic point functions receive the current box, including
when their other inputs are constant.

The neighbor graph uses O(N+E) storage, where E is the number of accepted cutoff
pairs. Dense physical neighborhoods can still make E quadratic; the implementation
does not promise linear memory regardless of density. A 4,096-particle allocation
test assigns every particle a distinct type and evaluates three-body interactions
at fixed density. It rejects any single allocation larger than 512 bytes per
particle, checks one returned-force allocation after warmup, and verifies no new
dense allocation during warmed host accumulation. This is an allocation bound,
not the sum of all buffers or total process RSS.

### Many-particle numerical coverage

The initial reference fixture covers 1,728 states and 131,328 scalar comparisons:
set sizes one through four, both permutation modes, all three cutoff methods,
three filter patterns, direct/nested CV use, four update/trajectory phases and
three force-group queries. It includes a fixed particle, an out-of-plane virtual
site, arbitrary signed type codes, changed exclusions and type assignments,
changed table dimensions, changed expressions/modes and periodic image shifts.
Maximum errors are 9.10e-13 kJ/mol in potential energy, 2.28e-13 kJ/mol/nm in force,
8.89e-16 nm in positions and 5.38e-11 nm/ps in velocities. Another 3,024 independent
force components use finite differences of the original source energy; their
maximum error is 9.86e-7 kJ/mol/nm.

Rust tests independently distinguish neighbor stars from cliques, verify exclusion
of noncentral pairs, exercise type-selected roles and compare geometry forces with
energy gradients. They also cover f32, sparse bond exclusions, table-triggered
recapture, one-particle behavior, arbitrary set sizes and cutoff rounding. The
cutoff predicate follows Reference's square-root distance comparison; squaring the
cutoff instead can include a pair whose computed distance is exactly on the boundary.

Four eight-particle systems run 25,000 Verlet steps at 0.0002 ps, for 5 ps each,
covering both permutation modes with and without periodic cutoffs. Harmonic bonds
keep every pair below 0.255 nm, well inside the 0.9 nm cutoff. The largest total
energy variation is 4.40e-7 kJ/mol in both implementations. At 0.1 ps, position and
velocity differences are below 1.07e-13 nm and 1.41e-12 nm/ps; at 5 ps, the recorded
maxima are 2.04e-11 nm and 7.67e-12 nm/ps. Rust's final total-momentum error is below
4.02e-15 Da nm/ps, and the source error is below 2.66e-10 Da nm/ps. These restrained
synthetic systems check deterministic integration away from cutoff crossings;
they do not establish liquid sampling or long-time thermodynamic convergence.

### Many-particle performance

`scripts/benchmark-custom-many.sh` measures f64 three-particle force/energy queries
on a jittered 0.32 nm lattice. The first model uses a symmetric exponential of the
three pair distances; the second uses a central-particle angular expression.
Per-particle amplitudes vary by index and lambda is 0.7. Cutoff methods use 0.65 nm.
NoCutoff measures 32/64/128 particles; cutoff methods measure 128/256/512.

Each condition has one warmup and three processes. Timed query counts are 16/4/1
for NoCutoff and 10/3/1 for cutoffs, in increasing particle-count order. The table
gives median queries/s and the largest process peak RSS. Construction is excluded
from timing; RSS includes Context storage, force workspaces and runtime overhead.
Energy and squared force norm are checked for every measured process.

| Model | Method | Particles | Rust queries/s | Reference queries/s | Rust peak KiB | Reference peak KiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Symmetric triples | NoCutoff | 32 | 411.506 | 16.115 | 4,216 | 7,704 |
| Symmetric triples | NoCutoff | 64 | 70.792 | 1.956 | 4,268 | 7,548 |
| Symmetric triples | NoCutoff | 128 | 8.676 | 0.236 | 4,244 | 7,532 |
| Symmetric triples | Nonperiodic cutoff | 128 | 522.833 | 16.564 | 4,480 | 7,504 |
| Symmetric triples | Nonperiodic cutoff | 256 | 247.177 | 4.826 | 4,424 | 7,488 |
| Symmetric triples | Nonperiodic cutoff | 512 | 77.089 | 1.051 | 4,688 | 7,760 |
| Symmetric triples | Periodic cutoff | 128 | 337.917 | 9.301 | 4,476 | 7,780 |
| Symmetric triples | Periodic cutoff | 256 | 152.198 | 2.564 | 4,464 | 7,472 |
| Symmetric triples | Periodic cutoff | 512 | 58.406 | 0.498 | 4,680 | 7,952 |
| Central triples | NoCutoff | 32 | 144.049 | 2.305 | 4,304 | 7,936 |
| Central triples | NoCutoff | 64 | 17.339 | 0.274 | 4,232 | 7,984 |
| Central triples | NoCutoff | 128 | 2.123 | 0.033 | 4,440 | 7,940 |
| Central triples | Nonperiodic cutoff | 128 | 91.166 | 1.383 | 4,492 | 8,008 |
| Central triples | Nonperiodic cutoff | 256 | 34.859 | 0.490 | 4,528 | 8,016 |
| Central triples | Nonperiodic cutoff | 512 | 15.398 | 0.169 | 4,772 | 8,020 |
| Central triples | Periodic cutoff | 128 | 50.418 | 0.763 | 4,512 | 7,956 |
| Central triples | Periodic cutoff | 256 | 20.460 | 0.283 | 4,488 | 7,948 |
| Central triples | Periodic cutoff | 512 | 8.138 | 0.086 | 4,720 | 7,992 |

Reference enumerates particle triples even with a cutoff. Rust prunes the search
using sparse neighbor cliques or center-neighbor combinations and reuses its
compiled derivative workspace. NoCutoff still has cubic cost for these models.
The generated summaries retain individual repetitions and timing ranges, including
variation from scheduling and CPU frequency. These are synthetic Reference-platform
force queries; optimized OpenMM CPU/GPU platforms and complete MD throughput are
outside this measurement.

## Constant-potential electrostatics

`ConstantPotentialForce` implements periodic PME electrostatics with fixed solute
charges and Gaussian electrode charges. The electrode charges minimize the
quadratic energy at prescribed potentials, with optional Thomas-Fermi terms,
a uniform external field and a constraint on the total system charge. This
force does not include Lennard-Jones interactions. A separate force must provide
them when needed. The core supports f64/f32 output and the Python binding exposes
both naming conventions, electrode sets, exceptions, charge queries and actual
Context PME parameters.

The direct electrostatic kernel combines the Ewald and Gaussian charge widths.
Reciprocal PME, Ewald self energy, neutralizing plasma, Gaussian self energy,
electrode potentials, Thomas-Fermi energy and the external field contribute to
the same minimized energy. Cartesian forces use the charge solution held fixed;
stationarity of the charge solution supplies the corresponding total derivative.
External-field energy uses the stored, unwrapped coordinates. The field is in
kJ/mol/nm per proton charge and electrode potentials are in kJ/mol per proton
charge. The total-charge constraint includes fixed solute charges.

### Charge solvers and storage

The CG solver retains its charge guess and reuses scalar work vectors. It can use
the source diagonal preconditioner, including the PME, self, plasma, Gaussian and
Thomas-Fermi diagonal contributions. Constraint projection and drift removal
follow the Reference iteration. The source has two stopping checks: the projected
gradient norm and a small matrix-vector step. Rust retains both, including the
iteration limit. The tolerance therefore controls both tests; the header only
describes the gradient test.

The matrix solver stores an in-place packed Cholesky factor with
`K*(K+1)/2` doubles for K electrode particles. It reuses that factor until the
electrode parameters, electrode positions or box change. Applying the total-charge
constraint uses a precomputed vector proportional to the inverse matrix times
an all-ones vector. Electrode masses must be zero for this solver. Manual position
and box changes rebuild the matrix, matching the source kernel and its tests.

Both solvers share reusable PME transforms and FFT workspace. Charge derivatives
interpolate the reciprocal potential directly, including at zero-charge particles;
no force evaluation or finite charge difference is needed. Charge arrays contain
only one scalar per particle, without unused Lennard-Jones fields. Finite-cutoff
cell traversal computes fixed-charge interactions immediately and stores energy
and force coefficients only for pairs involving an electrode. Charge solving and
final force evaluation reuse those coefficients, avoiding a second cell traversal
and repeated screening-function evaluations. Charge-only queries omit fixed-charge
interactions and force coefficients. CG storage is linear in particles plus these pairs and the PME grid. Dense electrode
clusters can still have quadratically many neighbor pairs. Only the matrix method
allocates a dense electrode factor.

The allocation regression uses 512 electrode particles for CG and 256 for the
matrix solver. It bounds matrix storage by the packed triangular size and rejects
a dense electrode matrix in the CG workspace. After warmup, the host accumulation
path makes no new dense allocation; the tensor API allocates its returned force
buffer. The bounds concern allocations inside these fixtures, not process RSS or
all possible electrode densities.

### Captured state and parameter updates

Context creation captures the method, cutoff, PME alpha/grid, CG tolerance,
preconditioner selection, charge-constraint enable flag and exception periodicity.
Changing these settings on the definition requires a new Context. Explicit updates
copy fixed charges, active exception parameters, electrode parameters, the external
field and the total-charge target. Updated input charges on electrode particles
are ignored, preserving the current iterative guess. Charge queries solve for the
current positions and box independently of force-group selection.

The supplied kernel permits regrouping electrode particles, including a changed
number of groups, as long as their union stays fixed. It also accepts changes to
active exception endpoints while retaining the original exclusion graph. An
initially nonzero exception may become zero and return to a nonzero value; an
initially zero exception cannot become nonzero. Adding a zero-valued exception
through an update does not add an exclusion to the captured graph. Rust follows
these measured implementation boundaries. Reinitialize for a consistently changed
exclusion topology. Thirty source/Rust update states retain these behaviors as
explicit probes rather than treating the header's stricter description as the
kernel behavior.

### Constant-potential validation

The main fixture compares 1,152 states and 100,224 scalar values across three
solver configurations, constrained/unconstrained charge, one/four electrode
particles, orthorhombic/triclinic cells, periodic/nonperiodic exceptions, direct
or nested CV use, four update/trajectory phases and three force-group selections.
It includes virtual-site force redistribution, a distance constraint, fixed
particles, explicit parameter changes, changed periodic images, manual electrode
movement and captured PME settings. At a CG tolerance of 1e-8 kJ/mol/e, maximum
energy, force and charge differences are 1.07e-14 kJ/mol, 8.96e-10 kJ/mol/nm and
1.20e-11 e. The actual PME parameters match exactly. A tighter 1e-11 source run
failed to converge in one condition; the complete comparison uses the same 1e-8
tolerance in both implementations.

Another 2,016 Cartesian components are checked against finite differences of the
source energy, with a maximum error of 3.34e-9 kJ/mol/nm. Independent Rust tests
check reciprocal charge derivatives, agreement with fixed-charge PME Coulomb
forces, CG/matrix agreement, total charge, relaxed energy gradients, update
invalidation and f32 output. Regression comparisons retain 41 nonbonded cases,
360 CV states in two cache modes, and force-group cases after the shared PME
refactoring.

Six restrained six-particle systems run 5,000 Verlet steps at 0.0004 ps, for 2 ps
each: three solver configurations with and without the total-charge constraint.
The maximum total-energy variation is 2.40e-6 kJ/mol in both implementations.
Final positions and velocities differ by at most 7.17e-12 nm and
1.30e-11 nm/ps. All pair distances stay below 1.323 nm, within the 1.4 nm cutoff,
and constrained total charge agrees to the displayed precision. These small
systems test deterministic propagation and charge equilibration, not electrode
force-field preparation, bulk sampling convergence or resident GPU execution.

### Constant-potential performance

`scripts/benchmark-constant-potential.sh` measures coordinate updates followed by
force/energy queries on a jittered 0.32 nm lattice. One quarter of particles form
two fixed electrodes with potentials +2/-2 kJ/mol/e, Gaussian widths 0.08/0.10 nm
and Thomas-Fermi scales 0.005/nm. The total charge is constrained to 0.25 e and
the field is `(0.1,-0.05,0.08)` kJ/mol/nm/e. The cutoff is 0.8 nm and alpha is
3/nm; the cubic PME mesh is the next power of two above six times the box side,
with a minimum of 16. Solute coordinates change smoothly on every timed query,
requiring a new charge solution while keeping the matrix geometry reusable.

Each condition has one warmup, 200/40/10 timed queries for 128/512/2,048 particles
and three processes. The final energy, squared force norm, squared charge norm
and total charge are verified for every process. Setup through warmup and the
first force query are timed separately. Peak RSS includes construction and
workspace initialization. This is a synthetic CPU comparison with Reference;
optimized OpenMM CPU/GPU platforms and molecular integration throughput are not
measured here.

| Solver | Particles | Rust queries/s | Reference queries/s | Speedup | Rust RSS (KiB) | Reference RSS (KiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CG | 128 | 213.65 | 50.70 | 4.21 | 4,680 | 7,712 |
| CG | 512 | 60.32 | 15.45 | 3.90 | 5,192 | 8,000 |
| CG | 2,048 | 10.17 | 4.08 | 2.50 | 6,384 | 9,564 |
| Preconditioned CG | 128 | 223.08 | 53.67 | 4.16 | 4,476 | 7,940 |
| Preconditioned CG | 512 | 62.39 | 16.12 | 3.87 | 5,224 | 8,008 |
| Preconditioned CG | 2,048 | 9.97 | 3.98 | 2.51 | 6,360 | 9,612 |
| Matrix | 128 | 750.40 | 295.77 | 2.54 | 4,592 | 7,752 |
| Matrix | 512 | 137.13 | 83.56 | 1.64 | 5,280 | 8,128 |
| Matrix | 2,048 | 34.60 | 23.18 | 1.49 | 7,492 | 13,528 |

First-query and setup medians in milliseconds:

| Solver | Particles | Rust first | Reference first | Rust setup | Reference setup |
| --- | ---: | ---: | ---: | ---: | ---: |
| CG | 128 | 6.84 | 20.81 | 7.93 | 22.02 |
| CG | 512 | 22.76 | 74.97 | 23.05 | 76.05 |
| CG | 2,048 | 134.24 | 321.83 | 134.83 | 323.79 |
| Preconditioned CG | 128 | 6.61 | 20.59 | 6.71 | 21.73 |
| Preconditioned CG | 512 | 18.95 | 73.29 | 19.18 | 74.66 |
| Preconditioned CG | 2,048 | 135.41 | 339.23 | 136.25 | 340.79 |
| Matrix | 128 | 12.35 | 35.40 | 12.51 | 36.55 |
| Matrix | 512 | 68.08 | 386.14 | 68.41 | 387.39 |
| Matrix | 2,048 | 1414.94 | 4378.37 | 1416.12 | 4380.25 |

Rust is 1.49-4.21 times faster in these conditions and uses less
process peak RSS. The matrix result benefits from fixed electrode geometry; its
first query includes matrix construction and factorization. These timings do not
cover moving electrodes or large dense electrode clusters.

## Documentation discrepancies found during implementation

- All four Monte Carlo barostat constructor comments call the rigid scaling center a
  center of mass. The Reference kernel instead uses the unweighted centroid,
  consistent with their getter/setter documentation. Rust follows the code.
- The membrane barostat surface-tension setter describes its argument as bar,
  although its getter, constructor and area-work calculation use bar nm. Rust
  uses bar nm for both getter and setter.

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

- `CompoundIntegrator.h` calls the collection a `CustomIntegrator` in its
  introduction, and the source's add-after-binding error repeats that name.
  Both refer to `CompoundIntegrator`. The header's requirement to add children
  before Context creation is not enforced by the supplied initialization code;
  the Rust ownership boundary above enforces it explicitly.

- `NoseHooverIntegrator.h` describes a collision-frequency setter argument as
  picoseconds although the getter, constructor and propagation use inverse ps.
- `NoseHooverChain.h` says changes to temperatures, collision frequencies and
  degrees of freedom only affect new Contexts. The integrator setters modify the
  same chains that its kernels read every step; these scalar changes are live.
- Nose-Hoover constructor diagnostics require a positive MTS count but the source
  check only rejects negative values, allowing zero. Rust enforces positivity.
- The Nose-Hoover pair documentation promises independent COM and relative
  thermostatting, but the source absolute-chain branch requires an individual
  atom and the low-absolute-energy guard also disables the relative chain.
  Its kinetic-energy API omits relative and unthermostated motion. The approved
  exceptions above correct all three implementation defects.

- `QTBIntegrator.h` says step size is fixed at Context creation, which matches
  its propagation kernel, but the supplied Reference clock and thermal velocity
  offset read the later requested value. The approved exception uses the captured
  physical step size consistently. Its documented adapted-friction grid also
  omits the even-length Nyquist bin and uses a frequency spacing inconsistent
  with odd-length FFTs; the corrected API exposes every actual FFT bin.

- `CustomIntegrator.h` describes access to numbered force groups and energies
  from per-DOF expressions, but Reference omits these bindings for vector
  expressions. Its clock also ignores later `dt` changes, and its derivative
  cache can return a different group's derivative. The exceptions above preserve
  the requested variable and time semantics and the derivative of the named energy.
- The custom-integrator analyzer rejects `fN` plus `energyN` in one computation
  despite the header only prohibiting dependence on multiple force groups.
  Rust preserves that rejection and records the restriction explicitly.

- `CustomGBForce.h` says parameter updates cannot change energy expressions and
  limit table changes to values at existing nodes. Reference rebuilds its
  expressions whenever a table update count changes, including changed formulas,
  table dimensions, bounds and names. Rust follows the observed code behavior.
- Its deprecated function comments refer to `getTabulatedFunctionParameters()`
  and `setTabulatedFunctionParameters()`, which are not members of the class.
  The actual object API is `getTabulatedFunction().get/setFunctionParameters()`.

- `LCPOForce.h` limits Context updates to surface tension and particle parameters.
  The supplied Reference kernel reinitializes its full interaction and also
  reloads the periodic-boundary flag. Rust follows that implementation.

- `GayBerneForce.h` restricts Context updates to numeric parameters and excludes
  changed orientation atoms, exception pairs/counts, methods and cutoffs. The
  Reference kernel reconstructs the full interaction and accepts those changes.
  Rust follows the implementation while retaining a fixed System particle count.
- The same header says `NoCutoff` ignores switching; the supplied Reference kernel
  applies it regardless of mode. Rust uses the approved mode guard described above.
- Its spherical Lennard-Jones equivalence claim omits the equal-sigma condition.
  Unequal diameters retain a different contact distance and shape prefactor, which
  Rust preserves and checks with the source energy probe.

- `CustomManyParticleForce.h` describes type filtering without the Reference
  implementation's single-order bypass. One-particle sets and two-particle
  central interactions ignore those filters; Rust preserves the source behavior.
- Its update description says expressions, exclusions, modes and cutoffs remain
  unchanged, and restricts tables to their old nodes/ranges. A changed table update
  count actually reconstructs the interaction, including those definitions and
  particle types. Only the kernel's outer periodic flag and box-check cutoff stay
  captured. Rust follows those measured update boundaries and rejects inconsistent
  periodic transitions that are undefined in the source.
- The nonbonded-method enum comments describe an all-pairs cutoff without stating
  the central-mode exception. The detailed class description and implementation
  apply center distances only in `UniqueCentralParticle` mode.

- `ConstantPotentialForce.h` restricts the matrix method to unchanged electrode
  positions and box vectors. The kernel instead requires zero electrode masses
  and rebuilds its factor after manual position or box changes; its own tests
  exercise both changes.
- Its update description omits the external field, which the implementation
  copies on every update. The header also prohibits electrode regrouping and
  changed exception endpoints, while the kernel accepts the update cases listed
  above and retains its original exclusion graph.
- The CG tolerance comment describes only projected-gradient convergence. The
  implementation also stops for a sufficiently small matrix-vector step; Rust
  preserves that additional stopping condition.
- The `setChargeConstraintTarget()` comment says "Get" although the method sets
  the target. The getter and setter use the same total-system-charge convention.

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

## Barostat compatibility exceptions

Rigid versus atomic scaling is captured at Context creation in Rust, for both
coordinate changes and the Jacobian particle count. In the supplied Reference
snapshot, changing this setting afterward changes only the particle count; the
coordinate kernel keeps its original mode. Rust retains a consistent captured mode.
Create a new Context to apply a changed mode.

Nonpositive proposed volumes are rejected before computing a box scale. The
supplied isotropic Reference implementation instead evaluates a fractional power of a
negative volume ratio, producing nonfinite box vectors and an exception. A seeded
32-atom, 16-molecule ideal-gas probe reproduced this after 35,111 completed steps: the current
volume was about 12.27 nm^3 and the proposal half-width about 13.69 nm^3. The regular
statistical fixtures use 128 atoms. A separate Rust regression tests rejection of
nonpositive proposals after a large external reduction of box volume.

The supplied anisotropic and membrane pressure APIs scale a whole box vector when probing an
axis but scale coordinates by Cartesian component. Those operations coincide
for orthorhombic cells and differ for triclinic cells. Rust instead scales each
Cartesian column of the box, matching the actual Monte Carlo proposals. For the
energy `E = bx`, box vectors `(3,0,0)`, `(0.4,3.3,0)`, `(-0.2,0.3,3.5)` and zero
velocities, the source reports `(0,-0.1916928,0)` bar. The Cartesian derivative,
returned by Rust, is `(-0.1916928,0,0)` bar. The fixture retains source API values
and separately evaluates source-energy finite differences, so this exception is
explicit rather than absorbed into a numerical tolerance.
