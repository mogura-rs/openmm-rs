# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`openmm-rs` is a pure-Rust implementation of [OpenMM](https://openmm.org/), a molecular dynamics simulation toolkit. Use a separate OpenMM checkout as the reference for API design and numerical correctness. Set `OPENMM_SOURCE` to select it; the default is the sibling directory `../openmm`. The reference is not a submodule.

The full compatibility and performance target is tracked in
`docs/compatibility.md`. The implementation remains incomplete.

## Key Crates

| Crate | Purpose |
|---|---|
| `burn` | Backend tensors for CPU and GPU feature configurations |
| `rustfft` | 3D FFT for Particle Mesh Ewald (PME) electrostatics |
| `nalgebra` | Coordinate transforms and small matrix ops outside burn tensors |
| `rayon` | CPU parallelism outside of burn tensor operations |
| `uom` | Physical unit type safety (nm, kJ/mol, Da, etc.) |
| `mdtraj-rs` | Planned trajectory file I/O; the dependency is currently disabled |
| `approx` | Floating-point comparison in tests |
| `pyo3` | Python bindings (thin layer only) |

## Workspace Layout

```
openmm-rs/
├── Cargo.toml                  # workspace root
└── crates/
    ├── openmm-core/            # all physics: forces, integrators, context
    │   └── src/
    │       ├── lib.rs
    │       ├── system.rs       # System<B>
    │       ├── context.rs      # Context<B> — MD loop driver
    │       ├── state.rs        # State<B> — immutable snapshot
    │       ├── units.rs        # uom type aliases
    │       ├── forces/         # Force<B> trait + implementations
    │       └── integrators/    # Integrator<B> trait + implementations
    └── openmm-py/              # pyo3 thin wrappers (NdArray<f64> fixed)
```

## Build & Development Commands

```bash
# Build (CPU/NdArray backend, default)
cargo build

# Compile the WGPU backend; this alone does not validate GPU force execution
cargo build -p openmm-core --features wgpu --no-default-features

# Release build
cargo build --release

# Run all tests
cargo test

# Run a single test by name
cargo test test_bonds

# Run tests in a specific module
cargo test -p openmm-core forces::harmonic_bond

# Lint
cargo clippy -- -D warnings

# Format
cargo fmt
```

## Architecture

### Backend Generics

All core types are generic over `burn::prelude::Backend`:

```rust
struct System<B: Backend> { ... }      // holds Box<dyn Force<B>>
struct Context<B: Backend> { ... }     // owns positions/velocities tensors
struct State<B: Backend> { ... }       // immutable snapshot
```

The Python binding layer (`openmm-py`) fixes the backend to `NdArray<f64>` — this is the only place where `B` is concrete. The CPU core also supports `NdArray<f32>`. Use `openmm_core::tensor::from_floats` to preserve the selected backend precision when constructing tensors. New backends (WGPU, CUDA) are enabled via Cargo features in `openmm-core`:

```toml
# openmm-core/Cargo.toml features
ndarray = ["burn/ndarray", "dep:ndarray"]   # default CPU
wgpu    = ["burn/wgpu"]
cuda    = ["burn/cuda"]
```

### Core Layer Hierarchy (mirrors OpenMM design)

```
System<B>       - holds particles (mass), box vectors, Box<dyn Force<B>>
Integrator<B>   - time-stepping (Verlet, Langevin, Brownian, …)
Context<B>      - owns System + Integrator + positions + velocities tensors
  └─ State<B>   - immutable snapshot (positions, velocities, forces, energy)
```

### Abbreviated Force Trait

The actual trait also provides cloning, downcasting and parameter methods.

```rust
pub trait Force<B: Backend>: Send + Sync {
    fn calc_forces_and_energy(
        &self,
        positions: &Tensor<B, 2>,           // [N, 3] nm
        box_vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64);               // ([N, 3] kJ/mol/nm, kJ/mol)

    fn uses_periodic_boundary_conditions(&self) -> bool { false }
}
```

Use analytic gradients for geometric force terms. Custom scalar expressions use the compiled expression graph and its reverse-mode derivatives; reuse that evaluator when extending custom forces.

### Python Binding Strategy

- All simulation logic lives in `openmm-core`
- `openmm-py` is a thin `pyo3` translation layer with no business logic
- The concrete backend is `NdArray<f64>`, selected in `openmm-py/src/lib.rs`
- Expose the same top-level API as OpenMM Python: `System`, `Context`, `State`, `*Force`, `*Integrator`

## Reference Tests

OpenMM test headers are in `$OPENMM_SOURCE/tests/Test*.h`. Each test constructs a `System`, adds particles, adds a `Force`, creates a `Context`, sets positions, and asserts energy/forces using `ASSERT_EQUAL_TOL` (tolerance ~1e-5). Paths prefixed with `openmm/` below refer to that external checkout.

Replicate the same numerical assertions in Rust tests using `approx::assert_relative_eq!`. Tests live in `#[cfg(test)]` modules within each source file. Mapping:

```
openmm/tests/TestHarmonicBondForce.h     → forces/harmonic_bond.rs  (implemented)
openmm/tests/TestHarmonicAngleForce.h    → forces/harmonic_angle.rs  (implemented)
openmm/tests/TestPeriodicTorsionForce.h  → forces/periodic_torsion.rs  (implemented)
openmm/tests/TestNonbondedForce.h        → forces/nonbonded.rs  (partial coverage)
openmm/tests/TestLangevinMiddleIntegrator.h → integrators/langevin.rs  (implemented)
```

## Implementation Status & Priorities

### Implemented Subsets
- `HarmonicBondForce` — analytic forces + PBC
- `HarmonicAngleForce` — analytic forces + PBC
- `PeriodicTorsionForce` — Blondel-Karplus formula + PBC
- `VerletIntegrator` — leapfrog
- `LangevinMiddleIntegrator` — LFMiddle splitting
- `BrownianIntegrator` — overdamped Langevin
- CPU stochastic integration — direct f64 state updates and per-integrator seeded random streams
- `NonbondedForce` — cutoff/reaction field, Ewald, PME, LJPME, parameter offsets, switching and dispersion correction
- Virtual sites — averages, out-of-plane, local coordinates, symmetry and nested dependencies
- Constraints — sparse SHAKE/RATTLE position and velocity projection
- Custom bond, angle, torsion, external, nonbonded, compound, centroid, hydrogen-bond and volume forces — compiled expressions, tabulated/geometric functions and analytic derivatives
- RMSD, radius of gyration and orientation restraint — quaternion alignment and analytic forces
- CustomCVForce — nested force composition and analytic chain-rule derivatives
- CMMotionRemover and AndersenThermostat — ordered velocity updates, constraint-group collisions and runtime thermostat parameters
- MonteCarloBarostat and MonteCarloAnisotropicBarostat — isotropic or axis-selected volume moves, molecular/atomic scaling, live frequency and instantaneous pressure
- Python wrappers — partial snake_case CPU interface with retained force definitions, Context snapshots, single-owner checks and shared runtime integrator settings

### Next
1. Complete the compatibility checklist in `docs/compatibility.md`.
2. Complete pressure control, custom expressions and numerical coverage.
3. Extend custom expressions/forces, integrators and Python API compatibility.
4. Implement resident GPU force evaluation and validate through Slurm.
5. Measure and optimize throughput and peak host/device memory without weakening numerical validation.

## Important Conventions

- Units follow OpenMM defaults: nm, kJ/mol, Da, ps, e (elementary charge)
- Public simulation methods currently accept scalar values in OpenMM units; `uom` aliases do not yet enforce units at those boundaries
- Periodic boundary conditions: use minimum image convention for all pairwise distances
- PME cutoff and Ewald parameters live in `NonbondedForce`, not in `Context`
- Reference implementation is `openmm/platforms/reference/src/SimTKReference/`; use it as the algorithmic ground truth
