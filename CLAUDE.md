# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`openmm-rs` is a pure-Rust implementation of [OpenMM](https://openmm.org/), a molecular dynamics simulation toolkit. The original OpenMM source is included as a git submodule at `./openmm/` and serves as the reference for API design and test correctness.

## Key Crates

| Crate | Purpose |
|---|---|
| `burn` | Tensor operations and automatic differentiation (force = -∇E) |
| `rustfft` | 3D FFT for Particle Mesh Ewald (PME) electrostatics |
| `nalgebra` | Coordinate transforms and small matrix ops outside burn tensors |
| `rayon` | CPU parallelism outside of burn tensor operations |
| `uom` | Physical unit type safety (nm, kJ/mol, Da, etc.) |
| `mdtraj-rs` | Trajectory file I/O (PDB, DCD, XTC) |
| `approx` | Floating-point comparison in tests |
| `pyo3` | Python bindings (thin layer only) |

## Workspace Layout

```
openmm-rs/
├── Cargo.toml                  # workspace root
├── openmm/                     # git submodule — reference implementation & tests
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
    └── openmm-py/              # pyo3 thin wrappers (NdArrayBackend fixed)
```

## Build & Development Commands

```bash
# Build (CPU/NdArray backend, default)
cargo build

# Build with WGPU (GPU) backend
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

The Python binding layer (`openmm-py`) fixes the backend to `NdArrayBackend` — this is the only place where `B` is concrete. New backends (WGPU, CUDA) are enabled via Cargo features in `openmm-core`:

```toml
# openmm-core/Cargo.toml features
ndarray = ["burn/ndarray"]   # default CPU
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

### Force Trait

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

Use analytic gradients for simple forces (HarmonicBond, HarmonicAngle). Use `burn` autodiff for complex ones (CustomGBForce, etc.).

### Python Binding Strategy

- All simulation logic lives in `openmm-core`
- `openmm-py` is a thin `pyo3` translation layer with no business logic
- Backend is fixed to `NdArrayBackend` in `openmm-py/Cargo.toml`
- Expose the same top-level API as OpenMM Python: `System`, `Context`, `State`, `*Force`, `*Integrator`

## Reference Tests

OpenMM test headers are in `openmm/tests/Test*.h`. Each test constructs a `System`, adds particles, adds a `Force`, creates a `Context`, sets positions, and asserts energy/forces using `ASSERT_EQUAL_TOL` (tolerance ~1e-5).

Replicate the same numerical assertions in Rust tests using `approx::assert_relative_eq!`. Tests live in `#[cfg(test)]` modules within each source file. Mapping:

```
openmm/tests/TestHarmonicBondForce.h     → forces/harmonic_bond.rs  (implemented)
openmm/tests/TestHarmonicAngleForce.h    → forces/harmonic_angle.rs  (implemented)
openmm/tests/TestPeriodicTorsionForce.h  → forces/periodic_torsion.rs  (implemented)
openmm/tests/TestNonbondedForce.h        → forces/nonbonded.rs  (TODO)
openmm/tests/TestLangevinMiddleIntegrator.h → integrators/langevin.rs  (implemented)
```

## Implementation Status & Priorities

### Done
- `HarmonicBondForce` — analytic forces + PBC
- `HarmonicAngleForce` — analytic forces + PBC
- `PeriodicTorsionForce` — Blondel-Karplus formula + PBC
- `VerletIntegrator` — leapfrog
- `LangevinMiddleIntegrator` — BAOAB splitting
- `BrownianIntegrator` — overdamped Langevin

### Next
1. `NonbondedForce` — cutoff direct space, then PME reciprocal (`rustfft`)
2. Constraints — CCMA (`openmm/platforms/reference/src/SimTKReference/ReferenceCCMAAlgorithm.cpp`)
3. Custom\*Force — expression parser + evaluator
4. `MonteCarloBarostat`
5. Python bindings (`openmm-py`)

## Important Conventions

- Units follow OpenMM defaults: nm, kJ/mol, Da, ps, e (elementary charge)
- `uom` types enforce unit correctness at compile time; convert at system boundaries
- Periodic boundary conditions: use minimum image convention for all pairwise distances
- PME cutoff and Ewald parameters live in `NonbondedForce`, not in `Context`
- Reference implementation is `openmm/platforms/reference/src/SimTKReference/`; use it as the algorithmic ground truth
