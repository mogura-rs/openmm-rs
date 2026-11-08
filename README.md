# openmm-rs
Rust reimplementation of OpenMM, with CPU and GPU backend support in progress.

This project is not yet an OpenMM-compatible replacement. See
[the compatibility checklist](docs/compatibility.md) for coverage, validation,
performance targets and known limitations.

```sh
cargo test --workspace --locked
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

Simulation values use OpenMM units: nm, ps, Da, kJ/mol and elementary charge.

Reference validation requires CMake, a C++17 compiler, `uv`, and a separate
OpenMM source checkout. The default source location is the sibling directory
`../openmm`; override it with `OPENMM_SOURCE`.

```sh
bash scripts/validate-reference.sh
bash scripts/benchmark-cpu.sh
```

GPU validation must use Slurm with at most two nodes active for this project.
The current numerical and performance evidence covers CPU execution. See the
compatibility document for the fractional `SymmetrySite` force exception.
