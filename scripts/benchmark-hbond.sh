#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
reference_source="${OPENMM_SOURCE:-../openmm}"
reference_build=target/reference-openmm
export CARGO_TARGET_DIR=target/benchmark-rust
jobs="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p benchmark-results/hbond
cargo build --release --locked -p openmm-core --example hbond_bench --jobs "$jobs"
"${CXX:-c++}" -std=c++17 -O3 tests/reference/hbond_bench.cpp \
    -I "$reference_source/openmmapi/include" -I "$reference_source/olla/include" \
    -I "$reference_source/serialization/include" \
    -L "$reference_build" -Wl,-rpath,"$(pwd)/$reference_build" -lOpenMM \
    -o target/reference-hbond-bench

time_command="$(type -P time)"
time_flags=(-v)
if [[ "$(uname -s)" == Darwin ]]; then
    time_flags=(-l)
fi
for side in 8 16 24; do
    for traversal in exhaustive cells; do
        "$time_command" "${time_flags[@]}" \
            "$CARGO_TARGET_DIR/release/examples/hbond_bench" "$side" 10 "$traversal" \
            > "benchmark-results/hbond/rust-${side}-${traversal}.json" \
            2> "benchmark-results/hbond/rust-${side}-${traversal}.memory.txt"
    done
    "$time_command" "${time_flags[@]}" \
        target/reference-hbond-bench "$side" 10 \
        > "benchmark-results/hbond/reference-${side}.json" \
        2> "benchmark-results/hbond/reference-${side}.memory.txt"
done
uv run --no-project python scripts/summarize-benchmarks.py benchmark-results/hbond
