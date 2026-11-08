#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
reference_source="${OPENMM_SOURCE:-../openmm}"
reference_build=target/reference-openmm
export CARGO_TARGET_DIR=target/benchmark-rust
jobs="${SLURM_CPUS_PER_TASK:-4}"
cargo build --release --locked -p openmm-core --example cv_bench --jobs "$jobs"
"${CXX:-c++}" -std=c++17 -O3 tests/reference/cv_bench.cpp \
    -I "$reference_source/openmmapi/include" -I "$reference_source/olla/include" \
    -I "$reference_source/serialization/include" \
    -L "$reference_build" -Wl,-rpath,"$(pwd)/$reference_build" -lOpenMM \
    -o target/reference-cv-bench
time_command="$(type -P time)"
time_flags=(-v)
if [[ "$(uname -s)" == Darwin ]]; then time_flags=(-l); fi
for variables in 2 8 32; do
    directory="benchmark-results/cv/${variables}"
    mkdir -p "$directory"
    for particles in 4096 65536; do
        iterations=5000
        if [[ "$particles" == 65536 ]]; then iterations=300; fi
        for repetition in 1 2 3; do
            for mode in stream bounded all; do
                "$time_command" "${time_flags[@]}" \
                    "$CARGO_TARGET_DIR/release/examples/cv_bench" "$particles" "$variables" "$iterations" "$mode" \
                    > "$directory/rust-${particles}-${mode}-${repetition}.json" \
                    2> "$directory/rust-${particles}-${mode}-${repetition}.memory.txt"
            done
            "$time_command" "${time_flags[@]}" \
                target/reference-cv-bench "$particles" "$variables" "$iterations" \
                > "$directory/reference-${particles}-${repetition}.json" \
                2> "$directory/reference-${particles}-${repetition}.memory.txt"
        done
    done
    uv run --no-project python scripts/summarize-benchmarks.py "$directory"
done
