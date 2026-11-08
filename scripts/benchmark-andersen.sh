#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
reference_source="${OPENMM_SOURCE:-../openmm}"
reference_build=target/reference-openmm
export CARGO_TARGET_DIR=target/benchmark-rust
jobs="${SLURM_CPUS_PER_TASK:-4}"
cargo build --release --locked -p openmm-core --example andersen_bench --jobs "$jobs"
"${CXX:-c++}" -std=c++17 -O3 tests/reference/andersen_bench.cpp \
    -I "$reference_source/openmmapi/include" -I "$reference_source/olla/include" \
    -I "$reference_source/serialization/include" \
    -L "$reference_build" -Wl,-rpath,"$(pwd)/$reference_build" -lOpenMM \
    -o target/reference-andersen-bench
time_command="$(type -P time)"
time_flags=(-v)
if [[ "$(uname -s)" == Darwin ]]; then time_flags=(-l); fi
directory=benchmark-results/andersen
mkdir -p "$directory"
for particles in 4096 65536; do
    iterations=1000
    if [[ "$particles" == 65536 ]]; then iterations=300; fi
    for repetition in 1 2 3; do
        order=(rust reference)
        if [[ "$repetition" == 2 ]]; then order=(reference rust); fi
        for implementation in "${order[@]}"; do
            if [[ "$implementation" == rust ]]; then
                benchmark_command=("$CARGO_TARGET_DIR/release/examples/andersen_bench")
            else
                benchmark_command=(target/reference-andersen-bench)
            fi
            "$time_command" "${time_flags[@]}" \
                "${benchmark_command[@]}" "$particles" "$iterations" \
                > "$directory/${implementation}-${particles}-${repetition}.json" \
                2> "$directory/${implementation}-${particles}-${repetition}.memory.txt"
        done
    done
done
uv run --no-project python scripts/summarize-benchmarks.py "$directory"
