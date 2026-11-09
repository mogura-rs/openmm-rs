#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
reference_source="${OPENMM_SOURCE:-../openmm}"
reference_build="${OPENMM_REFERENCE_BUILD:-target/reference-openmm}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target/benchmark-rust}"
cargo build --release --locked -p openmm-core --example atm_bench --jobs "${SLURM_CPUS_PER_TASK:-4}"
mkdir -p target
"${CXX:-c++}" -std=c++17 -O3 tests/reference/atm_bench.cpp \
    -I "$reference_source/openmmapi/include" -I "$reference_source/olla/include" -I "$reference_source/serialization/include" \
    -L "$reference_build" -Wl,-rpath,"$(cd "$reference_build" && pwd)" -lOpenMM -o target/reference-atm-bench
time_command="$(type -P time)"; time_flags=(-v)
if [[ "$(uname -s)" == Darwin ]]; then time_flags=(-l); fi
for model in 0 1; do
    directory="benchmark-results/atm/$model"; mkdir -p "$directory"
    for particles in 128 512 2048; do
        case "$particles" in 128) iterations=500 ;; 512) iterations=100 ;; 2048) iterations=20 ;; esac
        for repetition in 1 2 3; do
            order=(rust reference); if [[ "$repetition" == 2 ]]; then order=(reference rust); fi
            for implementation in "${order[@]}"; do
                if [[ "$implementation" == rust ]]; then benchmark_command=("$CARGO_TARGET_DIR/release/examples/atm_bench"); else benchmark_command=(target/reference-atm-bench); fi
                "$time_command" "${time_flags[@]}" "${benchmark_command[@]}" "$particles" "$iterations" "$model" \
                    > "$directory/${implementation}-${particles}-${repetition}.json" \
                    2> "$directory/${implementation}-${particles}-${repetition}.memory.txt"
            done
        done
    done
    uv run --no-project python scripts/summarize-benchmarks.py "$directory"
done
