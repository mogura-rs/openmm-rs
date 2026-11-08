#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
reference_source="${OPENMM_SOURCE:-../openmm}"
reference_build=target/reference-openmm
export CARGO_TARGET_DIR=target/benchmark-rust
cargo build --release --locked -p openmm-core --example custom_many_bench --jobs "${SLURM_CPUS_PER_TASK:-4}"
"${CXX:-c++}" -std=c++17 -O3 tests/reference/custom_many_bench.cpp \
    -I "$reference_source/openmmapi/include" -I "$reference_source/olla/include" \
    -I "$reference_source/serialization/include" \
    -L "$reference_build" -Wl,-rpath,"$(pwd)/$reference_build" -lOpenMM \
    -o target/reference-custom-many-bench
time_command="$(type -P time)"
time_flags=(-v)
if [[ "$(uname -s)" == Darwin ]]; then time_flags=(-l); fi
for model in 0 1; do
    for method in 0 1 2; do
        directory="benchmark-results/custom-many/$model-$method"
        mkdir -p "$directory"
        if [[ "$method" == 0 ]]; then sizes=(32 64 128); else sizes=(128 256 512); fi
        for particles in "${sizes[@]}"; do
            if [[ "$method" == 0 ]]; then
                case "$particles" in 32) iterations=16 ;; 64) iterations=4 ;; 128) iterations=1 ;; esac
            else
                case "$particles" in 128) iterations=10 ;; 256) iterations=3 ;; 512) iterations=1 ;; esac
            fi
            for repetition in 1 2 3; do
                order=(rust reference)
                if [[ "$repetition" == 2 ]]; then order=(reference rust); fi
                for implementation in "${order[@]}"; do
                    if [[ "$implementation" == rust ]]; then
                        benchmark_command=("$CARGO_TARGET_DIR/release/examples/custom_many_bench")
                    else
                        benchmark_command=(target/reference-custom-many-bench)
                    fi
                    "$time_command" "${time_flags[@]}" "${benchmark_command[@]}" "$particles" "$iterations" "$method" "$model" \
                        > "$directory/${implementation}-${particles}-${repetition}.json" \
                        2> "$directory/${implementation}-${particles}-${repetition}.memory.txt"
                done
            done
        done
        uv run --no-project python scripts/summarize-benchmarks.py "$directory"
    done
done
