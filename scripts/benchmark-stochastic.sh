#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
reference_source="${OPENMM_SOURCE:-../openmm}"
reference_build=target/reference-openmm
export CARGO_TARGET_DIR=target/benchmark-rust
jobs="${SLURM_CPUS_PER_TASK:-4}"
cargo build --release --locked -p openmm-core --example stochastic_bench --jobs "$jobs"
"${CXX:-c++}" -std=c++17 -O3 tests/reference/stochastic_bench.cpp \
    -I "$reference_source/openmmapi/include" -I "$reference_source/olla/include" \
    -I "$reference_source/serialization/include" \
    -L "$reference_build" -Wl,-rpath,"$(pwd)/$reference_build" -lOpenMM \
    -o target/reference-stochastic-bench
time_command="$(type -P time)"
time_flags=(-v)
if [[ "$(uname -s)" == Darwin ]]; then time_flags=(-l); fi
for kind in verlet langevin brownian; do
    temperatures=(0 300)
    if [[ "$kind" == verlet ]]; then temperatures=(0); fi
    for temperature in "${temperatures[@]}"; do
        directory="benchmark-results/stochastic/$kind/$temperature"
        mkdir -p "$directory"
        for particles in 8 4096 65536; do
            case "$particles" in
                8) iterations=30000 ;;
                4096) iterations=1500 ;;
                65536) iterations=200 ;;
            esac
            for repetition in 1 2 3; do
                order=(rust reference)
                if [[ "$repetition" == 2 ]]; then order=(reference rust); fi
                for implementation in "${order[@]}"; do
                    if [[ "$implementation" == rust ]]; then
                        benchmark_command=("$CARGO_TARGET_DIR/release/examples/stochastic_bench")
                    else
                        benchmark_command=(target/reference-stochastic-bench)
                    fi
                    "$time_command" "${time_flags[@]}" "${benchmark_command[@]}" "$kind" "$particles" "$iterations" "$temperature" \
                        > "$directory/${implementation}-${particles}-${repetition}.json" \
                        2> "$directory/${implementation}-${particles}-${repetition}.memory.txt"
                done
            done
        done
        uv run --no-project python scripts/summarize-benchmarks.py "$directory"
    done
done
