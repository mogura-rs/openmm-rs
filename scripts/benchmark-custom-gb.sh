#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
reference_source="${OPENMM_SOURCE:-../openmm}"
reference_build=target/reference-openmm
export CARGO_TARGET_DIR=target/benchmark-rust
cargo build --release --locked -p openmm-core --example custom_gb_bench --jobs "${SLURM_CPUS_PER_TASK:-4}"
"${CXX:-c++}" -std=c++17 -O3 tests/reference/custom_gb_bench.cpp \
    -I "$reference_source/openmmapi/include" -I "$reference_source/olla/include" \
    -I "$reference_source/serialization/include" \
    -L "$reference_build" -Wl,-rpath,"$(pwd)/$reference_build" -lOpenMM \
    -o target/reference-custom-gb-bench
time_command="$(type -P time)"
time_flags=(-v)
if [[ "$(uname -s)" == Darwin ]]; then time_flags=(-l); fi
for model in 0 1; do
for method in 0 1 2; do
    directory="benchmark-results/custom-gb/$model/$method"
    mkdir -p "$directory"
    for particles in 256 1024 4096; do
        case "$particles" in
            256) iterations=10 ;;
            1024) iterations=3 ;;
            4096) iterations=1 ;;
        esac
        for repetition in 1 2 3; do
            order=(rust reference)
            if [[ "$repetition" == 2 ]]; then order=(reference rust); fi
            for implementation in "${order[@]}"; do
                if [[ "$implementation" == rust ]]; then
                    benchmark_command=("$CARGO_TARGET_DIR/release/examples/custom_gb_bench")
                else
                    benchmark_command=(target/reference-custom-gb-bench)
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
