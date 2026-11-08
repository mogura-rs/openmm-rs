#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
reference_source="${OPENMM_SOURCE:-../openmm}"
kind="${1:-isotropic}"
case "$kind" in isotropic|anisotropic|membrane) ;; *) echo "Unknown barostat kind: $kind" >&2; exit 2 ;; esac
reference_build=target/reference-openmm
export CARGO_TARGET_DIR=target/benchmark-rust
jobs="${SLURM_CPUS_PER_TASK:-4}"
cargo build --release --locked -p openmm-core --example barostat_bench --jobs "$jobs"
"${CXX:-c++}" -std=c++17 -O3 tests/reference/barostat_bench.cpp \
    -I "$reference_source/openmmapi/include" -I "$reference_source/olla/include" \
    -I "$reference_source/serialization/include" \
    -L "$reference_build" -Wl,-rpath,"$(pwd)/$reference_build" -lOpenMM \
    -o target/reference-barostat-bench
time_command="$(type -P time)"
time_flags=(-v)
if [[ "$(uname -s)" == Darwin ]]; then time_flags=(-l); fi
for mode in rigid atomic; do
    directory="benchmark-results/barostat/$mode"
    if [[ "$kind" != isotropic ]]; then directory="benchmark-results/$kind-barostat/$mode"; fi
    mkdir -p "$directory"
    for particles in 512 4096 65536; do
        iterations=2000
        if [[ "$particles" == 512 ]]; then iterations=20000; fi
        if [[ "$particles" == 65536 ]]; then iterations=300; fi
        for repetition in 1 2 3; do
            order=(rust reference)
            if [[ "$repetition" == 2 ]]; then order=(reference rust); fi
            for implementation in "${order[@]}"; do
                if [[ "$implementation" == rust ]]; then
                    benchmark_command=("$CARGO_TARGET_DIR/release/examples/barostat_bench")
                else
                    benchmark_command=(target/reference-barostat-bench)
                fi
                "$time_command" "${time_flags[@]}" \
                    "${benchmark_command[@]}" "$particles" "$iterations" "$mode" "$kind" \
                    > "$directory/${implementation}-${particles}-${repetition}.json" \
                    2> "$directory/${implementation}-${particles}-${repetition}.memory.txt"
            done
        done
    done
    uv run --no-project python scripts/summarize-benchmarks.py "$directory"
done
