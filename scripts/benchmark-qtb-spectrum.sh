#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
reference_source="${OPENMM_SOURCE:-../openmm}"
reference_build=target/reference-openmm
export CARGO_TARGET_DIR=target/benchmark-rust
jobs="${SLURM_CPUS_PER_TASK:-4}"
cargo build --release --locked -p openmm-core --example qtb_spectrum_bench --jobs "$jobs"
"${CXX:-c++}" -std=c++17 -O3 tests/reference/qtb_spectrum.cpp \
    -I "$reference_source/openmmapi/include" -I "$reference_source/olla/include" \
    -I "$reference_source/serialization/include" \
    -L "$reference_build" -Wl,-rpath,"$(pwd)/$reference_build" -lOpenMM \
    -o target/reference-qtb-spectrum
directory=benchmark-results/qtb-spectrum
mkdir -p "$directory"
time_command="$(type -P time)"
time_flags=(-v)
if [[ "$(uname -s)" == Darwin ]]; then time_flags=(-l); fi
for repetition in 1 2 3; do
    order=(rust reference)
    if [[ "$repetition" == 2 ]]; then order=(reference rust); fi
    for implementation in "${order[@]}"; do
        if [[ "$implementation" == rust ]]; then
            executable="$CARGO_TARGET_DIR/release/examples/qtb_spectrum_bench"
        else
            executable=target/reference-qtb-spectrum
        fi
        "$time_command" "${time_flags[@]}" "$executable" \
            > "$directory/$implementation-$repetition.jsonl" \
            2> "$directory/$implementation-$repetition.memory.txt"
    done
    uv run --no-project python scripts/compare-qtb-spectrum.py \
        --reference "$directory/reference-$repetition.jsonl" \
        --rust "$directory/rust-$repetition.jsonl" \
        | tee "$directory/comparison-$repetition.json"
done
