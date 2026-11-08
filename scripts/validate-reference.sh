#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
reference_source="${OPENMM_SOURCE:-../openmm}"
reference_build="target/reference-openmm"
jobs="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p validation-results

git -C "$reference_source" rev-parse HEAD > validation-results/reference-revision.txt
cmake -S "$reference_source" -B "$reference_build" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    -DOPENMM_BUILD_CPU_LIB=OFF -DOPENMM_BUILD_CUDA_LIB=OFF \
    -DOPENMM_BUILD_OPENCL_LIB=OFF -DOPENMM_BUILD_HIP_LIB=OFF \
    -DOPENMM_BUILD_AMOEBA_PLUGIN=OFF -DOPENMM_BUILD_RPMD_PLUGIN=OFF \
    -DOPENMM_BUILD_DRUDE_PLUGIN=OFF -DOPENMM_BUILD_PME_PLUGIN=OFF \
    -DOPENMM_BUILD_PYTHON_WRAPPERS=OFF -DOPENMM_BUILD_C_AND_FORTRAN_WRAPPERS=OFF \
    -DOPENMM_BUILD_EXAMPLES=OFF
cmake --build "$reference_build" --target OpenMM --parallel "$jobs"

for fixture in integrators nonbonded virtual_sites bonded; do
    "${CXX:-c++}" -std=c++17 -O2 "tests/reference/${fixture}.cpp" \
        -I "$reference_source/openmmapi/include" -I "$reference_source/olla/include" \
        -I "$reference_source/serialization/include" \
        -L "$reference_build" -Wl,-rpath,"$(pwd)/$reference_build" -lOpenMM \
        -o "target/reference-${fixture}"
    "target/reference-${fixture}" > "validation-results/${fixture}-reference.jsonl"
done

export PYO3_PYTHON
PYO3_PYTHON="$(uv run --no-project python -c 'import sys; print(sys.executable)')"
export CARGO_TARGET_DIR=target/reference-rust
cargo build --locked -p openmm-py --jobs "$jobs"
extension="$CARGO_TARGET_DIR/debug/libopenmm.so"
if [[ "$(uname -s)" == Darwin ]]; then
    extension="$CARGO_TARGET_DIR/debug/libopenmm.dylib"
fi
uv run --no-project python scripts/compare-reference.py \
    --extension "$extension" \
    --reference validation-results/integrators-reference.jsonl \
    | tee validation-results/integrators-comparison.json
uv run --no-project python scripts/compare-nonbonded.py \
    --extension "$extension" \
    --reference validation-results/nonbonded-reference.jsonl \
    | tee validation-results/nonbonded-comparison.json

uv run --no-project python scripts/compare-virtual-sites.py \
    --extension "$extension" \
    --reference validation-results/virtual_sites-reference.jsonl \
    | tee validation-results/virtual_sites-comparison.json
uv run --no-project python scripts/compare-bonded.py \
    --extension "$extension" \
    --reference validation-results/bonded-reference.jsonl \
    | tee validation-results/bonded-comparison.json
