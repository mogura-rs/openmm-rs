#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
reference_source="${OPENMM_SOURCE:-../openmm}"
reference_build="target/reference-openmm"
jobs="${SLURM_CPUS_PER_TASK:-4}"
fixtures=(integrators nonbonded virtual_sites bonded groups molecules minimizer custom tables cmap custom_nonbonded geometry centroid)
if [[ $# -gt 0 ]]; then
    fixtures=("$@")
fi
for fixture in "${fixtures[@]}"; do
    case "$fixture" in
        integrators|nonbonded|virtual_sites|bonded|groups|molecules|minimizer|custom|tables|cmap|custom_nonbonded|geometry|centroid) ;;
        *) echo "Unknown reference fixture: $fixture" >&2; exit 2 ;;
    esac
done
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

for fixture in "${fixtures[@]}"; do
    "${CXX:-c++}" -std=c++17 -O2 "tests/reference/${fixture}.cpp" \
        -I "$reference_source/openmmapi/include" -I "$reference_source/olla/include" \
        -I "$reference_source/serialization/include" \
        -I "$reference_source/platforms/reference/include" -I "$reference_source/libraries/lepton/include" \
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
for fixture in "${fixtures[@]}"; do
    case "$fixture" in
        integrators) comparison=reference ;;
        virtual_sites) comparison=virtual-sites ;;
        *) comparison="$fixture" ;;
    esac
    uv run --no-project python "scripts/compare-${comparison}.py" \
        --extension "$extension" \
        --reference "validation-results/${fixture}-reference.jsonl" \
        | tee "validation-results/${fixture}-comparison.json"
done
