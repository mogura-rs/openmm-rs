#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
reference_source="${OPENMM_SOURCE:-../openmm}"
reference_build="${OPENMM_REFERENCE_BUILD:-target/reference-openmm}"
jobs="${SLURM_CPUS_PER_TASK:-4}"
fixtures=(integrators nonbonded virtual_sites bonded groups molecules minimizer custom tables cmap custom_nonbonded geometry centroid hbond collective cv cm_motion andersen ownership integrator_binding barostat anisotropic_barostat membrane_barostat flexible_barostat variable_integrators compound_integrator nose_hoover dpd qtb custom_integrator gbsa_obc custom_gb lcpo gay_berne custom_many constant_potential drude atm)
if [[ $# -gt 0 ]]; then
    fixtures=("$@")
fi
for fixture in "${fixtures[@]}"; do
    case "$fixture" in
        integrators|nonbonded|virtual_sites|bonded|groups|molecules|minimizer|custom|tables|cmap|custom_nonbonded|geometry|centroid|hbond|collective|cv|cm_motion|andersen|ownership|integrator_binding|barostat|anisotropic_barostat|membrane_barostat|flexible_barostat|variable_integrators|compound_integrator|nose_hoover|dpd|qtb|custom_integrator|gbsa_obc|custom_gb|lcpo|gay_berne|custom_many|constant_potential|drude|atm) ;;
        *) echo "Unknown reference fixture: $fixture" >&2; exit 2 ;;
    esac
done
with_drude=OFF
for fixture in "${fixtures[@]}"; do
    if [[ "$fixture" == drude ]]; then with_drude=ON; fi
done
mkdir -p validation-results

git -C "$reference_source" rev-parse HEAD > validation-results/reference-revision.txt
cmake -S "$reference_source" -B "$reference_build" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    -DOPENMM_BUILD_CPU_LIB=OFF -DOPENMM_BUILD_CUDA_LIB=OFF \
    -DOPENMM_BUILD_OPENCL_LIB=OFF -DOPENMM_BUILD_HIP_LIB=OFF \
    -DOPENMM_BUILD_AMOEBA_PLUGIN=OFF -DOPENMM_BUILD_RPMD_PLUGIN=OFF \
    -DOPENMM_BUILD_DRUDE_PLUGIN="$with_drude" -DOPENMM_BUILD_PME_PLUGIN=OFF \
    -DOPENMM_BUILD_PYTHON_WRAPPERS=OFF -DOPENMM_BUILD_C_AND_FORTRAN_WRAPPERS=OFF \
    -DOPENMM_BUILD_EXAMPLES=OFF
reference_targets=(OpenMM)
if [[ "$with_drude" == ON ]]; then reference_targets+=(OpenMMDrude OpenMMDrudeReference); fi
cmake --build "$reference_build" --target "${reference_targets[@]}" --parallel "$jobs"

for fixture in "${fixtures[@]}"; do
    plugin_flags=()
    if [[ "$fixture" == drude ]]; then
        plugin_flags=(-I "$reference_source/plugins/drude/openmmapi/include" -lOpenMMDrudeReference -lOpenMMDrude)
    fi
    "${CXX:-c++}" -std=c++17 -O2 "tests/reference/${fixture}.cpp" \
        -I "$reference_source/openmmapi/include" -I "$reference_source/olla/include" \
        -I "$reference_source/serialization/include" \
        -I "$reference_source/platforms/reference/include" -I "$reference_source/libraries/lepton/include" \
        -L "$reference_build" -Wl,-rpath,"$(cd "$reference_build" && pwd)" "${plugin_flags[@]}" -lOpenMM \
        -o "target/reference-${fixture}"
    "target/reference-${fixture}" > "validation-results/${fixture}-reference.jsonl"
done

export PYO3_PYTHON
PYO3_PYTHON="$(uv run --no-project python -c 'import sys; print(sys.executable)')"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target/reference-rust}"
profile="${OPENMM_RS_PROFILE:-release}"
build_flags=()
case "$profile" in
    release) build_flags=(--release) ;;
    debug) ;;
    *) echo "Unknown Rust validation profile: $profile" >&2; exit 2 ;;
esac
cargo build --locked -p openmm-py --jobs "$jobs" "${build_flags[@]}"
printf '%s\n' "$profile" > validation-results/rust-profile.txt
extension="$CARGO_TARGET_DIR/$profile/libopenmm.so"
if [[ "$(uname -s)" == Darwin ]]; then
    extension="$CARGO_TARGET_DIR/$profile/libopenmm.dylib"
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
