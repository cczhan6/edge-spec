#!/usr/bin/env bash
set -euo pipefail

ROOT="${PREFLIGHT_ROOT:-outputs/baseline_preflight}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/default.yaml}"
DATASET="${DATASET:-data/spec_bench/question.jsonl}"
RESIDENCY_MANIFEST="${DRAFTER_RESIDENCY_MANIFEST:-outputs/drafter_residency_manifest.json}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCENARIOS=(homogeneous combined_strong_heterogeneous)
SEEDS=(20260628 20260629)
METHODS=(
  target_only
  server_only_linear
  server_only_tree
  specedge_linear
  specedge_tree
  dip_sd
)

mkdir -p "${ROOT}/_configs"

if [[ ! -s "${RESIDENCY_MANIFEST}" ]]; then
  "${PYTHON_BIN}" scripts/check_drafter_residency.py \
    --config "${CONFIG}" \
    --output "${RESIDENCY_MANIFEST}"
fi

"${PYTHON_BIN}" scripts/collect_experiment_environment.py \
  --output "${ROOT}/environment_manifest.json"
"${PYTHON_BIN}" scripts/verify_baseline_preflight.py attach-residency \
  --environment "${ROOT}/environment_manifest.json" \
  --residency "${RESIDENCY_MANIFEST}"

for scenario in "${SCENARIOS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    cell="${ROOT}/${scenario}/${seed}"
    config_out="${ROOT}/_configs/${scenario}_${seed}.yaml"
    audit_out="${ROOT}/_configs/${scenario}_${seed}_audit.json"
    mkdir -p "${cell}/_raw"

    "${PYTHON_BIN}" scripts/verify_baseline_preflight.py prepare \
      --config "${CONFIG}" \
      --scenario "${scenario}" \
      --seed "${seed}" \
      --output "${config_out}"

    "${PYTHON_BIN}" scripts/audit_experiment_config.py \
      --config "${config_out}" \
      --scenario "${scenario}" \
      --methods "${METHODS[@]}" \
      --resolved-config-out "${audit_out}"

    command=(
      "${PYTHON_BIN}"
      -m scripts.run_all
      --config "${config_out}"
      --dataset "${DATASET}"
      --scenario "${scenario}"
      --methods "${METHODS[@]}"
      --out_dir "${cell}/_raw"
      --summary_out "${cell}/_raw/all_results.csv"
      --trace-bundle-root "${cell}"
    )

    printf 'Running baseline preflight: scenario=%s seed=%s\n' "${scenario}" "${seed}"
    "${command[@]}" >"${cell}/stdout.log" 2>&1
    for method in "${METHODS[@]}"; do
      cp "${cell}/stdout.log" "${cell}/${method}/stdout.log"
    done

    "${PYTHON_BIN}" scripts/verify_baseline_preflight.py materialize \
      --cell "${cell}" \
      --scenario "${scenario}" \
      --seed "${seed}" \
      --environment "${ROOT}/environment_manifest.json" \
      --command-text "${command[*]}"
  done
done

"${PYTHON_BIN}" scripts/verify_baseline_preflight.py --root "${ROOT}"
