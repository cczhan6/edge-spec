#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRACE_ROOT="${TRACE_ROOT:-outputs/baseline_trace}"
SCENARIO="baseline_trace"
METHODS=(
  target_only
  server_only_linear
  server_only_tree
  specedge_linear
  specedge_tree
  dip_sd
)

mkdir -p "${TRACE_ROOT}"

"${PYTHON_BIN}" -m scripts.baseline_trace prepare \
  --root "${TRACE_ROOT}" \
  --config configs/default.yaml

CONFIG="${TRACE_ROOT}/baseline_trace_config.yaml"
DATASET="${TRACE_ROOT}/baseline_trace_dataset.jsonl"

for method in "${METHODS[@]}"; do
  method_dir="${TRACE_ROOT}/${method}"
  mkdir -p "${method_dir}"
  printf 'Running baseline trace: %s\n' "${method}"
  "${PYTHON_BIN}" -m scripts.run_all \
    --config "${CONFIG}" \
    --dataset "${DATASET}" \
    --scenario "${SCENARIO}" \
    --method "${method}" \
    --out_dir "${method_dir}" \
    --summary_out "${method_dir}/metrics.csv" \
    --use-fake-model-runner \
    --trace-bundle-dir "${method_dir}"
done

"${PYTHON_BIN}" -m scripts.baseline_trace verify \
  --root "${TRACE_ROOT}" \
  --summary "${TRACE_ROOT}/summary.md"
