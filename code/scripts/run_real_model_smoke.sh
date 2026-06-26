#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT="${REAL_MODEL_SMOKE_ROOT:-outputs/real_model_smoke}"
CONFIG_TEMPLATE="${CONFIG:-configs/default.yaml}"
TARGET_MODEL="${TARGET_MODEL_PATH:-}"
DRAFT_MODEL="${DRAFT_MODEL_PATH:-}"
DATASET="${DATASET_PATH:-}"
TARGET_DEVICE="${TARGET_DEVICE:-cuda:1}"
DRAFT_DEVICE="${DRAFT_DEVICE:-cuda:0}"
NUM_REQUESTS="${NUM_REQUESTS:-4}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-8}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-}"
CACHE_DIR="${HF_CACHE_DIR:-}"
MODEL_REVISION="${MODEL_REVISION:-}"
METHODS="${REAL_MODEL_SMOKE_METHODS:-}"

usage() {
  cat >&2 <<'USAGE'
Usage:
  TARGET_MODEL_PATH=/path/to/target DRAFT_MODEL_PATH=/path/to/draft \
    bash scripts/run_real_model_smoke.sh [options]

Options:
  --root PATH              Output root (default: outputs/real_model_smoke)
  --config PATH            Base config (default: configs/default.yaml)
  --target-model PATH      Target HF model/path (or TARGET_MODEL_PATH)
  --draft-model PATH       Drafter HF model/path (or DRAFT_MODEL_PATH)
  --dataset PATH           Optional source dataset JSONL (or DATASET_PATH)
  --target-device DEVICE   Target torch device (default: cuda:1)
  --draft-device DEVICE    Drafter torch device (default: cuda:0)
  --num-requests N         2-4 requests (default: 4)
  --output-tokens N        8-16 tokens per request (default: 8)
  --local-files-only BOOL  Pass model_runner.local_files_only
  --cache-dir PATH         Hugging Face cache_dir
  --revision REV           Hugging Face revision
  --methods LIST           Comma-separated methods (default:
                           target_only,server_only_linear,specedge_linear,dip_sd).
                           target_only is added automatically as the greedy
                           reference when omitted.

This script never enables the fake runner. Missing model paths are a hard error
rather than a fallback to deterministic traces.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="$2"
      shift 2
      ;;
    --config)
      CONFIG_TEMPLATE="$2"
      shift 2
      ;;
    --target-model)
      TARGET_MODEL="$2"
      shift 2
      ;;
    --draft-model)
      DRAFT_MODEL="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --target-device)
      TARGET_DEVICE="$2"
      shift 2
      ;;
    --draft-device)
      DRAFT_DEVICE="$2"
      shift 2
      ;;
    --num-requests)
      NUM_REQUESTS="$2"
      shift 2
      ;;
    --output-tokens)
      OUTPUT_TOKENS="$2"
      shift 2
      ;;
    --local-files-only)
      LOCAL_FILES_ONLY="$2"
      shift 2
      ;;
    --cache-dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    --revision)
      MODEL_REVISION="$2"
      shift 2
      ;;
    --methods)
      METHODS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${TARGET_MODEL}" || -z "${DRAFT_MODEL}" ]]; then
  printf 'TARGET_MODEL_PATH and DRAFT_MODEL_PATH must be provided explicitly.\n' >&2
  usage
  exit 2
fi

mkdir -p "${ROOT}"

PREPARE_ARGS=(
  -m scripts.real_model_smoke prepare
  --root "${ROOT}"
  --config "${CONFIG_TEMPLATE}"
  --target-model "${TARGET_MODEL}"
  --draft-model "${DRAFT_MODEL}"
  --target-device "${TARGET_DEVICE}"
  --draft-device "${DRAFT_DEVICE}"
  --num-requests "${NUM_REQUESTS}"
  --output-tokens "${OUTPUT_TOKENS}"
)
if [[ -n "${DATASET}" ]]; then
  PREPARE_ARGS+=(--dataset "${DATASET}")
fi
if [[ -n "${LOCAL_FILES_ONLY}" ]]; then
  PREPARE_ARGS+=(--local-files-only "${LOCAL_FILES_ONLY}")
fi
if [[ -n "${CACHE_DIR}" ]]; then
  PREPARE_ARGS+=(--cache-dir "${CACHE_DIR}")
fi
if [[ -n "${MODEL_REVISION}" ]]; then
  PREPARE_ARGS+=(--revision "${MODEL_REVISION}")
fi

"${PYTHON_BIN}" "${PREPARE_ARGS[@]}"

CONFIG="${ROOT}/real_model_smoke_config.yaml"
DATASET_PREPARED="${ROOT}/real_model_smoke_dataset.jsonl"
SCENARIO="real_model_smoke"
DEFAULT_METHODS=(target_only server_only_linear specedge_linear dip_sd)
SUPPORTED_METHODS=(target_only server_only_linear server_only_tree specedge_linear specedge_tree dip_sd)

contains_method() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

validate_method() {
  local method="$1"
  if contains_method "${method}" "${SUPPORTED_METHODS[@]}"; then
    return 0
  fi
  printf 'Unknown real model smoke method: %s\n' "${method}" >&2
  exit 2
}

SELECTED_METHODS=()
if [[ -z "${METHODS}" ]]; then
  SELECTED_METHODS=("${DEFAULT_METHODS[@]}")
else
  IFS=',' read -r -a RAW_METHODS <<< "${METHODS}"
  for raw_method in "${RAW_METHODS[@]}"; do
    method="${raw_method//[[:space:]]/}"
    if [[ -z "${method}" ]]; then
      continue
    fi
    validate_method "${method}"
    if ! contains_method "${method}" "${SELECTED_METHODS[@]}"; then
      SELECTED_METHODS+=("${method}")
    fi
  done
  if [[ "${#SELECTED_METHODS[@]}" -eq 0 ]]; then
    SELECTED_METHODS=("${DEFAULT_METHODS[@]}")
  elif ! contains_method "target_only" "${SELECTED_METHODS[@]}"; then
    SELECTED_METHODS=(target_only "${SELECTED_METHODS[@]}")
  fi
fi

METHODS_CSV="$(IFS=','; printf '%s' "${SELECTED_METHODS[*]}")"

write_manifest() {
  local method="$1"
  local method_dir="$2"
  local command_text="$3"
  local return_code="$4"
  local log_path="$5"
  local skipped_reason="${6:-}"
  local gpu_peak="${7:-n/a}"
  local args=(
    -m scripts.real_model_smoke manifest
    --output-dir "${method_dir}"
    --method "${method}"
    --command-text "${command_text}"
    --return-code "${return_code}"
    --config "${CONFIG}"
    --dataset "${DATASET_PREPARED}"
    --target-model "${TARGET_MODEL}"
    --draft-model "${DRAFT_MODEL}"
    --target-device "${TARGET_DEVICE}"
    --draft-device "${DRAFT_DEVICE}"
    --stdout-log "${log_path}"
    --gpu-peak-mb "${gpu_peak}"
  )
  if [[ -n "${skipped_reason}" ]]; then
    args+=(--skipped-reason "${skipped_reason}")
  fi
  "${PYTHON_BIN}" "${args[@]}" >/dev/null
}

run_method() {
  local method="$1"
  local method_dir="${ROOT}/${method}"
  local log_path="${method_dir}/stdout.log"
  mkdir -p "${method_dir}"
  local cmd=(
    "${PYTHON_BIN}"
    -m scripts.run_all
    --config "${CONFIG}"
    --dataset "${DATASET_PREPARED}"
    --scenario "${SCENARIO}"
    --method "${method}"
    --out_dir "${method_dir}"
    --summary_out "${method_dir}/metrics.csv"
    --trace-bundle-dir "${method_dir}"
  )
  printf 'Running real model smoke: %s\n' "${method}"
  set +e
  "${cmd[@]}" >"${log_path}" 2>&1
  local status=$?
  set -e
  if [[ -f "${method_dir}/resolved_config.json" ]]; then
    cp "${method_dir}/resolved_config.json" "${method_dir}/resolved_config"
  fi
  write_manifest "${method}" "${method_dir}" "${cmd[*]}" "${status}" "${log_path}"
  if [[ "${status}" -ne 0 ]]; then
    printf 'real model smoke failed for %s; see %s\n' "${method}" "${log_path}" >&2
  fi
  return "${status}"
}

skip_method() {
  local method="$1"
  local reason="$2"
  local method_dir="${ROOT}/${method}"
  local log_path="${method_dir}/stdout.log"
  mkdir -p "${method_dir}"
  printf 'skipped: %s\n' "${reason}" >"${log_path}"
  write_manifest "${method}" "${method_dir}" "skipped" "125" "${log_path}" "${reason}"
}

run_status=0
for method in "${SELECTED_METHODS[@]}"; do
  if ! run_method "${method}"; then
    run_status=1
  fi
done

set +e
"${PYTHON_BIN}" -m scripts.real_model_smoke verify \
  --root "${ROOT}" \
  --summary "${ROOT}/summary.md" \
  --expected-requests "${NUM_REQUESTS}" \
  --methods "${METHODS_CSV}"
verify_status=$?
set -e

if [[ "${run_status}" -ne 0 ]]; then
  exit 1
fi
exit "${verify_status}"
