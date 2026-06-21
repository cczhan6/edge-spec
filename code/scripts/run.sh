#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-configs/default.yaml}"
DATASET="${DATASET:-data/spec_bench/question.jsonl}"
RUN_ROOT="${RUN_ROOT:-outputs/runs}"
RUN_ID="${RUN_ID:-}"
RUN_DIR="${RUN_DIR:-}"
OUT_DIR="${OUT_DIR:-}"
SUMMARY_OUT="${SUMMARY_OUT:-}"
SCENARIO="${SCENARIO:-combined_strong_heterogeneous}"
SCENARIOS="${SCENARIOS:-homogeneous combined_strong_heterogeneous}"
METHOD="${METHOD:-full}"
METHODS="${METHODS:-full target_only sync_batch_sd SpecEdge server_only}"
W_VALUES="${W_VALUES:-1 2 3 4}"
LANE_VALUES="${LANE_VALUES:-1 2 4 8}"
USE_FAKE_MODEL_RUNNER="${USE_FAKE_MODEL_RUNNER:-${USE_FAKE_ORACLE:-0}}"
SAMPLES_PER_CATEGORY="${SAMPLES_PER_CATEGORY:-}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
TREE_DRAFT_STRATEGY="${TREE_DRAFT_STRATEGY:-}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run.sh smoke
  bash scripts/run.sh single
  bash scripts/run.sh all
  bash scripts/run.sh sensitivity-w
  bash scripts/run.sh sensitivity-lanes
  bash scripts/run.sh help

Commands:
  smoke              Run combined_strong_heterogeneous + full with the fake model runner.
  single             Run one real-model scenario and method.
  all                Run all default scenarios with main baselines and Full.
  sensitivity-w      Legacy fixed-window sweep; current Full ignores W_default.
  sensitivity-lanes  Run Full with lane counts from LANE_VALUES.
  help               Show this help.

Environment overrides:
  CONFIG=configs/default.yaml
  DATASET=data/spec_bench/question.jsonl
  RUN_ROOT=outputs/runs
  RUN_ID=<YYYYMMDD-HHMMSS>
  RUN_DIR=outputs/runs/<RUN_ID>
  OUT_DIR=<RUN_DIR>/raw
  SUMMARY_OUT=<RUN_DIR>/summary/all_results.csv
  SCENARIO=combined_strong_heterogeneous
  SCENARIOS="homogeneous combined_strong_heterogeneous"
  METHOD=full
  METHODS="full target_only sync_batch_sd SpecEdge server_only"
  W_VALUES="1 2 3 4"
  LANE_VALUES="1 2 4 8"
  USE_FAKE_MODEL_RUNNER=1
  SAMPLES_PER_CATEGORY=10
  SUMMARY_ONLY=1
  TREE_DRAFT_STRATEGY=linear|specexec_approx
  PYTHON_BIN=python3

Examples:
  bash scripts/run.sh smoke
  SCENARIO=combined_strong_heterogeneous METHOD=full bash scripts/run.sh single
  SAMPLES_PER_CATEGORY=10 bash scripts/run.sh all
  SCENARIOS=combined_strong_heterogeneous SAMPLES_PER_CATEGORY=10 bash scripts/run.sh all
  SCENARIOS=combined_strong_heterogeneous SUMMARY_ONLY=1 bash scripts/run.sh all
  USE_FAKE_MODEL_RUNNER=1 LANE_VALUES="1 2" bash scripts/run.sh sensitivity-lanes
  TREE_DRAFT_STRATEGY=specexec_approx METHOD=SpecEdge bash scripts/run.sh single
EOF
}

enabled() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    0|false|no|off|"") return 1 ;;
    *)
      printf 'Invalid boolean value: %s\n' "$1" >&2
      exit 2
      ;;
  esac
}

print_and_run() {
  printf 'Running:'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

yaml_quote() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "${value}"
}

write_yaml_array() {
  local key="$1"
  shift
  if [[ "$#" -eq 0 ]]; then
    printf '%s: []\n' "${key}"
    return
  fi
  printf '%s:\n' "${key}"
  local item
  for item in "$@"; do
    printf '  - '
    yaml_quote "${item}"
    printf '\n'
  done
}

prepare_run_dir() {
  local base_id suffix candidate
  base_id="${RUN_ID:-$(date '+%Y%m%d-%H%M%S')}"
  if [[ -z "${RUN_DIR}" ]]; then
    candidate="${RUN_ROOT}/${base_id}"
    if [[ -z "${RUN_ID}" ]]; then
      suffix=1
      while [[ -e "${candidate}" ]]; do
        candidate="${RUN_ROOT}/${base_id}-$(printf '%02d' "${suffix}")"
        suffix=$((suffix + 1))
      done
    elif [[ -e "${candidate}" ]]; then
      printf 'Run directory already exists: %s\n' "${candidate}" >&2
      printf 'Choose a different RUN_ID or RUN_DIR to avoid overwriting a run.\n' >&2
      exit 2
    fi
    RUN_DIR="${candidate}"
  elif [[ -e "${RUN_DIR}" ]]; then
    printf 'Run directory already exists: %s\n' "${RUN_DIR}" >&2
    printf 'Choose a different RUN_DIR to avoid overwriting a run.\n' >&2
    exit 2
  fi
  RUN_ID="$(basename "${RUN_DIR}")"
  OUT_DIR="${OUT_DIR:-${RUN_DIR}/raw}"
  SUMMARY_OUT="${SUMMARY_OUT:-${RUN_DIR}/summary/all_results.csv}"
  mkdir -p "${RUN_DIR}" "${OUT_DIR}" "$(dirname -- "${SUMMARY_OUT}")"
}

write_manifest() {
  local command_name="$1"
  shift
  local manifest_path started_at git_commit git_dirty
  manifest_path="${RUN_DIR}/manifest.yaml"
  started_at="$(date '+%Y-%m-%d %H:%M:%S %z')"
  git_commit="unknown"
  git_dirty="unknown"
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git_commit="$(git rev-parse --short HEAD 2>/dev/null || printf 'unknown')"
    if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
      git_dirty="true"
    else
      git_dirty="false"
    fi
  fi

  {
    printf 'run_id: '
    yaml_quote "${RUN_ID}"
    printf '\nstarted_at: '
    yaml_quote "${started_at}"
    printf '\ncommand: '
    yaml_quote "bash scripts/run.sh ${command_name}"
    printf '\nconfig: '
    yaml_quote "${CONFIG}"
    printf '\ndataset: '
    yaml_quote "${DATASET}"
    printf '\nrun_dir: '
    yaml_quote "${RUN_DIR}"
    printf '\nout_dir: '
    yaml_quote "${OUT_DIR}"
    printf '\nsummary_out: '
    yaml_quote "${SUMMARY_OUT}"
    printf '\n'
    write_yaml_array "scenarios" "${MANIFEST_SCENARIOS[@]}"
    write_yaml_array "methods" "${MANIFEST_METHODS[@]}"
    if [[ "${#MANIFEST_VALUES[@]}" -gt 0 ]]; then
      write_yaml_array "${MANIFEST_VALUE_KEY}" "${MANIFEST_VALUES[@]}"
    fi
    printf 'samples_per_category: '
    if [[ -n "${SAMPLES_PER_CATEGORY}" ]]; then
      printf '%s\n' "${SAMPLES_PER_CATEGORY}"
    else
      printf 'null\n'
    fi
    printf 'use_fake_model_runner: '
    if [[ "${command_name}" == "smoke" ]] || enabled "${USE_FAKE_MODEL_RUNNER}"; then
      printf 'true\n'
    else
      printf 'false\n'
    fi
    printf 'summary_only: '
    if enabled "${SUMMARY_ONLY}"; then
      printf 'true\n'
    else
      printf 'false\n'
    fi
    printf 'tree_draft_strategy: '
    if [[ -n "${TREE_DRAFT_STRATEGY}" ]]; then
      yaml_quote "${TREE_DRAFT_STRATEGY}"
      printf '\n'
    else
      printf 'null\n'
    fi
    printf 'git_commit: '
    yaml_quote "${git_commit}"
    printf '\ngit_dirty: %s\n' "${git_dirty}"
  } > "${manifest_path}"

  printf 'Run directory: %s\n' "${RUN_DIR}" >&2
  printf 'Manifest: %s\n' "${manifest_path}" >&2
}

if [[ "$#" -ne 1 ]]; then
  usage >&2
  exit 2
fi

COMMAND="$1"
OPTIONAL_ARGS=()
MANIFEST_SCENARIOS=()
MANIFEST_METHODS=()
MANIFEST_VALUES=()
MANIFEST_VALUE_KEY="values"
if [[ -n "${SAMPLES_PER_CATEGORY}" ]]; then
  OPTIONAL_ARGS+=(--samples-per-category "${SAMPLES_PER_CATEGORY}")
fi
if [[ -n "${TREE_DRAFT_STRATEGY}" ]]; then
  OPTIONAL_ARGS+=(--tree-draft-strategy "${TREE_DRAFT_STRATEGY}")
fi

case "${COMMAND}" in
  smoke)
    OPTIONAL_ARGS=()
    if [[ -n "${TREE_DRAFT_STRATEGY}" ]]; then
      OPTIONAL_ARGS+=(--tree-draft-strategy "${TREE_DRAFT_STRATEGY}")
    fi
    if enabled "${SUMMARY_ONLY}"; then
      OPTIONAL_ARGS+=(--summary-only)
    fi
    MANIFEST_SCENARIOS=(combined_strong_heterogeneous)
    MANIFEST_METHODS=(full)
    prepare_run_dir
    write_manifest "${COMMAND}"
    print_and_run "${PYTHON_BIN}" -m scripts.run_all \
      --config "${CONFIG}" \
      --dataset "${DATASET}" \
      --scenario combined_strong_heterogeneous \
      --method full \
      --out_dir "${OUT_DIR}" \
      --summary_out "${SUMMARY_OUT}" \
      --use-fake-model-runner \
      "${OPTIONAL_ARGS[@]}"
    ;;
  single)
    if enabled "${USE_FAKE_MODEL_RUNNER}"; then
      OPTIONAL_ARGS+=(--use-fake-model-runner)
    fi
    if enabled "${SUMMARY_ONLY}"; then
      OPTIONAL_ARGS+=(--summary-only)
    fi
    MANIFEST_SCENARIOS=("${SCENARIO}")
    MANIFEST_METHODS=("${METHOD}")
    prepare_run_dir
    write_manifest "${COMMAND}"
    print_and_run "${PYTHON_BIN}" -m scripts.run_all \
      --config "${CONFIG}" \
      --dataset "${DATASET}" \
      --scenario "${SCENARIO}" \
      --method "${METHOD}" \
      --out_dir "${OUT_DIR}" \
      --summary_out "${SUMMARY_OUT}" \
      "${OPTIONAL_ARGS[@]}"
    ;;
  all)
    if enabled "${USE_FAKE_MODEL_RUNNER}"; then
      OPTIONAL_ARGS+=(--use-fake-model-runner)
    fi
    if enabled "${SUMMARY_ONLY}"; then
      OPTIONAL_ARGS+=(--summary-only)
    fi
    read -r -a SCENARIO_VALUES <<< "${SCENARIOS}"
    read -r -a METHOD_VALUES <<< "${METHODS}"
    MANIFEST_SCENARIOS=("${SCENARIO_VALUES[@]}")
    MANIFEST_METHODS=("${METHOD_VALUES[@]}")
    prepare_run_dir
    write_manifest "${COMMAND}"
    print_and_run "${PYTHON_BIN}" -m scripts.run_all \
      --config "${CONFIG}" \
      --dataset "${DATASET}" \
      --scenarios "${SCENARIO_VALUES[@]}" \
      --methods "${METHOD_VALUES[@]}" \
      --out_dir "${OUT_DIR}" \
      --summary_out "${SUMMARY_OUT}" \
      "${OPTIONAL_ARGS[@]}"
    ;;
  sensitivity-w)
    if enabled "${USE_FAKE_MODEL_RUNNER}"; then
      OPTIONAL_ARGS+=(--use-fake-model-runner)
    fi
    read -r -a VALUES <<< "${W_VALUES}"
    MANIFEST_SCENARIOS=("${SCENARIO}")
    MANIFEST_METHODS=(full)
    MANIFEST_VALUES=("${VALUES[@]}")
    MANIFEST_VALUE_KEY="w_values"
    prepare_run_dir
    write_manifest "${COMMAND}"
    print_and_run "${PYTHON_BIN}" -m scripts.run_sensitivity_w \
      --config "${CONFIG}" \
      --dataset "${DATASET}" \
      --scenario "${SCENARIO}" \
      --values "${VALUES[@]}" \
      --out "${OUT_DIR}/sensitivity_w.csv" \
      "${OPTIONAL_ARGS[@]}"
    ;;
  sensitivity-lanes)
    if enabled "${USE_FAKE_MODEL_RUNNER}"; then
      OPTIONAL_ARGS+=(--use-fake-model-runner)
    fi
    read -r -a VALUES <<< "${LANE_VALUES}"
    MANIFEST_SCENARIOS=("${SCENARIO}")
    MANIFEST_METHODS=(full)
    MANIFEST_VALUES=("${VALUES[@]}")
    MANIFEST_VALUE_KEY="lane_values"
    prepare_run_dir
    write_manifest "${COMMAND}"
    print_and_run "${PYTHON_BIN}" -m scripts.run_sensitivity_lanes \
      --config "${CONFIG}" \
      --dataset "${DATASET}" \
      --scenario "${SCENARIO}" \
      --values "${VALUES[@]}" \
      --out "${OUT_DIR}/sensitivity_lanes.csv" \
      "${OPTIONAL_ARGS[@]}"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    printf 'Unknown command: %s\n\n' "${COMMAND}" >&2
    usage >&2
    exit 2
    ;;
esac
