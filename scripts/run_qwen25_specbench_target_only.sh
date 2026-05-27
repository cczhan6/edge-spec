#!/usr/bin/env bash
set -euo pipefail

# Xet can stall behind local SOCKS proxies. Plain HTTP/LFS is slower sometimes
# but resumes cleanly and is more reliable for large Qwen checkpoints.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

DATASET_PATH="${DATASET_PATH:-data/spec_bench/question.jsonl}"
DATASET_MODE="${DATASET_MODE:-all}"
CATEGORY="${CATEGORY:-Sum}"
PROFILE_CONFIG="${PROFILE_CONFIG:-configs/edge_hetero.yaml}"
RESULTS_DIR="${RESULTS_DIR:-results/target_only/${CATEGORY}}"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SERVER_DEVICE="${SERVER_DEVICE:-cuda:1}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"

SEED="${SEED:-42}"
NETWORK_SEED="${NETWORK_SEED:-${SEED}}"
NETWORK_TRACE_SLOT_S="${NETWORK_TRACE_SLOT_S:-0.05}"

case "${CATEGORY}" in
  Sum|Math|MT|QA|RAG|Trans)
    ;;
  *)
    echo "Unsupported CATEGORY=${CATEGORY}. Use one of: Sum, Math, MT, QA, RAG, Trans." >&2
    exit 2
    ;;
esac

python -m edge_spec.run \
  --method target_only \
  --target-model "${TARGET_MODEL}" \
  --dataset-path "${DATASET_PATH}" \
  --dataset-mode "${DATASET_MODE}" \
  --category "${CATEGORY}" \
  --profile-config "${PROFILE_CONFIG}" \
  --results-dir "${RESULTS_DIR}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --top-k "${TOP_K}" \
  --seed "${SEED}" \
  --network-seed "${NETWORK_SEED}" \
  --network-trace-slot-s "${NETWORK_TRACE_SLOT_S}" \
  --server-device "${SERVER_DEVICE}" \
  --torch-dtype "${TORCH_DTYPE}"
