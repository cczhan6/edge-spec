#!/usr/bin/env bash
set -euo pipefail

# Xet can stall behind local SOCKS proxies. Plain HTTP/LFS is slower sometimes
# but resumes cleanly and is more reliable for large Qwen checkpoints.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

DATASET_PATH="${DATASET_PATH:-data/spec_bench/question.jsonl}"
DATASET_MODE="${DATASET_MODE:-all}"
CATEGORY="${CATEGORY:-Sum}"
PROFILE_CONFIG="${PROFILE_CONFIG:-configs/edge_hetero.yaml}"
RESULTS_DIR="${RESULTS_DIR:-results/sync_batch/${CATEGORY}}"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DRAFT_MODEL_0="${DRAFT_MODEL_0:-Qwen/Qwen2.5-0.5B-Instruct}"
DRAFT_MODEL_1="${DRAFT_MODEL_1:-Qwen/Qwen2.5-1.5B-Instruct}"
DRAFT_MODEL_2="${DRAFT_MODEL_2:-Qwen/Qwen2.5-3B-Instruct}"
CLIENT_DEVICE="${CLIENT_DEVICE:-cuda:0}"
SERVER_DEVICE="${SERVER_DEVICE:-cuda:1}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"

GAMMA="${GAMMA:-4}"
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
  --method sync_batch \
  --target-model "${TARGET_MODEL}" \
  --draft-models "${DRAFT_MODEL_0}" "${DRAFT_MODEL_1}" "${DRAFT_MODEL_2}" \
  --dataset-path "${DATASET_PATH}" \
  --dataset-mode "${DATASET_MODE}" \
  --category "${CATEGORY}" \
  --profile-config "${PROFILE_CONFIG}" \
  --results-dir "${RESULTS_DIR}" \
  --gamma "${GAMMA}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --top-k "${TOP_K}" \
  --seed "${SEED}" \
  --network-seed "${NETWORK_SEED}" \
  --network-trace-slot-s "${NETWORK_TRACE_SLOT_S}" \
  --client-device "${CLIENT_DEVICE}" \
  --server-device "${SERVER_DEVICE}" \
  --torch-dtype "${TORCH_DTYPE}"
