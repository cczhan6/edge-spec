#!/usr/bin/env bash
set -euo pipefail

# Xet can stall behind local SOCKS proxies. Plain HTTP/LFS is slower sometimes
# but resumes cleanly and is more reliable for large Qwen checkpoints.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
DATASET_MODE="${DATASET_MODE:-all}"
CATEGORY="${CATEGORY:-Sum}"
RESULTS_DIR="${RESULTS_DIR:-results/sync/${CATEGORY}}"

python -m edge_spec.run \
  --target-model Qwen/Qwen2.5-7B-Instruct \
  --draft-models \
    Qwen/Qwen2.5-0.5B-Instruct \
    Qwen/Qwen2.5-1.5B-Instruct \
    Qwen/Qwen2.5-3B-Instruct \
  --dataset-path data/spec_bench/question.jsonl \
  --dataset-mode "${DATASET_MODE}" \
  --category "${CATEGORY}" \
  --profile-config configs/edge_hetero.yaml \
  --results-dir "${RESULTS_DIR}" \
  --gamma 4 \
  --max-new-tokens 256 \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --client-device cuda:0 \
  --server-device cuda:1
  #--skip-target-baseline \
