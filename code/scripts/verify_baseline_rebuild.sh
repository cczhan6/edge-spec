#!/usr/bin/env bash
set -euo pipefail

pytest -q

pytest -q \
  tests/test_target_only.py \
  tests/test_linear_sd_core.py \
  tests/test_server_only_linear.py \
  tests/test_specedge_linear.py \
  tests/test_dip_sd.py \
  tests/test_server_only_tree.py \
  tests/test_specedge_tree.py

if grep -Rni \
  --exclude-dir=.git \
  --exclude-dir=__pycache__ \
  --exclude='*.md' \
  --exclude='*.pyc' \
  -E 'include_prefill|draft_prefill|target_prefill|prefill_latency' \
  src configs tests; then
    echo "Unexpected prefill execution code found"
    exit 1
fi
