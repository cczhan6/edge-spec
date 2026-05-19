#!/usr/bin/env bash
set -euo pipefail

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not a git repository." >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/push.sh \"commit message\"" >&2
  exit 1
fi

commit_message="$1"

git add .

git commit -m "$commit_message"

git push
