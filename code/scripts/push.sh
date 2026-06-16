#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/push.sh "commit message" [--workspace|--code-only]

Options:
  --workspace   Stage the whole Git worktree. This is the default.
  --code-only   Stage only the code/ directory that contains this script.
  -h, --help    Show this help.

Examples:
  bash scripts/push.sh "Reorganize project layout"
  bash scripts/push.sh "Update simulator code" --code-only
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

scope="workspace"
commit_message=""

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --workspace)
      scope="workspace"
      ;;
    --code-only)
      scope="code"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "${commit_message}" ]]; then
        printf 'Commit message was provided more than once.\n\n' >&2
        usage >&2
        exit 2
      fi
      commit_message="$1"
      ;;
  esac
  shift
done

if [[ -z "${commit_message}" ]]; then
  printf 'Missing commit message.\n\n' >&2
  usage >&2
  exit 2
fi

if ! git -C "${CODE_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  printf 'Not inside a Git repository: %s\n' "${CODE_ROOT}" >&2
  exit 1
fi

REPO_ROOT="$(git -C "${CODE_ROOT}" rev-parse --show-toplevel)"
BRANCH="$(git -C "${REPO_ROOT}" branch --show-current)"

if [[ -z "${BRANCH}" ]]; then
  printf 'Detached HEAD. Check out a branch before committing.\n' >&2
  exit 1
fi

case "${scope}" in
  workspace)
    STAGE_PATH="${REPO_ROOT}"
    ;;
  code)
    STAGE_PATH="${CODE_ROOT}"
    ;;
  *)
    printf 'Internal error: unknown scope %s\n' "${scope}" >&2
    exit 1
    ;;
esac

printf 'Repository: %s\n' "${REPO_ROOT}"
printf 'Branch: %s\n' "${BRANCH}"
printf 'Staging: %s\n' "${STAGE_PATH}"

git -C "${REPO_ROOT}" add -A -- "${STAGE_PATH}"

if git -C "${REPO_ROOT}" diff --cached --quiet; then
  printf 'No staged changes to commit.\n'
else
  git -C "${REPO_ROOT}" commit -m "${commit_message}"
fi

if git -C "${REPO_ROOT}" rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
  git -C "${REPO_ROOT}" push
elif git -C "${REPO_ROOT}" remote get-url origin >/dev/null 2>&1; then
  git -C "${REPO_ROOT}" push -u origin "${BRANCH}"
else
  printf 'No upstream branch or origin remote is configured. Commit is local only.\n' >&2
  printf 'Add a remote, then run: git push -u origin %s\n' "${BRANCH}" >&2
fi
