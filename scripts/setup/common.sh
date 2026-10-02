#!/usr/bin/env bash
set -euo pipefail

actual_root="$(git rev-parse --show-toplevel)"
EXPECTED_WORKTREE="${EDITMGT_WORKTREE:-$actual_root}"
if [[ "$(realpath "$actual_root")" != "$(realpath "$EXPECTED_WORKTREE")" ]]; then
  echo "WRONG_WORKTREE expected=$EXPECTED_WORKTREE actual=$actual_root" >&2
  exit 2
fi
