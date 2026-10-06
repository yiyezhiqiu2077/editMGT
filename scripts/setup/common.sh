#!/usr/bin/env bash
set -euo pipefail

actual_root="$(git rev-parse --show-toplevel)"
EXPECTED_WORKTREE="${EDITMGT_WORKTREE:-$actual_root}"
if [[ "$(realpath "$actual_root")" != "$(realpath "$EXPECTED_WORKTREE")" ]]; then
  echo "WRONG_WORKTREE expected=$EXPECTED_WORKTREE actual=$actual_root" >&2
  exit 2
fi

ensure_short_tmpdir() {
  local tmpdir="${TMPDIR:-}"
  local run_slug="$(basename "${RUN_ROOT:-editmgt-run}")"
  local short_tmp="/tmp/${run_slug:0:48}"
  if [[ -z "$tmpdir" || ${#tmpdir} -gt 80 ]]; then
    tmpdir="$short_tmp"
  fi
  mkdir -p "$tmpdir"
  export TMPDIR="$tmpdir"
  export TMP="$TMPDIR"
  export TEMP="$TMPDIR"
}

ensure_short_tmpdir
