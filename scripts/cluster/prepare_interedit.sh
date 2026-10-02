#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../setup/common.sh"
mode="${1:---print-command}"
commands=(
  "bash scripts/cluster/prepare_assets.sh"
  "bash scripts/cluster/prepare_interedit_translation.sh"
)
if [[ "$mode" == "--print-command" ]]; then
  printf '%s\n' "${commands[@]}"
elif [[ "$mode" == "--run" ]]; then
  [[ "${CONFIRM_FORMAL_RUN:-}" == "YES" ]] || {
    echo "CONFIRM_FORMAL_RUN=YES required" >&2
    exit 3
  }
  # These preparation helpers execute directly; only this top-level wrapper
  # owns the dry-run/formal-run switch.
  for command in "${commands[@]}"; do bash -lc "$command"; done
else
  echo "usage: $0 --print-command|--run" >&2
  exit 2
fi
