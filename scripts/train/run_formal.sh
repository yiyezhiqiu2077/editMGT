#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../setup/common.sh"
if [[ $# -ne 2 || ( "$2" != "--print-command" && "$2" != "--run" ) ]]; then
  echo "usage: $0 CONFIG --print-command|--run" >&2
  exit 2
fi
config="$1"
mode="$2"
cmd=(uv run accelerate launch --multi_gpu --num_processes 8 --gpu_ids 0,1,2,3,4,5,6,7 scripts/train/train_explicit_region.py --config "$config")
if [[ "$mode" == "--print-command" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
else
  if [[ "${CONFIRM_FORMAL_RUN:-}" != "YES" ]]; then echo "CONFIRM_FORMAL_RUN=YES required" >&2; exit 3; fi
  : "${DERIVED_ROOT:?DERIVED_ROOT required}"
  uv run python scripts/data/verify_corpus_ready.py "$DERIVED_ROOT/fixed200k/CORPUS_READY.json"
  "${cmd[@]}"
fi
