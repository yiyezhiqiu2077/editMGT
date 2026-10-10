#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mode=${1:-}
[[ "$mode" == --print-command || "$mode" == --run ]] || {
  echo 'usage: run_full_dense_dimo.sh --print-command|--run [--resume CHECKPOINT]' >&2
  exit 2
}
shift
resume=()
if [[ $# -gt 0 ]]; then
  [[ $# -eq 2 && "$1" == --resume ]] || exit 2
  resume=(--resume "$2")
fi
command=(env -u GH_TOKEN uv run torchrun --standalone --nnodes=1 --nproc-per-node=8
  scripts/train/train_dense_teacher_dimo.py --config configs/dimo/full_dense_d200k.yaml "${resume[@]}")
if [[ "$mode" == --print-command ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
