#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mode=${1:-}
if [[ "$mode" != --print-command && "$mode" != --run ]]; then
  echo "usage: $0 --print-command|--run [--resume CHECKPOINT]" >&2
  exit 2
fi
shift
resume=()
if [[ $# -gt 0 ]]; then
  [[ $# -eq 2 && "$1" == --resume ]] || exit 2
  resume=(--resume "$2")
fi
uv run python scripts/train/verify_e3_dense_config.py --static
command=(uv run accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16
  --gpu_ids 0,1,2,3,4,5,6,7 --main_process_port "${DENSE_MASTER_PORT:-29671}"
  scripts/train/train_dense_region.py --config configs/train/dense_d200k_5epoch.yaml "${resume[@]}")
if [[ "$mode" == --print-command ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
[[ ${CONFIRM_DENSE_RUN:-NO} == YES ]] || { echo 'Set CONFIRM_DENSE_RUN=YES' >&2; exit 3; }
uv run python scripts/train/verify_e3_dense_config.py "${resume[@]}"
exec "${command[@]}"
