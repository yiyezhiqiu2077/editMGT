#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../setup/common.sh"
config=""
mode=""
extra=()
preflight=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) [[ $# -ge 2 ]] || exit 2; config="$2"; shift 2 ;;
    --print-command|--run) [[ -z "$mode" ]] || exit 2; mode="$1"; shift ;;
    --resume) [[ $# -ge 2 ]] || exit 2; extra+=(--resume "$2"); preflight+=(--resume "$2"); shift 2 ;;
    --stop-after-steps) [[ $# -ge 2 ]] || exit 2; extra+=(--stop-after-steps "$2"); shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$config" || -z "$mode" ]]; then
  echo "usage: $0 --config CONFIG --print-command|--run [--resume CHECKPOINT] [--stop-after-steps N]" >&2
  exit 2
fi
port="$(pick_main_process_port)"
cmd=(uv run torchrun --nnodes 1 --nproc_per_node 8 --master_addr 127.0.0.1 --master_port "$port"
     scripts/train/train_dimo_editing.py --config "$config" --experiment-mode pilot_8gpu "${extra[@]}")
if [[ "$mode" == --print-command ]]; then
  uv run python scripts/train/preflight_dimo_8g.py --config "$config" --contract-only
  printf '%q ' "${cmd[@]}"
  printf '\n'
else
  uv run python scripts/train/preflight_dimo_8g.py --config "$config" "${preflight[@]}"
  exec "${cmd[@]}"
fi
