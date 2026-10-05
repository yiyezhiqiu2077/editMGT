#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:---print-command}"
COMMAND=(uv run python scripts/train/train_dimo_editing.py --config configs/dimo/prep_smoke.yaml --prep-smoke)

if [[ "$MODE" == "--print-command" ]]; then
  printf 'cd %q && CUDA_VISIBLE_DEVICES=0' "$ROOT"
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
elif [[ "$MODE" == "--run" ]]; then
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES=0 "${COMMAND[@]}"
else
  echo "usage: $0 [--print-command|--run]" >&2
  exit 2
fi
