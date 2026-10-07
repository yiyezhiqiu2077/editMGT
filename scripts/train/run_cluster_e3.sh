#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../setup/common.sh"
if [[ $# -lt 1 || ( "$1" != "--print-command" && "$1" != "--run" ) ]]; then
  echo "usage: $0 --print-command|--run [--resume CHECKPOINT]" >&2
  exit 2
fi
uv run python scripts/train/verify_e3_five_epoch_config.py
if [[ "$1" == "--run" && -n "$(git status --porcelain)" ]]; then
  echo "CURRENT_CODE_NOT_CLEAN: use the exact pushed SHA" >&2
  exit 3
fi
exec bash scripts/train/run_formal.sh configs/train/cluster_8g_e3.yaml "$@"
