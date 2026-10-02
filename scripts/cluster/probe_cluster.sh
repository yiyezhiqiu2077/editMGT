#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../setup/common.sh"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader
if [[ -z "${INTEREDIT_ROOT:-}" || ! -d "$INTEREDIT_ROOT" ]]; then
  echo "INTEREDIT_ROOT_NOT_FOUND" >&2
  exit 3
fi
echo "INTEREDIT_ROOT=$(realpath "$INTEREDIT_ROOT")"
du -sh "$INTEREDIT_ROOT"
find "$INTEREDIT_ROOT/metadata" -type f -name '*.jsonl.gz' | wc -l
find "$INTEREDIT_ROOT/source_shards" -type f -name '*.tar' | wc -l
find "$INTEREDIT_ROOT/asset_shards" -type f -name '*.tar' | wc -l
