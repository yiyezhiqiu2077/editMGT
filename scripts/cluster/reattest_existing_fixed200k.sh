#!/usr/bin/env bash
set -euo pipefail
: "${EDITMGT_RUN_ROOT:?existing RUN_ROOT required}"
: "${EDITMGT_ASSET_ROOT:?existing asset root required}"
: "${EDITMGT_MODEL_ROOT:?released model snapshot required}"
source "$(dirname "$0")/../setup/common.sh"
exec uv run python scripts/data/reattest_existing_fixed200k.py \
  --run-root "$EDITMGT_RUN_ROOT" --asset-root "$EDITMGT_ASSET_ROOT" \
  --model-root "$EDITMGT_MODEL_ROOT"
