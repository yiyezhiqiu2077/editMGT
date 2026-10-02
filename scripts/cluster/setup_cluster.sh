#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../setup/common.sh"
if [[ -z "${INTEREDIT_ROOT:-}" || ! -d "$INTEREDIT_ROOT" ]]; then
  echo "INTEREDIT_ROOT_NOT_FOUND" >&2
  exit 3
fi
bash scripts/setup_local_assets.sh interedit "$INTEREDIT_ROOT"
bash scripts/setup/bootstrap_uv.sh
uv run python scripts/setup/audit_assets.py
