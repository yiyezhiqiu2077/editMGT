#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../setup/common.sh"
if [[ -z "${DERIVED_ROOT:-}" ]]; then
  echo "DERIVED_ROOT_REQUIRED" >&2
  exit 2
fi
if [[ -z "${INTEREDIT_ROOT:-}" || ! -d "$INTEREDIT_ROOT" ]]; then
  echo "INTEREDIT_ROOT_NOT_FOUND" >&2
  exit 3
fi
metadata=("$INTEREDIT_ROOT"/metadata/*.jsonl.gz)
if [[ ! -e "${metadata[0]}" ]]; then echo "INTEREDIT_METADATA_NOT_FOUND" >&2; exit 4; fi
source_count="$(find "$INTEREDIT_ROOT/source_shards" -type f -name '*.tar' | wc -l)"
asset_count="$(find "$INTEREDIT_ROOT/asset_shards" -type f -name '*.tar' | wc -l)"
if [[ "$source_count" -eq 0 || "$asset_count" -eq 0 ]]; then echo "INTEREDIT_SHARDS_NOT_FOUND" >&2; exit 5; fi
echo "EXISTING_VALID metadata=${#metadata[@]} source_shards=$source_count asset_shards=$asset_count disk=$(du -sh "$INTEREDIT_ROOT" | cut -f1)"
mkdir -p "$DERIVED_ROOT/interedit"
uv run python scripts/data/audit_interedit.py "${metadata[@]}" --output "$DERIVED_ROOT/interedit/audit.json"
