#!/usr/bin/env bash
set -euo pipefail
: "${DERIVED_ROOT:?}"
fixed="$DERIVED_ROOT/fixed200k"
touch "$fixed/translation_cache.jsonl"
builder=(uv run python scripts/data/build_fixed_200k_corpus.py --config configs/data/fixed_200k.yaml
  --magicbrush-pool "$fixed/pools/magicbrush_train.jsonl" --crispedit-pool "$fixed/pools/crispedit.jsonl"
  --scaleedit-pool "$fixed/pools/scaleedit.jsonl" --interedit-pool "$fixed/pools/interedit.jsonl"
  --validation "$fixed/validation/magicbrush_official_dev.jsonl" --validation "$fixed/validation/crispedit_aux128.jsonl"
  --validation "$fixed/validation/scaleedit_aux128.jsonl" --validation "$fixed/validation/interedit_aux128.jsonl"
  --translation-cache "$fixed/translation_cache.jsonl" --output-dir "$fixed")
set +e
"${builder[@]}"
status=$?
set -e
if [[ "$status" != 0 && "$status" != 42 ]]; then exit "$status"; fi
[[ "$status" == 0 || -s "$fixed/translation_required.jsonl" ]] || { echo "selection produced no frozen corpus or translation request" >&2; exit 7; }
uv run python - <<'PY' "$fixed/selection_stage.json" "$status"
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"status":"FROZEN" if sys.argv[2]=="0" else "TRANSLATION_REQUIRED","builder_exit":int(sys.argv[2])},sort_keys=True)+"\n")
PY
