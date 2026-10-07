#!/usr/bin/env bash
set -euo pipefail
: "${DERIVED_ROOT:?}" "${TRANSLATOR_MODEL_ROOT:?}" "${TRANSLATOR_REVISION:?}"
fixed="$DERIVED_ROOT/fixed200k"
cache="$fixed/translation_cache.jsonl"
mkdir -p "$fixed/validation"
touch "$cache"
command=(uv run python scripts/data/freeze_fixed200k_validation.py
  --magicbrush-dev "$fixed/pools/magicbrush_dev.jsonl"
  --crispedit-pool "$fixed/pools/crispedit.jsonl"
  --scaleedit-pool "$fixed/pools/scaleedit.jsonl"
  --interedit-pool "$fixed/pools/interedit.jsonl"
  --translation-cache "$cache"
  --output-dir "$fixed/validation")
for _iteration in $(seq 1 32); do
  set +e
  "${command[@]}"
  status=$?
  set -e
  if [[ "$status" -eq 0 ]]; then
    exit 0
  fi
  if [[ "$status" -ne 42 ]]; then
    exit "$status"
  fi
  uv run python scripts/data/translate_canonical_candidates.py \
    --required "$fixed/validation/translation_required.jsonl" \
    --cache "$cache" \
    --qa-output "$fixed/validation/translation_qa_flags.jsonl" \
    --backend nllb --model-path "$TRANSLATOR_MODEL_ROOT" \
    --revision "$TRANSLATOR_REVISION" --src-lang zho_Hans --tgt-lang eng_Latn --formal
done
echo "validation translation/backfill did not converge after 32 iterations" >&2
exit 9
