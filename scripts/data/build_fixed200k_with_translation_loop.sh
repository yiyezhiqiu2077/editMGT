#!/usr/bin/env bash
set -euo pipefail
: "${DERIVED_ROOT:?}" "${TRANSLATOR_MODEL_ROOT:?}" "${TRANSLATOR_REVISION:?}"
fixed="$DERIVED_ROOT/fixed200k"
mkdir -p "$fixed"
if [[ ! -e "$fixed/translation_cache.jsonl" ]]; then
  : > "$fixed/translation_cache.jsonl"
fi
mode="${1:---train}"
if [[ "$mode" == "--validation" ]]; then
  request_root="$fixed/validation"
  builder=(uv run --frozen python scripts/data/freeze_fixed200k_validation.py
    --config configs/data/fixed_200k.yaml
    --magicbrush-dev "$fixed/pools/magicbrush_dev.jsonl"
    --crispedit-pool "$fixed/pools/crispedit.jsonl"
    --scaleedit-pool "$fixed/pools/scaleedit.jsonl"
    --interedit-pool "$fixed/pools/interedit.jsonl"
    --translation-cache "$fixed/translation_cache.jsonl" --output-dir "$request_root")
elif [[ "$mode" == "--train" ]]; then
  request_root="$fixed"
  builder=(uv run --frozen python scripts/data/build_fixed_200k_corpus.py --config configs/data/fixed_200k.yaml
  --magicbrush-pool "$fixed/pools/magicbrush_train.jsonl"
  --magicbrush-root "$MAGICBRUSH_ROOT"
  --crispedit-pool "$fixed/pools/crispedit.jsonl"
  --scaleedit-pool "$fixed/pools/scaleedit.jsonl"
  --interedit-pool "$fixed/pools/interedit.jsonl"
  --validation "$fixed/validation/magicbrush_official_dev.jsonl"
  --validation "$fixed/validation/crispedit_aux128.jsonl"
  --validation "$fixed/validation/scaleedit_aux128.jsonl"
  --validation "$fixed/validation/interedit_aux128.jsonl"
  --translation-cache "$fixed/translation_cache.jsonl" --output-dir "$fixed")
else
  printf 'usage: %s [--validation|--train]\n' "$0" >&2
  exit 2
fi
for iteration in $(seq 1 32); do
  set +e
  "${builder[@]}"
  status=$?
  set -e
  if [[ "$status" -eq 0 ]]; then exit 0; fi
  if [[ "$status" -ne 42 ]]; then exit "$status"; fi
  uv run --frozen python scripts/data/translate_canonical_candidates.py \
    --required "$request_root/translation_required.jsonl" \
    --cache "$fixed/translation_cache.jsonl" \
    --qa-output "$request_root/translation_qa_flags.jsonl" \
    --backend nllb --model-path "$TRANSLATOR_MODEL_ROOT" \
    --revision "$TRANSLATOR_REVISION" --src-lang zho_Hans --tgt-lang eng_Latn --formal
done
echo "translation/backfill did not converge after 32 iterations" >&2
exit 9
