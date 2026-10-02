#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../setup/common.sh"
mode="${1:---print-command}"
: "${DERIVED_ROOT:?}" "${MAGICBRUSH_ROOT:?}" "${MAGICBRUSH_DEV_ROOT:?}" "${CRISPEDIT_ROOT:?}" "${SCALEEDIT_ROOT:?}" "${INTEREDIT_ROOT:?}" "${EDITMGT_MODEL_ROOT:?}" "${TRANSLATOR_MODEL_ROOT:?}" "${TRANSLATOR_REVISION:?}"
: "${MAGICBRUSH_REVISION:?}" "${CRISPEDIT_REVISION:?}" "${SCALEEDIT_REVISION:?}" "${INTEREDIT_REVISION:?}" "${CRISPEDIT_SCHEMA_MAPPING:?}" "${SCALEEDIT_SCHEMA_MAPPING:?}"
fixed="$DERIVED_ROOT/fixed200k"
commands=(
  "uv run python scripts/data/audit_raw_dataset_schema.py --dataset-name magicbrush --root $MAGICBRUSH_ROOT --revision $MAGICBRUSH_REVISION --output $fixed/schema_audit/magicbrush.json"
  "uv run python scripts/data/audit_raw_dataset_schema.py --dataset-name crispedit --root $CRISPEDIT_ROOT --revision $CRISPEDIT_REVISION --output $fixed/schema_audit/crispedit.json"
  "uv run python scripts/data/audit_raw_dataset_schema.py --dataset-name scaleedit --root $SCALEEDIT_ROOT --revision $SCALEEDIT_REVISION --output $fixed/schema_audit/scaleedit.json"
  "uv run python scripts/data/audit_raw_dataset_schema.py --dataset-name interedit --root $INTEREDIT_ROOT --revision $INTEREDIT_REVISION --output $fixed/schema_audit/interedit.json"
  "uv run python scripts/data/build_magicbrush_canonical.py --manifest $MAGICBRUSH_ROOT/manifest.jsonl --dataset-root $MAGICBRUSH_ROOT --revision $MAGICBRUSH_REVISION --output $fixed/pools/magicbrush_train.jsonl"
  "uv run python scripts/data/build_magicbrush_canonical.py --manifest $MAGICBRUSH_DEV_ROOT/manifest.jsonl --dataset-root $MAGICBRUSH_DEV_ROOT --revision $MAGICBRUSH_REVISION --output $fixed/pools/magicbrush_dev.jsonl"
  "uv run python scripts/data/build_schema_mapped_canonical.py --root $CRISPEDIT_ROOT --mapping $CRISPEDIT_SCHEMA_MAPPING --edit-type-mapping configs/data/edit_type_mapping.yaml --output $fixed/pools/crispedit.jsonl"
  "uv run python scripts/data/build_schema_mapped_canonical.py --root $SCALEEDIT_ROOT --mapping $SCALEEDIT_SCHEMA_MAPPING --edit-type-mapping configs/data/edit_type_mapping.yaml --output $fixed/pools/scaleedit.jsonl"
  "uv run python scripts/data/build_interedit_canonical_pool.py $INTEREDIT_ROOT/metadata/*.jsonl.gz --interedit-root $INTEREDIT_ROOT --revision $INTEREDIT_REVISION --output $fixed/pools/interedit.jsonl"
  "uv run python scripts/data/freeze_fixed200k_validation.py --magicbrush-dev $fixed/pools/magicbrush_dev.jsonl --crispedit-pool $fixed/pools/crispedit.jsonl --scaleedit-pool $fixed/pools/scaleedit.jsonl --interedit-pool $fixed/pools/interedit.jsonl --output-dir $fixed/validation"
  "bash scripts/data/build_fixed200k_with_translation_loop.sh"
  "uv run python scripts/data/audit_fixed200k.py --manifest $fixed/train_200k.jsonl --magicbrush-root $MAGICBRUSH_ROOT --crispedit-root $CRISPEDIT_ROOT --scaleedit-root $SCALEEDIT_ROOT --interedit-root $INTEREDIT_ROOT --model-root $EDITMGT_MODEL_ROOT --output-dir $fixed/audit"
  "uv run python scripts/data/build_translation_manual_audit.py --manifest $fixed/train_200k.jsonl --output-jsonl $fixed/translation_manual_audit_500.jsonl --output-csv $fixed/translation_manual_audit_500.csv"
  "uv run python scripts/data/finalize_corpus_ready.py --fixed-root $fixed --selection-config configs/data/fixed_200k.yaml --translation-cache $fixed/translation_cache.jsonl --git-sha $(git rev-parse HEAD)"
  "uv run python scripts/data/verify_corpus_ready.py $fixed/CORPUS_READY.json"
)
if [[ "$mode" == "--print-command" ]]; then
  printf '%s\n' "${commands[@]}"
elif [[ "$mode" == "--run" ]]; then
  [[ "${CONFIRM_CORPUS_BUILD:-}" == "YES" ]] || { echo "CONFIRM_CORPUS_BUILD=YES required" >&2; exit 3; }
  mkdir -p "$fixed/schema_audit" "$fixed/pools" "$fixed/validation" "$fixed/audit"
  for command in "${commands[@]}"; do bash -lc "$command"; done
else
  echo "usage: $0 --print-command|--run" >&2; exit 2
fi
