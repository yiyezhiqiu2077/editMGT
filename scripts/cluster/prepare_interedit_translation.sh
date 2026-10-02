#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../setup/common.sh"
if [[ -z "${INTEREDIT_ROOT:-}" || ! -d "$INTEREDIT_ROOT" ]]; then echo "INTEREDIT_ROOT_NOT_FOUND" >&2; exit 3; fi
if [[ -z "${TRANSLATOR_MODEL_ROOT:-}" || ! -d "$TRANSLATOR_MODEL_ROOT" ]]; then echo "TRANSLATOR_MODEL_NOT_FOUND" >&2; exit 4; fi
if [[ -z "${TRANSLATOR_REVISION:-}" ]]; then echo "TRANSLATOR_IMMUTABLE_REVISION_REQUIRED" >&2; exit 5; fi
if [[ -z "${DERIVED_ROOT:-}" ]]; then echo "DERIVED_ROOT_REQUIRED" >&2; exit 6; fi
metadata=("$INTEREDIT_ROOT"/metadata/*.jsonl.gz)
derived="$DERIVED_ROOT/interedit"
mkdir -p "$derived"
uv run python scripts/data/translate_interedit.py "${metadata[@]}" \
  --cache "$derived/translation_cache.jsonl" \
  --qa-output "$derived/translation_qa_flags.jsonl" \
  --backend nllb --model-path "$TRANSLATOR_MODEL_ROOT" \
  --revision "$TRANSLATOR_REVISION" --src-lang zho_Hans --tgt-lang eng_Latn --formal
uv run python scripts/data/build_interedit_manifest.py "${metadata[@]}" \
  --cache "$derived/translation_cache.jsonl" \
  --output "$derived/train_manifest.jsonl" \
  --backend nllb --revision "$TRANSLATOR_REVISION"
uv run python scripts/data/freeze_group_splits.py \
  --input "$derived/train_manifest.jsonl" --output-dir "$derived/splits" \
  --group-key source_id --validation-fraction 0.05 --seed 42
magic_splits="$DERIVED_ROOT/magicbrush-splits"
uv run python scripts/data/freeze_group_splits.py \
  --input "$MAGICBRUSH_ROOT/manifest.jsonl" --output-dir "$magic_splits" \
  --group-key img_id --validation-fraction 0.05 --seed 42
