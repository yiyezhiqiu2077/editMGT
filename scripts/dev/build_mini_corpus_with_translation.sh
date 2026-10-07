#!/usr/bin/env bash
set -euo pipefail

: "${MINI_ROOT:?MINI_ROOT required}"
: "${TRANSLATOR_MODEL_ROOT:?TRANSLATOR_MODEL_ROOT required}"
: "${TRANSLATOR_REVISION:?TRANSLATOR_REVISION required}"

corpus="$MINI_ROOT/corpus"
validation="$corpus/validation"
cache="$corpus/translation_cache.jsonl"
mkdir -p "$validation"
touch "$cache"

translate_required() {
  local required="$1"
  local qa_output="$2"
  uv run python scripts/data/translate_canonical_candidates.py \
    --required "$required" --cache "$cache" --qa-output "$qa_output" \
    --backend nllb --model-path "$TRANSLATOR_MODEL_ROOT" \
    --revision "$TRANSLATOR_REVISION" --src-lang zho_Hans --tgt-lang eng_Latn --formal
}

validation_command=(uv run python scripts/data/freeze_fixed200k_validation.py
  --config configs/dev/four_dataset_512.yaml
  --magicbrush-dev "$MINI_ROOT/pools/magicbrush_dev.jsonl"
  --crispedit-pool "$MINI_ROOT/pools/crispedit.jsonl"
  --scaleedit-pool "$MINI_ROOT/pools/scaleedit.jsonl"
  --interedit-pool "$MINI_ROOT/pools/interedit.jsonl"
  --translation-cache "$cache" --output-dir "$validation" --count 8)

validation_converged=0
for _iteration in $(seq 1 32); do
  set +e
  "${validation_command[@]}"
  status=$?
  set -e
  if [[ "$status" -eq 0 ]]; then validation_converged=1; break; fi
  [[ "$status" -eq 42 ]] || exit "$status"
  translate_required "$validation/translation_required.jsonl" "$validation/translation_qa_flags.jsonl"
done
[[ "$validation_converged" -eq 1 ]] || { echo "Mini validation did not converge" >&2; exit 9; }

builder=(uv run python scripts/data/build_fixed_200k_corpus.py
  --config configs/dev/four_dataset_512.yaml
  --magicbrush-pool "$MINI_ROOT/pools/magicbrush_train.jsonl"
  --crispedit-pool "$MINI_ROOT/pools/crispedit.jsonl"
  --scaleedit-pool "$MINI_ROOT/pools/scaleedit.jsonl"
  --interedit-pool "$MINI_ROOT/pools/interedit.jsonl"
  --magicbrush-root "$MINI_ROOT/assets/magicbrush/train"
  --crispedit-root "$MINI_ROOT/assets/crispedit"
  --scaleedit-root "$MINI_ROOT/assets/scaleedit"
  --interedit-root "$MINI_ROOT/assets/interedit"
  --validation "$validation/magicbrush_official_dev.jsonl"
  --validation "$validation/crispedit_aux8.jsonl"
  --validation "$validation/scaleedit_aux8.jsonl"
  --validation "$validation/interedit_aux8.jsonl"
  --translation-cache "$cache" --output-dir "$corpus" --allow-nonproduction-total)

corpus_converged=0
for _iteration in $(seq 1 32); do
  set +e
  "${builder[@]}"
  status=$?
  set -e
  if [[ "$status" -eq 0 ]]; then corpus_converged=1; break; fi
  [[ "$status" -eq 42 ]] || exit "$status"
  translate_required "$corpus/translation_required.jsonl" "$corpus/translation_qa_flags.jsonl"
done
[[ "$corpus_converged" -eq 1 ]] || { echo "Mini corpus did not converge" >&2; exit 9; }

# A second warm build must be byte-identical.
first_sha="$(sha256sum "$corpus/train_512.jsonl" | awk '{print $1}')"
"${builder[@]}"
second_sha="$(sha256sum "$corpus/train_512.jsonl" | awk '{print $1}')"
[[ "$first_sha" == "$second_sha" ]] || { echo "MINI_WARM_BUILD_NONDETERMINISTIC" >&2; exit 10; }

uv run python scripts/dev/finalize_mini_corpus.py \
  --manifest "$corpus/train_512.jsonl" \
  --validation "$validation/magicbrush_official_dev.jsonl" \
  --validation "$validation/crispedit_aux8.jsonl" \
  --validation "$validation/scaleedit_aux8.jsonl" \
  --validation "$validation/interedit_aux8.jsonl" \
  --magicbrush-root "$MINI_ROOT/assets/magicbrush/train" \
  --crispedit-root "$MINI_ROOT/assets/crispedit" \
  --scaleedit-root "$MINI_ROOT/assets/scaleedit" \
  --interedit-root "$MINI_ROOT/assets/interedit" \
  --output "$corpus/MINI_CORPUS_READY.json"

printf 'MINI_CORPUS_SHA256=%s\n' "$second_sha"
