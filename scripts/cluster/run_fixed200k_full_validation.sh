#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../setup/common.sh"
mode="${1:---print-command}"
: "${DERIVED_ROOT:?}" "${EDITMGT_OUTPUT_ROOT:?}" "${EDITMGT_MODEL_ROOT:?}" "${MAGICBRUSH_DEV_ROOT:?}" "${CRISPEDIT_ROOT:?}" "${SCALEEDIT_ROOT:?}" "${INTEREDIT_ROOT:?}" "${CANDIDATE_CHECKPOINT:?}"
validation="$DERIVED_ROOT/fixed200k/validation";run="$EDITMGT_OUTPUT_ROOT/full-validation/$(basename "$CANDIDATE_CHECKPOINT")"
commands=("uv run python scripts/data/verify_corpus_ready.py $DERIVED_ROOT/fixed200k/CORPUS_READY.json")
for spec in "magicbrush:$MAGICBRUSH_DEV_ROOT:magicbrush_official_dev.jsonl" "crispedit:$CRISPEDIT_ROOT:crispedit_aux128.jsonl" "scaleedit:$SCALEEDIT_ROOT:scaleedit_aux128.jsonl" "interedit:$INTEREDIT_ROOT:interedit_aux128.jsonl"; do
  dataset="${spec%%:*}";rest="${spec#*:}";root="${rest%%:*}";manifest="${rest#*:}";destination="$run/$dataset"
  commands+=("uv run python scripts/eval/generate_evaluation.py --dataset $dataset --manifest $validation/$manifest --canonical-root $root --model-root $EDITMGT_MODEL_ROOT --checkpoint $CANDIDATE_CHECKPOINT --config configs/eval/formal.yaml --timestep-mode roi_relative --output-dir $destination")
  commands+=("uv run python scripts/eval/formal_eval.py --experiment fixed200k-validation-$dataset --checkpoint $CANDIDATE_CHECKPOINT --config configs/eval/formal.yaml --predictions-manifest $destination/predictions.jsonl --output $destination/metrics.json")
done
if [[ "$mode" == "--print-command" ]];then printf '%s\n' "${commands[@]}";elif [[ "$mode" == "--run" ]];then [[ "${CONFIRM_FORMAL_RUN:-}" == YES ]]||exit 3;for command in "${commands[@]}";do bash -lc "$command";done;else exit 2;fi
