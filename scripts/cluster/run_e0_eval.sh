#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../setup/common.sh"
mode="${1:---print-command}"
: "${DERIVED_ROOT:?DERIVED_ROOT required}" "${EDITMGT_OUTPUT_ROOT:?EDITMGT_OUTPUT_ROOT required}" "${EDITMGT_MODEL_ROOT:?EDITMGT_MODEL_ROOT required}" "${INTEREDIT_ROOT:?INTEREDIT_ROOT required}"
commands=()
for named_protocol in official:official_upstream_timestep region:roi_relative;do
 label="${named_protocol%%:*}";protocol="${named_protocol#*:}"
 for dataset in magicbrush interedit;do
  if [[ "$dataset" == magicbrush ]];then manifest="$DERIVED_ROOT/magicbrush-splits/validation.jsonl";extra=();else manifest="$DERIVED_ROOT/interedit/splits/validation.jsonl";extra=(--interedit-root "$INTEREDIT_ROOT");fi
  run="$EDITMGT_OUTPUT_ROOT/e0-${label}-${dataset}"
  commands+=("uv run python scripts/eval/generate_evaluation.py --dataset $dataset --manifest $manifest ${extra[*]} --model-root $EDITMGT_MODEL_ROOT --config configs/eval/formal.yaml --timestep-mode $protocol --output-dir $run")
  commands+=("uv run python scripts/eval/formal_eval.py --experiment E0 --checkpoint released --config configs/eval/formal.yaml --predictions-manifest $run/predictions.jsonl --output $run/metrics.json")
 done
done
if [[ "$mode" == --print-command ]];then printf '%s\n' "${commands[@]}";else [[ "${CONFIRM_FORMAL_RUN:-}" == YES ]]||exit 3;for command in "${commands[@]}";do bash -lc "$command";done;fi
