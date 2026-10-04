#!/usr/bin/env bash
set -euo pipefail
repo_root="$(realpath "$(dirname "$0")/../..")"
cd "$repo_root"
source "$repo_root/scripts/setup/common.sh"
mode="${1:---print-command}"
[[ "$mode" == "--run" || "$mode" == "--print-command" || "$mode" == "--inspect-gates" ]] || exit 2
: "${DERIVED_ROOT:?}" "${EDITMGT_OUTPUT_ROOT:?}" "${EDITMGT_MODEL_ROOT:?}" "${MAGICBRUSH_DEV_ROOT:?}" "${CRISPEDIT_ROOT:?}" "${SCALEEDIT_ROOT:?}" "${INTEREDIT_ROOT:?}" "${CANDIDATE_CHECKPOINT:?}" "${CANDIDATE_EXPERIMENT:?Use fixed name E1-E4 or LR-1e-5/LR-3e-5/LR-5e-5}"
: "${UV_PROJECT_ENVIRONMENT:?Source the frozen RUN_ROOT/scripts/env.sh first}"
python="$UV_PROJECT_ENVIRONMENT/bin/python"
validation="$DERIVED_ROOT/fixed200k/validation"
gate=("$python" -m src.explicit_region.gates --stage eval --corpus-ready "$DERIVED_ROOT/fixed200k/CORPUS_READY.json")
if [[ "$mode" == "--inspect-gates" ]]; then "${gate[@]}"; exit; fi
run_or_print() {
  if [[ "$mode" == "--print-command" ]]; then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
if [[ "$mode" == "--run" ]]; then
  [[ "${CONFIRM_FORMAL_RUN:-}" == "YES" ]] || { printf 'CONFIRM_FORMAL_RUN=YES required\n' >&2; exit 3; }
fi
run_or_print "${gate[@]}"
if [[ "$mode" == "--print-command" && ! -d "$CANDIDATE_CHECKPOINT" ]]; then
  identity_name="${CANDIDATE_EXPERIMENT}-CHECKPOINT_IDENTITY_COMPUTED_AT_RUN"
else
  identity_name="$("$python" scripts/eval/select_checkpoint.py output-name --experiment "$CANDIDATE_EXPERIMENT" --checkpoint "$CANDIDATE_CHECKPOINT")"
fi
run="$EDITMGT_OUTPUT_ROOT/full-validation/$identity_name"
datasets=(magicbrush crispedit scaleedit interedit)
roots=("$MAGICBRUSH_DEV_ROOT" "$CRISPEDIT_ROOT" "$SCALEEDIT_ROOT" "$INTEREDIT_ROOT")
manifests=(magicbrush_official_dev.jsonl crispedit_aux128.jsonl scaleedit_aux128.jsonl interedit_aux128.jsonl)
for index in 0 1 2 3; do
  dataset="${datasets[$index]}"; manifest="$validation/${manifests[$index]}"; destination="$run/$dataset"
  run_or_print "$python" scripts/eval/generate_evaluation.py --dataset "$dataset" --manifest "$manifest" \
    --canonical-root "${roots[$index]}" --model-root "$EDITMGT_MODEL_ROOT" --checkpoint "$CANDIDATE_CHECKPOINT" \
    --experiment "$CANDIDATE_EXPERIMENT" --config configs/eval/formal.yaml --timestep-mode roi_relative --output-dir "$destination"
  run_or_print "$python" scripts/eval/formal_eval.py --experiment "$CANDIDATE_EXPERIMENT" --checkpoint "$CANDIDATE_CHECKPOINT" \
    --dataset "$dataset" --manifest "$manifest" --config configs/eval/formal.yaml \
    --predictions-manifest "$destination/predictions.jsonl" --output "$destination/metrics.json"
done
