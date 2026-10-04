#!/usr/bin/env bash
# Released E0 full official DEV only; distinct protocols, never candidate ranking.
set -euo pipefail
repo_root="$(realpath "$(dirname "$0")/../..")"
cd "$repo_root"
source scripts/setup/common.sh
mode="${1:---print-command}"
[[ "$mode" == "--run" || "$mode" == "--print-command" || "$mode" == "--inspect-gates" ]] || exit 2
: "${DERIVED_ROOT:?}" "${MAGICBRUSH_DEV_ROOT:?}" "${EDITMGT_MODEL_ROOT:?}" "${EDITMGT_OUTPUT_ROOT:?}" "${RUN_ROOT:?}" "${UV_PROJECT_ENVIRONMENT:?}"
python="$UV_PROJECT_ENVIRONMENT/bin/python"
release="${EDITMGT_RELEASE_METADATA:-$RUN_ROOT/provenance/editmgt_remote.json}"
manifest="$DERIVED_ROOT/fixed200k/validation/magicbrush_official_dev.jsonl"
gate=("$python" -m src.explicit_region.gates --stage eval --corpus-ready "$DERIVED_ROOT/fixed200k/CORPUS_READY.json")
run_or_print() { if [[ "$mode" == "--print-command" ]]; then printf '%q ' "$@"; printf '\n'; else "$@"; fi; }
if [[ "$mode" == "--inspect-gates" ]]; then "${gate[@]}"; exit; fi
if [[ "$mode" == "--run" ]]; then [[ "${CONFIRM_FORMAL_RUN:-}" == YES ]] || exit 3; fi
run_or_print "${gate[@]}"
for spec in E0-official:official_upstream_timestep E0-region:roi_relative; do
  experiment="${spec%%:*}"; protocol="${spec#*:}"
  base="$EDITMGT_OUTPUT_ROOT/e0-full-dev"
  run_or_print "$python" scripts/eval/generate_evaluation.py --baseline-canonical --dataset magicbrush \
    --manifest "$manifest" --canonical-root "$MAGICBRUSH_DEV_ROOT" --model-root "$EDITMGT_MODEL_ROOT" \
    --release-metadata "$release" --experiment "$experiment" --config configs/eval/formal.yaml \
    --timestep-mode "$protocol" --output-dir "$base"
  if [[ "$mode" == "--print-command" ]]; then
    destination="$base/$experiment-IDENTITY_COMPUTED_AT_RUN"
  else
    # Generation creates one exclusive experiment+identity directory. Never reuse
    # an old basename or infer a fictitious finetuned checkpoint identity.
    matches=("$base/$experiment-"*/predictions.jsonl)
    [[ ${#matches[@]} == 1 && -f "${matches[0]}" ]] || { printf 'Ambiguous baseline outputs for %s\n' "$experiment" >&2; exit 3; }
    destination="$(dirname "${matches[0]}")"
  fi
  run_or_print "$python" scripts/eval/formal_eval.py --baseline-canonical --checkpoint released \
    --experiment "$experiment" --dataset magicbrush --manifest "$manifest" --model-root "$EDITMGT_MODEL_ROOT" \
    --release-metadata "$release" --timestep-mode "$protocol" --config configs/eval/formal.yaml \
    --predictions-manifest "$destination/predictions.jsonl" --output "$destination/metrics.json"
done
