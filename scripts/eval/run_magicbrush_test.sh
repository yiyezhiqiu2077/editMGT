#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../setup/common.sh"
mode="${1:---print-command}"
: "${ASSET_ROOT:?ASSET_ROOT required}" "${CANDIDATE_CHECKPOINT:?CANDIDATE_CHECKPOINT required}" "${EDITMGT_MODEL_ROOT:?EDITMGT_MODEL_ROOT required}" "${MAGICBRUSH_TEST_ROOT:?MAGICBRUSH_TEST_ROOT required}" "${EDITMGT_OUTPUT_ROOT:?EDITMGT_OUTPUT_ROOT required}"
out="$EDITMGT_OUTPUT_ROOT/magicbrush-test/$(basename "$CANDIDATE_CHECKPOINT")"
commands=(
 "uv run python scripts/eval/verify_magicbrush_test_contract.py --asset-root $ASSET_ROOT --checkpoint $CANDIDATE_CHECKPOINT"
 "uv run python scripts/eval/generate_evaluation.py --dataset magicbrush --manifest $MAGICBRUSH_TEST_ROOT/manifest.jsonl --model-root $EDITMGT_MODEL_ROOT --checkpoint $CANDIDATE_CHECKPOINT --config configs/eval/formal.yaml --timestep-mode roi_relative --output-dir $out --count 1053"
 "uv run python scripts/eval/formal_eval.py --experiment magicbrush-official-test --checkpoint $CANDIDATE_CHECKPOINT --config configs/eval/formal.yaml --predictions-manifest $out/predictions.jsonl --output $out/metrics.json"
)
if [[ "$mode" == "--print-command" ]];then printf '%s\n' "${commands[@]}";elif [[ "$mode" == "--run" ]];then [[ "${CONFIRM_FORMAL_RUN:-}" == YES ]]||{ echo "CONFIRM_FORMAL_RUN=YES required" >&2;exit 3;};for command in "${commands[@]}";do bash -lc "$command";done;else echo "usage: $0 --print-command|--run" >&2;exit 2;fi
