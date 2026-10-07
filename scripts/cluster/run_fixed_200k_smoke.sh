#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../setup/common.sh"
mode="${1:---print-command}"
: "${ASSET_ROOT:?ASSET_ROOT required}" "${DERIVED_ROOT:?DERIVED_ROOT required}" "${EDITMGT_OUTPUT_ROOT:?EDITMGT_OUTPUT_ROOT required}"
marker="$DERIVED_ROOT/fixed200k/CORPUS_READY.json"
formal_assets="$ASSET_ROOT/artifacts/formal_assets.json"
port="$(pick_main_process_port)"
launcher=(uv run accelerate launch --multi_gpu --num_processes 8 --gpu_ids 0,1,2,3,4,5,6,7 --main_process_port "$port" scripts/train/train_explicit_region.py)
commands=(
  "uv run python scripts/setup/verify_formal_pipeline.py --mode train --require-reattestation --formal-assets $formal_assets --corpus-ready $marker --pre-smoke"
  "${launcher[*]} --config configs/train/cluster_8g_smoke.yaml"
  "${launcher[*]} --config configs/train/cluster_8g_smoke_fresh10.yaml"
  "${launcher[*]} --config configs/train/cluster_8g_smoke_resume20.yaml --resume $EDITMGT_OUTPUT_ROOT/fixed200k-smoke-resume-fresh10/checkpoint-10"
  "uv run python scripts/train/verify_fixed200k_smoke.py --fresh20 $EDITMGT_OUTPUT_ROOT/cluster-smoke-20 --fresh10 $EDITMGT_OUTPUT_ROOT/fixed200k-smoke-resume-fresh10 --resume20 $EDITMGT_OUTPUT_ROOT/fixed200k-smoke-resume-to20 --output $EDITMGT_OUTPUT_ROOT/fixed200k-smoke-verification.json"
  "uv run python scripts/train/finalize_formal_ready.py --formal-assets $formal_assets --corpus-ready $marker --smoke-verification $EDITMGT_OUTPUT_ROOT/fixed200k-smoke-verification.json --output $ASSET_ROOT/artifacts/FORMAL_READY.json"
)
if [[ "$mode" == "--print-command" ]]; then
  printf '%s\n' "${commands[@]}"
elif [[ "$mode" == "--run" ]]; then
  [[ "${CONFIRM_FORMAL_RUN:-}" == "YES" ]] || { echo "CONFIRM_FORMAL_RUN=YES required" >&2; exit 3; }
  for command in "${commands[@]}"; do bash -lc "$command"; done
else
  echo "usage: $0 --print-command|--run" >&2; exit 2
fi
