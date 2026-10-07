#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../setup/common.sh"
if [[ ( $# -ne 2 && $# -ne 4 ) || ( "$2" != "--print-command" && "$2" != "--run" ) ]]; then
  echo "usage: $0 CONFIG --print-command|--run [--resume CHECKPOINT]" >&2
  exit 2
fi
config="$1"
mode="$2"
port="$(pick_main_process_port)"
cmd=(uv run accelerate launch --multi_gpu --num_processes 8 --gpu_ids 0,1,2,3,4,5,6,7 --main_process_port "$port" scripts/train/train_explicit_region.py --config "$config")
if [[ $# -eq 4 ]]; then
  [[ "$3" == "--resume" ]] || { echo "only --resume is accepted" >&2; exit 2; }
  cmd+=(--resume "$4")
fi
if [[ "$mode" == "--print-command" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
else
  if [[ "${CONFIRM_FORMAL_RUN:-}" != "YES" ]]; then echo "CONFIRM_FORMAL_RUN=YES required" >&2; exit 3; fi
  : "${ASSET_ROOT:?ASSET_ROOT required}" "${DERIVED_ROOT:?DERIVED_ROOT required}"
  gate_args=()
  if [[ "$(basename "$config")" == cluster_8g_e3.yaml || "$(basename "$config")" == cluster_8g_smoke*.yaml ]]; then
    gate_args+=(--require-reattestation)
  fi
  if [[ "$(basename "$config")" == cluster_8g_smoke*.yaml ]]; then
    uv run python scripts/setup/verify_formal_pipeline.py --mode train "${gate_args[@]}" --formal-assets "$ASSET_ROOT/artifacts/formal_assets.json" --corpus-ready "$DERIVED_ROOT/fixed200k/CORPUS_READY.json" --pre-smoke
  else
    uv run python scripts/setup/verify_formal_pipeline.py --mode train "${gate_args[@]}" --formal-assets "$ASSET_ROOT/artifacts/formal_assets.json" --corpus-ready "$DERIVED_ROOT/fixed200k/CORPUS_READY.json" --formal-ready "$ASSET_ROOT/artifacts/FORMAL_READY.json"
  fi
  "${cmd[@]}"
fi
