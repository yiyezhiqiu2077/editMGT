#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mode=${1:-}
[[ "$mode" == --print-command || "$mode" == --run ]] || exit 2
: "${DENSE_OUTPUT_ROOT:?Use an independent empty output directory}"
config=configs/train/dense_mini512_smoke.yaml
base=(uv run accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16
  --gpu_ids 0,1,2,3,4,5,6,7 --main_process_port "${DENSE_MASTER_PORT:-29671}"
  scripts/train/train_dense_region.py --config "$config")
if [[ "$mode" == --print-command ]]; then
  printf '%q ' "${base[@]}" --output-dir "$DENSE_OUTPUT_ROOT/fresh16" --stop-at-global-step 16
  printf '\n'
  exit 0
fi
uv run python scripts/train/preflight_dense_local.py
mkdir -p "$DENSE_OUTPUT_ROOT/logs"
"${base[@]}" --output-dir "$DENSE_OUTPUT_ROOT/fresh16" --stop-at-global-step 16 > "$DENSE_OUTPUT_ROOT/logs/fresh16.log" 2>&1
"${base[@]}" --output-dir "$DENSE_OUTPUT_ROOT/fresh8" --stop-at-global-step 8 > "$DENSE_OUTPUT_ROOT/logs/fresh8.log" 2>&1
"${base[@]}" --output-dir "$DENSE_OUTPUT_ROOT/resumed16" --stop-at-global-step 16 \
  --resume "$DENSE_OUTPUT_ROOT/fresh8/checkpoint-8" > "$DENSE_OUTPUT_ROOT/logs/resumed16.log" 2>&1
uv run python scripts/train/compare_dense_resume.py \
  --fresh "$DENSE_OUTPUT_ROOT/fresh16/checkpoint-16" \
  --resumed "$DENSE_OUTPUT_ROOT/resumed16/checkpoint-16" \
  --fresh-log "$DENSE_OUTPUT_ROOT/fresh16/train_metrics.jsonl" \
  --interrupted-logs "$DENSE_OUTPUT_ROOT/fresh8/train_metrics.jsonl" "$DENSE_OUTPUT_ROOT/resumed16/train_metrics.jsonl" \
  --output "$DENSE_OUTPUT_ROOT/DENSE_SMOKE.json"
