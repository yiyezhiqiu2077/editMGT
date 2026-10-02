#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${STAGE_A_CHECKPOINT:-}" ]]; then echo "STAGE_A_CHECKPOINT required" >&2; exit 3; fi
mode="${1:---print-command}"
cmd=(uv run accelerate launch --multi_gpu --num_processes 8 --gpu_ids 0,1,2,3,4,5,6,7 scripts/train/train_explicit_region.py --config configs/train/cluster_8g_stage_b.yaml --warm-start "$STAGE_A_CHECKPOINT")
if [[ "$mode" == "--print-command" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
else
  if [[ "${CONFIRM_FORMAL_RUN:-}" != "YES" ]]; then echo "CONFIRM_FORMAL_RUN=YES required" >&2; exit 4; fi
  : "${DERIVED_ROOT:?DERIVED_ROOT required}"
  uv run python scripts/data/verify_corpus_ready.py "$DERIVED_ROOT/fixed200k/CORPUS_READY.json"
  "${cmd[@]}"
fi
