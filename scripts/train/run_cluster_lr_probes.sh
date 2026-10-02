#!/usr/bin/env bash
set -euo pipefail
mode="${1:---print-command}"
for lr in 1e-5 3e-5 5e-5; do
  bash "$(dirname "$0")/run_formal.sh" "configs/train/cluster_8g_lr_${lr}.yaml" "$mode"
done
