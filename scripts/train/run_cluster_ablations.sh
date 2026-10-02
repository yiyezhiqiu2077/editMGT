#!/usr/bin/env bash
set -euo pipefail
mode="${1:---print-command}"
for experiment in e1 e2 e3 e4; do
  bash "$(dirname "$0")/run_formal.sh" "configs/train/cluster_8g_${experiment}.yaml" "$mode"
done
