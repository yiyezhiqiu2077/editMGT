#!/usr/bin/env bash
set -euo pipefail
exec bash scripts/train/run_cluster_ablations.sh "${1:---print-command}"
