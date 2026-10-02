#!/usr/bin/env bash
set -euo pipefail
exec bash scripts/train/run_cluster_stage_a.sh "${1:---print-command}"
