#!/usr/bin/env bash
set -euo pipefail
exec bash scripts/train/run_cluster_lr_probes.sh "${1:---print-command}"
