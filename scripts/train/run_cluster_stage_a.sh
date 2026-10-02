#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/run_formal.sh" configs/train/cluster_8g_stage_a.yaml "${1:---print-command}"
