#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/run_formal.sh" configs/train/cluster_8g_smoke.yaml "${1:---print-command}"
