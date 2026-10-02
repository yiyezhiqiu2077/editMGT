#!/usr/bin/env bash
set -euo pipefail
exec bash scripts/cluster/run_fixed_200k_smoke.sh "${1:---print-command}"
