#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/../cluster/run_e0_eval.sh" "${1:---print-command}"
