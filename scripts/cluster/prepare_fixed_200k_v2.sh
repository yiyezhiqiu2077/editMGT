#!/usr/bin/env bash
set -euo pipefail
echo "prepare_fixed_200k_v2.sh is deprecated; delegating to the Formal Pipeline v3 runner" >&2
exec bash "$(dirname "$0")/../setup/run_formal_prepare.sh" "${1:---print-command}"
