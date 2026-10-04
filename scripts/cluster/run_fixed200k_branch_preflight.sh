#!/usr/bin/env bash
# Exactly two approved E1/E2 optimizer updates each; not a smoke PASS or candidate.
set -euo pipefail
repo_root="$(realpath "$(dirname "$0")/../..")"
cd "$repo_root"
mode="${1:---print-command}"
[[ "$mode" == "--run" || "$mode" == "--print-command" || "$mode" == "--inspect-gates" ]] || exit 2
: "${EDITMGT_OUTPUT_ROOT:?}" "${UV_PROJECT_ENVIRONMENT:?Source frozen env first}"
if [[ "$mode" == "--run" ]]; then
  for branch in e1 e2; do
    destination="$EDITMGT_OUTPUT_ROOT/branch-preflight-${branch}-2updates"
    [[ ! -e "$destination" && ! -L "$destination" ]] || { printf 'Diagnostic needs fresh noncandidate directory: %s\n' "$destination" >&2; exit 3; }
  done
fi
for branch in e1 e2; do
  bash scripts/train/run_formal.sh "configs/train/cluster_8g_diagnostic_${branch}.yaml" "$mode"
done
