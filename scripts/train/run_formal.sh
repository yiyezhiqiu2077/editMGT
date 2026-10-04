#!/usr/bin/env bash
set -euo pipefail
repo_root="$(realpath "$(dirname "$0")/../..")"
cd "$repo_root"
source "$repo_root/scripts/setup/common.sh"
if [[ $# -ne 2 || ( "$2" != "--print-command" && "$2" != "--run" && "$2" != "--inspect-gates" ) ]]; then
  printf 'usage: %s CONFIG --print-command|--inspect-gates|--run\n' "$0" >&2
  exit 2
fi
config="$1"
mode="$2"
: "${UV_PROJECT_ENVIRONMENT:?Source the frozen RUN_ROOT/scripts/env.sh first}"
python="$UV_PROJECT_ENVIRONMENT/bin/python"
gate=("$python" -m src.explicit_region.gates --train-config "$config")
cmd=("$python" -m accelerate.commands.launch --multi_gpu --num_processes 8 --gpu_ids 0,1,2,3,4,5,6,7 scripts/train/train_explicit_region.py --config "$config")
if [[ "$mode" == "--print-command" ]]; then
  printf '%q ' "${gate[@]}"; printf '\n'
  printf '%q ' "${cmd[@]}"; printf '\n'
else
  "${gate[@]}"
  if [[ "$mode" == "--run" ]]; then
    [[ "${CONFIRM_FORMAL_RUN:-}" == "YES" ]] || { printf 'CONFIRM_FORMAL_RUN=YES required\n' >&2; exit 3; }
    "${cmd[@]}"
  fi
fi
