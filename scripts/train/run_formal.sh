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
pick_main_process_port() {
  if [[ -n "${EDITMGT_MAIN_PROCESS_PORT:-}" ]]; then
    printf '%s\n' "$EDITMGT_MAIN_PROCESS_PORT"
    return
  fi
  "$python" - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}
run_launch() {
  local port
  port="$(pick_main_process_port)"
  printf '[run_formal] %s using main process port %s\n' "$config" "$port" >&2
  "$python" -m accelerate.commands.launch --multi_gpu --num_processes 8 --gpu_ids 0,1,2,3,4,5,6,7 --main_process_port "$port" scripts/train/train_explicit_region.py --config "$config"
}
if [[ "$mode" == "--print-command" ]]; then
  printf '%q ' "${gate[@]}"; printf '\n'
  local_port="$(pick_main_process_port)"
  printf '%q ' "$python" -m accelerate.commands.launch --multi_gpu --num_processes 8 --gpu_ids 0,1,2,3,4,5,6,7 --main_process_port "$local_port" scripts/train/train_explicit_region.py --config "$config"; printf '\n'
else
  "${gate[@]}"
  if [[ "$mode" == "--run" ]]; then
    [[ "${CONFIRM_FORMAL_RUN:-}" == "YES" ]] || { printf 'CONFIRM_FORMAL_RUN=YES required\n' >&2; exit 3; }
    run_launch
  fi
fi
