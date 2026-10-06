#!/usr/bin/env bash
set -euo pipefail
repo_root="$(realpath "$(dirname "$0")/../..")"
cd "$repo_root"
source "$repo_root/scripts/setup/common.sh"
mode="${1:---print-command}"
[[ "$mode" == "--run" || "$mode" == "--print-command" || "$mode" == "--inspect-gates" ]] || exit 2
: "${DERIVED_ROOT:?}" "${EDITMGT_OUTPUT_ROOT:?}" "${UV_PROJECT_ENVIRONMENT:?Source the frozen RUN_ROOT/scripts/env.sh first}"
python="$UV_PROJECT_ENVIRONMENT/bin/python"
marker="$DERIVED_ROOT/fixed200k/CORPUS_READY.json"
gate=("$python" -m src.explicit_region.gates --stage smoke --corpus-ready "$marker")
accelerate_base=("$python" -m accelerate.commands.launch --multi_gpu --num_processes 8 --gpu_ids 0,1,2,3,4,5,6,7)
run_or_print() {
  if [[ "$mode" == "--print-command" ]]; then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
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
launch_training() {
  local port
  port="$(pick_main_process_port)"
  printf '[run_fixed_200k_smoke] using main process port %s\n' "$port" >&2
  run_or_print "${accelerate_base[@]}" --main_process_port "$port" scripts/train/train_explicit_region.py "$@"
}
if [[ "$mode" == "--inspect-gates" ]]; then "${gate[@]}"; exit; fi
if [[ "$mode" == "--run" ]]; then
  [[ "${CONFIRM_FORMAL_RUN:-}" == "YES" ]] || { printf 'CONFIRM_FORMAL_RUN=YES required\n' >&2; exit 3; }
  for directory in cluster-smoke-20 fixed200k-smoke-resume-fresh10 fixed200k-smoke-resume-to20; do
    [[ ! -e "$EDITMGT_OUTPUT_ROOT/$directory" ]] || { printf 'Smoke requires fresh output directories: %s\n' "$directory" >&2; exit 3; }
  done
fi
# Smoke deliberately does NOT depend on a pre-existing smoke PASS or preregistration.
run_or_print "${gate[@]}"
launch_training --config configs/train/cluster_8g_smoke.yaml
launch_training --config configs/train/cluster_8g_smoke_fresh10.yaml
launch_training --config configs/train/cluster_8g_smoke_resume20.yaml --resume "$EDITMGT_OUTPUT_ROOT/fixed200k-smoke-resume-fresh10/checkpoint-10"
run_or_print "$python" scripts/train/verify_fixed200k_smoke.py \
  --fresh20 "$EDITMGT_OUTPUT_ROOT/cluster-smoke-20" \
  --fresh10 "$EDITMGT_OUTPUT_ROOT/fixed200k-smoke-resume-fresh10" \
  --resume20 "$EDITMGT_OUTPUT_ROOT/fixed200k-smoke-resume-to20" \
  --output "$EDITMGT_OUTPUT_ROOT/fixed200k-smoke-verification.json"
