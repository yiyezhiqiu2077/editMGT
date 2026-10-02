#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../setup/common.sh"
mode="${1:---print-command}"
cmd1=(uv run python scripts/data/audit_change_containment.py --dataset magicbrush --manifest "$DERIVED_ROOT/magicbrush-splits/train.jsonl" --model-root "$EDITMGT_MODEL_ROOT" --output-dir "$DERIVED_ROOT/audits/containment-magicbrush")
cmd2=(uv run python scripts/data/audit_change_containment.py --dataset interedit --manifest "$DERIVED_ROOT/interedit/splits/train.jsonl" --interedit-root "$INTEREDIT_ROOT" --model-root "$EDITMGT_MODEL_ROOT" --output-dir "$DERIVED_ROOT/audits/containment-interedit")
if [[ "$mode" == --print-command ]];then printf '%q ' "${cmd1[@]}";printf '\n';printf '%q ' "${cmd2[@]}";printf '\n';else [[ "${CONFIRM_FORMAL_RUN:-}" == YES ]]||exit 3;"${cmd1[@]}";"${cmd2[@]}";fi
