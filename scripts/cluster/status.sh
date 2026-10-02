#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../setup/common.sh"
git status --short --branch
bash scripts/setup/probe_runtime.sh
for path in local_assets/models/EditMGT local_assets/datasets/magicbrush local_assets/datasets/magicbrush-test local_assets/datasets/interedit; do
  if [[ -L "$path" ]]; then echo "OK $path -> $(realpath "$path")"; else echo "MISSING $path"; fi
done
