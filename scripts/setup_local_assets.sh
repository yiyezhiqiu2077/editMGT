#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/setup/common.sh"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 model|magicbrush|magicbrush-test|interedit|outputs /absolute/path" >&2
  exit 2
fi
kind="$1"
source_path="$(realpath -e "$2")"
case "$kind" in
  model) target="local_assets/models/EditMGT" ;;
  magicbrush) target="local_assets/datasets/magicbrush" ;;
  magicbrush-test) target="local_assets/datasets/magicbrush-test" ;;
  interedit) target="local_assets/datasets/interedit" ;;
  outputs) target="local_assets/outputs" ;;
  *) echo "unsupported asset kind: $kind" >&2; exit 2 ;;
esac
mkdir -p "$(dirname "$target")"
if [[ -L "$target" ]]; then
  current="$(realpath -e "$target")"
  if [[ "$current" == "$source_path" ]]; then
    echo "EXISTING_VALID $target -> $current"
    exit 0
  fi
  echo "WRONG_SYMLINK $target -> $current expected=$source_path" >&2
  exit 3
fi
if [[ -e "$target" ]]; then
  echo "REAL_PATH_EXISTS_REFUSING_TO_OVERWRITE $target" >&2
  exit 4
fi
ln -s "$source_path" "$target"
echo "LINKED $target -> $source_path"
