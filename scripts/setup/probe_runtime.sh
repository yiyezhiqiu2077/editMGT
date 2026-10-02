#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

echo "worktree=$actual_root"
echo "git_sha=$(git rev-parse HEAD)"
echo "python=$(python3 --version 2>&1)"
echo "uv=$(uv --version 2>&1 || echo NOT_FOUND)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader || true
if command -v uv >/dev/null 2>&1 && [[ -d .venv ]]; then
  uv run python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
PY
fi
