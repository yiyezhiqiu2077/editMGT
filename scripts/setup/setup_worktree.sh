#!/usr/bin/env bash
set -euo pipefail
source_repo="${EDITMGT_REPO_ROOT:?set EDITMGT_REPO_ROOT to the upstream clone}"
worktree="${EDITMGT_WORKTREE:?set EDITMGT_WORKTREE to the new worktree path}"
branch="exp/explicit-region-sft-v1"
expected_origin="${EDITMGT_EXPECTED_ORIGIN:-https://github.com/weichow23/EditMGT.git}"
actual_origin="$(git -C "$source_repo" remote get-url origin)"
[[ "$actual_origin" == "$expected_origin" ]] || { echo "ORIGIN_MISMATCH $actual_origin" >&2; exit 3; }
if [[ -e "$worktree" ]]; then
  [[ "$(git -C "$worktree" rev-parse --show-toplevel)" == "$worktree" ]] || { echo WRONG_WORKTREE >&2; exit 4; }
  [[ "$(git -C "$worktree" branch --show-current)" == "$branch" ]] || { echo WRONG_BRANCH >&2; exit 5; }
  [[ "$(git -C "$worktree" remote get-url origin)" == "$expected_origin" ]] || { echo ORIGIN_MISMATCH >&2; exit 6; }
  echo "EXISTING_VALID $worktree branch=$branch origin=$actual_origin"
  exit 0
fi
if git -C "$source_repo" show-ref --verify --quiet "refs/heads/$branch"; then
  git -C "$source_repo" worktree add "$worktree" "$branch"
else
  git -C "$source_repo" worktree add -b "$branch" "$worktree"
fi
echo "CREATED $worktree branch=$branch origin=$actual_origin"
