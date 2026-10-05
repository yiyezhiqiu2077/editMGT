#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
mode="${1:---print-command}"
if [[ "$mode" != "--print-command" && "$mode" != "--dry-run" && "$mode" != "--run" ]]; then
  echo "usage: $0 --print-command|--dry-run|--run" >&2; exit 2
fi
: "${ASSET_ROOT:?ASSET_ROOT must point to a large writable disk}"
ASSET_ROOT="$(realpath -m "$ASSET_ROOT")"; export ASSET_ROOT
artifacts="$ASSET_ROOT/artifacts"; stages="$artifacts/stages"; fixed="$ASSET_ROOT/derived/fixed200k"
git_sha="$(git rev-parse HEAD)"
env_file="$artifacts/formal_env.sh"
stage_names=(00_environment 01_model 02_datasets 03_eval_models 04_schema_contract 05_canonical 06_validation_freeze 07_selection 08_translation 09_dedup_backfill 10_exact200k 11_corpus_audit 12_ready)
export EDITMGT_MODEL_ROOT="$ASSET_ROOT/models/editmgt" TRANSLATOR_MODEL_ROOT="$ASSET_ROOT/models/nllb"
export MAGICBRUSH_ROOT="$ASSET_ROOT/datasets/magicbrush/train" MAGICBRUSH_DEV_ROOT="$ASSET_ROOT/datasets/magicbrush/dev" MAGICBRUSH_TEST_ROOT="$ASSET_ROOT/datasets/magicbrush/test/canonical"
export CRISPEDIT_ROOT="$ASSET_ROOT/datasets/crispedit" SCALEEDIT_ROOT="$ASSET_ROOT/datasets/scaleedit" INTEREDIT_ROOT="$ASSET_ROOT/datasets/interedit"
export DERIVED_ROOT="$ASSET_ROOT/derived" EDITMGT_OUTPUT_ROOT="$ASSET_ROOT/experiments" TORCH_HOME="$ASSET_ROOT/models/torch"
export MAGICBRUSH_REVISION=1d8d4629150d18ca50afab66391866f2085be989 CRISPEDIT_REVISION=dcbd1c952e93e4361ad862b33f3acd1cc74bec5a SCALEEDIT_REVISION=f97ffb061d4275bcbbff6500572139b6f8df1c4e INTEREDIT_REVISION=b319d9fdb45cc670ec263fe4aeaa962f1623d5dc TRANSLATOR_REVISION=7be3e24664b38ce1cac29b8aeed6911aa0cf0576
if [[ "$mode" == "--run" ]]; then
  mkdir -p "$artifacts" "$stages"
  uv run python scripts/setup/write_formal_env.py --asset-root "$ASSET_ROOT" --output "$env_file" >/dev/null
  source "$env_file"
fi

commands=(
  "uv sync --frozen --group dev --group translation --group metrics"
  "uv run python scripts/setup/download_formal_assets.py --manifest configs/formal_assets.yaml --asset-root $ASSET_ROOT --asset editmgt --asset nllb --run"
  "uv run python scripts/setup/download_formal_assets.py --manifest configs/formal_assets.yaml --asset-root $ASSET_ROOT --asset magicbrush --asset magicbrush_test --asset crispedit --asset scaleedit --asset interedit --run"
  "uv run python scripts/setup/download_formal_assets.py --manifest configs/formal_assets.yaml --asset-root $ASSET_ROOT --asset dino --asset clip --run && uv run python scripts/setup/prefetch_lpips.py --root $TORCH_HOME"
  "uv run python scripts/setup/verify_schema_contracts.py --magicbrush-snapshot $ASSET_ROOT/datasets/magicbrush/snapshot --crispedit-root $CRISPEDIT_ROOT --scaleedit-root $SCALEEDIT_ROOT --interedit-root $INTEREDIT_ROOT --output $fixed/schema_contract_report.json"
  "uv run python scripts/data/prepare_magicbrush_assets.py --snapshot $ASSET_ROOT/datasets/magicbrush/snapshot --train-output $MAGICBRUSH_ROOT --dev-output $MAGICBRUSH_DEV_ROOT --test-root $ASSET_ROOT/datasets/magicbrush/test --test-output $MAGICBRUSH_TEST_ROOT && uv run python scripts/data/build_magicbrush_canonical.py --manifest $MAGICBRUSH_ROOT/manifest.jsonl --dataset-root $MAGICBRUSH_ROOT --revision $MAGICBRUSH_REVISION --output $fixed/pools/magicbrush_train.jsonl && uv run python scripts/data/build_magicbrush_canonical.py --manifest $MAGICBRUSH_DEV_ROOT/manifest.jsonl --dataset-root $MAGICBRUSH_DEV_ROOT --revision $MAGICBRUSH_REVISION --output $fixed/pools/magicbrush_dev.jsonl && uv run python scripts/data/build_schema_mapped_canonical.py --root $CRISPEDIT_ROOT --mapping configs/data/schema/crispedit.yaml --edit-type-mapping configs/data/edit_type_mapping.yaml --output $fixed/pools/crispedit.jsonl && uv run python scripts/data/build_schema_mapped_canonical.py --root $SCALEEDIT_ROOT --mapping configs/data/schema/scaleedit.yaml --edit-type-mapping configs/data/edit_type_mapping.yaml --output $fixed/pools/scaleedit.jsonl && uv run python scripts/data/build_interedit_canonical_pool.py $INTEREDIT_ROOT/metadata/*.jsonl.gz --interedit-root $INTEREDIT_ROOT --revision $INTEREDIT_REVISION --output $fixed/pools/interedit.jsonl"
  "uv run python scripts/data/freeze_fixed200k_validation.py --magicbrush-dev $fixed/pools/magicbrush_dev.jsonl --crispedit-pool $fixed/pools/crispedit.jsonl --scaleedit-pool $fixed/pools/scaleedit.jsonl --interedit-pool $fixed/pools/interedit.jsonl --output-dir $fixed/validation"
  "bash scripts/data/run_initial_fixed200k_selection.sh"
  "bash scripts/data/build_fixed200k_with_translation_loop.sh"
  "uv run python scripts/data/verify_fixed200k_artifacts.py --fixed-root $fixed --phase dedup"
  "uv run python scripts/data/verify_fixed200k_artifacts.py --fixed-root $fixed --phase exact"
  "uv run python scripts/data/audit_fixed200k.py --manifest $fixed/train_200k.jsonl --magicbrush-root $MAGICBRUSH_ROOT --crispedit-root $CRISPEDIT_ROOT --scaleedit-root $SCALEEDIT_ROOT --interedit-root $INTEREDIT_ROOT --model-root $EDITMGT_MODEL_ROOT --output-dir $fixed/audit && uv run python scripts/data/validate_fixed200k_machine.py --manifest $fixed/train_200k.jsonl --validation $fixed/validation/magicbrush_official_dev.jsonl --validation $fixed/validation/crispedit_aux128.jsonl --validation $fixed/validation/scaleedit_aux128.jsonl --validation $fixed/validation/interedit_aux128.jsonl --magicbrush-root $MAGICBRUSH_ROOT --crispedit-root $CRISPEDIT_ROOT --scaleedit-root $SCALEEDIT_ROOT --interedit-root $INTEREDIT_ROOT --audit-report $fixed/audit/audit_report.json --translation-report $fixed/translation_report.json --output $fixed/machine_qa.json; uv run python scripts/data/build_translation_manual_audit.py --manifest $fixed/train_200k.jsonl --output-jsonl $fixed/translation_manual_audit_500.jsonl --output-csv $fixed/translation_manual_audit_500.csv || echo 'diagnostic translation audit generation failed (non-gating)' >&2"
  "uv run python scripts/setup/build_formal_assets_record.py --asset-root $ASSET_ROOT --train-manifest $fixed/train_200k.jsonl --output $artifacts/formal_assets.json && uv run python scripts/data/finalize_corpus_ready.py --fixed-root $fixed --selection-config configs/data/fixed_200k.yaml --translation-cache $fixed/translation_cache.jsonl --formal-assets $artifacts/formal_assets.json --machine-qa $fixed/machine_qa.json --git-sha $git_sha && uv run python scripts/setup/build_formal_assets_record.py --asset-root $ASSET_ROOT --train-manifest $fixed/train_200k.jsonl --corpus-ready $fixed/CORPUS_READY.json --output $artifacts/formal_assets.json && uv run python scripts/data/verify_corpus_ready.py $fixed/CORPUS_READY.json"
)
stage_outputs=(
  "$env_file"
  "$ASSET_ROOT/models/editmgt/asset_identity.json|$ASSET_ROOT/models/nllb/asset_identity.json"
  "$ASSET_ROOT/datasets/magicbrush/snapshot/asset_identity.json|$ASSET_ROOT/datasets/magicbrush/test/asset_identity.json|$CRISPEDIT_ROOT/asset_identity.json|$SCALEEDIT_ROOT/asset_identity.json|$INTEREDIT_ROOT/asset_identity.json"
  "$ASSET_ROOT/models/dinov2-base/asset_identity.json|$ASSET_ROOT/models/clip-vit-large-patch14/asset_identity.json|$TORCH_HOME/asset_identity.json"
  "$fixed/schema_contract_report.json"
  "$fixed/pools/magicbrush_train.jsonl|$fixed/pools/magicbrush_dev.jsonl|$fixed/pools/crispedit.jsonl|$fixed/pools/scaleedit.jsonl|$fixed/pools/interedit.jsonl|$MAGICBRUSH_TEST_ROOT/manifest.jsonl"
  "$fixed/validation/validation.meta.json"
  "$fixed/selection_stage.json"
  "$fixed/train_200k.jsonl"
  "$fixed/duplicate_report.json|$fixed/translation_report.json"
  "$fixed/train_200k.meta.json"
  "$fixed/audit/audit_report.json|$fixed/machine_qa.json"
  "$artifacts/formal_assets.json|$fixed/CORPUS_READY.json"
)

if [[ "$mode" == "--print-command" ]]; then
  for index in "${!commands[@]}"; do printf '%s\t%s\n' "${stage_names[$index]}" "${commands[$index]}"; done
  exit 0
fi
if [[ "$mode" == "--dry-run" ]]; then
  uv run python scripts/setup/download_formal_assets.py --manifest configs/formal_assets.yaml --asset-root "$ASSET_ROOT" --dry-run
  uv run python - <<'PY'
from pathlib import Path
from src.explicit_region.formal_pipeline import STAGES,load_formal_assets,load_schema_contract
load_formal_assets("configs/formal_assets.yaml")
for path in Path("configs/data/schema").glob("*.yaml"): load_schema_contract(path)
assert len(STAGES)==13 and STAGES[0]=="00_environment" and STAGES[-1]=="12_ready"
print({"status":"PASS","stage_dependency_graph":list(STAGES),"note":"metadata only; no asset bodies downloaded"})
PY
  exit 0
fi

config_args=(--config configs/formal_assets.yaml --config configs/data/fixed_200k.yaml --config configs/data/edit_type_mapping.yaml --config configs/data/schema/magicbrush.yaml --config configs/data/schema/crispedit.yaml --config configs/data/schema/scaleedit.yaml --config configs/data/schema/interedit.yaml)
previous=""
for index in "${!stage_names[@]}"; do
  stage="${stage_names[$index]}"; marker="$stages/$stage/_SUCCESS.json"; input_args=()
  [[ -z "$previous" ]] || input_args=(--input "$previous")
  IFS='|' read -r -a output_paths <<< "${stage_outputs[$index]}";output_args=()
  for output_path in "${output_paths[@]}";do output_args+=(--output "$output_path");done
  set +e
  uv run python scripts/setup/formal_stage.py check --stage-root "$stages" --stage "$stage" --git-sha "$git_sha" "${config_args[@]}" "${input_args[@]}" "${output_args[@]}"
  check_status=$?
  set -e
  if [[ "$check_status" == 0 ]]; then
    echo "SKIP $stage (verified marker)"
  elif [[ "$check_status" == 10 ]]; then
    echo "RUN  $stage"
    mkdir -p "$(dirname "$marker")" "$fixed/pools" "$fixed/validation" "$fixed/audit"
    bash -lc "${commands[$index]}"
    uv run python scripts/setup/formal_stage.py write --stage-root "$stages" --stage "$stage" --git-sha "$git_sha" "${config_args[@]}" "${input_args[@]}" "${output_args[@]}"
  else
    exit "$check_status"
  fi
  previous="$marker"
done
echo "FORMAL PREPARE READY: $artifacts/formal_assets.json"
