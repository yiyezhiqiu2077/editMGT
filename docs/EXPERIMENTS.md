# EditMGT Explicit-Region SFT experiments

This is the end-to-end runbook for reproducing the D200K-v2 corpus and explicit-region experiments
on a fresh server. It describes the current code, including gates that have not yet been executed.
Commands are run from the repository root unless stated otherwise.

## 1. Experimental scope

The experiment series asks:

- Q1: does continued SFT improve the released EditMGT under a fixed evaluation protocol?
- Q2: does ROI hard-lock corruption improve provided-region editing and outside preservation?
- Q3: does a persistent edit-region embedding add value beyond ROI corruption?
- Q4: is target-side LoRA adaptation necessary, or is reference-only adaptation sufficient?
- Q5: which dense checkpoint should be the baseline for later cache, sparse, or shortcut acceleration?

This repository defines the experiments; it does not report answers. E0–E4 and all long formal runs
remain unrun.

## 2. Reproducibility contract

A formal run binds all of the following:

- one reviewed Git SHA and dirty-diff digest;
- `uv.lock` and the resolved YAML configuration;
- released EditMGT component/snapshot identity;
- immutable raw-dataset and NLLB revisions;
- canonical `sample_uid` plus source/target/region content SHA256;
- exact `train_200k.jsonl` SHA256 and all READY-attested artifacts;
- `fixed200k-hash-sort-v1` epoch permutation and no-replacement rank stride;
- optimizer-boundary committed sample cursor for resume;
- world size, batch/GPU, accumulation, seed, precision, LoRA scope, and scheduler state.

`torch_compile=false` and deterministic algorithms are part of the v1 correctness contract. A bad
frozen row terminates the run; runtime replacement is forbidden.

## 3. Empty-server setup

```bash
git clone git@github.com:yiyezhiqiu2077/editMGT.git
cd editMGT
git checkout FORMAL_EXPERIMENT_CODE_SHA

export EDITMGT_WORKTREE="$PWD"
uv sync --frozen --group dev --group translation --group metrics

uv run python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
PY
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader
```

If `uv` is not installed and the server can reach Astral, `bash scripts/setup/bootstrap_uv.sh`
installs it and runs the base frozen sync. On an isolated cluster, provision `uv` and a compatible
wheel cache in advance; do not remove `--frozen`.

Record the starting identity:

```bash
git rev-parse HEAD
git status --short
sha256sum uv.lock configs/data/fixed_200k.yaml configs/eval/formal.yaml
```

## 4. Directory and asset setup

Keep code, read-only raw assets, generated corpus, caches, and outputs separate. A generic layout is:

```text
WORK_ROOT/
├── editMGT/                         # Git checkout
├── models/
│   ├── EditMGT/IMMUTABLE_SNAPSHOT/
│   └── NLLB/IMMUTABLE_SNAPSHOT/
├── datasets/
│   ├── MagicBrush/{train,dev,test}/
│   ├── CrispEdit-labeling-39k/
│   ├── ScaleEdit-labeling-25k/
│   └── Inter-Edit-Train/
├── derived/editmgt-explicit-region/
├── outputs/editmgt-explicit-region/
└── cache/huggingface/
```

Export roots and immutable identity labels. Replace the generic values with audited local paths and
real immutable revisions:

```bash
export EDITMGT_WORKTREE="$PWD"
export EDITMGT_MODEL_ROOT=/models/EditMGT/IMMUTABLE_SNAPSHOT
export TRANSLATOR_MODEL_ROOT=/models/NLLB/IMMUTABLE_SNAPSHOT
export MAGICBRUSH_ROOT=/datasets/MagicBrush/train
export MAGICBRUSH_DEV_ROOT=/datasets/MagicBrush/dev
export CRISPEDIT_ROOT=/datasets/CrispEdit-labeling-39k
export SCALEEDIT_ROOT=/datasets/ScaleEdit-labeling-25k
export INTEREDIT_ROOT=/datasets/Inter-Edit-Train
export DERIVED_ROOT=/derived/editmgt-explicit-region
export EDITMGT_OUTPUT_ROOT=/outputs/editmgt-explicit-region
export HF_HOME=/cache/huggingface

export MAGICBRUSH_REVISION=IMMUTABLE_DATASET_ID
export CRISPEDIT_REVISION=IMMUTABLE_DATASET_ID
export SCALEEDIT_REVISION=IMMUTABLE_DATASET_ID
export INTEREDIT_REVISION=IMMUTABLE_DATASET_ID
export TRANSLATOR_REVISION=IMMUTABLE_MODEL_ID
```

Raw datasets and released snapshots are read-only. Only `DERIVED_ROOT`, `EDITMGT_OUTPUT_ROOT`, and
`HF_HOME` are writable. Nothing under those roots belongs in Git.

Optional local symlinks used by the asset audit can be created with:

```bash
bash scripts/setup_local_assets.sh model "$EDITMGT_MODEL_ROOT"
bash scripts/setup_local_assets.sh magicbrush "$MAGICBRUSH_ROOT"
bash scripts/setup_local_assets.sh magicbrush-test /datasets/MagicBrush/test
bash scripts/setup_local_assets.sh interedit "$INTEREDIT_ROOT"
bash scripts/setup_local_assets.sh outputs "$EDITMGT_OUTPUT_ROOT"
uv run python scripts/setup/audit_assets.py
```

See [ENVIRONMENT.md](ENVIRONMENT.md) for the full environment contract.

## 5. Released model assets

`EDITMGT_MODEL_ROOT` must be one complete released snapshot containing:

```text
editmgt/
text_encoder/
tokenizer/
llm_encoder/
vqvae/
scheduler/
```

The training path verifies these names, loads from local files, and records component identity. It
must not silently substitute another Gemma/text encoder, tokenizer, scheduler, or VQ-VAE. The VQ-VAE
is fixed to FP32 because the local BF16 audit did not preserve adequate token agreement.

## 6. Raw datasets

The formal data sources are:

- MagicBrush official TRAIN;
- MagicBrush official DEV, used as primary validation;
- MagicBrush official TEST, sealed for final evaluation only;
- CrispEdit-labeling-39k;
- ScaleEdit-labeling-25k;
- Inter-Edit-Train, restricted to quality-filtered `better_data` records.

Do not substitute similarly named local caches or test sets. Real schema audit is mandatory before
canonicalization because CrispEdit and ScaleEdit field names, storage layout, edit taxonomy, and mask
semantics are not inferred automatically.

Run read-only schema audits first:

```bash
mkdir -p "$DERIVED_ROOT/fixed200k/schema_audit" "$DERIVED_ROOT/schema_mappings"

uv run python scripts/data/audit_raw_dataset_schema.py \
  --dataset-name magicbrush --root "$MAGICBRUSH_ROOT" \
  --revision "$MAGICBRUSH_REVISION" \
  --output "$DERIVED_ROOT/fixed200k/schema_audit/magicbrush.json"

uv run python scripts/data/audit_raw_dataset_schema.py \
  --dataset-name crispedit --root "$CRISPEDIT_ROOT" \
  --revision "$CRISPEDIT_REVISION" \
  --output "$DERIVED_ROOT/fixed200k/schema_audit/crispedit.json"

uv run python scripts/data/audit_raw_dataset_schema.py \
  --dataset-name scaleedit --root "$SCALEEDIT_ROOT" \
  --revision "$SCALEEDIT_REVISION" \
  --output "$DERIVED_ROOT/fixed200k/schema_audit/scaleedit.json"

uv run python scripts/data/audit_raw_dataset_schema.py \
  --dataset-name interedit --root "$INTEREDIT_ROOT" \
  --revision "$INTEREDIT_REVISION" \
  --output "$DERIVED_ROOT/fixed200k/schema_audit/interedit.json"
```

After human inspection, copy and complete the explicit mappings:

```bash
export CRISPEDIT_SCHEMA_MAPPING="$DERIVED_ROOT/schema_mappings/crispedit.yaml"
export SCALEEDIT_SCHEMA_MAPPING="$DERIVED_ROOT/schema_mappings/scaleedit.yaml"
cp configs/data/crispedit_schema_mapping.template.yaml "$CRISPEDIT_SCHEMA_MAPPING"
cp configs/data/scaleedit_schema_mapping.template.yaml "$SCALEEDIT_SCHEMA_MAPPING"
```

Fill schema status, revision, format/glob, source/target/region storage and fields, instruction,
edit type, optional group ID, and mask semantics. Then complete the CrispEdit/ScaleEdit taxonomy in
`configs/data/edit_type_mapping.yaml`. Remaining `REPLACE`, `NOT_RUN`, or empty dataset mappings are
intentional hard failures.

## 7. D200K-v2 corpus preparation

The fixed policy in `configs/data/fixed_200k.yaml` is:

- use every eligible MagicBrush TRAIN row;
- take at most 39,000 eligible CrispEdit rows;
- take at most 25,000 eligible ScaleEdit rows;
- fill the remaining quota with `better_data` Inter-Edit rows;
- produce exactly 200,000 unique `sample_uid` rows.

Actual contributions cannot be reported until the real corpus is built. The build order is:

```text
real schema audit
  -> explicit canonical adapters
  -> MagicBrush official DEV + probe128 + three aux128 freezes
  -> eligibility filtering
  -> deterministic candidate and reserve order
  -> same-dataset exact-sample collapse
  -> cross-dataset exact-source resolution (validation wins)
  -> translate selected Han instructions
  -> translation QA and deterministic reserve backfill
  -> exact-200K integrity freeze
  -> content/geometry/VQ/montage audits
  -> CORPUS_READY.json
```

The dataset priority for cross-dataset source duplicates is MagicBrush, CrispEdit, ScaleEdit,
Inter-Edit. Selection is deterministic and without replacement. MagicBrush official DEV wins any
train/validation source or group conflict. TEST is never read by this pipeline.

Print the full orchestrated command list:

```bash
bash scripts/cluster/prepare_fixed_200k_v2.sh --print-command
```

After reviewing paths, revisions, and mappings, run:

```bash
export CONFIRM_CORPUS_BUILD=YES
bash scripts/cluster/prepare_fixed_200k_v2.sh --run
unset CONFIRM_CORPUS_BUILD
```

The primary outputs are under `$DERIVED_ROOT/fixed200k/`:

```text
train_200k.jsonl
train_200k.meta.json
candidate_selection.jsonl
reserve_order.jsonl
final_selection.jsonl
selection_report.json
duplicate_report.json
translation_report.json
translation_cache.jsonl
translation_manual_audit_500.{jsonl,csv}
validation/
audit/
CORPUS_READY.json
```

## 8. Translation

Han detection applies to every selected dataset, not just Inter-Edit. The frozen translation contract
is `facebook/nllb-200-distilled-1.3B`, `zho_Hans -> eng_Latn`, immutable revision, greedy decoding,
and `max_new_tokens=128`.

The cache key binds original text, backend, model revision, source/target language, and decoding.
Selection happens before translation. The following QA flags reject a translation:

- `empty_output`;
- `copy_output`;
- `han_remaining`;
- `length_ratio_outlier`;
- `digit_mismatch`.

A rejected selected row is replaced from the deterministic reserve for the requested stratum. The
loop is bounded to 32 iterations. The final build also freezes 500 translated rows for human audit;
the CSV and all flagged cases must be inspected before formal training.

## 9. Data audit

`scripts/data/audit_fixed200k.py` verifies all frozen rows with their real roots and produces:

- per-dataset counts and unique-source counts;
- missing/decode/hash/Han failures;
- edit-type and mask-semantics distributions;
- region-fraction and samples-per-source statistics;
- dimension and aspect-ratio summaries;
- FP32-VQ inside/outside change containment;
- 100-row montage per dataset and a final stratified montage.

Human reviewers must check source/target/region alignment, mask polarity, instruction correctness,
translation quality, duplicate/diversity reports, and anomalous long tails. Machine success alone is
not corpus approval.

## 10. Corpus-ready gate

`CORPUS_READY.json` is the only machine-trusted readiness marker. It binds the train manifest,
metadata, candidate/reserve/final selection, duplicate and translation reports, translation cache,
validation manifests, audit report, manual translation sample, config, and montages by SHA256.

Verify it independently:

```bash
uv run python scripts/data/verify_corpus_ready.py \
  "$DERIVED_ROOT/fixed200k/CORPUS_READY.json"
```

D200K smoke, formal training, and full validation re-run hash verification at launch. Missing files,
changed hashes, a non-READY status, or a total other than 200,000 blocks execution. Record a separate
human go/no-go note with reviewer, date, and marker SHA; READY does not attest licensing or visual
quality.

## 11. 8-GPU smoke and resume gate

The smoke uses the E3 path with 8 GPUs, batch/GPU 1, accumulation 4, and therefore global batch 32.
Twenty optimizer steps commit exactly 640 distinct rows.

```bash
bash scripts/cluster/run_8g_smoke.sh --print-command

export CONFIRM_FORMAL_RUN=YES
bash scripts/cluster/run_8g_smoke.sh --run
unset CONFIRM_FORMAL_RUN
```

The launcher executes:

1. READY hash verification;
2. uninterrupted fresh steps 1–20;
3. fresh steps 1–10 and checkpoint-10;
4. resume from checkpoint-10 through step 20;
5. sequence and final-state verification.

`$EDITMGT_OUTPUT_ROOT/fixed200k-smoke-verification.json` must report 640 unique rows and exact matches
for sample UID, geometry seed, corruption seed, and learning-rate sequence. OOM, NCCL failure,
NaN/Inf, missing state, or verifier failure is a no-go for all long runs.

This 8-GPU smoke has not yet been run for D200K-v2.

## 12. Training semantics

Let `M` be the token-level edit region and `S` the deterministically sampled masked subset.

### Full target

The corruption canvas is the target token grid. `S` is sampled over the whole grid, masked in the
input, and supervised against target tokens. No edit-region condition is passed in E1.

### ROI hard lock

`S` is a non-empty subset of `M`. The input contains:

```text
MASK token       on S
target token     on M \ S
source token     outside M
```

Labels are target tokens only on `S`; all other positions use ignore index. Thus optimization loss is
computed only on selected edit tokens, while the outside canvas is source-locked. The mask ratio is
`rho = cos(u * pi / 2)`; with probability 0.15, all ROI tokens are masked. E2 disables persistent
region conditioning; E3/E4 add the learned region embedding on `M` throughout the Transformer.

The loss is sample-normalized:

```text
L_i = sum_{p in S_i} CE(logits_i,p, target_i,p) / |S_i|
L   = mean_i L_i
```

This prevents large masks from receiving automatically larger sample weight. `token_mean` is logged
as a diagnostic but is not the optimization reduction.

Inference supports `official_upstream_timestep` and `roi_relative`. The latter feeds remaining ROI
mask ratio to the model instead of the upstream global discrete timestep. Training uses text-only
condition dropout 0.1; source/region conditioning is retained. LoRA can be active on both target and
reference branches or reference-only.

## 13. Formal budget

Values below are read from `configs/train/_cluster_8g_base.yaml` and `configs/eval/formal.yaml`:

| Item | Value |
| --- | --- |
| Fixed train rows | 200,000 |
| Epochs | 1 |
| GPUs | 8 |
| Batch / GPU | 1 |
| Gradient accumulation | 4 |
| Global batch | 32 |
| Optimizer steps / epoch | 6,250 |
| Resolution | 1024 |
| Train seed | 42 |
| Precision | BF16 Transformer/text, FP32 VQ |
| Optimizer | AdamW, betas 0.9/0.95, weight decay 0.01 |
| Base learning rate | 3e-5 |
| Scheduler | constant with 200 warmup steps |
| Gradient clip | 1.0 |
| LoRA | rank 64, alpha 64, dropout 0.05 |
| Checkpoints | every 625 steps, including 6,250 |
| Periodic validation | every 625 steps |
| Generation seeds | 0, 1, 2, 3 |
| Bootstrap | seed 42, 10,000 resamples, sample unit |

YAML is authoritative. If a reviewed YAML changes, update this table before running.

## 14. Learning-rate probes

The three E3 probes use `1e-5`, `3e-5`, and `5e-5`. Each runs 1,500 optimizer steps with the same
base seed, train manifest, epoch permutation, geometry/corruption sequence, batch, accumulation, and
scheduler horizon. Therefore each consumes the same first `1,500 × 32 = 48,000` rows.

```bash
bash scripts/cluster/run_lr_probes.sh --print-command

export CONFIRM_FORMAL_RUN=YES
bash scripts/cluster/run_lr_probes.sh --run
unset CONFIRM_FORMAL_RUN
```

The probe configs disable periodic generation and checkpoint at 500/1,000/1,500. Before executing,
preregister a rule based on finite/quality-gate behavior and a frozen validation protocol. Do not
choose the LR using final TEST. If the selected LR differs from the base `3e-5`, create and review a
new named formal config rather than silently editing an existing one.

## 15. E0 and E1–E4

| Run | Train | Corruption | Persistent region condition | LoRA scope | Timestep evaluation |
| --- | --- | --- | --- | --- | --- |
| E0-official | none | none | none | released | official upstream |
| E0-region | none | none | provided region | released | ROI-relative |
| E1 | fixed 200K | full target | OFF | both | ROI-relative |
| E2 | fixed 200K | ROI hard lock | OFF | both | ROI-relative |
| E3 | fixed 200K | ROI hard lock | ON | both | ROI-relative |
| E4 | fixed 200K | ROI hard lock | ON | reference only | ROI-relative |

E1–E4 share exactly the same 200K manifest, sample order, train budget, seed, optimizer family,
checkpoint schedule, and primary validation. Launch them only after corpus and 8-GPU smoke approval:

```bash
bash scripts/cluster/run_e1_e4.sh --print-command

export CONFIRM_FORMAL_RUN=YES
bash scripts/cluster/run_e1_e4.sh --run
unset CONFIRM_FORMAL_RUN
```

The current E0 launcher uses legacy frozen MagicBrush/Inter-Edit splits, not the D200K auxiliary
validation manifests. Prepare and label them explicitly:

```bash
bash scripts/cluster/prepare_interedit.sh --print-command

export CONFIRM_FORMAL_RUN=YES
bash scripts/cluster/prepare_interedit.sh --run
bash scripts/cluster/run_e0_eval.sh --run
unset CONFIRM_FORMAL_RUN
```

E0 reports must record those manifest paths and hashes so they are not confused with fixed200k full
validation.

## 16. Optional Stage A/B

Stage A/B is not part of the primary E0–E4 matrix. It may run only if the primary results justify an
additional curriculum experiment.

- Stage A: mixed corruption, ROI probability 0.50, full-ROI probability 0.15, persistent condition.
- Stage B: warm-started mixed corruption, ROI probability 0.85, full-ROI probability 0.15,
  persistent condition, warmup 100.

```bash
bash scripts/cluster/run_stage_a.sh --print-command
export STAGE_A_CHECKPOINT=/absolute/path/to/stage-a/checkpoint
bash scripts/cluster/run_stage_b.sh --print-command
```

Actual execution requires `CONFIRM_FORMAL_RUN=YES`; Stage B uses `--warm-start`, not `--resume`.
Neither stage has been run.

## 17. Validation protocol

### Periodic validation

- MagicBrush official DEV deterministic probe of 128 rows;
- generation seed 0;
- every 625 optimizer steps;
- diagnostic only and not a substitute for full DEV;
- validation saves/restores CPU/CUDA/Python/NumPy RNG and does not advance train state.

### Full validation

- complete MagicBrush official DEV with seeds `[0, 1, 2, 3]`: primary checkpoint evaluation;
- CrispEdit aux-128, ScaleEdit aux-128, Inter-Edit aux-128: separate diagnostic reports only;
- auxiliary results do not participate in primary checkpoint ranking.

```bash
export CANDIDATE_CHECKPOINT=/absolute/path/to/checkpoint
bash scripts/cluster/run_fixed200k_full_validation.sh --print-command

export CONFIRM_FORMAL_RUN=YES
bash scripts/cluster/run_fixed200k_full_validation.sh --run
unset CONFIRM_FORMAL_RUN
```

Outputs are written to
`$EDITMGT_OUTPUT_ROOT/full-validation/CHECKPOINT_NAME/DATASET/{predictions.jsonl,metrics.json}`.

## 18. Metrics

`scripts/eval/formal_eval.py` computes:

- inside-mask L1, PSNR, SSIM, and masked LPIPS-to-target;
- outside-mask L1, PSNR, SSIM, and masked LPIPS-to-source;
- full-image LPIPS-to-target;
- `d_ST`, `d_SO`, `d_OT`, and no-op progress diagnostics inside the edit region;
- optional DINO-I-to-target/source and CLIP-I-to-target/source with explicit local model roots;
- per-generation runtime;
- per-sample mean over repeated generation seeds;
- per-dataset and per-edit-type means, standard errors, and 95% sample-bootstrap intervals.

Masked LPIPS is feature-space masked aggregation: the spatial region is area-resized to each LPIPS
feature map and used to aggregate learned feature distance. It is not LPIPS on blacked-out images.

DINO/CLIP values are absent unless local embedding models are explicitly configured. The evaluator
does not download them silently.

## 19. Checkpoint selection

The desired preregistration shape is:

1. apply an outside-preservation gate on primary MagicBrush DEV;
2. among passing candidates, minimize inside masked LPIPS-to-target;
3. use preregistered no-op diagnostics as a secondary rule, never final TEST.

However, the current repository does **not** yet contain an executable selection rule. In
`configs/eval/formal.yaml`, `threshold_preserve`, `tau_edit`, and `tau_noop` are null and status is
`MUST_BE_PREREGISTERED_BEFORE_FORMAL`. The evaluator computes metrics and only emits a no-op value
when both tau thresholds are supplied; it does not rank checkpoints or apply `threshold_preserve`.

Before LR probes or formal candidate selection, freeze the exact preservation metric/direction,
threshold, tie-breaks, tau definitions, and failure policy in code/config, test them, and commit the
new experiment SHA. Do not fill thresholds after looking at candidate or TEST results.

## 20. Final TEST

Only the single method/checkpoint chosen without TEST may be evaluated on MagicBrush official TEST.
The locally audited official test contains 1,053 turns, but it is not part of corpus construction,
LR selection, periodic validation, full DEV ranking, or threshold tuning.

The repository currently has no dedicated D200K final-TEST launcher. Add and review a frozen TEST
manifest/launcher only after the selection contract is committed and the final candidate is locked;
do not repurpose the DEV launcher ad hoc. Record TEST manifest hash, code/config/checkpoint identity,
generation seeds, and all evaluator asset identities in the final report.

## 21. Current execution status

| Stage | Status |
| --- | --- |
| Local code/tests and real MagicBrush chain | PASS |
| Real CrispEdit-labeling-39k schema audit | NOT RUN |
| Real ScaleEdit-labeling-25k schema audit | NOT RUN |
| Real Inter-Edit-Train schema audit | NOT RUN |
| Real fixed 200K build and human audit | NOT BUILT / NOT RUN |
| `CORPUS_READY.json` | NOT CREATED |
| 8-GPU uninterrupted/resume smoke | NOT RUN |
| LR probes | NOT RUN |
| E0–E4 | NOT RUN |
| Stage A/B | NOT RUN |
| Final TEST | NOT RUN |

The local result is 41 unit tests passed plus a real 1024 one-step backward on one A100-PCIE-40GB.
It does not certify any missing real dataset or cluster stage.

## 22. Full command sequence

This compact sequence is an index; the review requirements in the preceding sections still apply.

```bash
# Clone and environment
git clone git@github.com:yiyezhiqiu2077/editMGT.git
cd editMGT
git checkout FORMAL_EXPERIMENT_CODE_SHA
export EDITMGT_WORKTREE="$PWD"
uv sync --frozen --group dev --group translation --group metrics

# Export all model/data/revision/derived/output/mapping variables from sections 4 and 6.
bash scripts/setup/probe_runtime.sh
bash scripts/cluster/probe_cluster.sh
uv run pytest -q

# Audit real schemas manually and freeze CrispEdit/ScaleEdit schema + edit-type mappings.

# Build and audit exact D200K
bash scripts/cluster/prepare_fixed_200k_v2.sh --print-command
export CONFIRM_CORPUS_BUILD=YES
bash scripts/cluster/prepare_fixed_200k_v2.sh --run
unset CONFIRM_CORPUS_BUILD
uv run python scripts/data/verify_corpus_ready.py \
  "$DERIVED_ROOT/fixed200k/CORPUS_READY.json"

# 8-GPU smoke gate
bash scripts/cluster/run_8g_smoke.sh --print-command
export CONFIRM_FORMAL_RUN=YES
bash scripts/cluster/run_8g_smoke.sh --run
unset CONFIRM_FORMAL_RUN

# After committing the preregistered selection rule: LR probes
bash scripts/cluster/run_lr_probes.sh --print-command
export CONFIRM_FORMAL_RUN=YES
bash scripts/cluster/run_lr_probes.sh --run
unset CONFIRM_FORMAL_RUN

# Released baselines
bash scripts/cluster/prepare_interedit.sh --print-command
bash scripts/cluster/run_e0_eval.sh --print-command

# Formal ablations
bash scripts/cluster/run_e1_e4.sh --print-command
export CONFIRM_FORMAL_RUN=YES
bash scripts/cluster/run_e1_e4.sh --run
unset CONFIRM_FORMAL_RUN

# Full DEV + auxiliary validation for a preregistered candidate
export CANDIDATE_CHECKPOINT=/absolute/path/to/checkpoint
bash scripts/cluster/run_fixed200k_full_validation.sh --print-command
export CONFIRM_FORMAL_RUN=YES
bash scripts/cluster/run_fixed200k_full_validation.sh --run
unset CONFIRM_FORMAL_RUN
```

Do not proceed to final TEST until one candidate is selected under the committed DEV-only rule.
