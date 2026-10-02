# Repository architecture

This repository keeps the released EditMGT model stack and adds an explicit-region experiment layer.
Reusable experiment logic lives in `src/explicit_region/`; executable workflows live in `scripts/`;
immutable parameters live in `configs/`; behavioral contracts live in `tests/`.

## Upstream and experiment layers

The retained upstream modules provide the Transformer, scheduler, pipeline, text encoders, VQ-VAE,
and prompt/image utilities:

- `src/editmgt.py`: released pipeline/component assembly.
- `src/transformer.py`: EditMGT Transformer, extended with region embedding and branch-scope control.
- `src/pipeline.py`: generation pipeline, extended with provided-mask hard lock and selectable timestep
  protocol.
- `src/scheduler.py`, `src/v2_model.py`, `src/v2_utils.py`: upstream discrete generation and VQ
  helpers.
- `src/dataset_utils.py`, `src/local_utils.py`: upstream prompt and image utilities.

`src/explicit_region/` is the new experiment/training layer. It depends on upstream primitives; the
upstream modules do not own corpus selection, DDP sampling, formal evaluation, or experiment gates.

## `src/explicit_region/`

| Module | Responsibility |
| --- | --- |
| `canonical.py` | Portable canonical records, FILE/TAR/PARQUET/bbox locators, content hashes, and strict asset verification |
| `fixed_corpus.py` | Deterministic D200K selection, source/group deduplication, Inter-Edit strata, backfill, and READY attestation |
| `fixed_dataset.py` | Immutable manifest-index runtime reader; integrity failures terminate instead of replacing rows |
| `epoch_sampler.py` | Versioned hash-sort epoch permutation, rank-stride DDP partition, and resume cursor |
| `geometry.py` | Shared paired geometry, mask-retention attempts, and deterministic contain/center-pad fallback |
| `corruption.py` | `full_target`, ROI hard-lock, mixed corruption, selected-token labels, and ROI-ratio schedule |
| `conditioning.py` | Region-mask expansion/duplication helpers for model and CFG paths |
| `masks.py` | Pixel-region to VQ-token mask conversion |
| `losses.py` | Per-sample normalized cross-entropy and token-mean diagnostic |
| `checkpoint.py` | LoRA/region-embedding state, optimizer/scheduler/quality state, fingerprint-checked resume, and warm start |
| `metrics.py` | Masked L1/PSNR/SSIM, feature-space masked LPIPS, cosine similarity, and finite aggregation |
| `quality.py` | Finite/logit/clip/update-ratio checks and rolling-loss quality gate |
| `language.py` | Han detection, revision-bound translation cache identity, and translation QA flags |
| `crispedit.py`, `scaleedit.py` | Explicit schema-mapped dataset adapters; no real-field guessing |
| `interedit.py` | Inter-Edit archive metadata and `better_data` parsing |
| `dataset.py` | MagicBrush/Inter-Edit aligned datasets and deterministic local/core data paths |
| `validation.py` | Frozen periodic generation/evaluation without advancing training RNG/state |
| `modeling.py`, `contracts.py` | Local-only component loading, model identity, freezing, trainable-parameter and asset contracts |
| `config.py`, `deterministic.py` | Strict YAML inheritance/environment expansion and stable seed utilities |

## Executable workflows

### `scripts/data/`

Owns the offline data lifecycle:

- inspect real schemas without inferring adapters;
- build MagicBrush, schema-mapped CrispEdit/ScaleEdit, and Inter-Edit canonical pools;
- freeze primary and auxiliary validation sets;
- select/deduplicate/backfill the fixed corpus;
- run revision-bound NLLB translation and QA;
- audit content, diversity, geometry, VQ containment, and montages;
- create and verify `CORPUS_READY.json`.

### `scripts/train/`

[`train_explicit_region.py`](../scripts/train/train_explicit_region.py) is the only formal
explicit-region training entry point. It resolves configuration, verifies the worktree/model/corpus,
injects LoRA, constructs the sampler, executes training/validation, and saves provenance and
checkpoint state.

The other files in this directory compose 8-GPU launch commands and verify uninterrupted versus
resumed smoke runs. The retained upstream `train/train.py` is not the formal entry point for this
experiment.

### `scripts/eval/`

- `generate_evaluation.py`: generate frozen-manifest predictions with explicit model/checkpoint and
  timestep protocol.
- `formal_eval.py`: compute per-generation, per-sample-after-seed-mean, grouped, and bootstrap metrics.
- `inference_smoke.py`: pipeline-only local smoke; its outputs are ineligible for checkpoint selection.
- audit/montage helpers: diagnose reference aliasing and compare local smoke variants.

### `scripts/cluster/`

Provides guarded orchestration for asset preparation, D200K build, 8-GPU smoke, LR probes, E0,
E1–E4, optional Stage A/B, and full four-dataset validation. Formal launchers default to
`--print-command`; state-changing execution requires `--run` and the relevant confirmation variable.

### `scripts/setup/`

Provides the frozen `uv` bootstrap, runtime probe, asset audit, worktree validation, and optional
upstream-worktree creation. Machine-specific locations enter only through environment variables.

## Configuration ownership

| Directory | Owns |
| --- | --- |
| `configs/data/` | Fixed-200K policy, raw schema mappings, canonical edit-type mapping |
| `configs/train/` | Local checks, smoke/resume runs, LR probes, E1–E4, Stage A/B |
| `configs/eval/` | Local smoke, periodic probe, formal generation/metric protocol |
| `configs/translation/` | NLLB backend, immutable revision, language pair, decoding |
| `configs/cluster/` | Cluster-visible model and dataset root contract |

YAML is authoritative for experiment numbers. Documentation summarizes it but must be updated if a
formal config changes.

## Dependency direction

```text
configs ───────────────┐
                      v
scripts/cluster -> scripts/{data,train,eval,setup} -> src/explicit_region -> upstream src modules
                                                     ^
tests ------------------------------------------------┘
```

In particular, `scripts -> src` and `tests -> src`. Reusable corpus/training/metric logic must not be
hidden in cluster shell scripts, and library modules must not depend on a specific server layout.

## Data path and readiness

```text
read-only raw datasets
  -> schema audit
  -> explicit schema/edit-type mappings
  -> canonical pools with sample_uid and content SHA256
  -> frozen validation
  -> deterministic candidate/reserve selection and source deduplication
  -> selected-row translation and deterministic QA backfill
  -> exact train_200k.jsonl
  -> hash/read/geometry/FP32-VQ/montage audits
  -> CORPUS_READY.json
  -> 8-GPU smoke
  -> formal training
```

`CORPUS_READY.json` binds the train manifest, metadata, selection/dedup/translation reports,
translation cache, validation manifests, audit report, manual translation sample, and montages by
hash. It proves machine-checkable corpus consistency; it does not replace human review of licensing,
montages, or translations.

## Training data path

For each frozen row:

1. Verify locators and source/target/region hashes.
2. Align source and target to the region coordinate system.
3. Apply identical deterministic geometry to all three.
4. Encode source/target with the FP32 VQ-VAE and map the pixel region to the VQ grid.
5. Construct full-target or ROI hard-lock corruption.
6. Encode text with frozen text encoders and run the LoRA-equipped Transformer.
7. Average cross-entropy within each sample's selected set, then average across the batch.

The base Transformer, text encoders, and VQ-VAE are frozen. Trainable state is LoRA plus
`edit_region_embedding`. E4 restricts active LoRA scope to the reference branch.

## Determinism and resume

The formal sampler schema is `fixed200k-hash-sort-v1`:

```text
permutation = hash_sort(base_seed, epoch, train_manifest_sha256)
rank r      = permutation[r::world_size]
```

It uses no padding, dropping, or replacement. Checkpoints bind code/config/data/model identity,
world size, batch/accumulation, sampler schema, epoch permutation, optimizer step, and committed
global sample cursor.

- `--resume`: restores trainable weights, optimizer, scheduler, quality state, step, and cursor after
  fingerprint validation.
- `--warm-start`: loads only LoRA and region embedding, then starts a new optimizer/scheduler/cursor.

These modes are mutually exclusive.

## Evaluation boundary

MagicBrush official DEV is primary validation. The fixed DEV probe-128/seed-0 run is diagnostic;
full DEV/seeds 0–3 is used for formal candidate evaluation. CrispEdit, ScaleEdit, and Inter-Edit
aux-128 sets are reported separately and are diagnostic. MagicBrush official TEST remains sealed
until one method/checkpoint is selected without using TEST.

The current formal selection thresholds in `configs/eval/formal.yaml` are intentionally null and
marked `MUST_BE_PREREGISTERED_BEFORE_FORMAL`; no completed checkpoint-selection rule or formal result
is claimed.

## Non-negotiable invariants

- Raw data and released model snapshots are read-only and never committed.
- Formal assets load locally; no silent model or tokenizer fallback is allowed.
- A missing or stale READY marker blocks D200K smoke/training/full-validation.
- A bad frozen row is an integrity error, not a signal to replace the sample at runtime.
- `torch_compile` is false for the v1 correctness contract.
- Periodic validation does not advance training RNG or optimizer state.
- Final TEST cannot be used for hyperparameter, threshold, or checkpoint selection.
