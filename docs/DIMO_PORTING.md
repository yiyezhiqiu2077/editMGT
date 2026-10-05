# Region-DiMO porting notes

Status: **CODE PREP ONLY**. `SELECTED_DIMO_TEACHER = NONE` and
`DIMO_EDIT_FORMAL_READY = false`.

This document records the code-level reference used for the Region-DiMO
preparation. It is not a result claim and none of the values below are frozen
formal hyperparameters.

## Reference and attribution

- Upstream method: *Di[M]O: Distilling Masked Diffusion Models into One-step
  Generator* (ICCV 2025).
- Upstream repository: <https://github.com/yuanzhi-zhu/DiMO>
- Reviewed commit: `24613741ec9ca730273a6ebc822327878d5a086b`
- Reviewed files: `train_DIMO_Meissonic.py`,
  `configs/DIMO_Meissonic.yaml`, `sample_Meissonic.py`, and
  `scripts/launch_DIMO_Meissonic.sh`.

The implementation in `src/dimo` is an independent, minimal adaptation. The
upstream repository is neither vendored nor imported at runtime.

## A. What the upstream Meissonic code actually does

The upstream training code loads a pretrained Meissonic transformer as a
frozen teacher, deep-copies it as the trainable generator, and deep-copies it
again as the trainable `fake_model` (the auxiliary approximation to the current
generator distribution). Its one-step generator first fills the whole image
token grid with uniform random codebook tokens, replaces an exact fixed-ratio
subset with MASK, runs one transformer forward at that ratio, filters logits
with optional temperature/top-k/top-p, and samples discrete tokens.

For a generator update, `get_mask_code` pseudo-masks an exact subset of the
sampled image. Teacher and fake model query that same pseudo-state. The code
forms `p_true` and `p_fake`; FKL uses `p_fake - p_true`, RKL uses
`p_fake * (log(p_fake) - log(p_true) - D_RKL)`, and Jeffreys is the configured
linear mixture. It zeros this vector outside the pseudo-mask. The generator is
trained through `0.5 * mse(z, stopgrad(z-g))`; sampled tokens are discrete and
do not provide a straight-through path.

The fake-model update independently generates samples, pseudo-masks them, and
trains on masked positions. The reference configuration uses hard-token CE
(`alpha_fake: 0`). Upstream may consume a different dataloader batch for this
update. Generator update, fake update, and then generator EMA make one logical
cycle. The reference configuration uses cosine ratios, fixed initial ratio
0.5, embedding perturbation 0.3, and EMA 0.9999. These are reference defaults,
not Region-DiMO choices.

The upstream perturbation is applied immediately after token embedding as
`sigma * randn_like(h) + sqrt(1-sigma**2) * h`. Upstream generates that noise
inside the transformer from the global RNG. This port preserves the formula,
but supplies deterministic noise from the caller and limits it to target ROI.

Upstream checkpoint resume uses Accelerate state plus an additional
`global_step`; sampling otherwise relies substantially on global RNG state.
This port intentionally strengthens that contract with stateless per-sample,
per-purpose streams and saves the fixed-corpus cursor and stream identity.

## B. Mapping to explicit-region EditMGT

The “Original” column describes the reviewed upstream code. The “Adaptation”
column is new EditMGT-specific design and must not be presented as an original
DiMO contribution.

| Concern | Original DiMO / Meissonic | EditMGT explicit-region adaptation |
| --- | --- | --- |
| Teacher | Pretrained masked generator | Selected dense explicit-region EditMGT checkpoint; selection is pending |
| Logical roles | Frozen teacher plus full generator and fake-model copies | Frozen shared base with teacher/student/auxiliary role adapters and role-specific region vectors |
| Generator input | Full-image mixture of MASK and random tokens | Source copied outside ROI; MASK/random initialization only inside ROI |
| Condition | Text | Clean source reference + editing instruction + full edit region |
| Forward corruption | Masks anywhere in image | Masks only inside editable ROI |
| Mask ratio | Relative to full image token grid | Relative to each sample's ROI token count |
| Generator output | Samples every image token | Samples ROI only; outside ROI is overwritten with source exactly |
| Embedding perturbation | Whole generator target embedding, noise drawn inside model | Target ROI only, externally generated deterministic noise; never reference/text/outside ROI |
| Auxiliary model | Tracks current generator distribution | Same source/text/full-region-conditioned EditMGT role as student |
| DiMO gradient support | Pseudo-masked full-grid positions | ROI pseudo-masked positions only, per-sample normalized |
| CFG unconditional branch | Text unconditional prediction | Empty text only; target state, clean source reference, region mask, and region embedding remain identical |
| Auxiliary batch | May consume another batch | Reuses the same fixed-corpus batch; no additional sample accounting |
| RNG | Predominantly global RNG plus resume state | Stateless seed of base seed, epoch, UID, committed DiMO step, and purpose namespace |
| Checkpoint boundary | Training loop checkpoint | Only after student + auxiliary + EMA complete |

### Role and gradient contract

All three roles originate from one `DIMO_TEACHER_CHECKPOINT`. The frozen base,
teacher adapter, and teacher region embedding never receive gradients and are
never optimizer members. Student and auxiliary adapters/region embeddings are
copied from the teacher state, then optimized by disjoint optimizers. Adapter
switching and `edit_region_embedding_override` allow one base allocation.

Student and auxiliary adapters use the teacher's rank, alpha, targets, and
initial weights, but their DiMO LoRA dropout is explicitly fixed to `0.0`.
The teacher retains its SFT dropout configuration and always runs in eval
mode. This is an initialization policy, not a change to the selected teacher.

The formal checkpoint contract includes compatible base identity, LoRA state,
region-embedding state, content fingerprint, training-config identity, source
Git SHA, selected checkpoint identity, and an explicit `formal_teacher` flag.
The normal training entry point rejects anything else with
`DIMO_TEACHER_NOT_SELECTED`. `--prep-smoke` may use a compatible temporary
checkpoint, emits `PREP_ONLY_TEACHER=true` and
`TEACHER_QUALITY_NOT_VALIDATED=true`, and is hard-limited to two optimizer
steps.

### One complete Region-DiMO step

One fixed-corpus batch supplies clean source tokens, instruction, complete ROI,
UID, and global sample index. The student receives a deterministic ROI-only
MASK/random initial state and full clean reference, then performs one forward.
ROI tokens are sampled from detached logits and outside tokens are hard-locked.
A teacher-query pseudo-mask is sampled inside ROI; teacher and auxiliary are
queried without gradient, and their divergence vector drives the student's
surrogate loss. After the student optimizer step, the same detached sample and
same conditioning batch receive an independent auxiliary pseudo-mask and
per-sample hard CE update. EMA runs last. Only then does
`committed_dimo_step` advance and become checkpointable.

### Determinism and resume

The independent namespaces are `dimo-init-mask`, `dimo-init-token`,
`dimo-student-sample`, `dimo-teacher-query-mask`, `dimo-aux-query-mask`, and
`dimo-embedding-noise`. A checkpoint contains committed step, both optimizers
and schedulers, both adapter/region states, student EMA, epoch, fixed-corpus
cursor, samples consumed, config hash, teacher bundle fingerprint/hash,
inference fingerprint, and this upstream commit.
Resume begins at the next complete step; partial student-only checkpoints are
invalid.

## PREP-v1.1 correctness patch

PREP-v1.1 leaves Region initialization, ROI pseudo-forward, FKL/RKL/Jeffreys
formulae, and the same-batch auxiliary update unchanged. It narrows six
correctness contracts:

- one-step student inference explicitly selects eval mode and runs under
  `torch.inference_mode()`; training calls retain their previous train-mode
  default;
- teacher/auxiliary distribution algebra and the complete vocabulary-SUM
  surrogate execute in FP32, with per-sample masked normalization unchanged;
- gradient-checkpoint recomputation is regression-tested after the intervening
  teacher and auxiliary adapter switches;
- teacher identity attests the adapter, region embedding, trainable config,
  optional fingerprint/manifest, and released base-model identity as one
  canonical bundle;
- inference selects `--weights student|ema` (default `student`) and fails closed
  on schema, teacher bundle, upstream commit, or inference fingerprint mismatch;
- student/auxiliary LoRA dropout is explicitly zero, while teacher dropout is
  inherited and inactive in eval mode.

`number_of_transformer_forwards=1` counts model invocations. With CFG greater
than one, that one invocation uses a batched unconditional/conditional input of
size `2B`; metadata records `effective_cfg_batch_multiplier=2`.

## Formal teacher handoff

唯一允许的正式 teacher 交接链为：

```text
E1/E2/E3/E4
→ MagicBrush DEV pre-registered selector
→ scripts/eval/register_selected_checkpoint.py
→ SELECTED_CHECKPOINT.json
→ scripts/dimo/register_teacher.py
→ dimo_teacher_manifest.json
→ DIMO_TEACHER_CHECKPOINT
```

注册命令：

```bash
source "$ASSET_ROOT/artifacts/formal_env.sh"

uv run python scripts/dimo/register_teacher.py \
  --selected-checkpoint "$CANDIDATE_CHECKPOINT" \
  --selected-checkpoint-record "$ASSET_ROOT/artifacts/SELECTED_CHECKPOINT.json" \
  --formal-assets "$ASSET_ROOT/artifacts/formal_assets.json" \
  --output-dir "$ASSET_ROOT/models/dimo_teacher"

export DIMO_TEACHER_CHECKPOINT="$ASSET_ROOT/models/dimo_teacher"
```

注册器不信任输入路径：它重新计算 adapter、region embedding、trainable config 和 fingerprint 的 SHA256，核对 selected-checkpoint identity、formal-assets hash、formal Git SHA，并规范化比较 checkpoint 与 formal ledger 中的 `repo_id` 和 40 位 `resolved_revision`。bundle 在同一文件系统的临时目录完成两次 contract 验证后原子 rename；相同 bundle 可幂等重放，不同的既有目录会 fail closed。

Formal teacher registration 只证明 handoff provenance 完整，不代表 Region-DiMO 已允许正式训练。`DIMO_EDIT_FORMAL_READY` 和配置中的 `formal_ready` 仍保持 false；本 milestone 不存在绕过参数。

### Formal teacher selection status

Current values are:

```text
SELECTED_DIMO_TEACHER = NONE
DIMO_EDIT_FORMAL_READY = false
```

After teacher selection, candidate studies may compare FKL, RKL, and Jeffreys,
initial mask ratio, perturbation sigma, and LoRA versus full finetuning. No such
matrix is generated or run in this preparation milestone. The first future
comparison should include the dense teacher, a naive teacher one-step baseline,
and Region-DiMO one-step student under a frozen evaluation protocol.
