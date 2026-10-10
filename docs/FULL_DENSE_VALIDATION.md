# Local Full DenseDiMO validation — 2026-10-10

Scope: `exp/e3-dense-fp32-v1`, based on
`21b5b26c1a81c9780d067cd33e1b92a2a39cef83`. No new experiment branch.
E3 Dense training configuration and mathematical implementation are unchanged;
the SFT entry only adds durable scalar-log flushing and excludes GH_TOKEN from
its launch environment.

## Executed locally

- Full CPU pytest: **262 passed**, 17 dependency deprecation warnings.
- Two-process Gloo: fresh versus resume, full bitwise comparison of role weights,
  both AdamW optimizers, schedulers, EMA, sampler/loader state and each rank RNG.
- Single-rank injected failure after Student optimizer, after Auxiliary
  optimizer, during EMA, after EMA, and before commit: both workers terminate;
  no false complete checkpoint or committed cursor.
- Analytical FP64/FP32 surrogate gradients, tiny fields and logit scales;
  CPU BF16-autocast layer outputs preserve the linear objective gradient.
- Tiny real repository Transformer, FP32 CPU complete steps, with gradient
  checkpointing enabled and disabled. All trainable gradients are covered.
- First Student zero gradient accepted; Auxiliary updates and subsequent
  Student signal verified. EMA arithmetic and count tested at decay 0.9995.
- Group/source lineage leakage rejection, sample-weighted cluster bootstrap,
  paired seeds/UIDs, exact eligibility boundaries and earlier-step tie-break.
- Inference-only loader cannot clone other Dense roles or load optimizer pickle.
- Monitor upload failures, missing authentication/CLI and SIGKILL do not control
  training. PNG whitelist, pending recovery and local report integrity tested.
- Separate two-rank synthetic smoke: **20 complete steps**, sampler cursor 40,
  both optimizer states and EMA; local plots and lightweight archive generated.
- compileall, shell syntax checks and `git diff --check`: PASS.

Development smoke artifacts remain under local `runs/` and are not published.
Synthetic artifacts created before the final commit record their then-current
Git parent; they are development evidence, not formal experiment artifacts.

## Not executed / admission requirements

Actual 8-GPU DDP/NCCL, CUDA BF16 real-model gradient audit, first-update VRAM,
real-data role isolation and GPU fresh/resume equivalence:
**NOT_RUN_RESOURCE_UNAVAILABLE**. This host exposes four A800 80GB GPUs; none
were allocated for this task. CPU BF16 tests do not establish CUDA acceptance.
The installed CPU attention path does not support every real Transformer BF16
operation; the real Transformer CPU test therefore uses FP32 explicitly.

FSDP candidate: **UNVALIDATED**, not a production fallback.
No full D200K download, SFT long run, formal DEV selection, selected Teacher
export, 25,000-step DiMO run or real quality conclusion was produced locally.
GitHub curve upload tests used fault/mocked publishers; no synthetic curves were
uploaded to an experiment release.

Before formal DEV: supply immutable canonical DEV/TEST manifests and reviewed
original-source/sequence evidence, audit D200K isolation, then preregister the
selection contract. Before formal DiMO: supply the selected Teacher, frozen
corpus READY, model identity, sufficient output disk and exclusive eight GPUs.
Runtime steps1/20/500 record measured acceptance and stop on hard failures.
Complete GPU resume evidence requires independent fresh20 and fresh10→resume20
runs followed by `scripts/train/compare_full_dense_resume.py`.

Executable stage commands: [EXPERIMENTS.md](EXPERIMENTS.md).
Algorithm/state details: [FULL_DENSE_PROTOCOL.md](FULL_DENSE_PROTOCOL.md).
