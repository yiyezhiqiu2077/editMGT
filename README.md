# EditMGT Explicit-Region Editing Experiments

This is a research fork and experimental repository derived from the released
[EditMGT](https://github.com/weichow23/editmgt) codebase. It studies image editing with a
provided region / explicit mask. The first milestone is an auditable mask-aware SFT baseline;
the resulting dense checkpoint is intended to support later dense, cache, sparse, and shortcut
generation-acceleration studies.

The repository contains experiment code and reproducibility gates, not completed formal results.
No performance improvement is claimed here. The complete runbook is in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Current status

| Item | Status |
| --- | --- |
| Local D200K-v2 implementation | DONE |
| Local unit tests | 41 passed |
| 1024 one-step backward on one A100 40GB | PASS |
| Fixed real 200K corpus | NOT BUILT |
| Real CrispEdit-labeling-39k audit | NOT RUN |
| Real ScaleEdit-labeling-25k audit | NOT RUN |
| Real Inter-Edit-Train audit | NOT RUN |
| 8-GPU fixed-corpus smoke / resume smoke | NOT RUN |
| LR probes and formal E1–E4 | NOT RUN |
| Optional Stage A/B | NOT RUN |

“Implementation complete” does not mean that the corpus or formal experiments are complete.
`CORPUS_READY.json` has not been created from the real four-dataset corpus, so formal training must
not be launched yet.

## Project structure

```text
editMGT/
├── configs/
│   ├── cluster/
│   ├── data/
│   ├── eval/
│   ├── train/
│   └── translation/
├── docs/
├── scripts/
│   ├── cluster/
│   ├── data/
│   ├── eval/
│   ├── setup/
│   └── train/
├── src/
│   └── explicit_region/
├── tests/
├── train/                    # retained upstream training code
├── pyproject.toml
└── uv.lock
```

The formal explicit-region training entry point is
[`scripts/train/train_explicit_region.py`](scripts/train/train_explicit_region.py), not the retained
upstream `train/train.py`. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Environment

The locked project environment uses Python `>=3.10,<3.11`, `uv`, PyTorch 2.1.2 with the CUDA 12.1
wheel index, Diffusers 0.32.1, Transformers 4.47.1, and PEFT 0.14.0.

```bash
git clone git@github.com:yiyezhiqiu2077/editMGT.git
cd editMGT
uv sync --frozen --group dev --group translation --group metrics

uv run python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

Models and data are local-only assets and are not downloaded silently. Full setup and storage
guidance is in [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).

## Models and data

This repository does not publish model weights, raw datasets, translation weights, checkpoints,
evaluation-model weights, derived corpora, caches, or experiment outputs. Runtime locations are
provided with environment variables:

```text
EDITMGT_MODEL_ROOT
MAGICBRUSH_ROOT
MAGICBRUSH_DEV_ROOT
CRISPEDIT_ROOT
SCALEEDIT_ROOT
INTEREDIT_ROOT
TRANSLATOR_MODEL_ROOT
DERIVED_ROOT
EDITMGT_OUTPUT_ROOT
```

Raw datasets are treated as immutable and read-only. Dataset and model revisions must be recorded
with immutable identifiers before a formal build.

## D200K-v2

The planned fixed corpus contains exactly 200,000 rows selected without replacement from:

- MagicBrush official train: all eligible rows;
- CrispEdit-labeling-39k: capped at 39,000 rows;
- ScaleEdit-labeling-25k: capped at 25,000 rows;
- quality-filtered Inter-Edit: fills the remaining quota.

Those are policies and caps, not observed final contributions; the real corpus has not been built.
The frozen epoch is deterministically permuted and rank-strided. With 8 GPUs, batch/GPU 1, and
gradient accumulation 4, global batch is 32 and one 200K epoch is 6,250 optimizer steps.

Schema audit, canonicalization, translation, duplicate handling, audit, and READY attestation are
described in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Explicit-region SFT

The experiment layer adds:

- shared source / target / region geometry;
- pixel-to-token edit-region mapping;
- ROI hard-lock corruption and loss only on the selected masked subset;
- persistent edit-region embedding;
- ROI-relative inference timestep;
- target/reference LoRA scope control over a frozen backbone;
- content-bound fingerprints and deterministic optimizer-boundary resume.

## Formal experiments

All long runs below are currently **NOT RUN**.

| Run | Corruption / mask condition | LoRA scope |
| --- | --- | --- |
| E0-official | released model, upstream timestep | released weights |
| E0-region | released model, ROI-relative timestep | released weights |
| E1 | `full_target`, mask condition OFF | both |
| E2 | ROI hard lock, mask condition OFF | both |
| E3 | ROI hard lock, mask condition ON | both |
| E4 | ROI hard lock, mask condition ON | reference only |

E1–E4 use the same fixed 200K rows, permutation, budget, and primary validation protocol. The only
intended differences are listed above.

## Training

Formal shell entry points default to printing commands. `--run` additionally requires the explicit
confirmation variable documented in the runbook.

```bash
# Fixed-corpus preparation
bash scripts/cluster/prepare_fixed_200k_v2.sh --print-command

# 8-GPU uninterrupted/resume smoke
bash scripts/cluster/run_8g_smoke.sh --print-command

# 1e-5 / 3e-5 / 5e-5 probes
bash scripts/cluster/run_lr_probes.sh --print-command

# E1–E4
bash scripts/cluster/run_e1_e4.sh --print-command
```

After review, the corresponding operation uses the same launcher with `--run`. See
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) before executing any of them.

## Evaluation

MagicBrush official DEV is the primary validation set. MagicBrush official TEST is final-test-only
and must not be used for tuning or checkpoint selection. The formal evaluator is
[`scripts/eval/formal_eval.py`](scripts/eval/formal_eval.py).

Implemented measurements include inside/outside L1, PSNR, SSIM, feature-space masked LPIPS,
full-image LPIPS-to-target, optional DINO-I and CLIP-I similarities to target/source, no-op raw
diagnostics, runtime, repeated-seed sample means, standard errors, and sample-level bootstrap
confidence intervals. DINO/CLIP evaluation requires explicit local model paths and does not download
weights implicitly.

## Tests

```bash
uv run pytest -q
uv run python -m compileall src scripts tests
```

The recorded `41 passed` result is the current D200K-v2 local validation result, not a GitHub CI
claim and not evidence that the unrun cluster experiments passed.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Environment and assets](docs/ENVIRONMENT.md)
- [Complete experiment runbook](docs/EXPERIMENTS.md)

## Upstream, citation, and license

This repository is derived from [the original EditMGT repository](https://github.com/weichow23/editmgt).
It is an experimental fork and does not imply that its maintainer is an author of the original
EditMGT paper.

Original paper: *EditMGT: Unleashing Potentials of Masked Generative Transformers in Image Editing*.

```bibtex
@article{chow2025editmgt,
  title={EditMGT: Unleashing Potentials of Masked Generative Transformers in Image Editing},
  author={Chow, Wei and Li, Linfeng and Kong, Lingdong and Li, Zefeng and Xu, Qi and Song, Hang and Ye, Tian and Wang, Xian and Bai, Jinbin and Xu, Shilin and others},
  journal={arXiv preprint arXiv:2512.11715},
  year={2025}
}
```

The upstream [CC-BY-4.0 license](LICENSE) is preserved unchanged.
