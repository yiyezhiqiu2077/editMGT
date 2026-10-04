# EditMGT Explicit-Region Editing Experiments

本仓库用于在 EditMGT 上进行带 provided region / explicit mask 的图像编辑实验，当前支持固定 200K mixed corpus 的构建流程、ROI hard-lock corruption、edit-region conditioning、LoRA 微调、DDP、确定性 resume 和 GT-mask 评测。

第一阶段目标是建立稳定的 mask-aware SFT baseline；后续将在得到的 dense checkpoint 上继续研究 dense、cache、sparse 和 shortcut 等生成加速方向。

完整实验流程见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 项目结构

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
├── train/
├── pyproject.toml
└── uv.lock
```

正式 explicit-region 训练入口为 [`scripts/train/train_explicit_region.py`](scripts/train/train_explicit_region.py)。保留的上游 `train/train.py` 不作为本实验的正式训练入口。

详细结构见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 环境安装

环境固定使用 Python `>=3.10,<3.11`、PyTorch 2.1.2（CUDA 12.1 wheel）、Diffusers 0.32.1、Transformers 4.47.1 和 PEFT 0.14.0。

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
详细环境配置见 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)。

## 模型与数据

模型、原始数据、翻译模型、checkpoint、缓存和实验输出均不提交到仓库，通过环境变量指定本地路径：

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

计划使用固定的 `D200K-v2` 训练语料，总计 200,000 条记录，数据来源和选择策略为：

| 数据来源 | 选择策略 |
| --- | --- |
| MagicBrush official train | 使用全部合格记录 |
| CrispEdit-labeling-39k | 最多 39,000 条 |
| ScaleEdit-labeling-25k | 最多 25,000 条 |
| Inter-Edit-Train | 经过质量筛选后补齐至 200,000 条 |

这些数值是选择策略和上限；实际数据集占比以冻结后的 manifest 为准。

数据准备流程为 raw datasets → schema audit → canonicalization → translation → deduplication / filtering → fixed D200K-v2 manifest → `CORPUS_READY.json`。


| 项目 | 值 |
| --- | --- |
| GPU | 8 |
| Batch / GPU | 1 |
| Gradient Accumulation | 4 |
| Global Batch | 32 |
| Samples / Epoch | 200,000 |
| Steps / Epoch | 6,250 |

详细数据准备和审计流程见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## Explicit-Region Training

当前训练实现支持：

- source / target / region shared geometry preprocessing
- pixel region → VQ token region mapping
- ROI hard-lock corruption
- masked-subset generation loss
- persistent edit-region embedding
- ROI-relative inference timestep
- target / reference LoRA scope control
- content-bound fingerprint
- optimizer-boundary deterministic resume

其中 `edit_region_hardlock` 模式只对 ROI 内选中的 target token subset 进行 corruption，并在该 subset 上计算 generation loss；ROI 外 token 保持锁定。

## 正式实验

正式实验包括两个 released-model baseline 和四组 SFT：

| Experiment | Corruption / Mask Condition | LoRA Scope |
| --- | --- | --- |
| E0-official | released model + upstream timestep | released weights |
| E0-region | released model + ROI-relative timestep | released weights |
| E1 | `full_target`, mask condition OFF | both |
| E2 | `edit_region_hardlock`, mask condition OFF | both |
| E3 | `edit_region_hardlock`, mask condition ON | both |
| E4 | `edit_region_hardlock`, mask condition ON | reference only |

E1–E4 使用相同的 D200K-v2 corpus、training permutation、global batch、training budget 和 primary validation protocol，主要用于比较 ROI hard-lock、explicit region conditioning 和 LoRA scope 的影响。

正式实验前进行 `1e-5 / 3e-5 / 5e-5` learning-rate probes。

## 训练

固定语料准备：

```bash
bash scripts/cluster/prepare_fixed_200k_v2.sh --print-command
bash scripts/cluster/prepare_fixed_200k_v2.sh --run
```

8-GPU uninterrupted / resume smoke：

```bash
bash scripts/cluster/run_8g_smoke.sh --print-command
bash scripts/cluster/run_8g_smoke.sh --run
```

Learning-rate probes：

```bash
bash scripts/cluster/run_lr_probes.sh --print-command
bash scripts/cluster/run_lr_probes.sh --run
```

E1–E4 正式训练：

```bash
bash scripts/cluster/run_e1_e4.sh --print-command
bash scripts/cluster/run_e1_e4.sh --run
```

Launcher 默认只打印命令。运行前检查 `--print-command` 输出，满足对应 corpus、asset 和 experiment gate，并设置 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) 规定的显式确认变量后，再使用 `--run` 执行。

## 分布式训练

正式训练 topology 为 `1 node × 8 GPUs`。

数据加载和训练采用 PyTorch distributed training，并使用固定的无放回 epoch permutation + rank-stride partition 保证各 rank 数据划分确定。

Resume 在 optimizer boundary 上恢复，并检查 corpus / model fingerprint。

## 评测

MagicBrush official DEV 为 primary validation set；MagicBrush official TEST 为 final test set，不参与 learning-rate selection、checkpoint selection 或其他超参数调节。

正式 evaluator 为 [`scripts/eval/formal_eval.py`](scripts/eval/formal_eval.py)。

当前主要指标和诊断包括：

- Inside / Outside L1
- PSNR / SSIM
- Feature-space masked LPIPS
- Full-image LPIPS-to-target
- 可选的 DINO-I target / source similarity
- 可选的 CLIP-I target / source similarity
- No-op 原始诊断
- Runtime

正式评测同时记录 multi-seed sample mean、standard error 和 sample-level bootstrap confidence interval。

DINO / CLIP evaluator 使用显式配置的本地模型权重，不自动下载模型。

详细评测协议见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。
。

## 测试

```bash
uv run pytest -q
uv run python -m compileall src scripts tests
```

## 文档

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
