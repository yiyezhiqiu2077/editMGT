# EditMGT 显式区域编辑实验

本仓库是基于已发布 [EditMGT](https://github.com/weichow23/editmgt) 代码建立的研究分支，主要研究带给定区域（provided region）或显式掩码（explicit mask）的图像编辑。当前第一个里程碑是建立一套可审计的 mask-aware SFT 基线；后续计划以得到的 dense checkpoint 为基础，继续研究 dense、cache、sparse 和 shortcut 等生成加速方向。

仓库目前提供的是实验代码和可复现门禁，并不代表正式实验已经完成，也不宣称获得了性能提升。完整实验流程见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 当前状态

| 项目 | 状态 |
| --- | --- |
| 本地 D200K-v2 实现 | 已完成 |
| 本地单元测试 | 41 passed |
| 单张 A100 40GB 上的 1024 一步反传 | 通过 |
| 真实固定 200K 语料 | 尚未构建 |
| 真实 CrispEdit-labeling-39k 审计 | 未运行（NOT RUN） |
| 真实 ScaleEdit-labeling-25k 审计 | 未运行（NOT RUN） |
| 真实 Inter-Edit-Train 审计 | 未运行（NOT RUN） |
| 8-GPU 固定语料 smoke / resume smoke | 未运行（NOT RUN） |
| LR probes 与正式 E1–E4 | 未运行（NOT RUN） |
| 可选 Stage A/B | 未运行（NOT RUN） |

“代码实现完成”不等于“语料或正式实验完成”。当前尚未从真实四数据集生成 `CORPUS_READY.json`，因此不得启动正式训练。

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
├── train/                    # 保留的上游训练代码
├── pyproject.toml
└── uv.lock
```

正式 explicit-region 训练入口是 [`scripts/train/train_explicit_region.py`](scripts/train/train_explicit_region.py)，不是保留的上游 `train/train.py`。代码组织详见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 环境安装

锁定环境使用 Python `>=3.10,<3.11`、`uv`、PyTorch 2.1.2（CUDA 12.1 wheel）、Diffusers 0.32.1、Transformers 4.47.1 和 PEFT 0.14.0。

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

模型和数据均使用本地资产，正式代码不会静默下载它们。完整的软件、存储和资产配置见 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)。

## 模型与数据

本仓库不提交模型权重、原始数据集、翻译模型权重、checkpoint、评估模型权重、衍生语料、缓存或实验输出。运行时通过以下环境变量提供路径：

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

所有原始数据集都应视为不可变只读资产。正式构建前，必须用不可变标识记录数据集和模型 revision。

## D200K-v2

计划中的固定语料恰好包含 200,000 条记录，来源及选择策略为：

- MagicBrush official train：使用全部合格记录；
- CrispEdit-labeling-39k：最多 39,000 条；
- ScaleEdit-labeling-25k：最多 25,000 条；
- 经过质量筛选的 Inter-Edit：补齐剩余额度。

以上是策略和上限，不是已经观测到的最终数据集占比；真实语料尚未构建。冻结语料采用确定性的无放回 epoch permutation 和 rank-stride 切分。使用 8 GPU、每卡 batch 1、梯度累积 4 时，global batch 为 32，一个 200K epoch 恰好包含 6,250 个 optimizer steps。

schema audit、canonicalization、翻译、去重、数据审计和 READY attestation 的完整说明见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## Explicit-region SFT

本实验层增加了以下能力：

- source、target、region 共用的配对几何变换；
- pixel region 到 VQ token region 的映射；
- ROI hard-lock corruption，并只在选中的 masked subset 上计算 loss；
- 持续生效的 edit-region embedding；
- ROI-relative inference timestep；
- 冻结 backbone 上的 target/reference LoRA scope 控制；
- 内容绑定的 fingerprint，以及 optimizer boundary 上的确定性 resume。

## 正式实验

以下所有 long run 当前均为**未运行（NOT RUN）**。

| 实验 | Corruption / mask condition | LoRA scope |
| --- | --- | --- |
| E0-official | released model，upstream timestep | released weights |
| E0-region | released model，ROI-relative timestep | released weights |
| E1 | `full_target`，mask condition OFF | both |
| E2 | ROI hard lock，mask condition OFF | both |
| E3 | ROI hard lock，mask condition ON | both |
| E4 | ROI hard lock，mask condition ON | reference only |

E1–E4 使用相同的固定 200K 记录、permutation、训练预算和 primary validation protocol，预期差异仅限表中项目。

## 训练入口

正式 shell 入口默认只打印命令。真正执行 `--run` 时，还必须设置实验文档中规定的显式确认变量。

```bash
# 构建固定语料
bash scripts/cluster/prepare_fixed_200k_v2.sh --print-command

# 8-GPU uninterrupted/resume smoke
bash scripts/cluster/run_8g_smoke.sh --print-command

# 1e-5 / 3e-5 / 5e-5 probes
bash scripts/cluster/run_lr_probes.sh --print-command

# E1–E4
bash scripts/cluster/run_e1_e4.sh --print-command
```

审核打印出的命令并满足所有门禁后，使用同一个 launcher 的 `--run` 模式执行。运行前必须阅读 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 评估

MagicBrush official DEV 是 primary validation set；MagicBrush official TEST 只能用于最终测试，不得参与调参或 checkpoint selection。正式 evaluator 位于 [`scripts/eval/formal_eval.py`](scripts/eval/formal_eval.py)。

当前实现的指标包括 inside/outside L1、PSNR、SSIM、feature-space masked LPIPS、full-image LPIPS-to-target、可选的 DINO-I/CLIP-I target/source similarity、no-op 原始诊断、运行时间、多 seed sample mean、standard error 和 sample-level bootstrap confidence interval。DINO/CLIP 评估必须显式配置本地模型路径，不会隐式下载权重。

## 测试

```bash
uv run pytest -q
uv run python -m compileall src scripts tests
```

记录中的 `41 passed` 是当前 D200K-v2 本地验证结果，不是 GitHub CI 声明，也不能证明尚未运行的集群实验已经通过。

## 文档

- [代码架构](docs/ARCHITECTURE.md)
- [环境与资产](docs/ENVIRONMENT.md)
- [完整实验流程](docs/EXPERIMENTS.md)


