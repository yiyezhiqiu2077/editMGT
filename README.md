# EditMGT Explicit-Region Editing

本仓库基于 [EditMGT](https://github.com/weichow23/editmgt)，研究 explicit-mask / provided-region 图像编辑，包括 Explicit-region SFT、D200K 数据流程和 Region-DiMO one-step distillation。

## 快速开始

```bash
git clone --branch fix/e3-quality-pipeline-v1 git@github.com:yiyezhiqiu2077/editMGT.git
cd editMGT

uv sync --frozen --group dev --group translation --group metrics

export ASSET_ROOT=/large_disk/editmgt_assets
export HF_HOME=/large_disk/editmgt_hf_cache   # 可选

bash scripts/setup/run_formal_prepare.sh --dry-run
bash scripts/setup/run_formal_prepare.sh --run
```

准备脚本下载模型和数据，生成后续训练使用的环境文件与 D200K manifest。资产版本定义在 `configs/formal_assets.yaml`。

## D200K

D200K 固定包含 200,000 条训练记录：

| 数据集 | 使用方式 |
|---|---|
| MagicBrush train | 使用全部合格样本 |
| CrispEdit-labeling-39k | 最多 39,000 条 |
| ScaleEdit-labeling-25k | 最多 25,000 条 |
| Inter-Edit-Train | 补齐至 200,000 条 |

四个数据集使用统一的 canonical record、edit type 和 mask 语义。中文 instruction 由 `facebook/nllb-200-distilled-1.3B` 从 `zho_Hans` 翻译到 `eng_Latn`，训练读取 `instruction_en`。

数据格式和选择规则见 [D200K 数据定义](docs/DATA_CONTRACT.md)。

## Explicit-region SFT

训练实现包括：

- source、target、mask 共享几何变换；
- pixel mask 到 VQ token mask 的映射；
- ROI hard-lock corruption；
- trainable region embedding；
- ROI-relative timestep；
- target / reference LoRA scope；
- per-sample normalized CE；
- deterministic sample order 与 resume。

单个训练入口：

```bash
uv run python scripts/train/train_explicit_region.py \
  --config configs/train/local_one_step.yaml
```

已有 frozen D200K 时，按 [实验流程](docs/EXPERIMENTS.md) 配置现有路径并导入 `formal_env.sh`，然后运行：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CONFIRM_FORMAL_RUN=YES

bash scripts/cluster/reattest_existing_fixed200k.sh
bash scripts/cluster/run_8g_smoke.sh --run
bash scripts/train/run_cluster_e3.sh --run
```

正式训练仅运行 E3：5 epochs / 31,250 optimizer steps，LR=3e-5。训练完成后基于 DEV 选择 checkpoint。

### E0–E4

| Experiment | Corruption / region condition | LoRA scope |
|---|---|---|
| E0-official | released model / official timestep | released weights |
| E0-region | released model / ROI-relative timestep | released weights |
| E1 | full target / OFF | both |
| E2 | ROI hard lock / OFF | both |
| E3 | ROI hard lock / ON | both |
| E4 | ROI hard lock / ON | reference only |

E0-official 为 released baseline；E0-region、E1、E2、E4 保留为后续可选 ablation，不在本次 E3-only 正式训练中执行。

## Region-DiMO

```text
Explicit-region SFT
→ selected dense checkpoint
→ teacher model
→ Region-DiMO
→ one-step editing
```

核心实现包括：

- teacher / student / auxiliary 三个角色；
- ROI 内 MASK / random-token 初始化；
- ROI 外 source token hard lock；
- teacher / auxiliary distribution matching；
- surrogate loss 与 auxiliary update；
- student EMA；
- raw student / EMA one-step inference。

将选中的 SFT checkpoint 注册为 teacher：

```bash
uv run python scripts/dimo/register_teacher.py \
  --selected-checkpoint "$CANDIDATE_CHECKPOINT" \
  --selected-checkpoint-record "$ASSET_ROOT/artifacts/SELECTED_CHECKPOINT.json" \
  --formal-assets "$ASSET_ROOT/artifacts/formal_assets.json" \
  --output-dir "$ASSET_ROOT/models/dimo_teacher"

export DIMO_TEACHER_CHECKPOINT="$ASSET_ROOT/models/dimo_teacher"
```

训练与推理入口为 `scripts/train/train_dimo_editing.py` 和 `scripts/eval/sample_dimo_editing.py`。方法说明见 [Region-DiMO](docs/DIMO_PORTING.md)。

## 项目结构

```text
configs/                 数据、训练、评测与 DiMO 配置
src/explicit_region/     canonical data、geometry、mask 与 SFT
src/dimo/                Region-DiMO 训练与 one-step generation
scripts/setup/           环境和资产准备
scripts/data/            D200K 构建与翻译
scripts/train/           SFT / DiMO 训练
scripts/eval/            validation、TEST 与推理
scripts/dimo/            teacher 注册
scripts/cluster/         多 GPU 运行入口
tests/                   单元测试与集成测试
docs/                    实验和方法文档
```

## 测试

```bash
uv run pytest -q
uv run python -m compileall src scripts tests
find scripts -name '*.sh' -print0 | xargs -0 -n1 bash -n
git diff --check
```

## 文档

- [实验流程](docs/EXPERIMENTS.md)
- [D200K 数据定义](docs/DATA_CONTRACT.md)
- [代码结构](docs/ARCHITECTURE.md)
- [Region-DiMO](docs/DIMO_PORTING.md)
- [环境配置](docs/ENVIRONMENT.md)
