# EditMGT Explicit-Region Editing

本仓库基于已发布的 [EditMGT](https://github.com/weichow23/editmgt)，用于研究带显式编辑区域（explicit mask / provided region）的离散图像编辑，以及在此基础上的 one-step Region-DiMO 蒸馏。

仓库主要包含三部分：

- **Explicit-region SFT**：在给定 edit mask 的条件下训练 EditMGT；
- **D200K 数据流程**：统一使用 MagicBrush、CrispEdit、ScaleEdit 和 Inter-Edit 构建固定训练语料；
- **Region-DiMO**：将多步编辑模型进一步蒸馏为 one-step editing model。

## 快速开始

```bash
git clone git@github.com:yiyezhiqiu2077/editMGT.git
cd editMGT

uv sync --frozen --group dev --group translation --group metrics
```

空服务器可以直接指定统一资产目录：

```bash
export ASSET_ROOT=/large_disk/editmgt_assets
export HF_HOME=/large_disk/editmgt_hf_cache   # 可选

bash scripts/setup/run_formal_prepare.sh --dry-run
bash scripts/setup/run_formal_prepare.sh --run
```

该流程会准备 EditMGT、NLLB、训练数据与评测模型，并生成 D200K 所需的数据文件。

## D200K

训练语料固定为 200,000 条，来源为：

| 数据集 | 使用方式 |
|---|---|
| MagicBrush train | 使用全部合格样本 |
| CrispEdit-labeling-39k | 最多 39,000 条 |
| ScaleEdit-labeling-25k | 最多 25,000 条 |
| Inter-Edit-Train | 补齐至 200,000 条 |

中文 instruction 使用 `facebook/nllb-200-distilled-1.3B` 统一翻译为英文，再写入训练使用的 `instruction_en`。

## Explicit-region SFT

核心训练入口：

```bash
scripts/train/train_explicit_region.py
```

主要实现：

- source / target / mask 共享几何变换；
- pixel mask → VQ token mask；
- ROI hard-lock corruption；
- edit-region embedding；
- ROI-relative timestep；
- target / reference LoRA scope；
- per-sample normalized masked-token loss；
- deterministic resume。

8-GPU 训练入口：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CONFIRM_FORMAL_RUN=YES

bash scripts/cluster/run_8g_smoke.sh --run
bash scripts/cluster/run_lr_probes.sh --run
bash scripts/cluster/run_e1_e4.sh --run
```

### E0–E4

| Experiment | Corruption / region condition | LoRA scope |
|---|---|---|
| E0-official | released model / official timestep | released weights |
| E0-region | released model / ROI-relative timestep | released weights |
| E1 | full target / OFF | both |
| E2 | ROI hard lock / OFF | both |
| E3 | ROI hard lock / ON | both |
| E4 | ROI hard lock / ON | reference only |

E1–E4 使用相同的 D200K、sample order、seed 和训练预算。

## Region-DiMO

Region-DiMO 用于把 explicit-region EditMGT 从多步生成蒸馏为 one-step editing：

```text
Explicit-region SFT
        ↓
selected dense checkpoint
        ↓
teacher model
        ↓
Region-DiMO
        ↓
one-step editing
```

实现包括：

- teacher / student / auxiliary 三个角色；
- ROI 内 MASK / random-token 初始化；
- ROI 外 source token hard lock；
- teacher / auxiliary distribution matching；
- surrogate loss；
- auxiliary update；
- student EMA；
- raw student / EMA one-step inference。

Teacher 注册入口：

```bash
uv run python scripts/dimo/register_teacher.py \
  --selected-checkpoint "$CANDIDATE_CHECKPOINT" \
  --selected-checkpoint-record "$ASSET_ROOT/artifacts/SELECTED_CHECKPOINT.json" \
  --formal-assets "$ASSET_ROOT/artifacts/formal_assets.json" \
  --output-dir "$ASSET_ROOT/models/dimo_teacher"
```

DiMO 训练与推理入口：

```text
scripts/train/train_dimo_editing.py
scripts/eval/sample_dimo_editing.py
```

详细方法见 [docs/DIMO_PORTING.md](docs/DIMO_PORTING.md)。

## 项目结构

```text
configs/
  data/                  D200K 与数据配置
  train/                 Explicit-region SFT 配置
  eval/                  评测配置
  dimo/                  Region-DiMO 配置

src/
  explicit_region/       数据、geometry、mask、SFT 公共组件
  dimo/                  Region-DiMO 实现

scripts/
  setup/                 资产与环境准备
  data/                  D200K 构建与翻译
  train/                 SFT / DiMO 训练
  eval/                  评测与 one-step inference
  cluster/               多 GPU 运行入口
  dimo/                  Teacher 注册

tests/                   单元测试与集成测试
docs/                    实验与实现说明
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
- [数据定义](docs/DATA_CONTRACT.md)
- [代码结构](docs/ARCHITECTURE.md)
- [Region-DiMO](docs/DIMO_PORTING.md)
- [环境配置](docs/ENVIRONMENT.md)

## 上游与许可证

本仓库基于官方 [EditMGT](https://github.com/weichow23/editmgt) 项目进行研究开发，上游 [CC-BY-4.0 许可证](LICENSE)保持不变。
