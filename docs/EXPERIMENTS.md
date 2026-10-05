# 实验流程

所有命令均在仓库根目录执行。环境和资产路径说明见 [ENVIRONMENT.md](ENVIRONMENT.md)，D200K 格式见 [DATA_CONTRACT.md](DATA_CONTRACT.md)。

## 1. 环境

```bash
git clone git@github.com:yiyezhiqiu2077/editMGT.git
cd editMGT

uv sync --frozen --group dev --group translation --group metrics

export ASSET_ROOT=/large_disk/editmgt_assets
export HF_HOME=/large_disk/editmgt_hf_cache   # 可选
# export HF_ENDPOINT=https://your-hf-mirror.example
```

`ASSET_ROOT` 保存模型、数据、D200K 和实验输出。Inter-Edit 体积较大，建议使用持久化大磁盘。

## 2. 数据准备

先检查下载和构建计划：

```bash
bash scripts/setup/run_formal_prepare.sh --dry-run
```

准备全部资产并构建 D200K：

```bash
bash scripts/setup/run_formal_prepare.sh --run
source "$ASSET_ROOT/artifacts/formal_env.sh"
```

该脚本准备 EditMGT、NLLB、MagicBrush、CrispEdit、ScaleEdit、Inter-Edit、DINO、CLIP 和 LPIPS，并生成 exact 200K 训练 manifest。中文 instruction 使用 NLLB 翻译为英文。

## 3. 8-GPU smoke

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash scripts/cluster/run_8g_smoke.sh --print-command
export CONFIRM_FORMAL_RUN=YES
bash scripts/cluster/run_8g_smoke.sh --run
```

该任务检查 8-GPU 训练、checkpoint 保存和 resume 后的 sample sequence。

## 4. LR probes

LR probes 使用 `1e-5`、`3e-5` 和 `5e-5`：

```bash
bash scripts/cluster/run_lr_probes.sh --print-command
export CONFIRM_FORMAL_RUN=YES
bash scripts/cluster/run_lr_probes.sh --run
```

checkpoint selection 只使用 MagicBrush DEV，不使用 final TEST。

## 5. E0–E4

| 实验 | 训练语义 | LoRA scope |
|---|---|---|
| E0-official | released model，official timestep | released weights |
| E0-region | released model，ROI-relative timestep | released weights |
| E1 | full target，region condition OFF | both |
| E2 | ROI hard lock，region condition OFF | both |
| E3 | ROI hard lock，region condition ON | both |
| E4 | ROI hard lock，region condition ON | reference only |

E1–E4 共享 D200K、sample order、seed、训练预算和 validation protocol。

```bash
bash scripts/cluster/run_e0_eval.sh --print-command
bash scripts/cluster/run_e0_eval.sh --run

bash scripts/cluster/run_e1_e4.sh --print-command
export CONFIRM_FORMAL_RUN=YES
bash scripts/cluster/run_e1_e4.sh --run
```

## 6. Validation

MagicBrush DEV 是 primary validation。CrispEdit、ScaleEdit 和 Inter-Edit 的 aux-128 分开报告。

```bash
export CANDIDATE_CHECKPOINT="$ASSET_ROOT/experiments/<run>/checkpoint-6250"

bash scripts/cluster/run_fixed200k_full_validation.sh --print-command
export CONFIRM_FORMAL_RUN=YES
bash scripts/cluster/run_fixed200k_full_validation.sh --run
```

## 7. Final TEST

先登记由 DEV selection 得到的 checkpoint：

```bash
uv run python scripts/eval/register_selected_checkpoint.py \
  --checkpoint "$CANDIDATE_CHECKPOINT" \
  --formal-assets "$ASSET_ROOT/artifacts/formal_assets.json" \
  --output "$ASSET_ROOT/artifacts/SELECTED_CHECKPOINT.json"
```

运行 MagicBrush TEST：

```bash
bash scripts/eval/run_magicbrush_test.sh --print-command
bash scripts/eval/run_magicbrush_test.sh --run
```

## 8. Region-DiMO

Region-DiMO 使用选中的 dense SFT checkpoint 作为 teacher：

```text
selected checkpoint
→ SELECTED_CHECKPOINT.json
→ dimo_teacher_manifest.json
→ DIMO_TEACHER_CHECKPOINT
→ Region-DiMO training
```

注册 teacher：

```bash
uv run python scripts/dimo/register_teacher.py \
  --selected-checkpoint "$CANDIDATE_CHECKPOINT" \
  --selected-checkpoint-record "$ASSET_ROOT/artifacts/SELECTED_CHECKPOINT.json" \
  --formal-assets "$ASSET_ROOT/artifacts/formal_assets.json" \
  --output-dir "$ASSET_ROOT/models/dimo_teacher"

export DIMO_TEACHER_CHECKPOINT="$ASSET_ROOT/models/dimo_teacher"
```

训练入口：

```bash
uv run python scripts/train/train_dimo_editing.py \
  --config configs/dimo/base.yaml
```

one-step inference 使用 `scripts/eval/sample_dimo_editing.py`，可通过 `--weights student` 或 `--weights ema` 选择权重。算法细节见 [DIMO_PORTING.md](DIMO_PORTING.md)。
