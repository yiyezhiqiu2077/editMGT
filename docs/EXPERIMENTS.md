# 实验流程

所有命令在仓库根目录执行。环境和数据定义见 [ENVIRONMENT.md](ENVIRONMENT.md) 与 [DATA_CONTRACT.md](DATA_CONTRACT.md)。

主流程：已有 frozen D200K → re-attestation → 8-GPU smoke → E3-only 五轮训练 → DEV checkpoint selection → TEST / teacher / Region-DiMO。

## 1. 环境准备

```bash
git clone --branch fix/e3-quality-pipeline-v1 git@github.com:yiyezhiqiu2077/editMGT.git
cd editMGT
uv sync --frozen --group dev --group translation --group metrics

export ASSET_ROOT=/large_disk/editmgt_assets
export HF_HOME=/large_disk/editmgt_hf_cache   # 可选
```

模型和数据版本由 `configs/formal_assets.yaml` 固定。使用持久化大磁盘，并保持工作树干净。

## 2. 数据

### 已有 frozen D200K（推荐）

使用实际运行目录、资产目录和已有环境文件：

```bash
export EDITMGT_RUN_ROOT=/path/to/existing/run
export EDITMGT_ASSET_ROOT=/path/to/existing/assets
export ASSET_ROOT="$EDITMGT_ASSET_ROOT"
source "$ASSET_ROOT/artifacts/formal_env.sh"

bash scripts/cluster/reattest_existing_fixed200k.sh
```

环境文件需提供 `EDITMGT_MODEL_ROOT` 及模型、canonical data、derived、output 路径；若文件位于其他位置，使用其实际路径。
re-attestation 不重建、不重新采样、不修改 D200K，只验证冻结数据并更新验证记录。通过后再运行 smoke。

### 从零准备（可选）

没有现成 D200K 时，下载资产并构建固定 200,000 条训练记录：

```bash
bash scripts/setup/run_formal_prepare.sh --dry-run
bash scripts/setup/run_formal_prepare.sh --run
source "$ASSET_ROOT/artifacts/formal_env.sh"

# 默认准备脚本将数据与验证记录放在 ASSET_ROOT 下。
export EDITMGT_RUN_ROOT="$ASSET_ROOT"
export EDITMGT_ASSET_ROOT="$ASSET_ROOT"
bash scripts/cluster/reattest_existing_fixed200k.sh
```

## 3. D200K 8-GPU smoke

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CONFIRM_FORMAL_RUN=YES
bash scripts/cluster/run_8g_smoke.sh --print-command
bash scripts/cluster/run_8g_smoke.sh --run
```

检查真实 D200K 的 20-step 训练与 10-step + resume、梯度、checkpoint 和恢复一致性。通过后进入正式训练。

## 4. E3-only formal training

E3 固定使用 ROI hard-lock、persistent region conditioning=ON、LoRA scope=both。
正式配置为 [cluster_8g_e3.yaml](../configs/train/cluster_8g_e3.yaml)。

```bash
export CONFIRM_FORMAL_RUN=YES
bash scripts/train/run_cluster_e3.sh --print-command
bash scripts/train/run_cluster_e3.sh --run
```

训练以数据 re-attestation 和 smoke 为前置条件，不要求 selection preregistration。

## 5. 正式训练预算

| 参数 | 设置 |
|---|---|
| D200K | 200,000 samples |
| GPU / batch/GPU / gradient accumulation | 8 / 1 / 4 |
| Global batch | 32 |
| Epochs | 5 |
| Optimizer steps | 每 epoch 6250，共 31250 |
| LR / warmup | 3e-5 / 200 steps，全程只 warmup 一次 |
| Scheduler | constant_with_warmup |

Checkpoint steps：`3125, 6250, 9375, 12500, 15625, 18750, 21875, 25000, 28125, 31250`。

最后一个 checkpoint 不一定最佳；训练完成后基于 DEV 选择。
独立 LR probes 保留用于未来实验，不属于本次固定 LR=3e-5 的 E3-only 流程。

## 6. Periodic validation

使用 [e3_periodic_probe.yaml](../configs/eval/e3_periodic_probe.yaml)，每 3125 steps 在已有固定 MagicBrush DEV128 上进行 diagnostic。
记录 inside/outside 指标与质量趋势，不自动选择 checkpoint 或 teacher。
CFG=5 仅用于训练期间 diagnostic，不是正式最优 CFG 声明。

## 7. Formal checkpoint selection

训练完成后，在 `configs/eval/formal.yaml` 预注册 selection 规则，再使用固定 DEV 比较各 checkpoint。
MagicBrush DEV 为 primary validation；CrispEdit、ScaleEdit、Inter-Edit 的 aux-128 分项报告。不得使用 TEST 选择 checkpoint。

DEV 生成和指标入口分别为 [generate_evaluation.py](../scripts/eval/generate_evaluation.py) 与 [formal_eval.py](../scripts/eval/formal_eval.py)。
确定所选 checkpoint 后，登记其内容身份：

```bash
export CANDIDATE_CHECKPOINT="$EDITMGT_OUTPUT_ROOT/e3-roi-condition-both/checkpoint-<selected_step>"
uv run python scripts/eval/register_selected_checkpoint.py \
  --checkpoint "$CANDIDATE_CHECKPOINT" \
  --formal-assets "$ASSET_ROOT/artifacts/formal_assets.json" \
  --output "$ASSET_ROOT/artifacts/SELECTED_CHECKPOINT.json"
```

## 8. TEST

只有 selected checkpoint 确定并登记后，才运行 MagicBrush TEST：

```bash
bash scripts/eval/run_magicbrush_test.sh --print-command
bash scripts/eval/run_magicbrush_test.sh --run
```

## 9. Region-DiMO

Region-DiMO 是 E3 checkpoint selection 之后的阶段，不默认使用最后一个 checkpoint：

```text
selected checkpoint → SELECTED_CHECKPOINT.json
→ teacher → Region-DiMO → one-step editing
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

训练入口为 `scripts/train/train_dimo_editing.py`，one-step inference 入口为 `scripts/eval/sample_dimo_editing.py`，支持 `--weights student` / `--weights ema`。
方法和配置见 [DIMO_PORTING.md](DIMO_PORTING.md)。

## Optional ablations

- E0-official：released baseline，official timestep。
- E0-region：released baseline，ROI-relative timestep。
- E1：full target、region conditioning OFF、LoRA both。
- E2：ROI hard-lock、region conditioning OFF、LoRA both。
- E4：ROI hard-lock、region conditioning ON、LoRA reference only。

E0-region / E1 / E2 / E4 保留为后续 ablation，当前 E3-only 正式训练不执行。
