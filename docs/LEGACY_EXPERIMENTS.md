# Historical experiment notes

Imported from 21b5b26c1a81c9780d067cd33e1b92a2a39cef83; historical claims below are not current Full DenseDiMO acceptance evidence.

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

## 10. Dense E3：独立开发、集群训练与 Dense Teacher

本节与前述 LoRA 流程独立。Dense 分支为 `exp/e3-dense-fp32-v1`，基线为
`36dff5ff9ad10cee3d3910eb980932ec0a251b62`。不要切换正在训练的 LoRA 工作树；使用独立 clone、输出目录及进程端口。
部署 SHA 使用本分支最终交付的 commit，以下记为 `$DENSE_COMMIT`。

### 本地验证与边界

本地测试入口为 `uv run pytest -q`，CPU 用例明确为合成模型，不代表 released 模型编辑质量或真实 GPU 验收。
真实 Mini-512 Smoke 使用 [run_dense_local_smoke.sh](../scripts/train/run_dense_local_smoke.sh)，执行 fresh16、fresh8 和 resume-to16。
结果位于 `$DENSE_OUTPUT_ROOT/DENSE_SMOKE.json`、`logs/`、各运行的 `train_metrics.jsonl`、`memory-rank*.jsonl` 及完整 checkpoint。
一致性标准预先固定为 bitwise exact，不在失败后放宽容差。
正式 D200K、正式 DEV selection 和 Region-DiMO 长训在本地均为 `NOT_RUN`；GPU 状态以实际结果文件为准，不能把 CPU PASS 复用为 GPU PASS。

2026-10-10 本地验收：真实八张 A100 40GB 上 fresh16 / fresh8→resume16 通过，完整权重、optimizer、scheduler、sampler、各 rank RNG、QualityGate 和样本 trace 均 bitwise exact。
训练执行 SHA 为 `b5f7be522cfe0f8c40fe95e7d618a46d99d2166e`；后续提交补充评测门禁、独立 Dense probe 配置和本文档，不改变训练数学。
可训练及 optimizer 参数均为 1,009,307,648，全部 FP32，无 LoRA、结构性 unused 或 encoder gradient。
首次更新 peak allocated 21.26 GiB；最高 allocated / reserved 为 28.08 / 28.63 GiB；普通 step 中位数7.11秒，不含 checkpoint I/O。

本地证据路径由交付报告记录，分别记作 `$LOCAL_DENSE_SMOKE_ROOT` 和 `$LOCAL_DENSE_AUX_ROOT`：

- `$LOCAL_DENSE_SMOKE_ROOT/DENSE_SMOKE.json`：真实 Dense 训练及严格恢复。
- `$LOCAL_DENSE_AUX_ROOT/real-mini-dev/`：固定 DEV32、CFG10、12 steps、seed0，四模型128张图，指标/配对CI/固定案例；仅 diagnostic，selection 为 PENDING。
- `$LOCAL_DENSE_AUX_ROOT/in-memory-probe-final/IN_MEMORY_PROBE.json`：真实 DEV8、CFG5 内存推理，权重、RNG、scheduler 和训练模式不受影响。
- `$LOCAL_DENSE_AUX_ROOT/DIMO_GPU_RESUME.json` 和 `dimo-one-step/DIMO_ONE_STEP.json`：真实八卡 DiMO 两步及严格恢复、四数据集 raw/EMA 八张单步图，finite 和 token-lock 检查通过。

已实际导出并验证 Mini16 provisional Teacher。Selected export 的门禁由 CPU 用例覆盖，真实 selected Teacher 尚未产生。
Smoke 仅16步、仍在200步 warmup 内，不能据此判断 Dense 学习质量，也未覆盖完整 QualityGate baseline 窗口或正式 DEV128。
首次开发 Smoke 暴露的 DataLoader generator 重复推进问题已修复；这里只采用修复后的严格恢复证明，不放宽容差。
配置文件 identity 可用 `sha256sum configs/train/dense_d200k_5epoch.yaml` 检查，实际 resolved config/fingerprint 保存在各运行的 `run_provenance.json`。

### 集群环境

先申请独占八张 A100，并保证 checkpoint 磁盘容量。每个 Dense checkpoint 约需完整 FP32 权重与两份 AdamW moment；十个 checkpoint 及原子写入临时目录需预留足够空间，建议不少于 160GB，不含数据、评测图像和 Teacher。

```bash
export DENSE_CODE_ROOT=/path/to/separate/dense/repository
export DENSE_COMMIT=<本次交付的完整commit_SHA>
git clone --branch exp/e3-dense-fp32-v1 git@github.com:yiyezhiqiu2077/editMGT.git "$DENSE_CODE_ROOT"
cd "$DENSE_CODE_ROOT"
git checkout --detach "$DENSE_COMMIT"
uv sync --frozen --group dev --group translation --group metrics

export FORMAL_ENV_FILE=/path/to/existing/formal_env.sh
source "$FORMAL_ENV_FILE"           # 只读取已有环境，不执行数据准备脚本
export DENSE_EXPECTED_GIT_SHA="$DENSE_COMMIT"
export DENSE_OUTPUT_ROOT=/path/to/new/dense/experiment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DENSE_MASTER_PORT=29671
```

环境文件须提供 `EDITMGT_MODEL_ROOT`、`DERIVED_ROOT`、`MAGICBRUSH_ROOT`、`CRISPEDIT_ROOT`、`SCALEEDIT_ROOT`、`INTEREDIT_ROOT`、`MAGICBRUSH_DEV_ROOT`、`TRANSLATOR_REVISION`。
`EDITMGT_OUTPUT_ROOT` 保留原环境定义；Dense 自己只向 `DENSE_OUTPUT_ROOT` 写产物。
`DERIVED_ROOT/fixed200k/` 必须包含已有 `train_200k.jsonl`、metadata、`CORPUS_READY.json`、翻译缓存及 validation manifests。
所有数据根目录必须指向原 canonical 数据，不重新下载、翻译、采样或修正 manifest。

### 只读门禁与启动

Dense **不要调用**前述会更新共享记录的 `reattest_existing_fixed200k.sh`。
新入口逐条验证原冻结数据/geometry、训练与验证隔离、已有 READY 文件身份，并只写 Dense 自己的证明。
它不会覆盖共享 `CORPUS_READY.json` 或 `formal_assets.json`。

```bash
uv run python scripts/train/verify_e3_dense_config.py --static
uv run python scripts/train/verify_e3_dense_config.py --reattest
bash scripts/train/run_dense_cluster.sh --print-command
export CONFIRM_DENSE_RUN=YES
bash scripts/train/run_dense_cluster.sh --run

# 完整恢复：相同配置、代码和数据，保留原训练输出目录
bash scripts/train/run_dense_cluster.sh --print-command --resume "$DENSE_OUTPUT_ROOT/train/checkpoint-6250"
bash scripts/train/run_dense_cluster.sh --run --resume "$DENSE_OUTPUT_ROOT/train/checkpoint-6250"
```

`--print-command` 只做静态检查；`--run` 检查代码 SHA、干净工作树、released 权重、Dense 独立 attestation、DEV identity、八卡进程占用/可用显存、输出路径和 resume 格式。
Dense Smoke 证明不能借用 LoRA 的 smoke PASS。集群正式任务本身在 step1/4/16 内嵌执行 dtype、梯度、AdamW 状态、全 rank 参数一致性和样本消费检查；通过后继续同一任务，不重启、不重置 warmup。
本地未执行真实八卡 Smoke 时，不得提前写成已通过；集群早期检查也不替代 fresh/resume 等价性的证明。

### 固定配方与监控

配置为 [dense_d200k_5epoch.yaml](../configs/train/dense_d200k_5epoch.yaml)：

| 参数 | Dense E3 |
|---|---|
| 初始化 / 训练范围 | released 原始 FP32 Transformer；完整 Transformer + region embedding |
| 冻结 / LoRA | CLIP、Gemma、VQ 冻结；不创建或加载任何 LoRA |
| 精度 / 前向 | trainable parameters FP32；BF16 autocast |
| 目标 | 原 E3 per-sample CE、ROI hard-lock、persistent conditioning |
| 数据 / 预算 | frozen D200K；8×1×4=global batch32；5 epochs / 31250 steps |
| LR / warmup | 2e-6；200 steps，全程一次；不搜索 LR |
| Optimizer | AdamW，betas=(0.9,0.95)，weight_decay=0.01，clip=1.0 |
| Checkpoint | 3125 的整数倍，共十个 half-epoch checkpoint |
| 后端 | 现有 math SDP、gradient checkpointing、Accelerate/DDP，不自动切 FSDP/ZeRO |

`trainable_parameter_report.json` 和 `initialization_report.json` 记录原始权重 dtype、直接加载一致性、optimizer 覆盖及冻结参数。
`train_metrics.jsonl` 记录 CE、成功更新/调度计数、cursor、样本 UID/geometry/corruption trace、step time 和 QualityGate。
详细更新诊断默认每100步及早期检查点采集，使用每参数前4096个元素的代表性采样，**不是完整模型更新范数**；未采集字段为 null。
显存记录包含加载、DDP 包装、首次 forward/backward 和 optimizer 更新后的 allocated/reserved/peak。
低 CE 不能证明编辑能力提升；参数和恢复正常也不自动完成 DEV selection。
训练期 DEV128 每3125步使用 CFG5 diagnostic，不声明最优 CFG。
使用 [dense_periodic_probe.yaml](../configs/eval/dense_periodic_probe.yaml)，不依赖可选 DINO/CLIP 资产；旧 LoRA probe 配置不变。

### 本地真实 Mini-512 Smoke

只在八卡完全空闲时执行。与正式任务保持1024、batch/GPU1、accumulation4、LR2e-6、warmup200、horizon31250和相同数学/后端；stop 参数只控制本次截止点。

```bash
export MINI_ROOT=/path/to/existing/four_dataset_512
export DENSE_OUTPUT_ROOT=/path/to/new/local/dense-smoke
bash scripts/train/run_dense_local_smoke.sh --print-command
bash scripts/train/run_dense_local_smoke.sh --run
```

Mini manifest 固定为256/64/64/128，预检只读，不改变样本身份。
如果 fresh16 发生 OOM，立即停止，不继续 fresh8/resume，不自动减小分辨率、batch 或更换训练精度。

### 十个 checkpoint 的 DEV 评测

评测提前实现，不需在集群修改 Python。使用 [dense_dev_plan.yaml](../configs/eval/dense_dev_plan.yaml) 与 [dense_dev.yaml](../configs/eval/dense_dev.yaml)。
所有模型使用相同输入、geometry、seeds、steps、CFG、FP32 Transformer/BF16 autocast 和 FP32 VQ。
E0-official 保留 official timestep；E0-region、LoRA E3 和 Dense E3 使用 ROI-relative timestep，分别报告，不能混成同一个基线。

```bash
export E3_LORA_CHECKPOINT=/path/to/frozen/lora/checkpoint
export DENSE_MAGICBRUSH_DEV_MANIFEST=/path/to/frozen/official/dev.jsonl
export DENSE_CRISPEDIT_AUX_MANIFEST=/path/to/frozen/crispedit/aux.jsonl
export DENSE_SCALEEDIT_AUX_MANIFEST=/path/to/frozen/scaleedit/aux.jsonl
export DENSE_INTEREDIT_AUX_MANIFEST=/path/to/frozen/interedit/aux.jsonl
export CUDA_VISIBLE_DEVICES=0          # 独立分配的空闲评测卡
uv run python scripts/eval/evaluate_dense_checkpoints.py --plan configs/eval/dense_dev_plan.yaml
```

输出目录为 `$DENSE_OUTPUT_ROOT/dev_evaluation`：per-generation/per-sample metrics、JSON/CSV comparison table、quality trend、配对 bootstrap CI 和固定顺序案例图。
Inside 对 target、outside 对 source；记录 LPIPS/PSNR/SSIM、full LPIPS 和同步计时 latency。
原始1024 PNG保留，案例图只是预览。ROI hard-lock 验证的是 token，不承诺 VQ 解码后的区域外像素完全相同。

默认 selection 为 `PENDING`，不默认选最后一份 checkpoint。
正式选择前，在计划配置中明确 preservation degradation 和 inside improvement 阈值，将状态改为 `PREREGISTERED`，并在任何本轮 DEV 生成前执行：

```bash
uv run python scripts/eval/evaluate_dense_checkpoints.py --plan configs/eval/dense_dev_plan.yaml --preregister-selection
```

之后再执行完整评测。选择规则为 MagicBrush DEV 上满足 preservation 限制的最低 inside LPIPS，默认要求配对 CI 支持改善，同分选较早 step。
没有合格候选仍为 PENDING；aux 不参与主选择；TEST 不得用于反向选择。
`--count` 或 `--checkpoint-steps` 仅用于小规模 diagnostic，不能生成正式 selected record。

### Dense Teacher 导出

导出只复制完整 safetensors 推理权重、region、architecture、fingerprint、身份和 Teacher manifest，不复制 optimizer。

```bash
# 早期 DEV-only 接口/pilot：不代表质量合格的正式 teacher
uv run python scripts/dimo/export_dense_teacher.py \
  --checkpoint "$DENSE_OUTPUT_ROOT/train/checkpoint-3125" \
  --status provisional --output-dir "$DENSE_OUTPUT_ROOT/teacher-provisional"

# 只有 DEV preregistered selection 返回 READY 才能导出 selected
export SELECTED_DENSE_CHECKPOINT=/path/to/actually/selected/checkpoint
uv run python scripts/dimo/export_dense_teacher.py \
  --checkpoint "$SELECTED_DENSE_CHECKPOINT" --status selected \
  --selected-record "$DENSE_OUTPUT_ROOT/dev_evaluation/SELECTED_DENSE_CHECKPOINT.json" \
  --output-dir "$DENSE_OUTPUT_ROOT/teacher-selected"

uv run python scripts/dimo/export_dense_teacher.py --verify "$DENSE_OUTPUT_ROOT/teacher-provisional"
```

Provisional 固定 `formal_teacher=false, dev_only=true`，不能绕过 selection gate。
Selected 由真实 READY record 登记，并不自动授权或启动 Region-DiMO 长训。
权重和身份全部 SHA256 绑定，不能将完整权重伪装成 `adapter_model.safetensors`。

### Region-DiMO 接口与限制

Dense 后端为一个 frozen FP32 Dense backbone + student/auxiliary 零增量 LoRA + 三份独立 region embedding。
Teacher 使用 backbone 原函数；初始 student/auxiliary 的 logits 必须与 teacher 一致，GPU 预声明容差 atol=rtol=1e-5，CPU toy 要求 exact。
原 distribution gradient、surrogate、pseudo masking、aux update、EMA 和 ROI hard-lock 数学不变。LoRA teacher 的旧入口仍可用。

```bash
export DIMO_TEACHER_CHECKPOINT="$DENSE_OUTPUT_ROOT/teacher-provisional"
export DENSE_DIMO_OUTPUT_ROOT=/path/to/new/dense-dimo-smoke
export CUDA_VISIBLE_DEVICES=0          # 单卡兼容性检查，最多2步
uv run python scripts/train/train_dimo_editing.py \
  --config configs/dimo/dense_teacher_smoke.yaml --prep-smoke --teacher-backend dense

# 独立八卡 prep 入口：显式 all-reduce 两个角色的梯度，最多2步，不是长训入口
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
uv run torchrun --nproc_per_node=8 --master_port=29672 scripts/train/train_dense_teacher_dimo.py \
  --config configs/dimo/dense_teacher_smoke.yaml --stop-at-global-step 2
# 可用 --stop-at-global-step 1 后，再以相同配置 --resume checkpoint-1 恢复到2
```

两种 prep 入口 checkpoint 格式不同，不能相互 resume。八卡 prep 的跨 rank 同步和恢复已通过本地真实测试，但正式八卡 DiMO 长训仍保持关闭，尚需正式 teacher selection 与独立长训方案；只有 YAML 或 CPU toy 不代表 GPU PASS。

```bash
export CUDA_VISIBLE_DEVICES=0
uv run python scripts/eval/sample_dimo_editing.py \
  --model-root "$EDITMGT_MODEL_ROOT" --teacher-backend dense \
  --teacher-checkpoint "$DIMO_TEACHER_CHECKPOINT" \
  --student-checkpoint "$DENSE_DIMO_OUTPUT_ROOT/checkpoint-2" \
  --source-image "$SOURCE_IMAGE" --edit-region-mask "$REGION_MASK" \
  --instruction "$EDIT_INSTRUCTION" --weights ema --output "$DENSE_DIMO_OUTPUT_ROOT/one-step.png"
```

八卡 prep checkpoint 的 one-step 推理只读取 identity 和 student/EMA safetensors，不读取 optimizer。

### 失败与恢复

- OOM：停止所有本实验 rank，保留日志/显存记录；不能自动改精度、batch、resolution 或并行算法。核实独占资源后再决定新任务。
- Bad data：停止，定位 UID/hash/geometry；不修改冻结 manifest，也不跳过错误样本。
- Rank failure：整个任务退出，不用 surviving ranks 继续提交更新；从最近 COMPLETED checkpoint 恢复。
- Checkpoint incomplete：没有有效 COMPLETED 或 hash 不符时禁止恢复；`.partial-*` 是未发布产物，不作为 checkpoint。
- Resume mismatch：核对代码、配置、model/data identity、world size、sampler 和 scheduler horizon；禁止把 resume 自动降级为 warm-start。
- Teacher identity mismatch：停止，重新核验 bundle 与 selected/provisional record；不能手动编辑 hash 来绕过门禁。
