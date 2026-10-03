# EditMGT 显式区域 SFT 实验

本文档给出在一台空服务器上复现 D200K-v2 语料和 explicit-region 实验的完整流程。内容以当前代码为准，同时明确标记尚未实际执行的门禁和实验。除非特别说明，所有命令都应在仓库根目录运行。

## 1. 实验范围

本实验系列研究以下问题：

- Q1：在固定评估协议下，continued SFT 是否能提升 released EditMGT？
- Q2：ROI hard-lock corruption 是否能改善 provided-region editing 和区域外内容保持？
- Q3：除 ROI corruption 外，持续生效的 edit-region embedding 是否有额外作用？
- Q4：target-side LoRA 是否必要，还是只训练 reference-side LoRA 已经足够？
- Q5：后续 cache、sparse 或 shortcut acceleration 应选用哪个 dense checkpoint 作为基线？

本仓库只定义实验，并未给出这些问题的答案。E0–E4 和所有正式 long run 当前均未运行。

## 2. 可复现性契约

一次正式运行必须绑定以下全部身份：

- 一个经过审核的 Git SHA 和 dirty-diff digest；
- `uv.lock` 和解析后的 YAML 配置；
- released EditMGT component/snapshot identity；
- 不可变的原始数据集 revision 和 NLLB revision；
- canonical `sample_uid` 以及 source/target/region 内容 SHA256；
- 精确的 `train_200k.jsonl` SHA256 和所有 READY attestation 文件；
- `fixed200k-hash-sort-v1` epoch permutation 和无放回 rank stride；
- optimizer boundary 上已提交的 resume sample cursor；
- world size、每卡 batch、梯度累积、seed、precision、LoRA scope 和 scheduler state。

`torch_compile=false` 和 deterministic algorithms 是 v1 正确性契约的一部分。冻结语料中的坏样本必须使运行失败，禁止在训练期动态换样本。

## 3. 空服务器环境安装

```bash
git clone git@github.com:yiyezhiqiu2077/editMGT.git
cd editMGT
git checkout FORMAL_EXPERIMENT_CODE_SHA

export EDITMGT_WORKTREE="$PWD"
uv sync --frozen --group dev --group translation --group metrics

uv run python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
PY
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader
```

如果服务器可以访问 Astral 且尚未安装 `uv`，可运行 `bash scripts/setup/bootstrap_uv.sh` 安装 `uv` 并执行基础 frozen sync。在隔离集群上，应提前准备 `uv` 和兼容的 wheel cache；不要通过删除 `--frozen` 绕过 lockfile。

记录开始时的代码身份：

```bash
git rev-parse HEAD
git status --short
sha256sum uv.lock configs/data/fixed_200k.yaml configs/eval/formal.yaml
```

## 4. 目录与资产配置

代码、只读原始资产、衍生语料、缓存和实验输出应彼此分离。推荐使用以下通用布局：

```text
WORK_ROOT/
├── editMGT/                         # Git checkout
├── models/
│   ├── EditMGT/IMMUTABLE_SNAPSHOT/
│   └── NLLB/IMMUTABLE_SNAPSHOT/
├── datasets/
│   ├── MagicBrush/{train,dev,test}/
│   ├── CrispEdit-labeling-39k/
│   ├── ScaleEdit-labeling-25k/
│   └── Inter-Edit-Train/
├── derived/editmgt-explicit-region/
├── outputs/editmgt-explicit-region/
└── cache/huggingface/
```

导出路径和不可变身份标识。下面是通用示例，必须替换成经过审计的本地路径和真实 immutable revision：

```bash
export EDITMGT_WORKTREE="$PWD"
export EDITMGT_MODEL_ROOT=/models/EditMGT/IMMUTABLE_SNAPSHOT
export TRANSLATOR_MODEL_ROOT=/models/NLLB/IMMUTABLE_SNAPSHOT
export MAGICBRUSH_ROOT=/datasets/MagicBrush/train
export MAGICBRUSH_DEV_ROOT=/datasets/MagicBrush/dev
export CRISPEDIT_ROOT=/datasets/CrispEdit-labeling-39k
export SCALEEDIT_ROOT=/datasets/ScaleEdit-labeling-25k
export INTEREDIT_ROOT=/datasets/Inter-Edit-Train
export DERIVED_ROOT=/derived/editmgt-explicit-region
export EDITMGT_OUTPUT_ROOT=/outputs/editmgt-explicit-region
export HF_HOME=/cache/huggingface

export MAGICBRUSH_REVISION=IMMUTABLE_DATASET_ID
export CRISPEDIT_REVISION=IMMUTABLE_DATASET_ID
export SCALEEDIT_REVISION=IMMUTABLE_DATASET_ID
export INTEREDIT_REVISION=IMMUTABLE_DATASET_ID
export TRANSLATOR_REVISION=IMMUTABLE_MODEL_ID
```

原始数据集和 released snapshot 必须只读。只有 `DERIVED_ROOT`、`EDITMGT_OUTPUT_ROOT` 和 `HF_HOME` 可以写入；这些目录中的任何内容都不应进入 Git。

资产审计使用的可选本地软链接可通过以下命令建立：

```bash
bash scripts/setup_local_assets.sh model "$EDITMGT_MODEL_ROOT"
bash scripts/setup_local_assets.sh magicbrush "$MAGICBRUSH_ROOT"
bash scripts/setup_local_assets.sh magicbrush-test /datasets/MagicBrush/test
bash scripts/setup_local_assets.sh interedit "$INTEREDIT_ROOT"
bash scripts/setup_local_assets.sh outputs "$EDITMGT_OUTPUT_ROOT"
uv run python scripts/setup/audit_assets.py
```

完整环境契约见 [ENVIRONMENT.md](ENVIRONMENT.md)。

## 5. Released model 资产

`EDITMGT_MODEL_ROOT` 必须指向一个完整的 released snapshot，并包含：

```text
editmgt/
text_encoder/
tokenizer/
llm_encoder/
vqvae/
scheduler/
```

训练入口会验证这些目录名，从本地文件加载组件，并记录 component identity。禁止静默替换 Gemma/text encoder、tokenizer、scheduler 或 VQ-VAE。VQ-VAE 固定使用 FP32，因为本地 BF16 audit 未达到足够的 token agreement。

## 6. 原始数据集

正式数据来源包括：

- MagicBrush official TRAIN；
- MagicBrush official DEV，作为 primary validation；
- MagicBrush official TEST，只能用于最终评估；
- CrispEdit-labeling-39k；
- ScaleEdit-labeling-25k；
- Inter-Edit-Train，仅使用经过质量筛选的 `better_data` 记录。

不得用名称相似的本地 cache 或 test set 替代目标数据。canonicalization 前必须先审计真实 schema，因为代码不会自动猜测 CrispEdit/ScaleEdit 的字段名、存储格式、edit taxonomy 或 mask semantics。

首先运行只读 schema audit：

```bash
mkdir -p "$DERIVED_ROOT/fixed200k/schema_audit" "$DERIVED_ROOT/schema_mappings"

uv run python scripts/data/audit_raw_dataset_schema.py \
  --dataset-name magicbrush --root "$MAGICBRUSH_ROOT" \
  --revision "$MAGICBRUSH_REVISION" \
  --output "$DERIVED_ROOT/fixed200k/schema_audit/magicbrush.json"

uv run python scripts/data/audit_raw_dataset_schema.py \
  --dataset-name crispedit --root "$CRISPEDIT_ROOT" \
  --revision "$CRISPEDIT_REVISION" \
  --output "$DERIVED_ROOT/fixed200k/schema_audit/crispedit.json"

uv run python scripts/data/audit_raw_dataset_schema.py \
  --dataset-name scaleedit --root "$SCALEEDIT_ROOT" \
  --revision "$SCALEEDIT_REVISION" \
  --output "$DERIVED_ROOT/fixed200k/schema_audit/scaleedit.json"

uv run python scripts/data/audit_raw_dataset_schema.py \
  --dataset-name interedit --root "$INTEREDIT_ROOT" \
  --revision "$INTEREDIT_REVISION" \
  --output "$DERIVED_ROOT/fixed200k/schema_audit/interedit.json"
```

人工检查审计结果后，复制并填写显式 mapping：

```bash
export CRISPEDIT_SCHEMA_MAPPING="$DERIVED_ROOT/schema_mappings/crispedit.yaml"
export SCALEEDIT_SCHEMA_MAPPING="$DERIVED_ROOT/schema_mappings/scaleedit.yaml"
cp configs/data/crispedit_schema_mapping.template.yaml "$CRISPEDIT_SCHEMA_MAPPING"
cp configs/data/scaleedit_schema_mapping.template.yaml "$SCALEEDIT_SCHEMA_MAPPING"
```

mapping 必须明确填写 schema status、revision、format/glob、source/target/region 的 storage 和字段、instruction、edit type、可选 group ID 以及 mask semantics。随后在 `configs/data/edit_type_mapping.yaml` 中补全 CrispEdit/ScaleEdit taxonomy。残留的 `REPLACE`、`NOT_RUN` 或空数据集映射会触发预期的 hard failure。

## 7. D200K-v2 语料构建

`configs/data/fixed_200k.yaml` 定义的固定策略为：

- 使用全部合格 MagicBrush TRAIN 记录；
- 最多选择 39,000 条合格 CrispEdit；
- 最多选择 25,000 条合格 ScaleEdit；
- 使用 `better_data` Inter-Edit 补齐剩余额度；
- 最终必须得到恰好 200,000 个唯一 `sample_uid`。

真实语料构建完成前，不能报告各数据集的实际贡献数量。完整构建顺序为：

```text
真实 schema audit
  -> 显式 canonical adapters
  -> MagicBrush official DEV + probe128 + 三套 aux128 冻结
  -> eligibility filtering
  -> 确定性 candidate 和 reserve order
  -> 同数据集 exact-sample collapse
  -> 跨数据集 exact-source resolution（validation 优先）
  -> 翻译已选中的 Han instruction
  -> translation QA 和确定性 reserve backfill
  -> exact-200K integrity freeze
  -> 内容、几何、VQ 和 montage 审计
  -> CORPUS_READY.json
```

处理跨数据集 source duplicate 时的优先级为 MagicBrush、CrispEdit、ScaleEdit、Inter-Edit。选择过程确定、无放回。MagicBrush official DEV 在所有 train/validation source 或 group 冲突中优先。该流水线永远不读取 TEST。

先打印完整编排命令：

```bash
bash scripts/cluster/prepare_fixed_200k_v2.sh --print-command
```

确认路径、revision 和 mapping 后运行：

```bash
export CONFIRM_CORPUS_BUILD=YES
bash scripts/cluster/prepare_fixed_200k_v2.sh --run
unset CONFIRM_CORPUS_BUILD
```

主要产物位于 `$DERIVED_ROOT/fixed200k/`：

```text
train_200k.jsonl
train_200k.meta.json
candidate_selection.jsonl
reserve_order.jsonl
final_selection.jsonl
selection_report.json
duplicate_report.json
translation_report.json
translation_cache.jsonl
translation_manual_audit_500.{jsonl,csv}
validation/
audit/
CORPUS_READY.json
```

## 8. 翻译

Han detection 对所有已选数据集生效，而不仅限于 Inter-Edit。冻结的翻译契约是 `facebook/nllb-200-distilled-1.3B`、`zho_Hans -> eng_Latn`、immutable revision、greedy decoding 和 `max_new_tokens=128`。

cache key 绑定原始文本、backend、模型 revision、源/目标语言和 decoding 参数。流程先选择样本，再进行翻译。以下 QA flag 会拒绝翻译结果：

- `empty_output`；
- `copy_output`；
- `han_remaining`；
- `length_ratio_outlier`；
- `digit_mismatch`。

被拒绝的已选记录会由对应 stratum 的确定性 reserve 替换，循环最多执行 32 轮。最终构建还会冻结 500 条翻译记录供人工审计；正式训练前必须检查 CSV 和所有 flagged case。

## 9. 数据审计

`scripts/data/audit_fixed200k.py` 使用真实数据根目录验证全部冻结记录，并生成：

- 每个数据集的记录数和 unique-source 数；
- missing、decode、hash 和 Han failure；
- edit-type 和 mask-semantics 分布；
- region-fraction 和 samples-per-source 统计；
- 尺寸和宽高比分布；
- FP32-VQ 的区域内外 change containment；
- 每个数据集 100 条 montage，以及最终 stratified montage。

人工审阅必须检查 source/target/region 对齐、mask polarity、instruction 正确性、翻译质量、duplicate/diversity 报告和异常长尾。机器检查成功并不等于语料已经获批。

## 10. Corpus-ready 门禁

`CORPUS_READY.json` 是唯一由机器信任的 readiness marker。它通过 SHA256 绑定 train manifest、metadata、candidate/reserve/final selection、duplicate/translation report、translation cache、validation manifest、audit report、manual translation sample、config 和 montage。

独立验证命令：

```bash
uv run python scripts/data/verify_corpus_ready.py \
  "$DERIVED_ROOT/fixed200k/CORPUS_READY.json"
```

D200K smoke、正式训练和 full validation 在启动时会重新检查这些 hash。文件缺失、hash 改变、status 不是 READY 或总数不是 200,000 都必须阻止执行。还应另行保存人工 go/no-go 记录，包括审阅人、日期和 marker SHA；READY 不证明数据许可证或视觉质量已经通过人工审核。

## 11. 8-GPU smoke 与 resume 门禁

Smoke 使用 E3 路径、8 张 GPU、每卡 batch 1、梯度累积 4，因此 global batch 为 32。20 个 optimizer steps 应提交恰好 640 条不同记录。

```bash
bash scripts/cluster/run_8g_smoke.sh --print-command

export CONFIRM_FORMAL_RUN=YES
bash scripts/cluster/run_8g_smoke.sh --run
unset CONFIRM_FORMAL_RUN
```

launcher 按顺序执行：

1. 验证 READY hash；
2. fresh uninterrupted step 1–20；
3. fresh step 1–10 并保存 checkpoint-10；
4. 从 checkpoint-10 resume 到 step 20；
5. 对比 sample sequence 和最终状态。

`$EDITMGT_OUTPUT_ROOT/fixed200k-smoke-verification.json` 必须报告 640 条 unique rows，并确认 sample UID、geometry seed、corruption seed 和 learning-rate sequence 完全一致。OOM、NCCL failure、NaN/Inf、状态缺失或 verifier failure 都会阻止所有 long run。

当前尚未对 D200K-v2 实际运行这套 8-GPU smoke。

## 12. 训练语义

令 `M` 表示 token-level edit region，`S` 表示从中确定性采样的 masked subset。

### Full target

corruption canvas 是完整 target token grid。`S` 从全图采样，在输入中替换为 MASK，并以 target token 为监督目标。E1 不传入 edit-region condition。

### ROI hard lock

`S` 是 `M` 的非空子集，输入由以下部分组成：

```text
MASK token       位于 S
target token     位于 M \ S
source token     位于 M 之外
```

label 仅在 `S` 上使用 target token，其它位置使用 ignore index。因此优化 loss 只计算选中的 edit token，而区域外 canvas 始终由 source hard lock。mask ratio 为 `rho = cos(u * pi / 2)`；另有 0.15 概率 mask 全部 ROI token。E2 关闭 persistent region conditioning，E3/E4 则在 Transformer 中持续向 `M` 注入学习得到的 region embedding。

loss 采用 sample-normalized reduction：

```text
L_i = sum_{p in S_i} CE(logits_i,p, target_i,p) / |S_i|
L   = mean_i L_i
```

这样可避免大 mask 自动获得更大的 sample weight。`token_mean` 仅作为诊断日志，不是实际优化 reduction。

推理支持 `official_upstream_timestep` 和 `roi_relative`。后者向模型输入剩余 ROI mask ratio，而不是上游的全局 discrete timestep。训练使用概率 0.1 的 text-only condition dropout，同时保留 source/region conditioning。LoRA 可作用于 target 和 reference 两个分支，也可只作用于 reference 分支。

## 13. 正式训练预算

下表数值来自 `configs/train/_cluster_8g_base.yaml` 和 `configs/eval/formal.yaml`：

| 项目 | 数值 |
| --- | --- |
| 固定训练记录 | 200,000 |
| Epochs | 1 |
| GPUs | 8 |
| Batch / GPU | 1 |
| Gradient accumulation | 4 |
| Global batch | 32 |
| Optimizer steps / epoch | 6,250 |
| Resolution | 1024 |
| Train seed | 42 |
| Precision | BF16 Transformer/text，FP32 VQ |
| Optimizer | AdamW，betas 0.9/0.95，weight decay 0.01 |
| Base learning rate | 3e-5 |
| Scheduler | constant with 200 warmup steps |
| Gradient clip | 1.0 |
| LoRA | rank 64，alpha 64，dropout 0.05 |
| Checkpoint | 每 625 steps，包含 step 6,250 |
| Periodic validation | 每 625 steps |
| Generation seeds | 0、1、2、3 |
| Bootstrap | seed 42，10,000 resamples，sample unit |

YAML 是最终依据。如果经过审核的 YAML 发生变化，必须在运行前同步更新本表。

## 14. Learning-rate probes

三个 E3 probe 分别使用 `1e-5`、`3e-5` 和 `5e-5`。每组运行 1,500 个 optimizer steps，并共享 base seed、train manifest、epoch permutation、geometry/corruption sequence、batch、accumulation 和 scheduler horizon。因此每组都消费相同的前 `1,500 × 32 = 48,000` 条记录。

```bash
bash scripts/cluster/run_lr_probes.sh --print-command

export CONFIRM_FORMAL_RUN=YES
bash scripts/cluster/run_lr_probes.sh --run
unset CONFIRM_FORMAL_RUN
```

probe 配置关闭 periodic generation，并在 step 500、1,000、1,500 保存 checkpoint。执行前必须基于 finite/quality-gate behavior 和冻结的 validation protocol 预注册选择规则。禁止使用 final TEST 选择 LR。如果最终 LR 不是 base config 中的 `3e-5`，应创建并审核一个新的命名配置，而不是静默修改已有配置。

## 15. E0 与 E1–E4

| 实验 | 训练 | Corruption | Persistent region condition | LoRA scope | 评估 timestep |
| --- | --- | --- | --- | --- | --- |
| E0-official | 无 | 无 | 无 | released | official upstream |
| E0-region | 无 | 无 | provided region | released | ROI-relative |
| E1 | fixed 200K | full target | OFF | both | ROI-relative |
| E2 | fixed 200K | ROI hard lock | OFF | both | ROI-relative |
| E3 | fixed 200K | ROI hard lock | ON | both | ROI-relative |
| E4 | fixed 200K | ROI hard lock | ON | reference only | ROI-relative |

E1–E4 使用完全相同的 200K manifest、sample order、训练预算、seed、optimizer family、checkpoint schedule 和 primary validation。只有 corpus 与 8-GPU smoke 均获批后才能启动：

```bash
bash scripts/cluster/run_e1_e4.sh --print-command

export CONFIRM_FORMAL_RUN=YES
bash scripts/cluster/run_e1_e4.sh --run
unset CONFIRM_FORMAL_RUN
```

当前 E0 launcher 使用旧版 frozen MagicBrush/Inter-Edit split，而不是 D200K auxiliary validation manifest。必须单独准备并清晰标注：

```bash
bash scripts/cluster/prepare_interedit.sh --print-command

export CONFIRM_FORMAL_RUN=YES
bash scripts/cluster/prepare_interedit.sh --run
bash scripts/cluster/run_e0_eval.sh --run
unset CONFIRM_FORMAL_RUN
```

E0 报告必须记录这些 manifest 的路径和 hash，避免与 fixed200k full validation 混淆。

## 16. 可选 Stage A/B

Stage A/B 不属于主要 E0–E4 矩阵。只有主要实验结果支持继续研究 curriculum 时才应执行。

- Stage A：mixed corruption，ROI probability 0.50，full-ROI probability 0.15，persistent condition ON。
- Stage B：从 Stage A warm-start，mixed corruption，ROI probability 0.85，full-ROI probability 0.15，persistent condition ON，warmup 100。

```bash
bash scripts/cluster/run_stage_a.sh --print-command
export STAGE_A_CHECKPOINT=/absolute/path/to/stage-a/checkpoint
bash scripts/cluster/run_stage_b.sh --print-command
```

真正执行仍要求 `CONFIRM_FORMAL_RUN=YES`。Stage B 使用 `--warm-start`，不是 `--resume`。这两个阶段目前都未运行。

## 17. Validation protocol

### Periodic validation

- 使用 MagicBrush official DEV 中确定性冻结的 128 条 probe；
- generation seed 为 0；
- 每 625 optimizer steps 运行一次；
- 仅用于诊断，不能替代完整 DEV；
- validation 会保存并恢复 CPU/CUDA/Python/NumPy RNG，不推进训练状态。

### Full validation

- 完整 MagicBrush official DEV，seeds `[0, 1, 2, 3]`：用于 primary checkpoint evaluation；
- CrispEdit aux-128、ScaleEdit aux-128、Inter-Edit aux-128：分别报告，仅用于诊断；
- auxiliary 结果不参与 primary checkpoint ranking。

```bash
export CANDIDATE_CHECKPOINT=/absolute/path/to/checkpoint
bash scripts/cluster/run_fixed200k_full_validation.sh --print-command

export CONFIRM_FORMAL_RUN=YES
bash scripts/cluster/run_fixed200k_full_validation.sh --run
unset CONFIRM_FORMAL_RUN
```

输出位于 `$EDITMGT_OUTPUT_ROOT/full-validation/CHECKPOINT_NAME/DATASET/{predictions.jsonl,metrics.json}`。

## 18. 指标

`scripts/eval/formal_eval.py` 计算：

- 区域内 L1、PSNR、SSIM 和 masked LPIPS-to-target；
- 区域外 L1、PSNR、SSIM 和 masked LPIPS-to-source；
- full-image LPIPS-to-target；
- edit region 内的 `d_ST`、`d_SO`、`d_OT` 和 no-op progress diagnostics；
- 使用显式本地模型时的 DINO-I-to-target/source 和 CLIP-I-to-target/source；
- 每次 generation 的运行时间；
- 多 generation seed 的 per-sample mean；
- 按数据集和 edit type 汇总的 mean、standard error 和 95% sample-bootstrap interval。

Masked LPIPS 使用 feature-space masked aggregation：将 spatial region 通过 area resize 映射到各层 LPIPS feature map，再聚合 learned feature distance。它不是对黑掉区域后的图像直接计算 LPIPS。

只有显式配置本地 embedding model 时才会产生 DINO/CLIP 数值；evaluator 不会静默下载这些模型。

## 19. Checkpoint selection

计划预注册的选择顺序是：

1. 在 primary MagicBrush DEV 上应用区域外 preservation gate；
2. 在通过门禁的候选中最小化区域内 masked LPIPS-to-target；
3. 使用预注册的 no-op diagnostics 作为 secondary rule，绝不使用 final TEST。

但是，当前仓库**尚未包含可执行的 checkpoint selection rule**。`configs/eval/formal.yaml` 中的 `threshold_preserve`、`tau_edit` 和 `tau_noop` 均为 null，status 是 `MUST_BE_PREREGISTERED_BEFORE_FORMAL`。evaluator 只计算指标；只有同时提供两个 tau threshold 时才产生 no-op 值，并不会对 checkpoint 排名，也不会应用 `threshold_preserve`。

在运行 LR probes 或正式 checkpoint selection 前，必须在代码和配置中冻结具体的 preservation metric/direction、threshold、tie-break、tau 定义和 failure policy，为其增加测试，并提交新的 experiment SHA。禁止在查看 candidate 或 TEST 结果后再填写阈值。

## 20. 最终 TEST

只有完全不使用 TEST 选出的唯一 method/checkpoint 才能在 MagicBrush official TEST 上评估。本地审计记录显示 official test 包含 1,053 turns，但它不参与 corpus construction、LR selection、periodic validation、full DEV ranking 或 threshold tuning。

当前仓库还没有 D200K 专用 final-TEST launcher。只有在 selection contract 已提交且最终 candidate 已锁定后，才能新增并审核冻结的 TEST manifest/launcher；不要临时改用 DEV launcher。最终报告必须记录 TEST manifest hash、代码/config/checkpoint identity、generation seeds 和全部 evaluator asset identity。

## 21. 当前执行状态

| 阶段 | 状态 |
| --- | --- |
| 本地代码/测试和真实 MagicBrush 链路 | 通过 |
| 真实 CrispEdit-labeling-39k schema audit | 未运行（NOT RUN） |
| 真实 ScaleEdit-labeling-25k schema audit | 未运行（NOT RUN） |
| 真实 Inter-Edit-Train schema audit | 未运行（NOT RUN） |
| 真实固定 200K 构建和人工审核 | 尚未构建 / 未运行 |
| `CORPUS_READY.json` | 尚未创建 |
| 8-GPU uninterrupted/resume smoke | 未运行（NOT RUN） |
| LR probes | 未运行（NOT RUN） |
| E0–E4 | 未运行（NOT RUN） |
| Stage A/B | 未运行（NOT RUN） |
| Final TEST | 未运行（NOT RUN） |

当前本地证据包括 41 个单元测试通过，以及在一张 A100-PCIE-40GB 上完成真实 1024 一步反传。它不能证明任何缺失的真实数据集或集群阶段已经通过。

## 22. 完整命令顺序

下面是完整流程的紧凑索引；前文规定的人工审核和门禁仍然全部生效。

```bash
# 克隆和环境安装
git clone git@github.com:yiyezhiqiu2077/editMGT.git
cd editMGT
git checkout FORMAL_EXPERIMENT_CODE_SHA
export EDITMGT_WORKTREE="$PWD"
uv sync --frozen --group dev --group translation --group metrics

# 按第 4、6 节导出全部 model/data/revision/derived/output/mapping 变量。
bash scripts/setup/probe_runtime.sh
bash scripts/cluster/probe_cluster.sh
uv run pytest -q

# 人工审计真实 schema，冻结 CrispEdit/ScaleEdit schema mapping 和 edit-type mapping。

# 构建并审计 exact D200K
bash scripts/cluster/prepare_fixed_200k_v2.sh --print-command
export CONFIRM_CORPUS_BUILD=YES
bash scripts/cluster/prepare_fixed_200k_v2.sh --run
unset CONFIRM_CORPUS_BUILD
uv run python scripts/data/verify_corpus_ready.py \
  "$DERIVED_ROOT/fixed200k/CORPUS_READY.json"

# 8-GPU smoke gate
bash scripts/cluster/run_8g_smoke.sh --print-command
export CONFIRM_FORMAL_RUN=YES
bash scripts/cluster/run_8g_smoke.sh --run
unset CONFIRM_FORMAL_RUN

# 提交预注册的选择规则后，运行 LR probes
bash scripts/cluster/run_lr_probes.sh --print-command
export CONFIRM_FORMAL_RUN=YES
bash scripts/cluster/run_lr_probes.sh --run
unset CONFIRM_FORMAL_RUN

# Released baselines
bash scripts/cluster/prepare_interedit.sh --print-command
bash scripts/cluster/run_e0_eval.sh --print-command

# Formal ablations
bash scripts/cluster/run_e1_e4.sh --print-command
export CONFIRM_FORMAL_RUN=YES
bash scripts/cluster/run_e1_e4.sh --run
unset CONFIRM_FORMAL_RUN

# 对预注册 candidate 运行完整 DEV + auxiliary validation
export CANDIDATE_CHECKPOINT=/absolute/path/to/checkpoint
bash scripts/cluster/run_fixed200k_full_validation.sh --print-command
export CONFIRM_FORMAL_RUN=YES
bash scripts/cluster/run_fixed200k_full_validation.sh --run
unset CONFIRM_FORMAL_RUN
```

只有在使用已提交的 DEV-only 规则选出唯一 candidate 后，才能进入 final TEST。

## 23. Future stage: Region-DiMO one-step distillation

本阶段当前仅完成代码准备，不是正式实验。只有
`FIXED_200K_FORMAL_READY=true`、E1–E4 全部完成、且按预注册 DEV-only
selector 冻结 selected dense teacher 后，才允许设置
`DIMO_TEACHER_CHECKPOINT` 并进入 formal Region-DiMO。当前
`SELECTED_DIMO_TEACHER=NONE`、`DIMO_EDIT_FORMAL_READY=false`。完整设计、
upstream code reference 和后续门禁见 [DIMO_PORTING.md](DIMO_PORTING.md)。
