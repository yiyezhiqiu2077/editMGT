# EditMGT Explicit-Region SFT 正式实验

本文是从空服务器到最终评测的执行入口。数据选择、去重、翻译、几何和 READY 语义见 [DATA_CONTRACT.md](DATA_CONTRACT.md)。所有命令均在仓库根目录执行。

## 环境安装

```bash
git clone git@github.com:yiyezhiqiu2077/editMGT.git
cd editMGT
git checkout <formal-sha>
uv sync --frozen --group dev --group translation --group metrics
```

正式 SHA 必须包含已冻结的 checkpoint-selection 规则。目前 `configs/eval/formal.yaml` 仍为 `MUST_BE_PREREGISTERED_BEFORE_FORMAL`，因此 LR probe 和 E1–E4 会按设计拒绝运行。

## 空服务器目录

只需指定大磁盘根目录；`HF_HOME` 和镜像端点可选：

```bash
export ASSET_ROOT=/large_disk/editmgt_assets
export HF_HOME=/large_disk/editmgt_hf_cache
# export HF_ENDPOINT=https://your-hf-mirror.example
```

脚本生成以下布局：

```text
$ASSET_ROOT/
├── models/                 # EditMGT、NLLB、DINO、CLIP、LPIPS/AlexNet
├── datasets/               # MagicBrush train/dev/test、Crisp、Scale、Inter-Edit
├── derived/fixed200k/      # canonical pools、validation、D200K、QA
├── experiments/            # smoke、训练、评测输出
└── artifacts/
    ├── formal_env.sh
    ├── formal_assets.json
    ├── FORMAL_READY.json
    └── stages/*/_SUCCESS.json
```

存储规模以 Inter-Edit 为主，准备至少数 TB 可用空间。镜像只改变传输路径，不改变 pinned commit。

## 资产与 D200K 准备

先检查计划或只做在线元数据检查：

```bash
bash scripts/setup/run_formal_prepare.sh --print-command
bash scripts/setup/run_formal_prepare.sh --dry-run
```

正式准备：

```bash
bash scripts/setup/run_formal_prepare.sh --run
```

该命令自动完成依赖安装、pinned 资产下载、schema contract 比较、canonicalization、NLLB 翻译、QA、去重与确定性 backfill、exact 200K、FP32 VQ 审计和 `CORPUS_READY.json`。

00–12 每阶段都绑定 Git、配置、上游 marker 和 resolved revisions。身份未变时重跑会跳过；发生变化时，该阶段及下游 marker 自动失效。

## 8-GPU smoke

先打印，再执行：

```bash
bash scripts/cluster/run_8g_smoke.sh --print-command
export CONFIRM_FORMAL_RUN=YES
bash scripts/cluster/run_8g_smoke.sh --run
```

smoke 对比 uninterrupted 1→20 与 fresh 1→10 + resume 11→20，必须满足 640 条 distinct rows、sample/geometry/corruption/LR 序列一致且无 NaN/Inf。通过后生成 `$ASSET_ROOT/artifacts/FORMAL_READY.json`。

## LR probes

```bash
bash scripts/cluster/run_lr_probes.sh --print-command
bash scripts/cluster/run_lr_probes.sh --run
```

launcher 会同时验证 `formal_assets.json`、`CORPUS_READY.json`、`FORMAL_READY.json` 和 selection rule。selection 状态不是 `READY` 或阈值为空时会报 `SELECTION_RULE_NOT_PREREGISTERED`；不要在服务器上临时补阈值。

## E0–E4

| 实验 | 训练语义 | LoRA scope |
|---|---|---|
| E0-official | released model，official timestep | released weights |
| E0-region | released model，ROI-relative timestep | released weights |
| E1 | full target，region condition OFF | both |
| E2 | ROI hard lock，region condition OFF | both |
| E3 | ROI hard lock，region condition ON | both |
| E4 | ROI hard lock，region condition ON | reference only |

E1–E4 共享 exact D200K、permutation、seed、预算和 validation protocol。

```bash
bash scripts/cluster/run_e0_eval.sh --print-command
bash scripts/cluster/run_e1_e4.sh --print-command
bash scripts/cluster/run_e1_e4.sh --run
```

## Validation

MagicBrush DEV 是 primary validation。CrispEdit、ScaleEdit 和 Inter-Edit aux-128 分开报告，只作诊断。TEST 不参与 D200K、阈值、LR 或 checkpoint 选择。

```bash
export CANDIDATE_CHECKPOINT=$ASSET_ROOT/experiments/<run>/checkpoint-6250
bash scripts/cluster/run_fixed200k_full_validation.sh --print-command
```

## MagicBrush TEST

在 formal SHA 中冻结 selection config 后，先登记唯一 checkpoint：

```bash
uv run python scripts/eval/register_selected_checkpoint.py \
  --checkpoint "$CANDIDATE_CHECKPOINT" \
  --formal-assets "$ASSET_ROOT/artifacts/formal_assets.json" \
  --output "$ASSET_ROOT/artifacts/SELECTED_CHECKPOINT.json"
```

TEST launcher 硬验证 pinned TEST identity、535 sessions、1053 turns、checkpoint、Git 和 formal-assets hash：

```bash
bash scripts/eval/run_magicbrush_test.sh --print-command
bash scripts/eval/run_magicbrush_test.sh --run
```

禁止用 DEV 路径替代 TEST。

## 结果打包

```bash
uv run python scripts/tools/package_formal_results.py \
  --experiment-root "$ASSET_ROOT/experiments" \
  --eval-root "$ASSET_ROOT/experiments/magicbrush-test" \
  --formal-assets "$ASSET_ROOT/artifacts/formal_assets.json" \
  --output "$ASSET_ROOT/editmgt-formal-results.tar.gz"
```

包内包含 configs、provenance、metrics、summary、READY marker 和 audit report；不含 raw data、模型/checkpoint 权重、cache、token payload 或全分辨率生成图。

## 测试

```bash
uv run pytest -q
uv run python -m compileall src scripts tests
bash -n scripts/setup/run_formal_prepare.sh scripts/cluster/*.sh scripts/train/*.sh scripts/eval/*.sh
git diff --check
```

代码完成不等于实验完成。若没有真实 `CORPUS_READY.json` 和 `FORMAL_READY.json`，正式训练仍是 NOT RUN。

## Region-DiMO

Region-DiMO 是 E1–E4 之后的候选第二阶段，目前仅为 CODE PREP，不是正式实验。完整 handoff 顺序为：

```text
E1–E4
→ 使用预注册规则在 MagicBrush DEV 上选择唯一 candidate
→ scripts/eval/register_selected_checkpoint.py
→ SELECTED_CHECKPOINT.json
→ scripts/dimo/register_teacher.py
→ dimo_teacher_manifest.json
→ DIMO_TEACHER_CHECKPOINT
→ future Region-DiMO
```

teacher 注册示例：

```bash
uv run python scripts/dimo/register_teacher.py \
  --selected-checkpoint "$CANDIDATE_CHECKPOINT" \
  --selected-checkpoint-record "$ASSET_ROOT/artifacts/SELECTED_CHECKPOINT.json" \
  --formal-assets "$ASSET_ROOT/artifacts/formal_assets.json" \
  --output-dir "$ASSET_ROOT/models/dimo_teacher"

export DIMO_TEACHER_CHECKPOINT="$ASSET_ROOT/models/dimo_teacher"
```

注册器会重新验证 checkpoint 文件、selection record、formal assets、Git 与 released EditMGT identity，并原子生成 teacher bundle。teacher registration 不等于 formal DiMO ready；当前必须保持 `SELECTED_DIMO_TEACHER=NONE`、`DIMO_EDIT_FORMAL_READY=false`，不得启动正式 DiMO long run。算法、role 管理、one-step inference 与后续门禁详见 [DIMO_PORTING.md](DIMO_PORTING.md)。
