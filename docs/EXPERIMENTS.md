# E3 Dense → Full DenseDiMO 实验运行手册

在仓库根目录执行；只使用 `exp/e3-dense-fp32-v1` 的固定干净 commit。
历史实验记录见 [LEGACY_EXPERIMENTS.md](LEGACY_EXPERIMENTS.md)，不作为本次 GPU 验收证据。

## 1. 环境和冻结数据

```bash
git clone --single-branch --branch exp/e3-dense-fp32-v1 \
  https://github.com/yiyezhiqiu2077/editMGT.git editMGT
cd editMGT
git checkout --detach "$DENSE_COMMIT"
bash scripts/setup/bootstrap_uv.sh
uv sync --frozen --group dev --group metrics
source "$FORMAL_ENV_FILE"  # 已有集群环境文件，只读取
export DENSE_EXPECTED_GIT_SHA="$DENSE_COMMIT"
export DENSE_OUTPUT_ROOT=/path/to/new/e3-dense-run
export DIMO_OUTPUT_ROOT=/path/to/new/full-dense-dimo-run
export FIXED200K_MANIFEST="$DERIVED_ROOT/fixed200k/train_200k.jsonl"
export CORPUS_READY="$DERIVED_ROOT/fixed200k/CORPUS_READY.json"
export DIMO_DEV_PLAN=configs/eval/full_dense_dimo_plan.yaml
```

环境文件须提供 `EDITMGT_MODEL_ROOT`、`DERIVED_ROOT`、`TRANSLATOR_REVISION`、
`MAGICBRUSH_ROOT`、`CRISPEDIT_ROOT`、`SCALEEDIT_ROOT`、`INTEREDIT_ROOT`、`MAGICBRUSH_DEV_ROOT`。
使用已有 Frozen D200K；不运行重建、翻译或共享 READY 更新脚本。

先申请独占 8 GPU，确认单卡容量、输出磁盘空间和模型/数据路径。
DiMO 运行时按参数量估算六份恢复 checkpoint、八份推理导出和原子保存临时空间。
约 10 亿参数时，每份完整恢复约 28 GB，每份 Raw+EMA 导出约 8 GB。
GPU 验收必须覆盖两份 AdamW 状态首次分配、EMA 和 checkpoint 保存。

## 2. E3 Dense SFT

固定配置：`configs/train/dense_d200k_5epoch.yaml`。
Released FP32 初始化；完整 Transformer+region；CLIP/Gemma/VQ 冻结。
8×1×accum4=32；5 epochs/31,250 updates；LR=2e-6。
AdamW=(0.9,0.95)、WD=0.01、clip=1；constant warmup=200；BF16 autocast。
ROI hard-lock、persistent conditioning、geometry、CE 和采样配方保持原实现。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
uv run python scripts/train/verify_e3_dense_config.py --static
uv run python scripts/train/verify_e3_dense_config.py --reattest
bash scripts/train/run_dense_cluster.sh --print-command
export CONFIRM_DENSE_RUN=YES
bash scripts/train/run_dense_cluster.sh --run
bash scripts/train/run_dense_cluster.sh --run \
  --resume "$DENSE_OUTPUT_ROOT/train/checkpoint-6250"
```

checkpoint：3125、6250、9375、12500、15625、18750、21875、25000、28125、31250。
标量：`$DENSE_OUTPUT_ROOT/train/train_metrics.jsonl`；恢复保持原输出目录。
只读 attestation 写 Dense 自己的记录，不修改共享数据。

## 3. DEV Selection 和 Dense Teacher

先准备冻结 DEV/TEST canonical manifests 与经审查的 source/group 语义证据。
合同示例结构见 `configs/eval/group_identity_contract.example.json`；未审查状态会拒绝 selection。
必须确认 group 表示原始 source 或完整编辑序列；图像 hash/多轮链也参与隔离审计。

```bash
export DENSE_MAGICBRUSH_DEV_MANIFEST=/path/to/frozen/magicbrush-dev.jsonl
export DENSE_CRISPEDIT_AUX_MANIFEST=/path/to/frozen/crispedit-dev.jsonl
export DENSE_SCALEEDIT_AUX_MANIFEST=/path/to/frozen/scaleedit-dev.jsonl
export DENSE_INTEREDIT_AUX_MANIFEST=/path/to/frozen/interedit-dev.jsonl
export DENSE_GROUP_IDENTITY_CONTRACT=/path/to/reviewed/group-identity.json
export DENSE_TEST_MANIFESTS_JSON='["/path/to/frozen/test.jsonl"]'
export E3_LORA_CHECKPOINT=""  # 有兼容权重时填路径
uv run python scripts/eval/evaluate_dense_checkpoints.py \
  --plan configs/eval/dense_dev_plan.yaml --preregister-selection
export CUDA_VISIBLE_DEVICES=0
uv run python scripts/eval/evaluate_dense_checkpoints.py \
  --plan configs/eval/dense_dev_plan.yaml
```

Released E0-region 为 baseline，使用相同 ROI-relative timestep 与推理协议。
资格：平均 Inside 改善 ≥0.005，配对 95% CI 支持严格正改善，平均 Outside 退化 ≤0.01。
差值均为 masked LPIPS 绝对值。source/group cluster bootstrap：seed42、10,000 repeats。
先对每 sample 的 generation seeds 平均，再做 sample-weighted 配对 cluster bootstrap。
合格候选按 Inside LPIPS 最低选择，并列较早 step；TEST 不参与选择。
无合格候选：PENDING/NO_ELIGIBLE_CHECKPOINT，停止正式 Teacher 导出。
预注册文件不可变；绑定 baseline、manifest、推理协议、选择规则和语义证据身份。

```bash
export SELECTED_DENSE_CHECKPOINT=/path/to/selected/dense/checkpoint
uv run python scripts/dimo/export_dense_teacher.py \
  --checkpoint "$SELECTED_DENSE_CHECKPOINT" --status selected \
  --selected-record "$DENSE_OUTPUT_ROOT/dev_evaluation/SELECTED_DENSE_CHECKPOINT.json" \
  --output-dir "$DENSE_OUTPUT_ROOT/teacher-selected"
export DIMO_TEACHER_CHECKPOINT="$DENSE_OUTPUT_ROOT/teacher-selected"
uv run python scripts/dimo/export_dense_teacher.py --verify "$DIMO_TEACHER_CHECKPOINT"
```

输出：`dev_evaluation/` 和 `teacher-selected/`。Teacher 不携带 SFT optimizer。
provisional 仅用于接口验证，不能用于正式 DiMO。

## 4. Full DenseDiMO

Teacher 冻结 eval；Student/Auxiliary 完整独立 Dense；不创建 LoRA。
普通 DDP；8×1×accum1=8；1 epoch/25,000 updates；两个 LR 均 1e-6。
AdamW=(0.9,0.999)、WD=0、constant/no warmup；EMA=0.9995。
角色参数、梯度、AdamW moments 和 EMA 均 FP32；forward BF16 autocast。
正式 objective：linear/roi-vocabulary-sum-v1；平方实现保留兼容。
FKL、sampling、pseudo-mask、noise、reference conditioning 使用固定配置。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash scripts/train/run_full_dense_dimo.sh --print-command
bash scripts/train/run_full_dense_dimo.sh --run
bash scripts/train/run_full_dense_dimo.sh --run \
  --resume "$DIMO_OUTPUT_ROOT/checkpoints/checkpoint-5000"
```

输出：`metrics.jsonl`、`memory_rank*.jsonl`、manifest、failure/acceptance 记录。
恢复 checkpoint：500、5000、10000、15000、20000、25000；提前停止也保存完整状态。
推理导出：0、500、2500、5000、10000、15000、20000、25000。
不为每 500 步监控保存全模型；step500 是首次验收的独立恢复点。
首次 step1/20 验证角色、梯度、首次 optimizer states、EMA、rank/cursor 和真实 logits 梯度。
step500 执行固定四例 DEV diagnostic 后继续同一训练任务，不重启 step0。
两 optimizer、EMA 完成且 rank 检查通过后才 commit step/cursor。
失败不假装内存回滚；必须重新加载最近完整 checkpoint。
NaN/Inf、OOM、rank mismatch、身份/数据/状态错误停止所有 rank。
小梯度、CE约9、低 surrogate、质量未改善触发诊断，不按绝对 loss 自动停止。
FSDP candidate 独立配置和代码未通过 GPU 验证，不由 DDP OOM 自动启用。

## 5. One-Step DEV Evaluation

```bash
export CUDA_VISIBLE_DEVICES=0
uv run python scripts/eval/evaluate_full_dense_dimo.py \
  --plan "$DIMO_DEV_PLAN" --run-root "$DIMO_OUTPUT_ROOT"
uv run python scripts/eval/evaluate_full_dense_dimo.py \
  --plan "$DIMO_DEV_PLAN" --run-root "$DIMO_OUTPUT_ROOT" --steps 500
```

输出：`$DIMO_OUTPUT_ROOT/evaluation/full_dev/`。
Teacher 多步、Raw Student 和 EMA 单步分别评测；推理只加载需要的一个 Dense role。
Inside 对 target；Outside preservation 对 source；算法说明见 `FULL_DENSE_PROTOCOL.md`。
token hard-lock 不保证 VQ 解码像素保持。固定 UID、mask、geometry、seed。
计时包含编码/Transformer/VQ解码，排除 canonical geometry 和图像文件 I/O。
默认 warmup1/repeats3；CUDA 同步；记录 phase timings 和真实 forward count。
四例 diagnostic 不能替代完整 DEV 或 cluster CI。尚未执行项标为 NOT_AVAILABLE。

## 6. Loss / Gradient Curves

Monitor 是独立 CPU 进程，训练不等待其绘图、上传或退出。
每 500 committed updates 只追加 Loss PNG 和 Gradient PNG 到 run prerelease。
不上传标量、模型、DEV 图像、报告或私人路径。Release 绑定实验开始的代码 SHA。

```bash
uv run python scripts/monitor/live_monitor.py --stage dimo \
  --log "$DIMO_OUTPUT_ROOT/metrics.jsonl" --output "$DIMO_OUTPUT_ROOT/curves" \
  --run-id "$RUN_ID" --git-sha "$DENSE_COMMIT" --upload
uv run python scripts/monitor/live_monitor.py --stage e3 \
  --log "$DENSE_OUTPUT_ROOT/train/train_metrics.jsonl" --output "$DENSE_OUTPUT_ROOT/curves" \
  --run-id "$RUN_ID" --git-sha "$DENSE_COMMIT" --upload
```

在独立终端/tmux 启动；GH_TOKEN 仅给 Monitor，或使用已认证 gh CLI。
无认证退化为本地绘图；上传错误有限重试，训练继续；缓存 PNG 数量有界。
重启补传可加 `--retry-pending`；只绘图去掉 `--upload`；单次处理加 `--once`。
Loss 图注明 objective，线性 loss 可为负；独立显示 Aux CE 与 0.5 mean_roi(||g||²)。
梯度未采集值为 null，曲线不伪造 0 或测量点；原始尖峰保留。

## 7. Lightweight Packaging 和验证

```bash
uv run python scripts/report/build_experiment_report.py --stage all \
  --e3-root "$DENSE_OUTPUT_ROOT" --dimo-root "$DIMO_OUTPUT_ROOT" \
  --output /path/to/experiment_report.tar.gz
CUDA_VISIBLE_DEVICES='' uv run pytest -q
uv run python -m compileall -q src scripts tests
git diff --check
CUDA_VISIBLE_DEVICES='' uv run torchrun --standalone --nproc-per-node=2 \
  scripts/train/train_full_dense_dimo.py --config configs/dimo/full_dense_local_smoke.yaml \
  --synthetic --output-dir /path/to/new/cpu-smoke
```

报告仅本地保存，包含标量/身份/曲线/DEV指标/固定 montage/错误；排除权重与完整数据。
支持 partial report，附内部文件校验与 tar.gz SHA256 sidecar。
CPU/Gloo PASS 不等于 GPU PASS；首次真实 8-GPU 完整 step 和恢复测试单独验收。
真实 resume 等价性按固定配置运行 fresh20、fresh10→resume20，比较完整训练状态。
使用两个独立输出目录，并在各自相同目录恢复：

```bash
CUDA_VISIBLE_DEVICES='' uv run python scripts/train/compare_full_dense_resume.py \
  --fresh /path/to/fresh/checkpoints/checkpoint-20 \
  --resumed /path/to/resumed/checkpoints/checkpoint-20 \
  --output /path/to/new/resume_acceptance.json
```

完整恢复测试结果未产生前，不宣称 GPU resume 等价性通过。
