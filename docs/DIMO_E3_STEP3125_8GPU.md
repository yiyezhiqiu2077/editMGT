# E3 step3125 → Region-DiMO 探索性实验

使用独立分支 `exp/dimo-e3-step3125-8gpu-v1` 和独立 worktree。不要在 E3 正在训练的目录中切换分支、安装依赖或修改代码。以下命令均在独立 worktree 中执行，GPU 必须独立分配。

teacher 是 E3 checkpoint-3125（0.5 epoch），不是选出的最佳 checkpoint。所有运行均为 `dev_only=true / formal_teacher=false`；正式 DiMO gate 保持关闭，不生成 selection 或 formal registration 文件，不运行 TEST。

## 环境与已有资产

在集群的独立 shell 中设置现有资产的绝对路径，不重新构建 D200K：

```bash
export EDITMGT_WORKTREE=/absolute/path/to/independent/worktree
export EDITMGT_MODEL_ROOT=/absolute/path/to/released/EditMGT/snapshot
export E3_CHECKPOINT_DIR=/absolute/path/to/E3/checkpoint-3125
export DIMO_TEACHER_CHECKPOINT=/absolute/path/to/dimo/assets/teacher-step3125
export DIMO_D200K_MANIFEST=/absolute/path/to/frozen/train_200k.jsonl
export DIMO_CORPUS_READY=/absolute/path/to/frozen/CORPUS_READY.json
export MAGICBRUSH_ROOT=/absolute/path/to/magicbrush/train
export CRISPEDIT_ROOT=/absolute/path/to/crispedit
export SCALEEDIT_ROOT=/absolute/path/to/scaleedit
export INTEREDIT_ROOT=/absolute/path/to/interedit
export MAGICBRUSH_DEV_ROOT=/absolute/path/to/magicbrush/dev
export DIMO_DEV128_MANIFEST=/absolute/path/to/frozen/validation/magicbrush_probe128.jsonl
export DIMO_RUN_ROOT=/absolute/path/to/separate/dimo/run
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

先按 [ENVIRONMENT.md](ENVIRONMENT.md) 配置独立 Python 环境和现有模型缓存。LPIPS 评估需要缓存中的预训练权重。输出必须与 E3 训练目录、teacher bundle 分离。启动器检查已有 GPU 进程和可用显存，遇到资源冲突只退出，不停止其他训练。

## 导出 provisional teacher

```bash
uv run python scripts/dimo/export_provisional_teacher.py \
  --checkpoint "$E3_CHECKPOINT_DIR" --model-root "$EDITMGT_MODEL_ROOT" \
  --output-dir "$DIMO_TEACHER_CHECKPOINT"
```

检查 step=3125、已消费样本=100000、E3 配方、模型身份和文件 hash；只复制 LoRA/region sidecar，不复制 E3 optimizer。manifest 为 `dimo_provisional_teacher_manifest.json`。临时目录完成验证后原子发布；重用 bundle 时再次校验全部 hash。

## 8-GPU smoke：fresh20 与 10+resume20

使用同一份 [smoke 配置](../configs/dimo/smoke_e3_step3125_8g.yaml)。`--stop-after-steps` 只控制本次暂停点，不改变配置 hash 或 scheduler horizon。

```bash
export DIMO_OUTPUT_ROOT="$DIMO_RUN_ROOT/fresh20"
bash scripts/train/run_dimo_8g.sh --config configs/dimo/smoke_e3_step3125_8g.yaml --print-command
bash scripts/train/run_dimo_8g.sh --config configs/dimo/smoke_e3_step3125_8g.yaml --run

export DIMO_OUTPUT_ROOT="$DIMO_RUN_ROOT/first10"
bash scripts/train/run_dimo_8g.sh --config configs/dimo/smoke_e3_step3125_8g.yaml --run --stop-after-steps 10

export DIMO_OUTPUT_ROOT="$DIMO_RUN_ROOT/resume20"
bash scripts/train/run_dimo_8g.sh --config configs/dimo/smoke_e3_step3125_8g.yaml --run \
  --resume "$DIMO_RUN_ROOT/first10/smoke-e3-step3125-8g/checkpoint-10"

uv run python scripts/train/verify_dimo_8g_smoke.py \
  --fresh20 "$DIMO_RUN_ROOT/fresh20/smoke-e3-step3125-8g" \
  --fresh10 "$DIMO_RUN_ROOT/first10/smoke-e3-step3125-8g" \
  --resume20 "$DIMO_RUN_ROOT/resume20/smoke-e3-step3125-8g" \
  --output "$DIMO_RUN_ROOT/smoke-report.json"
```

验收逐 rank 梯度隔离、更新前 AllReduce、EMA、outside token hard-lock、无重复样本、配置/teacher 身份、完整 checkpoint 和逐 rank RNG 恢复。报告分别记录 `bitwise_exact` 与 `numerical_tolerance`，不把 CPU/Gloo 测试计作真实 GPU smoke。

## 500-step pilot

先按下节的 teacher 命令验证 provisional teacher 的 DEV 编辑质量。只有真实 smoke 全部必要检查通过，才可运行 pilot：

```bash
export DIMO_SMOKE_REPORT="$DIMO_RUN_ROOT/smoke-report.json"
export DIMO_OUTPUT_ROOT="$DIMO_RUN_ROOT/pilot500"
bash scripts/train/run_dimo_8g.sh --config configs/dimo/pilot_e3_step3125_8g.yaml --print-command
bash scripts/train/run_dimo_8g.sh --config configs/dimo/pilot_e3_step3125_8g.yaml --run
uv run python scripts/train/summarize_dimo_pilot.py \
  --run-dir "$DIMO_OUTPUT_ROOT/pilot-e3-step3125-8g" --output "$DIMO_RUN_ROOT/pilot-report.json"
```

8 ranks × batch1 × accumulation1 = global batch8；500 更新消费4000条样本，保存 step0/50/100/250/500。每 rank 一份 frozen backbone，三角色共享主干；student/auxiliary 梯度分别按稳定顺序合桶平均后更新各自 optimizer。

参考超参数：FKL、初始化 MASK 比例0.5、embedding noise σ=0.3、每次 student 更新对应一次同 batch auxiliary 更新、EMA=0.9999、GT anchor=0、student/aux LR=1e-6、constant scheduler、warmup=0。这些不是已验证最优参数。ROI-only 初始化、distribution gradient、surrogate、auxiliary CE、sampling 与 EMA 数学方法不变。

`metrics.jsonl` 记录损失、梯度、更新比例、EMA 距离、ROI 比例、finite、样本游标、耗时和逐 rank peak allocated VRAM；`rank_consistency.jsonl` 记录角色/EMA/optimizer/scheduler hash。损失大小不能代替编辑质量判断。

checkpoint 在更新边界由全部 rank 捕获 RNG，仅 rank0 写入临时目录并原子发布。恢复严格绑定 teacher、base、Git SHA、完整配置、world size、sampler、optimizer/scheduler 和游标。可恢复到新输出目录；原目录恢复要求日志恰好截止于被恢复 checkpoint，拒绝从旧 checkpoint 追加到较新日志。

## 固定 DEV128 比较

teacher 先用12步 E3推理做诊断。随后比较 raw student0/50/100/250/500、EMA50/100/250/500；输入、geometry、mask、base seed 固定。teacher 与 one-step sampler 的随机子流不同，不能声称两者使用相同采样算法。

默认 teacher CFG=5、student CFG=1，仅用于诊断，不声明最优 CFG。比较 inside LPIPS/PSNR/SSIM、outside LPIPS-to-source、full LPIPS 和 warm-model 推理延迟；延迟包含输入编码、模型推理和 VQ decode，不含模型加载及 PNG IO。

在独立空闲 GPU 上执行（不要占用 E3 或仍在训练的 DiMO GPU）：

```bash
export CUDA_VISIBLE_DEVICES=0
pilot="$DIMO_RUN_ROOT/pilot500/pilot-e3-step3125-8g"
common=(--teacher-checkpoint "$DIMO_TEACHER_CHECKPOINT" \
  --manifest "$DIMO_DEV128_MANIFEST" --corpus-ready "$DIMO_CORPUS_READY" \
  --canonical-root "$MAGICBRUSH_DEV_ROOT" --model-root "$EDITMGT_MODEL_ROOT")

uv run python scripts/eval/evaluate_dimo_dev128.py "${common[@]}" \
  --role teacher --output-dir "$DIMO_RUN_ROOT/dev/teacher-3125"

for step in 0 50 100 250 500; do
  uv run python scripts/eval/evaluate_dimo_dev128.py "${common[@]}" --role student \
    --student-checkpoint "$pilot/checkpoint-$step" --output-dir "$DIMO_RUN_ROOT/dev/student-$step"
done
for step in 50 100 250 500; do
  uv run python scripts/eval/evaluate_dimo_dev128.py "${common[@]}" --role ema \
    --student-checkpoint "$pilot/checkpoint-$step" --output-dir "$DIMO_RUN_ROOT/dev/ema-$step"
done
uv run python scripts/eval/compare_dimo_dev128.py --runs \
  "$DIMO_RUN_ROOT/dev/teacher-3125" \
  "$DIMO_RUN_ROOT/dev/student-0" "$DIMO_RUN_ROOT/dev/student-50" \
  "$DIMO_RUN_ROOT/dev/student-100" "$DIMO_RUN_ROOT/dev/student-250" "$DIMO_RUN_ROOT/dev/student-500" \
  "$DIMO_RUN_ROOT/dev/ema-50" "$DIMO_RUN_ROOT/dev/ema-100" \
  "$DIMO_RUN_ROOT/dev/ema-250" "$DIMO_RUN_ROOT/dev/ema-500" \
  --output-dir "$DIMO_RUN_ROOT/dev/comparison"
```

DEV 入口仅接受 `CORPUS_READY` 中冻结的验证 manifest，并检查 train/DEV 源图及 group 无泄漏；不接受 TEST。已有合法 CrispEdit/ScaleEdit/InterEdit aux128 可另设 `--dataset`、manifest 和相应 canonical root，分数据集比较，不重采样。拼图固定使用 manifest 前6条，不按生成质量挑选。

## 完整 epoch 入口

[完整 epoch 配置](../configs/dimo/d200k_e3_step3125_8g.yaml) 为200000样本、global batch8、25000更新，每2500步保存。先审阅 pilot 和 DEV 结果，再单独决定是否运行；不自动启动。入口另要求 `CONFIRM_DIMO_LONG_RUN=YES` 和匹配的真实 smoke 报告。不要直接从500-step配置恢复到25000-step配置：完整配置不同，strict resume 会拒绝，应另开实验目录从同一 teacher 初始化。

## 本地检查

```bash
CUDA_VISIBLE_DEVICES='' uv run pytest -q
CUDA_VISIBLE_DEVICES='' uv run python -m compileall -q src scripts tests
find scripts -name '*.sh' -print0 | xargs -0 -n1 bash -n
git diff --check
```

CPU 测试包含实际8进程 Gloo同步、20-step fresh/resume、逐 rank RNG、sampler 和 teacher加载；它们只验基础设施，不证明真实1024分辨率8-GPU训练可运行或有质量收益。
