# Soft-DiMO-Editing-Reward：审计、实现与执行边界

本方案作为可验证的短程实验是合理的，不能预先保证改善硬 Token 编辑质量。
依据当前 DEV128，4000步 EMA 的 LPIPS 优于8000/12000步，选4000完整断点有依据。
主要风险是冻结解码器对概率期望 embedding 的响应与真实 categorical Token 不一致。
必须以硬采样固定 DEV 判断效果，soft Reward loss 下降不能代替这个判断。

## 来源与隔离

- 实际 DiMO-v2 基线：`23c4fb181731376f19e580cbfd53ae3e77b4fbdd`。
- 独立分支：`exp/soft-dimo-editing-reward`。
- 工作树：`/data02/zhangyuyang/code/editMGT-soft-dimo-editing-reward`。
- 原工作树、原训练、所有历史 checkpoint/日志/评测不修改。原训练保持停止。
- fork：`/data/zhangyuyang/editmgt_dimo_v2_20261013/train/checkpoints/checkpoint-4000`。
- Student 从 **Raw4000** 初始化，Auxiliary 从同一完整断点初始化，Teacher 保持原 selected SFT。
- EMA 保留同一4000断点的 trainable shadow 值和 decay=0.9995，**新实验 update_count=0**。
- 两个 AdamW optimizer、scheduler、新实验 sampler cursor、RNG 全部独立初始化，不读父断点 optimizer state。
- 这是显式 weights-only fork，不是原训练的中断续跑；新断点的严格 resume 会恢复本实验 optimizer、sampler、RNG、EMA、Reward 校准状态。
- 父断点全部30个文件的 SHA256 已核验，870个 Student 张量；约27.96 GB，未复制/覆盖父权重。

官方参考固定到 `yuanzhi-zhu/SoftDiMO@7d2a01b81c5c0b0729ddfe6ef8e43718dcabae01`：
[`Meissonic_src/train_SoftDiMO_Meissonic.py` L1388–1415](https://github.com/yuanzhi-zhu/SoftDiMO/blob/7d2a01b81c5c0b0729ddfe6ef8e43718dcabae01/Meissonic_src/train_SoftDiMO_Meissonic.py#L1388-L1415)。
其 reward 分支使用 FP32 softmax、概率乘 VQ codebook、post_quant_conv/decoder，并把 Reward 加到 Generator loss。
同文件 L245–255 的 Transformer embedding 用于其他 soft DiMO/GAN 路径，不能拿来代替 reward 的 VQ embedding。
官方全图文本 Reward 没有编辑任务的 ROI 外锁定与配对 Target，因此本实现是编辑扩展。

## 保留的基线机制

Full Dense、FKL、FP32 Linear Surrogate、Fresh Batch B、冻结 Student/Auxiliary Embedding/MLM Head、
Teacher CFG4 query、Student CFG1、r_init=1、sigma=0.3、ROI token hard lock、AdamW 和原 EMA update 公式保持不变。
`src/explicit_region/metrics.py` 与原 hard inference/evaluation 实现不改动。

`soft_reward` 是 content-bound recipe extension：checkpoint 继续使用原 Full Dense v3 布局，
完整 identity 加入 fork provenance、Reward 配置、模型文件哈希和库版本。
不同 Reward/父断点/配置/git SHA 无法作为同一 recipe resume；不额外训练 Reward 模型。

## 可微路径与三个 Reward

```
Student logits [B,64,64,V]
   → ROI 内 FP32 softmax，只取8192个有效类别
   → probabilities @ vqvae.quantize.embedding.weight
   → ROI 外填 Source 的冻结 codebook embedding
   → post_quant_conv → decoder → clamp(0,1)
   → Masked LPIPS(Target) + DINO(Target ROI) + CLIP-I(Target ROI)
   → 同一个 Student total loss.backward() → 全局 clip10 → Student optimizer
```

实际 VQGAN 的 hard decode 使用 `clamp(0,1)`，不添加 `[-1,1]→[0,1]` 映射，不添加 scaling_factor。
冻结解码器参数不等于切断输入梯度：生成图像路径不经过 no_grad、detach、PIL、NumPy 或硬 categorical 索引。
Source lookup 和目标特征是常量；无梯度 hard decode 仅用于诊断，与 Reward backward 分开。
ROI 外 embedding 和 logits 梯度严格锁定；VQGAN 卷积/GroupNorm 的空间耦合意味着 ROI 外**像素**不保证逐位相等，仍测 outside LPIPS。

- `DifferentiableMaskedLPIPS` 保留原 LPIPS 多层特征与 area mask 加权，移除生成输入的 no_grad。训练 Reward 使用512分辨率，Target/生成图像完全同样 resize；固定 DEV 保持原1024评测。
- DINO/CLIP-I：同一个 pixel mask bbox + max(8px, 0.1 longest side) 上下文 crop，对生成/Target 完全一致。
  Torch bicubic antialias resize + center crop + 固定模型 RGB mean/std；输入已经是[0,1]，不重复除255。
  DINO 使用裁剪后的 CLS 特征局部语义相似度，避免整图背景占优；CLIP-I 同样在 ROI 上比较。
  训练的 torch 插值不声称与 PIL 量化插值逐像素相同，hard/soft DEV 均用原评测 processor。
- 不使用编辑指令充当完整 Target caption；本版本关闭 CLIP-T。将来有可靠完整目标描述时才单独注册文本 Reward recipe。
- Frozen VQ decoder/DINO/CLIP 用 non-reentrant activation checkpointing 控制内存；生成路径使用 FP32，Student 前向仍 BF16 autocast。

## 梯度校准与日志

前4个真实 A batch 先测 `dL_linear/dlogits` 和三个未加权 Reward 的 logits 梯度 L2 范数，
跨 DDP rank 求平均并累加；这4步仍只优化原 Linear DiMO。
第5步起固定 `lambda_i = 0.1 * sum(norm_linear) / sum(norm_reward_i)`。
不按 loss 标量匹配权重，不动态追逐训练 loss；三项初始各取约0.1倍原 logits 梯度尺度，
合成梯度可能相互加强/抵消，因此同时记录总 logits 与实际 Student 参数梯度范数。
lambda>100、任一 rank 的单项梯度为0/非有限时集体中止，不静默剪成一个未经审核的权重。

每步记录原 `loss_dimo`、三个 Reward loss、原始/加权单项梯度范数、lambda、总 logits 范数、
总 loss、Student 全局 pre/post clip 范数。每20步及初始步记录同 A batch 的 categorical hard/soft gap。
这些 batch gap 只是诊断，不能把不同 batch 的 gap 波动解释为固定样本趋势。
校准计数、范数累积和冻结 lambda 随完整 checkpoint 保存；校准计数必须等于 min(step,4)。
Auxiliary Fresh B 阶段不调用 Reward，仍使用 Student 更新后重新生成的 B。

## 固定 DEV 与扩展门槛

固定计划复用 `/data02/zhangyuyang/experiments/editmgt_dimo_v2_20261013/dev128_plan_save2000.json`。
128样本、seed0/1、106个 source-group clusters，Raw/EMA categorical hard 一步，CFG1、r_init1、sigma0.3。
step0 必须逐个生成 PNG 与原4000步 Raw/EMA 匹配；否则停止，先处理 fork/protocol 差异。
step20、step500 同时生成 hard 和 soft diagnostic，比较原 DiMO4000 Raw/EMA。
配对统计先平均两个 seed，再做10000次 source-cluster bootstrap，seed42。
固定前8例 seed0生成拼图，不按质量挑样本；Teacher一步列只读复用已有基线。

Gate 为 `STOP_REVIEW` 的条件：

1. 任一 Raw/EMA hard 的 inside/outside LPIPS、ROI CLIP/DINO 出现配对95% CI不跨0的退化。
2. 同样本的 hard–soft Target distance gap，相对step0显著扩大：LPIPS CI下限>0.01，CLIP/DINO distance gap CI下限>0.02。
3. step0硬结果不同于原4000、Reward梯度断连/非有限、冻结权重违例、Smoke失败/严重OOM。

`PASS` 只表示未检出这些停止条件，不证明统计等效或质量提升。
20步 Smoke proof和固定DEV20 Gate 都通过，才允许**继续到累计500步**，不是再加500步。
入口和直接 trainer 都限制最多500步，不提供自动长训。GPU失败不自动重试，也不抢占其他任务。
保留父 EMA 时，500次新update后原shadow仍约占0.9995^500≈77.9%；短程要重点看 Raw，EMA改善会滞后。

建议后续增加相同4000初始化、相同 fresh optimizer/seed 的无 Reward 500步对照，才能区分 Reward 与继续训练本身的影响。
这个对照及单项 Reward ablation **尚未执行**，不将本联合实验称为三项 Reward 的独立因果验证。

## CPU 验证与未执行项目

证据目录：`/data02/zhangyuyang/experiments/soft_dimo_editing_reward_from4000_20261013`。

- `SOURCE_AUDIT.json`：实际父完整断点哈希、配置审计。
- `REAL_CPU_DECODE_REWARD.json`：真实8192类 VQGAN，4×4 latent→64×64图像；真实LPIPS/DINO/CLIP均有非零有限 logits 梯度。
  梯度范数：LPIPS≈0.01715、DINO≈0.01942、CLIP≈0.02443，总范数≈0.03600。
  ROI外和mask类梯度为0；one-hot极限与native hard decode最大误差0。
  这不是实际训练 batch 的 lambda 校准，不是1024 GPU显存/质量证据。
- 数值梯度、ROI锁定、LPIPS评测聚合一致性、FP32/BF16输入、冻结权重、完整reward事务、fresh B、EMA、recipe隔离、恢复校准和下一步逐位重放均有CPU测试。
- `cpu_ddp_final/CPU_DDP_PASS.json`：2-rank CPU DDP参数和校准一致、完整断点下一步逐位重放、单rank reward失败集体中止。
- 全套CPU测试最终 **436 passed**（297.36s），见 `CPU_TESTS_FINAL.xml`；定向38项见 `CPU_FOCUSED_FINAL.xml`。

**未执行：1024分辨率GPU可微反向、20-step GPU Smoke、500-step短训、fixed DEV20/500、实际拼图及Soft/Hard质量改善、真实GPU恢复对照、无Reward控制实验、长训练。**
CPU数据流正确不能当作以上实验已通过。

## 资源与真正执行的命令

原DiMO-v2实测每卡peak reserved≈47.68 GiB；新Reward路径粗估每张A800需60–75 GiB，batch/GPU=1、4卡DDP、1024图像。
这是估计，1024反向峰值尚未测；Smoke reserved峰值>75 GiB则不扩大训练。
实际需要由20-step memory_rank日志确认。用户 continue 后已启动GPU Smoke；第一次在首步提交前因CUDA抗锯齿resize的非确定性backward报错退出，详见下文。

完整断点≈27.96 GB，Raw+EMA推理导出≈8.07 GB。
新实验完整断点在 `/data/zhangyuyang/soft_dimo_editing_reward_from4000_20261013_retry01/train`，
新推理导出在 `/data01/zhangyuyang/soft_dimo_editing_reward_from4000_20261013_retry01/inference`，
通过新实验自身的 inference symlink 关联，不改变原实验目录。
包含Smoke20/short500保留和atomic空间，预算 `/data`≈94.88 GB、`/data01`≈39.18 GB；
检查时剩余约108.56/43.84 GB。命令会再次实际检查，空间变化会拒绝运行，不自动清理文件。

以下为修复后的独立 retry01 命令；是否执行以 GPU_EXECUTION_STATUS.json 和实际输出为准。须保持新工作树 clean commit，与checkpoint identity绑定。

```bash
cd /data02/zhangyuyang/code/editMGT-soft-dimo-editing-reward
SOFT_DIMO_PYTHON=/data02/zhangyuyang/code/editMGT/.venv/bin/python
SOFT_DIMO_RUN=/data/zhangyuyang/soft_dimo_editing_reward_from4000_20261013_retry01/train
SOFT_DIMO_PLAN=/data02/zhangyuyang/experiments/editmgt_dimo_v2_20261013/dev128_plan_save2000.json
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1

# 1. 显式fork，20步Smoke；新输出目录必须不存在，不自动后续训练。
CUDA_VISIBLE_DEVICES=0,1,2,3 "$SOFT_DIMO_PYTHON" scripts/train/run_soft_dimo_reward.py \
  --config configs/dimo/soft_reward_from4000_deterministic_resize.yaml --mode smoke

# 2. 仅评测，包含step0硬图与原4000一致性以及step20固定Soft/Hard gate。
CUDA_VISIBLE_DEVICES=0 "$SOFT_DIMO_PYTHON" scripts/eval/evaluate_soft_dimo_reward.py \
  --run "$SOFT_DIMO_RUN" --step 20 --plan "$SOFT_DIMO_PLAN"

# 3. 只有GPU_SMOKE_PASS + QUALITY_GATE_step20都PASS才恢复同一新recipe，累计500步。
CUDA_VISIBLE_DEVICES=0,1,2,3 "$SOFT_DIMO_PYTHON" scripts/train/run_soft_dimo_reward.py \
  --config configs/dimo/soft_reward_from4000_deterministic_resize.yaml --mode short

# 4. 仅评测500步hard质量与Soft/Hard gap，输出报告和固定拼图；不扩展训练。
CUDA_VISIBLE_DEVICES=0 "$SOFT_DIMO_PYTHON" scripts/eval/evaluate_soft_dimo_reward.py \
  --run "$SOFT_DIMO_RUN" --step 500 --plan "$SOFT_DIMO_PLAN"
```

命令CPU预览可加 `--print-command`（只用于train launcher），不会启动GPU。

CPU重现：

```bash
PATH=/data02/zhangyuyang/code/editMGT/.venv/bin:/data02/zhangyuyang/miniconda3/envs/lumina_dimoo/bin:$PATH \
  UV_PROJECT_ENVIRONMENT=/data02/zhangyuyang/code/editMGT/.venv CUDA_VISIBLE_DEVICES='' \
  "$SOFT_DIMO_PYTHON" -m pytest tests -q
CUDA_VISIBLE_DEVICES='' "$SOFT_DIMO_PYTHON" scripts/diagnostics/verify_soft_reward_decode_cpu.py --output /tmp/soft_reward_real_cpu.json
CUDA_VISIBLE_DEVICES='' "$SOFT_DIMO_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests/dimo/soft_reward_ddp_worker.py --output /tmp/soft_reward_cpu_ddp_new
```

GPU Smoke 首次兼容性修复（2026-10-14 北京时间）：严格确定性模式下，当前PyTorch的
CUDA `interpolate(antialias=True)` backward 报错；4 rank均在step1提交前一致abort，
没有optimizer/EMA更新。原失败目录和step0导出保留，不覆盖。
修复使用CPU预计算固定抗锯齿轴滤波矩阵，生成图像仅在原设备上做可微separable matmul；
生成图像没有CPU/NumPy/PIL/detach中转，保持严格确定性和gen/target一致的滤波规则。
CPU double forward/backward与原interpolate对齐并做gradcheck；CUDA测试包括1024→512以及bicubic ROI resize。
滤波backend显式绑定进recipe，修改后仅能使用新的checkpoint。
新推理存储移至/data01独立目录，避免保留失败现场后/data02空间不足；无历史文件清理。
