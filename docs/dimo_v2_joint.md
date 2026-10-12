# DiMO-v2 五项联合配方与验收记录

本实现位于独立 worktree `/data02/zhangyuyang/code/editMGT-dimo-v2`，分支
`exp/dimo-paper-aligned-v2`，基线为 `047100ae10d84f6c76971abe337a13c65bf60649`。
参考 DiMO 源码版本为 `24613741ec9ca730273a6ebc822327878d5a086b`。
原工作树、原 checkpoint、日志和正在运行的四卡任务不作修改。

## 配方及预算

新配置为 [dimo_v2_mixed20k.yaml](../configs/dimo/dimo_v2_mixed20k.yaml)。
Student/Auxiliary 在 DDP 包装及 optimizer 创建前冻结完整 `embed`、`mlm_layer`；
`edit_region_embedding` 保持训练。冻结 MLM 的 forward 保留 autograd。
optimizer 精确覆盖对应角色的全部可训练参数，冻结权重及梯度单独审计。
Student 在 DDP 梯度同步后按全部可训练梯度作 global L2 clipping，阈值为 10；
Auxiliary 不裁剪。Teacher 查询 CFG=4，Student/Auxiliary CFG=1。
CFG conditional/unconditional logits 在组合前转 FP32，使用空字符串 unconditional。

保留现有 Linear Surrogate、FKL、r_init=1、ROI hard lock、sigma=0.3、
AdamW LR=1e-6、恒定 LR、EMA decay=0.9995。未修改 surrogate.py/divergence.py 的数学实现。
没有新增 SFT anchor、negative prompt、warmup、EMA ramp-up 或单因素实验。

Auxiliary 对 pseudo-mask 选中的 hard token 计算 per-sample normalized CE，
`label_smoothing=0.1`，目标 detach，`soft_target_weight=0`。
有效 VQ 词表为 8192 类，不包含输入 MASK。PyTorch smoothing 对应
`0.9 * one_hot(y) + 0.1 * Uniform(8192)`；接受它改变 Auxiliary 分布和 DiMO 梯度场，
不做逆变换。smoothing=0 的数值及梯度旧逻辑等价性有回归测试。

每个 data epoch 使用一个确定性全局 permutation，连续消费全局 A、B 两批。
所有 rank 的 A/B UID 联合校验，禁止重叠、补齐、丢弃和跨 epoch 配对。
A 更新 Student 后，B 用更新后的 Student 在 no_grad 下重新构造全 MASK 初态、
生成 ROI noise 并采样；B 使用自己的 source、ROI、文本条件和 model kwargs。
B 初态、noise、categorical sampling、pseudo masking 使用独立 RNG namespace。

| 预算项 | Mixed20K v2 |
| --- | ---: |
| rows/data epoch | 20000 |
| world size / batch per GPU | 4 / 1 |
| global batches/完整 optimizer step | 2 |
| 样本消费/完整 step | 8 |
| optimizer steps/data epoch | 2500 |
| data epochs | 5 |
| Student / Auxiliary updates | 各 12500 |
| total sample consumptions | 100000 |
| 每角色用于更新的 sample consumptions | 各 50000 |

身份记录明确绑定 `budget_basis: total_data_consumption`。不能把它解释成每角色独立看完五轮数据。
当前 Teacher 沿用用户已选择的 Mixed20K SFT epoch 3（step 3750），
Student/Auxiliary/optimizer/EMA 全新初始化，不继承旧 DiMO optimizer。

真实 Teacher safetensors header 统计：每角色参数 1009307648，冻结 18941952，
可训练 990365696（98.1233%）。GPU 运行时还会输出实际角色/optimizer 参数审计报告。

## 事务、身份、EMA 与诊断

继续复用 `StepCommit`；只有 Student、fresh Student B、Auxiliary、EMA、rank state 审计
全部成功后，完整 step/cursor 才提交。任何部分更新失败都使事务不可继续，
必须从最后一个完整 checkpoint 重载并重放；不尝试原地回滚 optimizer。

旧 recipe/旧 RNG identity/旧 checkpoint schema `full-dense-dimo-ddp-v2` 保留。
新 recipe 是 `dimo-v2`，checkpoint schema 为 `full-dense-dimo-ddp-v3`。
新身份绑定冻结策略、预算、按顺序列出的 optimizer 参数 name/shape/dtype、
EMA 参数集合、RNG namespace、Teacher 身份、配置哈希和 Git SHA。
v2 不能恢复旧 checkpoint 或不兼容 optimizer。恢复还验证完整文件哈希、
模型 shape/dtype、AdamW moments、sampler/cursor/epoch、EMA count、累计裁剪计数。

EMA shadow 只覆盖训练时可训练参数，部分 shadow 必须严格等于该集合。
Raw/EMA inference export 都包含完整 Student state（参数与 buffer）；EMA 在完整 state
上覆盖 shadow，冻结参数不丢失。加载继续严格校验完整权重集合、shape、dtype、哈希。

日志包含 preclip/postclip norm、触发标志和累计 clip_fraction，以及 attention/FFN/embed/MLM
分组范数。冻结的 embed/MLM 范数必须为零。
16 坐标抽样更新量明确记录为 `student_parameter_update_ratio_sampled`。
指定诊断步 `[1,20,2000,6000,12000,12500]` 才计算全部可训练坐标的 full update norm/ratio，
分母为更新前可训练参数的全 L2 范数；CPU snapshot 避免新增整模型 GPU 副本。
累计裁剪计数随 checkpoint 保存与恢复。如实际长期 clip_fraction 接近 100%，
应在实验报告中标记，代码不自动修改阈值。

## DEV128 协议

协议构建入口为 [prepare_dimo_v2_dev128.py](../scripts/eval/prepare_dimo_v2_dev128.py)。
它只读检查已有 DEV128 manifest、原评测 plan、Teacher12 plan、旧 Raw/EMA 的每 rank
身份和预测/PNG 哈希，并输出独立 v2 plan。已通过实文件验证的计划为：
`/data02/zhangyuyang/experiments/editmgt_dimo_v2_20261013/dev128_plan_save2000.json`。

manifest SHA256：`5dcbb70bc28126e226026c5f3f2836ce58c0642780b027a097f49626a986f731`。
共 128 个样本、106 个 source groups；seeds=[0,1]；native 1024×1024 source/target/mask。
严格验证它们的对应关系、文件哈希、非空 ROI 和分组身份，避免随机 crop 改变旧比较条件。

| 模型 | CFG | steps | Student 初态/noise |
| --- | ---: | ---: | --- |
| 训练查询 Teacher | 4 | 单次分布查询 | 不属于 DEV Teacher 生成配置 |
| DEV SFT Teacher | 5 | 12 | 复用已验证结果 |
| DEV 旧 Raw/EMA | 1 | 1 | r_init=1, sigma=0.3 |
| DEV v2 Raw/EMA | 1 | 1 | r_init=1, sigma=0.3 |

所有 DEV 模型使用 FP32 参数、BF16 autocast forward、FP32 VQ、math SDP、关闭 TF32，
reference_strength=1、roi_relative timestep、temperature=1、top_k=0、top_p=0。
所有五个模型的指标用同一个 evaluator 从 PNG 重算；先对每个样本的两个 seed 求均值，
再按 source group 作 paired cluster bootstrap。包含 inside/outside 像素及 LPIPS、
完整图 CLIP/DINO，以及与旧代码相同的 contextual ROI CLIP/DINO crop。
同时生成五模型比较图。旧 Raw/EMA 使用已有 step 5000 结果，需在解读中保留其步数差异；
该联合比较不构成对单个修改的因果归因。

Mixed20K 非 formal Teacher 的评测由 checkpoint 注册实验身份和 Teacher bundle 验证，
不强制 formal=True，也不跳过身份验证。单图 r_init 从 checkpoint identity 读取；
无记录时必须显式给值，不默认为 0.5。v2 单图配置必须符合已记录 recipe。

## 验证、资源和后续入口

CPU 测试覆盖：既有 suite、Linear Surrogate 数值回归、8192 类 smoothing 数值/梯度、
人工大梯度裁剪、冻结 MLM 的反传、optimizer/EMA membership、完整 Raw/EMA 加载、
fresh B 的更新时序/独立条件/noise、CFG FP32、严格配置及 checkpoint 拒绝测试。
两进程 Gloo 在三轮数据中验证连续运行与 epoch 边界/中途恢复的 Student、Auxiliary、
optimizer、scheduler、EMA、rank RNG、loader RNG、sampler、cursor、clip stats 全状态一致。
在 Student optimizer 后、fresh forward、Auxiliary optimizer 后、EMA 中途/完成和
commit 前注入单 rank 故障，验证所有 rank 退出、整步不提交、重载回放一致。
CPU synthetic 结果仅是实现验收证据。

实际报告位于 `/data02/zhangyuyang/experiments/editmgt_dimo_v2_20261013/`：
`cpu_pytest_final.xml` / `.log`、`preflight.json`、`resolved_mixed20k_save2000.yaml`、
`gpu_smoke_preflight.log`，以及独立 CPU 连续/恢复运行与推理的验收结果。

GPU 20-step smoke 入口如下（应等原四卡任务释放资源后使用）：

```bash
cd /data02/zhangyuyang/code/editMGT-dimo-v2
/data02/zhangyuyang/code/editMGT/.venv/bin/python scripts/train/run_dimo_v2.py \
  --config /data02/zhangyuyang/experiments/editmgt_dimo_v2_20261013/resolved_mixed20k_save2000.yaml \
  --mode smoke --output-dir /data/zhangyuyang/editmgt_dimo_v2_20261013/gpu_smoke \
  --inference-output-dir /data02/zhangyuyang/experiments/editmgt_dimo_v2_20261013/gpu_smoke_inference
```

入口在启动 torchrun 前只读检查 GPU 占用和磁盘预算；禁止占用已有任务的 GPU。
`--inference-output-dir` 将新 Raw/EMA 导出通过符号链接放到独立输出目录，预算按实际
文件系统分别校验；完整恢复 checkpoint 保留在原 output 的 checkpoints 中。
该参数只影响存储位置，不改变训练配方，不覆盖已有权重。
smoke 必须完成 20 个完整 step（160 sample consumptions）、完整 checkpoint 校验、
checkpoint 重载、Raw/EMA 单图推理及 frozen-weight 审计，才能产生 `GPU_SMOKE_PASS.json`。
自然梯度不触发裁剪可以通过 smoke；人工构造大梯度的单测负责验证裁剪行为。

用户已授权 smoke 通过后自动启动长训练。按后续要求，长训每 2000 步保存完整可恢复
checkpoint 与 Raw/EMA inference export：2000、4000、6000、8000、10000、12000，
并额外保留最终 12500 步；step 0 仍保存初始 Raw/EMA。DEV plan 使用同一组导出步数。
完整训练保存预算约 315611482685 bytes（约 316 GB）；保留 smoke 后完整链路约需 352 GB。

正式启动入口为同一个脚本的 `--mode train`，由授权的连续执行脚本传入 `--confirm-long-run` 和
同一 Git SHA/配置的 `--smoke-proof`，使用独立 train output。只有 GPU smoke 验收通过
才执行后续长训练；启动授权已具备，不需再次确认。
实际运行时必须保持工作树 clean，恢复要求同一配置、输出目录及 Git SHA。

最新资源检查发现四张 A800 正在运行旧模型 step15000 DEV128 评测，smoke 的只读 admission 拒绝新任务。
无 GPU 训练、实模型 Raw/EMA 单图生成或新 DEV 质量结果。
最新检查时 `/data` 仅余约 66 GB；
需要满足空间预算，代码不会自动删除旧数据或 checkpoint。
本次未预设联合配方会改善质量。

## 修改入口

- 训练：`scripts/train/train_full_dense_dimo.py`、`run_dimo_v2.py`。
- 配方/预算：`src/dimo/recipe.py`、`runtime.py`、`mixed_experiment.py`、`mixed_storage.py`。
- 角色/冻结/EMA：`dense_roles.py`、`roles.py`、`ema.py`。
- 更新过程：`step.py`、`fresh_batch.py`、`gradient_stats.py`、`auxiliary.py`、`conditioning.py`、`rng.py`。
- 事务持久化：`full_checkpoint.py`、`dense_distributed_checkpoint.py`、`resume_comparison.py`；复用现有 `transaction.py`。
- 推理/DEV：`evaluation.py`、`dev_protocol.py`、`native_evaluation.py`，以及 eval 三个脚本。
- CPU 验证：`synthetic.py`、`dimo_v2_cpu_smoke.yaml`、`tests/dimo/test_v2_*.py`。
