# Region-DiMO

Region-DiMO 将 DiMO 的 one-step distillation 方法应用到 explicit-region EditMGT。模型输入包含 source image、editing instruction 和 edit mask，生成只发生在 ROI 内。

## 1. 原始 DiMO

参考方法：*Di[M]O: Distilling Masked Diffusion Models into One-step Generator*（ICCV 2025）。

- GitHub：<https://github.com/yuanzhi-zhu/DiMO>
- 参考 commit：`24613741ec9ca730273a6ebc822327878d5a086b`

DiMO 使用三个模型角色：

- teacher model：预训练 masked generator，参数冻结；
- student / generator：从随机 token 与 MASK 的混合状态生成样本；
- auxiliary / fake model：近似 student 的输出分布。

Student 先执行一次生成。生成结果经过 pseudo masking 后，同时查询 teacher 和 auxiliary。两者的分布差异形成 student 的更新方向。Auxiliary 再用 student sample 更新，最后更新 student EMA。

## 2. 迁移到 EditMGT

Region-DiMO 将整图生成改为 ROI 编辑：

| 项目 | Original DiMO / Meissonic | Region-DiMO |
|---|---|---|
| Teacher | pretrained masked generator | selected Explicit-region SFT checkpoint |
| 模型条件 | text | source image + instruction + edit mask |
| 初始化区域 | 全图 | ROI 内 |
| ROI 外 token | 同样参与生成 | 始终复制 source |
| Mask ratio | 相对全图 token 数 | 相对每个样本的 ROI token 数 |
| Student sampling | 全图采样 | 只采样 ROI |
| Pseudo mask | 全图位置 | ROI 内位置 |
| Embedding noise | 整个 target embedding | 只作用于 target ROI |
| Loss normalization | masked token 聚合 | 每个样本按 ROI masked count 归一化 |
| Auxiliary batch | 可使用另一 batch | 与 student 使用同一 batch |
| 模型存储 | 三份完整模型 | 一个 backbone + 三组 role parameters |

ROI 外 hard lock 在初始化、student sample、teacher query、auxiliary query 和输出阶段都生效。Reference branch 不注入 target region embedding，也不加入 embedding noise。

## 3. Model roles

三个角色共享 frozen EditMGT backbone：

```text
shared backbone
├── teacher LoRA + teacher region embedding
├── student LoRA + student region embedding
└── auxiliary LoRA + auxiliary region embedding
```

Teacher 的 LoRA 和 region embedding 来自选中的 SFT checkpoint，参数冻结并使用 eval mode。Student 和 auxiliary 从 teacher 初始化，之后分别更新自己的 LoRA 与 region embedding。

Student 与 auxiliary 使用独立 optimizer。两者的 LoRA dropout 为 `0.0`。Role 切换由 `src/dimo/roles.py` 管理，`edit_region_embedding_override` 为当前 role 提供对应的 region vector。

## 4. One training step

一个完整步骤使用同一批 D200K 样本：

1. VQ 编码 source，得到 source token grid 和 ROI token mask。
2. 在 ROI 内按 `r_init` 选择 MASK，其余 ROI 位置填入随机 codebook token；ROI 外复制 source。
3. Student 接收初始化 token、source reference、instruction 和完整 ROI，执行一次 forward。
4. 从 student logits 采样 ROI token，ROI 外再次覆盖为 source。
5. 在 student sample 的 ROI 内生成 teacher-query pseudo mask。
6. Teacher 与 auxiliary 查询同一 pseudo state，计算 distribution gradient `g`。
7. 使用 surrogate loss 更新 student。
8. 生成独立的 auxiliary pseudo mask，更新 auxiliary，然后更新 student EMA。

Student sample 在步骤 4 后 detach。离散采样不使用 straight-through gradient。Teacher query 和 auxiliary query 使用 FP32 probability algebra。

每个完成步骤才增加 `committed_dimo_step`。Checkpoint 保存 student、auxiliary、EMA、两个 optimizer/scheduler、epoch、D200K cursor、RNG stream identity 和 teacher identity。

## 5. Loss

设 teacher 与 auxiliary 在 vocabulary 上的概率分别为 `p_teacher` 和 `p_aux`。

FKL 使用：

```text
g = p_aux - p_teacher
```

Student logits 记为 `z`，surrogate loss 为：

```text
L_surrogate = 1/2 || z - sg(z - g) ||²
```

`sg` 表示 stop-gradient。该形式在反向传播时将 `g` 作为 student logits 的梯度方向。Loss 先在 vocabulary 维求和，再按每个样本的 pseudo-masked ROI token 数归一化。

实现还支持 RKL 与 Jeffreys：

```text
g_RKL = p_aux * (log p_aux - log p_teacher - D_RKL)
g_Jeffreys = beta * g_FKL + (1 - beta) * g_RKL
```

Auxiliary 默认在 pseudo-masked ROI 位置使用 hard-token CE，也可加入 soft target term。

## 6. Embedding perturbation

Student target embedding 可加入确定性 Gaussian noise：

```text
h_noisy = sigma * noise + sqrt(1 - sigma²) * h
```

混合只作用于 target ROI。Noise 由训练循环按 sample UID、epoch、step 和用途生成；Transformer 内部不调用随机 API。`sigma=0` 时直接返回原 tensor。

## 7. One-step inference

```text
source + instruction + mask
→ VQ source tokens
→ ROI MASK / random-token initialization
→ one student forward
→ sample ROI
→ copy source outside ROI
→ VQ decode
```

推理入口为 `scripts/eval/sample_dimo_editing.py`。`--weights student` 使用 raw student，`--weights ema` 使用 EMA。Temperature、top-k、top-p、CFG、初始 mask ratio 和 embedding-noise sigma 由命令行指定。

## 8. Teacher handoff

Teacher 由 DEV selection 得到的 SFT checkpoint 注册：

```text
scripts/eval/register_selected_checkpoint.py
→ SELECTED_CHECKPOINT.json
→ scripts/dimo/register_teacher.py
→ dimo_teacher_manifest.json
→ DIMO_TEACHER_CHECKPOINT
```

```bash
uv run python scripts/dimo/register_teacher.py \
  --selected-checkpoint "$CANDIDATE_CHECKPOINT" \
  --selected-checkpoint-record "$ASSET_ROOT/artifacts/SELECTED_CHECKPOINT.json" \
  --formal-assets "$ASSET_ROOT/artifacts/formal_assets.json" \
  --output-dir "$ASSET_ROOT/models/dimo_teacher"

export DIMO_TEACHER_CHECKPOINT="$ASSET_ROOT/models/dimo_teacher"
```

Teacher bundle 包含 LoRA、region embedding、trainable config、SFT fingerprint 和 `dimo_teacher_manifest.json`。注册脚本核对 checkpoint record、文件 SHA256、Git commit 和 EditMGT model revision，再以独立目录保存 bundle。

## 9. 代码入口

| 文件 | 作用 |
|---|---|
| `scripts/train/train_dimo_editing.py` | 训练入口 |
| `scripts/eval/sample_dimo_editing.py` | one-step inference |
| `src/dimo/roles.py` | role adapters 与 region vectors |
| `src/dimo/step.py` | 完整 student / auxiliary / EMA update |
| `src/dimo/divergence.py` | distribution gradient |
| `src/dimo/surrogate.py` | student surrogate loss |
| `src/dimo/checkpoint.py` | checkpoint 与 resume |
