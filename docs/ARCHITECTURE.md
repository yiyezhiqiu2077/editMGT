# 代码结构

仓库在 EditMGT 模型栈上增加 explicit-region 数据处理、SFT、评测和 Region-DiMO。主要训练入口位于 `scripts/train/`。

## 目录

```text
configs/                 数据、训练、评测和 DiMO 配置
src/explicit_region/     D200K reader、geometry、mask 与 SFT 组件
src/dimo/                Region-DiMO 训练和 one-step generation
scripts/setup/           环境与资产准备
scripts/data/            canonical data、翻译和 D200K 构建
scripts/train/           Explicit-region SFT 与 Region-DiMO
scripts/eval/            validation、TEST 与 one-step inference
scripts/dimo/            SFT checkpoint 到 teacher bundle 的转换
scripts/cluster/         多 GPU 运行入口
tests/dimo/              Region-DiMO 单元测试
tests/integration/       SFT、D200K 与 DiMO 接口测试
docs/                    实验和方法说明
```

## Explicit-region

`src/explicit_region/` 提供 SFT 和 Region-DiMO 共用的数据层。

| 模块 | 作用 |
|---|---|
| `canonical.py` | canonical record、locator、内容 hash 与图像读取 |
| `fixed_corpus.py` | D200K selection、dedup、split 与 backfill |
| `fixed_dataset.py` | 按固定 manifest 读取四个数据集 |
| `geometry.py` | source / target / mask 共享 resize、crop 与 fallback |
| `masks.py` | pixel mask 到 VQ token mask |
| `corruption.py` | full-target 与 ROI hard-lock corruption |
| `conditioning.py` | region embedding 注入 |
| `epoch_sampler.py` | hash-sort permutation、DDP rank stride 与 resume cursor |
| `checkpoint.py` | LoRA、region embedding 和 optimizer state |
| `metrics.py` | masked pixel、LPIPS、DINO 与 CLIP 指标 |

`scripts/train/train_explicit_region.py` 组合这些模块。`src/transformer.py` 接收 target/reference tokens、ROI mask、region embedding 和 LoRA scope。

## Region-DiMO

| 模块 | 作用 |
|---|---|
| `roles.py` | 在共享 backbone 上切换 teacher、student、auxiliary adapters |
| `initialization.py` | ROI 内 MASK / random-token 初始化与采样 |
| `forward_process.py` | ROI pseudo masking 与 embedding perturbation |
| `divergence.py` | FKL、RKL 与 Jeffreys distribution gradient |
| `surrogate.py` | student surrogate loss |
| `auxiliary.py` | auxiliary hard / soft target update |
| `step.py` | student、auxiliary 与 EMA 的完整训练步骤 |
| `ema.py` | student EMA 保存与加载 |
| `checkpoint.py` | DiMO optimizer、role state、cursor 与 resume |
| `one_step.py` | 单次 student forward 和 ROI sampling |

Region-DiMO 直接复用 `FixedCorpusDataset`、mask conversion、geometry 和 released component loader，不维护单独的数据 reader。

## 数据流

```text
raw datasets
→ canonical records
→ fixed D200K
→ Explicit-region SFT
→ selected checkpoint
→ teacher bundle
→ Region-DiMO
→ one-step editing
```

## 模型训练

Explicit-region SFT 冻结 EditMGT backbone、text encoders 和 VQ-VAE，只更新 LoRA 与 `edit_region_embedding`。loss 在每个样本的 selected ROI tokens 上归一化，再对 batch 求平均。

target branch 接收 corruption 后的 token 和 region condition；reference branch 接收干净 source token。`lora_scope` 控制 LoRA 应用于 target、reference 或两者。

## Region-DiMO 关系

teacher、student 和 auxiliary 共享一个 frozen EditMGT backbone，并各自使用 LoRA adapter 与 region embedding。Teacher 参数冻结；student 和 auxiliary 分别由独立 optimizer 更新。

每个训练步骤先更新 student，再更新 auxiliary，最后更新 student EMA。外部区域始终复制 source tokens。teacher bundle 由 `scripts/dimo/register_teacher.py` 从选中的 SFT checkpoint 生成。
