# 代码架构

仓库保留 released EditMGT 模型栈，在其上增加 explicit-region 数据、训练、评估与可复现门禁。正式入口是 `scripts/train/train_explicit_region.py`，不是保留的上游 `train/train.py`。

## 目录职责

```text
configs/                 冻结资产、schema、数据、训练和评估协议
src/explicit_region/     可复用的数据/训练/门禁逻辑
scripts/setup/           空服务器下载、阶段 runner、provenance
scripts/data/            canonical、D200K、翻译、QA、READY
scripts/train/           训练、resume、smoke verifier
scripts/eval/            生成、指标、TEST contract
scripts/cluster/         8-GPU 编排入口
scripts/tools/           结果打包等离线工具
tests/                   行为契约
docs/                    runbook 与设计说明
```

依赖方向固定为：

```text
configs -> scripts -> src/explicit_region -> released src modules
              ^              ^
              └──── tests ───┘
```

## 核心模块

| 模块 | 职责 |
|---|---|
| `formal_pipeline.py` | asset manifest、revision、schema、stage marker、FORMAL_READY |
| `canonical.py` | FILE/TAR/PARQUET/bbox locator、content hash、canonical record |
| `fixed_corpus.py` | exact D200K 选择、去重、分层、backfill、CORPUS_READY |
| `fixed_dataset.py` | frozen index 读取；坏行直接终止，不运行时替换 |
| `geometry.py` | source/target/mask 共用几何与 deterministic fallback |
| `language.py` | Han 检测、NLLB cache identity、翻译 QA |
| `epoch_sampler.py` | hash-sort permutation、rank-stride DDP、resume cursor |
| `corruption.py` / `conditioning.py` | ROI hard lock 与 region embedding |
| `checkpoint.py` | LoRA、region embedding、optimizer/resume fingerprint |
| `metrics.py` | masked pixel/LPIPS、DINO/CLIP、finite aggregation |

## Formal prepare

`scripts/setup/run_formal_prepare.sh` 编排 00–12 阶段。下载器仅解析 `configs/formal_assets.yaml`，schema builder 仅解析 `configs/data/schema/*.yaml`，不存在运行时手填 mapping。每个阶段 marker 绑定 Git、全部 contract config、上游 marker hash 和 resolved revisions。

数据流为：

```text
pinned snapshots
  -> schema contract
  -> four canonical pools
  -> frozen DEV/aux
  -> deterministic selection
  -> NLLB QA/backfill
  -> exact train_200k.jsonl
  -> exhaustive content/geometry/VQ QA
  -> CORPUS_READY
  -> 8-GPU smoke/resume
  -> FORMAL_READY
  -> formal launchers
```

`formal_assets.json` 记录 Git、public asset identity、schema hash、D200K hash 和 CORPUS_READY hash。`CORPUS_READY` 是唯一数据机器 gate；`FORMAL_READY` 再绑定 smoke verifier。正式 launcher 必须同时验证三者。

## 训练语义边界

基础 Transformer、text encoders 与 VQ-VAE 冻结；可训练参数为 LoRA 和 `edit_region_embedding`。source/target/region 共用几何；VQ 固定 FP32；loss 先按样本 selected tokens 归一化再跨 batch 平均。E1–E4 的 corruption、region condition、LoRA scope、batch、累积、LR、scheduler 和 epoch 都由现有配置定义，pipeline v3 不改变它们。

正式 sampler 是 `fixed200k-hash-sort-v1`：无放回 hash-sort permutation，rank `r` 读取 `permutation[r::world_size]`，不 padding、不 drop、不运行时 replacement。

## 评估边界

DEV 用于 validation/selection；三套 aux 独立报告。TEST 只有在 selection config 与唯一 checkpoint identity 冻结后才能启动。TEST launcher 不接受 DEV 替代路径，并硬验证 535 sessions / 1053 turns。

结果包只收集小型配置、provenance、metrics、summary、READY 和 audit report，主动排除数据、权重、cache、token 与全分辨率图像。
