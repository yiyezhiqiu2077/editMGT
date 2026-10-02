# Environment and assets

本文给出本地有限 GPU 服务器和 8-GPU 正式集群共用的环境、模型与数据契约。命令默认从仓库根目录执行。

## Hardware and software

- Linux，NVIDIA driver 能支持 CUDA 12.1 wheels。
- Python `>=3.10,<3.11`。
- 当前真实 1024 one-step 验证使用 A100-PCIE-40GB；这是一条验证记录，不表示代码只支持 A100。
- 正式 topology 是单节点 8 张可见 GPU；launcher 固定使用 GPU `0,1,2,3,4,5,6,7`，但 D200K-v2 8-GPU smoke 尚未实际运行。
- 足够容纳四个 raw datasets、一个 NLLB snapshot、D200K 衍生文件、checkpoint 和验证图像的本地存储。

Python 依赖由 [pyproject.toml](../pyproject.toml) 和 [uv.lock](../uv.lock) 冻结。核心版本包括 PyTorch 2.1.2/cu121、torchvision 0.16.2、diffusers 0.32.1、transformers 4.47.1 和 accelerate 1.0.1。

## Bootstrap

```bash
git clone git@github.com:yiyezhiqiu2077/editMGT.git
cd editMGT
export EDITMGT_WORKTREE="$PWD"
bash scripts/setup/bootstrap_uv.sh
uv sync --frozen --group dev --group translation --group metrics
bash scripts/setup/probe_runtime.sh
```

`bootstrap_uv.sh` 在 `uv` 不存在时会从 Astral 安装脚本联网安装它，然后执行 `uv sync --frozen`。隔离集群若禁止联网，应提前部署 `uv` 和 wheel cache；不要删掉 `--frozen` 来绕过 lockfile。

所有 shell launcher 都通过 `scripts/setup/common.sh` 检查实际 Git root。未设置 `EDITMGT_WORKTREE` 时默认采用当前 Git root；显式设置后，它必须与实际 root 一致，否则会以 `WRONG_WORKTREE` 退出。

## Storage layout

推荐将代码、只读 raw assets、衍生语料和实验输出分开：

```text
/workspace/EditMGT-explicit-region-sft-v1/   # Git worktree
/models/EditMGT/IMMUTABLE_SNAPSHOT/          # released EditMGT
/models/NLLB/IMMUTABLE_SNAPSHOT/             # nllb-200-distilled-1.3B
/datasets/MagicBrush/train/
/datasets/MagicBrush/dev/
/datasets/MagicBrush/test/                   # final-test-only
/datasets/CrispEdit-labeling-39k/
/datasets/ScaleEdit-labeling-25k/
/datasets/Inter-Edit-Train/
/derived/editmgt-explicit-region/            # canonical pools / fixed200k
/outputs/editmgt-explicit-region/            # logs / checkpoints / predictions
/cache/huggingface/                          # optional Hugging Face cache
```

raw dataset 和 model snapshot 应只读。`DERIVED_ROOT` 与 `EDITMGT_OUTPUT_ROOT` 必须是可写、空间充足且不会被临时清理的目录。

## Required environment variables

将实际绝对路径和真实 immutable revisions 填入当前 shell 或集群作业环境：

```bash
export EDITMGT_WORKTREE=/workspace/EditMGT-explicit-region-sft-v1
export EDITMGT_MODEL_ROOT=/models/EditMGT/IMMUTABLE_SNAPSHOT
export TRANSLATOR_MODEL_ROOT=/models/NLLB/IMMUTABLE_SNAPSHOT
export EDITMGT_OUTPUT_ROOT=/outputs/editmgt-explicit-region
export DERIVED_ROOT=/derived/editmgt-explicit-region
export HF_HOME=/cache/huggingface

export MAGICBRUSH_ROOT=/datasets/MagicBrush/train
export MAGICBRUSH_DEV_ROOT=/datasets/MagicBrush/dev
export CRISPEDIT_ROOT=/datasets/CrispEdit-labeling-39k
export SCALEEDIT_ROOT=/datasets/ScaleEdit-labeling-25k
export INTEREDIT_ROOT=/datasets/Inter-Edit-Train

export MAGICBRUSH_REVISION=IMMUTABLE_DATASET_ID
export CRISPEDIT_REVISION=IMMUTABLE_DATASET_ID
export SCALEEDIT_REVISION=IMMUTABLE_DATASET_ID
export INTEREDIT_REVISION=IMMUTABLE_DATASET_ID
export TRANSLATOR_REVISION=IMMUTABLE_MODEL_ID

export CRISPEDIT_SCHEMA_MAPPING="$DERIVED_ROOT/schema_mappings/crispedit.yaml"
export SCALEEDIT_SCHEMA_MAPPING="$DERIVED_ROOT/schema_mappings/scaleedit.yaml"
```

revision 必须能唯一标识已审计资产，例如版本库 commit、snapshot hash 或组织内不可变发布 ID；不要填 `main`、`latest` 或临时日期标签。

仅在真正执行受保护操作的那个 shell 中设置确认变量：

```bash
export CONFIRM_CORPUS_BUILD=YES   # 允许构建/覆盖 D200K 衍生目录中的产物
export CONFIRM_FORMAL_RUN=YES     # 允许 8-GPU smoke/formal workload
```

不要长期写入 shell profile。先用 `--print-command` 审核将执行的命令，再临时 export。

## EditMGT model assets

`EDITMGT_MODEL_ROOT` 必须直接指向完整 snapshot，并至少包含以下 subfolders：

```text
editmgt/
text_encoder/
tokenizer/
llm_encoder/
vqvae/
scheduler/
```

训练入口验证这些名称，按本地文件加载组件，并将 snapshot identity 写入 provenance。VQ-VAE 在训练和审计中固定使用 FP32。

为兼容现有资产审计脚本，在 repo 内建立受控软链接：

```bash
bash scripts/setup_local_assets.sh model "$EDITMGT_MODEL_ROOT"
bash scripts/setup_local_assets.sh magicbrush "$MAGICBRUSH_ROOT"
bash scripts/setup_local_assets.sh magicbrush-test /datasets/MagicBrush/test
bash scripts/setup_local_assets.sh interedit "$INTEREDIT_ROOT"
bash scripts/setup_local_assets.sh outputs "$EDITMGT_OUTPUT_ROOT"

uv run python scripts/setup/audit_assets.py
```

脚本拒绝覆盖已有真实目录或指向不同位置的 symlink。CrispEdit、ScaleEdit 和 NLLB 由环境变量直接引用，不通过 `setup_local_assets.sh`。

## Dataset contracts

### MagicBrush

train 和 official dev 根目录都必须含 `manifest.jsonl`，其中 locator 指向可读的 source、target 和 region。official test 必须独立保存，只能用于最终评估。

### CrispEdit and ScaleEdit

真实格式必须先由 `audit_raw_dataset_schema.py` 审计，再从模板生成显式 mapping：

- `configs/data/crispedit_schema_mapping.template.yaml`
- `configs/data/scaleedit_schema_mapping.template.yaml`

将模板复制到 `DERIVED_ROOT/schema_mappings/`，根据真实 schema 填写 `format`、glob、字段、storage、mask semantics、dataset revision 和 `schema_audit_status`。同时在 `configs/data/edit_type_mapping.yaml` 中冻结真实 taxonomy 到 canonical edit types 的映射。

模板中的 `REPLACE`、`NOT_RUN` 或空的 CrispEdit/ScaleEdit edit-type mapping 会使正式 corpus preparation 失败，这是预期门禁。

### Inter-Edit

当前脚本期望：

```text
$INTEREDIT_ROOT/metadata/*.jsonl.gz
$INTEREDIT_ROOT/source_shards/*.tar
$INTEREDIT_ROOT/asset_shards/*.tar
```

只允许 `better_data` 进入正式候选池。用以下命令检查根目录：

```bash
bash scripts/cluster/probe_cluster.sh
```

## NLLB translation asset

`TRANSLATOR_MODEL_ROOT` 指向本地 `facebook/nllb-200-distilled-1.3B` snapshot，`TRANSLATOR_REVISION` 必须与它一致。正式设置为 `zho_Hans -> eng_Latn`、greedy、`max_new_tokens=128`；cache key 绑定 backend、revision、语言和 decoding 参数。

安装翻译附加依赖：

```bash
uv sync --frozen --group translation
```

若需完整 LPIPS/SSIM 指标环境，可同时安装 metrics group：

```bash
uv sync --frozen --group metrics
```

## Preflight checklist

在构建语料或启动 GPU job 前运行：

```bash
git status --short --branch
bash scripts/setup/probe_runtime.sh
bash scripts/cluster/status.sh
bash scripts/cluster/probe_cluster.sh
uv run pytest -q
```

然后确认：

- `nvidia-smi` 显示期望的 8 张 GPU，且无占用冲突。
- `git status` 与计划中的 commit/diff 一致；dirty hash 会进入 fingerprint。
- 模型和 raw dataset 均为正确 revision 且可读。
- derived/output 路径可写，磁盘和 inode 充足。
- 不存在从 train 到 validation/test 的 group 或 source SHA 泄漏。
- `configs/eval/formal.yaml` 的正式 checkpoint selection rule 已在看结果前预注册。

## Common failures

| 错误 | 含义 / 处理 |
|---|---|
| `WRONG_WORKTREE` | `EDITMGT_WORKTREE` 与实际 Git root 不同；重新 export |
| `ASSET_NOT_SYMLINK` | `local_assets/` 不是由 setup 脚本建立的 symlink |
| `INTEREDIT_ROOT_NOT_FOUND` | 未挂载真实 Inter-Edit-Train，不能以 Test 替代 |
| `CORPUS_NOT_READY` | marker 缺失、总数不为 200K 或任一绑定 hash 改变 |
| `FROZEN_CORPUS_INTEGRITY_ERROR` | frozen row 的文件、hash、instruction 或 mask 已损坏；修复资产并重建，不要跳过 |
| `RESUME_FINGERPRINT_MISMATCH` | 代码、配置、数据、world size 或 sampler state 与 checkpoint 不一致 |
| CUDA OOM | 不要直接更改 formal batch 契约；先用 smoke 复现并记录，再形成新配置/实验版本 |

## Security and reproducibility notes

- 不要把 API token 写入 repo；当前正式数据/模型路径不需要运行时下载 token。
- 不要改动 raw assets 来“修”某一条样本；修复应发生在 mapping/builder，并产生新的 revision 与 READY marker。
- 不要手工编辑 `train_200k.jsonl`、translation cache 或 READY marker。
- 复制实验时连同 Git commit、dirty patch、`uv.lock`、configs、READY marker 和 run provenance 一起归档。
