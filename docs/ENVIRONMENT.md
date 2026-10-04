# 环境与资产

## 软件与 GPU

- Linux；NVIDIA driver 能运行 CUDA 12.1 wheel。
- Python `>=3.10,<3.11`，依赖由 `uv.lock` 锁定。
- 正式拓扑是单节点 8 GPU；launcher 固定 world size 8。
- PyTorch 2.1.2、torchvision 0.16.2、Diffusers 0.32.1、Transformers 4.47.1。

```bash
uv sync --frozen --group dev --group translation --group metrics
bash scripts/setup/probe_runtime.sh
```

## 存储

只需手工设置：

```bash
export ASSET_ROOT=/large_disk/editmgt_assets
export HF_HOME=/large_disk/editmgt_hf_cache   # 可选
export HF_ENDPOINT=https://huggingface.co     # 可选镜像
```

Inter-Edit 发布约为 TB 级，CrispEdit/ScaleEdit 也需要大量空间。`ASSET_ROOT` 和 `HF_HOME` 应位于持久化大磁盘，不要指向 Git worktree 或临时盘。下载器使用 Hugging Face cache/断点续传；已验证的 `_SUCCESS` 与 `asset_identity.json` 会阻止重复下载。

## 资产身份

`configs/formal_assets.yaml` 是唯一 public-asset 清单。HF 资产必须使用 40 位 commit，禁止 `main`、`latest` 或可变 tag。准备脚本会比较 requested/resolved revision，并验证 EditMGT 六个 component、数据结构和 TEST 计数。

正式资产包括 EditMGT、NLLB、MagicBrush TRAIN/DEV/TEST、CrispEdit、ScaleEdit、Inter-Edit、DINO、CLIP 和 LPIPS/AlexNet。评估期间全部以 local-only 方式读取；silent download 属于错误。

MagicBrush TEST 使用公开 pinned mirror 作为自动传输载体，并以 535 sessions / 1053 turns 的官方内容契约硬校验。该 transport 选择与 TEST 隔离规则无关。

## 网络

`HF_ENDPOINT` 只改变传输端点。最终 identity 仍必须解析到 manifest 中相同 commit。中断后重复运行 `run_formal_prepare.sh --run` 即可续传；不要删除未完成 cache。

若资产需要 Hugging Face token，只在作业环境注入 `HF_TOKEN`，不得写入仓库、配置或结果包。

## 自动生成的环境

`run_formal_prepare.sh` 写入 `$ASSET_ROOT/artifacts/formal_env.sh`，其中包含模型、数据、derived、output 和 evaluator 路径。后续 launcher 只要求 `ASSET_ROOT`，`scripts/setup/common.sh` 会自动载入该文件。

执行 GPU workload 另需短期设置：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CONFIRM_FORMAL_RUN=YES
```

不要把确认变量长期写入 shell profile。

## 常见错误

| 错误 | 含义 |
|---|---|
| `ASSET_NOT_PINNED` | manifest 缺少 immutable revision |
| `ASSET_REVISION_MISMATCH` | 远端 resolved commit 与冻结值不同 |
| `EDITMGT_SNAPSHOT_INVALID` | released model component 不完整 |
| `SCHEMA_CONTRACT_MISMATCH` | 实际数据结构与提交的 contract 漂移 |
| `CORPUS_NOT_READY` | D200K 或绑定 artifact 缺失/变更 |
| `FORMAL_PIPELINE_NOT_READY` | assets/corpus/smoke/Git hash 不一致 |
| `SELECTION_RULE_NOT_PREREGISTERED` | selection status 或阈值尚未冻结 |

原始 snapshot 不可修改。出现坏数据时应修复 contract/builder、更新版本并重建 READY，不能直接编辑 raw data 或 frozen manifest。
