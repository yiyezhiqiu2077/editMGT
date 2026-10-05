# 环境配置

## 软件

- Linux
- Python `>=3.10,<3.11`
- CUDA 12.1 compatible NVIDIA driver
- PyTorch 2.1.2 / torchvision 0.16.2
- Diffusers 0.32.1
- Transformers 4.47.1
- `uv`

依赖版本由 `uv.lock` 固定。

## 安装

```bash
git clone git@github.com:yiyezhiqiu2077/editMGT.git
cd editMGT

uv sync --frozen --group dev --group translation --group metrics
bash scripts/setup/probe_runtime.sh
```

## GPU

E1–E4 使用单节点 8 GPU，推荐 8×A100。单卡可运行数据链路、单步 SFT、Region-DiMO smoke 和 one-step inference。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash scripts/cluster/probe_cluster.sh
```

## 存储与网络

```bash
export ASSET_ROOT=/large_disk/editmgt_assets
export HF_HOME=/large_disk/editmgt_hf_cache   # 可选
export HF_ENDPOINT=https://huggingface.co     # 可选镜像
```

`ASSET_ROOT` 保存模型、数据、derived manifests 和实验输出。`HF_HOME` 保存 Hugging Face cache。Inter-Edit 为 TB 级数据，建议把两个目录放在持久化大磁盘。

需要访问权限时，通过作业环境设置 `HF_TOKEN`，不要写入仓库文件。

## 资产

`scripts/setup/run_formal_prepare.sh` 准备以下内容：

- EditMGT released model；
- `facebook/nllb-200-distilled-1.3B`；
- MagicBrush train / DEV / TEST；
- CrispEdit-labeling-39k；
- ScaleEdit-labeling-25k；
- Inter-Edit-Train；
- DINOv2、CLIP 和 LPIPS/AlexNet。

具体版本由 `configs/formal_assets.yaml` 固定。下载支持 Hugging Face cache 和断点续传。

## 常用命令

只检查资产计划和远端 metadata：

```bash
bash scripts/setup/run_formal_prepare.sh --dry-run
```

下载资产并构建 D200K：

```bash
bash scripts/setup/run_formal_prepare.sh --run
source "$ASSET_ROOT/artifacts/formal_env.sh"
```

`formal_env.sh` 提供模型、数据、derived 和 output 路径，后续训练脚本可直接使用这些环境变量。
