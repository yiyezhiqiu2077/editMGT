# EditMGT 显式区域编辑 SFT

本仓库基于已发布的 [EditMGT](https://github.com/weichow23/editmgt)，提供 explicit-mask / provided-region 图像编辑 SFT 的可复现实验工程。目标是让一台空服务器从 pinned public assets 自动得到 exact D200K、机器 READY 门禁、8-GPU smoke 和正式 E1–E4 入口。

代码与文档完成不代表实验完成；除非服务器真实生成 `CORPUS_READY.json` 与 `FORMAL_READY.json`，D200K、8-GPU smoke、LR probes、E1–E4 和 final TEST 都仍是 NOT RUN。

## 当前状态

| 项目 | 状态 |
|---|---|
| Formal Pipeline v3 工程 | 已实现，待完整资产/集群实跑 |
| Public asset immutable manifest | 已提交 |
| CrispEdit/ScaleEdit schema 与 mask polarity | pinned metadata + 真实 shard 样本已核验 |
| 完整 public assets 下载 | NOT RUN |
| exact D200K 构建 | NOT RUN |
| 8-GPU smoke / resume | NOT RUN |
| LR probes / E1–E4 / TEST | NOT RUN |

## 快速入口

```bash
git clone git@github.com:yiyezhiqiu2077/editMGT.git
cd editMGT
git checkout <formal-sha>

export ASSET_ROOT=/large_disk/editmgt_assets
export HF_HOME=/large_disk/editmgt_hf_cache   # 可选

bash scripts/setup/run_formal_prepare.sh --dry-run
bash scripts/setup/run_formal_prepare.sh --run
```

prepare 会自动安装 locked 环境，下载 EditMGT、NLLB、四套训练数据、MagicBrush DEV/TEST 与评测模型，验证 schema，构建/翻译/去重 D200K，完成内容、几何和 VQ QA，并生成 `$ASSET_ROOT/artifacts/formal_assets.json` 与 `CORPUS_READY.json`。运行中不要求手填 mapping 或人工审批。

## 8-GPU 与正式实验

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CONFIRM_FORMAL_RUN=YES

bash scripts/cluster/run_8g_smoke.sh --run
bash scripts/cluster/run_lr_probes.sh --run
bash scripts/cluster/run_e1_e4.sh --run
```

正式 launcher 验证 `formal_assets.json`、`CORPUS_READY.json`、`FORMAL_READY.json`、Git SHA 和 selection config。当前 selection 阈值故意未冻结，因此 LR/E1–E4 会报 `SELECTION_RULE_NOT_PREREGISTERED`；不要根据已看结果临时补阈值。

完整命令见 [实验 runbook](docs/EXPERIMENTS.md)。

## D200K

固定语料恰好 200,000 条：

- MagicBrush official train：全部 eligible 记录；
- CrispEdit-labeling-39k：最多 39,000；
- ScaleEdit-labeling-25k：最多 25,000；
- Inter-Edit：仅 `better_data=true`，补齐余额。

语料使用内容寻址 locator、确定性 selection/dedup/translation backfill、无放回 hash-sort epoch permutation 和 rank-stride DDP。MagicBrush DEV 与三套 aux validation 先冻结，TEST 永不参与 corpus 或 checkpoint selection。详细定义见 [数据契约](docs/DATA_CONTRACT.md)。

## 实验定义

| 实验 | Corruption / region condition | LoRA scope |
|---|---|---|
| E0-official | released model / official timestep | released weights |
| E0-region | released model / ROI-relative timestep | released weights |
| E1 | full target / OFF | both |
| E2 | ROI hard lock / OFF | both |
| E3 | ROI hard lock / ON | both |
| E4 | ROI hard lock / ON | reference only |

E1–E4 共享 exact D200K、permutation、seed、训练预算和 validation protocol。本轮 pipeline 工程不改变训练语义。

## 项目结构

```text
configs/                 formal assets、schema、data/train/eval contracts
src/explicit_region/     canonical、geometry、selection、sampler、训练与门禁
scripts/setup/           空服务器下载与 00–12 prepare
scripts/data/            D200K、翻译、QA、CORPUS_READY
scripts/train/           训练、resume 与 smoke verifier
scripts/eval/            validation、正式 TEST 与指标
scripts/cluster/         8-GPU 编排入口
scripts/tools/           结果打包
tests/                   行为契约
docs/                    中文 runbook 与设计文档
```

正式训练入口是 `scripts/train/train_explicit_region.py`，不是保留的上游 `train/train.py`。详见 [代码架构](docs/ARCHITECTURE.md)。

## Final TEST 与结果包

selection rule 和唯一 checkpoint 冻结后：

```bash
export CANDIDATE_CHECKPOINT=$ASSET_ROOT/experiments/<run>/checkpoint-6250
uv run python scripts/eval/register_selected_checkpoint.py \
  --checkpoint "$CANDIDATE_CHECKPOINT" \
  --formal-assets "$ASSET_ROOT/artifacts/formal_assets.json" \
  --output "$ASSET_ROOT/artifacts/SELECTED_CHECKPOINT.json"
bash scripts/eval/run_magicbrush_test.sh --run
```

TEST launcher 硬验证 pinned test identity、535 sessions / 1053 turns、checkpoint、Git 与 provenance。结果可用 `scripts/tools/package_formal_results.py` 打包；默认排除数据、权重、cache 与全分辨率图像。

## 验证

```bash
uv run pytest -q
uv run python -m compileall src scripts tests
bash -n scripts/setup/run_formal_prepare.sh scripts/cluster/*.sh scripts/train/*.sh scripts/eval/*.sh
git diff --check
```

## 文档

- [实验流程](docs/EXPERIMENTS.md)
- [环境与资产](docs/ENVIRONMENT.md)
- [数据契约](docs/DATA_CONTRACT.md)
- [代码架构](docs/ARCHITECTURE.md)

## 上游与许可证

本仓库衍生自官方 EditMGT 项目，是独立实验工程，不代表原论文作者。上游 [CC-BY-4.0 许可证](LICENSE)保持不变。
