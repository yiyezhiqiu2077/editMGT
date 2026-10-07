# E3-only D200K 5-epoch：准备与集群交接

本轮先完成本地可执行验证；本机没有 frozen D200K，不重建、下载或伪造该数据集。
本地 Mini/Tiny 验证不能替代真实 D200K re-attestation 或 8-GPU smoke。

## 冻结训练预算

继续使用 `configs/train/cluster_8g_e3.yaml`，不复制训练 recipe：

- 200000 frozen rows，8 GPU，batch/GPU=1，grad accumulation=4，global batch=32。
- 5 epochs，6250 steps/epoch，31250 successful optimizer updates，1000000 exposures。
- LR=3e-5，warmup=200（全程仅一次），constant_with_warmup。
- 每 3125 步 checkpoint 和 fixed DEV diagnostic，共 10 个 checkpoint。
- core E3 数学保持不变；训练/selection gate 分离，selection 与 formal DiMO gate 仍关闭。

epoch sampler 沿用 `fixed200k-hash-sort-v1`，每 epoch 重新置换且不补齐、不重复。
dataset 通过既有 `set_epoch` 改变 deterministic geometry；原 UID/index corruption seed contract 保持不变，
没有引入另一套 corruption schedule 或 sampling 数学。
每步审计全局 UID 序列，并记录 optimizer/scheduler updates。
每 epoch boundary 记录完整 epoch train CE mean、LR、参数/LoRA update/region norm、消费与 unique row 数。
resume 保存 epoch 内 cursor、CE 累积值、sampler state、optimizer、scheduler、各 rank RNG。

## 本地检验

```bash
uv run pytest -q
uv run python -m compileall -q src scripts tests
find scripts -name '*.sh' -print0 | xargs -0 -n1 bash -n
git diff --check
bash scripts/train/run_cluster_e3.sh --print-command
```

真实模型跨 epoch replay 的最小检验配置为：
`configs/dev/e3_cross_epoch_fresh4.yaml`、`e3_cross_epoch_first2.yaml`、`e3_cross_epoch_resume4.yaml`。
使用既有 Tiny-64，2 epochs × 2 optimizer steps；fresh4 对比 first2（epoch0 boundary）+ resume4。
它验证同一 production trainer 的数据序列、geometry/corruption seeds、权重、optimizer、scheduler 和 rank RNG，
不用于报告 editing quality，也不是 D200K smoke。

## 已知集群路径

代码仓库：`/opt/tiger/tanyue/editMGT`。
现有运行根目录：`/mnt/bn/strategy-mllm-train/user/tanyue/experiments/lumina/20261003T065800Z_editmgt_d200k_v2`。
历史记录：`docs/RUN_20261003_D200K_V2.md`（集群仓库中）。
还需从现有实验记录确认 asset root、released model snapshot 与已有 literal `formal_env.sh`。
没有这些记录时停止；不猜 canonical roots，不生成新 corpus。

## 集群执行顺序（本轮本地不执行）

1. fetch personal 并 detached checkout 本次推送的精确 SHA；工作树须干净。
2. 设置 `EDITMGT_RUN_ROOT` 为上述现有目录，以及实际的 `EDITMGT_ASSET_ROOT`、`EDITMGT_MODEL_ROOT`。
3. 运行 `bash scripts/cluster/reattest_existing_fixed200k.sh`，此命令不训练。
4. 从现有环境记录导入 canonical roots、DERIVED_ROOT、ASSET_ROOT、EDITMGT_OUTPUT_ROOT。
5. re-attestation 全部 PASS 后，运行现有 `run_fixed_200k_smoke.sh --run`（需 `CONFIRM_FORMAL_RUN=YES`），保留 fresh20 与 fresh10+resume20。
6. smoke PASS 且 FORMAL_READY hash-bound 后，仅用 `run_cluster_e3.sh --run` 启动 E3。
7. 恢复可使用 `run_cluster_e3.sh --run --resume /absolute/checkpoint-path`；不得修改 frozen recipe。

smoke 必须绑定当前 Git SHA、train manifest 与 CORPUS_READY hash；除权重之外，
optimizer、scheduler、sampler cursor 与各 rank RNG 也必须精确恢复。
不能复用旧代码的 smoke PASS 或只比较生成图来签发 FORMAL_READY。

re-attestation 只接受唯一、hash-bound v3 CORPUS_READY 与匹配 formal asset ledger，
要求 current-code production loader 验证、完整 train/DEV hash 不变、32 条 MagicBrush alpha 检查。
优先直接读 pinned parquet mask_img 的 alpha；无 raw 时仅接受旧 READY 已 hash-bound 的可信 `255-alpha` export evidence。
缺证据、旧 schema 不支持、歧义路径、坏 hash、坏 mask 或 loader 失败均 STOP，不修数据。
旧 READY 与 ledger 先备份；所有检查完成后才 atomic replace，失败可字节精确恢复。

## Diagnostic 与后续 selection

`configs/eval/e3_periodic_probe.yaml` 采用 CFG=5，仅为观察设置，不宣称最优。
复用已有 frozen MagicBrush probe128，不重新采样。每 checkpoint 保留七项指标与趋势。
`POSSIBLE_OVERFIT` 为观察性提示，不自动选 checkpoint、TEST、teacher 或修改训练。
正常指标波动不停止；既有异常 loss/梯度/数据/分布式质量 gate 仍生效。
最后一个 checkpoint 不默认是 teacher。正式 selection、MagicBrush TEST、formal DiMO 留待下一阶段。
