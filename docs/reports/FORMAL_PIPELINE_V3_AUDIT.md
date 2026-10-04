# Formal Pipeline v3 改造前审计

审计基线：`6d6bab14582faefa83de7db5c69ff1cf8b0bad55`。

## 已确认缺口

- `EDITMGT_MODEL_ROOT` 与 `TRANSLATOR_MODEL_ROOT` 依赖服务器预存 EditMGT/NLLB。
- MagicBrush、CrispEdit、ScaleEdit、Inter-Edit root 均需人工提前准备。
- CrispEdit/ScaleEdit 只有 schema mapping 模板，edit-type mapping 为空。
- 原文档要求 schema 后手填 YAML、人工检查 500 条翻译、人工查看 montage。
- `CORPUS_READY` 把人工抽样与 montage 当 required files，却没有可验证的 human approval；这不是可靠机器 gate。
- LR/E1–E4 launcher 只验证旧 `CORPUS_READY`，未强制 smoke verifier、formal asset 或 selection-ready。
- MagicBrush TEST 缺少固定 identity、535/1053 contract、checkpoint/Git/provenance 联合门禁。
- 准备流程要求十余个环境变量，没有可恢复的统一下载器和阶段 marker。

## v3 决策

- public assets 统一由 `configs/formal_assets.yaml` 固定，HF revision 必须为完整 commit。
- schema 与 label mapping 提交到仓库；actual/contract mismatch 或未知 label 立即失败。
- translation/montage 抽样改为非阻塞诊断；所有正式 gate 由机器 invariant 决定。
- `CORPUS_READY` 只证明数据，`FORMAL_READY` 证明 8-GPU smoke；formal launcher 同时验证 provenance 和 selection rule。
- 00–12 阶段 marker 负责安全续跑和依赖失效。

本审计只描述代码基线与工程修复范围，不表示资产已完整下载、D200K 已构建或正式实验已运行。

## Pinned schema 核验记录

- CrispEdit commit `dcbd1c952e93e4361ad862b33f3acd1cc74bec5a`：读取 `dataset_manifest.json` 与真实 `shards/add/add_00000.parquet`。确认 embedded `input_img` / `output_img`、`instruction`、`type`、`mask__mask_png` 以及 quality/scene/grounding/mask QC 字段；manifest 记录 38,971 rows、1,844 shards。
- ScaleEdit commit `f97ffb061d4275bcbbff6500572139b6f8df1c4e`：读取 `dataset_manifest.json` 与真实 `shards/expand-20260924-00000.parquet`。确认 embedded `source_image` / `edited_image`、`final_instruction`、`final_task`、`mask__mask_png` 及四组 QC 字段；manifest 记录 25,664 rows、1,073 shards。
- 两个 sampled shard 的 mask 像素均为 `{0,255}`。白区 source/target 平均变化显著大于黑区（CrispEdit 样本约 57.34 vs 11.54；ScaleEdit 样本约 166.58 vs 2.21），与字段/数据说明一致，因此 contract 固定 `white_is_edit_region`。
- Inter-Edit commit `b319d9fdb45cc670ec263fe4aeaa962f1623d5dc` 的 `manifest.json` 记录 1,099,964 samples / 610,186 unique sources，README/manifest 字段与提交的 JSONL+TAR contract 一致。
- MagicBrush TEST mirror commit `8c8eb2cfeac96f635ed83512c60c3502b06e00c2` 的 annotation metadata 实测为 535 sessions / 1053 turns；正式 prepare 仍会逐文件复核。

本地开发阶段没有下载完整百 GB/TB 级 snapshot；全量每-shard schema 与每-row 内容检查由 04/11 阶段执行。
