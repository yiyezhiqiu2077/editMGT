# D200K 数据契约

## Canonical record

每条记录绑定 dataset/revision、稳定 `sample_uid`、source/target/region locator 及 SHA256、原始与英文 instruction、group identity、edit type、mask semantics、region fraction 和 translation identity。支持 file、tar member、parquet cell 与明确 bbox locator；路径不得逃逸 dataset root。

## Schema 与 mask

四套正式 contract 位于 `configs/data/schema/`。运行时 actual schema 与 committed contract 不一致即 `SCHEMA_CONTRACT_MISMATCH`。CrispEdit 与 ScaleEdit 已使用 pinned revision 的真实 parquet schema/样本核对；其二值 mask 白色/非零区域表示 edit region。未知 edit label 不做隐式兜底，而是 `UNMAPPED_EDIT_TYPE`。

## Selection

总量严格为 200,000：MagicBrush 使用全部 eligible train；CrispEdit cap 39,000；ScaleEdit cap 25,000；仅 `better_data=true` 的 Inter-Edit 补齐。selection key 绑定 seed、dataset、revision 和 sample UID。

MagicBrush official DEV 先冻结；每个其他数据集冻结 group-disjoint aux-128。validation 对 source SHA 与 `(dataset, group_id)` 拥有优先权。

## 去重与 backfill

同数据集完全重复由确定性顺序保留一条；跨数据集 source duplicate 按固定 dataset priority 解决。Inter-Edit 在 edit type × region-size strata 内选取，reserve 顺序固定。任何 QA rejection 从同 stratum reserve 确定性回填，不足时按提交的 fallback 顺序处理。

## 翻译

只翻译被选择的中文 instruction。NLLB contract 固定 `zho_Hans -> eng_Latn`、`do_sample=false`、`num_beams=1`、`max_new_tokens=128`；cache key 绑定模型 commit、语言与 decoding。

hard QA 包含 `empty_output`、`copy_output`、`han_remaining`、`length_ratio_outlier`、`digit_mismatch`。失败样本 reject/backfill；最终 `instruction_en` 不得含 Han。500 条人工抽样文件只是诊断，不参与 READY。

## 几何

source、target 和 region 在 mask 坐标系预对齐后，共用 deterministic resize/crop/flip。最多八次随机尝试，region retention 至少 0.75；失败时使用 contain/center-pad fallback。最终 region 必须非空且三者尺寸一致。

## Machine QA

`validate_fixed200k_machine.py` 穷举 200K 行并检查 locator、内容 hash、解码、对齐、mask、post-geometry、翻译、Inter-Edit eligibility、train/DEV 泄漏和 TEST 排除。`audit_fixed200k.py` 使用 FP32 VQ 做 finite/containment 审计；NaN/Inf 数必须为零。

montage 与 translation audit 会保留用于排障，但缺少人工签字不会阻塞 READY。

## READY 语义

`CORPUS_READY.json` 包含 Git、formal asset identity、selection/config/manifest hash、数量与 unique-source 统计，以及 schema、asset、translation、dedup、geometry、leakage、exact-200K 和 VQ PASS。文件哈希变化后 verifier 必须失败。

`FORMAL_READY.json` 由 8-GPU uninterrupted/resume smoke 生成，绑定 Git、`formal_assets.json`、`CORPUS_READY.json`、smoke report 与 world size 8。

这两个 marker 都是机器生成的 attestation，不包含 reviewer、human approval 或手工 go/no-go。

## Determinism

manifest index 在 freeze 后连续且不可变。epoch permutation、DDP rank partition、geometry/corruption seed、translation cache、reserve/backfill 与 resume cursor 都由稳定内容 identity 驱动。坏 frozen row 是 integrity error，不能在训练时换样本。
