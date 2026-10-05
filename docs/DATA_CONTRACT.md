# D200K 数据定义

D200K 是 Explicit-region SFT 使用的固定 200,000 条训练语料。四个来源先转换为统一 canonical record，再进行 selection、去重、翻译和几何检查。

## Canonical record

每行 JSONL 记录一条编辑样本，主要字段包括：

| 字段 | 含义 |
|---|---|
| `sample_uid` | 由内容生成的稳定样本标识 |
| `source_locator` | source image 的 file、tar 或 parquet 位置 |
| `target_locator` | target image 的位置 |
| `region_locator` | edit mask 或 bbox 的位置 |
| `instruction_original` | 原始 instruction |
| `instruction_en` | 英文训练 instruction |
| `dataset_name` | 数据集名称 |
| `dataset_revision` | 数据版本 |
| `group_id` | session / image group，用于 split 隔离 |
| `edit_type_original` | 原始 edit type |
| `edit_type_canonical` | 统一 edit type |
| `mask_semantics` | mask 前景含义 |

记录还保存 source、target 和 region 的 SHA256、region fraction、translation 状态与连续 `manifest_index`。

## 数据来源

| 数据集 | 选择规则 |
|---|---|
| MagicBrush train | 使用全部合格样本 |
| CrispEdit-labeling-39k | 最多 39,000 条 |
| ScaleEdit-labeling-25k | 最多 25,000 条 |
| Inter-Edit-Train | 使用 `better_data=true` 的样本补齐余额 |

总量固定为 200,000。MagicBrush DEV 与其他数据集的 aux validation 在训练集选择前分离，split 之间保持 group-disjoint。

## Mask

白色或非零像素表示 edit region。bbox 数据先转换为同样的二值区域。

训练时将 pixel mask 映射到 VQ token grid。`any_overlap` 模式选择与区域相交的 token，也可通过 coverage threshold 和 token dilation 调整。映射后的 ROI 至少包含一个 token。

## Selection 与去重

选择过程使用固定 seed 和稳定 sample identity，不做 replacement：

1. 先分离 validation group；
2. 删除同数据集完全重复记录；
3. 按固定 dataset priority 处理跨数据集 source duplicate；
4. 在 edit type 与 region-size strata 中选择 Inter-Edit；
5. 被过滤的记录从对应 reserve 顺序 backfill。

相同输入、配置和数据版本会生成相同 manifest。

## Translation

中文 instruction 使用 `facebook/nllb-200-distilled-1.3B`，语言方向为 `zho_Hans → eng_Latn`。解码使用 greedy：`do_sample=false`、`num_beams=1`、`max_new_tokens=128`。

翻译 QA 检查：

- empty output；
- 与输入完全相同；
- 仍包含 Han 字符；
- 长度比例异常；
- 数字不一致。

通过检查的结果写入 `instruction_en`。缓存 key 包含模型版本、语言方向、解码配置和输入文本。

## Geometry

source、target 和 mask 先对齐到 mask 坐标系，然后共享 resize、crop 和 flip 参数。随机 crop 会检查 mask retention；多次尝试仍不满足时使用 contain / center-pad fallback。

几何变换后，三者尺寸一致，edit region 保持非空。mask 使用 nearest-neighbor resize，图像使用对应的连续插值。

## Determinism

训练 manifest 固定后，`manifest_index` 和 `sample_uid` 不再变化。epoch sample order 使用 hash-sort permutation；DDP rank `r` 读取 `permutation[r::world_size]`。

geometry、corruption 和翻译缓存使用稳定 seed。checkpoint 保存 epoch、cursor 和已消费样本数，resume 延续同一 sample sequence。
