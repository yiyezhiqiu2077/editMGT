# Screened total20K and epoch7 Dense SFT continuation

This is an independent exploratory dataset and run on the existing
`exp/e3-dense-fp32-v1` maintenance branch. It does not change Frozen D200K,
the old MagicBrush manifest, the paused DiMO checkpoints, the formal
DiMO recipe, or the shared geometry rules.

The latest user instruction reduces **the final total including MagicBrush**
to 20–30K and requests preparation during downloads. The registered fast
plan uses 20K total. The superseded 38,784-row plan never started training;
its configs remain historical templates. The existing experiment directory
name still contains `mixed38k` for continuity; its active manifest is 20K.
All new downloads use `https://hf-mirror.com`, immutable revisions,
content checksums, and `requests.Session.trust_env = False`. Downloader
children also remove proxy environment variables. No reverse proxy is used.

## Preregistered candidate and final selection

Final target: CrispEdit 2,000 + ScaleEdit 7,000 + InterEdit 2,216 +
the exact existing 8,784 MagicBrush training turns = 20,000.

The initial 12K/12K/6K proposal was revised **before any training or DEV
evaluation**: a real CrispEdit geometry pilot found frequent source/target
aspect mismatches. For example 1536×1024 versus 1537×1024 fails the existing
strict aspect gate. Such examples are excluded; the gate is not relaxed.
The revised quotas provide reserves without altering image/mask alignment.

Pinned repositories:

| Dataset | Revision |
|---|---|
| TTangenty/CrispEdit-labeling-39k | dcbd1c952e93e4361ad862b33f3acd1cc74bec5a |
| TTangenty/ScaleEdit-labeling-25k | f97ffb061d4275bcbbff6500572139b6f8df1c4e |
| a1557811266/Inter-Edit-Train | b319d9fdb45cc670ec263fe4aeaa962f1623d5dc |
| facebook/nllb-200-distilled-1.3B | 7be3e24664b38ce1cac29b8aeed6911aa0cf0576 |

CrispEdit/ScaleEdit shards are ordered by seed42, dataset, revision and
path hashes, then downloaded to provide at least 6K/9K candidate rows.
Already downloaded files outside this smaller preregistered cohort remain
unused. No additional CrispEdit download is required for the active prefix.
All existing schema quality, scene, grounding and mask flags must pass.
Candidates must decode, have a nonempty instruction and mask, pass the
unaltered 1024 geometry gate, and have different source/target bytes.
Automatic mask QC is not a claim of human verification or semantic correctness.

InterEdit uses one fixed early asset shard and their required
source archives, with `better_data=true` only. Random later asset shards
would require sources throughout the full >2 TB corpus. This bounded
subset is **not** a globally representative random InterEdit sample.
Its masks remain labeled `user_guidance_region`, distinct from precise
edit-region masks. Final InterEdit selection uses the existing edit-type /
region-size stratification and source round robin.

Native source AND target hashes from all 528 original MagicBrush DEV
records, including the excluded geometry case, protect evaluation lineage.
All original MagicBrush Train records take priority over new data. Every
connected candidate group touching protected content is removed. New
source-group disjoint auxiliary DEV sets contain 128 records per dataset;
all their connected groups/content are excluded from training. Official
TEST remains unread and its content-level isolation is `NOT_TESTED`.

Chinese prompts from the bounded InterEdit `better_data` metadata cohort
are pretranslated while image archives download, using pinned local NLLB,
deterministic greedy decoding and a revision-keyed cache. These provisional
candidate/reserve translations do not confer training eligibility: pixel
screening, final group selection and translation QA are still mandatory.
Only one cache writer runs, and final selection waits for it to finish. Empty/copy/
remaining-Han/length/digit QA failures are rejected and deterministically
backfilled. A fixed 500-prompt review artifact is generated; automatic QA
does not replace human semantic review, whose status remains explicit.

Original image/mask bytes are materialized for the selected rows and auxiliary
DEV only. Canonical identities remain unchanged. Every frozen training row
is checked by the real loader, and FP32-VQ containment / fixed data montages
are generated before publishing independent `MIXED_READY.json`.

Insufficient eligible candidates, bad translations, changed hashes, or
insufficient disk cause a hard stop. No substitution, padding, geometry
relaxation, or training recipe changes happen automatically.

## Training and evaluation

Warm start is the verified full FP32 Dense SFT **epoch7 / step3843**
checkpoint. Only weights are inherited. This is a new dataset, optimizer,
scheduler and step counter; it is not exact resume of the old optimizer.

Four A800, batch1 per GPU, accumulation4, global batch16, LR2e-6,
200-step warmup, FP32 trainable parameters/gradients/Adam moments,
BF16 autocast forward, unchanged E3 corruption/conditioning/mask rules.
Five additional epochs: 1,250 steps each, 6,250 total. Save all five epoch
checkpoints. The sampler never pads or drops rows.

Before the long run: CPU acceptance; two-step real four-GPU smoke;
fresh20 versus 10+resume20 full bitwise state/sample/RNG comparison.
Disk is budgeted for all new smoke and SFT checkpoints plus an atomic save.
Old checkpoints and datasets are retained. Curves remain local; no Release upload.

After training, four GPU evaluation queues compare epoch7 and all five
new checkpoints on the identical existing DEV527 / 266 source groups,
seeds [0,1], 1024, Teacher 12 steps / CFG5, FP32 weights / BF16 forward.
Seed means precede source-group paired bootstrap (10K resamples, seed42).
Report all inside/outside metrics, DINO/CLIP when available, no-edit diagnostics,
and fixed comparison images. Loss alone is not an editing-quality criterion.

Preregistered exploratory candidate recommendation: inside LPIPS mean delta
≤−0.005, paired 95% CI upper bound <0, outside LPIPS mean degradation ≤0.01;
choose minimum inside LPIPS, ties favor earlier steps. These are individual
exploratory CIs across five candidates, not multiplicity-adjusted confirmatory
tests. PSNR/SSIM and semantic case review must accompany the recommendation.
No eligible candidate means stop before new DiMO. The coordinator finishes
SFT/evaluation and records `SFT_EVALUATED_DIMO_PENDING` for quality review
before preparing the new Teacher and DiMO run; it never restarts the paused
old DiMO run or claims that LPIPS alone proves instruction-following success.

## Commands

```bash
python scripts/data/download_mixed30k_mirror.py --experiment-root "$MIXED_EXPERIMENT_ROOT"
python scripts/data/incremental_mixed_preparation.py --experiment-root "$MIXED_EXPERIMENT_ROOT"
python scripts/data/prepare_mixed30k_corpus.py --experiment-root "$MIXED_EXPERIMENT_ROOT" \
  --previous-experiment "$PREVIOUS_EXPERIMENT_ROOT"
python scripts/train/run_mixed30k_continuation.py --experiment-root "$MIXED_EXPERIMENT_ROOT" \
  --previous-experiment "$PREVIOUS_EXPERIMENT_ROOT"
```

The coordinator expects a running downloader and current committed CPU
acceptance. It screens each complete, checksum-verified planned shard during downloads;
partial or unplanned files cannot enter the cache. Per-shard output hashes
and code/schema/geometry contract fingerprints guard reuse. Stable merging
reproduces the original builder file/row order. NLLB and InterEdit downloads
are prioritized. `PREPARATION_STATUS.json` tracks actual screening and
translation. The coordinator monitors downloads, freezes the corpus,
checks admission, trains and evaluates automatically. Hard failures are
recorded in `STATUS.json`; there is no implicit failure recovery.
