# Full DenseDiMO implementation contract

The formal branch is exp/e3-dense-fp32-v1. E3 Dense training mathematics and
configs/train/dense_d200k_5epoch.yaml are unchanged. Historical GPU records in
LEGACY_EXPERIMENTS.md are not acceptance evidence for this implementation.

## DEV teacher selection

The committed YAML declares the protocol; actual preregistration is an immutable
selection_preregistration.json written BEFORE any formal DEV generation. It binds
the resolved plan, released FP32 baseline weight hashes, canonical DEV manifest
hashes, inference protocol, selection rules, train/test hashes and group evidence.

The baseline is E0-region (released weights with ROI-relative timestep). Let
d_i = candidate Inside LPIPS - baseline Inside LPIPS after averaging generation
seeds for each sample. Eligibility requires mean(d_i) <= -0.005 and the upper
endpoint of the paired 95% percentile cluster CI < 0. Outside degradation uses
candidate-minus-baseline LPIPS to SOURCE; mean degradation must be <= 0.01.
These are absolute LPIPS differences, not percentages. The CI proves positive
improvement; it does not require the whole CI to exceed 0.005 improvement.

All ten registered checkpoints must be evaluated. Among eligible candidates,
minimum mean Inside LPIPS wins; exact ties choose earlier optimizer step. No
eligible candidate leaves PENDING/NO_ELIGIBLE_CHECKPOINT and blocks formal export.
TEST manifests are read only for split-isolation identity; no TEST predictions or
scores enter the selection procedure.

The semantic evidence document must be reviewed from upstream dataset metadata,
not produced by treating a group_id field name as proof. The example contract is
deliberately reviewed=false and cannot run formal selection. Missing/unconfirmed
evidence or conflicting UIDs fail closed. Source and target image content hashes
join groups transitively, covering reused images and target-to-source multi-turn
links even when local group labels differ. Any cluster crossing train/DEV/TEST
stops selection. Exact content identity cannot prove absence of all transformed
near-duplicates; reviewed original-source/sequence identity remains necessary.

Paired candidates must have the same UID, group/cluster, source/target/region
hashes, manifest identity and generation seed set. Duplicate UIDs/seeds and
incomplete seed sets fail. Seeds are averaged within sample first. Cluster
bootstrap samples G clusters WITH replacement, includes ALL samples from each
draw (including repeated clusters), and divides their total paired difference by
their total sample count. This is the sample-weighted estimand, not an average
of group means. Fixed seed 42, 10,000 resamples, percentile CI at 2.5/97.5%.
Report sample count, group count, group sizes and empty-region exclusions.
Fewer than two independent groups cannot establish a formal CI.

## Objective and precision

Teacher, Student, Auxiliary parameters, parameter gradients, AdamW moments and
EMA are FP32. CUDA Transformer forward uses BF16 autocast; VQ stays FP32. Frozen
CLIP/Gemma/VQ are shared outside the role manager. Student/Auxiliary each own a
complete independently cloned Transformer including region embedding, with
independent optimizers and DDP wrappers. Teacher is always frozen/eval.

Formal student objective is linear/roi-vocabulary-sum-v1:

    mean_b sum_selected_tokens sum_v(z[b,t,v] * detached_g[b,t,v]) / count[b]

FKL field is auxiliary_probability - teacher_probability in FP32. Vocabulary
is SUMMED, tokens are normalized per sample, and samples are batch-averaged.
Analytical gradient is detached_g/(batch_size*selected_token_count) inside the
mask and zero outside. Squared surrogate is retained as an explicitly separate
compatibility objective; it may lose small g through cancellation in z-(z-g).
Both objectives are audited at the real differentiable logits boundary at steps
1 and 20, before student backward. CPU tests also cover double references,
logit scales, tiny g, and actual CPU BF16-autocast layer outputs and FP32 weight
gradients. CPU BF16 evidence is not CUDA BF16 acceptance.

Linear objective may be negative and is not a CE or a monotonic quality score.
The loss figure separately shows linear objective, Auxiliary CE, and the
nonnegative diagnostic 0.5*mean_roi(vocabulary_sum(g²)). Teacher/Auxiliary logits
and divergence must be finite BEFORE any replacement; nan_to_num is not used.
Initial teacher/auxiliary parity can give exactly zero student gradient at step1.
That step can succeed; zero parameter displacement is not a failure criterion.

## Commit, checkpoint and memory

Commit is owned by StepCommit. Two backward/optimizer/scheduler sequences and
every EMA update finish before rank checks and global step/cursor assignment.
CUDA EMA uses FP32 same-device shadows and waits for its stream completion;
it does not copy all dense weights to CPU each step. Failures invalidate the
in-memory transaction and require reload, rather than pretending to roll back.
Fault tests cover both optimizer boundaries, partial EMA, completed EMA before
commit, and faults raised on only one Gloo rank.

DDP averages complete trainable gradients. find_unused_parameters=False.
Routine norms use FP32 tensor reductions with one scalar extraction, not full
FP64 scans or dense snapshots. Update ratios inspect 16 fixed evenly spaced
coordinates per tensor at steps1/20/every100; absent points are null. Rank
diagnostics inspect these coordinates plus four optimizer coordinates, counters
and scheduler state, and explicitly declare fixed_coordinate_sample scope.
They are not presented as full-state bitwise hashes. The independent two-rank
fresh/resume test compares full weights, optimizer moments, schedulers, EMA,
sampler and each rank RNG bitwise.

Memory evidence covers role/DDP loading, student forward, teacher/auxiliary
query (while student graph is live), student backward and first optimizer,
auxiliary backward and first optimizer, completed EMA, and checkpoint save.
This is the actual execution order, not a claim that student backward precedes
the query needed to form its gradient. GPU startup must measure both optimizer
state allocations. Approximately 1B parameters per role implies 36–40 GB of
FP32 parameters/moments/EMA/gradient residency before encoders, activations,
communication buckets and allocator temporaries. DDP does not shard residency.

Only rank0 saves weights/optimizer state. Weight/EMA shards materialize bounded
CPU weight chunks, not full GPU clones. Completion ledger hashes every file;
files and directories are fsynced around atomic publication. Partial directories
are never resumable. Full checkpoint includes role weights, EMA count/decay,
two optimizers/schedulers, sampler, loader generator replay state, per-rank RNG,
teacher/data/config/code identity. Inference exports omit optimizer state.
Recovery preserves the abandoned scalar-log tail and future inference exports
in a local recovery archive; active logs resume at the loaded checkpoint cursor.
Loading an older checkpoint when a newer complete recovery checkpoint exists
is rejected. Existing checkpoint directories must pass integrity and identity
checks before reuse. Offline compare_full_dense_resume checks all tensors and
training states, including each rank RNG, without allocating Dense GPU models.

Control/status collectives use a separate Gloo group with a two-hour timeout,
so rank0's large checkpoint I/O or step500 local evaluation does not expire a
short NCCL timeout on waiting ranks. Gradient collectives retain NCCL. A stalled
worker is still subject to the configured process-group deadlines.

DDP OOM stops all workers with stage evidence. FSDP FULL_SHARD wrapper and
separate candidate configuration/state context exist, but are UNVALIDATED and
not wired as a production fallback. Its state schema differs from DDP.

## Metrics and inference

Pixel metric inputs are aligned RGB in [0,1]. Inside compares generated to target;
Outside compares generated to source. PSNR uses masked RGB-channel MSE with
epsilon=1e-12. SSIM uses an 11x11 sigma1.5 Gaussian full-image local map, zero
padding5 and channel mean, followed by ROI-weighted spatial mean. Neighborhoods
at ROI boundaries can include exterior pixels; no boundary erosion is applied.
LPIPS uses AlexNet learned per-level feature-distance maps, area-resizes the
pixel mask to each map, computes a weighted mean at each level and sums levels.
Feature receptive fields can cross ROI boundaries. Full LPIPS uses full images.
An empty region yields invalid/null, never zero. Paired missing validity must
match between candidate and baseline; exclusions are counted explicitly.

Source tokens outside ROI are copied exactly. VQ decoding receptive fields and
reconstruction can still alter exterior pixels; pixel hard-lock is not claimed.
Student/EMA loaders construct one Transformer only and never create teacher or
auxiliary roles or load AdamW. Fixed manifests carry UID/group/source/geometry
identity. CUDA-synchronized timings default to warmup1/repeats3, record CLIP,
Gemma, VQ encode, Transformer and VQ decode separately, plus total generation
call time. Dataset geometry and image file I/O are outside total latency;
pipeline preprocessing/sampling is included in the unassigned remainder.
Transformer forward counts are measured, not inferred from requested steps.

## Monitor and report

Independent CPU monitor reads only complete committed JSONL records, once per
500 optimizer steps. It uploads exactly two immutable PNG assets per milestone,
under a run prerelease tied to fixed code SHA. Linear/CE/energy use separate
panels. Gradient null points are absent measured points. No sample/image data,
private paths, model or report archive is sent. GH_TOKEN is removed from the
DiMO launcher environment; monitor credentials are supplied separately.

Authentication/network/rate/timeout/plot/process errors never govern torchrun
lifecycle. Retry attempts are finite; --retry-pending re-enables later retries.
Owned PNG cache has a bounded pending-pair count, and evicted plots can be
regenerated from authoritative scalar logs. Report archives stay local, support
partial results, and contain internal SHA256 and an archive sidecar. Unavailable
quality/GPU measurements are NOT_AVAILABLE/NOT_RUN_RESOURCE_UNAVAILABLE.
