# Mixed20K: automatic final-SFT to Full DenseDiMO

The user explicitly requests automatic DiMO after five additional SFT epochs,
using the **last** SFT weights, then selects the **same mixed20K, five DiMO
epochs**. This overrides the earlier DEV-based Teacher recommendation gate
for this independent exploratory run. DEV remains diagnostic; no editing
quality improvement is claimed from training loss or this policy change.

SFT warm starts the previous Dense epoch7/step3843 weights and completes
6,250 optimizer steps on the frozen 20K. The final checkpoint is
`sft/train/checkpoint-6250`: added epoch5, cumulative SFT epoch12, exactly
100,000 committed samples. Handoff requires both successful completion of
the SFT training command and an intact, content-verified final checkpoint.
The dataset hash, original warm-start identity, continuation contract and
SFT horizon must match. An intermediate checkpoint is never substituted.

The final checkpoint is exported unchanged to `selected_teacher`. Its
Teacher contract is `mixed20k_user_last`, explicitly exploratory and rejected
by the formal D200K Teacher gate. Fresh Student, Auxiliary and EMA start from
this exact Teacher. Optimizers and counters are new; the paused old DiMO run
is not resumed. Four A800, batch1 per GPU and accumulation1 give global
batch4: five epochs on 20,000 rows require **25,000 Student steps**.

`full_dense_mixed20k_4a800_5epoch.yaml` inherits the validated Full DenseDiMO
math: all ROI tokens MASK (`r_init=1.0`), sigma0.3, CFG1, linear surrogate,
FP32 parameters/gradients/Adam moments/EMA, BF16 forward, constant Student
and Auxiliary LR1e-6, EMA0.9995. Each Student step has one Auxiliary update.
The manifest, data roots, geometry, sampling and conditioning remain fixed.
No formal D200K or existing MagicBrush configuration is changed.

The standby coordinator waits without holding the SFT coordinator lock.
After the SFT command succeeds and final weights pass validation, it ends
only the owned **post-training evaluation coordinator**, preserving partial
DEV outputs, and waits for its lock and GPU processes to exit. It never
terminates an SFT training worker or another user's GPU job. This avoids
waiting for the earlier five-candidate quality-selection sweep. If that
sweep already finished, its results are preserved and the user-selected
last checkpoint still applies.

Before the long DiMO run, the coordinator executes a two-step four-GPU smoke
and compares fresh20 with 10+resume20, including model/optimizer/scheduler/
EMA/RNG/sampler state and every rank's sample sequence. A failed admission,
SFT run, identity check or acceptance stops handoff; no implicit retry or
recipe adjustment occurs. A user pause does not trigger a restart.

Full resume checkpoints are retained at steps5,000/15,000/25,000. Raw/EMA
inference exports are retained at steps0/5,000/15,000/25,000. No step500 DEV
or other evaluator runs alongside live training. The unchanged DEV527,
266 source groups and seeds[0,1] are evaluated after DiMO training: Teacher
12 steps/CFG5, Raw Student and EMA one step/CFG1, 1024 resolution. Curves
remain local; GitHub Release upload stays disabled.

Storage is budgeted across two filesystems without deleting any old weights,
optimizer states or data. SFT, Teacher, DiMO smoke2 and main resume states
stay on the experiment filesystem; DiMO fresh/resume acceptance states and
main raw/EMA exports use the secondary filesystem. The main inference
directory is an explicit symlink. Per-invocation disk admission budgets
only the states that invocation can write, with an atomic-save reserve.

The current SFT controller pins commit07f3fcdd and its clean worktree.
The new coordinator is committed in a detached worktree, then pushed to the
existing `exp/e3-dense-fp32-v1` remote branch. No new branch is created and
the running SFT worktree stays pinned. Its immutable CPU acceptance must
match the new coordinator commit before the standby starts.

## Launch and evidence

Create `USER_AUTO_LAST_SFT_DIMO.json` from the explicit user request and
answer, then write successful CPU acceptance for the committed code into
`admission/DIMO_AUTO_CPU_ACCEPTANCE.json`. Launch from the new committed
worktree:

```bash
python -u scripts/train/auto_mixed_last_sft_dimo.py \
  --experiment-root "$MIXED_EXPERIMENT_ROOT" \
  --previous-experiment "$PREVIOUS_EXPERIMENT_ROOT" \
  --secondary-root "$MIXED_DIMO_SECONDARY_ROOT"
```

`DIMO_HANDOFF_STATUS.json` records waiting and ownership separately from
the active SFT `STATUS.json`. After ownership, both record the new run.
`DIMO_AUTO_HANDOFF_STARTED.json`, `TEACHER_SELECTION.json`, the exported
Teacher contract, `DIMO_STORAGE_LOCATIONS.json`, disk budgets, CUDA acceptance,
resolved config, metrics and code SHA preserve the complete handoff identity.
