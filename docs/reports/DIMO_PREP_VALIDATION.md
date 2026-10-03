# DiMO-Edit PREP validation

Status: `PREP_ONLY_TEACHER=true`, `TEACHER_QUALITY_NOT_VALIDATED=true`,
`DIMO_EDIT_FORMAL_READY=false`. These are plumbing/correctness observations,
not formal model results.

## Reference

- DiMO upstream: <https://github.com/yuanzhi-zhu/DiMO>
- Commit: `24613741ec9ca730273a6ebc822327878d5a086b`
- Reviewed: `train_DIMO_Meissonic.py`, `configs/DIMO_Meissonic.yaml`,
  `sample_Meissonic.py`, `scripts/launch_DIMO_Meissonic.sh`

## Real-model preparation smoke

- Temporary compatible checkpoint: explicit-region `local-one-step/final`
  (not a selected teacher)
- GPU: NVIDIA A100-PCIE-40GB
- Resolution / batch: 1024 / 1
- Complete DiMO steps: 2
- Step 2 student loss: `1.4603137969970703e-06`
- Step 2 auxiliary loss: `10.5`
- Step 2 student grad norm: `0.0003273052112825123`
- Step 2 auxiliary grad norm: `1.1116785511365541`
- Teacher grad norm: `0.0`
- Outside mismatch: `0`
- NaN/Inf count: `0`
- Peak allocated / reserved: `11,924,503,552 / 13,042,188,288` bytes

Step 1 has exactly zero DiMO/student gradient because teacher and auxiliary are
contractually initialized from identical state. Its auxiliary hard-CE update
makes the step-2 auxiliary distribution different, after which student gradient
is nonzero.

The one-step inference smoke decoded a 1024 image with
`number_of_transformer_forwards=1` and `outside_mismatch_count=0`.

## Resume comparison

The uninterrupted two-step run and `step 1 -> checkpoint -> resume -> step 2`
run used identical sample UIDs. SHA-256 traces match exactly for initialization
mask, initialization tokens, sampled student tokens, teacher pseudo-mask,
auxiliary pseudo-mask, and embedding noise. CPU and CUDA RNG states also match.

At step 2, hard auxiliary loss matches exactly; student loss differs by less
than `1e-6`. BF16 state comparisons satisfy absolute tolerance `5e-6`:

- student maximum absolute difference: `1.4789402484893799e-06`
- auxiliary maximum absolute difference: `3.978610038757324e-06`
- EMA maximum absolute difference: `1.478940303556442e-10`

The pure-tensor resume unit test is bitwise exact; the real BF16 CUDA comparison
uses the explicit tolerance above.

## Formal status

```text
SELECTED_DIMO_TEACHER = NONE
DIMO_EDIT_FORMAL_READY = false
```

No formal or long DiMO training was started.

## Regression checks

- Full pytest suite: `75 passed` (existing SFT plus new DiMO tests)
- Existing explicit-region SFT real 1024 one-step regression: PASS
- `python -m compileall src scripts tests`: PASS
- All repository shell scripts under `scripts/`: `bash -n` PASS
- `git diff --check`: PASS

## PREP-v1.1 validation

This section supplements rather than replaces the PREP-v1 results above.
Status remains `PREP_ONLY_TEACHER=true`, `TEACHER_QUALITY_NOT_VALIDATED=true`,
and `DIMO_EDIT_FORMAL_READY=false`.

### Correctness and numerics

- Inference student eval mode and `torch.inference_mode()`: PASS
- BF16 divergence input versus its FP32-cast reference: maximum absolute
  difference `0.0`; returned field is detached FP32
- FP32 surrogate retains vocabulary SUM and manual
  `g / masked_count / batch_size` gradient: PASS
- Gradient-checkpoint OFF versus ON: student loss maximum difference `0.0`;
  student adapter/region-gradient maximum absolute difference `0.0`
- Teacher bundle attestation and resume equality gate: PASS
- Raw student and strict FP32-shadow EMA loading: PASS
- Teacher-inherit / student-zero / auxiliary-zero LoRA dropout policy: PASS
- Fail-closed inference checkpoint compatibility gate: PASS

The real-model non-training `epsilon=1e-3` auxiliary-only diagnostic perturbed
420 LoRA tensors and then restored them. It recorded
`||p_teacher-p_aux||=1.6169142723083496`,
`||g_dimo||=1.0216221809387207`, student gradient norm
`0.0031216828887180186`, and zero teacher/auxiliary gradients during student
backward. The artifact is
`dimo-prep-v11-smoke-r2/dimo_nonzero_signal_test.json`; it is diagnostic only.

### Real 1024 regression

- GPU / resolution / batch: NVIDIA A100-PCIE-40GB / 1024 / 1
- Existing explicit-region SFT one-step: PASS; finite loss/logits/parameters
- DiMO two-complete-step prep smoke: PASS
- Step-2 student loss / auxiliary loss:
  `6.466890454248642e-07` / `9.375`
- Outside mismatch / NaN-or-Inf count: `0` / `0`
- Peak allocated / reserved:
  `11,923,454,976` / `13,035,896,832` bytes
- One-step raw-student inference: PASS
- One-step EMA inference: PASS
- Both inference paths: one transformer invocation, CFG multiplier `1`, outside
  mismatch `0`, NaN-or-Inf count `0`
- Repeated raw-student output PNG SHA-256:
  `0318dad88c8ac2c77c42678833c1c91d88a6a2a3b2453f4413cc38f37ce77329`
  for both runs

### PREP-v1.1 regression checks

- Full pytest suite: `97 passed`
- `python -m compileall src scripts tests`: PASS
- All repository shell scripts under `scripts/`: `bash -n` PASS
- `git diff --check`: PASS

No formal teacher was selected and no formal or long DiMO training was started.
