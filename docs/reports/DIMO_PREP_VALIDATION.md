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
