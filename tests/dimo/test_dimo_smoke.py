import torch

from src.dimo.step import complete_dimo_step
from tests.dimo.toy import batch, make_bundle, step_config


def test_tiny_complete_dimo_step_has_required_diagnostics():
    roles, student_opt, aux_opt, student_sched, aux_sched, ema = make_bundle()
    diagnostics, artifacts = complete_dimo_step(
        roles, **batch(), student_optimizer=student_opt, auxiliary_optimizer=aux_opt,
        student_scheduler=student_sched, auxiliary_scheduler=aux_sched, student_ema=ema,
        base_seed=42, epoch=0, committed_dimo_step=0, mask_token_id=5,
        codebook_size=5, config=step_config(),
    )
    assert diagnostics["committed_dimo_step"] == 1
    assert diagnostics["loss_dimo"] > 0
    assert diagnostics["loss_aux"] > 0
    assert diagnostics["student_grad_norm"] > 0
    assert diagnostics["aux_grad_norm"] > 0
    assert diagnostics["teacher_grad_norm"] == 0
    assert diagnostics["outside_mismatch_count"] == 0
    assert diagnostics["nan_inf_count"] == 0
    assert diagnostics["student_parameter_update_ratio"] > 0
    assert diagnostics["aux_parameter_update_ratio"] > 0
    region = batch()["edit_region_mask"]
    assert torch.equal(artifacts["student_tokens"][~region], batch()["source_tokens"][~region])
