"""A complete student + auxiliary + EMA Region-DiMO optimizer step."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import time

import torch

from .auxiliary import auxiliary_loss
from .conditioning import predict_role_logits
from .diagnostics import tensor_grad_norm, token_diagnostics
from .divergence import dimo_divergence_gradient
from .forward_process import mask_student_prediction, schedule_mask_ratio
from .initialization import build_editing_initial_state, sample_student_tokens
from .rng import normal_noise_per_sample, sample_seeds
from .surrogate import surrogate_logit_loss


def _role_grad_norm(roles, role: str) -> float:
    return tensor_grad_norm(parameter for _, parameter in roles.role_named_parameters(role))


def _snapshot(roles, role: str) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().float().clone() for name, parameter in roles.role_named_parameters(role)}


def _update_ratio(before: Mapping[str, torch.Tensor], roles, role: str) -> float:
    current = dict(roles.role_named_parameters(role))
    delta = sum(
        float((current[name].detach().float() - value.to(current[name].device)).square().sum())
        for name, value in before.items()
    ) ** 0.5
    denominator = sum(float(value.square().sum()) for value in before.values()) ** 0.5
    return delta / max(denominator, 1e-12)


def complete_dimo_step(
    roles,
    *,
    source_tokens: torch.Tensor,
    edit_region_mask: torch.Tensor,
    sample_uids: Sequence[str],
    prompt_condition: Mapping[str, Mapping[str, torch.Tensor]],
    model_kwargs: Mapping[str, object],
    student_optimizer,
    auxiliary_optimizer,
    student_scheduler,
    auxiliary_scheduler,
    student_ema,
    base_seed: int,
    epoch: int,
    committed_dimo_step: int,
    mask_token_id: int,
    codebook_size: int,
    config: Mapping[str, object],
    gradient_sync=None,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    """Perform one checkpointable same-batch step and return diagnostics/state."""
    started = time.perf_counter()
    if len(sample_uids) != source_tokens.shape[0]:
        raise ValueError("one sample UID is required per batch row")
    if config["auxiliary"]["batch_policy"] != "same_batch":
        raise RuntimeError("only same_batch auxiliary updates are supported")
    if int(config["auxiliary"]["updates_per_student"]) != 1:
        raise RuntimeError("v1 requires exactly one auxiliary update per student")
    init_mask_seed = sample_seeds(
        base_seed=base_seed, epoch=epoch, sample_uids=sample_uids,
        global_dimo_step=committed_dimo_step, namespace="dimo-init-mask",
    )
    init_token_seed = sample_seeds(
        base_seed=base_seed, epoch=epoch, sample_uids=sample_uids,
        global_dimo_step=committed_dimo_step, namespace="dimo-init-token",
    )
    categorical_seed = sample_seeds(
        base_seed=base_seed, epoch=epoch, sample_uids=sample_uids,
        global_dimo_step=committed_dimo_step, namespace="dimo-student-sample",
    )
    teacher_seed = sample_seeds(
        base_seed=base_seed, epoch=epoch, sample_uids=sample_uids,
        global_dimo_step=committed_dimo_step, namespace="dimo-teacher-query-mask",
    )
    auxiliary_seed = sample_seeds(
        base_seed=base_seed, epoch=epoch, sample_uids=sample_uids,
        global_dimo_step=committed_dimo_step, namespace="dimo-aux-query-mask",
    )
    noise_seed = sample_seeds(
        base_seed=base_seed, epoch=epoch, sample_uids=sample_uids,
        global_dimo_step=committed_dimo_step, namespace="dimo-embedding-noise",
    )
    initial = build_editing_initial_state(
        source_tokens, edit_region_mask, mask_token_id, codebook_size,
        float(config["initialization"]["mask_ratio"]),
        mask_seeds=init_mask_seed, token_seeds=init_token_seed,
    )

    student_before = _snapshot(roles, "student")
    auxiliary_before = _snapshot(roles, "auxiliary")
    student_optimizer.zero_grad(set_to_none=True)
    auxiliary_optimizer.zero_grad(set_to_none=True)
    sigma = float(config["embedding_perturbation"]["sigma"])
    noise = None
    if config["embedding_perturbation"]["enabled"] and sigma:
        inner_dim = int(roles.base_model.inner_dim)
        reference = next(roles.base_model.parameters())
        shape = (source_tokens.shape[0], inner_dim, source_tokens.shape[1], source_tokens.shape[2])
        noise = normal_noise_per_sample(
            torch.empty(shape, device=source_tokens.device, dtype=reference.dtype), noise_seed
        )
    student_kwargs = dict(model_kwargs)
    student_kwargs.update(target_embedding_noise=noise, target_embedding_noise_sigma=sigma)
    student_logits = predict_role_logits(
        roles, "student", initial.initial_tokens, source_tokens, prompt_condition,
        edit_region_mask, initial.initial_roi_mask_ratio.to(source_tokens.device),
        float(config["distillation"].get("student_cfg", 1.0)), model_kwargs=student_kwargs,
    )
    student_tokens = sample_student_tokens(
        student_logits, source_tokens, edit_region_mask, seeds=categorical_seed,
        **config["sampling"],
    )
    teacher_rho = [
        schedule_mask_ratio(
            seed=seed, mode=config["pseudo_forward"]["teacher_ratio_mode"],
            r_min=float(config["pseudo_forward"].get("ratio_min", 0.02)),
            r_max=float(config["pseudo_forward"].get("ratio_max", 0.98)),
        ) for seed in teacher_seed
    ]
    teacher_query = mask_student_prediction(
        student_tokens, source_tokens, edit_region_mask, teacher_rho, mask_token_id,
        seeds=[seed + 1 for seed in teacher_seed],
    )
    with torch.no_grad():
        teacher_logits = predict_role_logits(
            roles, "teacher", teacher_query.pseudo_tokens, source_tokens, prompt_condition,
            edit_region_mask, teacher_query.rho_actual.to(source_tokens.device),
            float(config["distillation"]["teacher_cfg"]), model_kwargs=model_kwargs,
        )
        auxiliary_query_logits = predict_role_logits(
            roles, "auxiliary", teacher_query.pseudo_tokens, source_tokens, prompt_condition,
            edit_region_mask, teacher_query.rho_actual.to(source_tokens.device),
            float(config["distillation"]["auxiliary_cfg"]), model_kwargs=model_kwargs,
        )
        gradient = dimo_divergence_gradient(
            teacher_logits, auxiliary_query_logits,
            temperature_teacher=float(config["distillation"]["teacher_temperature"]),
            temperature_auxiliary=float(config["distillation"]["auxiliary_temperature"]),
            mode=config["distillation"]["mode"],
            beta=float(config["distillation"].get("jeffreys_beta", 0.5)),
            pseudo_mask=teacher_query.pseudo_mask,
        )
    student_loss = surrogate_logit_loss(student_logits, gradient, teacher_query.pseudo_mask)
    # With gradient checkpointing, recomputation happens during backward and
    # must see the same logical adapter as the original forward.
    roles.activate_adapter("student")
    student_loss.backward()
    if gradient_sync is not None:
        gradient_sync('student')
    student_grad = _role_grad_norm(roles, "student")
    gradient_norm = float(gradient.detach().float().norm())
    if (
        (gradient_norm > 1e-12 and student_grad == 0)
        or _role_grad_norm(roles, "auxiliary") != 0
        or _role_grad_norm(roles, "teacher") != 0
    ):
        raise RuntimeError("student backward violated role gradient isolation")
    student_optimizer.step()
    student_scheduler.step()
    student_optimizer.zero_grad(set_to_none=True)

    aux_rho = [
        schedule_mask_ratio(
            seed=seed, mode=config["pseudo_forward"]["auxiliary_ratio_mode"],
            r_min=float(config["pseudo_forward"].get("ratio_min", 0.02)),
            r_max=float(config["pseudo_forward"].get("ratio_max", 0.98)),
        ) for seed in auxiliary_seed
    ]
    aux_query = mask_student_prediction(
        student_tokens.detach(), source_tokens, edit_region_mask, aux_rho, mask_token_id,
        seeds=[seed + 1 for seed in auxiliary_seed],
    )
    auxiliary_logits = predict_role_logits(
        roles, "auxiliary", aux_query.pseudo_tokens, source_tokens, prompt_condition,
        edit_region_mask, aux_query.rho_actual.to(source_tokens.device),
        float(config["distillation"].get("auxiliary_train_cfg", 1.0)), model_kwargs=model_kwargs,
    )
    aux_loss = auxiliary_loss(
        auxiliary_logits, student_tokens.detach(), aux_query.pseudo_mask,
        soft_target_weight=float(config["auxiliary"]["soft_target_weight"]),
        student_logits=student_logits.detach(),
    )
    roles.activate_adapter("auxiliary")
    aux_loss.backward()
    if gradient_sync is not None:
        gradient_sync('auxiliary')
    aux_grad = _role_grad_norm(roles, "auxiliary")
    if aux_grad == 0 or _role_grad_norm(roles, "teacher") != 0 or _role_grad_norm(roles, "student") != 0:
        raise RuntimeError("auxiliary backward violated role gradient isolation")
    auxiliary_optimizer.step()
    auxiliary_scheduler.step()
    auxiliary_optimizer.zero_grad(set_to_none=True)
    student_ema.update(roles.role_named_parameters("student"))

    diagnostics = token_diagnostics(student_logits, student_tokens, source_tokens, edit_region_mask)
    diagnostics["nan_inf_count"] = sum(
        int((~torch.isfinite(value.detach())).sum())
        for value in (student_logits, teacher_logits, auxiliary_logits)
    )
    diagnostics.update(
        loss_dimo=float(student_loss.detach()), loss_aux=float(aux_loss.detach()),
        student_grad_norm=student_grad, aux_grad_norm=aux_grad, teacher_grad_norm=0.0,
        dimo_gradient_norm=gradient_norm,
        initial_roi_mask_ratio=float(initial.initial_roi_mask_ratio.mean()),
        teacher_query_roi_mask_ratio=float(teacher_query.rho_actual.mean()),
        aux_query_roi_mask_ratio=float(aux_query.rho_actual.mean()),
        max_abs_teacher_logit=float(teacher_logits.detach().abs().max()),
        max_abs_aux_logit=float(auxiliary_logits.detach().abs().max()),
        student_parameter_update_ratio=_update_ratio(student_before, roles, "student"),
        aux_parameter_update_ratio=_update_ratio(auxiliary_before, roles, "auxiliary"),
        step_wall_time=time.perf_counter() - started,
        committed_dimo_step=committed_dimo_step + 1,
    )
    return diagnostics, {
        "initial_mask": initial.initial_mask.detach(),
        "initial_tokens": initial.initial_tokens.detach(),
        "student_tokens": student_tokens.detach(),
        "teacher_pseudo_mask": teacher_query.pseudo_mask.detach(),
        "auxiliary_pseudo_mask": aux_query.pseudo_mask.detach(),
        "embedding_noise": noise.detach() if noise is not None else torch.empty(0),
    }
