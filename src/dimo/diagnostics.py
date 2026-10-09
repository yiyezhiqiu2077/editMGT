"""Small, side-effect-free Region-DiMO diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .conditioning import predict_role_logits
from .divergence import dimo_divergence_gradient
from .surrogate import surrogate_logit_loss


def tensor_grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum())
    return total**0.5


def token_diagnostics(
    student_logits: torch.Tensor,
    student_tokens: torch.Tensor,
    source_tokens: torch.Tensor,
    edit_region_mask: torch.Tensor,
) -> dict[str, float | int]:
    region = edit_region_mask.bool()
    outside = ~region
    prob = torch.softmax(student_logits.detach().float(), dim=-1)
    entropy = -(prob * torch.log(prob.clamp_min(1e-12))).sum(-1)
    inside_count = max(int(region.sum()), 1)
    inside_tokens = student_tokens[region]
    return {
        "student_entropy_inside_roi": float(entropy[region].mean()),
        "student_source_copy_ratio_inside_roi": float((inside_tokens == source_tokens[region]).sum()) / inside_count,
        "student_unique_token_fraction_inside_roi": float(inside_tokens.unique().numel()) / inside_count,
        "outside_mismatch_count": int((student_tokens[outside] != source_tokens[outside]).sum()),
        "max_abs_student_logit": float(student_logits.detach().abs().max()),
        "nan_inf_count": int((~torch.isfinite(student_logits.detach())).sum()),
    }


def run_nonzero_signal_diagnostic(
    roles,
    *,
    target_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    edit_region_mask: torch.Tensor,
    prompt_condition,
    model_kwargs,
    output_path: str | Path,
    epsilon: float = 1e-3,
) -> dict[str, object]:
    """Perturb only auxiliary LoRA, prove the DiMO gradient path, then restore."""
    if epsilon <= 0:
        raise ValueError("diagnostic epsilon must be positive")
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(torch.cuda.current_device()) if torch.cuda.is_initialized() else None
    auxiliary_parameters = roles.role_named_parameters("auxiliary")
    adapter_items = [(name, value) for name, value in auxiliary_parameters if name != "edit_region_embedding"]
    if not adapter_items:
        raise RuntimeError("nonzero signal diagnostic requires an auxiliary adapter tensor")
    originals = {name: parameter.detach().clone() for name, parameter in adapter_items}
    for _, value in roles.role_named_parameters("student") + roles.role_named_parameters("auxiliary"):
        value.grad = None
    try:
        with torch.no_grad():
            for index, (_, parameter) in enumerate(adapter_items):
                pattern = torch.linspace(-1.0, 1.0, parameter.numel(), device=parameter.device)
                if index % 2:
                    pattern = pattern.flip(0)
                parameter.add_(pattern.reshape_as(parameter).to(parameter.dtype), alpha=epsilon)
        timestep = (
            edit_region_mask.bool().flatten(1).sum(1).float()
            / edit_region_mask[0].numel()
        ).to(target_tokens.device)
        student_logits = predict_role_logits(
            roles, "student", target_tokens, reference_tokens, prompt_condition,
            edit_region_mask, timestep, 1.0, model_kwargs=model_kwargs, training=True,
        )
        with torch.no_grad():
            teacher_logits = predict_role_logits(
                roles, "teacher", target_tokens, reference_tokens, prompt_condition,
                edit_region_mask, timestep, 1.0, model_kwargs=model_kwargs, training=False,
            )
            auxiliary_logits = predict_role_logits(
                roles, "auxiliary", target_tokens, reference_tokens, prompt_condition,
                edit_region_mask, timestep, 1.0, model_kwargs=model_kwargs, training=False,
            )
            gradient = dimo_divergence_gradient(
                teacher_logits, auxiliary_logits, pseudo_mask=edit_region_mask
            )
            probability_difference_norm = float(
                (torch.softmax(teacher_logits.float(), -1)
                 - torch.softmax(auxiliary_logits.float(), -1)).norm()
            )
        loss = surrogate_logit_loss(student_logits, gradient, edit_region_mask)
        roles.activate_adapter("student")
        loss.backward()
        student_grad_norm = tensor_grad_norm(
            value for _, value in roles.role_named_parameters("student")
        )
        teacher_grad_norm = tensor_grad_norm(
            value for _, value in roles.role_named_parameters("teacher")
        )
        auxiliary_grad_norm = tensor_grad_norm(
            value for _, value in roles.role_named_parameters("auxiliary")
        )
        result = {
            "epsilon": float(epsilon),
            "perturbed_auxiliary_parameter_count": len(adapter_items),
            "probability_difference_norm": probability_difference_norm,
            "dimo_gradient_norm": float(gradient.norm()),
            "student_grad_norm": student_grad_norm,
            "teacher_grad_norm": teacher_grad_norm,
            "auxiliary_grad_norm": auxiliary_grad_norm,
            "passed": probability_difference_norm > 0 and float(gradient.norm()) > 0
            and student_grad_norm > 0 and teacher_grad_norm == 0 and auxiliary_grad_norm == 0,
        }
        if not result["passed"]:
            raise RuntimeError(f"DIMO_NONZERO_SIGNAL_DIAGNOSTIC_FAILED: {result}")
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
    finally:
        with torch.no_grad():
            for name, parameter in adapter_items:
                parameter.copy_(originals[name])
        for _, value in roles.role_named_parameters("student") + roles.role_named_parameters("auxiliary"):
            value.grad = None
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device=torch.cuda.current_device())
