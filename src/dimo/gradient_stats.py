"""Student clipping and explicitly sampled versus full update measurements."""
import math
import torch
from .diagnostics import tensor_grad_norm
from .distributed import collective_require


def clip_student_gradients(named_parameters, max_norm):
    items = [(n, p) for n, p in named_parameters if p.requires_grad]
    device = items[0][1].device
    pre = tensor_grad_norm(p for _, p in items)
    collective_require(math.isfinite(pre), "DIMO_NONFINITE_STUDENT_GRADIENT", device)
    torch.nn.utils.clip_grad_norm_([p for _, p in items], max_norm=max_norm,
                                  norm_type=2., error_if_nonfinite=True)
    post = tensor_grad_norm(p for _, p in items)
    groups = {"attention": lambda n: ".attn." in n or ".attention." in n,
              "ffn": lambda n: ".ff." in n or "proj_mlp" in n or ".ffn." in n,
              "embed": lambda n: n.startswith("embed."),
              "mlm_head": lambda n: n.startswith("mlm_layer.")}
    return {"student_grad_norm_preclip": pre, "student_grad_norm_postclip": post,
            "student_clip_triggered": pre > max_norm,
            **{f"student_grad_norm_{group}": tensor_grad_norm(p for n, p in items if select(n))
               for group, select in groups.items()}}


def snapshot_full_trainable(named_parameters):
    # A CPU copy avoids adding a full dense model to the GPU's peak usage.
    return {n: p.detach().cpu().clone() for n, p in named_parameters if p.requires_grad}


def full_update_statistics(before, named_parameters):
    numerator = denominator = 0.
    for n, p in named_parameters:
        if p.requires_grad:
            old = before[n]
            # Per-tensor transfer/reduction limits temporary storage.
            numerator += float((p.detach().cpu().double() - old.double()).square().sum())
            denominator += float(old.double().square().sum())
    norm = math.sqrt(numerator)
    return {"student_parameter_update_norm_full": norm,
            "student_parameter_update_ratio_full": norm / max(math.sqrt(denominator), 1e-12),
            "full_update_parameter_scope": "student_trainable_parameters"}
