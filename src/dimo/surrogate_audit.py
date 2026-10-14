"""Compare objective gradients at the real student-logit boundary."""
import torch
from .surrogate import surrogate_logit_loss, linear_surrogate_logit_loss


def audit_surrogate_gradients(logits, gradient, mask):
    if not logits.requires_grad:
        raise ValueError("audit requires real differentiable student logits")
    counts = mask.flatten(1).sum(1)
    shape = [len(counts)] + [1] * (logits.ndim - 1)
    # Reverse-mode traverses batch mean -> token mean -> vocabulary sum ->
    # product. Calculate its FP32 scale before multiplying g, rather than
    # dividing g first. Mathematically equal expressions can straddle a BF16
    # rounding midpoint and differ by one BF16 ULP at the logit boundary.
    # Keep the strict comparison; this is a reference correction, not a
    # relaxed tolerance or a change to the training objective.
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    scale = (torch.ones_like(counts, dtype=dtype) / len(counts)) / counts
    expected = (gradient.detach().to(dtype) * scale.reshape(shape)).masked_fill(~mask[..., None].bool(), 0)
    expected = expected.to(logits.dtype).to(dtype)
    original = (gradient.detach().to(dtype) * mask[..., None] / counts.reshape(shape) / len(counts)).to(logits.dtype).to(dtype)
    result = {"logits_dtype": str(logits.dtype), "expected_norm": float(expected.norm()),
        "reference_version": "reverse-roi-reduction-v2", "comparison_rtol": 1e-6,
        "comparison_atol": 0., "division_first_reference_max_absolute_difference": float((expected - original).abs().max()),
        "division_first_reference_differing_elements": int((expected != original).sum()), "objectives": {}}
    for name, function in (("squared", surrogate_logit_loss), ("linear", linear_surrogate_logit_loss)):
        actual = torch.autograd.grad(function(logits, gradient, mask), logits, retain_graph=True)[0].to(dtype)
        difference = actual - expected
        norm = float(actual.norm())
        reference = float(expected.norm())
        result["objectives"][name] = {"max_absolute_error": float(difference.abs().max()),
            "max_relative_error_nonzero": float((difference.abs()[expected != 0] / expected.abs()[expected != 0]).max()) if (expected != 0).any() else 0.0,
            "actual_norm": norm, "norm_ratio": norm / reference if reference else None,
            "cosine_direction": float((actual * expected).sum()) / (norm * reference) if norm and reference else None,
            "matches_expected": bool(torch.allclose(actual, expected, atol=0, rtol=1e-6))}
    if not result["objectives"]["linear"]["matches_expected"]:
        raise RuntimeError(f"LINEAR_SURROGATE_REAL_LOGIT_GRADIENT_FAILED: {result}")
    return result
