"""Regression for a real BF16 audit false alarm, without weakening the gate."""
import pytest
import torch

from src.dimo.surrogate import linear_surrogate_logit_loss
import src.dimo.surrogate_audit as audit


@pytest.mark.parametrize("batch", [1, 3])
def test_bf16_rounding_midpoint_uses_reverse_reduction_reference(batch):
    # This exactly representable FP32 g lands on different sides of a BF16
    # rounding midpoint for g / 7 and g * (1 / 7).
    value = float.fromhex("0x1.3f60000000000p-16")
    z = torch.ones(batch, 8, 2, dtype=torch.bfloat16, requires_grad=True)
    g = torch.full(z.shape, value, dtype=torch.float32, requires_grad=True)
    mask = torch.ones(batch, 8, dtype=torch.bool)
    mask[:, -1] = False
    report = audit.audit_surrogate_gradients(z, g, mask)
    assert report["objectives"]["linear"]["matches_expected"]
    assert report["comparison_rtol"] == 1e-6 and report["comparison_atol"] == 0
    if batch == 1:
        assert report["division_first_reference_differing_elements"] == 14
    actual = torch.autograd.grad(linear_surrogate_logit_loss(z, g, mask), z)[0]
    assert actual[:, -1].eq(0).all() and actual[:, :-1].ne(0).all()
    assert g.grad is None


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_unequal_roi_and_batch_normalization(dtype):
    z = torch.full((3, 7, 5), 10., dtype=dtype, requires_grad=True)
    g = torch.linspace(-1e-8, 1e-8, z.numel(), dtype=torch.float32).reshape_as(z)
    mask = torch.arange(7)[None, :] < torch.tensor([1, 3, 7])[:, None]
    report = audit.audit_surrogate_gradients(z, g, mask)
    assert report["objectives"]["linear"]["matches_expected"]


@pytest.mark.parametrize("broken", ["zero", "vocabulary_mean", "ignore_roi"])
def test_audit_still_rejects_actual_gradient_loss(monkeypatch, broken):
    correct = audit.linear_surrogate_logit_loss
    def wrong(z, g, mask):
        if broken == "zero":
            return z.float().sum() * 0
        if broken == "vocabulary_mean":
            return correct(z, g, mask) / z.shape[-1]
        return correct(z, g, torch.ones_like(mask))
    monkeypatch.setattr(audit, "linear_surrogate_logit_loss", wrong)
    z = torch.ones(2, 7, 5, dtype=torch.bfloat16, requires_grad=True)
    mask = torch.arange(7)[None, :] < torch.tensor([3, 7])[:, None]
    with pytest.raises(RuntimeError, match="LINEAR_SURROGATE_REAL_LOGIT_GRADIENT_FAILED"):
        audit.audit_surrogate_gradients(z, torch.ones_like(z.float()), mask)
