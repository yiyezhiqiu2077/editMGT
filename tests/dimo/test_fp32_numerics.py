import pytest
import torch

from src.dimo.divergence import dimo_divergence_gradient
from src.dimo.surrogate import surrogate_logit_loss


@pytest.mark.parametrize("mode", ["FKL", "RKL", "Jeffreys"])
def test_bf16_divergence_runs_entirely_from_fp32_inputs(mode):
    torch.manual_seed(91)
    teacher = torch.randn(2, 3, 7).bfloat16()
    auxiliary = torch.randn(2, 3, 7).bfloat16()
    mask = torch.tensor([[True, False, True], [True, True, False]])
    actual = dimo_divergence_gradient(teacher, auxiliary, mode=mode, pseudo_mask=mask)
    expected = dimo_divergence_gradient(teacher.float(), auxiliary.float(), mode=mode, pseudo_mask=mask)
    assert actual.dtype == torch.float32
    assert not actual.requires_grad
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_surrogate_gradient_is_manual_g_over_mask_count_and_batch(dtype):
    raw = torch.linspace(-0.5, 0.5, 2 * 3 * 5).reshape(2, 3, 5)
    logits = raw.to(dtype).requires_grad_()
    gradient = torch.linspace(-0.25, 0.25, logits.numel()).reshape_as(raw).to(dtype)
    mask = torch.tensor([[True, False, False], [True, True, False]])
    loss = surrogate_logit_loss(logits, gradient, mask)
    loss.backward()
    expected = torch.zeros_like(raw)
    expected[0, 0] = gradient.float()[0, 0] / 2
    expected[1, :2] = gradient.float()[1, :2] / 4
    assert loss.dtype == torch.float32
    tolerance = 2e-3 if dtype == torch.bfloat16 else 1e-6
    assert torch.allclose(logits.grad.float(), expected, atol=tolerance, rtol=tolerance)
