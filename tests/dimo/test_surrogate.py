import torch

from src.dimo.surrogate import surrogate_logit_loss


def test_surrogate_gradient_is_masked_per_sample_normalized_g():
    logits = torch.randn(2, 3, 5, dtype=torch.float32, requires_grad=True)
    gradient = torch.randn_like(logits)
    mask = torch.tensor([[True, False, False], [True, True, False]])
    loss = surrogate_logit_loss(logits, gradient, mask)
    loss.backward()
    expected = torch.zeros_like(logits)
    expected[0, 0] = gradient[0, 0] / 1 / 2
    expected[1, :2] = gradient[1, :2] / 2 / 2
    assert torch.allclose(logits.grad, expected, atol=1e-6, rtol=1e-6)


def test_surrogate_value_is_finite():
    logits = torch.randn(2, 2, 2, 7, requires_grad=True)
    gradient = torch.randn_like(logits)
    mask = torch.tensor([[[True, False], [False, False]], [[True, True], [True, False]]])
    assert torch.isfinite(surrogate_logit_loss(logits, gradient, mask))
