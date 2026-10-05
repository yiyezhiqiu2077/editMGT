import pytest
import torch

from src.dimo.divergence import dimo_divergence_gradient


@pytest.mark.parametrize("seed", [0, 3, 19])
def test_fkl_matches_autograd(seed):
    torch.manual_seed(seed)
    teacher = torch.randn(2, 3, 5, dtype=torch.float64)
    auxiliary = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    teacher_prob = torch.softmax(teacher, -1)
    loss = (teacher_prob * (torch.log_softmax(teacher, -1) - torch.log_softmax(auxiliary, -1))).sum()
    expected, = torch.autograd.grad(loss, auxiliary)
    actual = dimo_divergence_gradient(teacher, auxiliary.detach(), mode="FKL")
    assert torch.allclose(actual.double(), expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("seed", [1, 7, 23])
def test_rkl_matches_autograd(seed):
    torch.manual_seed(seed)
    teacher = torch.randn(2, 3, 5, dtype=torch.float64)
    auxiliary = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    auxiliary_prob = torch.softmax(auxiliary, -1)
    loss = (auxiliary_prob * (torch.log_softmax(auxiliary, -1) - torch.log_softmax(teacher, -1))).sum()
    expected, = torch.autograd.grad(loss, auxiliary)
    actual = dimo_divergence_gradient(teacher, auxiliary.detach(), mode="RKL")
    assert torch.allclose(actual.double(), expected, atol=2e-6, rtol=2e-6)


def test_jeffreys_endpoints_and_mask():
    teacher = torch.randn(2, 3, 5)
    auxiliary = torch.randn(2, 3, 5)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    fkl = dimo_divergence_gradient(teacher, auxiliary, mode="FKL", pseudo_mask=mask)
    rkl = dimo_divergence_gradient(teacher, auxiliary, mode="RKL", pseudo_mask=mask)
    assert torch.equal(
        dimo_divergence_gradient(teacher, auxiliary, mode="Jeffreys", beta=0, pseudo_mask=mask), fkl
    )
    assert torch.equal(
        dimo_divergence_gradient(teacher, auxiliary, mode="Jeffreys", beta=1, pseudo_mask=mask), rkl
    )
    mixed = dimo_divergence_gradient(teacher, auxiliary, mode="Jeffreys", beta=0.3, pseudo_mask=mask)
    assert torch.allclose(mixed, 0.7 * fkl + 0.3 * rkl)
    assert torch.equal(mixed[~mask], torch.zeros_like(mixed[~mask]))
