import torch
import torch.nn.functional as F

from src.dimo.auxiliary import auxiliary_loss


def test_hard_ce_is_pseudo_masked_and_per_sample_normalized():
    logits = torch.randn(2, 2, 2, 5, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, 5, (2, 2, 2))
    mask = torch.tensor([[[True, False], [False, False]], [[True, True], [False, False]]])
    loss = auxiliary_loss(logits, targets, mask)
    raw = F.cross_entropy(logits.reshape(-1, 5), targets.reshape(-1), reduction="none").reshape(2, -1)
    expected = torch.stack([raw[0, 0], raw[1, :2].mean()]).mean()
    assert torch.allclose(loss, expected)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.equal(logits.grad[~mask], torch.zeros_like(logits.grad[~mask]))
