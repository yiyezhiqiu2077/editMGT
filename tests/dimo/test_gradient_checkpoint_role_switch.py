import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.dimo.divergence import dimo_divergence_gradient
from src.dimo.roles import DiMOModelRoles
from src.dimo.surrogate import surrogate_logit_loss


class CheckpointAdapterModel(nn.Module):
    def __init__(self, checkpointing):
        super().__init__()
        self.vocab = 5
        self.checkpointing = checkpointing
        self.edit_region_embedding = nn.Parameter(torch.linspace(0, 0.2, self.vocab))
        self.adapters = nn.ModuleDict({
            role: nn.Linear(self.vocab, self.vocab, bias=False)
            for role in ("teacher", "student", "auxiliary")
        })
        with torch.no_grad():
            self.adapters["teacher"].weight.copy_(torch.eye(self.vocab))
        self.active_adapter = "teacher"

    def set_adapter(self, name):
        self.active_adapter = name

    def forward(self, hidden_states, edit_region_mask, edit_region_embedding_override, **kwargs):
        encoded = torch.nn.functional.one_hot(hidden_states, self.vocab).float()

        def block(value, region_embedding):
            logits = self.adapters[self.active_adapter](value)
            return logits + edit_region_mask[..., None] * region_embedding

        if self.checkpointing and self.training:
            return checkpoint(
                block, encoded, edit_region_embedding_override, use_reentrant=False
            )
        return block(encoded, edit_region_embedding_override)


def _loss_and_grads(checkpointing):
    torch.manual_seed(3)
    roles = DiMOModelRoles(CheckpointAdapterModel(checkpointing))
    with torch.no_grad():
        for _, parameter in roles.role_named_parameters("auxiliary"):
            parameter.add_(torch.linspace(-0.03, 0.03, parameter.numel()).reshape_as(parameter))
    tokens = torch.arange(16).reshape(1, 4, 4) % 5
    mask = torch.ones_like(tokens, dtype=torch.bool)
    student = roles.forward_student(hidden_states=tokens, edit_region_mask=mask)
    with torch.no_grad():
        teacher = roles.forward_teacher(hidden_states=tokens, edit_region_mask=mask)
        auxiliary = roles.forward_aux(hidden_states=tokens, edit_region_mask=mask)
        gradient = dimo_divergence_gradient(teacher, auxiliary, pseudo_mask=mask)
    loss = surrogate_logit_loss(student, gradient, mask)
    roles.activate_adapter("student")
    loss.backward()
    grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in roles.role_named_parameters("student")
    }
    assert all(parameter.grad is None for _, parameter in roles.role_named_parameters("teacher"))
    assert all(parameter.grad is None for _, parameter in roles.role_named_parameters("auxiliary"))
    return loss.detach(), grads


def test_gradient_checkpointing_role_switch_matches_disabled_path():
    off_loss, off_grads = _loss_and_grads(False)
    on_loss, on_grads = _loss_and_grads(True)
    assert torch.equal(off_loss, on_loss)
    assert off_grads.keys() == on_grads.keys()
    for name in off_grads:
        assert torch.equal(off_grads[name], on_grads[name])
