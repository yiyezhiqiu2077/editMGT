import torch
from torch import nn

from src.dimo.roles import DiMOModelRoles, audit_role_optimizers


class ToyAdapterModel(nn.Module):
    def __init__(self, vocab=5):
        super().__init__()
        self.vocab = vocab
        self.inner_dim = vocab
        self.edit_region_embedding = nn.Parameter(torch.linspace(0.0, 0.4, vocab))
        self.adapters = nn.ModuleDict({
            role: nn.Linear(vocab, vocab, bias=False)
            for role in ("teacher", "student", "auxiliary")
        })
        with torch.no_grad():
            self.adapters["teacher"].weight.copy_(torch.eye(vocab))
            self.adapters["student"].weight.zero_()
            self.adapters["auxiliary"].weight.fill_(2)
        self.active_adapter = "teacher"

    def set_adapter(self, name):
        self.active_adapter = name

    def forward(self, hidden_states, edit_region_mask, edit_region_embedding_override, **kwargs):
        one_hot = torch.nn.functional.one_hot(hidden_states, self.vocab).float()
        logits = self.adapters[self.active_adapter](one_hot)
        return logits + edit_region_mask[..., None] * edit_region_embedding_override


def make_roles():
    return DiMOModelRoles(ToyAdapterModel())


def test_role_initialization_and_trainability():
    roles = make_roles()
    teacher = roles.role_state_dict("teacher")
    student = roles.role_state_dict("student")
    auxiliary = roles.role_state_dict("auxiliary")
    assert teacher.keys() == student.keys() == auxiliary.keys()
    for key in teacher:
        assert torch.equal(teacher[key], student[key])
        assert torch.equal(teacher[key], auxiliary[key])
    assert not any(p.requires_grad for _, p in roles.role_named_parameters("teacher"))
    assert all(p.requires_grad for _, p in roles.role_named_parameters("student"))
    assert all(p.requires_grad for _, p in roles.role_named_parameters("auxiliary"))


def test_student_and_auxiliary_gradients_are_isolated(tmp_path):
    roles = make_roles()
    student_params = [p for _, p in roles.role_named_parameters("student")]
    aux_params = [p for _, p in roles.role_named_parameters("auxiliary")]
    student_opt = torch.optim.SGD(student_params, lr=0.1)
    aux_opt = torch.optim.SGD(aux_params, lr=0.1)
    report = audit_role_optimizers(roles, student_opt, aux_opt, tmp_path / "report.json")
    assert report["optimizer_overlap"] == 0
    tokens = torch.tensor([[[0, 1], [2, 3]]])
    region = torch.ones_like(tokens, dtype=torch.bool)

    aux_before = {k: v.clone() for k, v in roles.role_state_dict("auxiliary").items()}
    roles.forward_student(hidden_states=tokens, edit_region_mask=region).sum().backward()
    assert sum(float(p.grad.abs().sum()) for p in student_params if p.grad is not None) > 0
    assert all(p.grad is None for p in aux_params)
    assert all(p.grad is None for _, p in roles.role_named_parameters("teacher"))
    student_opt.step(); student_opt.zero_grad(set_to_none=True)
    assert all(torch.equal(aux_before[k], roles.role_state_dict("auxiliary")[k]) for k in aux_before)

    student_before = {k: v.clone() for k, v in roles.role_state_dict("student").items()}
    roles.forward_aux(hidden_states=tokens, edit_region_mask=region).sum().backward()
    assert sum(float(p.grad.abs().sum()) for p in aux_params if p.grad is not None) > 0
    assert all(p.grad is None for p in student_params)
    assert all(p.grad is None for _, p in roles.role_named_parameters("teacher"))
    aux_opt.step()
    assert all(torch.equal(student_before[k], roles.role_state_dict("student")[k]) for k in student_before)
