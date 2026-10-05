import torch
from torch import nn

from src.dimo.one_step import one_step_edit_tokens
from src.dimo.roles import DiMOModelRoles


class ModeRecordingModel(nn.Module):
    def __init__(self, vocab=7):
        super().__init__()
        self.vocab = vocab
        self.edit_region_embedding = nn.Parameter(torch.zeros(vocab))
        self.adapters = nn.ModuleDict({
            role: nn.Linear(vocab + 1, vocab, bias=False)
            for role in ("teacher", "student", "auxiliary")
        })
        self.active_adapter = "teacher"
        self.modes = []

    def set_adapter(self, name):
        self.active_adapter = name

    def forward(self, hidden_states, edit_region_mask, edit_region_embedding_override, **kwargs):
        self.modes.append((self.active_adapter, self.training, torch.is_inference_mode_enabled()))
        encoded = torch.nn.functional.one_hot(hidden_states, self.vocab + 1).float()
        encoded = torch.nn.functional.dropout(encoded, p=0.8, training=self.training)
        return self.adapters[self.active_adapter](encoded)


def _run(roles):
    source = torch.arange(16).reshape(1, 4, 4) % 7
    region = torch.ones_like(source, dtype=torch.bool)
    return one_step_edit_tokens(
        roles, source_tokens=source, edit_region_mask=region,
        prompt_condition={"conditional": {}, "unconditional": {}},
        timestep_model_kwargs={}, mask_token_id=7, codebook_size=7, r_init=0.5,
        init_mask_seeds=1, init_token_seeds=2, sample_seeds=3,
    )


def test_one_step_forces_student_eval_and_inference_mode_deterministically():
    roles = DiMOModelRoles(ModeRecordingModel())
    roles.eval()
    first = _run(roles)
    second = _run(roles)
    assert torch.equal(first.tokens, second.tokens)
    assert torch.equal(first.logits, second.logits)
    assert roles.base_model.training is False
    assert all(role == "student" and not training and inference for role, training, inference in roles.base_model.modes)


def test_role_training_defaults_and_explicit_override():
    roles = DiMOModelRoles(ModeRecordingModel())
    tokens = torch.zeros(1, 2, 2, dtype=torch.long)
    region = torch.ones_like(tokens, dtype=torch.bool)
    roles.forward_student(hidden_states=tokens, edit_region_mask=region)
    roles.forward_aux(hidden_states=tokens, edit_region_mask=region)
    roles.forward_teacher(hidden_states=tokens, edit_region_mask=region)
    roles.forward_student(hidden_states=tokens, edit_region_mask=region, training=False)
    assert [item[1] for item in roles.base_model.modes] == [True, True, False, False]
