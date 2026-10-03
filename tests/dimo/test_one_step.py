import torch

from src.dimo.one_step import one_step_edit_tokens


class CountingRoles:
    def __init__(self, vocab):
        self.vocab = vocab
        self.calls = 0

    def forward_role(self, role, hidden_states, **kwargs):
        assert role == "student"
        self.calls += 1
        return torch.nn.functional.one_hot((hidden_states + 1) % self.vocab, self.vocab).float() * 20


def test_exactly_one_student_forward_and_final_outside_lock():
    source = torch.arange(16).reshape(1, 4, 4) % 7
    region = torch.zeros_like(source, dtype=torch.bool)
    region[:, 1:3, 1:3] = True
    roles = CountingRoles(7)
    output = one_step_edit_tokens(
        roles, source_tokens=source, edit_region_mask=region,
        prompt_condition={"conditional": {}, "unconditional": {}},
        timestep_model_kwargs={}, mask_token_id=8, codebook_size=7, r_init=0.5,
        init_mask_seeds=1, init_token_seeds=2, sample_seeds=3,
    )
    assert roles.calls == 1
    assert output.metadata["number_of_transformer_forwards"] == 1
    assert torch.equal(output.tokens[~region], source[~region])


def test_cfg_is_batched_into_one_student_forward():
    source = torch.arange(16).reshape(1, 4, 4) % 7
    region = torch.ones_like(source, dtype=torch.bool)
    roles = CountingRoles(7)
    condition = torch.zeros(1, 2, 3)
    output = one_step_edit_tokens(
        roles, source_tokens=source, edit_region_mask=region,
        prompt_condition={
            "conditional": {"encoder_hidden_states": condition + 1},
            "unconditional": {"encoder_hidden_states": condition},
        },
        timestep_model_kwargs={}, mask_token_id=8, codebook_size=7, r_init=0.5,
        init_mask_seeds=1, init_token_seeds=2, sample_seeds=3, cfg_scale=4,
    )
    assert roles.calls == 1
    assert output.metadata["number_of_transformer_forwards"] == 1
