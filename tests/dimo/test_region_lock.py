import pytest
import torch

from src.dimo.forward_process import mask_student_prediction
from src.dimo.initialization import sample_student_tokens


def test_all_token_states_enforce_outside_lock():
    source = torch.arange(9).reshape(1, 3, 3)
    region = torch.tensor([[[False, False, False], [False, True, True], [False, True, True]]])
    logits = torch.randn(1, 3, 3, 12)
    sampled = sample_student_tokens(logits, source, region, seeds=3)
    pseudo = mask_student_prediction(sampled, source, region, 0.5, 99, seeds=4)
    assert torch.equal(sampled[~region], source[~region])
    assert torch.equal(pseudo.pseudo_tokens[~region], source[~region])


def test_invalid_shapes_fail_instead_of_silently_unlocking():
    with pytest.raises(ValueError):
        sample_student_tokens(torch.randn(1, 2, 2, 4), torch.zeros(1, 3, 3, dtype=torch.long), torch.ones(1, 3, 3), seeds=1)
