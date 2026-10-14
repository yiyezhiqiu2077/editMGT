"""Small complete Transformer surrogate for CPU wiring tests, never quality evidence."""
from types import SimpleNamespace
import torch


class SyntheticDenseTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = {"class": "CPU_SYNTHETIC", "vocab_size": 6, "codebook_size": 5}
        self.inner_dim = 5
        self.projection = torch.nn.Linear(6, 5)
        self.edit_region_embedding = torch.nn.Parameter(torch.linspace(.01, .2, 5))

    def forward(self, hidden_states, edit_region_mask, **kwargs):
        x = torch.nn.functional.one_hot(hidden_states, 6).float()
        return self.projection(x) + edit_region_mask[..., None] * self.edit_region_embedding


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, length=512):
        self.rows = [{"sample_uid": f"CPU_SYNTHETIC-{i:06d}"} for i in range(length)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return index

    def set_epoch(self, epoch):
        self.epoch = epoch


def synthetic_batch(indices, dataset):
    source = torch.stack([(torch.arange(16).reshape(4, 4) + int(i)) % 5 for i in indices]).long()
    region = torch.zeros_like(source, dtype=torch.bool)
    region[:, 1:4, 1:4] = True
    return {"source_tokens": source, "edit_region_mask": region,
            "sample_uids": [dataset.rows[int(i)]["sample_uid"] for i in indices],
            "prompt_condition": {"conditional": {}, "unconditional": {}}, "model_kwargs": {}}
