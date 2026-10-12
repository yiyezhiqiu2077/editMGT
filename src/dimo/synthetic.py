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


class SyntheticDenseTransformerV2(torch.nn.Module):
    """Exercise actual frozen modules, text CFG, and ROI noise on CPU."""
    def __init__(self):
        super().__init__()
        self.config = {"class": "CPU_SYNTHETIC_V2", "vocab_size": 6, "codebook_size": 5}
        self.inner_dim = 5
        self.embed = torch.nn.Sequential(torch.nn.Embedding(6, 5), torch.nn.LayerNorm(5))
        block = torch.nn.Module()
        block.attn = torch.nn.Linear(5, 5)
        block.ff = torch.nn.Sequential(torch.nn.Linear(5, 9), torch.nn.GELU(), torch.nn.Linear(9, 5))
        self.transformer_blocks = torch.nn.ModuleList([block])
        self.mlm_layer = torch.nn.Sequential(torch.nn.LayerNorm(5), torch.nn.Linear(5, 5))
        self.edit_region_embedding = torch.nn.Parameter(torch.linspace(.01, .2, 5))

    def forward(self, hidden_states, edit_region_mask, encoder_hidden_states=None,
                target_embedding_noise=None, target_embedding_noise_sigma=0., **kwargs):
        x = self.embed(hidden_states)
        region = edit_region_mask[..., None]
        if target_embedding_noise is not None:
            x = x + target_embedding_noise.permute(0, 2, 3, 1) * target_embedding_noise_sigma * region
        x = x + self.edit_region_embedding * region
        if encoder_hidden_states is not None:
            x = x + encoder_hidden_states[:, None, None, :]
        for block in self.transformer_blocks:
            x = x + torch.tanh(block.attn(x))
            x = x + block.ff(x)
        return self.mlm_layer(x)


def synthetic_v2_batch(indices, dataset):
    batch = synthetic_batch(indices, dataset)
    values = torch.stack([torch.linspace(.05, .4, 5) * (1 + int(i) % 3) for i in indices])
    batch["prompt_condition"] = {"conditional": {"encoder_hidden_states": values},
                                 "unconditional": {"encoder_hidden_states": torch.zeros_like(values)}}
    return batch
