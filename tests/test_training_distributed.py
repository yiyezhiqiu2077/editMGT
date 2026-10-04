"""Two CPU/Gloo ranks exercise DDP and collective per-rank checkpoint replay."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import random
import socket

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset


class DistributedTiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.ones(4))
        self.edit_region_embedding = torch.nn.Parameter(torch.zeros(4))
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, value, active):
        result = self.dropout(value) * self.lora_weight
        return result + self.edit_region_embedding if active else result


class RankRows(Dataset):
    def __init__(self, rank, start=0, stop=24):
        self.indices = list(range(rank + start, stop, 2))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return torch.full((4,), (1 + self.indices[index]) / 24)


def gloo_worker(rank, rendezvous, output):
    # No GPU context, large model, or pretrained asset is used in this worker.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    from scripts.train.train_explicit_region import loader_generator, resume_training_iterator
    from scripts.train.verify_fixed200k_smoke import assert_exact
    from src.explicit_region import checkpoint
    from src.explicit_region.contracts import audit_trainable_parameters, configure_region_trainability

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=120))
    try:
        # Two iterations are necessary: DDP reports missing reductions from an
        # inactive trainable vector at the *next* forward, not the first one.
        for mode, persistent in (("full_target", False), ("roi_hardlock", False), ("roi_hardlock", True)):
            model = DistributedTiny()
            active = configure_region_trainability(model, {"mode": mode, "persistent_conditioning": persistent})
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
            audit_trainable_parameters({"transformer": model}, optimizer, region_conditioning_active=active)
            ddp = DistributedDataParallel(model, find_unused_parameters=False)
            for _ in range(3):
                ddp(torch.ones(4), active).sum().backward()
                assert (model.edit_region_embedding.grad is not None) is active
                optimizer.step(); optimizer.zero_grad()

        checkpoint.get_peft_model_state_dict = lambda model: {"lora_weight": model.lora_weight}
        def set_lora(model, state):
            with torch.no_grad():
                model.lora_weight.copy_(state["lora_weight"])
        checkpoint.set_peft_model_state_dict = set_lora
        payload = {"test": "two-rank-replay", "world_size": 2}
        directory = Path(output) / "checkpoint-3"

        def setup(start):
            model = DistributedTiny()
            ddp = DistributedDataParallel(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, foreach=False)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, step / 4))
            loader = DataLoader(RankRows(rank, start), batch_size=1, num_workers=0,
                                generator=loader_generator(42, rank, 0))
            return model, ddp, optimizer, scheduler, loader

        def step(ddp, optimizer, scheduler, batch):
            # Rank-specific Python/NumPy draws influence actual dropout gradients.
            loss = ddp(batch, True).square().mean() * (1 + random.random() + np.random.random())
            loss.backward(); optimizer.step(); scheduler.step(); optimizer.zero_grad()

        model, ddp, optimizer, scheduler, loader = setup(0)
        random.seed(100 + rank); np.random.seed(200 + rank); torch.manual_seed(300 + rank)
        for index, batch in enumerate(resume_training_iterator(loader), 1):
            step(ddp, optimizer, scheduler, batch)
            if index == 3:
                checkpoint.save_resume_state(directory, model=model, optimizer=optimizer, scheduler=scheduler,
                                             global_optimizer_step=3, committed_global_sample_count=6,
                                             sampler_schema="test", fingerprint_payload=payload,
                                             resolved_config={"fixture": True})
        expected = deepcopy((model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), checkpoint.capture_rank_rng_state()))
        model, ddp, optimizer, scheduler, loader = setup(6)
        # Prove that restore overwrites rank-local startup draws, not only rank 0.
        random.random(); np.random.random(11); torch.rand(17)
        state = checkpoint.load_resume_state(directory, model=model, optimizer=optimizer, scheduler=scheduler,
                                               fingerprint_payload=payload, restore_rng=False)
        assert not torch.equal(state["rank_rng"]["states"]["0"]["torch_cpu"], state["rank_rng"]["states"]["1"]["torch_cpu"])
        for batch in resume_training_iterator(loader, state):
            step(ddp, optimizer, scheduler, batch)
        actual = (model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), checkpoint.capture_rank_rng_state())
        assert_exact(expected, actual)
        (Path(output) / f"rank-{rank}.passed").write_text("DDP inactive E1/E2 + exact collective RNG/optimizer replay\n")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_two_process_gloo_inactive_region_and_rng_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    # RUN_ROOT may be an object-backed mount without flock/FileStore support.
    # A loopback TCP rendezvous avoids placing any temp file outside RUN_ROOT.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(gloo_worker, args=(f"tcp://127.0.0.1:{port}", str(tmp_path)), nprocs=2, join=True)
    assert (tmp_path / "rank-0.passed").is_file()
    assert (tmp_path / "rank-1.passed").is_file()
