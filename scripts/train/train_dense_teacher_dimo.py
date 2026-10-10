#!/usr/bin/env python3
"""Real role-gradient allreduce + complete resume. Prep-only, at most 2 steps.

This is a separate distributed preparation entry, not a formal DiMO launcher.
No mathematical changes to complete_dimo_step. Gradient averaging is installed
before each optimizer update; student and auxiliary remain separately isolated.
"""
import argparse
import functools
import json
import os
from pathlib import Path
import sys
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("NCCL_ALGO", "Ring")
os.environ.setdefault("NCCL_PROTO", "Simple")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from scripts.train.train_dimo_editing import prepare_batch
from scripts.train.train_dense_region import configure_determinism
from src.explicit_region.config import load_config
from src.explicit_region.fixed_dataset import FixedCorpusDataset
from src.explicit_region.fixed_corpus import verify_mini_corpus_ready
from src.explicit_region.epoch_sampler import DeterministicEpochSampler
from src.explicit_region.contracts import sha256_file, freeze_modules
from src.explicit_region.modeling import load_released_components
from src.explicit_region.checkpoint import recipe_fingerprint
from src.explicit_region.dense_checkpoint import write_json
from src.dimo.dense_roles import initialize_teacher_roles, initial_logits_parity
from src.dimo.contracts import released_base_model_identity, teacher_bundle_fingerprint, DIMO_UPSTREAM_COMMIT, DIMO_MODEL_ROLES_V11, build_inference_fingerprint
from src.dimo.ema import TrainableEMA
from src.dimo.roles import audit_role_optimizers
from src.dimo.step import complete_dimo_step
from src.dimo.distributed import ExplicitGradientSync, assert_rank_consistency
from src.dimo.dense_distributed_checkpoint import save_checkpoint, load_checkpoint


def attach_sync(optimizer, sync, role):
    original = optimizer.step
    @functools.wraps(original)
    def step(*args, **kwargs):
        sync(role)
        return original(*args, **kwargs)
    optimizer.step = step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/dimo/dense_teacher_smoke.yaml")
    p.add_argument("--resume"); p.add_argument("--stop-at-global-step", type=int, default=2)
    a = p.parse_args()
    config = load_config(a.config)
    if config.get("formal_ready") or config["max_optimizer_steps"] != 2 or a.stop_at_global_step not in (1, 2):
        raise RuntimeError("DENSE_DIMO_PREP_ONLY_MAX_2_STEPS")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank, local = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    if world not in (1, 8) or not torch.cuda.is_available():
        raise RuntimeError("DENSE_DIMO_REAL_SMOKE_REQUIRES_1_OR_8_GPUS")
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl")
    device = torch.device("cuda", local)
    configure_determinism(); torch.manual_seed(config["seed"])
    output = Path(config["output_dir"])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    components = load_released_components(config["model"]["repo_or_root"], torch_dtype=torch.bfloat16,
                                          transformer_dtype=torch.float32, vq_dtype=torch.float32)
    base = released_base_model_identity(components.identity)
    bundle = teacher_bundle_fingerprint(config["teacher_checkpoint"], base_model_identity=base, formal=False)
    roles = initialize_teacher_roles(components.transformer, config["teacher_checkpoint"], teacher_backend="dense")
    roles.to(device); roles.base_model.enable_gradient_checkpointing()
    freeze_modules([components.text_encoder, components.llm_encoder, components.vqvae])
    components.text_encoder.to(device); components.llm_encoder.to(device); components.vqvae.to(device)
    optimizers = [torch.optim.AdamW([p for _, p in roles.role_named_parameters(role)],
        lr=config["optimizer"][f"{role}_learning_rate"], betas=tuple(config["optimizer"]["betas"]),
        weight_decay=config["optimizer"]["weight_decay"], foreach=False) for role in ("student", "auxiliary")]
    from diffusers.optimization import get_scheduler
    schedulers = [get_scheduler(config["optimizer"]["scheduler"], optimizer=o,
        num_warmup_steps=config["optimizer"]["warmup_steps"], num_training_steps=config["optimizer"]["horizon_steps"]) for o in optimizers]
    ema = TrainableEMA(roles.role_named_parameters("student"), decay=config["ema"]["decay"])
    audit_role_optimizers(roles, *optimizers)
    sync = ExplicitGradientSync(roles)
    for o, role in zip(optimizers, ("student", "auxiliary")):
        attach_sync(o, sync, role)
    inference = build_inference_fingerprint(teacher_bundle_sha256=bundle["bundle_sha256"], base_model_identity=base,
        model_roles=DIMO_MODEL_ROLES_V11, upstream_commit=DIMO_UPSTREAM_COMMIT)
    identity = {"teacher_bundle_fingerprint": bundle, "inference_fingerprint": inference,
                "upstream_commit": DIMO_UPSTREAM_COMMIT, "world_size": world}
    step = cursor = 0
    if a.resume:
        state = load_checkpoint(a.resume, roles=roles, optimizers=optimizers, schedulers=schedulers, ema=ema,
            expected_identity=identity, expected_config_hash=recipe_fingerprint(config))
        step, cursor = state["step"], state["cursor"]
    data = config["data"]
    verify_mini_corpus_ready(data["corpus_ready"], expected_rows=data["expected_rows"])
    dataset = FixedCorpusDataset(data["manifest"], data["roots"], resolution=1024, base_seed=config["seed"], verify_hashes=True)
    sampler = DeterministicEpochSampler(len(dataset), base_seed=config["seed"], epoch=0,
        train_manifest_sha256=sha256_file(data["manifest"]), rank=rank, world_size=world, samples_consumed_in_epoch=cursor)
    loader = DataLoader(dataset, sampler=sampler, batch_size=1, generator=torch.Generator().manual_seed(config["seed"]))
    for batch in loader:
        if step >= a.stop_at_global_step:
            break
        prepared = prepare_batch(batch, components, config, device)
        if step == 0:
            kwargs = dict(prepared["model_kwargs"], **prepared["prompt_condition"]["conditional"],
                hidden_states=prepared["source_tokens"], reference_image_hidden_states=prepared["source_tokens"],
                edit_region_mask=prepared["edit_region_mask"], timestep=torch.ones(1, device=device))
            parity = initial_logits_parity(roles, kwargs, **config["parity"])
            if rank == 0:
                write_json(output / "initial_parity.json", parity)
        diagnostics, _ = complete_dimo_step(roles, **prepared, student_optimizer=optimizers[0],
            auxiliary_optimizer=optimizers[1], student_scheduler=schedulers[0], auxiliary_scheduler=schedulers[1],
            student_ema=ema, base_seed=config["seed"], epoch=0, committed_dimo_step=step,
            mask_token_id=roles.base_model.config.vocab_size - 1, codebook_size=roles.base_model.config.codebook_size, config=config)
        step += 1; cursor += world
        proof = assert_rank_consistency(roles, ema, *optimizers, *schedulers, step)
        save_checkpoint(output / f"checkpoint-{step}", roles=roles, optimizers=optimizers, schedulers=schedulers,
            ema=ema, step=step, cursor=cursor, identity=identity, config_hash=recipe_fingerprint(config))
        if rank == 0:
            write_json(output / "rank_consistency.json", proof)
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(diagnostics | {"formal_ready": False, "teacher_backend": "dense"}) + "\n")
    if step != a.stop_at_global_step:
        raise RuntimeError("DENSE_DIMO_SMOKE_DID_NOT_REACH_STOP")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
