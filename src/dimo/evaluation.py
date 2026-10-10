"""Fixed canonical DEV generation using one selected full Dense role."""
import json
from pathlib import Path

import torch

from src.explicit_region.contracts import sha256_file
from src.explicit_region.fixed_dataset import CanonicalAlignedDataset
from src.explicit_region.dense_inference import pil
from src.explicit_region.group_statistics import audited_clusters, read_rows
from src.explicit_region.latency import repeated_latency
from .one_step import one_step_edit_tokens
from .rng import stable_seed, normal_noise_per_sample


@torch.inference_mode()
def generate_one_step_evaluation(components, roles, *, manifest, canonical_root, dataset_name,
        config, output_dir, group_contract, device="cuda", count=None):
    from scripts.train.train_dimo_editing import prepare_batch
    rows = read_rows(manifest)
    if any("test" in str(row.get("split", "")).lower() for row in rows):
        raise RuntimeError("DIMO_DEV_MANIFEST_CONTAINS_TEST")
    clusters = audited_clusters(rows, group_contract)
    dataset = CanonicalAlignedDataset(rows, canonical_root, resolution=config["resolution"], base_seed=42,
        minimum_mask_retention=.75, max_random_attempts=8, verify_hashes=True)
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("DIMO_EVALUATION_OUTPUT_EXISTS")
    out.mkdir(parents=True, exist_ok=True)
    predictions = []
    prepare_config = {"resolution": config["resolution"], "token_mask": {
        "mode": "any_overlap", "coverage_threshold": .5, "dilation_tokens": 0, "minimum_edit_tokens": 1}}
    for i in range(min(len(dataset), count if count is not None else len(dataset))):
        item, row = dataset[i], rows[i]
        source, target = pil(item["source_image"]), pil(item["target_image"])
        from PIL import Image
        import numpy as np
        mask_image = Image.fromarray(item["edit_region_mask"].numpy().astype(np.uint8)*255)
        for seed in config["generation_seeds"]:
            batch = {"source_image": item["source_image"][None], "edit_region_mask": item["edit_region_mask"][None],
                     "instruction_en": [item["instruction_en"]], "sample_uid": [row["sample_uid"]]}
            seed_args = (seed, 0, row["sample_uid"], 0)
            def operation():
                prepared = prepare_batch(batch, components, prepare_config, torch.device(device))
                tokens = prepared["source_tokens"]
                noise = normal_noise_per_sample(torch.empty(tokens.shape[0], roles.base_model.inner_dim,
                    *tokens.shape[1:], device=device, dtype=torch.float32), [stable_seed(*seed_args, "dimo-embedding-noise")])
                generated = one_step_edit_tokens(roles, source_tokens=tokens,
                    edit_region_mask=prepared["edit_region_mask"], prompt_condition=prepared["prompt_condition"],
                    timestep_model_kwargs=prepared["model_kwargs"], mask_token_id=roles.base_model.config.vocab_size-1,
                    codebook_size=roles.base_model.config.codebook_size, r_init=.5,
                    init_mask_seeds=stable_seed(*seed_args, "dimo-init-mask"),
                    init_token_seeds=stable_seed(*seed_args, "dimo-init-token"),
                    sample_seeds=stable_seed(*seed_args, "dimo-student-sample"),
                    temperature=1., top_k=0, top_p=0., cfg_scale=1., target_embedding_noise=noise, embedding_noise_sigma=.3)
                decoded = components.vqvae.decode(generated.tokens, force_not_quantize=True,
                    shape=(*tokens.shape, components.vqvae.config.latent_channels)).sample.clip(0, 1)
                if not torch.isfinite(decoded).all():
                    raise FloatingPointError("DIMO_NONFINITE_INFERENCE_DECODE")
                return pil(decoded[0].float().cpu())
            result, latency = repeated_latency(operation, {"transformer": (roles.base_model, "forward"),
                "vq_encode": (components.vqvae, "encode"), "vq_decode": (components.vqvae, "decode"),
                "clip": (components.text_encoder, "forward"), "gemma": (components.llm_encoder, "forward")},
                device, **config["latency"])
            if any(p["transformer"]["calls"] != 1 for p in latency["phases"]):
                raise RuntimeError("DIMO_ONE_STEP_FORWARD_COUNT_VIOLATION")
            stem = f"{dataset_name}_{i:06d}_seed{seed}"
            paths = {}
            for name, image in (("source", source), ("target", target), ("mask", mask_image), ("output", result)):
                path = out / f"{stem}_{name}.png"; image.save(path); paths[name] = str(path.resolve())
            predictions.append(paths | {"dataset_name": dataset_name, "sample_key": item["sample_key"],
                "sample_uid": row["sample_uid"], "seed": seed, "group_id": row["group_id"],
                "cluster_id": clusters[row["sample_uid"]], "source_sha256": row["source_sha256"],
                "target_sha256": row["target_sha256"], "region_sha256": row["region_sha256"],
                "geometry": item["geometry"], "manifest_sha256": sha256_file(manifest),
                "runtime_seconds": latency["mean_seconds"], "latency_protocol": latency,
                "transformer_forward_count": 1, "token_outside_lock": "PASS", "pixel_outside_lock": "NOT_GUARANTEED",
                "inference_protocol": {"cfg": 1., "steps": 1, "temperature": 1., "r_init": .5, "sigma": .3}})
    path = out / "predictions.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True)+"\n" for row in predictions))
    return path
