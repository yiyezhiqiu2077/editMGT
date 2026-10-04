"""Frozen validation generation/evaluation without advancing training state."""
from __future__ import annotations
import json, random, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from src.explicit_region.config import load_config
from src.explicit_region.dataset import MagicBrushAlignedDataset, InterEditArchiveDataset
from src.explicit_region.fixed_dataset import CanonicalAlignedDataset
from src.pipeline import Pipeline


def _pil(tensor):
    array=(tensor.cpu().numpy().transpose(1,2,0)*255).round().clip(0,255).astype(np.uint8)
    return Image.fromarray(array)


def run_validation(*, components, transformer, train_config, step, output_dir, device):
    validation=train_config["validation"]; infer=load_config(validation["inference_config"])
    rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all(),random.getstate(),np.random.get_state())
    root=Path(output_dir)/f"validation-{step}";root.mkdir(parents=True,exist_ok=True)
    pipe=Pipeline(transformer=transformer,tokenizer=components.tokenizer,text_encoder=components.text_encoder,
                  vqvae=components.vqvae,scheduler=components.scheduler,tokenizer_t5=components.llm_tokenizer,
                  text_encoder_t5=components.llm_encoder)
    pipe.set_progress_bar_config(disable=True);transformer.eval(); records=[]
    if validation.get("protocol") == "fixed200k_magicbrush_probe":
        rows=[json.loads(line) for line in Path(validation["manifest"]).open(encoding="utf-8") if line.strip()]
        datasets=[("magicbrush",CanonicalAlignedDataset(
            rows,validation["dataset_root"],resolution=infer["resolution"],
            base_seed=train_config["seed"],verify_hashes=True,
        ))]
    else:
        datasets=[("magicbrush",MagicBrushAlignedDataset(validation["magicbrush_manifest"],resolution=infer["resolution"],base_seed=train_config["seed"]))]
        datasets.append(("interedit",InterEditArchiveDataset(validation["interedit_manifest"],train_config["data"]["interedit_root"],resolution=infer["resolution"],base_seed=train_config["seed"])))
    try:
        for dataset_name,dataset in datasets:
            for index in range(len(dataset)):
                item=dataset[index];source=_pil(item["source_image"]);target=_pil(item["target_image"]);mask=Image.fromarray(item["edit_region_mask"].numpy().astype(np.uint8)*255)
                for seed in infer["generation_seeds"]:
                    generator=torch.Generator(device=device).manual_seed(seed);started=time.perf_counter()
                    generated=pipe(prompt=item["instruction_en"],reference_image=source,mask_image=mask,
                      height=infer["resolution"],width=infer["resolution"],num_inference_steps=infer["steps"],
                      guidance_scale=infer["guidance_scale"],reference_strength=infer["reference_strength"],
                      generator=generator,lora_scope=train_config["lora"]["scope"],
                      persistent_conditioning=train_config["corruption"]["persistent_conditioning"],
                      inference_timestep_mode=infer["inference_timestep_mode"]).images[0]
                    stem=f"{dataset_name}_{index:06d}_seed{seed}"; paths={}
                    for name,image in (("source",source),("target",target),("mask",mask),("output",generated)):
                        path=root/f"{stem}_{name}.png";image.save(path);paths[name]=str(path)
                    records.append(paths|{"sample_key":item["sample_key"],"dataset_name":dataset_name,"edit_type":item["edit_type"],"seed":seed,"runtime_seconds":time.perf_counter()-started})
        manifest=root/"predictions.jsonl";manifest.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in records),encoding="utf-8")
        from scripts.eval.formal_eval import evaluate
        result=evaluate(manifest,infer,root/"validation_metrics.json",str(device))
        result["validation_wall_seconds"] = sum(record["runtime_seconds"] for record in records)
        (root/"validation_metrics.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    finally:
        torch.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1]);random.setstate(rng[2]);np.random.set_state(rng[3]);transformer.train()
        components.text_encoder.eval();components.llm_encoder.eval();components.vqvae.eval()
    return result
