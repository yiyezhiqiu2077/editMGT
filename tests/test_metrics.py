import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from src.explicit_region.metrics import MaskedLPIPS, masked_l1, masked_psnr, masked_ssim
from scripts.eval.formal_eval import evaluate


def test_masked_pixel_metrics_boundary_masks_are_not_nan():
    source=torch.zeros(1,3,32,32);target=source.clone();target[:,:,8:24,8:24]=1
    masks=[torch.ones(1,1,32,32,dtype=torch.bool),torch.zeros(1,1,32,32,dtype=torch.bool),
           torch.nn.functional.pad(torch.ones(1,1,1,1,dtype=torch.bool),(15,16,15,16)),
           torch.rand(1,1,32,32)>0.7]
    for mask in masks:
        for fn in (masked_l1,masked_psnr,masked_ssim):
            value,valid=fn(source,target,mask)
            assert not torch.isnan(value).any()
            assert bool(valid[0]) == bool(mask.any())


def test_formal_evaluator_synthetic_with_feature_masked_lpips(tmp_path):
    rng=np.random.default_rng(4); paths={}
    for name in ("source","target","output"):
        array=(rng.random((64,64,3))*255).astype(np.uint8)
        path=tmp_path/f"{name}.png";Image.fromarray(array).save(path);paths[name]=str(path)
    masks = {
        "all_one": np.full((64, 64), 255, np.uint8),
        "all_zero": np.zeros((64, 64), np.uint8),
        "small": np.pad(np.full((2, 2), 255, np.uint8), ((31, 31), (31, 31))),
        "irregular": (rng.random((64, 64)) > 0.73).astype(np.uint8) * 255,
    }
    manifest=tmp_path/"predictions.jsonl"
    records=[]
    for name,mask in masks.items():
        mask_path=tmp_path/f"mask_{name}.png";Image.fromarray(mask).save(mask_path)
        records.append(paths|{"mask":str(mask_path),"sample_key":name,"dataset_name":"synthetic","edit_type":"Local","seed":0,"runtime_seconds":0.1})
    manifest.write_text("".join(json.dumps(row)+"\n" for row in records))
    config={"selection":{"tau_edit":None,"tau_noop":None},"bootstrap":{"seed":42,"resamples":100}}
    result=evaluate(manifest,config,tmp_path/"metrics.json","cpu")
    by_key={row["sample_key"]:row["metrics"] for row in result["per_generation"]}
    names=("inside_psnr","inside_ssim","inside_masked_lpips","outside_psnr","outside_ssim","outside_masked_lpips","full_lpips_to_target")
    for key,metrics in by_key.items():
        for name in names:
            value=metrics[name]
            assert value is None or np.isfinite(value), (key,name,value)
        assert metrics["no_op"] is None
    assert by_key["all_zero"]["inside_masked_lpips"] is None
    assert by_key["all_one"]["outside_masked_lpips"] is None
    for key in ("small","irregular"):
        assert all(by_key[key][name] is not None for name in names)


def test_masked_lpips_boundary_masks_are_finite():
    generator = torch.Generator().manual_seed(7)
    source = torch.rand(1, 3, 64, 64, generator=generator)
    target = torch.rand(1, 3, 64, 64, generator=generator)
    masks = [
        torch.ones(1, 1, 64, 64, dtype=torch.bool),
        torch.zeros(1, 1, 64, 64, dtype=torch.bool),
        torch.nn.functional.pad(torch.ones(1, 1, 1, 1, dtype=torch.bool), (31, 32, 31, 32)),
        torch.rand(1, 1, 64, 64, generator=generator) > 0.7,
    ]
    metric = MaskedLPIPS(device="cpu")
    for mask in masks:
        value, valid = metric(source, target, mask)
        assert torch.isfinite(value).all()
        assert bool(valid[0]) == bool(mask.any())
