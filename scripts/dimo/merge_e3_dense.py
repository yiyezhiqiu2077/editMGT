#!/usr/bin/env python3
"""FP32 PEFT merge with preregistered real-image dual-branch CUDA parity."""
import argparse
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from safetensors import safe_open
from src.explicit_region.contracts import sha256_file
from src.explicit_region.dense_checkpoint import save_dense_checkpoint, load_dense_weights, write_json
from src.explicit_region.dense_inference import load_evaluation_pipeline, inference_safety
from src.explicit_region.checkpoint import recipe_fingerprint
from scripts.train.train_dense_region import configure_determinism
from scripts.train.train_dimo_editing import prepare_batch
from src.dimo.conditioning import predict_role_logits

TOLERANCES = {
    'fp32': {'maximum_absolute_error': 0.001, 'relative_rms_error': 0.00002},
    'bf16_autocast': {'maximum_absolute_error': 0.75, 'relative_rms_error': 0.01},
    'image_quality': {'maximum_mean_inside_lpips_degradation': 0.01},
}
ROUNDING_TOLERANCES = {
    'fp32': {'maximum_absolute_error': 0.05, 'relative_rms_error': 0.001},
    'bf16_autocast': {'maximum_absolute_error': 3.0, 'relative_rms_error': 0.075},
    'image_quality': {'maximum_mean_inside_lpips_degradation': 0.01},
}


def compare_logits(before, after, mode):
    a, b = before.float(), after.float()
    delta = a - b
    result = {'maximum_absolute_error': float(delta.abs().max()),
              'relative_rms_error': float(delta.norm() / a.norm().clamp_min(1e-12)),
              'greedy_token_agreement': float((a.argmax(-1) == b.argmax(-1)).float().mean()),
              'finite': bool(torch.isfinite(a).all() and torch.isfinite(b).all())}
    result['pass'] = result['finite'] and all(result[k] <= v for k, v in TOLERANCES[mode].items())
    return result


class Forward:
    def __init__(self, model, autocast):
        self.model, self.autocast = model, autocast

    def forward_role(self, role, training=None, **kwargs):
        kwargs['lora_scope'] = 'both'
        context = torch.autocast('cuda', dtype=torch.bfloat16) if self.autocast else nullcontext()
        with context:
            return self.model(**kwargs)


def inputs(pipe, row):
    def tensor(path, mask=False):
        a = np.array(Image.open(path).convert('L' if mask else 'RGB'), copy=True)
        return torch.from_numpy(a > 0) if mask else torch.from_numpy(a).permute(2, 0, 1).float() / 255
    batch = {'source_image': tensor(row['source'])[None], 'edit_region_mask': tensor(row['mask'], True)[None],
             'instruction_en': [row['instruction_en']], 'sample_uid': [row['sample_uid']]}
    from types import SimpleNamespace
    c = SimpleNamespace(vqvae=pipe.vqvae, tokenizer=pipe.tokenizer, llm_tokenizer=pipe.tokenizer_t5,
                        text_encoder=pipe.text_encoder, llm_encoder=pipe.text_encoder_t5)
    return prepare_batch(batch, c, {'resolution': 1024, 'token_mask': {
        'mode': 'any_overlap', 'coverage_threshold': .5, 'dilation_tokens': 0, 'minimum_edit_tokens': 1}}, torch.device('cuda'))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-root', required=True); p.add_argument('--lora', type=Path, required=True)
    p.add_argument('--dev-manifest', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--rounding-aware-holdout', action='store_true')
    a = p.parse_args()
    if a.output.exists() or a.evidence.exists():
        raise FileExistsError('Merge outputs must be new')
    if a.rounding_aware_holdout:
        # The original trial is preserved. This profile must use unseen cases
        # and a separately preregistered protocol, never silently accept it.
        TOLERANCES.clear(); TOLERANCES.update(ROUNDING_TOLERANCES)
    a.evidence.mkdir(parents=True)
    write_json(a.evidence / 'PARITY_PREREGISTRATION.json', {'tolerances': TOLERANCES,
               'script_sha256': sha256_file(__file__), 'greedy_vocabulary_axis': -1,
               'case_offset': 4 if a.rounding_aware_holdout else 0,
               'protocol': 'rounding-aware-holdout-v2' if a.rounding_aware_holdout else 'strict-initial-v1'})
    configure_determinism(); torch.set_num_threads(4)
    recipe = json.loads((a.lora / 'trainable_config.json').read_text())
    if recipe['lora']['scope'] != 'both':
        raise RuntimeError('STATIC_MERGE_REQUIRES_BOTH_BRANCH_SCOPE')
    artifacts = {n: sha256_file(a.lora / n) for n in ('adapter_model.safetensors', 'mask_conditioning.safetensors',
                 'trainable_config.json', 'fingerprint.json', 'mask_conditioning_config.json')}
    storage = {}
    for name in ('adapter_model.safetensors', 'mask_conditioning.safetensors'):
        with safe_open(a.lora / name, framework='pt') as h:
            storage[name] = {k: {'shape': h.get_slice(k).get_shape(), 'dtype': h.get_slice(k).get_dtype()} for k in h.keys()}
    pipe = load_evaluation_pipeline(a.model_root, backend='lora', checkpoint=a.lora)
    model = pipe.transformer
    # Confirm that PEFT installed every published adapter tensor, including shape.
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    published = load_file(a.lora / 'adapter_model.safetensors')
    loaded = get_peft_model_state_dict(model)
    if set(published) != set(loaded) or any(not torch.equal(v.float(), loaded[k].cpu().float()) for k, v in published.items()):
        raise RuntimeError('MERGE_ADAPTER_KEYS_OR_VALUES_MISMATCH')
    if not torch.equal(load_file(a.lora / 'mask_conditioning.safetensors')['edit_region_embedding'].float(), model.edit_region_embedding.cpu()):
        raise RuntimeError('MERGE_REGION_MISMATCH')
    rows = [json.loads(l) for l in a.dev_manifest.read_text().splitlines()]
    offset = 4 if a.rounding_aware_holdout else 0
    rows = rows[offset:offset+4]
    if len(rows) != 4: raise RuntimeError('MERGE_REQUIRES_FOUR_FIXED_CASES')
    calls, images = [], []
    for case, row in enumerate(rows):
        batch = inputs(pipe, row)
        # Vary ROI masking, timestep, reference image and CFG. All calls contain both branches.
        for ratio, cfg in ((.25, 1.), (1., 5.)):
            source, roi = batch['source_tokens'], batch['edit_region_mask']
            target = source.clone()
            random = torch.rand(source.shape, device='cuda', generator=torch.Generator('cuda').manual_seed(case))
            selected = roi & (random < ratio)
            target[selected] = 8255
            for mode in ('fp32', 'bf16_autocast'):
                condition = {k: {n: v.float() for n, v in sub.items()} for k, sub in batch['prompt_condition'].items()}
                kwargs = dict(target_tokens=target, reference_tokens=source, prompt_condition=condition,
                              region_mask=roi, timestep=torch.tensor([ratio], device='cuda'), cfg_scale=cfg,
                              model_kwargs={k: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                                            for k, v in batch['model_kwargs'].items()}, training=False)
                logits = predict_role_logits(Forward(model, mode != 'fp32'), 'teacher', **kwargs)
                calls.append((case, ratio, cfg, mode, kwargs, logits.cpu()))
        with inference_safety(pipe):
            image = pipe(prompt=row['instruction_en'], reference_image=Image.open(row['source']).convert('RGB'),
                         mask_image=Image.open(row['mask']).convert('L'), height=1024, width=1024,
                         num_inference_steps=12, guidance_scale=5., reference_strength=1.,
                         generator=torch.Generator('cuda').manual_seed(0), lora_scope='both',
                         inference_timestep_mode='roi_relative').images[0]
        image.save(a.evidence / f'case{case}_lora.png'); images.append(image)
    model.fuse_lora(safe_fusing=True)
    model.unload_lora()
    if any('lora_' in k or '.base_layer.' in k for k in model.state_dict()) or any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError('MERGE_DID_NOT_PRODUCE_FULL_FP32_DENSE')
    results = []
    for case, ratio, cfg, mode, kwargs, before in calls:
        after = predict_role_logits(Forward(model, mode != 'fp32'), 'teacher', **kwargs).cpu()
        result = compare_logits(before, after, mode)
        results.append(dict(case=case, mask_ratio=ratio, cfg=cfg, mode=mode, **result))
        print(json.dumps(results[-1]), flush=True)
    from src.explicit_region.metrics import MaskedLPIPS
    metric = MaskedLPIPS(device='cuda')
    quality = []
    for case, row in enumerate(rows):
        with inference_safety(pipe):
            image = pipe(prompt=row['instruction_en'], reference_image=Image.open(row['source']).convert('RGB'),
                         mask_image=Image.open(row['mask']).convert('L'), height=1024, width=1024,
                         num_inference_steps=12, guidance_scale=5., reference_strength=1.,
                         generator=torch.Generator('cuda').manual_seed(0), lora_scope='both',
                         inference_timestep_mode='roi_relative').images[0]
        image.save(a.evidence / f'case{case}_dense.png')
        def t(im): return torch.from_numpy(np.array(im, copy=True)).permute(2, 0, 1).float()[None].to('cuda') / 255
        mask = torch.from_numpy(np.array(Image.open(row['mask'])) > 0)[None, None].to('cuda')
        target = t(Image.open(row['target']).convert('RGB'))
        lo_value, lo_valid = metric(t(images[case]), target, mask)
        de_value, de_valid = metric(t(image), target, mask)
        if not bool(lo_valid.all() and de_valid.all()): raise RuntimeError('MERGE_EMPTY_METRIC_ROI')
        lo, de = float(lo_value.mean()), float(de_value.mean())
        quality.append({'case': case, 'lora_inside_lpips': lo, 'dense_inside_lpips': de, 'delta': de-lo})
    report = {'lora_artifact_sha256': artifacts, 'stored_adapter_tensors': storage, 'lora': recipe['lora'],
              'tested_sample_uids': [r['sample_uid'] for r in rows],
              'protocol': 'rounding-aware-holdout-v2' if a.rounding_aware_holdout else 'strict-initial-v1',
              'tolerances': TOLERANCES, 'logits_parity': results, 'quality': quality,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'all_fp32': True, 'no_lora': True, 'status': 'PASS' if all(r['pass'] for r in results) and
              sum(r['delta'] for r in quality)/len(quality) <= .01 else 'FAILED'}
    write_json(a.evidence / 'MERGE_PARITY.json', report)
    if report['status'] != 'PASS':
        raise RuntimeError('MERGE_PARITY_FAILED_DO_NOT_TRAIN')
    from src.explicit_region.contracts import audit_component_identity
    identity = audit_component_identity(a.model_root)
    payload = {'schema': 'e3-lora-merged-dense-v1', 'training_mode': 'full_transformer',
               'git_sha': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
               'model_identity': {'repo_id': identity['repo_id'], 'resolved_revision': identity['resolved_revision'],
                                  'components': {n: {'config_sha256': r['config_sha256']} for n, r in identity['components'].items()}},
               'corruption': recipe['corruption'], 'merge_provenance': report}
    saved = save_dense_checkpoint(a.output, model=model, fingerprint_payload=payload)
    # Independent strict loader checks every shard and all keys/shapes.
    from src.transformer import Transformer2DModel
    independent = Transformer2DModel.from_config(model.config).float()
    load_dense_weights(independent, a.output)
    for k, v in model.state_dict().items():
        if not torch.equal(v.cpu(), independent.state_dict()[k]): raise RuntimeError(f'MERGE_RELOAD_MISMATCH: {k}')
    write_json(a.evidence / 'MERGED_CHECKPOINT.json', saved | {'independent_reload': 'BITWISE_EXACT'})


if __name__ == '__main__':
    main()
