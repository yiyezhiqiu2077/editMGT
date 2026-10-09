#!/usr/bin/env python3
"""One model load per teacher/raw/EMA DEV128 diagnostic; no TEST entry point."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
from torch.utils.data import default_collate

from scripts.eval.formal_eval import evaluate
from scripts.train.train_dimo_editing import prepare_batch
from src.dimo.checkpoint import config_hash, read_dimo_state, validate_inference_checkpoint
from src.dimo.contracts import (DIMO_MODEL_ROLES_V11, DIMO_UPSTREAM_COMMIT, _sha256,
    build_inference_fingerprint, released_base_model_identity, teacher_bundle_fingerprint)
from src.dimo.dev import verified_dev128
from src.dimo.ema import apply_ema_to_student_role
from src.dimo.initialization import initialize_shared_model_roles
from src.dimo.one_step import one_step_edit_tokens
from src.dimo.pilot_contract import gpu_allocation_report, protect_output
from src.dimo.provisional_teacher import load_provisional_teacher
from src.dimo.rng import normal_noise_per_sample, stable_seed
from src.editmgt import init_edit_mgt
from src.explicit_region.config import load_config
from src.explicit_region.contracts import audit_component_identity
from src.explicit_region.fixed_dataset import CanonicalAlignedDataset
from src.explicit_region.modeling import load_released_components


def image(tensor):
    return Image.fromarray((tensor.detach().cpu().float().permute(1, 2, 0).numpy() * 255)
                          .round().clip(0, 255).astype(np.uint8))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=['teacher', 'student', 'ema'], required=True)
    parser.add_argument('--student-checkpoint')
    parser.add_argument('--teacher-checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--corpus-ready', required=True)
    parser.add_argument('--dataset', choices=['magicbrush', 'crispedit', 'scaleedit', 'interedit'], default='magicbrush')
    parser.add_argument('--canonical-root', required=True)
    parser.add_argument('--model-root', required=True)
    parser.add_argument('--config', default='configs/eval/dimo_e3_step3125_dev.yaml')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if (config.get('dev_only') is not True or config.get('formal_teacher') is not False
            or config['selection']['status'] != 'DIAGNOSTIC_ONLY' or config['resolution'] != 1024
            or config['steps'] != 12 or config['inference_timestep_mode'] != 'roi_relative'
            or len(config['generation_seeds']) != 1):
        raise RuntimeError('INVALID_DIMO_DEV_DIAGNOSTIC_CONFIG')
    gpu_allocation_report({'distributed': {'minimum_free_memory_gib': 24}}, expected_count=1)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('DEV inference requires exactly one independently allocated visible GPU')
    base = released_base_model_identity(audit_component_identity(args.model_root))
    contract = load_provisional_teacher(args.teacher_checkpoint, expected_base_identity=base)
    rows, dev_identity = verified_dev128(args.manifest, args.corpus_ready, dataset=args.dataset)
    if contract.manifest['source_d200k_sha256'] != dev_identity['train_manifest_sha256']:
        raise RuntimeError('DIMO_DEV_CORPUS_DIFFERS_FROM_TEACHER_E3_D200K')
    output = Path(args.output_dir).resolve()
    protect_output({'output_dir': str(output)}, contract)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('DEV_OUTPUT_ALREADY_EXISTS')
    output.mkdir(parents=True, exist_ok=True)
    dataset = CanonicalAlignedDataset(rows, args.canonical_root, resolution=1024,
        base_seed=int(config['geometry_seed']), verify_hashes=True, max_random_attempts=8,
        minimum_mask_retention=.75)
    teacher_bundle = teacher_bundle_fingerprint(args.teacher_checkpoint, base_model_identity=base)
    state, roles, components, pipe = None, None, None, None
    if args.role == 'teacher':
        if args.student_checkpoint:
            raise RuntimeError('TEACHER_ROLE_DOES_NOT_ACCEPT_STUDENT_CHECKPOINT')
        step = 3125
        pipe = init_edit_mgt('cuda', enable_bf16=True, base_model_path=args.model_root,
                            local_files_only=True, trainable_state_path=args.teacher_checkpoint)
        pipe.set_progress_bar_config(disable=True)
    else:
        if not args.student_checkpoint:
            raise RuntimeError('STUDENT_CHECKPOINT_REQUIRED')
        state = read_dimo_state(args.student_checkpoint)
        inference = build_inference_fingerprint(teacher_bundle_sha256=teacher_bundle['bundle_sha256'],
            base_model_identity=base, model_roles=DIMO_MODEL_ROLES_V11, upstream_commit=DIMO_UPSTREAM_COMMIT)
        validate_inference_checkpoint(state, expected_teacher_bundle_fingerprint=teacher_bundle,
            expected_inference_fingerprint=inference, expected_upstream_commit=DIMO_UPSTREAM_COMMIT)
        step = state['global_dimo_step']
        components = load_released_components(args.model_root, torch_dtype=torch.bfloat16, vq_dtype=torch.float32)
        roles = initialize_shared_model_roles(components.transformer, args.teacher_checkpoint,
                                              model_roles=DIMO_MODEL_ROLES_V11).to('cuda').eval()
        if args.role == 'student':
            roles.load_role_state_dict('student', state['student_state'])
        else:
            apply_ema_to_student_role(roles, state['student_ema'])
        for module, dtype in ((components.text_encoder, torch.bfloat16),
                              (components.llm_encoder, torch.bfloat16), (components.vqvae, torch.float32)):
            module.to(device='cuda', dtype=dtype).eval().requires_grad_(False)
        # Reuse the exact token-mask and initialization reference config used in training.
        train_config = state.get('config_hash')
        run_manifest = Path(args.student_checkpoint).parent / 'run_manifest.json'
        if not run_manifest.is_file():
            raise RuntimeError('STUDENT_RUN_MANIFEST_REQUIRED')
        runtime = json.loads(run_manifest.read_text())
        if (runtime['config_hash'] != train_config or runtime.get('teacher_hash') != contract.checkpoint_hash
                or config_hash(runtime['training_identity']) != train_config
                or runtime['resolved_config'] != runtime['training_identity']['resolved_config']):
            raise RuntimeError('STUDENT_RUN_MANIFEST_IDENTITY_MISMATCH')
        train_config = runtime['resolved_config']
    predictions = []
    for index in range(len(dataset)):
        item = dataset[index]
        source, target = image(item['source_image']), image(item['target_image'])
        mask = Image.fromarray(item['edit_region_mask'].numpy().astype(np.uint8) * 255)
        seed = int(config['generation_seeds'][0])
        uid = item['sample_uid']
        seeds = lambda purpose: stable_seed(seed, 0, uid, 0, purpose)
        # The base seed is identical for every checkpoint; algorithm-specific
        # substreams are explicit, not falsely claimed to be identical samplers.
        def generate():
            if pipe is not None:
                return pipe(prompt=item['instruction_en'], reference_image=source, mask_image=mask,
                    height=1024, width=1024, num_inference_steps=12, guidance_scale=config['guidance_scale'],
                    reference_strength=config['reference_strength'],
                    generator=torch.Generator(device='cuda').manual_seed(seed), lora_scope='both',
                    inference_timestep_mode='roi_relative').images[0]
            prepared = prepare_batch(default_collate([item]), components, train_config, 'cuda')
            grid = prepared['source_tokens'].shape[-1]
            noise = normal_noise_per_sample(torch.empty(1, roles.base_model.inner_dim, grid, grid,
                                                        device='cuda', dtype=torch.bfloat16),
                                             [seeds('dimo-embedding-noise')])
            token_output = one_step_edit_tokens(roles, source_tokens=prepared['source_tokens'],
                edit_region_mask=prepared['edit_region_mask'], prompt_condition=prepared['prompt_condition'],
                timestep_model_kwargs=prepared['model_kwargs'], mask_token_id=roles.base_model.config.vocab_size - 1,
                codebook_size=roles.base_model.config.codebook_size,
                r_init=train_config['initialization']['mask_ratio'], init_mask_seeds=seeds('dimo-init-mask'),
                init_token_seeds=seeds('dimo-init-token'), sample_seeds=seeds('dimo-student-sample'),
                temperature=train_config['sampling']['temperature'], top_k=train_config['sampling']['top_k'],
                top_p=train_config['sampling']['top_p'], cfg_scale=config['dimo_student_cfg'],
                target_embedding_noise=noise, embedding_noise_sigma=train_config['embedding_perturbation']['sigma']
                if train_config['embedding_perturbation']['enabled'] else 0)
            decoded = components.vqvae.decode(token_output.tokens, force_not_quantize=True,
                shape=(1, grid, grid, components.vqvae.config.latent_channels)).sample.clip(0, 1)
            if not torch.isfinite(decoded).all() or not torch.isfinite(token_output.logits).all():
                raise RuntimeError('DIMO_DEV_NONFINITE_OUTPUT')
            return image(decoded[0])
        if index == 0:
            generate()  # Untimed warmup, same stateless generation seeds.
        torch.cuda.synchronize()
        start = time.perf_counter()
        generated = generate()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        paths = {}
        for name, value in [('source', source), ('target', target), ('mask', mask), ('output', generated)]:
            path = output / f'{index:04d}_{name}.png'
            value.save(path)
            paths[name] = str(path)
        predictions.append(paths | {'sample_key': uid, 'dataset_name': args.dataset,
            'edit_type': item['edit_type'], 'seed': seed, 'runtime_seconds': elapsed,
            'output_sha256': _sha256(Path(paths['output'])),
            'aligned_input_sha256': {name: _sha256(Path(paths[name])) for name in ('source', 'target', 'mask')},
            'dev_only': True, 'formal_teacher': False})
    manifest = output / 'predictions.jsonl'
    manifest.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in predictions))
    del state, roles, components, pipe
    gc.collect(); torch.cuda.empty_cache()
    result = evaluate(manifest, config, output / 'metrics.json', 'cuda')
    metadata = dev_identity | {'schema': 'dimo-dev-diagnostic-v1', 'dev_only': True, 'formal_teacher': False,
        'role': args.role, 'step': step, 'teacher_hash': contract.checkpoint_hash, 'config': config,
        'label': f'{args.role}-{step}',
        'latency_scope': 'warm_model_source_tokens_text_encoding_forward_vq_decode_excludes_model_load_and_png_io',
        'teacher_cfg_diagnostic_only': config['guidance_scale'], 'dimo_student_cfg': config['dimo_student_cfg'],
        'prediction_manifest_sha256': _sha256(manifest),
        'metrics_sha256': _sha256(output / 'metrics.json'),
        'aggregate': result['aggregate'][f'dataset:{args.dataset}']}
    (output / 'diagnostic.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': 'DEV_DIAGNOSTIC_ONLY', 'role': args.role, 'step': step, 'rows': 128}))


if __name__ == '__main__':
    main()
