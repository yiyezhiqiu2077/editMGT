#!/usr/bin/env python3
"""Explicit 20-step smoke or gated 500-step continuation; never long training."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.explicit_region.config import load_config
from src.explicit_region.contracts import sha256_file
from src.explicit_region.dense_checkpoint import write_json
from src.dimo.runtime import validate_full_config


def validate_short_gate(root):
    root=Path(root)
    proof=json.loads((root/'GPU_SMOKE_PASS.json').read_text())
    quality=json.loads((root/'evaluation/soft_reward/QUALITY_GATE_step20.json').read_text())
    identity=sha256_file(root/'experiment_manifest.json')
    if (proof.get('gpu')!='REAL_GPU' or proof.get('step')!=20 or quality.get('checkpoint_step')!=20
            or any(d.get('status')!='PASS' or d.get('run_manifest_sha256')!=identity for d in [proof,quality])):
        raise RuntimeError('SOFT_REWARD_SMOKE_OR_FIXED_DEV_GATE_FAILED_DO_NOT_EXPAND')
    if proof['checkpoint_completed_sha256'] != sha256_file(root/'checkpoints/checkpoint-20/COMPLETED.json'):
        raise RuntimeError('SOFT_REWARD_SMOKE_CHECKPOINT_CHANGED')
    if (proof.get('inference_completed_sha256') != sha256_file(root/'inference/checkpoint-20/COMPLETED.json')
            or quality.get('checkpoint_completed_sha256') != proof.get('inference_completed_sha256')):
        raise RuntimeError('SOFT_REWARD_FIXED_DEV_CHECKPOINT_BINDING_CHANGED')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--mode',choices=['smoke','short'],required=True)
    p.add_argument('--print-command',action='store_true',help='CPU-only command preview; execute no experiments')
    args=p.parse_args();c=load_config(args.config);validate_full_config(c)
    if not c.get('soft_reward',{}).get('enabled'):
        raise RuntimeError('SOFT_REWARD_CONFIG_REQUIRED')
    root=Path(c['output_dir'])
    stop=20 if args.mode=='smoke' else 500
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
         str(ROOT/'scripts/train/train_full_dense_dimo.py'),'--config',str(Path(args.config).resolve()),
         '--stop-at-global-step',str(stop)]
    cmd+=(['--fork-dimo-checkpoint',c['soft_reward']['fork_checkpoint']] if args.mode=='smoke' else
          ['--resume',str(root/'checkpoints/checkpoint-20')])
    if args.print_command:
        print(json.dumps({'command':cmd,'gpu_experiment':'NOT_RUN','mode':args.mode}));return
    if args.mode=='smoke':
        if root.exists():raise RuntimeError('SOFT_REWARD_SMOKE_REQUIRES_NEW_OUTPUT_NO_OVERWRITE')
        inference=Path(c['soft_reward']['inference_output_dir'])
        if inference.exists():raise RuntimeError('SOFT_REWARD_INFERENCE_OUTPUT_EXISTS_NO_OVERWRITE')
        inference.mkdir(parents=True)
        root.mkdir(parents=True)
        (root/'inference').symlink_to(inference,target_is_directory=True)
    else:validate_short_gate(root)
    from src.dimo.mixed_storage import segment_budget
    parent=json.loads((Path(c['soft_reward']['fork_checkpoint'])/'inference_identity.json').read_text())
    trainable=sum(math.prod(r['shape']) for r in parent['recipe_contract']['optimizer_parameters']['student'])
    from safetensors import safe_open
    checkpoint=Path(c['soft_reward']['fork_checkpoint'])
    index=json.loads((checkpoint/'student.index.json').read_text())['weight_map']
    parameters=0
    for filename in set(index.values()):
        with safe_open(checkpoint/filename,framework='pt',device='cpu') as f:
            parameters+=sum(math.prod(f.get_slice(k).get_shape()) for k in f.keys())
    budget=segment_budget(parameters,c,root,stop,trainable_parameters=trainable)
    write_json(root/f'preflight_disk_budget_step{stop}.json',budget)
    subprocess.run(cmd,cwd=ROOT,check=True)
    if args.mode=='smoke':
        evidence=[]
        for rank in range(4):
            reward=json.loads((root/f'reward_acceptance_rank{rank}.json').read_text())
            acceptance=json.loads((root/f'v2_acceptance_rank{rank}.json').read_text())
            memory=[json.loads(l) for l in (root/f'memory_rank{rank}.jsonl').read_text().splitlines()]
            peak=max(r.get('peak_reserved_bytes',0) for r in memory)
            if (reward['status']!='PASS' or reward['step']!=20 or reward['calibration']['count']!=4
                    or reward['calibration']['weights'] is None or acceptance['status']!='PASS'
                    or acceptance['completed_updates_per_role']!=20 or peak>75*1024**3):
                raise RuntimeError('SOFT_REWARD_SMOKE_ACCEPTANCE_OR_MEMORY_MARGIN_FAILED')
            evidence.append({'rank':rank,'peak_reserved_bytes':peak,'calibration':reward['calibration']})
        proof={'status':'PASS','step':20,'gpu':'REAL_GPU','ranks':evidence,
            'run_manifest_sha256':sha256_file(root/'experiment_manifest.json'),
            'checkpoint_completed_sha256':sha256_file(root/'checkpoints/checkpoint-20/COMPLETED.json'),
            'inference_completed_sha256':sha256_file(root/'inference/checkpoint-20/COMPLETED.json'),
            'fixed_dev_gate':'REQUIRED_SEPARATELY_BEFORE_SHORT_TRAINING'}
        write_json(root/'GPU_SMOKE_PASS.json',proof)
    print(json.dumps({'mode':args.mode,'stop_step':stop,'output':str(root),'status':'COMPLETE'}))


if __name__=='__main__':main()
