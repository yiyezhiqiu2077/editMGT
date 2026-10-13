#!/usr/bin/env python3
"""Explicitly authorized long run, retained 2000-step checkpoints, no short500."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.explicit_region.config import load_config
from src.explicit_region.dense_checkpoint import write_json
from src.dimo.runtime import validate_full_config
from src.dimo.reward_continuation import validate_continuation
from src.dimo.mixed_storage import segment_budget


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--resume',help='latest complete checkpoint of this long run')
    parser.add_argument('--print-command',action='store_true')
    args=parser.parse_args();config=load_config(args.config);validate_full_config(config)
    if not config.get('soft_reward_continuation'):
        raise RuntimeError('LONG_REQUIRES_EXPLICIT_USER_AUTHORIZATION')
    parent=validate_continuation(config)
    manifest=config['soft_reward_continuation']
    source=Path(manifest['source'])
    from scripts.train.run_soft_dimo_reward import validate_short_gate
    validate_short_gate(source.parent.parent)
    output=Path(config['output_dir']);inference=Path(config['soft_reward']['inference_output_dir'])
    initial_step=manifest['source_step']
    resume=Path(args.resume).resolve() if args.resume else None
    if resume:
        if resume.parent!=output.resolve()/'checkpoints':
            raise RuntimeError('LONG_RESUME_MUST_USE_THIS_RUN_CHECKPOINT')
        initial_step=json.loads((resume/'COMPLETED.json').read_text())['committed_step']
    stops=sorted(s for s in set(config['resume_checkpoint_steps'])|{manifest['target_step']} if s>initial_step)
    def command(stop,checkpoint):
        return [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
            str(ROOT/'scripts/train/train_full_dense_dimo.py'),'--config',str(Path(args.config).resolve()),
            '--stop-at-global-step',str(stop),
            '--resume' if checkpoint else '--continue-soft-reward-checkpoint',str(checkpoint or source)]
    if args.print_command:
        print(json.dumps({'first_command':command(stops[0],resume),'long_target_step':manifest['target_step'],
                         'save_steps':stops,'training':'NOT_RUN','short500':'CANCELLED_BY_USER'}));return
    if not resume:
        if output.exists() or inference.exists():raise RuntimeError('LONG_NEW_OUTPUT_REQUIRED_NO_OVERWRITE')
        inference.mkdir(parents=True);output.mkdir(parents=True)
        (output/'inference').symlink_to(inference,target_is_directory=True)
    from safetensors import safe_open
    mapping=json.loads((source/'student.index.json').read_text())['weight_map']
    parameters=0
    for filename in set(mapping.values()):
        with safe_open(source/filename,framework='pt',device='cpu') as f:
            parameters+=sum(math.prod(f.get_slice(k).get_shape()) for k in f.keys())
    trainable=sum(math.prod(p['shape']) for p in parent['recipe_contract']['optimizer_parameters']['student'])
    status={'mode':'USER_AUTHORIZED_LONG','target_step':manifest['target_step'],'initial_step':initial_step,
            'checkpoint_interval':2000,'short500':'CANCELLED_BY_USER','pid':os.getpid(),
            'source_complete_checkpoint':str(source),'config':str(Path(args.config).resolve())}
    child=None
    def save(phase,**extra):
        status.update(phase=phase,updated_utc=datetime.now(timezone.utc).isoformat(),**extra)
        write_json(output/'LONG_TRAINING_STATUS.json',status);print(json.dumps(status),flush=True)
    def stop(signum,frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
        save('STOPPED_BY_SIGNAL',signal=signum);raise SystemExit(128+signum)
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
    try:
        for endpoint in stops:
            budget=segment_budget(parameters,config,output,endpoint,trainable_parameters=trainable)
            write_json(output/f'preflight_disk_budget_step{endpoint}.json',budget)
            cmd=command(endpoint,resume)
            child=subprocess.Popen(cmd,cwd=ROOT,start_new_session=True)
            save('TRAINING',segment_stop_step=endpoint,child_pid=child.pid,command=cmd)
            rc=child.wait()
            if rc:raise RuntimeError(f'LONG_TRAINING_SEGMENT_FAILED_RETURN_CODE_{rc}')
            checkpoint=output/f'checkpoints/checkpoint-{endpoint}'
            marker=json.loads((checkpoint/'COMPLETED.json').read_text())
            if marker['committed_step']!=endpoint:raise RuntimeError('LONG_CHECKPOINT_STEP_MISMATCH')
            for rank in range(4):
                reward=json.loads((output/f'reward_acceptance_rank{rank}.json').read_text())
                if reward['status']!='PASS' or reward['step']!=endpoint:
                    raise RuntimeError('LONG_FROZEN_REWARD_ACCEPTANCE_FAILED')
            resume=checkpoint
            save('CHECKPOINT_COMPLETE',latest_complete_step=endpoint,child_pid=None)
        save('COMPLETE',next_training='NONE')
    except Exception as exc:
        save('STOPPED_REVIEW',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':main()
