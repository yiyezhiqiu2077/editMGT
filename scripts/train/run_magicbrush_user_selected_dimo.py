#!/usr/bin/env python3
"""Continue only DiMO after an explicit user-selected SFT Teacher transition."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.train.run_magicbrush_two_stage import Run
from src.explicit_region.config import load_config
from src.explicit_region.contracts import sha256_file
from src.explicit_region.checkpoint import recipe_fingerprint
from src.explicit_region.dense_checkpoint import verify_dense_checkpoint,write_json
from src.explicit_region.magicbrush_experiment import verify_magicbrush_ready
from src.dimo.magicbrush_selection import export_teacher,user_select
from src.dimo.runtime import gpu_admission


class UserSelectedRun(Run):
    def execute(self):
        if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
            raise RuntimeError('MAGICBRUSH_REQUIRES_CLEAN_COMMITTED_CODE')
        if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='exp/e3-dense-fp32-v1':
            raise RuntimeError('MAGICBRUSH_REQUIRES_EXISTING_MAINTENANCE_BRANCH')
        marker=self.root/'DIMO_USER_SELECTED_STARTED.json'
        if marker.exists():raise RuntimeError('MAGICBRUSH_USER_SELECTED_ALREADY_STARTED_NO_IMPLICIT_RECOVERY')
        cpu=json.loads((self.root/'admission/DIMO_EPOCH7_CPU_ACCEPTANCE.json').read_text())
        if cpu.get('status')!='PASS' or cpu.get('git_sha')!=self.sha:
            raise RuntimeError('MAGICBRUSH_CURRENT_CODE_CPU_ACCEPTANCE_REQUIRED')
        request=json.loads(self.a.user_request_record.read_text())
        if (request.get('requested_action')!='stop_sft_and_use_epoch7_for_full_dense_dimo'
                or request.get('teacher_epoch')!=self.a.teacher_epoch
                or Path(request.get('teacher_checkpoint','')).resolve()!=self.a.teacher_checkpoint.resolve()):
            raise RuntimeError('MAGICBRUSH_EXPLICIT_USER_TEACHER_REQUEST_MISMATCH')
        dimo=load_config(self.root/'configs/dimo.yaml')
        verify_magicbrush_ready(dimo)
        plan=json.loads((self.root/'evaluation_plan.json').read_text())
        if Path(dimo['teacher_checkpoint']).resolve()!=self.root/'selected_teacher':
            raise RuntimeError('MAGICBRUSH_USER_SELECTED_TEACHER_DESTINATION_CHANGED')
        if (self.root/'selected_teacher').exists() or (self.root/'dimo').exists():
            raise RuntimeError('MAGICBRUSH_USER_SELECTED_REQUIRES_NEW_DIMO_OUTPUTS')
        record=user_select(plan,self.a.teacher_checkpoint,epoch=self.a.teacher_epoch,
                           instruction=self.a.user_instruction,
                           authorization_sha256=sha256_file(self.a.user_request_record))
        # Teacher export, all smoke outputs, training and atomic checkpoint save.
        source=verify_dense_checkpoint(self.a.teacher_checkpoint)
        from safetensors import safe_open
        count=0
        index=json.loads((self.a.teacher_checkpoint/'model.safetensors.index.json').read_text())
        for shard in sorted(set(index['weight_map'].values())):
            with safe_open(self.a.teacher_checkpoint/shard,framework='pt',device='cpu') as weights:
                for key in weights.keys():
                    import math
                    count+=math.prod(weights.get_slice(key).get_shape())
        estimate=int(count*(4+4*28+3*8+6*28+6*8+28)*1.15)+32*1024**3
        free=shutil.disk_usage(self.root).free
        write_json(self.root/'admission/DIMO_EPOCH7_DISK_BUDGET.json',{
            'status':'PASS' if free>=estimate else 'FAIL','parameters_per_role':count,
            'required_bytes':estimate,'free_bytes':free,'retention':'all planned checkpoints; no deletion'})
        if free<estimate:raise RuntimeError('MAGICBRUSH_USER_SELECTED_DISK_BUDGET_FAILED')
        gpu_admission(magicbrush=True)
        write_json(marker,{'git_sha':self.sha,'plan_sha256':recipe_fingerprint(plan),
                          'teacher_checkpoint_sha256':source['checkpoint_sha256'],
                          'user_request_record_sha256':sha256_file(self.a.user_request_record),
                          'selection_method':'USER_SELECTED','full_dev_teacher_selection_performed':False})
        self.phase='dimo:export_user_selected_teacher';self.state('RUNNING')
        write_json(self.root/'TEACHER_SELECTION.json',record)
        export_teacher(record,self.root/'selected_teacher')
        self.command([sys.executable,ROOT/'scripts/report/report_magicbrush_two_stage.py',
                      '--experiment-root',self.root,'--stage','sft'],'sft_user_stopped_report')
        smoke=self.root/'dimo/smoke2';fresh=self.root/'dimo/fresh20';resumed=self.root/'dimo/resume20'
        self.train('dimo',smoke,2)
        self.train('dimo',fresh,20)
        self.train('dimo',resumed,10)
        self.train('dimo',resumed,20,resumed/'checkpoints/checkpoint-10')
        self.acceptance('dimo',smoke,fresh,resumed)
        train=self.root/'dimo/train'
        self.monitor('dimo',train)
        self.train('dimo',train,0)
        self.dimo_eval(train,0)
        previous=0
        for stop in dimo['resume_checkpoint_steps']:
            self.train('dimo',train,stop,train/'checkpoints'/f'checkpoint-{previous}')
            self.dimo_eval(train,stop)
            previous=stop
        for step in dimo['evaluation_steps']:
            self.dimo_eval(train,step,dev32=False)
        self.command([sys.executable,ROOT/'scripts/report/report_magicbrush_two_stage.py',
                      '--experiment-root',self.root,'--stage','dimo'],'dimo_report')
        self.phase='complete';self.state('COMPLETE',teacher_source=record['teacher_source'],
                                       dimo_steps=dimo['max_optimizer_steps'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment-root',type=Path,required=True)
    p.add_argument('--teacher-checkpoint',type=Path,required=True)
    p.add_argument('--teacher-epoch',type=int,required=True)
    p.add_argument('--user-request-record',type=Path,required=True)
    p.add_argument('--user-instruction',required=True)
    args=p.parse_args()
    args.upload_curves=False;args.monitor_token_file=None
    lock=(args.experiment_root/'.coordinator.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=UserSelectedRun(args)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        run.execute()
    except BaseException as error:
        run.failed.set();run.stop_children()
        run.state('STOPPED_HARD_ERROR',error_class=type(error).__name__,error=str(error),
                  automatic_recipe_changes=False,automatic_failure_retry=False)
        raise
    finally:
        for monitor in run.monitors:monitor.terminate()
        lock.close()


if __name__=='__main__':main()
