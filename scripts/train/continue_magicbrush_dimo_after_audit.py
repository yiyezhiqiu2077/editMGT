#!/usr/bin/env python3
"""One explicitly authorized continuation after the BF16 audit reference fix.

Retains all failed-run evidence, repeats acceptance under one code SHA, and
never resumes optimizer state across the code change or retries a new failure.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.train.run_magicbrush_user_selected_dimo import UserSelectedRun
from src.explicit_region.config import load_config
from src.explicit_region.contracts import sha256_file
from src.explicit_region.dense_checkpoint import write_json
from src.explicit_region.magicbrush_experiment import verify_magicbrush_ready
from src.dimo.dense_teacher import load_dense_teacher_contract
from src.dimo.magicbrush_selection import verify_selection_record
from src.dimo.runtime import gpu_admission


def verify_recovery_request(root,request):
    """Bind a specific stopped run and its immutable configuration to consent."""
    if request.get('schema')!='magicbrush-dimo-audit-recovery-v1' or not request.get('user_instruction','').strip():
        raise RuntimeError('DIMO_AUDIT_RECOVERY_EXPLICIT_REQUEST_REQUIRED')
    status_path=root/'STATUS.json'
    status=json.loads(status_path.read_text())
    if (status.get('status')!='STOPPED_HARD_ERROR' or status.get('phase')!='dimo:fresh20:step20'
            or sha256_file(status_path)!=request.get('failed_status_sha256')):
        raise RuntimeError('DIMO_AUDIT_RECOVERY_STOPPED_RUN_MISMATCH')
    for relative,digest in request.get('evidence',{}).items():
        path=(root/relative).resolve()
        if root.resolve() not in path.parents or sha256_file(path)!=digest:
            raise RuntimeError('DIMO_AUDIT_RECOVERY_EVIDENCE_CHANGED')
    for required in ('configs/dimo.yaml','TEACHER_SELECTION.json','dimo/fresh20/failure_rank1.json','dimo/fresh20/failure_rank2.json'):
        if required not in request.get('evidence',{}):
            raise RuntimeError('DIMO_AUDIT_RECOVERY_EVIDENCE_MISSING')
    for rank in (1,2):
        failure=json.loads((root/f'dimo/fresh20/failure_rank{rank}.json').read_text())
        if failure.get('last_committed_step')!=19 or not failure.get('message','').startswith('LINEAR_SURROGATE_REAL_LOGIT_GRADIENT_FAILED:'):
            raise RuntimeError('DIMO_AUDIT_RECOVERY_UNEXPECTED_FAILURE')
    if (root/'dimo/train').exists() or (root/'DIMO_AUDIT_RECOVERY_STARTED.json').exists():
        raise RuntimeError('DIMO_AUDIT_RECOVERY_ALREADY_STARTED_NO_IMPLICIT_RETRY')
    return status


class AuditRecoveryRun(UserSelectedRun):
    def execute(self):
        if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
            raise RuntimeError('MAGICBRUSH_REQUIRES_CLEAN_COMMITTED_CODE')
        if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='exp/e3-dense-fp32-v1':
            raise RuntimeError('MAGICBRUSH_REQUIRES_EXISTING_MAINTENANCE_BRANCH')
        request=json.loads(self.a.recovery_request.read_text())
        old_status=verify_recovery_request(self.root,request)
        cpu=json.loads(self.a.cpu_acceptance.read_text())
        if cpu.get('status')!='PASS' or cpu.get('git_sha')!=self.sha:
            raise RuntimeError('MAGICBRUSH_CURRENT_CODE_CPU_ACCEPTANCE_REQUIRED')
        config=load_config(self.root/'configs/dimo.yaml')
        verify_magicbrush_ready(config)
        teacher=load_dense_teacher_contract(config['teacher_checkpoint'])
        record=json.loads((self.root/'TEACHER_SELECTION.json').read_text())
        verify_selection_record(record,teacher.manifest['dense_checkpoint_sha256'])
        if (record.get('selection_status')!='USER_SELECTED' or record.get('selected_step')!=3843
                or Path(config['teacher_checkpoint']).resolve()!=self.root/'selected_teacher'):
            raise RuntimeError('DIMO_AUDIT_RECOVERY_EPOCH7_TEACHER_CHANGED')
        historical=json.loads((self.root/'admission/DIMO_EPOCH7_DISK_BUDGET.json').read_text())
        free=shutil.disk_usage(self.root).free
        required=historical['required_bytes']
        if free<required:raise RuntimeError('DIMO_AUDIT_RECOVERY_DISK_BUDGET_FAILED')
        gpu_admission(magicbrush=True)
        directory=self.root/'dimo/audit_recovery_v1'
        if directory.exists():raise RuntimeError('DIMO_AUDIT_RECOVERY_OUTPUT_EXISTS')
        directory.mkdir()
        write_json(directory/'PREVIOUS_FAILURE_STATUS.json',old_status)
        write_json(self.root/'DIMO_AUDIT_RECOVERY_STARTED.json',{
            'git_sha':self.sha,'previous_git_sha':old_status['git_sha'],
            'recovery_request_sha256':sha256_file(self.a.recovery_request),
            'cpu_acceptance_sha256':sha256_file(self.a.cpu_acceptance),
            'teacher_checkpoint_sha256':teacher.manifest['dense_checkpoint_sha256'],
            'config_sha256':sha256_file(self.root/'configs/dimo.yaml'),
            'required_bytes':required,'free_bytes':free,
            'output_directory':str(directory),'automatic_failure_retry':False,
            'reference_fix':'reverse-roi-reduction-v2','training_recipe_changed':False,
            'old_acceptance_outputs_preserved':True})
        self.dimo_pipeline(directory/'smoke2',directory/'fresh20',directory/'resume20')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-root',type=Path,required=True)
    p.add_argument('--recovery-request',type=Path,required=True)
    p.add_argument('--cpu-acceptance',type=Path,required=True)
    args=p.parse_args();args.upload_curves=False;args.monitor_token_file=None
    lock=(args.experiment_root/'.coordinator.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=AuditRecoveryRun(args)
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
