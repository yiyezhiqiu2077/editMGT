#!/usr/bin/env python3
"""Wait for successful SFT completion, then run user-requested last-weight DiMO."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.train.run_magicbrush_two_stage import Run
from src.explicit_region.config import load_config
from src.explicit_region.contracts import sha256_file
from src.explicit_region.dense_checkpoint import verify_dense_checkpoint, write_json
from src.dimo.mixed_experiment import FINAL_SFT_STEP, verify_ready, verify_request, select_last_teacher, export_last_teacher
from src.dimo.mixed_storage import whole_handoff_budget
from src.dimo.runtime import gpu_admission


def live_process(pid):
    proc = Path(f'/proc/{pid}')
    try:
        return not proc.joinpath('stat').read_text().split(') ')[1].startswith('Z')
    except FileNotFoundError:
        return False


def sft_success_state(state):
    return state.get('phase') in ('four_gpu_full_dev_evaluation', 'sft_quality_review_before_new_dimo')


def safe_stop_sft_evaluation(root, state):
    """Stop only the owned post-training evaluator coordinator, never SFT workers."""
    if state.get('phase') != 'four_gpu_full_dev_evaluation' or not live_process(state.get('pid', 0)):
        return False
    pid = state['pid']; proc = Path(f'/proc/{pid}')
    command = proc.joinpath('cmdline').read_bytes().split(b'\0')
    if (not any(part.endswith(b'/run_mixed30k_continuation.py') or part == b'scripts/train/run_mixed30k_continuation.py'
                for part in command) or str(root).encode() not in command):
        raise RuntimeError('MIXED_DIMO_PREDECESSOR_PROCESS_IDENTITY_CHANGED')
    write_json(root / 'SFT_EVALUATION_USER_HANDOFF.json', {'previous_state': state,
               'reason': 'explicit user request: automatic DiMO with last SFT epoch; DEV is diagnostic only',
               'sft_training_command_completed_successfully': True, 'sft_checkpoint': FINAL_SFT_STEP,
               'evaluation_partial_outputs_retained': True})
    os.kill(pid, signal.SIGTERM)
    return True


class AutoRun(Run):
    def __init__(self, args):
        super().__init__(args)
        self.owns_training = False
        self.source_lock = None
        self.request_path = self.root / 'USER_AUTO_LAST_SFT_DIMO.json'
        self.request_sha = sha256_file(self.request_path)
        self.env.update(MIXED_EXPERIMENT_ROOT=str(self.root),
                        PREVIOUS_EXPERIMENT_ROOT=str(args.previous_experiment),
                        EDITMGT_MODEL_ROOT='/data01/zhangyuyang/editmgt_eval_20261010/assets/editmgt',
                        HF_ENDPOINT='https://hf-mirror.com', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        for key in list(self.env):
            if key.lower().endswith('_proxy'): self.env.pop(key)
        if getattr(args, 'allow_shared_gpus', False): self.env['EDITMGT_ALLOW_SHARED_A800'] = '1'
        os.environ.update({key: self.env[key] for key in ('MIXED_EXPERIMENT_ROOT', 'PREVIOUS_EXPERIMENT_ROOT', 'EDITMGT_MODEL_ROOT')})

    def state(self, status, **values):
        payload = {'status': status, 'phase': self.phase, 'git_sha': self.sha, 'pid': os.getpid(),
                   'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   'sft_final_step': FINAL_SFT_STEP, 'dimo_epochs': 5, 'dimo_steps': 25000, **values}
        write_json(self.root / 'DIMO_HANDOFF_STATUS.json', payload)
        if self.owns_training:
            super().state(status, **values)
        else:
            print(json.dumps(payload), flush=True)

    def assert_code_request(self):
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        if (sha != self.sha or subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
                or sha256_file(self.request_path) != self.request_sha):
            raise RuntimeError('MIXED_DIMO_HANDOFF_CODE_OR_REQUEST_CHANGED')

    def wait_sft(self, config):
        self.phase = 'waiting_for_successful_sft_final_epoch'
        checkpoint = Path(config['sft_final_checkpoint'])
        while True:
            self.assert_code_request()
            state = json.loads((self.root / 'STATUS.json').read_text())
            if state.get('status') == 'USER_PAUSED':
                self.state('WAITING_USER_PAUSED_SFT'); time.sleep(30); continue
            if sft_success_state(state) and (checkpoint / 'COMPLETED.json').is_file():
                return state
            if state.get('status') == 'STOPPED_HARD_ERROR':
                raise RuntimeError('MIXED_DIMO_SFT_FAILED; final Teacher will not be substituted')
            if not live_process(state.get('pid', 0)):
                raise RuntimeError('MIXED_DIMO_SFT_COORDINATOR_STOPPED_BEFORE_SUCCESS')
            self.state('WAITING_FOR_SFT', sft_phase=state.get('phase'))
            time.sleep(30)

    def acquire_training(self, state):
        safe_stop_sft_evaluation(self.root, state)
        self.phase = 'waiting_for_sft_processes_and_gpu_release'
        self.source_lock = (self.root / '.coordinator.lock').open('a')
        while True:
            try:
                fcntl.flock(self.source_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                self.state('WAITING_FOR_SFT_COORDINATOR_EXIT'); time.sleep(10)
        while True:
            try:
                gpu_admission(magicbrush=True, allow_other_jobs=self.a.allow_shared_gpus)
                break
            except RuntimeError as exc:
                if 'GPU_HAS_OTHER_JOB' not in str(exc): raise
                self.state('WAITING_FOR_IDLE_GPUS', reason=str(exc)); time.sleep(30)
        self.owns_training = True

    def execute(self):
        self.assert_code_request()
        cpu = json.loads((self.root / 'admission/DIMO_AUTO_CPU_ACCEPTANCE.json').read_text())
        if cpu.get('status') != 'PASS' or cpu.get('git_sha') != self.sha:
            raise RuntimeError('MIXED_DIMO_HANDOFF_CURRENT_CPU_ACCEPTANCE_REQUIRED')
        marker = self.root / 'DIMO_AUTO_HANDOFF_STARTED.json'
        if marker.exists(): raise RuntimeError('MIXED_DIMO_HANDOFF_ALREADY_STARTED_NO_IMPLICIT_RETRY')
        config = load_config(ROOT / 'configs/dimo/full_dense_mixed20k_4a800_5epoch.yaml')
        request = verify_request(config)
        budget = whole_handoff_budget(self.root, self.a.secondary_root, 1009307648,
                                      evaluation_root=self.a.evaluation_root)
        write_json(self.root / 'admission/DIMO_AUTO_WHOLE_DISK_BUDGET.json', budget)
        if budget['status'] != 'PASS': raise RuntimeError('MIXED_DIMO_WHOLE_HANDOFF_STORAGE_INSUFFICIENT')
        write_json(marker, {'git_sha': self.sha, 'authorization_sha256': self.request_sha,
                           'teacher_selection': 'USER_LAST_EPOCH', 'final_step': FINAL_SFT_STEP,
                           'dimo_epochs': 5, 'dimo_steps': 25000, 'old_dimo_restart': False})
        state = self.wait_sft(config)
        self.assert_code_request(); ready = verify_ready(config); record = select_last_teacher(config)
        self.acquire_training(state)
        if (self.root / 'selected_teacher').exists() or (self.root / 'dimo').exists():
            raise RuntimeError('MIXED_DIMO_REQUIRES_NEW_OUTPUTS')
        self.phase = 'exporting_user_requested_final_sft_teacher'; self.state('RUNNING')
        write_json(self.root / 'TEACHER_SELECTION.json', record)
        export_last_teacher(record, self.root / 'selected_teacher')
        import yaml
        (self.root / 'configs/dimo.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        plan = json.loads((self.root / 'evaluation_plan.json').read_text())
        write_json(self.root / 'evaluation_plan.before_user_last_dimo.json', plan)
        plan.update(teacher_checkpoint=config['teacher_checkpoint'],
                    selection_rules={'method': 'USER_LAST_EPOCH', 'sft_final_step': FINAL_SFT_STEP,
                                     'dev_quality_gate': False, 'authorization_sha256': self.request_sha})
        write_json(self.root / 'evaluation_plan.json', plan)
        secondary = self.a.secondary_root
        fresh = secondary / 'fresh20'; resumed = secondary / 'resume20'
        smoke = self.root / 'dimo/smoke2'; train = self.root / 'dimo/train'
        for path in (fresh, resumed, train):
            if path.exists(): raise RuntimeError(f'MIXED_DIMO_OUTPUT_ALREADY_EXISTS: {path}')
        secondary.mkdir(parents=True, exist_ok=True)
        inference = secondary / 'training_inference'; inference.mkdir()
        train.mkdir(parents=True); (train / 'inference').symlink_to(inference, target_is_directory=True)
        evaluation = self.a.evaluation_root
        if evaluation is not None:
            if evaluation.exists(): raise RuntimeError(f'MIXED_DIMO_EVALUATION_OUTPUT_ALREADY_EXISTS: {evaluation}')
            evaluation.mkdir(parents=True)
            (train / 'evaluation').symlink_to(evaluation, target_is_directory=True)
        write_json(self.root / 'DIMO_STORAGE_LOCATIONS.json', {'smoke2': str(smoke), 'fresh20': str(fresh),
                   'resume20': str(resumed), 'train': str(train), 'inference': str(inference),
                   'evaluation': str(evaluation or train / 'evaluation'), 'retention': 'all checkpoints'})
        self.assert_code_request()
        self.train('dimo', smoke, 2); self.train('dimo', fresh, 20); self.train('dimo', resumed, 10)
        self.train('dimo', resumed, 20, resumed / 'checkpoints/checkpoint-10')
        self.acceptance('dimo', smoke, fresh, resumed)
        self.assert_code_request(); self.monitor('dimo', train); self.train('dimo', train)
        for step in config['evaluation_steps']:
            self.dimo_eval(train, step, dev32=False)
        report = ['# Mixed20K automatic final-SFT DiMO', '', f'Code: {self.sha}',
                  f'Teacher: {record["selected_checkpoint"]}', f'Teacher hash: {record["selected_checkpoint_hash"]}',
                  'Explicit user last-epoch choice; DEV scores were not a Teacher gate.',
                  'Same frozen 20K manifest; 5 DiMO epochs / 25,000 Student updates.',
                  'r_init=1.0; sigma=0.3; CFG1; linear surrogate; FP32 parameters and moments; BF16 forward.',
                  'Teacher 12 steps/CFG5; Raw Student and EMA 1 step/CFG1; unchanged DEV527 and seeds [0,1].',
                  'Full paired metrics and cases: dimo/train/evaluation/full_dev.',
                  'All old and new checkpoints retained; no Release uploads; TEST unread.']
        (self.root / 'DIMO_REPORT.md').write_text('\n'.join(report) + '\n')
        self.phase = 'complete'; self.state('COMPLETE', teacher_source=record['teacher_source'],
                                          train_manifest_sha256=ready['train_manifest_sha256'])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--previous-experiment', type=Path, required=True)
    parser.add_argument('--secondary-root', type=Path, required=True)
    parser.add_argument('--evaluation-root', type=Path, help='optional separate filesystem for final DEV images')
    parser.add_argument('--allow-shared-gpus', action='store_true', help='explicit user-authorized GPU coexistence')
    args = parser.parse_args(); args.upload_curves = False; args.monitor_token_file = None
    args.experiment_root = args.experiment_root.resolve(); args.previous_experiment = args.previous_experiment.resolve()
    args.secondary_root = args.secondary_root.resolve()
    if args.evaluation_root is not None: args.evaluation_root = args.evaluation_root.resolve()
    lock = (args.experiment_root / '.dimo_handoff.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run = AutoRun(args)
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('automatic DiMO handoff interrupted')))
    try:
        run.execute()
    except BaseException as exc:
        run.failed.set(); run.stop_children()
        run.state('STOPPED_HARD_ERROR', error_class=type(exc).__name__, error=str(exc), automatic_failure_retry=False)
        raise
    finally:
        for monitor in run.monitors: monitor.terminate()
        if run.source_lock is not None: run.source_lock.close()
        lock.close()


if __name__ == '__main__':
    main()
