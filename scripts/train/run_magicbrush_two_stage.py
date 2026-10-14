#!/usr/bin/env python3
"""Four-A800 staged experiment. Hard errors stop; only planned resumes occur."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal
import threading
import fcntl
import shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.explicit_region.config import load_config
from src.explicit_region.contracts import sha256_file
from src.explicit_region.checkpoint import recipe_fingerprint
from src.explicit_region.dense_checkpoint import verify_dense_checkpoint,write_json
from src.explicit_region.magicbrush_experiment import verify_magicbrush_ready
from src.dimo.magicbrush_selection import RULES,register,select,export_teacher
from src.dimo.runtime import gpu_admission
from scripts.train.compare_dense_resume import compare as compare_sft,exact
from src.dimo.resume_comparison import compare_complete_checkpoints


def plan_for(root,previous,sft):
    protocol=json.loads((previous/'protocol.json').read_text())
    steps=sft['checkpoint_steps']
    return {'schema':'magicbrush-two-stage-evaluation-v1','selection_scope':'magicbrush_exploratory',
        'model_root':sft['model']['repo_or_root'],'baseline_checkpoint':sft['initial_dense_checkpoint'],
        'teacher_checkpoint':str(root/'selected_teacher'),
        'train_manifest':sft['data']['manifest'],'dev_manifest':str(root/'data/dev_full527.jsonl'),
        'dev32_manifest':str(root/'data/dev32.jsonl'), 'candidate_steps':steps,
        'candidate_checkpoints':{str(step):str(root/'sft/train'/f'checkpoint-{step}') for step in steps},
        'selection_rules':RULES,'bootstrap':protocol['bootstrap'],
        'embedding_models':{'dino':str(previous/'assets/dinov2-base'),'clip':str(previous/'assets/clip-vit-large-patch14')},
        'inference':{'resolution':1024,'generation_seeds':[0,1], 'teacher_steps':12,'teacher_cfg':5.,
            'reference_strength':1.,'timestep':'roi_relative','student_steps':1,'student_cfg':1.,
            'r_init':1.,'sigma':.3,'temperature':1.,'top_k':0,'top_p':0.,
            'parameters':'fp32','forward':'bf16_autocast','vq':'fp32','attention_backend':'math_sdp','tf32':False,
            'token_mask':sft['token_mask']}}


def memory_peaks(root,stage):
    pattern='memory-rank*.jsonl' if stage=='sft' else 'memory_rank*.jsonl'
    result={}
    for file in root.glob(pattern):
        rows=[json.loads(l) for l in file.read_text().splitlines()]
        result[file.stem]=max((r.get('peak') or r.get('peak_allocated_bytes') or 0) for r in rows)
    if len(result)!=4:raise RuntimeError('MAGICBRUSH_FOUR_RANK_MEMORY_EVIDENCE_MISSING')
    return result


def pipeline_disk_budget(root,parameters):
    # Retain every smoke and planned checkpoint, plus one atomic full save.
    # SFT: four smoke checkpoints and six epoch candidates (weights+AdamW).
    # DiMO: four smoke resumes, three initial exports, six formal resumes
    # and six formal raw/EMA exports. No deletion is needed for this budget.
    factors={'sft_smoke':4*12,'sft_candidates':6*12,'selected_teacher':4,
             'dimo_smoke':4*28+3*8,'dimo_training':6*28+6*8,'atomic_full_save':28}
    ledger={name:factor*parameters for name,factor in factors.items()}
    images_and_logs=32*1024**3
    required=int(sum(ledger.values())*1.15)+images_and_logs
    free=shutil.disk_usage(root).free
    result={'status':'PASS' if free>=required else 'FAIL','parameters_per_role':parameters,
            'estimated_bytes':ledger,'images_and_logs_margin_bytes':images_and_logs,
            'required_bytes':required,'free_bytes':free,'retention':'all checkpoints; no automatic deletion'}
    write_json(root/'admission/DISK_BUDGET.json',result)
    if free<required:raise RuntimeError('MAGICBRUSH_WHOLE_PIPELINE_DISK_BUDGET_FAILED')
    return result


class Run:
    def __init__(self,args):
        self.a=args;self.root=args.experiment_root.resolve();self.phase='admission';self.monitors=[]
        self.sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        self.env=os.environ.copy()
        for key in ('GH_TOKEN','GITHUB_TOKEN'):self.env.pop(key,None)
        self.env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='4',
                        CUBLAS_WORKSPACE_CONFIG=':4096:8',NCCL_ALGO='Ring',NCCL_PROTO='Simple',
                        MB_EXPERIMENT_ROOT=str(self.root))
        self.logs=self.root/'coordinator_logs';self.logs.mkdir(exist_ok=True)
        self.children=set();self.children_lock=threading.Lock();self.failed=threading.Event()
    def state(self,status,**values):
        payload={'status':status,'phase':self.phase,'git_sha':self.sha,'pid':os.getpid(),
                 'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**values}
        temp=self.root/'STATUS.partial.json';write_json(temp,payload);os.replace(temp,self.root/'STATUS.json')
        print(json.dumps(payload),flush=True)
    def command(self,argv,name,*,env=None):
        if self.failed.is_set():raise RuntimeError('MAGICBRUSH_PEER_PROCESS_FAILED')
        argv=[str(x) for x in argv]
        with (self.logs/(name+'.log')).open('a') as log:
            log.write(json.dumps({'argv':argv,'git_sha':self.sha})+'\n');log.flush()
            process=subprocess.Popen(argv,cwd=ROOT,env=env or self.env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with self.children_lock:self.children.add(process)
            try:
                code=process.wait()
                if code:
                    self.failed.set();self.stop_children()
                    raise subprocess.CalledProcessError(code,argv)
            finally:
                with self.children_lock:self.children.discard(process)
    def stop_children(self):
        with self.children_lock:children=list(self.children)
        for process in children:
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
    def train(self,stage,output,stop=None,resume=None):
        self.phase=f'{stage}:{output.name}:step{stop or "full"}';self.state('RUNNING')
        gpu_admission(magicbrush=True, allow_other_jobs=self.env.get('EDITMGT_ALLOW_SHARED_A800') == '1')
        entry='train_dense_region.py' if stage=='sft' else 'train_full_dense_dimo.py'
        argv=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',
              str(ROOT/'scripts/train'/entry),'--config',str(self.root/'configs'/f'{stage}.yaml'),
              '--output-dir',str(output)]
        if stop is not None:argv += ['--stop-at-global-step',str(stop)]
        if resume:argv += ['--resume',str(resume)]
        self.command(argv,f'{stage}_{output.name}_to{stop if stop is not None else "full"}')
    def monitor(self,stage,train):
        label='e3' if stage=='sft' else 'dimo'
        argv=[sys.executable,str(ROOT/'scripts/monitor/live_monitor.py'),'--stage',label,
              '--log',str(train/('train_metrics.jsonl' if stage=='sft' else 'metrics.jsonl')),
              '--output',str(self.root/stage/'curves'),'--run-id',self.root.name+'-'+stage,
              '--git-sha',self.sha]
        if self.a.upload_curves:argv+=['--upload']
        if self.a.monitor_token_file:argv += ['--token-file',str(self.a.monitor_token_file)]
        log=(self.logs/f'monitor_{stage}.log').open('a')
        process=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();self.monitors.append(process)
        return process
    def eval_teacher(self,checkpoint,output,gpu):
        env=self.env|{'CUDA_VISIBLE_DEVICES':str(gpu)}
        self.command([sys.executable,ROOT/'scripts/eval/evaluate_magicbrush_dense.py',
            '--experiment-root',self.root,'--checkpoint',checkpoint,'--output',output],f'eval_{output.name}',env=env)
    def dimo_eval(self,train,step,dev32=True):
        self.phase=f'dimo:dev{32 if dev32 else "full"}:step{step}';self.state('RUNNING')
        argv=[sys.executable,ROOT/'scripts/eval/evaluate_magicbrush_dense.py','--experiment-root',self.root,
              '--dimo-root',train,'--step',str(step)]
        if dev32:argv+=['--dev32']
        self.command(argv,f'dimo_dev{32 if dev32 else "full"}_{step}',env=self.env|{'CUDA_VISIBLE_DEVICES':'0'})
    def acceptance(self,stage,smoke,fresh,resumed):
        if stage=='sft':
            result=compare_sft(fresh/'checkpoint-20',resumed/'checkpoint-20',fresh_log=fresh/'train_metrics.jsonl',
                               interrupted_logs=[resumed/'train_metrics.jsonl'])
        else:
            result=compare_complete_checkpoints(fresh/'checkpoints/checkpoint-20',resumed/'checkpoints/checkpoint-20')
            for rank in range(4):
                def trace(root):return [json.loads(l) for l in (root/f'trace_rank{rank}.jsonl').read_text().splitlines()]
                exact(trace(fresh),trace(resumed),f'sample_uid_rank{rank}')
            result['sample_uid_sequence']='ALL_FOUR_RANKS_BITWISE_EXACT'
        result.update(gpu_peak_allocated_bytes={'smoke2':memory_peaks(smoke,stage),
            'fresh20':memory_peaks(fresh,stage),'resume20':memory_peaks(resumed,stage)},
            evidence='REAL_4_A800',no_recipe_adjustment=True)
        write_json(self.root/'admission'/f'{stage.upper()}_CUDA_ACCEPTANCE.json',result)
    def execute(self):
        if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
            raise RuntimeError('MAGICBRUSH_REQUIRES_CLEAN_COMMITTED_CODE')
        if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='exp/e3-dense-fp32-v1':
            raise RuntimeError('MAGICBRUSH_REQUIRES_EXISTING_MAINTENANCE_BRANCH')
        if (self.root/'PIPELINE_STARTED.json').exists():raise RuntimeError('MAGICBRUSH_PIPELINE_ALREADY_STARTED_NO_IMPLICIT_RECOVERY')
        cpu=json.loads((self.root/'admission/CPU_ACCEPTANCE.json').read_text())
        if cpu.get('status')!='PASS' or cpu.get('git_sha')!=self.sha:raise RuntimeError('MAGICBRUSH_CPU_ACCEPTANCE_REQUIRED')
        sft=load_config(self.root/'configs/sft.yaml');dimo=load_config(self.root/'configs/dimo.yaml')
        verify_magicbrush_ready(sft);verify_magicbrush_ready(dimo)
        merge=json.loads((self.root/'admission/merge_holdout_v2/MERGE_PARITY.json').read_text())
        fp64=json.loads((self.root/'admission/MERGE_FP64_COMPLETE_REFERENCE.json').read_text())
        if merge['status']!='PASS' or fp64['static_semantics_status']!='PASS':raise RuntimeError('MAGICBRUSH_MERGE_ACCEPTANCE_REQUIRED')
        verify_dense_checkpoint(sft['initial_dense_checkpoint'])
        pipeline_disk_budget(self.root,merge['parameter_count'])
        gpu_admission(magicbrush=True)
        plan=plan_for(self.root,self.a.previous_eval.resolve(),sft)
        write_json(self.root/'evaluation_plan.json',plan)
        write_json(self.root/'PIPELINE_STARTED.json',{'git_sha':self.sha,'plan_sha256':recipe_fingerprint(plan),'selection_rules':RULES})
        smoke=self.root/'sft/smoke2';fresh=self.root/'sft/fresh20';resumed=self.root/'sft/resume20'
        self.train('sft',smoke,2);self.train('sft',fresh,20);self.train('sft',resumed,10)
        self.train('sft',resumed,20,resumed/'checkpoint-10');self.acceptance('sft',smoke,fresh,resumed)
        train=self.root/'sft/train';self.monitor('sft',train);self.train('sft',train)
        self.phase='sft:registered_full_dev_selection';self.state('RUNNING')
        registration=register(plan,self.root/'selection_preregistration.json')
        jobs=[('baseline',plan['baseline_checkpoint'])]+[(str(step),plan['candidate_checkpoints'][str(step)]) for step in plan['candidate_steps']]
        # One serial queue per GPU. No simultaneous workers share the same GPU.
        def queue(gpu):
            for name,checkpoint in jobs[gpu::4]:
                self.eval_teacher(checkpoint,self.root/'sft/dev_evaluation'/name,gpu)
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(queue,range(4)))
        evaluation=self.root/'sft/dev_evaluation'
        baseline=json.loads((evaluation/'baseline/metrics.json').read_text())
        candidates={step:json.loads((evaluation/str(step)/'metrics.json').read_text()) for step in plan['candidate_steps']}
        from src.explicit_region.dense_evaluation import assert_matching_predictions,comparison_visual
        for name,_ in jobs[1:]:
            assert_matching_predictions([evaluation/'baseline/predictions.jsonl',evaluation/name/'predictions.jsonl'],expected_seeds=[0,1])
            comparison_visual({'Merged baseline':evaluation/'baseline/predictions.jsonl','Dense SFT':evaluation/name/'predictions.jsonl'},
                              evaluation/name/'comparison.jpg',count=8)
        record=select(plan,registration,baseline,candidates);write_json(self.root/'TEACHER_SELECTION.json',record)
        export_teacher(record,self.root/'selected_teacher')
        self.command([sys.executable,ROOT/'scripts/report/report_magicbrush_two_stage.py','--experiment-root',self.root,'--stage','sft'], 'sft_report')
        smoke=self.root/'dimo/smoke2';fresh=self.root/'dimo/fresh20';resumed=self.root/'dimo/resume20'
        self.train('dimo',smoke,2);self.train('dimo',fresh,20);self.train('dimo',resumed,10)
        self.train('dimo',resumed,20,resumed/'checkpoints/checkpoint-10');self.acceptance('dimo',smoke,fresh,resumed)
        train=self.root/'dimo/train';self.monitor('dimo',train)
        self.train('dimo',train,0);self.dimo_eval(train,0)
        previous=0
        for stop in dimo['resume_checkpoint_steps']:
            self.train('dimo',train,stop,train/'checkpoints'/f'checkpoint-{previous}')
            self.dimo_eval(train,stop);previous=stop
        for step in dimo['evaluation_steps']:self.dimo_eval(train,step,dev32=False)
        self.command([sys.executable,ROOT/'scripts/report/report_magicbrush_two_stage.py','--experiment-root',self.root,'--stage','dimo'], 'dimo_report')
        self.phase='complete';self.state('COMPLETE',teacher_source=record['teacher_source'],sft_steps=sft['max_optimizer_steps'],dimo_steps=dimo['max_optimizer_steps'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment-root',type=Path,required=True)
    p.add_argument('--previous-eval',type=Path,required=True);p.add_argument('--monitor-token-file',type=Path)
    p.add_argument('--upload-curves',action='store_true',help='Only enable when the user explicitly requests Release uploads')
    a=p.parse_args()
    lock=(a.experiment_root/'.coordinator.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (a.experiment_root/'PIPELINE_STARTED.json').exists():raise RuntimeError('PIPELINE_ALREADY_STARTED_NO_IMPLICIT_RECOVERY')
    run=Run(a)
    try:run.execute()
    except BaseException as exc:
        run.failed.set();run.stop_children()
        run.state('STOPPED_HARD_ERROR',error_class=type(exc).__name__,error=str(exc),
                  automatic_recipe_changes=False,automatic_failure_retry=False)
        raise
    finally:
        for monitor in run.monitors:
            monitor.terminate()
        lock.close()


if __name__=='__main__':main()
