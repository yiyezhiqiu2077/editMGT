"""GPU inference/audit coordinator. Never launches a training entry point."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--previous-eval',type=Path,required=True)
    p.add_argument('--work-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',type=int,nargs='+',default=[1,2,3]);p.add_argument('--audit-gpu',type=int,default=0)
    p.add_argument('--wait-for-running-pilot',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);a.work_root.mkdir(parents=True,exist_ok=True)
    def status(state,**more):
        x={'state':state,'utc':datetime.now(timezone.utc).isoformat(),'checkpoint_status':'RECONSTRUCTED_STEP0',**more}
        temp=a.output/'STATUS.json.tmp';temp.write_text(json.dumps(x,indent=2)+'\n');temp.replace(a.output/'STATUS.json')
        print(json.dumps(x),flush=True)
    def idle(gpu):
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        for row in rows.splitlines():
            i,m,u=map(int,row.split(','))
            if i==gpu:return m<128 and u<5
        return False
    def launch(script,args,gpu,log):
        deadline=time.time()+3600
        while not idle(gpu):
            if time.time()>deadline:raise RuntimeError(f'No idle GPU {gpu}')
            status('WAITING_FOR_IDLE_GPU',gpu=gpu);time.sleep(20)
        handle=log.open('w')
        env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'}
        process=subprocess.Popen([sys.executable,'-u',str(HERE/script),*args],env=env,stdout=handle,stderr=subprocess.STDOUT)
        return process,handle,log
    def wait_all(jobs,state):
        while True:
            pending=[]
            for process,handle,log in jobs:
                code=process.poll()
                if code is None:pending.append({'pid':process.pid,'log':str(log)})
                elif code:raise RuntimeError(f'Diagnostic worker exit {code}: {log}')
            status(state,active=pending)
            if not pending:break
            time.sleep(20)
        for _,handle,_ in jobs:handle.close()
    common=['--previous-eval',str(a.previous_eval),'--work-root',str(a.work_root)]
    jobs=[]
    try:
        if not (a.work_root/'PREFLIGHT.json').exists():
            jobs=[launch('dimo_step0_preflight.py',common,a.audit_gpu,a.work_root/'preflight.log')]
            wait_all(jobs,'CHECKING_ORIGINAL_INFERENCE_EQUIVALENCE')
        completion=a.work_root/'pilot128/shard0/COMPLETED.json'
        if a.wait_for_running_pilot:
            deadline=time.time()+7200
            while not completion.exists():
                if time.time()>deadline:raise RuntimeError('Waiting for externally running DEV128 timed out')
                status('DEV128_RUNNING');time.sleep(20)
        elif not completion.exists():
            jobs=[launch('dimo_step0_worker.py',common+['--stage','pilot128'],a.gpus[0],a.work_root/'pilot128.log')]
            wait_all(jobs,'DEV128_RUNNING')
        completed=json.loads(completion.read_text())
        if completed['samples']!=128 or not completed['student_unchanged']:raise RuntimeError('DEV128_PREFLIGHT_FAILED')
        # All finite image metrics, 2 seeds, one forward, and exterior lock required.
        for variant in ['r05','r10']:
            data=json.loads((completion.parent/variant/'metrics.json').read_text())['per_generation']
            if len(data)!=256:raise RuntimeError('DEV128_SAMPLE_COUNT_FAILED')
            for row in data:
                if row['transformer_forwards']!=1 or row['outside_token_mismatch_count']!=0:raise RuntimeError('DEV128_TOKEN_CONTRACT_FAILED')
                for n in ['inside_masked_lpips','inside_psnr','inside_ssim','outside_masked_lpips']:
                    import math
                    if not math.isfinite(row['metrics'][n]):raise RuntimeError('DEV128_NONFINITE_METRIC')
        jobs=[]
        for shard,gpu in enumerate(a.gpus):
            if not (a.work_root/f'full528/shard{shard}/COMPLETED.json').exists():
                jobs.append(launch('dimo_step0_worker.py',common+['--stage','full528','--shard',str(shard),'--shards',str(len(a.gpus))],gpu,a.work_root/f'full528_shard{shard}.log'))
        wait_all(jobs,'FULL_DEV_RUNNING')
        if not (a.output/'surrogate_real_logits.json').exists():
            jobs=[launch('dimo_step0_real_audit.py',['--previous-eval',str(a.previous_eval),'--output',str(a.output)],a.audit_gpu,a.work_root/'real_audit.log')]
            wait_all(jobs,'REAL_LOGIT_AUDIT_RUNNING')
        status('SUMMARIZING')
        subprocess.run([sys.executable,str(HERE/'dimo_step0_summarize.py'),*common,'--output',str(a.output),'--full-shards',str(len(a.gpus))],check=True)
        status('RESULTS_READY_REQUIRES_REPORT_AND_PACKAGE')
    except Exception as exc:
        for process,handle,_ in jobs:
            if process.poll() is None:process.terminate()
            handle.close()
        status('FAILED',error=str(exc));raise


if __name__=='__main__':main()
