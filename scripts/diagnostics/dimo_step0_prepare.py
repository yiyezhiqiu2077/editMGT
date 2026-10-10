"""Record immutable protocol and production-file integrity before diagnosis."""
from pathlib import Path
import argparse,importlib.metadata,json,platform,subprocess,sys
from dimo_step0_runtime import load_legacy,verify_assets,sha,LABEL


def main():
    p=argparse.ArgumentParser();p.add_argument('--previous-eval',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[2];a.output.mkdir(parents=True,exist_ok=True)
    branch=subprocess.check_output(['git','-C',str(repo),'branch','--show-current'],text=True).strip()
    if branch!='exp/e3-dense-fp32-v1':raise RuntimeError('Expected existing maintenance branch')
    legacy=load_legacy(a.previous_eval);_,_,identity=verify_assets(a.previous_eval,legacy)
    protected=list((repo/'src/dimo').glob('*.py'))+list((repo/'scripts/train').glob('*.py'))+list((repo/'configs/dimo').glob('*'))
    before={str(p.relative_to(repo)):sha(p) for p in protected if p.is_file()}
    path=a.output/'production_integrity_before.json'
    if path.exists() and json.loads(path.read_text())!=before:raise RuntimeError('Production integrity changed since previous preparation')
    path.write_text(json.dumps(before,indent=2)+'\n')
    protocol=json.loads((a.previous_eval/'protocol.json').read_text())
    protocol.update(schema='reconstructed-lora-dimo-step0-diagnosis-v1',checkpoint_status=LABEL,
        diagnostic_branch=branch,diagnostic_parent_git_sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        checkpoint_not_available_confirmed_by_user=True,rinit_values=[.5,1.],rinit_ablation_is_training_config=False,
        timestep_tied_to_actual_initial_mask_ratio=True,training_started=False,optimizer_steps=0,
        timing_scope='conditioning cached; do not compare latency with earlier evaluator',
        manifests={s:{'path':str(a.previous_eval/'data'/f'{s}.jsonl'),'sha256':sha(a.previous_eval/'data'/f'{s}.jsonl'),
            'rows':len((a.previous_eval/'data'/f'{s}.jsonl').read_text().splitlines())} for s in ['pilot128','full528']})
    if [protocol['manifests'][s]['rows'] for s in ['pilot128','full528']]!=[128,527]:raise RuntimeError('Frozen DEV sample counts changed')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (a.output/'asset_identity_preflight.json').write_text(json.dumps(identity,indent=2)+'\n')
    env={'python':sys.version,'platform':platform.platform(),'packages':sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions() if d.metadata.get('Name'))}
    (a.output/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
    print('PREPARATION_PASS no model weights loaded or saved',flush=True)


if __name__=='__main__':main()
