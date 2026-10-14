"""Package code and lightweight diagnostic evidence, excluding all weights."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,tarfile,tempfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def export(repo,dest,revision):
    dest.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        snapshot=Path(temporary)/'code.tar'
        subprocess.run(['git','-C',str(repo),'archive','-o',str(snapshot),revision],check=True)
        subprocess.run(['tar','-xf',str(snapshot),'-C',str(dest)],check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--previous-eval',type=Path,required=True);p.add_argument('--work-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if subprocess.check_output(['git','-C',str(a.repo),'branch','--show-current'],text=True).strip()!='exp/e3-dense-fp32-v1':
        raise RuntimeError('Maintenance branch changed')
    required=['REPORT.md','step0_metrics.json','rinit_ablation_metrics.json','surrogate_gradient_audit.json',
        'surrogate_gradient_audit.csv','checkpoint_identity.json','protocol.json','CLUSTER_COMMANDS.sh']
    for n in required:
        if not (a.output/n).is_file():raise RuntimeError('Missing required artifact: '+n)
    protected=json.loads((a.output/'production_integrity_before.json').read_text())
    after={name:sha(a.repo/name) for name in protected}
    if after!=protected:raise RuntimeError('Production training code/config changed')
    (a.output/'production_integrity_after.json').write_text(json.dumps({'unchanged':True,'sha256':after},indent=2)+'\n')
    maintenance=a.output/'code/maintenance'
    revision=subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'],text=True).strip()
    export(a.repo,maintenance,revision)
    for source in (a.repo/'scripts/diagnostics').glob('*.py'):
        dest=maintenance/source.relative_to(a.repo);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    test=a.repo/'tests/dimo/test_step0_precision_diagnosis.py'
    shutil.copyfile(test,maintenance/test.relative_to(a.repo))
    compatible=a.output/'code/compatible_lora_runtime'
    export(a.previous_eval/'code',compatible,'3132e24355f947aaeb604278514d9460806de0b6')
    (compatible/'DIAGNOSIS_REVISION.json').write_text(json.dumps({'revision':'3132e24355f947aaeb604278514d9460806de0b6'})+'\n')
    (a.output/'code/REVISIONS.json').write_text(json.dumps({'maintenance_branch':'exp/e3-dense-fp32-v1',
        'maintenance_parent_sha':revision,'diagnostic_tools_are_additions':True,
        'compatible_lora_runtime_sha':'3132e24355f947aaeb604278514d9460806de0b6',
        'historical_training_sha':'c4f92a7c696150c40f07959c72aa435cb6f67624','historical_training_source_unavailable':True},indent=2)+'\n')
    previous=a.output/'code/previous_eval'
    previous.mkdir(parents=True,exist_ok=True)
    for n in ['run_worker.py','summarize.py','prepare_data.py','run_all.py','protocol.json','preflight_checkpoint.json']:
        shutil.copyfile(a.previous_eval/n,previous/n)
    metadata=a.output/'checkpoint_metadata'
    for folder in ['editmgt-e3-quality-epoch1-step6250','editmgt-dimo-e3-step7500']:
        for source in (a.previous_eval/'assets/checkpoints'/folder).glob('*.json'):
            dest=metadata/folder/source.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    manifests=a.output/'frozen_manifests';manifests.mkdir(exist_ok=True)
    for name in ['pilot128.jsonl','full528.jsonl','audit.json']:
        shutil.copyfile(a.previous_eval/'data'/name,manifests/name)
    logs=a.output/'logs';logs.mkdir(exist_ok=True)
    for source in a.work_root.glob('*.log'):shutil.copyfile(source,logs/source.name)
    shutil.copyfile(a.work_root/'PREFLIGHT.json',a.output/'PREFLIGHT.json')
    environment=Path('/data01/zhangyuyang/packages/editmgt_eval_environment.json')
    if environment.exists() and not (a.output/'environment.json').exists():shutil.copyfile(environment,a.output/'environment.json')
    info={'schema':'dimo-step0-diagnosis-package-v1','checkpoint_status':'RECONSTRUCTED_STEP0',
        'repo':str(a.repo),'work_root':str(a.work_root),'previous_eval':str(a.previous_eval),
        'models_and_optimizer_excluded':True,'full_dataset_and_individual_outputs_excluded':True,
        'production_code_and_configs_unchanged':True,'archive_layout':'dimo_step0_diagnosis/'}
    (a.output/'PACKAGE_INFO.json').write_text(json.dumps(info,indent=2)+'\n')
    (a.output/'STATUS.json').write_text(json.dumps({'state':'COMPLETE','checkpoint_status':'RECONSTRUCTED_STEP0',
        'report':'REPORT.md','inference_and_audits_completed':True,'production_unchanged':True},indent=2)+'\n')
    archive=a.output.parent/'dimo_step0_diagnosis.tar.gz'
    if archive.exists():raise RuntimeError('Refusing to overwrite '+str(archive))
    files=sorted(p for p in a.output.rglob('*') if p.is_file() and p.relative_to(a.output).as_posix()!='SHA256SUMS.txt')
    for f in files:
        if f.suffix in ['.pt','.pth','.ckpt','.bin','.safetensors','.pyc'] or f.stat().st_size>100*1024*1024:
            raise RuntimeError('Unexpected payload in lightweight package: '+str(f))
    sums={p.relative_to(a.output).as_posix():sha(p) for p in files}
    (a.output/'SHA256SUMS.txt').write_text(''.join(f'{digest}  {name}\n' for name,digest in sums.items()))
    partial=Path(str(archive)+'.partial')
    subprocess.run(['tar','--sort=name','--owner=0','--group=0','--numeric-owner','-czf',str(partial),
        '-C',str(a.output.parent),a.output.name],check=True)
    actual={}
    with tarfile.open(partial,'r:gz') as t:
        for member in t:
            if member.isfile():
                name=str(Path(member.name).relative_to(a.output.name))
                if name=='SHA256SUMS.txt':continue
                h=hashlib.sha256();handle=t.extractfile(member)
                for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
                actual[name]=h.hexdigest()
    if actual!=sums:raise RuntimeError('Archive file integrity failed')
    partial.replace(archive);digest=sha(archive)
    Path(str(archive)+'.sha256').write_text(f'{digest}  {archive.name}\n')
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':digest,'verified_files':len(actual)}),flush=True)


if __name__=='__main__':main()
