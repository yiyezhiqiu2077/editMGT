#!/usr/bin/env python3
"""Freeze a native, whole-session train subset and the existing aligned DEV."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
import yaml
from audit_magicbrush_training_input import session_subset
from src.explicit_region.canonical import SCHEMA_VERSION,sample_uid_for
from src.explicit_region.contracts import sha256_file
from src.explicit_region.dense_checkpoint import write_json
from src.explicit_region.config import load_config
from src.explicit_region.fixed_dataset import FixedCorpusDataset


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--previous-eval',type=Path,required=True)
    p.add_argument('--experiment-root',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2)
    out=a.experiment_root/'data'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    audit=json.loads(a.audit.read_text())
    if audit['train_dev_group_leakage'] or audit['train_dev_exact_content_leakage']:
        raise RuntimeError('MAGICBRUSH_SPLIT_LEAKAGE_STOP')
    plan=session_subset(audit['records'])
    write_json(out/'SESSION_SUBSET.json',plan)
    keep=set(plan['kept_sample_keys']);rows=[]
    images=out/'train/images';images.mkdir(parents=True)
    revision=audit['dataset_revision']
    for path in sorted((a.snapshot/'data').glob('train-*.parquet')):
        # Raw audit hash binding is independently checked again before use.
        expected=next(r['sha256'] for r in audit['parquets'] if r['name']==path.name)
        if sha256_file(path)!=expected:raise RuntimeError('MAGICBRUSH_RAW_CHANGED_AFTER_AUDIT')
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
            for raw in batch.to_pylist():
                key=f"{raw['img_id']}_turn{raw['turn_index']}"
                if key not in keep:continue
                locators={};hashes={}
                for role,column in (('source','source_img'),('target','target_img')):
                    target=images/f'{key}_{role}.png';target.write_bytes(raw[column]['bytes'])
                    locators[role]={'backend':'file','relative_path':str(target.relative_to(out/'train'))}
                    hashes[role]=sha256_file(target)
                with Image.open(io.BytesIO(raw['mask_img']['bytes'])) as native:
                    if native.mode!='RGBA':raise RuntimeError('OFFICIAL_MASK_SEMANTICS_CHANGED')
                    mask=Image.fromarray(np.where(np.asarray(native.getchannel('A'))<255,255,0).astype('uint8'))
                target=images/f'{key}_region.png';mask.save(target)
                locators['region']={'backend':'file','relative_path':str(target.relative_to(out/'train'))}
                hashes['region']=sha256_file(target)
                row={'schema_version':SCHEMA_VERSION,'dataset_name':'magicbrush','dataset_revision':revision,
                     'group_id':str(raw['img_id']),'official_sample_key':key,'turn_index':int(raw['turn_index']),
                     'manifest_index':len(rows),'instruction_original':raw['instruction'],'instruction_en':raw['instruction'],
                     'language_original':'en','edit_type_original':'unknown','edit_type_canonical':'other',
                     'mask_semantics':'edit_region','region_fraction':float((np.asarray(mask)>0).mean()),
                     'translation_status':'passthrough_en','translation_cache_key':None,
                     **{r+'_locator':v for r,v in locators.items()},**{r+'_sha256':v for r,v in hashes.items()}}
                row['sample_uid']=sample_uid_for(row);rows.append(row)
        print(json.dumps({'materialized_train_rows':len(rows),'file':path.name}),flush=True)
    if len(rows)!=plan['kept_rows']:raise RuntimeError('MAGICBRUSH_SUBSET_COUNT_MISMATCH')
    manifest=out/'train_8800.jsonl' # Filename identifies nominal requested experiment; total_rows is authoritative.
    manifest.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    write_json(out/'KEPT_UIDS.json',[r['sample_uid'] for r in rows])
    # Preserve exactly the previous contain/pad geometry, sample order, pixels and masks.
    full=a.previous_eval/'data/full528.jsonl';pilot=a.previous_eval/'data/pilot128.jsonl'
    for name,source in (('dev_full527.jsonl',full),('dev128.jsonl',pilot)):
        (out/name).write_bytes(source.read_bytes())
    dev=[json.loads(l) for l in pilot.read_text().splitlines()][:32]
    (out/'dev32.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in dev))
    ds=FixedCorpusDataset(manifest,{'magicbrush':out/'train'},resolution=1024,base_seed=42,verify_hashes=True,
                         max_random_attempts=8,minimum_mask_retention=.75)
    for i in range(len(ds)):
        ds[i]
        if (i+1)%500==0:print(f'CANONICAL_LOADER_VERIFIED={i+1}',flush=True)
    files={p.name:{'path':str(p.resolve()),'sha256':sha256_file(p)} for p in (
        manifest,out/'SESSION_SUBSET.json',out/'KEPT_UIDS.json',out/'dev_full527.jsonl',out/'dev128.jsonl',out/'dev32.jsonl',a.audit)}
    ready={'schema':'magicbrush-training-ready-v1','status':'VERIFIED','selection_scope':'magicbrush_exploratory',
           'total_rows':len(rows),'train_manifest_sha256':sha256_file(manifest),'files':files,
           'session_consistent':True,'train_dev_leakage':'PASS','loader_geometry':'PASS','padding':False,
           'dataset_revision':revision,'raw_audit_sha256':sha256_file(a.audit),'excluded_rows':plan['excluded_rows'],
           'budget_recalculation':{'reason':'whole_session_geometry_exclusions','requested':8800,'actual':len(rows),
               'sft_steps':10*len(rows)//16,'dimo_steps':10*len(rows)//4}}
    write_json(out/'MAGICBRUSH_READY.json',ready)
    os.environ['MB_EXPERIMENT_ROOT']=str(a.experiment_root.resolve())
    os.environ['EDITMGT_MODEL_ROOT']=str((a.previous_eval/'assets/editmgt').resolve())
    config_out=a.experiment_root/'configs';config_out.mkdir()
    for stage,path in (('sft','configs/train/dense_magicbrush_4a800_10epoch.yaml'),
                       ('dimo','configs/dimo/full_dense_magicbrush_4a800_10epoch.yaml')):
        c=load_config(ROOT/path);c['data']['expected_rows']=len(rows)
        per_epoch=len(rows)//(16 if stage=='sft' else 4)
        c['max_optimizer_steps']=10*per_epoch
        if stage=='sft':
            c['scheduler_horizon_steps']=10*per_epoch
            c['checkpoint_steps']=[e*per_epoch for e in (1,2,3,5,7,10)]
        else:
            c['optimizer']['horizon_steps']=10*per_epoch
            c['resume_checkpoint_steps']=[e*per_epoch for e in (1,3,5,7,10)]
            c['evaluation_steps']=[0,*c['resume_checkpoint_steps']]
        (config_out/f'{stage}.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
    print(json.dumps(ready),flush=True)


if __name__=='__main__':main()
