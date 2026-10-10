"""Local exploratory selection with the unique LoRA-merged Dense fallback."""
import json
import os
from pathlib import Path
import tempfile
from src.explicit_region.checkpoint import recipe_fingerprint
from src.explicit_region.contracts import sha256_file
from src.explicit_region.dense_checkpoint import verify_dense_checkpoint, write_json
from src.explicit_region.dense_evaluation import paired_comparison

RULES={'primary':'inside_masked_lpips','minimum_inside_lpips_improvement':.005,
       'maximum_outside_lpips_degradation':.01,'inside_ci_upper_strictly_negative':True,
       'bootstrap_seed':42,'bootstrap_resamples':10000,'confidence_level':.95,
       'bootstrap_unit':'source_group_cluster','tie_break':'earlier_step',
       'fallback':'E3_EPOCH1_MERGED_DENSE','candidate_epochs':[1,2,3,5,7,10]}


def choose(candidates):
    if not candidates:raise RuntimeError('MAGICBRUSH_SELECTION_REQUIRES_ALL_CANDIDATES')
    eligible=[]
    for row in candidates:
        inside,outside=row['paired_inside'],row['paired_outside']
        row['eligible']=(row['identity_protocol_valid'] is True and row['numerical_roi_valid'] is True
                         and inside['mean_delta'] <= -.005 and inside['bootstrap_95_ci'][1] < 0
                         and outside['mean_delta'] <= .01)
        if row['eligible']:eligible.append(row)
    return min(eligible,key=lambda r:(r['inside_mean'],r['step'])) if eligible else None


def register(plan, output):
    rows=sum(1 for line in Path(plan['train_manifest']).read_text().splitlines() if line.strip())
    if plan['candidate_steps'] != [e*(rows//16) for e in RULES['candidate_epochs']] or rows%16:
        raise RuntimeError('MAGICBRUSH_SELECTION_REGISTERED_CANDIDATES_CHANGED')
    baseline=verify_dense_checkpoint(plan['baseline_checkpoint'])
    payload={'schema':'magicbrush-selection-prereg-v1','selection_scope':'magicbrush_exploratory',
             'rules':RULES,'plan':plan,'baseline_checkpoint_hash':baseline['checkpoint_sha256'],
             'train_manifest_sha256':sha256_file(plan['train_manifest']),
             'dev_manifest_sha256':sha256_file(plan['dev_manifest']),
             'dev32_manifest_sha256':sha256_file(plan['dev32_manifest']),
             'candidate_checkpoint_hashes':{str(s):verify_dense_checkpoint(plan['candidate_checkpoints'][str(s)])['checkpoint_sha256']
                                            for s in plan['candidate_steps']}}
    payload['sha256']=recipe_fingerprint(payload)
    output=Path(output)
    if output.exists():
        if json.loads(output.read_text())!=payload:raise RuntimeError('MAGICBRUSH_SELECTION_PREREGISTRATION_CHANGED')
    else:write_json(output,payload)
    return payload


def select(plan, registration, baseline_metrics, candidate_metrics):
    if registration['rules']!=RULES or registration['plan']!=plan:
        raise RuntimeError('MAGICBRUSH_SELECTION_RULE_OR_PLAN_CHANGED')
    if set(candidate_metrics)!=set(plan['candidate_steps']):
        raise RuntimeError('MAGICBRUSH_SELECTION_MISSING_CANDIDATES')
    if registration['sha256']!=recipe_fingerprint({k:v for k,v in registration.items() if k!='sha256'}):
        raise RuntimeError('MAGICBRUSH_SELECTION_REGISTRATION_HASH_CHANGED')
    candidates=[]
    for step in plan['candidate_steps']:
        result=candidate_metrics[step]
        pairs=paired_comparison(result,baseline_metrics,strict_groups=True)['magicbrush_dev']
        source=plan['candidate_checkpoints'][str(step)]
        identity=verify_dense_checkpoint(source)
        if identity['checkpoint_sha256']!=registration['candidate_checkpoint_hashes'][str(step)]:
            raise RuntimeError('MAGICBRUSH_CANDIDATE_WEIGHTS_CHANGED')
        inside=[r['metrics']['inside_masked_lpips'] for r in result['per_sample_after_seed_mean']]
        candidates.append({'step':step,'checkpoint':source,'checkpoint_hash':identity['checkpoint_sha256'],
                           'inside_mean':sum(inside)/len(inside),'paired_inside':pairs['inside_masked_lpips'],
                           'paired_outside':pairs['outside_masked_lpips'],
                           'identity_protocol_valid':True,'numerical_roi_valid':True})
    best=choose(candidates)
    source=best['checkpoint'] if best else plan['baseline_checkpoint']
    identity=verify_dense_checkpoint(source)
    return {'schema':'magicbrush-teacher-selection-v1','selection_scope':'magicbrush_exploratory',
            'teacher_source':'dense_sft_selected' if best else 'lora_merged_fallback',
            'fallback_reason':None if best else 'NO_ELIGIBLE_DENSE_SFT_CANDIDATE',
            'selected_checkpoint':source,'selected_step':best['step'] if best else 0,
            'selected_checkpoint_hash':identity['checkpoint_sha256'],
            'baseline_checkpoint_hash':registration['baseline_checkpoint_hash'],
            'dev_plan_hash':recipe_fingerprint(plan),'quality_metrics':candidates,
            'selection_status':'SELECTED' if best else 'FALLBACK', 'formal_teacher':False,
            'selection_preregistration':registration}


def verify_selection_record(record, checkpoint_hash):
    prereg=record.get('selection_preregistration',{})
    if (record.get('schema')!='magicbrush-teacher-selection-v1' or record.get('selection_scope')!='magicbrush_exploratory'
            or record.get('formal_teacher') is not False or record.get('selection_status') not in ('SELECTED','FALLBACK')
            or record.get('selected_checkpoint_hash')!=checkpoint_hash or prereg.get('rules')!=RULES
            or prereg.get('sha256')!=recipe_fingerprint({k:v for k,v in prereg.items() if k!='sha256'})
            or record.get('dev_plan_hash')!=recipe_fingerprint(prereg.get('plan',{}))):
        raise RuntimeError('MAGICBRUSH_EXPLORATORY_TEACHER_RECORD_INVALID')
    expected=prereg['plan']['candidate_steps']
    candidates=record['quality_metrics']
    if [r['step'] for r in candidates]!=expected:raise RuntimeError('MAGICBRUSH_SELECTION_CANDIDATE_SET_CHANGED')
    best=choose(candidates)
    expected_hash=best['checkpoint_hash'] if best else prereg['baseline_checkpoint_hash']
    if (checkpoint_hash!=expected_hash or record['teacher_source'] !=
            ('dense_sft_selected' if best else 'lora_merged_fallback')):
        raise RuntimeError('MAGICBRUSH_TEACHER_DOES_NOT_MATCH_SELECTION')


def export_teacher(record, output):
    from .dense_teacher import export_dense_teacher,load_dense_teacher_contract
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    verify_selection_record(record,record['selected_checkpoint_hash'])
    temporary=Path(tempfile.mkdtemp(prefix='.magicbrush-teacher-',dir=output.parent))
    target=temporary/'teacher'
    manifest=export_dense_teacher(record['selected_checkpoint'],target,status='provisional')
    write_json(target/'selection_record.json',record)
    manifest.update(status='magicbrush_exploratory',selection_scope='magicbrush_exploratory',formal_teacher=False,
                    teacher_source=record['teacher_source'],selected_checkpoint_identity=record['selected_checkpoint_hash'],
                    selection_record_sha256=sha256_file(target/'selection_record.json'))
    write_json(target/'dimo_teacher_manifest.json',manifest)
    identity=json.loads((target/'identity.json').read_text())
    ledger={p.name:sha256_file(p) for p in target.iterdir() if p.name!='COMPLETED.json'}
    write_json(target/'COMPLETED.json',{'schema':identity['schema'],'files':ledger})
    load_dense_teacher_contract(target)
    os.rename(target,output);temporary.rmdir()
    return manifest
