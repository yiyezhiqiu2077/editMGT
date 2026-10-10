"""Verify reconstruction and original/cached one-step equivalence before DEV."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from dimo_step0_runtime import configure_torch,make_generator,sha,state_digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--previous-eval',type=Path,required=True)
    p.add_argument('--work-root',type=Path,required=True);a=p.parse_args();configure_torch()
    g,legacy=make_generator(a.previous_eval)
    row=json.loads((a.previous_eval/'data/pilot128.jsonl').read_text().splitlines()[0])
    results=[]
    for ratio in [.5,1.]:
        image,_,_=g.generate(row,0,ratio)
        logits=g.last_result.logits.clone();initial=g.last_result.initial_tokens.clone()
        legacy.PROTOCOL['dimo']['r_init']=ratio
        old_image,_,_=legacy.Generator.generate(g,row,0)
        result={'r_init':ratio,'sample_key':row['sample_key'],'seed':0,
            'image_pixels_bitwise_equal':np.array_equal(np.asarray(image),np.asarray(old_image)),
            'student_logits_bitwise_equal':torch.equal(logits,g.last_result.logits),
            'initial_tokens_bitwise_equal':torch.equal(initial,g.last_result.initial_tokens),
            'logits_dtype':str(logits.dtype),'logits_shape':list(logits.shape),
            'logits_max_abs':float(logits.abs().max())}
        if not all(result[n] for n in ['image_pixels_bitwise_equal','student_logits_bitwise_equal','initial_tokens_bitwise_equal']):
            raise RuntimeError('CACHED_INFERENCE_NOT_EXACT: '+str(result))
        results.append(result)
    ck=a.previous_eval/'assets/checkpoints/editmgt-dimo-e3-step7500'
    marker=json.loads((ck/'CHECKPOINT_READY.json').read_text())
    fingerprint=sha(ck/'training_state.pt')
    if fingerprint!=marker['training_state_sha256']:raise RuntimeError('HISTORICAL_STEP7500_CONTENT_CHANGED')
    report={'status':'PASS','identity':g.identity,'original_inference_equivalence':results,
        'historical_step7500_training_state_sha256':fingerprint,
        'historical_checkpoint_binary_loaded':False,'student_unchanged':state_digest(g.roles.role_state_dict('student'))==g.identity['role_state_sha256']['student']}
    temporary=a.work_root/'PREFLIGHT.json.tmp'
    temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(a.work_root/'PREFLIGHT.json')
    print(json.dumps({'status':'PASS','equivalence':results}),flush=True)


if __name__=='__main__':main()
