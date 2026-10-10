"""Source-group paired statistics and compact diagnostic report artifacts."""
from pathlib import Path
from collections import defaultdict
import argparse,json,math,textwrap
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dimo_step0_runtime import LABEL,sha
from surrogate_precision import write_csv


def grouped(rows):
    buckets=defaultdict(list)
    for r in rows:buckets[r['sample_key']].append(r)
    result={}
    for key,items in buckets.items():
        if sorted(r['seed'] for r in items)!=[0,1]:raise RuntimeError('Incomplete paired seeds')
        names=set().union(*(r['metrics'] for r in items))
        values={n:np.mean([r['metrics'][n] for r in items if r['metrics'].get(n) is not None and np.isfinite(r['metrics'][n])])
                if any(r['metrics'].get(n) is not None and np.isfinite(r['metrics'][n]) for r in items) else None for n in names}
        result[key]={'cluster_id':items[0]['cluster_id'],'metrics':values,'rows':items}
    return result


def ci(values,clusters):
    pairs=[(float(v),str(c)) for v,c in zip(values,clusters) if v is not None and np.isfinite(v)]
    if not pairs:return {'mean':None,'ci95':None,'samples':0,'groups':0}
    buckets=defaultdict(list)
    for v,c in pairs:buckets[c].append(v)
    keys=sorted(buckets)
    sums=np.array([sum(buckets[k]) for k in keys]);counts=np.array([len(buckets[k]) for k in keys])
    idx=np.random.default_rng(42).integers(len(keys),size=(10000,len(keys)))
    boots=sums[idx].sum(1)/counts[idx].sum(1)
    return {'mean':float(sum(sums)/sum(counts)),'ci95':np.quantile(boots,[.025,.975]).tolist(),
        'samples':len(pairs),'groups':len(keys)}


def paired(a,b,keys,names):
    result={};clusters=[a[k]['cluster_id'] for k in keys]
    for n in names:
        ds=[]
        for k in keys:
            x,y=a[k]['metrics'].get(n),b[k]['metrics'].get(n)
            ds.append(x-y if x is not None and y is not None else None)
        result[n]=ci(ds,clusters)
    return result


def collect(work,previous,stage,shards):
    data={v:[] for v in ['r05','r10']};changes=[];identities=[]
    for i in range(shards):
        folder=work/stage/f'shard{i}'
        if not (folder/'COMPLETED.json').exists():raise RuntimeError('Stage incomplete: '+str(folder))
        identities.append(json.loads((folder/'identity.json').read_text()))
        for v in data:data[v]+=json.loads((folder/v/'metrics.json').read_text())['per_generation']
        changes+=json.loads((folder/'output_changes.json').read_text())['per_generation']
    if len({x['role_state_sha256']['student'] for x in identities})!=1:raise RuntimeError('Shard student weights differ')
    if any(not x['raw_student_exactly_teacher'] for x in identities):raise RuntimeError('Reconstruction identity failed')
    for v in ['sft','raw']:
        data['teacher' if v=='sft' else 'step7500']=json.loads((previous/'results'/stage/v/'metrics.json').read_text())['per_generation']
    means={v:grouped(rows) for v,rows in data.items()};keys=sorted(means['r05'])
    if any(set(x)!=set(keys) for x in means.values()):raise RuntimeError('Comparison sample sets differ')
    for key in keys:
        for seed in [0,1]:
            for field in ['source_sha256','target_sha256','region_sha256','sample_uid','manifest_sha256']:
                values={next(r[field] for r in means[v][key]['rows'] if r['seed']==seed) for v in means}
                if len(values)!=1:raise RuntimeError('Paired protocol/input mismatch: '+field)
    original_protocol=json.loads((previous/'protocol.json').read_text())
    assert original_protocol['dimo']=={'steps':1,'cfg':1.,'r_init':.5,'embedding_noise_sigma':.3,'temperature':1.,'top_k':0,'top_p':0.}
    old_id=json.loads((previous/'results'/stage/'raw/identity.json').read_text())
    assert old_id['inference_contract_verified'] and old_id['base_identity_verified'] and old_id['teacher_artifacts_verified']
    clusters=[means['r05'][k]['cluster_id'] for k in keys]
    names=sorted(set().union(*(x['metrics'] for v in means.values() for x in v.values())))
    aggregate={v:{n:ci([values[k]['metrics'].get(n) for k in keys],clusters) for n in names} for v,values in means.items()}
    output={'stage':stage,'checkpoint_status':LABEL,'samples':len(keys),'source_groups':len(set(clusters)),
        'generation_seeds':[0,1],'per_variant_generations':len(keys)*2,'variants':aggregate,
        'paired_delta_r05_minus_teacher':paired(means['r05'],means['teacher'],keys,names),
        'paired_delta_r05_minus_step7500':paired(means['r05'],means['step7500'],keys,names),
        'paired_delta_r10_minus_r05':paired(means['r10'],means['r05'],keys,names),
        'historical_quality_pairing_verified':True,'historical_runtime_comparison_valid':False,
        'runtime_note':'Step0 caches deterministic source/text conditioning; historical evaluator included it in timing. No speedup conclusion.',
        'bootstrap':{'seed':42,'resamples':10000,'unit':'source_group_cluster','seed_reduction':'mean within each sample first'},
        'identity':identities[0],'improved_sample_proportion':{},'per_sample_rinit_deltas':[]}
    higher=['inside_ssim','outside_ssim','inside_psnr','outside_psnr','DINO-I-to-Target','CLIP-I-to-Target']
    for n in ['inside_masked_lpips','inside_ssim','outside_masked_lpips','outside_ssim','inside_psnr','outside_psnr']:
        ds=[]
        for k in keys:
            x,y=means['r10'][k]['metrics'].get(n),means['r05'][k]['metrics'].get(n)
            ds.append(x-y if x is not None and y is not None else None)
        signs=[float(d>0 if n in higher else d<0) if d is not None else None for d in ds]
        output['improved_sample_proportion'][n]=ci(signs,clusters)
    changed=grouped([r|{'metrics':{k:v for k,v in r.items() if k.startswith('output_change_')}} for r in changes])
    output['raw_output_changes']={n:ci([changed[k]['metrics'][n] for k in keys],clusters) for n in changed[keys[0]]['metrics']}
    for k in keys:
        output['per_sample_rinit_deltas'].append({'sample_key':k,'cluster_id':means['r05'][k]['cluster_id'],
            'delta_inside_lpips':means['r10'][k]['metrics']['inside_masked_lpips']-means['r05'][k]['metrics']['inside_masked_lpips'],
            'delta_inside_ssim':means['r10'][k]['metrics']['inside_ssim']-means['r05'][k]['metrics']['inside_ssim'],
            'delta_outside_lpips':means['r10'][k]['metrics']['outside_masked_lpips']-means['r05'][k]['metrics']['outside_masked_lpips'],
            'raw_output_changes':changed[k]['metrics']})
    return output,means


def montage(output,means,previous,stage):
    rows=[json.loads(l) for l in (previous/'data'/f'{stage}.jsonl').read_text().splitlines()][:12]
    for mode in ['teacher_step0','rinit_ablation']:
        for page in range(3):
            canvas=Image.new('RGB',(5*224,4*290),'white');draw=ImageDraw.Draw(canvas)
            for y,r in enumerate(rows[page*4:page*4+4]):
                key=r['sample_key']
                names=['Source','Mask','Target']+(['Teacher 12 / CFG5','Step0 r0.5'] if mode=='teacher_step0' else ['Step0 r0.5','Step0 r1.0'])
                variants=['teacher','r05'] if mode=='teacher_step0' else ['r05','r10']
                paths=[r['source'],r['mask'],r['target']]+[next(x['output'] for x in means[v][key]['rows'] if x['seed']==0) for v in variants]
                for x,(name,path) in enumerate(zip(names,paths)):
                    canvas.paste(Image.open(path).convert('RGB').resize((224,224)),(x*224,y*290+20))
                    draw.text((x*224+3,y*290+3),name,fill='black')
                draw.multiline_text((4,y*290+249),key+' seed0 | RECONSTRUCTED_STEP0\n'+'\n'.join(textwrap.wrap(r['instruction_en'],150)[:2]),fill='black')
            canvas.save(output/f'{stage}_{mode}_{page:02d}.jpg',quality=92)


def metric_plot(output,report):
    names=['inside_masked_lpips','inside_ssim','outside_masked_lpips','outside_ssim']
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    variants=['teacher','r05','r10','step7500']
    for ax,name in zip(axes,names):
        values=[report['variants'][v][name]['mean'] for v in variants]
        intervals=[report['variants'][v][name]['ci95'] for v in variants]
        errors=np.array([[v-c[0],c[1]-v] for v,c in zip(values,intervals)]).T
        ax.bar(range(4),values,yerr=errors,color=['#4177b4','#c55252','#46a582','#aaa'],capsize=3)
        ax.set_xticks(range(4),['Teacher','Step0 .5','Step0 1.0','Step7500'],rotation=25)
        ax.set_title(name);ax.grid(axis='y',alpha=.2)
    fig.suptitle(f"{report['stage']}: {report['samples']} samples, source-group 95% CI; RECONSTRUCTED_STEP0")
    fig.tight_layout();fig.savefig(output/f"{report['stage']}_inside_outside_metrics.png",dpi=160);plt.close(fig)


def numerical_plot(output,rows):
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for magnitude in [1,5,10,20]:
        selected=[r for r in rows if r['kind']=='synthetic' and r['objective']=='fp32_squared' and r['logit_magnitude']==magnitude]
        scales=sorted({r['g_magnitude'] for r in selected})
        for ax,key in zip(axes,['relative_l2_error','nonzero_reference_computed_zero_fraction']):
            ax.plot(scales,[np.mean([r[key] for r in selected if r['g_magnitude']==s]) for s in scales],marker='o',label=f'|z| scale {magnitude}')
    linear=[r for r in rows if r['kind']=='synthetic' and r['objective']=='fp32_linear']
    scales=sorted({r['g_magnitude'] for r in linear})
    for ax,key in zip(axes,['relative_l2_error','nonzero_reference_computed_zero_fraction']):
        ax.plot(scales,[np.mean([r[key] for r in linear if r['g_magnitude']==s]) for s in scales],
                linestyle='--',color='black',label='FP32 linear (all z scales)')
    real=[r for r in rows if r['kind']=='real_logits_controlled_nonzero_g' and r['objective']=='fp32_squared']
    if real:
        scales=sorted({r['g_magnitude'] for r in real})
        for ax,key in zip(axes,['relative_l2_error','nonzero_reference_computed_zero_fraction']):
            ax.plot(scales,[np.mean([r[key] for r in real if r['g_magnitude']==s]) for s in scales],
                    linestyle='-.',marker='x',color='#8743a6',label='Real Student z, controlled g')
    for ax in axes:ax.set_xscale('log');ax.set_xlabel('controlled g RMS scale');ax.grid(alpha=.25);ax.legend()
    axes[0].set_ylabel('relative L2 gradient error');axes[1].set_ylabel('nonzero reference -> zero fraction')
    axes[0].set_title('FP32 squared surrogate');axes[1].set_title('Cancellation before ROI normalization')
    fig.tight_layout();fig.savefig(output/'g_magnitude_vs_gradient_error.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--previous-eval',type=Path,required=True)
    p.add_argument('--work-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--full-shards',type=int,default=3);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    reports={}
    for stage,shards in [('pilot128',1),('full528',a.full_shards)]:
        report,means=collect(a.work_root,a.previous_eval,stage,shards)
        reports[stage]=report;montage(a.output,means,a.previous_eval,stage);metric_plot(a.output,report)
    (a.output/'step0_metrics.json').write_text(json.dumps({s:{k:v for k,v in r.items() if k not in ['per_sample_rinit_deltas','improved_sample_proportion','raw_output_changes']} for s,r in reports.items()},indent=2)+'\n')
    (a.output/'rinit_ablation_metrics.json').write_text(json.dumps(reports,indent=2)+'\n')
    (a.output/'checkpoint_identity.json').write_text(json.dumps(reports['full528']['identity'],indent=2)+'\n')
    synthetic=json.loads((a.output/'surrogate_synthetic.json').read_text())
    real=json.loads((a.output/'surrogate_real_logits.json').read_text())
    rows=synthetic['rows']+real['rows']
    combined={'synthetic':synthetic,'real_logits':real,'rows':rows}
    (a.output/'surrogate_gradient_audit.json').write_text(json.dumps(combined,indent=2)+'\n')
    write_csv(a.output/'surrogate_gradient_audit.csv',rows);numerical_plot(a.output,rows)
    (a.output/'SUMMARY_READY.json').write_text(json.dumps({'samples':reports['full528']['samples'],'groups':reports['full528']['source_groups'],'gradient_rows':len(rows)},indent=2)+'\n')
    print('SUMMARY_COMPLETE',flush=True)


if __name__=='__main__':main()
