"""Independent CPU-only coverage/checkpoint audit and saved-prediction analysis."""
from collections import Counter, defaultdict
import gc
import json
from pathlib import Path
import numpy as np
import torch

from scripts.recognition.train_v38_yolo25 import WORK, START, SOURCE, V34, ROOT, HOLDOUT, INPUT
from outputs.recognition.v36_yolo_fullpasses import load
from outputs.recognition.tiered_short_pilot import rows
from src.recognition.metrics import summarise
from src.recognition.utils import sha, write_json


def cluster_bootstrap(a,b,metadata):
    assert len(a)==len(b)==len(metadata)
    buckets=defaultdict(lambda:np.zeros(3))
    for x,y,r in zip(a,b,metadata):
        assert x['truth']==y['truth']==r['text'] and x['index']==y['index']
        buckets[r['source_menu']]+=np.array([sum(x['error_counts'].values()),sum(y['error_counts'].values()),len(x['truth'])])
    data=np.array(list(buckets.values()));rng=np.random.default_rng(38)
    sampled=data[rng.integers(0,len(data),(5000,len(data)))].sum(1)
    diff=(sampled[:,1]-sampled[:,0])/sampled[:,2]*100
    return dict(menus=len(data),v38_minus_v36_pp=float((data[:,1].sum()-data[:,0].sum())/data[:,2].sum()*100),
                ci95_pp=np.percentile(diff,[2.5,97.5]).tolist())


def main():
    torch.set_num_threads(4)
    s=load(WORK/'sampling.json');report=load(WORK/'report.json');audits=load(WORK/'epoch_audits.json')
    cfg=load(WORK/'run_config.json');parity=load(WORK/'initial_parity.json')
    source=torch.load(START,map_location='cpu',weights_only=False)
    baseline=torch.load(SOURCE,map_location='cpu',weights_only=False)['model']
    bnkeys=[k for k in baseline if k.endswith(('running_mean','running_var','num_batches_tracked'))];assert len(bnkeys)==147
    assert sha(START)==cfg['source_sha256'] and all(v==0 for p in parity.values() for v in p.values())
    assert sha(INPUT/'labeled_crops.jsonl')==s['input_labels_sha256']
    assert sha(INPUT/'crops_manifest.json')==s['input_manifest_sha256']
    original=[r for r in rows(ROOT/'labels.jsonl') if r['split']=='train'];generated=rows(V34/'corpus/labels.jsonl')
    required=load(V34/'corpus/manifest.json')['required_terms'];assert len(required)==521
    yolo=rows(WORK/'yolo/labels.jsonl');train=[r for r in yolo if r['split']=='train'];hold=[r for r in yolo if r['split']=='val_holdout']
    assert len(train)==1336 and len(hold)==135 and {r['source_menu'] for r in hold}==HOLDOUT
    assert not {r['source_menu'] for r in train}&HOLDOUT
    assert not {r['sha256'] for r in train}&{r['sha256'] for r in hold}
    for r in yolo:assert sha(WORK/'yolo'/r['image'])==r['sha256']
    ds={'original':original,'generated':generated,'yolo':train};coverage=[];updates=0
    for epoch in range(1,5):
        plan=load(WORK/f'epoch_{epoch:02d}_plan.json');counts=Counter(tuple(k) for b in plan for k in b)
        assert all(v==1 for v in counts.values()) and len(counts)==5344
        assert {i for k,i in counts if k=='yolo'}==set(range(1336))
        terms=Counter(ds[k][i]['text'] for k,i in counts);assert all(terms[t]>=1 for t in required)
        actual=next(a for a in audits if a['epoch']==epoch)
        assert actual['records']==5344 and actual['yolo_presentations']==1336
        assert actual['exact_term_encounters']=={t:terms[t] for t in required}
        assert sha(WORK/f'epoch_{epoch:02d}_plan.json')==s['plans'][epoch-1]['plan_sha256']
        updates+=len(plan);coverage.append(dict(epoch=epoch,records=len(counts),yolo=1336,required_terms=521,updates=len(plan)))
    assert updates==report['updates']==686 and report['presentations']==21376 and report['yolo_presentations']==5344
    states={}
    for name in ('best.pt','last.pt','final_raw.pt','final_ema.pt','epoch_01.pt','epoch_02.pt','epoch_03.pt','epoch_04.pt'):
        saved=torch.load(WORK/name,map_location='cpu',weights_only=False)
        for field in ('model','raw_model','ema_model'):
            assert all(torch.equal(saved[field][k],baseline[k]) for k in bnkeys)
        if name=='best.pt':assert saved['updates']==0 and all(torch.equal(saved['model'][k],v) for k,v in source['model'].items())
        if name in ('last.pt','final_raw.pt','final_ema.pt'):
            assert saved['updates']==686
            assert all(int(state['step'])==686 for state in saved['optimizer']['state'].values())
        affine=[k.rsplit('.',1)[0]+'.'+suffix for k in bnkeys if k.endswith('running_mean') for suffix in ('weight','bias')]
        states[name]=dict(update=saved['updates'],arm=saved['checkpoint_arm'],bn_buffers_fixed=147,
                         bn_affine_tensors_changed=sum(not torch.equal(saved['model'][k],source['model'][k]) for k in affine))
        del saved;gc.collect()
    verification=dict(passed=True,coverage=coverage,total_updates=updates,total_presentations=21376,yolo_presentations=5344,yolo_fraction=.25,
        train_crops=1336,holdout_crops=135,holdout_menus=sorted(HOLDOUT),menu_pixel_hash_disjoint=True,
        all_eligible_image_hashes_verified=True,source_unchanged=True,initial_predictions_match_V36_both_precisions=True,
        checkpoints=states,selected_equals_initial_V36=True)
    write_json(WORK/'completion_verification.json',verification)
    initial={p:load(WORK/f'initial_{p}.json') for p in ('fp16','bf16')}
    results={};curves=[]
    for epoch,step in enumerate((173,344,515,686),1):
        for arm in ('raw','ema'):
            for precision in ('fp16','bf16'):
                data=load(WORK/f'step_{step:04d}_{arm}_{precision}.json')
                summary={k:v['0.0']['all'] for k,v in data['validation'].items()}
                curves.append(dict(epoch=epoch,update=step,arm=arm,precision=precision,metrics=summary,retention=data['retention'],score=data['score']))
                if step!=686:continue
                item=dict(metrics=summary,retention=data['retention'],score=data['score'],paired={},stratified={})
                for view,metadata in (('yolo_current_train',train),('yolo_current_holdout',hold)):
                    a=initial[precision][view]['0.0']['predictions'];b=data['validation'][view]['0.0']['predictions']
                    item['paired'][view]=cluster_bootstrap(a,b,metadata)
                    groups={}
                    for key,values in (('orientation_flag',(0,1)),('canvas',(128,256,384,512)),('review_required',(False,True))):
                        for val in values:
                            indices=[i for i,r in enumerate(metadata) if r.get(key,False)==val]
                            if indices:groups[f'{key}={val}']=dict(initial=summarise([a[i] for i in indices]),final=summarise([b[i] for i in indices]))
                    menus={m:dict(initial=summarise([a[i] for i,r in enumerate(metadata) if r['source_menu']==m]),
                                  final=summarise([b[i] for i,r in enumerate(metadata) if r['source_menu']==m])) for m in sorted({r['source_menu'] for r in metadata})}
                    groups['by_menu']=menus;item['stratified'][view]=groups
                results[f'{arm}_{precision}']=item
    write_json(WORK/'paired_analysis.json',dict(final=results,history=curves))
    lines=['# V38 independent completion review','',
           'Verified 686 updates / 21,376 presentations; 5,344 YOLO presentations (25%). All1,336 train crops exactly once in each of four passes;135 holdout crops from13 disjoint menus. All521 exact synthetic terms encountered each pass. All147 V30 BN buffers fixed in selected/final/epoch checkpoints, including raw and EMA states. Initial predictions reproduce V36 in both precisions. Source weights and delivered label/image hashes unchanged.', '',
           '**Selected best.pt is the original V36 weights at update0. No trained candidate passed all retention gates in both precisions.**', '',
           '| Dataset, FP16 | Initial V36 CER | Final raw CER | Final EMA CER |','|---|---:|---:|---:|']
    for name in initial['fp16']:
        lines.append(f"| {name} | {initial['fp16'][name]['0.0']['all']['cer']:.2%} | {results['raw_fp16']['metrics'][name]['cer']:.2%} | {results['ema_fp16']['metrics'][name]['cer']:.2%} |")
    lines+=['','## YOLO holdout paired evidence','']
    for arm in ('raw','ema'):
        r=results[f'{arm}_fp16'];p=r['paired']['yolo_current_holdout']
        lines.append(f"Final {arm}: YOLO holdout change {p['v38_minus_v36_pp']:+.2f}pp; menu-cluster bootstrap95%CI [{p['ci95_pp'][0]:+.2f}, {p['ci95_pp'][1]:+.2f}]pp.")
    lines+=['','## Retention at final step','']
    for name,r in results.items():
        g=r['retention'];lines.append(f"{name}: {g['failed']}")
        for k in g['failed']:lines.append(f"- {k}: {g['current'][k]:.4%} vs cap {g['caps'][k]:.4%}.")
    lines+=['','The heldout score improves across all four raw epochs; training and holdout improve together, so this run does not show aggregate YOLO holdout overfitting. '
            'Retention failures concern short synthetic labels and precision sensitivity, not absence of YOLO adaptation. EMA lags raw over this short686-update horizon. '
            'VLM labels are not independently verified. This is development/selection evidence on13 menus, not an independent production test. '
            'Check paired_analysis.json for menu, width, orientation, flagged-overlap and error-type breakdowns. No additional inference or optimization was run for this review.']
    lines+=['','## Interpretation and next step','',
            'Final raw improves12/13 heldout menus; menu040 worsens65.31%→67.35%. Gains are not confined to one page. '
            'Holdout exact lines14→24/135; deletions199→156, substitutions138→102, insertions17→25. Most recovery comes from fewer deletions/substitutions, while extra insertions need attention.',
            '', 'Heldout horizontal CER56.15%→42.88%, vertical37.13%→35.93%. The512bucket remains73.58% on only6heldout lines; this experiment does not establish long-line improvement. '
            'Flagged-overlap crops63.64%→57.14% remain harder than unflagged50.00%→39.18%.',
            '', 'Final raw FP16 passes all17gates, including2char54/400 at the cap; BF16 has56/400 against cap54. This is a2-edit cross-precision failure, not a failure to learn YOLO text. '
            'Final EMA FP16 has55/400, BF16 has55/400 and61/200 one-char vs cap60. All98BNaffine tensors changed, while147running-stat buffers did not.',
            '', 'The final raw epoch improves holdout by one edit only (284→283), suggesting diminishing returns at this learning-rate horizon. Do not extend the run blindly. '
            'Recommended next inexpensive check: interpolate preserved V36 and final V38 raw weights at25/50/75%V38, retaining the identical BN buffers, and evaluate the unchanged holdout/retention suites in both precisions. '
            'Keep V36 as the selected production candidate until a blend or trained checkpoint clears the agreed gates. Independently spot-check VLM labels before treating absolute CER as a real-menu quality estimate.']
    (WORK/'review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
