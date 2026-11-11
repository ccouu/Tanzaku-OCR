"""Frozen V31 one-character decoding:blank penalty,threshold,and empty-only fallback."""
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from outputs.recognition.application_mixture_pilot import validations,RenderedLines
from outputs.recognition.application_pilot import state_hash
from outputs.recognition.evaluate_v29_decoding import collapse,unit_checks
from src.recognition.data import collate,decode
from src.recognition.metrics import align_errors,summarise
from src.recognition.model import MenuRecognizer
from src.recognition.utils import sha,write_json

OUT=Path('outputs/recognition/evaluation/v31_onechar_blank_decoders')
V30=Path('outputs/recognition/pilots/v30_short_retention_2000')
V31=Path('outputs/recognition/pilots/v31_curated_short_2000')
CHECKPOINT=V31/'curated_70/step_2000.pt'
PENALTIES=[round(n/10,1) for n in range(21)]
THRESHOLDS=[round(n/10,1) for n in range(11)]+[.95,.99]


def read(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def make_views(cfg,mapping):
    views=validations(cfg,mapping,False)
    for view in ('clean','noisy'):
        old=[r for r in read(V30/'short_probe_labels.jsonl') if r['view']==view]
        views['fresh_short_'+view]=RenderedLines(old,V30,cfg,mapping)
        kind='clean_control' if view=='clean' else 'texture_color_noise'
        new=[r for r in read(V31/'menu_probe_labels.jsonl') if r['view']==kind]
        views['menu_'+view]=RenderedLines(new,V31,cfg,mapping)
    for folder,prefix in [('v30_onechar_fonts_1000','font_audit'),('v30_meaningful_onechar_terms','meaningful')]:
        root=Path('outputs/recognition/evaluation')/folder
        source=[dict(r,text=r['truth'],preferred_font=r.get('setting',r.get('font'))) for r in read(root/'labels.jsonl')]
        for view in ('clean','noisy'):
            selected=[r for r in source if r['view']==view and (prefix=='font_audit' or r['group']=='term')]
            views[prefix+'_'+view]=RenderedLines(selected,root,cfg,mapping)
    return {k:ds for k,ds in views.items() if any(len(r['text'])==1 for r in ds.rows)}


def collect(model,views,mapping):
    cache=[];reference=json.loads((V31/'curated_70/step_2000.json').read_text(encoding='utf-8'))['validation']
    with torch.inference_mode():
        for name,ds in views.items():
            expected={r['index']:r['prediction'] for r in reference[name]['0.0']['predictions']} if name in reference else None
            before=len(cache)
            # Keep original evaluation batch composition for exact baseline reproducibility.
            for canvas in (128,256,384):
                ids=[i for i in range(len(ds)) if ds.canvas_at(i)==canvas]
                for start in range(0,len(ids),32):
                    selected=ids[start:start+32]
                    if not any(len(ds.rows[i]['text'])==1 for i in selected):continue
                    b=collate([ds[i] for i in selected],padded_width=canvas)
                    with torch.autocast('cuda',dtype=torch.bfloat16):f=model(b['images'].cuda(),b['orientation'].cuda(),b['input_lengths']).float()
                    top,tokens=f[...,1:].max(-1);tokens+=1;lse=torch.logsumexp(f,-1)
                    gap=(f[...,0]-top).cpu().numpy();prob=(f[...,0]-lse).exp().cpu().numpy()
                    confidence=(top-lse).cpu().numpy();tokens=tokens.cpu().numpy();standard=decode(f,mapping,b['input_lengths'])
                    for col,i in enumerate(selected):
                        row=ds.rows[i];truth=row['text']
                        if len(truth)!=1:continue
                        n=int(b['input_lengths'][col]);t=tokens[:n,col];g=gap[:n,col];p=prob[:n,col]
                        if expected is not None:assert standard[col]==expected[i],(name,i,truth,standard[col],expected[i])
                        assert collapse(np.where(g>=0,0,t),{v:k for k,v in mapping.items()})==standard[col]
                        cache.append(dict(dataset=name,index=i,truth=truth,font=row.get('preferred_font'),
                            baseline=standard[col],nonblank=t.tolist(),blank_gap=g.tolist(),p_blank=p.tolist(),
                            fallback_token=int(t[int(confidence[:n,col].argmax())]),valid_t=n))
            print('CACHED',name,len(cache)-before,flush=True)
    return cache


def metrics(rows):
    value=summarise(rows)
    value['empty_predictions']=sum(not r['prediction'] for r in rows)
    value['multiple_character_predictions']=sum(len(r['prediction'])>1 for r in rows)
    value['baseline_empty']=sum(not r['baseline'] for r in rows)
    value['baseline_empty_recovered_exact']=sum(not r['baseline'] and r['prediction']==r['truth'] for r in rows)
    return value


def main():
    unit_checks();OUT.mkdir(parents=True,exist_ok=False);started=time.perf_counter();torch.set_num_threads(4)
    before=sha(CHECKPOINT);saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    assert saved['updates']==2000 and saved['arm']=='curated_70'
    mapping=saved['vocab']['char_to_id'];reverse={v:k for k,v in mapping.items()}
    model=MenuRecognizer(len(mapping)+1,use_bilstm=True).cuda();model.load_state_dict(saved['model']);model.eval();before_state=state_hash(model)
    cache=collect(model,make_views(saved['config'],mapping),mapping)
    modes={f'penalty_{b:.1f}':('penalty',b) for b in PENALTIES}
    modes.update({f'threshold_{t:g}':('threshold',t) for t in sorted(THRESHOLDS)})
    modes['nonempty_fallback']=('fallback',None)
    result={};lookup={}
    with (OUT/'predictions.jsonl').open('w',encoding='utf-8') as output:
        for mode,(kind,parameter) in modes.items():
            rows=[]
            for r in cache:
                if kind=='fallback':prediction=r['baseline'] or reverse[r['fallback_token']]
                else:
                    blank=np.asarray(r['blank_gap'])>=parameter if kind=='penalty' else np.asarray(r['p_blank'])>parameter
                    prediction=collapse(np.where(blank,0,np.asarray(r['nonblank'])),reverse)
                c=Counter(op['type'] for op in align_errors(r['truth'],prediction) if op['type']!='match')
                row={k:r[k] for k in ('dataset','index','truth','font','baseline')}
                row.update(mode=mode,prediction=prediction,error_counts={k:c[k] for k in ('deletion','insertion','substitution')})
                rows.append(row);output.write(json.dumps(row,ensure_ascii=False)+'\n')
                if mode=='penalty_0.0':assert prediction==r['baseline']
                if kind=='fallback':
                    assert prediction
                    assert r['baseline']==prediction or not r['baseline']
                    if not r['baseline']:assert sum(c.values())<=1
            groups={name:metrics([r for r in rows if r['dataset']==name]) for name in sorted({r['dataset'] for r in rows})}
            pooled=[r for r in rows if r['dataset'] in ('seen','heldout','original_bucket')]
            assert len(pooled)==200
            groups['pooled_original_1char']=metrics(pooled)
            result[mode]=groups;lookup[mode]=rows
    assert sha(CHECKPOINT)==before and state_hash(model)==before_state
    with (OUT/'frame_scores.jsonl').open('w',encoding='utf-8') as f:
        for r in cache:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    report=dict(results=result,checkpoint=str(CHECKPOINT),checkpoint_sha256=before,model_state_unchanged=True,
        training_updates=0,images=len(cache),penalty_range=PENALTIES,thresholds=sorted(THRESHOLDS),
        baseline_predictions_reproduced=True,script_sha256=sha(Path(__file__)),elapsed_seconds=time.perf_counter()-started,
        lengths='Unchanged valid_t,exclude artificial right padding',
        threshold_rule='Blank iff unmodified full-class softmax P(blank)>threshold;otherwise best nonblank;standard CTC collapse',
        fallback_rule='Only if standard greedy output is empty,return one nonblank token from the valid frame with highest nonblank softmax probability. No known-length or dictionary constraint.',
        limits='All evaluated crops have one true character. No detector false-positive or blank-negative evaluation;fallback assumes a text-containing crop.')
    write_json(OUT/'report.json',report)
    keys=('menu_clean','menu_noisy','meaningful_clean','meaningful_noisy','pooled_original_1char','fresh_short_clean','fresh_short_noisy','font_audit_clean','font_audit_noisy')
    lines=['# V31 single-character blank decoding','',
        f'Frozen curated step2000 checkpoint;{len(cache)} one-character images. No training or lexicon. Original full-batch baseline predictions reproduced on existing V31 sets.',
        '', '| Decoder | Menu clean | Menu noisy |16terms clean |16terms noisy |Original200 |Broad clean |Broad noisy |Font clean |Font noisy |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode in modes:lines.append('| '+mode+' | '+' | '.join(f"{result[mode][k]['cer']:.2%}" for k in keys)+' |')
    lines.extend(['','## Best observed settings per set','', '| Set | Best penalty | CER | Best threshold | CER | Empty-only fallback CER |','|---|---|---:|---|---:|---:|'])
    for key in keys:
        pen=min((m for m in modes if m.startswith('penalty_')),key=lambda m:(result[m][key]['edits'],m))
        threshold=min((m for m in modes if m.startswith('threshold_')),key=lambda m:(result[m][key]['edits'],m))
        lines.append(f"| {key} | {pen} | {result[pen][key]['cer']:.2%} | {threshold} | {result[threshold][key]['cer']:.2%} | {result['nonempty_fallback'][key]['cer']:.2%} |")
    lines.extend(['','Nonempty text detection and CTC temporal blank are different. Thresholds apply per frame,not per crop. Threshold1 removes every blank frame and can create many insertions. Empty-only fallback preserves all existing nonempty predictions and all temporal decoding;it changes only an entirely empty result.',
        '', 'For a true one-character crop,forcing one character only when the output was empty cannot worsen edit distance:correct fallback saves an error,incorrect fallback trades one deletion for one substitution. This guarantee does not extend to blank detector false positives. Per-dataset optimal settings are development diagnostics,not independently validated universal decoder choices.'])
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Yen-specific results across the old16-term audit and new menu probe.
    yen={mode:{name:metrics([r for r in rows if r['dataset']==name and r['truth']=='円']) for name in ('meaningful_clean','meaningful_noisy','menu_clean','menu_noisy')} for mode,rows in lookup.items()}
    write_json(OUT/'yen_results.json',yen)
    print('COMPLETE',len(cache),'images,',len(modes),'decoders,',time.perf_counter()-started,flush=True)


if __name__=='__main__':main()
