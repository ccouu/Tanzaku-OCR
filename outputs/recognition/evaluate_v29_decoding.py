"""Frozen V29 inference: additive blank penalty sweep versus P(blank)>0.5 gate."""
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from outputs.recognition.application_mixture_pilot import validations
from outputs.recognition.application_pilot import state_hash
from src.recognition.data import collate, decode
from src.recognition.metrics import align_errors, summarise
from src.recognition.model import MenuRecognizer
from src.recognition.utils import sha, write_json

OUT=Path('outputs/recognition/evaluation/v29_decoding_arms')
RUN=Path('outputs/recognition/pilots/v29_application_mixture_2000')
CHECKPOINT=RUN/'best.pt'
PENALTIES=[round(i/10,1) for i in range(21)]


def collapse(path,reverse):
    result=[];previous=None
    for token in path:
        token=int(token)
        if token!=previous and token!=0:result.append(reverse[token])
        previous=token
    return ''.join(result)


def path_for(nonblank,gap,probability,penalty=None):
    blank=(probability>.5) if penalty is None else (gap>=penalty)
    return np.where(blank,0,nonblank)


def unit_checks():
    # A non-majority blank can still be argmax; the threshold arm must reject it.
    p=torch.tensor([.4,.35,.25]); logits=p.log()
    top=int(logits[1:].argmax())+1;gap=float(logits[0]-logits[top])
    assert path_for(np.array([top]),np.array([gap]),np.array([.4]),0).tolist()==[0]
    assert path_for(np.array([top]),np.array([gap]),np.array([.4]),None).tolist()==[1]
    assert path_for(np.array([1]),np.array([1.]),np.array([.5]),None).tolist()==[1]
    assert path_for(np.array([1]),np.array([1.]),np.array([.6]),None).tolist()==[0]
    assert path_for(np.array([1]),np.array([1.]),np.array([.8]),1.).tolist()==[0]
    assert path_for(np.array([1]),np.array([1.]),np.array([.8]),1.1).tolist()==[1]
    assert collapse([1,1,0,1],{1:'日'})=='日日'
    assert collapse([1,1],{1:'日'})=='日'


def summarise_set(rows):
    result=dict(all=summarise(rows),by_length={str(n):summarise([r for r in rows if len(r['truth'])==n]) for n in sorted({len(r['truth']) for r in rows})})
    fonts=sorted({r['font'] for r in rows if r['font']})
    result['by_font']={f:summarise([r for r in rows if r['font']==f]) for f in fonts}
    for key in ('contrast','texture'):
        values=sorted({r[key] for r in rows if r.get(key)})
        if values:result['by_'+key]={v:summarise([r for r in rows if r.get(key)==v]) for v in values}
    return result


def main():
    unit_checks();OUT.mkdir(parents=True,exist_ok=False);started=time.perf_counter();torch.set_num_threads(4)
    before=sha(CHECKPOINT);saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    assert saved['updates']==2000
    mapping=saved['vocab']['char_to_id'];reverse={v:k for k,v in mapping.items()}
    model=MenuRecognizer(len(mapping)+1,use_bilstm=True).cuda();model.load_state_dict(saved['model']);model.eval()
    before_state=state_hash(model);views=validations(saved['config'],mapping,False)
    reference=json.loads((RUN/'step_2000.json').read_text(encoding='utf-8'))['validation']
    modes=[f'penalty_{b:.1f}' for b in PENALTIES]+['threshold_0.5']
    report={mode:{} for mode in modes};cache=[]
    with (OUT/'predictions.jsonl').open('w',encoding='utf-8') as predictions:
        with torch.inference_mode():
            for name,ds in views.items():
                rows_by_mode={mode:[] for mode in modes}
                for canvas in (128,256,384):
                    ids=[i for i in range(len(ds)) if ds.canvas_at(i)==canvas]
                    for start in range(0,len(ids),32):
                        selected=ids[start:start+32];batch=collate([ds[i] for i in selected],padded_width=canvas)
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            logits=model(batch['images'].cuda(),batch['orientation'].cuda(),batch['input_lengths'])
                        f=logits.float();top,indices=f[...,1:].max(-1);indices+=1
                        gaps=(f[...,0]-top).cpu().numpy();pblank=(f[...,0]-torch.logsumexp(f,-1)).exp().cpu().numpy();indices=indices.cpu().numpy()
                        standard=decode(logits,mapping,batch['input_lengths'])
                        for col,(i,truth) in enumerate(zip(selected,batch['texts'])):
                            n=int(batch['input_lengths'][col]);gap=gaps[:n,col];prob=pblank[:n,col];tokens=indices[:n,col]
                            row=ds.rows[i];base=dict(dataset=name,index=i,truth=truth,font=row.get('preferred_font'),contrast=row.get('contrast'),texture=row.get('texture'))
                            cache.append(dict(base,nonblank=tokens.tolist(),blank_gap=gap.tolist(),p_blank=prob.tolist(),valid_t=n))
                            for mode in modes:
                                beta=None if mode=='threshold_0.5' else float(mode.split('_')[1])
                                prediction=collapse(path_for(tokens,gap,prob,beta),reverse)
                                if beta==0:assert prediction==standard[col]
                                c=Counter(op['type'] for op in align_errors(truth,prediction) if op['type']!='match')
                                value=dict(base,mode=mode,prediction=prediction,error_counts={k:c[k] for k in ('deletion','insertion','substitution')})
                                rows_by_mode[mode].append(value);predictions.write(json.dumps(value,ensure_ascii=False)+'\n')
                for mode,rows in rows_by_mode.items():report[mode][name]=summarise_set(rows)
                assert report['penalty_0.0'][name]['all']==reference[name]['0.0']['all'],name
                # Verify each baseline transcription too, rather than only its aggregate CER.
                expected={r['index']:r['prediction'] for r in reference[name]['0.0']['predictions']}
                assert all(r['prediction']==expected[r['index']] for r in rows_by_mode['penalty_0.0'])
                print(name,'baseline',report['penalty_0.0'][name]['all']['cer'],'threshold',report['threshold_0.5'][name]['all']['cer'],flush=True)
    assert sha(CHECKPOINT)==before and state_hash(model)==before_state
    with (OUT/'frame_scores.jsonl').open('w',encoding='utf-8') as f:
        for row in cache:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    write_json(OUT/'report.json',dict(results=report,checkpoint=str(CHECKPOINT),checkpoint_sha256=before,
        checkpoint_update=2000,penalties=PENALTIES,threshold=.5,training_updates=0,model_state_unchanged=True,
        original_predictions_identical=True,script_sha256=sha(Path(__file__)),elapsed_seconds=time.perf_counter()-started,
        arm1='Subtract beta from blank logit before greedy argmax; no multiplication',
        arm2='P(blank) from unmodified full-class softmax; if >0.5 choose blank, otherwise best nonblank; standard CTC collapse afterward',
        lengths='Same valid_t as V29, excludes artificial right padding'))
    keys=('application_clean_control','application_texture_color','application_texture_color_noise','seen','heldout','original_bucket','historical_bucket','yolo_provisional','strong_appearance')
    lines=['# V29 frozen decoding comparison','',
        'Same nine frozen evaluation sets and V29 update2000. Identical model forward passes for all22 modes. No training, lexicon or beam search. Baseline predictions exactly reproduced for every line.',
        '', '| Decoder | App clean | App texture | App noisy | Seen | Held-out | Original969 | Historical | YOLO | Strong |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode in modes:lines.append('| '+mode+' | '+' | '.join(f"{report[mode][key]['all']['cer']:.2%}" for key in keys)+' |')
    lines.extend(['','## Best penalty on each development set','', '| Dataset | Lowest CER | Penalties attaining minimum |','|---|---:|---|'])
    for key in keys:
        lowest=min(report[mode][key]['all']['edits'] for mode in modes[:-1])
        best=[mode for mode in modes[:-1] if report[mode][key]['all']['edits']==lowest]
        lines.append(f"| {key} | {report[best[0]][key]['all']['cer']:.2%} | {', '.join(best)} |")
    lines.extend(['','Arm1 subtracts beta from the blank logit. Arm2 uses unmodified softmax over all1723 classes: choose blank iff P(blank)>0.5, otherwise choose the highest-probability nonblank token. Both apply ordinary CTC repeat-collapse and blank removal, using unchanged valid lengths.',
        '', 'Per-dataset best penalties are descriptive development-set optima, not independently validated recommendations. The same application-generator family was used in training; YOLO labels are provisional. Frame scores, per-line predictions and length/font/contrast/texture metrics are saved for inspection.'])
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('COMPLETE',time.perf_counter()-started,flush=True)


if __name__=='__main__':main()
