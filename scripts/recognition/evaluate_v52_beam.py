"""Frozen V43/V48 prefix-beam and optional menu-lexicon reranking evaluation."""
from collections import defaultdict,Counter
from pathlib import Path
import json
import re
import sqlite3
import time
import numpy as np
import torch
from scripts.recognition.evaluate_v51_resolution import build_views,rows
from scripts.recognition.train_v38_yolo25 import BucketCrops
from scripts.recognition.review_v50 import paired
from src.recognition.model import MenuRecognizer
from src.recognition.data import collate,decode
from src.recognition.metrics import summarise,align_errors
from src.recognition.canonical_ctc import fold_parenthesis_logits
from src.recognition.label_normalization import normalize_parentheses,normalize_digits
from src.recognition.prefix_beam import decode_beams,MenuLexiconScore
from src.recognition.utils import write_json,sha

WORK=Path('outputs/recognition/evaluation/v52_beam_lexicon')
DB=Path('data/processed/lexicon/culinary_menu.db')
ARMS=('greedy','beam5','beam10','beam25','beam25_lex0.1','beam25_lex0.3')


def assess(records):
    out=summarise(records)
    out['by_length']={str(n):summarise([r for r in records if len(r['truth'])==n]) for n in sorted({len(r['truth']) for r in records})}
    symbols={c:[0,0] for c in '()0123456789'}
    for r in records:
        for op in align_errors(r['truth'],r['prediction']):
            if op['truth'] in symbols:
                symbols[op['truth']][0]+=1;symbols[op['truth']][1]+=op['type']=='match'
    out['symbol_recall']={c:dict(reference=v[0],correct=v[1],recall=v[1]/v[0] if v[0] else None) for c,v in symbols.items()}
    digits=[v['recall'] for c,v in out['symbol_recall'].items() if c.isdigit() and v['recall'] is not None]
    out['digit_macro_recall']=sum(digits)/len(digits) if digits else None
    digit_lines=[r for r in records if any(c.isdigit() for c in r['truth'])]
    out['digit_sequence_exact']=dict(lines=len(digit_lines),correct=sum(''.join(c for c in r['truth'] if c.isdigit())==''.join(c for c in r['prediction'] if c.isdigit()) for r in digit_lines))
    price_lines=[r for r in records if 'price' in r.get('kind','') or
        (('円' in r['truth'] or '¥' in r['truth']) and re.fullmatch(r'[¥￥]?[0-9][0-9,]*(?:円)?',r['truth'].strip()))]
    out['price_line_exact']=dict(lines=len(price_lines),correct=sum(r['truth']==r['prediction'] for r in price_lines))
    return out


def collect(model,ds,mapping,cache):
    meta_file=cache.with_suffix('.json');npz_file=cache.with_suffix('.npz')
    if meta_file.exists() and npz_file.exists():return json.loads(meta_file.read_text(encoding='utf-8')),dict(np.load(npz_file))
    groups=defaultdict(list);samples=[ds[i] for i in range(len(ds))]
    for i,s in enumerate(samples):groups[s[0].shape[-1]].append(i)
    output={};packed={};missing_mass=[]
    with torch.inference_mode():
        for canvas,ids in sorted(groups.items()):
            for start in range(0,len(ids),32):
                indices=ids[start:start+32];b=collate([samples[i] for i in indices],padded_width=canvas)
                with torch.autocast('cuda',dtype=torch.float16):z=model(b['images'].cuda(),b['orientation'].cuda(),b['input_lengths'])
                assert torch.isfinite(z).all()
                pooled=fold_parenthesis_logits(z,mapping);pred=decode(pooled,mapping,b['input_lengths']);raw=decode(z,mapping,b['input_lengths'])
                lp=pooled.float().log_softmax(-1);blank=lp[:,:,0].clone();lp[:,:,0]=float('-inf')
                values,tokens=lp.topk(40,-1)
                for j,i in enumerate(indices):
                    n=int(b['input_lengths'][j]);r=ds.rows[i]
                    output[i]=dict(index=i,truth=normalize_parentheses(b['texts'][j]),greedy=pred[j],literal_prediction=raw[j],
                        font=r.get('preferred_font'),source_menu=r.get('source_menu'),crop_id=r.get('crop_id'),kind=r.get('kind',''))
                    packed[f'b{i}']=blank[:n,j].cpu().numpy();packed[f'p{i}']=values[:n,j].cpu().numpy();packed[f'i{i}']=tokens[:n,j].cpu().numpy().astype(np.int16)
                    missing_mass.extend((1-blank[:n,j].exp()-values[:n,j].exp().sum(-1)).clamp_min(0).cpu().tolist())
    metadata=dict(records=[output[i] for i in range(len(ds))],candidate_top_k=40,relative_logit_cutoff=12,
        discarded_topk_mass_mean=float(np.mean(missing_mass)),discarded_topk_mass_max=float(max(missing_mass)))
    np.savez_compressed(npz_file,**packed);write_json(meta_file,metadata)
    return metadata,packed


def main():
    WORK.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);assert torch.cuda.is_available()
    paths={'V43':Path('checkpoints/menu_recognizer_v43.pt'),
        'V48_final_raw':Path('outputs/recognition/pilots/v48_parentheses_2000/final_raw.pt')}
    with sqlite3.connect('file:'+str(DB)+'?mode=ro',uri=True) as db:
        terms=[normalize_parentheses(normalize_digits(r[0])) for r in db.execute('select canonical_jp from menu_dictionary')]
    lex=MenuLexiconScore(terms);started=time.time()
    report=dict(completed=False,training=False,production_unchanged=True,lexicon_sha256=sha(DB),lexicon_terms=len(lex.terms),
        lexicon_policy='Nbest25rerank;bonus=min(covered_chars,10)*weight;unknownallowed;digitslockedtoacousticbeam25',
        canonical_parentheses=True,precision='fp16',candidate_top_k=40,relative_cutoff=12,
        selection_note='Existing156YOLOholdoutisdevelopmentdata;no untouchedhumanverifiedconfirmationsetavailable.',models={})
    # Compile/JIT before timing decoder calls.
    decode_beams(np.array([-.1]),np.array([[1]]),np.array([[-3.]]),5,{'a':1})
    for tag,path in paths.items():
        saved=torch.load(path,map_location='cpu',weights_only=False);mapping=saved['vocab']['char_to_id'];cfg=saved['config']
        model=MenuRecognizer(len(mapping)+1,use_bilstm=True).cuda().eval();model.load_state_dict(saved['model']);del saved
        frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        wanted=('yolo_current_holdout','brackets_known','brackets_heldout','digits_known','digits_heldout',
            'original_bucket','historical_bucket','fresh_short_clean','fresh_short_noisy','application_texture_color_noise')
        all_views=build_views(cfg,mapping);views={n:all_views[n] for n in wanted}
        longroot=Path('outputs/recognition/pilots/v37_long512_2000');rr=rows(longroot/'long_labels.jsonl')
        for view in ('clean','noisy'):
            views['long512_'+view]=BucketCrops([r for r in rr if r['split']=='holdout' and r['view']==view],longroot,cfg,mapping)
        out={};profile={}
        for name,ds in views.items():
            assert len(ds)>0,name
            completed=WORK/f'{tag}_{name}.json'
            if completed.exists():
                previous=json.loads(completed.read_text(encoding='utf-8'))
                assert set(previous['metrics'])==set(ARMS) and len(previous['predictions']['greedy'])==len(ds)
                out[name]=dict(metrics=previous['metrics'],paired=previous['paired'],latency=previous['latency'],records=len(ds))
                continue
            write_json(WORK/'status.json',dict(state='forward',model=tag,probe=name,seconds=time.time()-started))
            cache=WORK/f'{tag}_{name}_cache';meta,arr=collect(model,ds,mapping,cache)
            decoded={a:[] for a in ARMS};milliseconds=defaultdict(list)
            for i,r in enumerate(meta['records']):
                if i%100==0:write_json(WORK/'status.json',dict(state='decoding',model=tag,probe=name,record=i,total=len(ds),seconds=time.time()-started))
                predictions={'greedy':r['greedy']};nbest=None
                for width in (5,10,25):
                    begin=time.perf_counter();beam=decode_beams(arr[f'b{i}'],arr[f'i{i}'],arr[f'p{i}'],width,mapping)
                    milliseconds[f'beam{width}'].append((time.perf_counter()-begin)*1000);predictions[f'beam{width}']=beam[0][0]
                    if width==25:nbest=beam
                for weight in (.1,.3):
                    arm=f'beam25_lex{weight}';begin=time.perf_counter();predictions[arm]=lex.rerank(nbest,weight)
                    milliseconds[arm].append((time.perf_counter()-begin)*1000+milliseconds['beam25'][-1])
                for arm,text in predictions.items():
                    errs=Counter(o['type'] for o in align_errors(r['truth'],text) if o['type']!='match')
                    decoded[arm].append(dict(r,prediction=text,error_counts={k:errs[k] for k in ('deletion','insertion','substitution')}))
            metrics={a:assess(rs) for a,rs in decoded.items()}
            effects={a:paired(decoded['greedy'],rs,[r['source_menu'] for r in rs] if name=='yolo_current_holdout' else None) for a,rs in decoded.items() if a!='greedy'}
            latency={a:dict(mean_ms=float(np.mean(ts)),p95_ms=float(np.percentile(ts,95))) for a,ts in milliseconds.items()}
            write_json(WORK/f'{tag}_{name}.json',dict(metrics=metrics,paired=effects,predictions=decoded,latency=latency,
                pruning={k:v for k,v in meta.items() if k!='records'}))
            out[name]=dict(metrics=metrics,paired=effects,latency=latency,records=len(ds));print(tag,name,{a:round(m['cer']*100,2) for a,m in metrics.items()},flush=True)
        assert all(torch.equal(v.cpu(),frozen[k]) for k,v in model.state_dict().items())
        guards={}
        for arm in ARMS:
            why=[];digit=out['digits_heldout']['metrics']
            if digit[arm]['digit_macro_recall']<digit['greedy']['digit_macro_recall']-.01:why.append('digit_recall_drop_gt1pp')
            short=out['fresh_short_clean']['metrics']
            for n in ('1','2'):
                if short[arm]['by_length'][n]['cer']>short['greedy']['by_length'][n]['cer']+.01:why.append('short'+n+'_drop_gt1pp')
            guards[arm]=why
        eligible=[a for a in ARMS if not guards[a]]
        selected=min(eligible,key=lambda a:out['yolo_current_holdout']['metrics'][a]['cer'])
        report['models'][tag]=dict(checkpoint=str(path),sha256=sha(path),all_weights_bn_buffers_unchanged=True,probes=out,
            development_selected_decoder=selected,decoder_guard_failures=guards)
        write_json(WORK/'report.json',report);del model;torch.cuda.empty_cache()
    report['completed']=True;report['seconds']=time.time()-started;write_json(WORK/'report.json',report)
    lines=['# V52: beam and menu-lexicon decoding','','FP16; pooled canonical parentheses; no training or promotion.','',
        '|Model|Decoder|YOLO CER|YOLO exact|Digit recall|Clean1char CER|Clean2char CER|Guard failures|','|---|---|---:|---:|---:|---:|---:|---|']
    for tag,r in report['models'].items():
        for arm in ARMS:
            y=r['probes']['yolo_current_holdout']['metrics'][arm];d=r['probes']['digits_heldout']['metrics'][arm];s=r['probes']['fresh_short_clean']['metrics'][arm]
            lines.append(f"|{tag}|{arm}|{y['cer']:.2%}|{y['exact']}/156|{d['digit_macro_recall']:.2%}|{s['by_length']['1']['cer']:.2%}|{s['by_length']['2']['cer']:.2%}|{','.join(r['decoder_guard_failures'][arm]) or 'none'}|")
    lines+=['','Decoder selection is development-only; fresh human-verified menu confirmation is still required.',
        'Desktop CPU decoding timings exclude JIT compilation and model forward; they are not mobile benchmarks.',
        'Search is approximate: top40nonblank tokens/frame and a relative12nat cutoff. Log probabilities are not renormalized after pruning.',
        'Lexicon only reranks the25acoustic candidates and cannot change their best digit sequence; unknown terms remain allowed.']
    (WORK/'analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(WORK/'status.json',dict(state='completed',seconds=time.time()-started))


if __name__=='__main__':main()
