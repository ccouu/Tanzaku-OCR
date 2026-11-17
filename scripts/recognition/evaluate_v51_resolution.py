"""Forward-only matched stride2 / H96 resolution diagnostics; no training."""
from collections import defaultdict
from pathlib import Path
import json
import math
import time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from outputs.recognition.v37_long512 import make_views
from outputs.recognition.application_mixture_pilot import RenderedLines
from scripts.recognition.train_v38_yolo25 import BucketCrops,INPUT
from scripts.recognition.prepare_v43 import normalize_dataset
from src.recognition.model import MenuRecognizer
from src.recognition.data import collate,decode
from src.recognition.canonical_ctc import fold_parenthesis_logits
from src.recognition.label_normalization import normalize_parentheses
from src.recognition.metrics import align_errors,summarise
from src.recognition.utils import sha,write_json
from scripts.recognition.review_v50 import paired

WORK=Path('outputs/recognition/evaluation/v51_resolution_zero_update')
V50=Path('outputs/recognition/pilots/v50_normalized_matched_2000')


def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]


def build_views(cfg,mapping):
    _,all_views=make_views(cfg,mapping)
    wanted=('original_bucket','historical_bucket','fresh_short_clean','fresh_short_noisy',
        'application_clean_control','application_texture_color_noise','strong_appearance')
    views={k:all_views[k] for k in wanted}
    roots={'brackets':Path('outputs/recognition/pilots/v48_parentheses_2000/targets'),
        'digits':Path('outputs/recognition/pilots/v43_digits_prices_2000/digits'),
        'target_multifont':Path('outputs/recognition/pilots/v44_latin_parentheses_long_2000/targeted')}
    for key,root in roots.items():
        rr=rows(root/'labels.jsonl')
        for cohort in ('known','heldout'):
            selected=[r for r in rr if r['split']=='probe' and r['cohort']==cohort and (key!='target_multifont' or r['kind']=='latin')]
            views[key+'_'+cohort]=BucketCrops(selected,root,cfg,mapping)
    yr=[r for r in rows(Path('outputs/recognition/training/v40_new_characters_2000/yolo_labels.jsonl')) if r['split']=='val_holdout']
    views['yolo_current_holdout']=BucketCrops(yr,INPUT,cfg,mapping)
    for ds in views.values():
        # Target tensors are unused in forward-only scoring. Keep OOV reference
        # text intact, rather than dropping probes or extending the model head.
        ds.mapping=defaultdict(lambda:0,mapping);normalize_dataset(ds)
    return views


def native_sample(row,cfg,height):
    with Image.open(INPUT/row['display_crop']) as im:
        strip=im.convert('RGB')
        if row['is_vertical']:strip=strip.transpose(Image.Transpose.ROTATE_90)
    # Same aspect-preserving fit as production; every limit scales with height.
    scale=min(height/strip.height,(height*8)/strip.width)
    w=min(height*8,max(1,round(strip.width*scale)));h=min(height,max(1,round(strip.height*scale)))
    resized=strip.resize((w,h),Image.Resampling.BICUBIC)
    canvas=next(c for c in (height*2,height*4,height*6,height*8) if w<=c)
    top=(height-h)//2
    arr=np.pad(np.asarray(resized),((top,height-h-top),(0,canvas-w),(0,0)),mode='edge')
    tensor=torch.from_numpy(arr.copy()).permute(2,0,1).float()/255
    tensor=(tensor-torch.tensor(cfg['mean'])[:,None,None])/torch.tensor(cfg['std'])[:,None,None]
    return tensor,w


def evaluate(model,ds,arm,cfg,mapping,native=False):
    samples=[ds[i] for i in range(len(ds))]
    groups=defaultdict(list)
    for i,s in enumerate(samples):groups[s[0].shape[-1]].append(i)
    records={};gpu_ms=0.;frame_total=0;peak_bytes=0;native_equal=0
    stride=2 if arm=='stride2' else 4
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for canvas,indices in sorted(groups.items()):
            for start in range(0,len(indices),32):
                ids=indices[start:start+32];ss=[samples[i] for i in ids]
                b=collate(ss,padded_width=canvas);images=b['images'];widths=[s[3] for s in ss]
                if native:
                    ns=[native_sample(ds.rows[i],cfg,96 if arm=='h96' else 64) for i in ids]
                    images=torch.stack([t for t,w in ns]);widths=[w for t,w in ns]
                    if arm=='baseline':native_equal+=sum(torch.equal(t,s[0]) and w==s[3] for (t,w),s in zip(ns,ss))
                elif arm=='h96':
                    images=F.interpolate(images,size=(96,round(canvas*1.5)),mode='bicubic',align_corners=False,antialias=True)
                    # Match legal RGB range after bicubic overshoot.
                    lo=(-torch.tensor(cfg['mean'])/torch.tensor(cfg['std']))[None,:,None,None]
                    hi=((1-torch.tensor(cfg['mean']))/torch.tensor(cfg['std']))[None,:,None,None]
                    images=torch.maximum(lo,torch.minimum(hi,images));widths=[round(w*1.5) for w in widths]
                lengths=torch.tensor([(w+stride-1)//stride for w in widths])
                images=images.cuda();orient=b['orientation'].cuda()
                torch.cuda.synchronize();begin=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.float16):logits=model(images,orient,lengths)
                torch.cuda.synchronize();gpu_ms+=(time.perf_counter()-begin)*1000
                frame_total+=int(lengths.sum());pooled=fold_parenthesis_logits(logits,mapping)
                raw=decode(logits,mapping,lengths);canon=decode(pooled,mapping,lengths)
                lp=pooled.float().log_softmax(-1)
                for j,i in enumerate(ids):
                    text=normalize_parentheses(b['texts'][j]);pred=canon[j];ops=align_errors(text,pred)
                    errs={k:sum(o['type']==k for o in ops) for k in ('deletion','insertion','substitution')}
                    valid=int(lengths[j]);prob=lp[:valid,j,mapping['(']].exp();rank=(pooled[:valid,j]>pooled[:valid,j,mapping['('],None]).sum(-1)+1
                    symbols={c:dict(reference=sum(o['truth']==c for o in ops),correct=sum(o['truth']==c and o['type']=='match' for o in ops)) for c in '()0123456789'}
                    records[i]=dict(index=i,truth=text,prediction=pred,literal_truth=b['texts'][j],literal_prediction=raw[j],
                        error_counts=errs,symbols=symbols,input_height=images.shape[-2],input_width=images.shape[-1],valid_t=valid,
                        peak_p_open=float(prob.max()),best_open_rank=int(rank.min()),
                        font=ds.rows[i].get('preferred_font',ds.rows[i].get('font_path')),
                        source_menu=ds.rows[i].get('source_menu'))
    ordered=[records[i] for i in range(len(ds))];stats=summarise(ordered)
    stats['by_length']={str(n):summarise([r for r in ordered if len(r['truth'])==n]) for n in sorted({len(r['truth']) for r in ordered})}
    stats['symbol_recall']={c:dict(reference=sum(r['symbols'][c]['reference'] for r in ordered),correct=sum(r['symbols'][c]['correct'] for r in ordered)) for c in '()0123456789'}
    stats['oov_reference_characters']=sorted({c for r in ordered for c in r['truth'] if c not in mapping})
    return dict(summary=stats,predictions=ordered,forward_cuda_ms=gpu_ms,ms_per_line=gpu_ms/len(ordered),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),valid_frames=frame_total,
        native_h64_exact_tensor_matches=native_equal if native and arm=='baseline' else None)


def main():
    WORK.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);assert torch.cuda.is_available()
    paths={'V43':Path('checkpoints/menu_recognizer_v43.pt'),'V50_control_selected':V50/'control/best.pt'}
    report=dict(training=False,precisions=['fp16'],arms=['baseline','stride2','h96'],checkpoints={},
        caveats=['Frozen-weight, frozen-BN zero-update diagnostic; changed geometry is out of training distribution.',
            'H96 synthetic inputs enlarge existing H64 pixels; they do not recover lost source detail.',
            'Separate yolo_native arm resizes from original display crops with correct vertical rotation.',
            'V43 lacks33Latinletters; reference text is retained and OOV targets are unused placeholders. Use V50 control for coverage-complete Latin comparisons.',
            'CUDA timings are local diagnostic measurements, not mobile latency benchmarks.'])
    started=time.time()
    for tag,path in paths.items():
        saved=torch.load(path,map_location='cpu',weights_only=False);mapping=saved['vocab']['char_to_id'];cfg=saved['config']
        model=MenuRecognizer(len(mapping)+1,use_bilstm=True).cuda().eval();model.load_state_dict(saved['model']);del saved
        initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        views=build_views(cfg,mapping);result={}
        for arm in report['arms']:
            model.encoder.blocks[0][0].conv.stride=(2,1) if arm=='stride2' else (2,2)
            model.eval();result[arm]={}
            for name,ds in list(views.items())+[('yolo_native',views['yolo_current_holdout'])]:
                write_json(WORK/'status.json',dict(state='evaluating',model=tag,arm=arm,probe=name,seconds=time.time()-started))
                d=evaluate(model,ds,arm,cfg,mapping,native=name=='yolo_native');result[arm][name]=d
                write_json(WORK/f'{tag}_{arm}_{name}.json',d)
                print(tag,arm,name,f"CER={d['summary']['cer']:.4f}",flush=True)
            assert all(torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items())
        report['checkpoints'][tag]=dict(path=str(path),sha256=sha(path),all_parameters_and_buffers_bit_identical=True,results={})
        for name in result['baseline']:
            base=result['baseline'][name];entry=dict(baseline=base['summary'],arms={})
            for arm in ('stride2','h96'):
                d=result[arm][name];menus=[r['source_menu'] for r in base['predictions']] if name.startswith('yolo') else None
                entry['arms'][arm]=dict(summary=d['summary'],paired=paired(base['predictions'],d['predictions'],menus),
                    forward_time_ratio=d['ms_per_line']/base['ms_per_line'],valid_frame_ratio=d['valid_frames']/base['valid_frames'])
            report['checkpoints'][tag]['results'][name]=entry
        del model;torch.cuda.empty_cache();write_json(WORK/'report.json',report)
    report['completed']=True;report['seconds']=time.time()-started;write_json(WORK/'report.json',report)
    write_json(WORK/'status.json',dict(state='completed',seconds=time.time()-started))
    lines=['# V51 frozen-weight resolution tests','','No training or promotion. All weights and BN buffers remained bit-identical.','',
        '|Checkpoint|Probe|Baseline CER|Stride2 CER|H96 CER|','|---|---|---:|---:|---:|']
    for tag,d in report['checkpoints'].items():
        for name,e in d['results'].items():lines.append(f"|{tag}|{name}|{e['baseline']['cer']:.2%}|{e['arms']['stride2']['summary']['cer']:.2%}|{e['arms']['h96']['summary']['cer']:.2%}|")
    lines+=['',*report['caveats']];(WORK/'analysis.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':main()
