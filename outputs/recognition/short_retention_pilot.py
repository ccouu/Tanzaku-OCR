"""V30: matched V29 continuations; current versus 20/20/60 length sampling."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import time
import traceback
import zipfile

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from fontTools.ttLib import TTFont
import torch
from torch import nn
from torch.utils.data import DataLoader

from outputs.recognition.application_pilot import glyph_mask, appearance, scaled_bucket
from outputs.recognition.application_mixture_pilot import (
    ROOT, RenderedLines, Mixture, validations, selection as application_selection, cpu_copy)
from src.recognition.datasets import FontLines
from src.recognition.font_training import evaluate, compact, gate, retention_values, row_canvas
from src.recognition.data import collate
from src.recognition.metrics import summarise
from src.recognition.model import MenuRecognizer
from src.recognition.optimization import pilot_optimizer
from src.recognition.checkpointing import save_atomic
from src.recognition.utils import seed_all, sha, write_json, loss_for, training_device

V29=Path('outputs/recognition/pilots/v29_application_mixture_2000')
SOURCE=V29/'best.pt'


def build_probe(work, train, args):
    rng=random.Random(args.seed+37);nrng=np.random.default_rng(args.seed+37)
    source=json.loads((ROOT/'manifest.json').read_text())['paths']
    known=[Path(p) for c in ('old','new') for p in source[c]]
    held=[Path(p) for p in source['heldout']]
    cmaps={}
    for path in known+held:
        with TTFont(path,fontNumber=0 if path.suffix.lower()=='.ttc' else -1) as f:cmaps[str(path)]=set(f.getBestCmap())
    used={json.loads(s)['text'] for s in (V29/'labels.jsonl').read_text(encoding='utf-8').splitlines()}
    # New full strings, drawn from menu-training characters/substrings; no new classes.
    pools={n:sorted({r['text'][i:i+n] for r in train.rows for i in range(len(r['text'])-n+1)}-used) for n in (1,2)}
    folder=work/'short_probe';folder.mkdir();rows=[];selected=set()
    count=20 if args.smoke else 200
    for n in (1,2):
        for j in range(count):
            cohort='known' if j<count//2 else 'heldout'
            roster=known if cohort=='known' else held
            path=roster[j%len(roster)]
            candidates=[s for s in pools[n] if s not in selected and all(ord(c) in cmaps[str(path)] for c in s)]
            assert candidates,(path,n)
            text=rng.choice(candidates);selected.add(text);size=rng.randint(24,120)
            mask=glyph_mask(text,path,size,rng)
            assert round(mask.width*64/mask.height)<=384
            pair=(n-1)*count+j;texture=('paper','wood','chalkboard','cloth')[pair%4];contrast=('low','medium','high')[pair%3]
            color,meta=appearance(mask,texture,contrast,rng,nrng)
            alpha=np.asarray(mask,dtype=float)/255
            clean=Image.fromarray(np.rint(242-222*alpha).astype('uint8')).convert('RGB')
            sigma=rng.uniform(1.5,3.5);blur=rng.uniform(.25,.55)
            a=np.asarray(color.filter(ImageFilter.GaussianBlur(blur)),dtype=float)
            noisy=Image.fromarray(np.rint(a+nrng.normal(0,sigma,a.shape)).clip(0,255).astype('uint8'))
            for view,image in (('clean',clean),('noisy',noisy)):
                padded,width,canvas=scaled_bucket(image);dest=folder/f'{pair:03d}_{view}.png';padded.save(dest)
                rows.append(dict(index=pair,text=text,truth=text,view=view,font=path.name,preferred_font=path.name,
                    cohort=cohort,orientation_flag=0,image=str(dest.relative_to(work)),sha256=sha(dest),
                    valid_width=width,canvas=canvas,native_font_size=size,native_size=list(mask.size),
                    texture=texture,contrast=contrast,noise_sigma=sigma,blur_radius=blur,**meta))
    assert len(selected)==2*count and not selected.intersection(used)
    (work/'short_probe_labels.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    write_json(work/'short_probe_manifest.json',dict(seed=args.seed+37,examples=2*count,images=len(rows),
        unique_transcriptions=len(selected),known_and_heldout_font_fraction=[.5,.5],
        generated_v29_training_text_overlap=0,current_original_replay_excludes_probe_text=True,
        caveat='New renders and full strings relative to V29 generated images; constituent glyphs were trained. Original historical training may already contain these strings. Some two-character labels are menu substrings, not complete dish names.'))
    preview=Image.new('RGB',(800,20*90),'#dddddd');draw=ImageDraw.Draw(preview)
    for j,pair in enumerate(range(0,2*count,max(1,2*count//20))):
        if j>=20:break
        for col,view in enumerate(('clean','noisy')):
            r=next(r for r in rows if r['index']==pair and r['view']==view)
            with Image.open(work/r['image']) as im:preview.paste(im,(col*400,j*90+23))
            draw.text((col*400,j*90+3),f'{pair} {view} {r["cohort"]} {r["contrast"]}',fill='black')
    preview.save(work/'short_probe_preview.png')
    return rows,selected


def make_plans(train,generated,excluded,args):
    rng=random.Random(args.seed+71)
    old=json.loads((V29/'batches.json').read_text())[:args.steps]
    # Replay the V29 control exactly except newly reserved probe strings in original rows.
    by_group=defaultdict(list)
    for i,r in enumerate(train.rows):
        if r['text'] not in excluded:by_group[(row_canvas(r),len(r['text']),r['cohort'])].append(i)
    control=[];replaced=0
    for batch in old:
        new=[]
        for kind,i in batch:
            if kind=='original' and train.rows[i]['text'] in excluded:
                row=train.rows[i];i=rng.choice(by_group[(row_canvas(row),len(row['text']),row['cohort'])]);replaced+=1
            new.append((kind,i))
        control.append(new)
    # Immutable paired images from V29 are shared between the arms; only sampling changes.
    clean={r['index']:i for i,r in enumerate(generated) if r['view']=='clean_control'}
    noisy={r['index']:i for i,r in enumerate(generated) if r['view']=='texture_color_noise'}
    assert clean.keys()==noisy.keys()
    original=defaultdict(list);pairs=defaultdict(list)
    for (canvas,n,c),ids in by_group.items():original[(canvas,1 if n==1 else 2 if n==2 else 3,c)].extend(ids)
    for pair,i in clean.items():
        r=generated[i];assert r['text'] not in excluded
        n=len(r['text']);pairs[(r['canvas'],1 if n==1 else 2 if n==2 else 3,r['cohort'])].append(pair)
    short=args.steps//5;categories=[1]*short+[2]*short+[3]*(args.steps-2*short);rng.shuffle(categories)
    treatment=[]
    for cat in categories:
        canvases=[w for w in (128,256,384) if all(original[(w,cat,c)] and pairs[(w,cat,c)] for c in ('old','new'))]
        assert canvases,cat
        weights=[sum(len(pairs[(w,cat,c)]) for c in ('old','new')) for w in canvases]
        canvas=rng.choices(canvases,weights=weights)[0];batch=[]
        for cohort in ('old','new'):
            for _ in range(8):batch.append(('original',rng.choice(original[(canvas,cat,cohort)])))
            for _ in range(4):
                pair=rng.choice(pairs[(canvas,cat,cohort)]);batch.extend([('clean_control',clean[pair]),('texture_color_noise',noisy[pair])])
        rng.shuffle(batch);treatment.append(batch)
    plans={'control':control,'short_20_20':treatment};audits={}
    for arm,batches in plans.items():
        counts=Counter();lengths=defaultdict(Counter);fonts=defaultdict(Counter);cohorts=defaultdict(Counter);canvases=Counter();original_ids=[]
        for batch in batches:
            assert len(batch)==32 and Counter(k for k,i in batch)=={'original':16,'clean_control':8,'texture_color_noise':8}
            widths=set()
            for kind,i in batch:
                row=train.rows[i] if kind=='original' else generated[i]
                assert row['text'] not in excluded
                widths.add(row_canvas(row) if kind=='original' else row['canvas'])
                counts[kind]+=1;lengths[kind][len(row['text'])]+=1
                fonts[kind][row.get('preferred_font',row.get('font'))]+=1;cohorts[kind][row['cohort']]+=1
                if kind=='original':original_ids.append(i)
            assert len(widths)==1;canvases[next(iter(widths))]+=1
        if arm=='short_20_20':
            for stream,mult in (('original',16),('clean_control',8),('texture_color_noise',8)):
                assert lengths[stream][1]==short*mult and lengths[stream][2]==short*mult
        audits[arm]=dict(updates=len(batches),presentations=sum(counts.values()),stream_counts=dict(counts),
            lengths={k:dict(v) for k,v in lengths.items()},font_counts={k:dict(v) for k,v in fonts.items()},
            cohort_counts={k:dict(v) for k,v in cohorts.items()},canvas_batches=dict(canvases),
            unique_original_records=len(set(original_ids)),probe_text_overlap=0,
            plan_sha256=hashlib.sha256(json.dumps(batches).encode()).hexdigest(),
            control_reserved_text_replacements=replaced if arm=='control' else None)
    return plans,audits


def short_metrics(report):
    values=[]
    for key in ('fresh_short_clean','fresh_short_noisy'):
        for n in ('1','2'):values.append(report[key]['0.0']['by_length'][n]['cer'])
    return sum(values)/4


def selection(report):return (short_metrics(report)+application_selection(report))/2


def extra_retention(report):
    result={f'application_{view}':report[f'application_{view}']['0.0']['all']['cer'] for view in ('clean_control','texture_color_noise')}
    result['heldout_total']=report['heldout']['0.0']['all']['cer']
    for n in (1,2):
        rows=[r for key in ('seen','heldout','original_bucket') for r in report[key]['0.0']['predictions'] if len(r['truth'])==n]
        result[f'pooled_original_{n}char']=summarise(rows)['cer']
    return result


def retention(report,baseline):
    basic=gate(report,retention_values(baseline));values=extra_retention(report);caps={k:v+.01 for k,v in extra_retention(baseline).items()}
    failed=basic['failed']+[k for k in values if values[k]>caps[k]+1e-12]
    return dict(eligible=not failed,failed=failed,current=dict(basic['current'],**values),caps=dict(basic['caps'],**caps))


def run(args):
    work=args.output;work.mkdir(parents=True,exist_ok=False);started=time.time()
    def status(**kw):write_json(work/'status.json',dict(elapsed_seconds=time.time()-started,**kw))
    try:
        seed_all(args.seed);torch.set_num_threads(4);device=training_device('cuda:0','RTX 3070')
        digest=sha(SOURCE);saved=torch.load(SOURCE,map_location='cpu',weights_only=False);assert saved['updates']==2000
        cfg=saved['config'];mapping=saved['vocab']['char_to_id'];train=FontLines(ROOT,'train',cfg['mean'],cfg['std'])
        status(state='freezing_probe');probe,excluded=build_probe(work,train,args)
        generated=[json.loads(s) for s in (V29/'labels.jsonl').read_text(encoding='utf-8').splitlines()]
        plans,audits=make_plans(train,generated,excluded,args);write_json(work/'sampling.json',audits)
        for arm,plan in plans.items():write_json(work/f'{arm}_batches.json',plan)
        views=validations(cfg,mapping,False)
        if args.smoke:
            for name,ds in views.items():
                if name in ('seen','heldout'):
                    fonts=sorted({r['preferred_font'] for r in ds.rows})
                    ds.rows=[next(r for r in ds.rows if r['preferred_font']==f and len(r['text'])==n) for f in fonts for n in (1,2)]
                else:
                    ds.rows=[r for n in range(1,11) for r in [r for r in ds.rows if len(r['text'])==n][:4]]
        for view in ('clean','noisy'):views[f'fresh_short_{view}']=RenderedLines([r for r in probe if r['view']==view],work,cfg,mapping)
        config=dict(source=str(SOURCE),source_sha256=digest,updates_per_arm=args.steps,seed=args.seed,smoke=args.smoke,
            encoder_lr=5e-7,head_lr=3e-6,schedule='constant',batch=32,mixture=[.5,.25,.25],
            reused_training_images=True,generated_labels_sha256=sha(V29/'labels.jsonl'),
            probe_labels_sha256=sha(work/'short_probe_labels.jsonl'),
            treatment='20% one character,20% two characters,60%3-10 within every image stream over complete run',
            selection='half fresh short macro CER,half V29 application/clean-font composite; primary beta0',
            retention='+1 absolute CER point: original4 gates,heldout total,pooled original1char/2char,application clean/noisy',
            matching_limits='Same source pixels, architecture,learning rates,budget,decoder. Sampling short lengths necessarily shifts bucket/font/orientation frequencies and unique presentations; counts audited. Treatment balances font cohorts50/50, control retains prior approximately50/50 mix. Batch length grouping differs.')
        write_json(work/'run_config.json',config)
        with zipfile.ZipFile(work/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in list(Path('src/recognition').glob('*.py'))+[Path(__file__),Path('outputs/recognition/application_mixture_pilot.py'),Path('outputs/recognition/application_pilot.py')]:z.write(p,p.as_posix())
        model=MenuRecognizer(len(mapping)+1,use_bilstm=True).to(device);model.load_state_dict(saved['model']);model.vocabulary=mapping
        def validation():return {name:evaluate(model,ds,device) for name,ds in views.items()}
        status(state='initial_validation');initial=validation();write_json(work/'initial.json',initial)
        mixture=Mixture(train,RenderedLines(generated,V29,cfg,mapping));all_history={}
        for arm,batches in plans.items():
            seed_all(args.seed+101);model.load_state_dict(saved['model']);optimizer=pilot_optimizer(model,5e-7,3e-6)
            criterion=nn.CTCLoss(blank=0,zero_infinity=False);folder=work/arm;folder.mkdir()
            best_score=selection(initial);best_step=0;history=[dict(step=0,score=best_score,retention=retention(initial,initial),validation=compact(initial))]
            def snapshot(step):return dict(model=cpu_copy(model.state_dict()),optimizer=cpu_copy(optimizer.state_dict()),
                vocab=saved['vocab'],config=dict(cfg,short_retention_pilot=config),updates=step,arm=arm,
                best_step=best_step,best_cer=best_score,rng_python=random.getstate(),rng_numpy=np.random.get_state(),
                rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all())
            save_atomic(folder/'best.pt',snapshot(0));write_json(folder/'best_validation.json',dict(step=0,score=best_score,validation=initial))
            loss_sum=0.
            for step,batch in enumerate(DataLoader(mixture,batch_sampler=batches,num_workers=0,collate_fn=collate),1):
                model.train();optimizer.zero_grad(set_to_none=True);loss=loss_for(model,batch,device,criterion)
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();loss_sum+=float(loss.detach())
                if step%50==0:status(state='training',arm=arm,update=step,total_updates_per_arm=args.steps,mean_loss=loss_sum/step,best_step=best_step)
                if step%500==0 or step==args.steps:
                    status(state='validation',arm=arm,update=step);report=validation();value=selection(report);gate_result=retention(report,initial)
                    if gate_result['eligible'] and value<best_score:
                        best_score=value;best_step=step;save_atomic(folder/'best.pt',snapshot(step))
                        write_json(folder/'best_validation.json',dict(step=step,score=value,retention=gate_result,validation=report))
                    entry=dict(step=step,score=value,retention=gate_result,mean_loss=loss_sum/step,best_step=best_step,validation=compact(report))
                    history.append(entry);write_json(folder/f'step_{step:04d}.json',dict(entry,validation=report))
                    with (folder/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
                    save_atomic(folder/f'step_{step:04d}.pt',snapshot(step));save_atomic(folder/'last.pt',snapshot(step))
                    print('EVAL',arm,step,'short',short_metrics(report),'score',value,'eligible',gate_result['eligible'],'best',best_step,flush=True)
            all_history[arm]=dict(best_step=best_step,best_score=best_score,history=history)
            write_json(folder/'report.json',all_history[arm])
        assert sha(SOURCE)==digest
        write_json(work/'report.json',dict(arms=all_history,sampling=audits,source_unchanged=True,updates_per_arm=args.steps,elapsed_seconds=time.time()-started))
        lines=['# V30 short-text retention pilot','', 'Both arms continue V29 update2000 for2,000 updates, encoder5e-7/head3e-6,50/25/25 image streams, unchanged source images. Standard greedy CTC at beta0. Fresh probe:200 examples per length, clean/noisy paired views, half known and half held-out fonts.',
            '', '| Arm | Update | Fresh clean1 | Fresh clean2 | Fresh noisy1 | Fresh noisy2 | App clean | App noisy | Seen | Held-out | Original | Retention |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        for arm,data in all_history.items():
            for entry in data['history']:
                r=entry['validation'];vals=[r[k]['0.0']['by_length'][str(n)]['cer'] for k in ('fresh_short_clean','fresh_short_noisy') for n in (1,2)]
                vals += [r[k]['0.0']['all']['cer'] for k in ('application_clean_control','application_texture_color_noise','seen','heldout','original_bucket')]
                lines.append('| '+arm+' | '+str(entry['step'])+' | '+' | '.join(f'{v:.2%}' for v in vals)+' | '+('pass' if entry['retention']['eligible'] else ', '.join(entry['retention']['failed']))+' |')
            lines.extend(['',f"{arm}: best eligible update{data['best_step']}. Step0 remains eligible.",''])
        lines.extend(['','Selection and constraints were fixed before training. Fresh probes use new renderings/full strings relative to V29 generated training, but trained characters and procedural textures; this is not real-menu accuracy. Historical training may already contain some probe strings. Current replay excludes all fresh-probe strings in both arms.',
            '', 'Control reuses V29 batches except reserved-probe original labels are replaced within the same length/cohort/bucket. Treatment uses20% one-character and20% two-character batches, with60%3–10, sampling with replacement. Homogeneous length grouping, bucket mix, marginal font/orientation frequencies and unique-image coverage can differ; sampling.json audits these limits. This compares length-focused replay policies, not a perfectly isolated per-example weighting effect.'])
        (work/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        status(state='completed',updates_per_arm=args.steps,best_steps={a:r['best_step'] for a,r in all_history.items()},source_unchanged=True)
        print('COMPLETE',work,flush=True)
    except Exception:
        status(state='failed',error=traceback.format_exc());raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/recognition/pilots/v30_short_retention_2000'))
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--seed',type=int,default=20261202);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();assert args.steps>0 and args.steps<=2000;run(args)


if __name__=='__main__':main()
