"""V31 matched continuations: fresh unconstrained versus70% curated short renders."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time
import traceback
import zipfile

import numpy as np
from PIL import Image,ImageFilter
from fontTools.ttLib import TTFont
import torch
from torch import nn
from torch.utils.data import DataLoader

from outputs.recognition.application_pilot import glyph_mask,appearance,scaled_bucket
from outputs.recognition.application_mixture_pilot import ROOT,RenderedLines,Mixture,validations,cpu_copy
from outputs.recognition.short_retention_pilot import selection as previous_selection,retention as previous_retention
from src.recognition.datasets import FontLines
from src.recognition.font_training import evaluate,compact
from src.recognition.data import collate
from src.recognition.model import MenuRecognizer
from src.recognition.optimization import pilot_optimizer
from src.recognition.checkpointing import save_atomic
from src.recognition.utils import sha,seed_all,write_json,loss_for,training_device

CURATION=Path('outputs/recognition/curation/v31_short_terms')
V29=Path('outputs/recognition/pilots/v29_application_mixture_2000')
V30=Path('outputs/recognition/pilots/v30_short_retention_2000')
SOURCE=V30/'control/best.pt'


def rows(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def render(text,path,size,seed,texture,contrast,canvas=None):
    rng=random.Random(seed);nrng=np.random.default_rng(seed)
    for _ in range(1000):
        mask=glyph_mask(text,path,size,rng);width=round(mask.width*64/mask.height)
        if width<=384 and (canvas is None or (128 if width<=128 else 256 if width<=256 else 384)==canvas):break
    else:raise ValueError(f'Cannot fit{text!r} into{canvas}')
    color,_=appearance(mask,texture,contrast,rng,nrng);a=np.asarray(mask,dtype=float)/255
    clean=Image.fromarray(np.rint(242-222*a).astype('uint8')).convert('RGB')
    sigma=rng.uniform(1.5,3.5);blur=rng.uniform(.25,.55);pixels=np.asarray(color.filter(ImageFilter.GaussianBlur(blur)),dtype=float)
    noisy=Image.fromarray(np.rint(pixels+nrng.normal(0,sigma,pixels.shape)).clip(0,255).astype('uint8'))
    return {view:scaled_bucket(im) for view,im in (('clean_control',clean),('texture_color_noise',noisy))}


def prepare(work,args,cfg,mapping,status):
    approved=rows(CURATION/'approved_terms.jsonl');manifest=json.loads((ROOT/'manifest.json').read_text())
    paths={Path(p).name:Path(p) for values in manifest['paths'].values() for p in values}
    coverage={}
    for name,p in paths.items():
        with TTFont(p,fontNumber=0 if p.suffix.lower()=='.ttc' else -1) as f:coverage[name]={chr(c) for c in f.getBestCmap()}
    known=[Path(p) for c in ('old','new') for p in manifest['paths'][c]];held=[Path(p) for p in manifest['paths']['heldout']]
    generated=rows(V29/'labels.jsonl')
    fixed_wide_terms={r['text'] for r in generated if len(r['text'])<=2 and r['canvas']>128}
    rng=random.Random(args.seed);must=set('円茶酒水湯氷米肉魚塩酢卵貝牛豚鶏')
    holdout=set()
    for n in (1,2):
        pool=sorted({r['normalized'] for r in approved if r['length']==n and r['normalized'] not in must and r['normalized'] not in fixed_wide_terms})
        holdout.update(rng.sample(pool,max(1,round(len(pool)*.2))))
    write_json(work/'term_split.json',dict(held_from_curated_render_stream=sorted(holdout),
        caveat='Original replay and earlier training can contain these strings; this is not unseen-character or fully unseen-label evaluation.'))
    probe=[];(work/'menu_probe').mkdir()
    for n in (1,2):
        pool=[r['normalized'] for r in approved if r['length']==n and all(all(c in coverage[p.name] for c in r['normalized']) for p in known+held)]
        take=5 if args.smoke else 50
        reserved=[s for s in pool if s in holdout];seen=[s for s in pool if s not in holdout]
        count=min(take//2,len(reserved));selected=rng.sample(reserved,count)+rng.sample(seen,take-count)
        if n==1 and not args.smoke:
            for term in ('円','豚','鶏','魚'):
                if term in pool and term not in selected:selected[-1-len([s for s in selected if s in must])%max(1,take-count)]=term
        assert len(set(selected))==len(selected)
        for ti,text in enumerate(selected):
            for fi in range(4):
                path=known[(ti*2+fi)%len(known)] if fi<2 else held[(ti*2+fi-2)%len(held)]
                seed=args.seed+900001+n*100000+ti*101+fi;size=random.Random(seed).randint(24,120)
                images=render(text,path,size,seed,('paper','wood','chalkboard','cloth')[(ti+fi)%4],('low','medium','high')[ti%3])
                for view,(image,width,canvas) in images.items():
                    dest=work/'menu_probe'/f'{n}_{ti:02d}_{fi}_{view}.png';image.save(dest)
                    probe.append(dict(text=text,truth=text,image=str(dest.relative_to(work)),sha256=sha(dest),
                        canvas=canvas,valid_width=width,view=view,preferred_font=path.name,
                        cohort='heldout' if fi>=2 else 'known',term_split='held_curated' if text in holdout else 'seen_curated'))
    (work/'menu_probe_labels.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in probe),encoding='utf-8')
    batches=json.loads((V30/'control_batches.json').read_text())[:args.steps]
    used={i for batch in batches for k,i in batch if k!='original'}
    pair_views={}
    for i in used:pair_views.setdefault(generated[i]['index'],{})[generated[i]['view']]=i
    short_pairs=sorted(pair for pair,ix in pair_views.items() if len(generated[next(iter(ix.values()))]['text'])<=2)
    curated=set()
    for n in (1,2):
        pool=[pair for pair in short_pairs if len(generated[next(iter(pair_views[pair].values()))]['text'])==n]
        eligible=[p for p in pool if generated[next(iter(pair_views[p].values()))]['canvas']==128]
        assert len(eligible)>=round(len(pool)*.7)
        curated.update(rng.sample(eligible,round(len(pool)*.7)))
    variants={arm:[dict(r,image=str((V29/r['image']).resolve())) for r in generated] for arm in ('control','curated_70')}
    for arm in variants:(work/arm).mkdir();(work/arm/'rendered').mkdir()
    random_pools={}
    for r in generated:
        n=len(r['text'])
        if n<=2 and r['text'] not in holdout:random_pools.setdefault((r['font'],n),set()).add(r['text'])
    for count,pair in enumerate(short_pairs,1):
        indices=pair_views[pair];base=generated[indices['clean_control']];n=len(base['text']);path=paths[base['font']]
        if base['canvas']>128:
            # Do not force ordinary terms into rare, inherently wide legacy crop layouts.
            # Both arms replay these exact pixels; selection still totals70% per length.
            assert pair not in curated and base['text'] not in holdout
            continue
        control_text=base['text']
        if control_text in holdout:control_text=rng.choice(sorted(random_pools[(base['font'],n)]))
        target=control_text
        if pair in curated:
            pool=[r['normalized'] for r in approved if r['length']==n and r['normalized'] not in holdout and all(c in coverage[path.name] for c in r['normalized'])]
            assert pool
            target=rng.choices(pool,weights=[3 if s in ('円','豚','鶏','魚') else 1 for s in pool])[0]
        for arm,text in (('control',control_text),('curated_70',target)):
            if arm=='curated_70' and target==control_text:
                for i in indices.values():variants[arm][i]=dict(variants['control'][i],curated_slot=pair in curated)
                continue
            images=render(text,path,base['native_font_size'],args.seed+pair*1009,base['texture'],base['contrast'],base['canvas'])
            for view,(image,width,canvas) in images.items():
                i=indices[view];dest=work/arm/'rendered'/f'{pair:05d}_{view}.png';image.save(dest,compress_level=1)
                variants[arm][i]=dict(generated[i],text=text,truth=text,image=str(dest.resolve()),sha256=sha(dest),
                    valid_width=width,canvas=canvas,curated_slot=pair in curated,term_split='seen_curated')
        if count%250==0 or count==len(short_pairs):status(state='rendering',pairs_done=count,pairs_total=len(short_pairs))
    for pair in short_pairs:
        for view,i in pair_views[pair].items():
            a,b=variants['control'][i],variants['curated_70'][i]
            assert len(a['text'])==len(b['text']) and a['canvas']==b['canvas'] and a['font']==b['font']
            assert a['text'] not in holdout and b['text'] not in holdout
            if pair not in curated:assert a['image']==b['image'] and a['sha256']==b['sha256']
    audit=dict(updates_per_arm=args.steps,presentations_per_arm=args.steps*32,original_presentations=args.steps*16,
        clean_presentations=args.steps*8,noisy_presentations=args.steps*8,short_generated_pairs=len(short_pairs),curated_pairs=len(curated),
        curated_fraction=len(curated)/len(short_pairs) if short_pairs else None,
        fixed_wide_short_pairs=sum(generated[next(iter(pair_views[p].values()))]['canvas']>128 for p in short_pairs),
        per_length={str(n):dict(short_pairs=sum(len(generated[next(iter(pair_views[p].values()))]['text'])==n for p in short_pairs),
            curated_pairs=sum(len(generated[next(iter(pair_views[p].values()))]['text'])==n for p in curated)) for n in (1,2)},
        original_records_and_batch_order_identical=True,font_length_orientation_canvas_assignments_identical=True,
        generated_holdout_label_overlap=0,random30_pixels_shared=True,
        actual_changed_short_pairs=sum(variants['control'][pair_views[p]['clean_control']]['text']!=variants['curated_70'][pair_views[p]['clean_control']]['text'] for p in short_pairs),
        rendering_note='Same per-pair font,size,seed,appearance policy and natural bucket; real crop dimensions and valid lengths can differ with glyph identity. No glyph clipping/squeezing.',
        treatment_short_counts=dict(Counter(variants['curated_70'][indices['clean_control']]['text'] for p,indices in pair_views.items() if p in curated)))
    write_json(work/'sampling.json',audit);write_json(work/'batches.json',batches)
    for arm,records in variants.items():(work/arm/'labels.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    return probe,variants,batches,audit


def menu_score(report):return sum(report[k]['0.0']['by_length'][n]['cer'] for k in ('menu_clean','menu_noisy') for n in ('1','2'))/4
def selection(report):return (menu_score(report)+previous_selection(report))/2
def retention(report,initial):
    base=previous_retention(report,initial)
    for key in ('fresh_short_clean','fresh_short_noisy'):
        for n in ('1','2'):
            name=key+n;v=report[key]['0.0']['by_length'][n]['cer'];cap=initial[key]['0.0']['by_length'][n]['cer']+.01
            base['current'][name]=v;base['caps'][name]=cap
            if v>cap+1e-12:base['failed'].append(name)
    base['eligible']=not base['failed'];return base


def run(args):
    work=args.output;work.mkdir(parents=True,exist_ok=False);start=time.time()
    status=lambda **kw:write_json(work/'status.json',dict(elapsed_seconds=time.time()-start,**kw))
    try:
        torch.set_num_threads(4);seed_all(args.seed);device=training_device('cuda:0','RTX 3070')
        digest=sha(SOURCE);saved=torch.load(SOURCE,map_location='cpu',weights_only=False);assert saved['arm']=='control' and saved['updates']==2000
        cfg=saved['config'];mapping=saved['vocab']['char_to_id'];train=FontLines(ROOT,'train',cfg['mean'],cfg['std'])
        probe,variants,batches,audit=prepare(work,args,cfg,mapping,status)
        views=validations(cfg,mapping,False)
        for view in ('clean','noisy'):
            old=rows(V30/'short_probe_labels.jsonl');views['fresh_short_'+view]=RenderedLines([r for r in old if r['view']==view],V30,cfg,mapping)
            kind='clean_control' if view=='clean' else 'texture_color_noise'
            views['menu_'+view]=RenderedLines([r for r in probe if r['view']==kind],work,cfg,mapping)
        if args.smoke:
            for name,ds in views.items():
                if name in ('seen','heldout'):
                    fs=sorted({r['preferred_font'] for r in ds.rows});ds.rows=[next(r for r in ds.rows if r['preferred_font']==f and len(r['text'])==n) for f in fs for n in (1,2)]
                elif not name.startswith('menu_'):ds.rows=[r for n in range(1,11) for r in [r for r in ds.rows if len(r['text'])==n][:4]]
        config=dict(source=str(SOURCE),source_sha256=digest,steps_per_arm=args.steps,seed=args.seed,smoke=args.smoke,
            encoder_lr=5e-7,head_lr=3e-6,lr_schedule='constant',mixture=[.5,.25,.25],short_text_quota_unchanged=True,
            curated_short_fraction=.7,approved_sha256=sha(CURATION/'approved_terms.jsonl'),
            selection='equal new menu short macro and prior V30 composite; beta0; existing retention plus original fresh short strata +1pp',
            limits='Random control uses existing unconstrained V29 short labels,rerendered fresh. Treatment changes70%of short generated pairs only. Original50% replay and all longer generated images remain identical.')
        config['wide_short_policy']='Rare256px short pairs replay unchanged in both arms;70%overall short-generated selection is drawn from eligible128px pairs.'
        write_json(work/'run_config.json',config)
        with zipfile.ZipFile(work/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in list(Path('src/recognition').glob('*.py'))+[Path(__file__),Path('outputs/recognition/application_pilot.py'),Path('outputs/recognition/application_mixture_pilot.py'),Path('outputs/recognition/short_retention_pilot.py')]:z.write(p,p.as_posix())
        model=MenuRecognizer(len(mapping)+1,use_bilstm=True).to(device);model.load_state_dict(saved['model']);model.vocabulary=mapping
        validation=lambda:{name:evaluate(model,ds,device) for name,ds in views.items()}
        status(state='initial_validation');initial=validation();write_json(work/'initial.json',initial);histories={}
        for arm,records in variants.items():
            seed_all(args.seed+901);model.load_state_dict(saved['model']);optimizer=pilot_optimizer(model,5e-7,3e-6);criterion=nn.CTCLoss(blank=0,zero_infinity=False)
            folder=work/arm;mixture=Mixture(train,RenderedLines(records,work,cfg,mapping));best_score=selection(initial);best_step=0
            history=[dict(step=0,score=best_score,retention=retention(initial,initial),validation=compact(initial))]
            def snapshot(step):return dict(model=cpu_copy(model.state_dict()),optimizer=cpu_copy(optimizer.state_dict()),vocab=saved['vocab'],
                config=dict(cfg,curated_short_pilot=config),updates=step,arm=arm,best_step=best_step,best_cer=best_score,
                rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all())
            save_atomic(folder/'best.pt',snapshot(0));write_json(folder/'best_validation.json',dict(step=0,score=best_score,validation=initial));total_loss=0.
            for step,batch in enumerate(DataLoader(mixture,batch_sampler=batches,num_workers=0,collate_fn=collate),1):
                model.train();optimizer.zero_grad(set_to_none=True);loss=loss_for(model,batch,device,criterion);loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();total_loss+=float(loss.detach())
                if step%50==0:status(state='training',arm=arm,update=step,total_updates_per_arm=args.steps,mean_loss=total_loss/step,best_step=best_step)
                if step%500==0 or step==args.steps:
                    status(state='validation',arm=arm,update=step);report=validation();value=selection(report);g=retention(report,initial)
                    if g['eligible'] and value<best_score:
                        best_score=value;best_step=step;save_atomic(folder/'best.pt',snapshot(step));write_json(folder/'best_validation.json',dict(step=step,score=value,retention=g,validation=report))
                    entry=dict(step=step,score=value,retention=g,mean_loss=total_loss/step,best_step=best_step,validation=compact(report));history.append(entry)
                    write_json(folder/f'step_{step:04d}.json',dict(entry,validation=report))
                    with (folder/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
                    save_atomic(folder/f'step_{step:04d}.pt',snapshot(step));save_atomic(folder/'last.pt',snapshot(step))
                    print('EVAL',arm,step,'menu',menu_score(report),'eligible',g['eligible'],'best',best_step,flush=True)
            histories[arm]=dict(best_step=best_step,best_score=best_score,history=history);write_json(folder/'report.json',histories[arm])
        assert sha(SOURCE)==digest;write_json(work/'report.json',dict(arms=histories,source_unchanged=True,steps_per_arm=args.steps,sampling=audit,elapsed_seconds=time.time()-start))
        lines=['# V31 reviewed short-term pilot','', 'Matched2,000-update arms from V30 control. Original replay,batch order,lengths,font settings and physical buckets are identical. Fresh short renders in both arms; treatment changes70%of short generated labels to approved menu terms. No higher overall short-text quota.',
            '', '| Arm | Update | Menu clean1 | Menu clean2 | Menu noisy1 | Menu noisy2 | App clean | App noisy | Seen | Held-out | Retention |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        for arm,data in histories.items():
            for e in data['history']:
                r=e['validation'];v=[r[k]['0.0']['by_length'][n]['cer'] for k in ('menu_clean','menu_noisy') for n in ('1','2')]+[r[k]['0.0']['all']['cer'] for k in ('application_clean_control','application_texture_color_noise','seen','heldout')]
                lines.append('| '+arm+' | '+str(e['step'])+' | '+' | '.join(f'{x:.2%}' for x in v)+' | '+('pass' if e['retention']['eligible'] else ', '.join(e['retention']['failed']))+' |')
        lines.extend(['', 'See sampling.json for exact rounded70/30 fractions and per-term counts. Term split refers only to the fresh generated short stream,not historical exposure. New menu-probe images are frozen before training. No lexicon corrections are applied. best.pt may remain the starting checkpoint if no eligible improvement occurs.'])
        (work/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        status(state='completed',steps_per_arm=args.steps,best_steps={a:r['best_step'] for a,r in histories.items()},source_unchanged=True)
    except Exception:status(state='failed',error=traceback.format_exc());raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/recognition/pilots/v31_curated_short_2000'))
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--seed',type=int,default=20261206);p.add_argument('--smoke',action='store_true');a=p.parse_args();assert 0<a.steps<=2000;run(a)


if __name__=='__main__':main()
