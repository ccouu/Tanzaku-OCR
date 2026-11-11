"""Single-arm 50/25/25 application-render continuation of V27; no active code changes."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random
import time
import traceback
import zipfile

import numpy as np
from PIL import Image, ImageFilter
from fontTools.ttLib import TTFont
import torch
from torch import nn
from torch.utils.data import DataLoader

from outputs.recognition.application_pilot import glyph_mask, appearance, scaled_bucket
from src.recognition.data import preprocess, collate, minimum_ctc_steps
from src.recognition.datasets import FontLines
from src.recognition.font_training import (ValidationFonts, ValidationClean, ValidationHistorical,
    FrozenInputs, evaluate, compact, score, retention_values, gate, row_canvas)
from src.recognition.model import MenuRecognizer
from src.recognition.optimization import pilot_optimizer
from src.recognition.checkpointing import save_atomic
from src.recognition.utils import sha, write_json, seed_all, loss_for, training_device

ROOT=Path('data/datasets/recognition/v27_fonts')
PROBE=Path('outputs/recognition/evaluation/v27_application_synthetic')
SOURCE=Path('outputs/recognition/training/v27_font_full20/epoch_20.pt')
DEFAULT_OUT=Path('outputs/recognition/pilots/v29_application_mixture_2000')


class RenderedLines:
    def __init__(self, rows, root, cfg, mapping):
        self.rows=rows;self.root=root;self.cfg=cfg;self.mapping=mapping;self.verified=set()
    def __len__(self):return len(self.rows)
    def canvas_at(self,i):return self.rows[i]['canvas']
    def __getitem__(self,i):
        row=self.rows[i];path=self.root/row['image']
        if i not in self.verified:
            assert sha(path)==row['sha256'];self.verified.add(i)
        with Image.open(path) as image:tensor,w,compressed=preprocess(image,self.cfg['mean'],self.cfg['std'],'batch')
        assert w==row['canvas'] and not compressed
        assert (row['valid_width']+3)//4>=minimum_ctc_steps(row['text'])
        return tensor,torch.tensor([self.mapping[c] for c in row['text']]),0,row['valid_width'],row['text'],False


class Mixture:
    def __init__(self, original, rendered):self.original=original;self.rendered=rendered
    def __getitem__(self,key):
        kind,index=key
        return self.original[index] if kind=='original' else self.rendered[index]


def validations(cfg,mapping,smoke):
    views={s:ValidationFonts(ROOT,s,cfg['mean'],cfg['std']) for s in ('seen','heldout')}
    clean=Path('data/datasets/recognition/v27_validation/clean')
    original=ValidationClean(clean,'validation',cfg['mean'],cfg['std']);rng=random.Random(2701)
    original.rows=[r for n in range(1,11) for r in rng.sample([r for r in original.rows if len(r['text'])==n],min(100,sum(len(r['text'])==n for r in original.rows)))]
    views['original_bucket']=original
    views['historical_bucket']=ValidationHistorical(Path('data/datasets/recognition/v27_validation/historical'),'validation',cfg['mean'],cfg['std'])
    transfer=json.loads(Path('data/datasets/recognition/v27_validation/transfer.json').read_text())
    for key in ('yolo_provisional','strong_appearance'):views[key]=FrozenInputs(transfer[key],cfg['mean'],cfg['std'])
    probes=[dict(r,text=r['truth'],preferred_font=r['font']) for r in map(json.loads,(PROBE/'labels.jsonl').read_text(encoding='utf-8').splitlines())]
    for view in ('clean_control','texture_color','texture_color_noise'):
        views['application_'+view]=RenderedLines([r for r in probes if r['view']==view],PROBE,cfg,mapping)
    if smoke:
        for name,ds in views.items():
            if name in ('seen','heldout'):
                fonts=sorted({r['preferred_font'] for r in ds.rows})
                ds.rows=[r for f in fonts for r in [x for x in ds.rows if x['preferred_font']==f][:2]]
            else:ds.rows=ds.rows[:32]
    return views


def prepare(train,views,work,args,status):
    rng=random.Random(args.seed);nrng=np.random.default_rng(args.seed)
    paths=json.loads((ROOT/'manifest.json').read_text())['paths']
    fonts={c:[Path(p) for p in paths[c]] for c in ('old','new')}
    cmaps={}
    for paths in fonts.values():
        for path in paths:
            with TTFont(path) as font:cmaps[str(path)]=set(font.getBestCmap())
    # All generated labels are disjoint from all monitored validation transcriptions.
    excluded={r.get('text',r.get('truth')) for ds in views.values() for r in ds.rows}
    excluded.update(json.loads(s)['truth'] for s in (PROBE/'labels.jsonl').read_text(encoding='utf-8').splitlines())
    pools={}
    for c,paths in fonts.items():
        for path in paths:
            for n in range(1,11):
                pools[(str(path),n)]=list(dict.fromkeys(r['text'] for r in train.rows if len(r['text'])==n and r['text'] not in excluded and all(ord(char) in cmaps[str(path)] for char in r['text'])))
    groups={w:[i for i,r in enumerate(train.rows) if row_canvas(r)==w] for w in (128,256,384)}
    # Approximate original natural-bucket proportions; preserve every batch's 16/8/8 ratio.
    counts={w:int(args.steps*len(ids)/len(train)) for w,ids in groups.items()}
    for w in sorted(groups,key=lambda w:args.steps*len(groups[w])/len(train)-counts[w],reverse=True)[:args.steps-sum(counts.values())]:counts[w]+=1
    canvases=[w for w,count in counts.items() for _ in range(count)];rng.shuffle(canvases)
    selected={}
    for w,ids in groups.items():
        needed=16*counts[w];assert needed<=len(ids)
        selected[w]=iter(rng.sample(ids,needed))
    records=[];batches=[];rejected=0
    folder=work/'rendered';folder.mkdir()
    target_lengths=[1,2,3,4,5,6,7,8,9,10]
    length_weights=[.05,.15,.15,.15,.35/3,.35/3,.35/3,.05,.05,.05]
    for step,canvas in enumerate(canvases,1):
        batch=[('original',next(selected[canvas])) for _ in range(16)]
        for j in range(8):
            # Both font cohorts get exactly four pairs per update, with random font within cohort.
            cohort='old' if j<4 else 'new';path=rng.choice(fonts[cohort]);size=rng.randint(24,120)
            for attempt in range(2000):
                n=rng.choices(target_lengths,weights=length_weights)[0];pool=pools[(str(path),n)]
                if not pool:continue
                text=rng.choice(pool);mask=glyph_mask(text,path,size,rng)
                width=round(mask.width*64/mask.height)
                if width<=384 and (128 if width<=128 else 256 if width<=256 else 384)==canvas:break
                rejected+=1
            else:raise ValueError(f'Cannot render matching bucket {canvas} / {path}')
            pair=(step-1)*8+j;texture=('paper','wood','chalkboard','cloth')[pair%4];contrast=('low','medium','high')[pair%3]
            colored,meta=appearance(mask,texture,contrast,rng,nrng)
            alpha=np.asarray(mask,dtype=float)/255
            clean=Image.fromarray(np.rint(242-222*alpha).astype('uint8')).convert('RGB')
            sigma=rng.uniform(1.5,3.5);blur=rng.uniform(.25,.55)
            noisy=np.asarray(colored.filter(ImageFilter.GaussianBlur(blur)),dtype=float)
            noisy=Image.fromarray(np.rint(noisy+nrng.normal(0,sigma,noisy.shape)).clip(0,255).astype('uint8'))
            for view,image in (('clean_control',clean),('texture_color_noise',noisy)):
                padded,valid,actual=scaled_bucket(image);assert actual==canvas
                dest=folder/f'{pair:05d}_{view}.png';padded.save(dest,compress_level=1)
                row=dict(text=text,truth=text,view=view,index=pair,image=str(dest.relative_to(work)),sha256=sha(dest),
                    canvas=canvas,valid_width=valid,font=path.name,preferred_font=path.name,cohort=cohort,
                    native_font_size=size,native_size=list(mask.size),texture=texture,contrast=contrast,
                    noise_sigma=sigma,blur_radius=blur,orientation_flag=0,**meta)
                assert text not in excluded and (valid+3)//4>=minimum_ctc_steps(text)
                batch.append((view,len(records)));records.append(row)
        rng.shuffle(batch);assert Counter(k for k,i in batch)=={'original':16,'clean_control':8,'texture_color_noise':8}
        batches.append(batch)
        if step%25==0 or step==args.steps:
            status(state='rendering',prepared_batches=step,total_batches=args.steps,rendered_images=len(records))
            print('PREPARE',step,'/',args.steps,flush=True)
    assert len(records)==args.steps*16 and len(batches)==args.steps
    audit=dict(updates=args.steps,batch_size=32,presentations=args.steps*32,
        original_presentations=args.steps*16,clean_presentations=args.steps*8,noisy_presentations=args.steps*8,
        original_unique_records=len({i for b in batches for k,i in b if k=='original'}),
        rendered_unique_texts=len({r['text'] for r in records}),generated_validation_text_overlap=0,
        generated_pairs=args.steps*8,canvas_batches=counts,layout_rejections=rejected,
        original_length_counts=dict(Counter(len(train.rows[i]['text']) for b in batches for k,i in b if k=='original')),
        generated_length_counts=dict(Counter(len(r['text']) for r in records if r['view']=='clean_control')),
        original_cohort_counts=dict(Counter(train.rows[i]['cohort'] for b in batches for k,i in b if k=='original')),
        generated_font_counts=dict(Counter(r['font'] for r in records if r['view']=='clean_control')),
        caveat='Generated length mix is conditioned on matching the original natural bucket; sampled with replacement. Original retained split can contain historical text overlap.')
    write_json(work/'sampling.json',audit);write_json(work/'batches.json',batches)
    (work/'labels.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    return records,batches,audit


def selection(report):
    return (score(report)+report['application_clean_control']['0.0']['all']['cer']+report['application_texture_color_noise']['0.0']['all']['cer'])/3


def cpu_copy(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def run(args):
    work=args.output;work.mkdir(parents=True,exist_ok=False);started=time.time()
    status=lambda **kw:write_json(work/'status.json',dict(elapsed_seconds=time.time()-started,**kw))
    try:
        torch.set_num_threads(4);seed_all(args.seed);device=training_device('cuda:0','RTX 3070')
        source_sha=sha(SOURCE);saved=torch.load(SOURCE,map_location='cpu',weights_only=False);cfg=saved['config'];mapping=saved['vocab']['char_to_id']
        assert sha(ROOT/'labels.jsonl')==json.loads((ROOT/'manifest.json').read_text())['labels_sha256']
        train=FontLines(ROOT,'train',cfg['mean'],cfg['std']);assert mapping==train.mapping
        views=validations(cfg,mapping,args.smoke)
        config=dict(source=str(SOURCE),source_sha256=source_sha,steps=args.steps,seed=args.seed,batch=32,
            mixture=[.5,.25,.25],encoder_lr=5e-7,head_lr=3e-6,schedule='constant',optimizer='fresh AdamW; norm/bias no weight decay',
            recipe='Original clean canonical V27 FontLines, no new morphology/jitter; application native-size/margin rendering, RGB textured color and mild native Gaussian blur/noise',
            generated_orientation='horizontal only; originals retain orientation flags',
            buckets=[128,256,384],valid_lengths='ceil(real crop width/4), excluding artificial edge fill',
            selection='equal clean-font composite / application clean CER / application noisy CER; beta0; +1pp original V27 retention gates',
            no_matched_control=True,smoke=args.smoke,source_labels_sha256=sha(ROOT/'labels.jsonl'),probe_labels_sha256=sha(PROBE/'labels.jsonl'))
        write_json(work/'run_config.json',config)
        with zipfile.ZipFile(work/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in list(Path('src/recognition').glob('*.py'))+[Path(__file__),Path('outputs/recognition/application_pilot.py')]:z.write(p,p.as_posix())
        status(state='preparing');records,batches,audit=prepare(train,views,work,args,status)
        mixture=Mixture(train,RenderedLines(records,work,cfg,mapping))
        model=MenuRecognizer(len(mapping)+1,use_bilstm=True).to(device);model.load_state_dict(saved['model']);model.vocabulary=mapping
        optimizer=pilot_optimizer(model,5e-7,3e-6);criterion=nn.CTCLoss(blank=0,zero_infinity=False)
        def validation():return {name:evaluate(model,ds,device) for name,ds in views.items()}
        status(state='initial_validation');initial=validation();write_json(work/'initial.json',initial)
        baseline=retention_values(initial);best_score=selection(initial);best_step=0;history=[]
        def snapshot(step):return dict(model=cpu_copy(model.state_dict()),optimizer=cpu_copy(optimizer.state_dict()),
            vocab=saved['vocab'],config=dict(cfg,application_mixture_pilot=config),updates=step,source_epoch=20,
            best_step=best_step,best_cer=best_score,baseline_retention=baseline,
            rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all())
        save_atomic(work/'best.pt',snapshot(0));write_json(work/'best_validation.json',dict(step=0,score=best_score,validation=initial))
        history.append(dict(step=0,score=best_score,retention=gate(initial,baseline),validation=compact(initial)))
        loss_sum=0.;train_started=time.time()
        for step,batch in enumerate(DataLoader(mixture,batch_sampler=batches,num_workers=0,collate_fn=collate),1):
            assert len(batch['texts'])==32
            model.train();optimizer.zero_grad(set_to_none=True)
            loss=loss_for(model,batch,device,criterion);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();loss_sum+=float(loss.detach())
            if step%50==0:status(state='training',update=step,total_updates=args.steps,mean_loss=loss_sum/step,best_step=best_step)
            if step%500==0 or step==args.steps:
                status(state='validation',update=step,total_updates=args.steps)
                report=validation();value=selection(report);retention=gate(report,baseline)
                if retention['eligible'] and value<best_score:
                    best_score=value;best_step=step;save_atomic(work/'best.pt',snapshot(step))
                    write_json(work/'best_validation.json',dict(step=step,score=value,retention=retention,validation=report))
                entry=dict(step=step,score=value,retention=retention,mean_loss=loss_sum/step,validation=compact(report),best_step=best_step)
                write_json(work/f'step_{step:04d}.json',dict(entry,validation=report));history.append(entry)
                with (work/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
                save_atomic(work/f'step_{step:04d}.pt',snapshot(step));save_atomic(work/'last.pt',snapshot(step))
                print('EVAL',step,'score',value,'retention',retention['eligible'],'best',best_step,flush=True)
        assert sha(SOURCE)==source_sha
        write_json(work/'report.json',dict(history=history,best_step=best_step,best_score=best_score,sampling=audit,
            source_checkpoint_unchanged=True,updates=args.steps,elapsed_seconds=time.time()-started,training_and_eval_seconds=time.time()-train_started))
        lines=['# V29 application-render mixture pilot','', 'V27 epoch20 continuation; fresh AdamW; encoder5e-7/head3e-6, constant; normal training BatchNorm. Each32-image batch:16 original,8 clean application renders,8 textured/color mild-noise renders. No additional morphology. Original images unchanged.',
            '', '| Step | Seen CER | Held-out CER | Application clean CER | Noisy CER | Original CER | Historical CER | YOLO CER | Strong CER | Retention |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        keys=('seen','heldout','application_clean_control','application_texture_color_noise','original_bucket','historical_bucket','yolo_provisional','strong_appearance')
        for entry in history:
            vals=[f"{entry['validation'][key]['0.0']['all']['cer']:.2%}" for key in keys]
            lines.append('| '+str(entry['step'])+' | '+' | '.join(vals)+' | '+('pass' if entry['retention']['eligible'] else ', '.join(entry['retention']['failed']))+' |')
        lines.extend(['',f'Best eligible checkpoint: update{best_step}; step0 remains eligible. Selection equally weights clean-font composite, application clean CER, application noisy CER. Retention caps are starting V27 +1 absolute percentage point on four historical metrics.',
            '', 'The generated training transcriptions exclude monitored validation text. Original training records retain the existing historical split. The application probe was designed and previously examined before this experiment; it is a development set, not independent final evidence. No matched original-only continuation arm was run. YOLO labels are provisional. All decoding is raw greedy CTC, primary blank bias0; beta0.2 is secondary in JSON.',
            '', 'See run_config.json, sampling.json, metrics.jsonl, report.json and step checkpoints for full audit.'])
        (work/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        status(state='completed',update=args.steps,total_updates=args.steps,best_step=best_step,source_checkpoint_unchanged=True)
        print('COMPLETE',work,'best step',best_step,flush=True)
    except Exception:
        status(state='failed',error=traceback.format_exc());raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--seed',type=int,default=20261201);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();assert args.steps>0;run(args)


if __name__=='__main__':main()
