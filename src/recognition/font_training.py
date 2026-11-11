"""V27: 20 full passes, production buckets, margin variation and retention gates."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time
import zipfile

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import align_errors,summarise,decode_with_blank_penalty
from .utils import fit_margins,seed_all,sha,loss_for,training_device,write_json
from .checkpointing import cpu_snapshot,save_atomic
from .data import collate,preprocess
from .datasets import FontLines,Minimum128,GuardedLines,canvas_for
from .model import MenuRecognizer
from .optimization import pilot_optimizer


def row_canvas(row):
    guard=4 if len(row['text'])==1 else 16
    return canvas_for(min(384,row['glyph_box'][2]-row['glyph_box'][0]+2*guard))


def epoch_plan(rows,epoch,seed):
    """Exactly once per record; never promote a remainder to another bucket."""
    rng=random.Random(seed+epoch*100003);groups={w:[] for w in (128,256,384)}
    augmented=set()
    for op in ('dilate','erode'):
        pool=[i for i,r in enumerate(rows) if r['morphology_actual']==op]
        count=round(len(rows)*.025)
        if len(pool)<count:raise ValueError('Insufficient stored morphology variants')
        augmented.update(rng.sample(pool,count))
    jitter=set(rng.sample(range(len(rows)),len(rows)//2))
    for i,row in enumerate(rows):groups[row_canvas(row)].append((i,i in augmented,i in jitter,rng.randrange(2**32)))
    batches=[]
    for canvas,keys in groups.items():
        rng.shuffle(keys)
        for start in range(0,len(keys),32):batches.append([(*k,canvas) for k in keys[start:start+32]])
    rng.shuffle(batches)
    ids=[k[0] for b in batches for k in b]
    if Counter(ids)!=Counter(range(len(rows))):raise ValueError('Epoch is not a full pass')
    audit=dict(epoch=epoch,records=len(rows),presentations=len(ids),unique_records=len(set(ids)),oversampling=0,
               updates=len(batches),morphology_requested=len(augmented),jitter_views=len(jitter),
               cohorts=dict(Counter(r['cohort'] for r in rows)),
               canvas_counts={str(w):len(keys) for w,keys in groups.items()},
               plan_sha256=hashlib.sha256(json.dumps(batches).encode()).hexdigest())
    return batches,audit


class TrainingLines(FontLines):
    def __getitem__(self,key):
        i,aug,jitter,seed,canvas=key;row=self.rows[i]
        relative=row['aug_image'] if aug else row['image'];box=row['aug_box'] if aug else row['glyph_box']
        check=(i,aug)
        if check not in self.verified:
            if sha(self.root/relative)!=(row['aug_sha256'] if aug else row['sha256']):raise ValueError('Training mask changed')
            self.verified.add(check)
        with Image.open(self.root/relative) as im:alpha=np.asarray(im)[:,box[0]:box[2]].copy()
        span=alpha.shape[1];guard=4 if len(row['text'])==1 else 16
        head,tail,capped=fit_margins(span,guard,guard,canvas)
        if min(head,tail)<1:raise ValueError('Cannot fit safe guards without clipping')
        if jitter:head=random.Random(seed).randint(min(4,head),canvas-span-tail)
        valid=head+span+tail;bg=row['background'];ink=20 if bg==242 else 235
        glyph=np.rint(bg+(ink-bg)*alpha.astype(np.float32)/255).clip(0,255).astype(np.uint8)
        image=Image.new('L',(canvas,64),bg);image.paste(Image.fromarray(glyph),(head,0))
        tensor,_,compressed=preprocess(image.convert('RGB'),self.mean,self.std,'batch')
        if compressed:raise ValueError('Glyph resizing is prohibited')
        meta=dict(index=i,augmented=aug,jittered=jitter,head=head,tail=tail,canvas=canvas,guards_capped=capped)
        return tensor,torch.tensor([self.mapping[c] for c in row['text']]),row['orientation_flag'],valid,row['text'],False,meta


def training_collate(samples):
    batch=collate(samples);batch['audit']=[s[6] for s in samples]
    if len({s[0].shape[-1] for s in samples})!=1:raise ValueError('Mixed physical widths in batch')
    return batch


class ValidationFonts(FontLines):
    def canvas_at(self,i):return row_canvas(self.rows[i])


class ValidationClean(Minimum128):
    def canvas_at(self,i):return row_canvas(self.rows[i])


class ValidationHistorical(GuardedLines):
    def canvas_at(self,i):return canvas_for(self.widths[i])
    def __getitem__(self,i):
        sample=super().__getitem__(i)
        return (sample[0][...,:self.canvas_at(i)],*sample[1:])


class FrozenInputs:
    def __init__(self,rows,mean,std):
        self.rows=[dict(r,text=r['truth']) for r in rows];self.mean=mean;self.std=std;self.verified={}
    def __len__(self):return len(self.rows)
    def canvas_at(self,i):return self.rows[i]['canvas']
    def __getitem__(self,i):
        row=self.rows[i];path=Path(row['image']);digest=sha(path)
        if i in self.verified and self.verified[i]!=digest:raise ValueError('Frozen validation input changed')
        self.verified[i]=digest
        with Image.open(path) as im:tensor,_,compressed=preprocess(im.convert('RGB'),self.mean,self.std,'batch')
        if compressed or tensor.shape[-1]!=row['canvas']:raise ValueError('Invalid frozen bucket')
        return tensor,torch.empty(0,dtype=torch.long),row['orientation_flag'],row['original_width'],row['text'],False


def evaluate(model,ds,device):
    groups={w:[] for w in (128,256,384)};model.eval();by_beta={str(b):{} for b in (0.,.2)}
    for i in range(len(ds)):groups[ds.canvas_at(i)].append(i)
    with torch.inference_mode():
        for canvas,indices in groups.items():
            for start in range(0,len(indices),32):
                ids=indices[start:start+32];batch=collate([ds[i] for i in ids],padded_width=canvas)
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(batch['images'].to(device),batch['orientation'].to(device),batch['input_lengths'])
                for beta in (0.,.2):
                    for i,truth,pred in zip(ids,batch['texts'],decode_with_blank_penalty(logits,ds.mapping if hasattr(ds,'mapping') else model.vocabulary,batch['input_lengths'],beta)):
                        errors=Counter(op['type'] for op in align_errors(truth,pred) if op['type']!='match')
                        row=ds.rows[i];by_beta[str(beta)][i]=dict(index=i,truth=truth,prediction=pred,font=row.get('preferred_font'),cohort=row.get('cohort'),
                            error_counts={k:errors[k] for k in ('deletion','insertion','substitution')})
    result={}
    for beta,values in by_beta.items():
        ordered=[values[i] for i in range(len(ds))];fonts=sorted({r['font'] for r in ordered if r['font']})
        by_font={f:summarise([r for r in ordered if r['font']==f]) for f in fonts}
        cohorts={c:[v['cer'] for f,v in by_font.items() if any(r['font']==f and r['cohort']==c for r in ordered)] for c in ('old','new')}
        result[beta]=dict(all=summarise(ordered),by_length={str(n):summarise([r for r in ordered if len(r['truth'])==n]) for n in sorted({len(r['truth']) for r in ordered})},
            by_font=by_font,cohort_macro={c:sum(a)/len(a) for c,a in cohorts.items() if a},
            macro_font_cer=sum(v['cer'] for v in by_font.values())/len(by_font) if by_font else None,predictions=ordered)
    return result


def score(report):
    seen=report['seen']['0.0']['cohort_macro']
    return (seen['old']+seen['new']+report['heldout']['0.0']['macro_font_cer'])/3


def retention_values(report):
    return dict(original_fonts=report['seen']['0.0']['cohort_macro']['old'],
        original_bucket=report['original_bucket']['0.0']['all']['cer'],
        historical_bucket=report['historical_bucket']['0.0']['all']['cer'],
        yolo_provisional=report['yolo_provisional']['0.0']['all']['cer'])


def gate(report,baseline,tolerance=.01):
    current=retention_values(report);caps={k:v+tolerance for k,v in baseline.items()}
    return dict(eligible=all(current[k]<=caps[k]+1e-12 for k in current),current=current,caps=caps,
                failed=[k for k in current if current[k]>caps[k]+1e-12])


def lr_factor(epoch,epochs):
    if epoch<=2:return 1.
    progress=min(1.,(epoch-2)/max(1,epochs-2))
    return .1+.9*.5*(1+math.cos(math.pi*progress))


def compact(report):return {name:{b:{k:v for k,v in value.items() if k!='predictions'} for b,value in result.items()} for name,result in report.items()}


def run(args):
    work=args.output;work.mkdir(parents=True,exist_ok=True);status=lambda **kw:write_json(work/'status.json',kw)
    try:
        torch.set_num_threads(4);seed_all(args.seed);device=training_device('cuda:0','RTX 3070')
        source=torch.load(args.checkpoint,map_location='cpu',weights_only=False);cfg=source['config'];source_sha=sha(args.checkpoint)
        root=Path('data/datasets/recognition/v27_fonts');manifest=json.loads((root/'manifest.json').read_text())
        if sha(root/'labels.jsonl')!=manifest['labels_sha256']:raise ValueError('Font dataset changed')
        train=TrainingLines(root,'train',cfg['mean'],cfg['std'])
        if train.mapping!=source['vocab']['char_to_id']:raise ValueError('Vocabulary mismatch')
        if args.smoke:
            rng=random.Random(27)
            train.rows=[r for n in range(1,11) for cohort in ('old','new') for r in rng.sample([r for r in train.rows if len(r['text'])==n and r['cohort']==cohort],32)]
        if Counter(r['cohort'] for r in train.rows)['old']*2!=len(train):raise ValueError('Font cohort balance is not50/50')
        views={s:ValidationFonts(root,s,cfg['mean'],cfg['std']) for s in ('seen','heldout')}
        original=ValidationClean(Path('data/datasets/recognition/v27_validation/clean'),'validation',cfg['mean'],cfg['std']);rng=random.Random(2701)
        original.rows=[r for n in range(1,11) for r in rng.sample([r for r in original.rows if len(r['text'])==n],min(100,sum(len(r['text'])==n for r in original.rows)))]
        views['original_bucket']=original
        legacy=Path('data/datasets/recognition/v27_validation/historical');views['historical_bucket']=ValidationHistorical(legacy,'validation',cfg['mean'],cfg['std'],None)
        replay=Path('data/datasets/recognition/v27_validation/transfer.json');transfer=json.loads(replay.read_text())
        cases=transfer
        views['yolo_provisional']=FrozenInputs(cases['yolo_provisional'],cfg['mean'],cfg['std'])
        strong=cases['strong_appearance'];texts=sorted({r['truth'] for r in strong});selected=set(random.Random(2702).sample(texts,len(texts)//2))
        views['strong_monitor']=FrozenInputs([r for r in strong if r['truth'] in selected],cfg['mean'],cfg['std'])
        reserved=FrozenInputs([r for r in strong if r['truth'] not in selected],cfg['mean'],cfg['std'])
        code_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
        config=dict(source=str(args.checkpoint),source_sha256=source_sha,epochs=args.epochs,seed=args.seed,smoke=args.smoke,
            dataset_labels_sha256=manifest['labels_sha256'],training_records=len(train),row_ids_sha256=hashlib.sha256(json.dumps([r['image'] for r in train.rows]).encode()).hexdigest(),
            code_hashes=code_hashes,batch=32,full_pass='every labelled training record exactly once per epoch; no oversampling or dropped tail',
            fonts=15,font_cohorts=[.5,.5],buckets=[128,256,384],separate_fixed_width_stream=False,
            morphology_fraction=.05,dilation_fraction=.025,erosion_fraction=.025,morphology_strength=.25,
            margin_variation_fraction=.5,margin_policy='within the assigned nearest bucket; canonical guards or random left offset retaining right guard',
            initial_encoder_lr=5e-6,initial_head_lr=3e-5,lr_schedule='2 epochs constant, cosine to10% by final epoch',
            selection='equal original/new/heldout-font macro CER at beta0; retention gates relative to starting V26',
            retention_tolerance_absolute=.01,retention_sets=['original_fonts','original_bucket','historical_bucket','yolo_provisional'],
            strong_monitor_ids=[r['id'] for r in views['strong_monitor'].rows],reserved_ids=[r['id'] for r in reserved.rows],
            strong_reserved_scope='reserved from this run monitoring/selection; images were used in prior diagnostics',
            source_replay_sha256=sha(replay))
        config=json.loads(json.dumps(config))
        if (work/'run_config.json').exists() and json.loads((work/'run_config.json').read_text())!=config:raise ValueError('Resume configuration/code differs')
        write_json(work/'run_config.json',config)
        if not (work/'source.zip').exists():
            with zipfile.ZipFile(work/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in Path(__file__).parent.glob('*.py'):z.write(p,'src/recognition/'+p.name)
        model=MenuRecognizer(len(train.mapping)+1,use_bilstm=True).to(device);model.load_state_dict(source['model']);model.vocabulary=train.mapping
        optimizer=pilot_optimizer(model,5e-6,3e-5);criterion=nn.CTCLoss(blank=0,zero_infinity=False)
        def validation():return {name:evaluate(model,ds,device) for name,ds in views.items()}
        endpoint_cfg=dict(cfg,long_font_training=config)
        start=0;updates=0;best_score=float('inf');best_state=None;baseline=None
        def snapshot(epoch):return dict(model=cpu_snapshot(model.state_dict()),optimizer=cpu_snapshot(optimizer.state_dict()),vocab=source['vocab'],config=endpoint_cfg,
            epoch=epoch,updates=updates,best_cer=best_score,baseline_retention=baseline,
            rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all())
        if (work/'last.pt').exists():
            saved=torch.load(work/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
            start=saved['epoch'];updates=saved['updates'];best_score=saved['best_cer'];best_state=saved['best_checkpoint'];baseline=saved['baseline_retention'];save_atomic(work/'best.pt',best_state)
            if best_state['epoch']==0:
                initial=json.loads((work/'initial.json').read_text(encoding='utf-8'))
                write_json(work/'best_validation.json',dict(epoch=0,score=best_score,validation=initial,retention=gate(initial,baseline)))
            else:write_json(work/'best_validation.json',json.loads((work/f"epoch_{best_state['epoch']:02d}.json").read_text(encoding='utf-8')))
            random.setstate(saved['rng_python']);np.random.set_state(saved['rng_numpy']);torch.set_rng_state(saved['rng_torch']);torch.cuda.set_rng_state_all(saved['rng_cuda'])
        else:
            status(state='initial_validation');initial=validation();write_json(work/'initial.json',initial)
            baseline=retention_values(initial);best_score=score(initial);best_state=snapshot(0)
            save_atomic(work/'best.pt',best_state);save_atomic(work/'last.pt',dict(best_state,best_checkpoint=best_state))
            write_json(work/'best_validation.json',dict(epoch=0,score=best_score,validation=initial,retention=gate(initial,baseline)))
        started=time.time()
        for epoch in range(start+1,args.epochs+1):
            batches,audit=epoch_plan(train.rows,epoch,args.seed);write_json(work/f'epoch_{epoch:02d}_sampling.json',audit)
            factor=lr_factor(epoch,args.epochs)
            for group in optimizer.param_groups:group['lr']=(5e-6 if group['role']=='encoder' else 3e-5)*factor
            loss_sum=0.;seen_ids=set();seen_count=0;morph_count=0;jitter_count=0
            for step,batch in enumerate(DataLoader(train,batch_sampler=batches,num_workers=0,collate_fn=training_collate),1):
                ids=[r['index'] for r in batch['audit']]
                if seen_ids.intersection(ids) or len(set(ids))!=len(ids):raise ValueError('Duplicate record during full pass')
                seen_ids.update(ids);seen_count+=len(ids);morph_count+=sum(r['augmented'] for r in batch['audit']);jitter_count+=sum(r['jittered'] for r in batch['audit'])
                model.train();optimizer.zero_grad(set_to_none=True);loss=loss_for(model,batch,device,criterion);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
                updates+=1;loss_sum+=float(loss.detach())*len(ids)
                if step%100==0:
                    status(state='training',epoch=epoch,epochs=args.epochs,step=step,steps_per_epoch=len(batches),records_seen=seen_count,total_records=len(train),updates=updates)
                    print('TRAIN',epoch,step,'/',len(batches),'loss',loss_sum/seen_count,flush=True)
            if seen_ids!=set(range(len(train))) or seen_count!=len(train) or morph_count!=audit['morphology_requested']:raise ValueError('Incomplete epoch coverage')
            status(state='validation',epoch=epoch,epochs=args.epochs);val=validation();value=score(val);gates=gate(val,baseline)
            improved=gates['eligible'] and value<best_score
            if improved:best_score=value
            report=dict(epoch=epoch,updates=updates,score=value,retention=gates,validation=val,sampling=audit,
                        training_loss=loss_sum/seen_count,lr_factor=factor,observed_records=seen_count,observed_morphology=morph_count,observed_jitter=jitter_count)
            write_json(work/f'epoch_{epoch:02d}.json',report)
            row={k:v for k,v in report.items() if k!='validation'};row['metrics']=compact(val);row['best_score']=best_score
            metrics_path=work/'metrics.jsonl';history=[json.loads(s) for s in metrics_path.read_text(encoding='utf-8').splitlines()] if metrics_path.exists() else []
            history=[r for r in history if r['epoch']<epoch]+[row];temporary=metrics_path.with_suffix('.tmp');temporary.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in history),encoding='utf-8');temporary.replace(metrics_path)
            snap=snapshot(epoch)
            if improved:best_state=snap;save_atomic(work/'best.pt',best_state);write_json(work/'best_validation.json',report)
            recovery=dict(snap,best_checkpoint=best_state);save_atomic(work/'last.pt',recovery)
            if epoch%5==0:save_atomic(work/f'epoch_{epoch:02d}.pt',recovery)
            print('EPOCH',epoch,'score',value,'eligible',gates['eligible'],'best',best_state['epoch'],best_score,flush=True)
            if args.stop_after_epoch and epoch==args.stop_after_epoch:
                status(state='smoke_checkpoint_boundary',epoch=epoch,updates=updates);return
        final_tests={}
        for name,state in [('baseline',source),('best',best_state),('last',torch.load(work/'last.pt',map_location='cpu',weights_only=False))]:
            model.load_state_dict(state['model']);final_tests[name]=evaluate(model,reserved,device)
        write_json(work/'reserved_strong_final.json',final_tests)
        last=torch.load(work/'last.pt',map_location='cpu',weights_only=False)
        deltas={int(v-source['model'][k]) for k,v in last['model'].items() if k.endswith('num_batches_tracked')}
        if deltas!={updates} or sha(args.checkpoint)!=source_sha:raise ValueError('Final checkpoint audit failed')
        write_json(work/'completion.json',dict(epochs=args.epochs,updates=updates,best_epoch=best_state['epoch'],best_score=best_score,
            unique_records_per_epoch=len(train),presentations=args.epochs*len(train),BN_counter_deltas=list(deltas),elapsed_seconds=time.time()-started,
            source_unchanged=True,best_sha256=sha(work/'best.pt'),last_sha256=sha(work/'last.pt')))
        status(state='completed',epochs=args.epochs,updates=updates,best_epoch=best_state['epoch'],best_score=best_score)
    except Exception as exc:status(state='failed',error=repr(exc));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=20);p.add_argument('--seed',type=int,default=20261127)
    p.add_argument('--checkpoint',type=Path,default=Path('outputs/recognition/training/v27_font_full20/initial.pt'))
    p.add_argument('--output',type=Path,default=Path('outputs/recognition/training/v27_new_run'))
    p.add_argument('--smoke',action='store_true');p.add_argument('--stop-after-epoch',type=int)
    run(p.parse_args())

if __name__=='__main__':main()
