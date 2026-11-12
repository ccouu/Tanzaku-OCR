"""V38: four complete eligible YOLO passes, exactly 25% crop presentations.

V36-selected initialization, fixed V30 BN statistics, trainable affine, EMA.
Labels are VLM-produced, not independently verified human transcriptions.
"""
import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import random
import shutil
import time
import traceback
import zipfile

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader

from outputs.recognition.v36_yolo_fullpasses import Combined, load, dump_rows, factor, WORK as V36
from outputs.recognition.v37_long512 import make_views, evaluate_any
from outputs.recognition.application_mixture_pilot import RenderedLines, cpu_copy, ROOT
from outputs.recognition.tiered_short_pilot import SOURCE, rows
from outputs.recognition.evaluate_bn_swap import V34, gates as legacy_gates
from outputs.recognition.evaluate_fp16_retention import evaluate_fp16
from outputs.recognition.short_retention_pilot import selection
from src.recognition.datasets import FontLines
from src.recognition.font_training import evaluate, compact, row_canvas
from src.recognition.data import preprocess, collate, minimum_ctc_steps
from src.recognition.model import MenuRecognizer
from src.recognition.optimization import pilot_optimizer
from src.recognition.checkpointing import save_atomic
from src.recognition.utils import sha, seed_all, write_json, loss_for, training_device
from src.recognition.retention import apply_retention_policy, POLICY_PATH

INPUT=Path('data/processed/gemini_banana_mobile_menus_50_crops')
WORK=Path('outputs/recognition/training/v38_yolo25_fullpasses4')
START=V36/'best.pt'
HOLDOUT={f'menu-{n:03d}' for n in (10,20,30,40,50,60,70,80,86,100,109,120,130)}
SEED=20261118


def gates(report,baseline):
    return apply_retention_policy(legacy_gates(report,baseline))


class BucketCrops(RenderedLines):
    def __getitem__(self,i):
        r=self.rows[i];path=self.root/r['image']
        if i not in self.verified:
            assert sha(path)==r['sha256'];self.verified.add(i)
        with Image.open(path) as im:
            tensor,w,compressed=preprocess(im,self.cfg['mean'],self.cfg['std'],'batch',max_width=r['canvas'])
        assert not compressed and w==r['canvas']
        return tensor,torch.tensor([self.mapping[c] for c in r['text']]),r['orientation_flag'],r['valid_width'],r['text'],False


def prepare(work,cfg,mapping):
    folder=work/'yolo';folder.mkdir(exist_ok=True);(folder/'images').mkdir(exist_ok=True)
    delivered=rows(INPUT/'labeled_crops.jsonl');manifest=load(INPUT/'crops_manifest.json')
    reference={r['crop_id']:r for r in manifest['crops']}
    assert len(delivered)==len(reference)==1894 and len({r['crop_id'] for r in delivered})==len(delivered)
    exclusions=[];eligible=[]
    for r in delivered:
        reason=None
        if r['status']!='OK':reason='status_'+r['status']
        elif not r['text']:reason='empty'
        elif any(c not in mapping for c in r['text']):reason='out_of_vocabulary'
        elif (r['valid_width']+3)//4<minimum_ctc_steps(r['text']):reason='ctc_infeasible'
        if reason:
            exclusions.append(dict(crop_id=r['crop_id'],text=r['text'],reason=reason));continue
        ref=reference[r['crop_id']]
        for key in ('ctc_crop','ctc_sha256','canvas','valid_width','orientation_flag','source_menu'):
            assert r[key]==ref[key],(r['crop_id'],key)
        assert r['orientation_flag'] in (0,1) and r['canvas'] in (128,256,384,512)
        assert 0<r['valid_width']<=r['canvas']
        source=INPUT/r['ctc_crop'];assert source.resolve().is_relative_to(INPUT.resolve())
        assert sha(source)==r['ctc_sha256']
        with Image.open(source) as image:assert image.size==(r['canvas'],64) and image.mode=='RGB'
        eligible.append(dict(r,split='val_holdout' if r['source_menu'] in HOLDOUT else 'train'))
    held_hashes={r['ctc_sha256'] for r in eligible if r['split']=='val_holdout'}
    kept=[]
    for r in eligible:
        if r['split']=='train' and r['ctc_sha256'] in held_hashes:
            exclusions.append(dict(crop_id=r['crop_id'],text=r['text'],reason='holdout_pixel_duplicate'));continue
        dest=folder/'images'/(r['crop_id']+'.png')
        if dest.exists():assert sha(dest)==r['ctc_sha256']
        else:shutil.copy2(INPUT/r['ctc_crop'],dest)
        kept.append(dict(r,image=str(dest.relative_to(folder)),sha256=r['ctc_sha256']))
    train=[r for r in kept if r['split']=='train'];hold=[r for r in kept if r['split']=='val_holdout']
    assert train and hold and not {r['source_menu'] for r in train}&{r['source_menu'] for r in hold}
    dump_rows(folder/'labels.jsonl',kept);dump_rows(folder/'excluded.jsonl',exclusions)
    original=FontLines(ROOT,'train',cfg['mean'],cfg['std']);generated=rows(V34/'corpus/labels.jsonl')
    corpus=load(V34/'corpus/manifest.json')
    assert sha(ROOT/'labels.jsonl')==corpus['original_labels_sha256']
    assert sha(V34/'corpus/labels.jsonl')==corpus['labels_sha256']
    required=corpus['required_terms'];assert len(required)==521
    protected_o=set();protected_p=set();pairs=defaultdict(list)
    for i,r in enumerate(generated):pairs[r['index']].append(i)
    assert all(len(v)==2 for v in pairs.values())
    for term in required:
        hits=[i for i,r in enumerate(original.rows) if r['text']==term]
        if hits:protected_o.add(hits[0])
        else:protected_p.add(next(r['index'] for r in generated if r['text']==term))
    n=len(train);synthetic=3*n
    pair_count=synthetic//4;orig_count=synthetic-2*pair_count
    assert len(protected_o)<=orig_count and len(protected_p)<=pair_count
    mix=Combined(original,RenderedLines(generated,V34/'corpus',cfg,mapping),BucketCrops(train,folder,cfg,mapping))
    plans=[];audits=[]
    for epoch in range(1,5):
        rng=random.Random(SEED+epoch*100003)
        optional_o=[i for i in range(len(original)) if i not in protected_o]
        optional_p=[i for i in pairs if i not in protected_p]
        selected_o=sorted(protected_o|set(rng.sample(optional_o,orig_count-len(protected_o))))
        selected_p=sorted(protected_p|set(rng.sample(optional_p,pair_count-len(protected_p))))
        keys=[('original',i) for i in selected_o]+[('generated',i) for p in selected_p for i in pairs[p]]+[('yolo',i) for i in range(n)]
        assert len(keys)==4*n and len(set(keys))==len(keys)
        groups=defaultdict(list)
        for k,i in keys:
            row=mix.datasets[k].rows[i];width=row_canvas(row) if k=='original' else row['canvas']
            # Homogeneous domain AND physical width; no synthetic stretching to512.
            groups[(k,width)].append((k,i))
        plan=[]
        for group,items in sorted(groups.items()):
            rng.shuffle(items);plan.extend(items[j:j+32] for j in range(0,len(items),32))
        rng.shuffle(plan)
        assert Counter(tuple(x) for b in plan for x in b)==Counter(keys)
        terms=Counter(mix.datasets[k].rows[i]['text'] for k,i in keys)
        assert all(terms[t]>=1 for t in required)
        write_json(work/f'epoch_{epoch:02d}_plan.json',plan)
        audits.append(dict(epoch=epoch,updates=len(plan),records=len(keys),yolo=n,original=orig_count,
            application_clean=pair_count,application_noisy=pair_count,exact_term_counts={t:terms[t] for t in required},
            canvas_presentations={str(w):sum(len(v) for (k,c),v in groups.items() if c==w) for w in (128,256,384,512)},
            plan_sha256=sha(work/f'epoch_{epoch:02d}_plan.json')))
        plans.append(plan)
    _,views=make_views(cfg,mapping)
    views['yolo_current_train']=mix.datasets['yolo']
    views['yolo_current_holdout']=BucketCrops(hold,folder,cfg,mapping)
    sampling=dict(input=str(INPUT),input_labels_sha256=sha(INPUT/'labeled_crops.jsonl'),input_manifest_sha256=sha(INPUT/'crops_manifest.json'),
        delivered=len(delivered),eligible=len(kept),training_crops=n,holdout_crops=len(hold),epochs=4,
        total_presentations=16*n,yolo_presentations=4*n,yolo_fraction=.25,updates=sum(map(len,plans)),
        training_menus=sorted({r['source_menu'] for r in train}),holdout_menus=sorted({r['source_menu'] for r in hold}),
        exclusions=dict(Counter(r['reason'] for r in exclusions)),overlap_flagged_training_crops=sum(r.get('review_required',False) for r in train),
        plans=audits,labels_policy='Exact VLM OK/in-vocab labels; no whitespace stripping, punctuation replacement or human-verification claim. Unresolved overlap flags retained; OK is trusted provisionally.',
        mixture='25%YOLO; remaining75%approximately50/25/25 original/application_clean/application_noisy. Four full YOLO passes, NOT four full passes of the 128000 synthetic corpus.',
        bucket_policy='Delivered RGB H64 128/256/384/512 crops unchanged; same-width/domain batches; incomplete batches kept. No additional augmentation.',
        holdout_pixel_hash_overlap=0)
    write_json(work/'sampling.json',sampling)
    return mix,plans,views,sampling


def run(args):
    work=args.output;work.mkdir(parents=True,exist_ok=args.resume or args.prepare_only)
    started=time.time()
    def status(**kw):write_json(work/'status.json',dict(elapsed_seconds=time.time()-started,**kw))
    try:
        torch.set_num_threads(4);seed_all(SEED)
        source=torch.load(START,map_location='cpu',weights_only=False);cfg=source['config'];mapping=source['vocab']['char_to_id'];source_sha=sha(START)
        status(state='preparing');mix,plans,views,sampling=prepare(work,cfg,mapping);total=sum(map(len,plans))
        config=dict(source=str(START),source_sha256=source_sha,total_updates=total,epochs=4,seed=SEED,
            encoder_lr=5e-7,head_lr=3e-6,warmup_updates=200,min_lr_factor=.1,ema_decay=.998,
            effective_batch=32,training_microbatch=4,gradient_accumulation='sample-count-weighted mean over each planned batch; partial batches retained',
            bn_running_stats_frozen=True,bn_affine_trainable=True,decoding='greedy beta0',
            retention_policy=load(POLICY_PATH)['id'],
            selection='0.7 current YOLO holdout CER +0.3 existing application/short composite; all17 retention gates in FP16/BF16; initial V36 fallback',
            sampling_sha256=sha(work/'sampling.json'))
        if (work/'run_config.json').exists():assert load(work/'run_config.json')==config,'Resume config changed'
        write_json(work/'run_config.json',config)
        if args.prepare_only:status(state='prepared',total_updates=total);return
        device=training_device('cuda:0','RTX 3070')
        model=MenuRecognizer(len(mapping)+1,use_bilstm=True).to(device);model.vocabulary=mapping;model.load_state_dict(source['model'])
        model.recurrent.lstm.flatten_parameters();ema=copy.deepcopy(model).eval();ema.requires_grad_(False);ema.recurrent.lstm.flatten_parameters()
        optimizer=pilot_optimizer(model,5e-7,3e-6);criterion=nn.CTCLoss(blank=0,zero_infinity=False)
        v30=torch.load(SOURCE,map_location='cpu',weights_only=False)['model']
        buffers={k:v for k,v in v30.items() if k.endswith(('running_mean','running_var','num_batches_tracked'))};assert len(buffers)==147
        bn=[m for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)]
        def check_bn():
            for candidate in (model,ema):assert all(torch.equal(candidate.state_dict()[k].cpu(),v) for k,v in buffers.items())
        check_bn()
        baselines={'fp16':load('outputs/recognition/evaluation/v35_paired_short_inspection/fp16_full_source.json'),'bf16':load(V34/'initial.json')}
        def validation(candidate,precision):
            result={}
            for name,ds in views.items():
                fn=evaluate_fp16 if precision=='fp16' else evaluate
                result[name]=evaluate_any(candidate,ds,device,precision) if name.startswith('yolo_current_') else fn(candidate,ds,device)
            return result
        def score(result):return .7*result['yolo_current_holdout']['0.0']['all']['cer']+.3*selection(result)
        step=0;best_step=0;best_arm='initial_V36';best_score=None;history=[];epoch_audits=[]
        def snapshot(epoch,position,candidate=None,arm='raw'):
            return dict(model=cpu_copy((candidate or model).state_dict()),raw_model=cpu_copy(model.state_dict()),ema_model=cpu_copy(ema.state_dict()),
                optimizer=cpu_copy(optimizer.state_dict()),vocab=source['vocab'],config=dict(cfg,v38_training=config),updates=step,epoch=epoch,batch_in_epoch=position,
                best_update=best_step,best_arm=best_arm,best_score=best_score,checkpoint_arm=arm,
                rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all())
        with zipfile.ZipFile(work/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in [Path(__file__),*Path('src/recognition').glob('*.py')]:z.write(p,p.as_posix())
        if args.resume and (work/'last.pt').exists():
            saved=torch.load(work/'last.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(saved['raw_model']);ema.load_state_dict(saved['ema_model']);optimizer.load_state_dict(saved['optimizer'])
            step=saved['updates'];best_step=saved['best_update'];best_arm=saved['best_arm'];best_score=saved['best_score']
            random.setstate(saved['rng_python']);np.random.set_state(saved['rng_numpy']);torch.set_rng_state(saved['rng_torch']);torch.cuda.set_rng_state_all(saved['rng_cuda'])
            history=load(work/'history.json');epoch_audits=load(work/'epoch_audits.json') if (work/'epoch_audits.json').exists() else []
        else:
            initial={};parity={}
            for precision in ('fp16','bf16'):
                status(state='initial_evaluation',precision=precision);initial[precision]=validation(model,precision)
                expected=load(V36/f'step_07000_raw_{precision}.json')['validation']
                parity[precision]={k:sum(a['prediction']!=b['prediction'] for a,b in zip(initial[precision][k]['0.0']['predictions'],expected[k]['0.0']['predictions'])) for k in expected}
                assert all(len(initial[precision][k]['0.0']['predictions'])==len(expected[k]['0.0']['predictions']) for k in expected)
                assert not any(parity[precision].values()) and gates(initial[precision],baselines[precision])['eligible']
                write_json(work/f'initial_{precision}.json',initial[precision])
            write_json(work/'initial_parity.json',parity);best_score=score(initial['fp16']);save_atomic(work/'best.pt',snapshot(0,0,model,best_arm))
        epoch_offset=0
        for epoch,plan in enumerate(plans,1):
            end=epoch_offset+len(plan)
            if step>=end:epoch_offset=end;continue
            offset=max(0,step-epoch_offset);encountered=Counter(tuple(k) for b in plan[:offset] for k in b);loss_sum=0.;loss_count=0
            loader=DataLoader(mix,batch_sampler=plan[offset:],num_workers=0,collate_fn=collate,generator=torch.Generator().manual_seed(SEED+epoch))
            for position,batch in enumerate(loader,offset+1):
                f=factor(step+1,total)
                for group in optimizer.param_groups:group['lr']=(5e-7 if group['role']=='encoder' else 3e-6)*f
                model.train()
                for layer in bn:layer.eval()
                optimizer.zero_grad(set_to_none=True)
                # Bound activation memory while preserving the planned update's
                # sample-weighted mean CTC loss. BN statistics are frozen.
                keys=plan[position-1];batch_loss=0.;target_offset=0
                for start in range(0,len(keys),4):
                    micro={k:v[start:start+4] for k,v in batch.items() if k!='targets'}
                    count=int(micro['lengths'].sum())
                    micro['targets']=batch['targets'][target_offset:target_offset+count];target_offset+=count
                    loss=loss_for(model,micro,device,criterion)
                    weight=len(micro['texts'])/len(keys)
                    (loss*weight).backward();batch_loss+=float(loss.detach())*weight
                assert all(m.weight.requires_grad and m.bias.requires_grad and m.weight.grad is not None and m.bias.grad is not None for m in bn)
                nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();step+=1
                with torch.no_grad():
                    for p,q in zip(ema.parameters(),model.parameters()):p.lerp_(q,.002)
                encountered.update(tuple(k) for k in plan[position-1]);loss_sum+=batch_loss;loss_count+=1
                if position==1 or step%50==0:
                    check_bn();status(state='training',epoch=epoch,update=step,total_updates=total,batch=position,batches_per_epoch=len(plan),mean_loss=loss_sum/loss_count,best_update=best_step)
                if position==len(plan):
                    entry=dict(epoch=epoch,update=step,lr_factor=f,mean_training_loss=loss_sum/loss_count,arms={})
                    for arm,candidate in (('raw',model),('ema',ema)):
                        status(state='evaluating',epoch=epoch,update=step,arm=arm,precision='fp16')
                        report=validation(candidate,'fp16');g=gates(report,baselines['fp16']);s=score(report)
                        write_json(work/f'step_{step:04d}_{arm}_fp16.json',dict(validation=report,retention=g,score=s))
                        details=dict(fp16=compact(report),fp16_retention=g,score=s)
                        # Both precisions every epoch: four scheduled checks only.
                        status(state='evaluating',epoch=epoch,update=step,arm=arm,precision='bf16')
                        bf=validation(candidate,'bf16');bg=gates(bf,baselines['bf16'])
                        write_json(work/f'step_{step:04d}_{arm}_bf16.json',dict(validation=bf,retention=bg,score=score(bf)))
                        details.update(bf16=compact(bf),bf16_retention=bg)
                        if g['eligible'] and bg['eligible'] and s<best_score:
                            best_score=s;best_step=step;best_arm=arm;save_atomic(work/'best.pt',snapshot(epoch,position,candidate,arm))
                        entry['arms'][arm]=details
                    check_bn();entry.update(best_update=best_step,best_arm=best_arm,best_score=best_score);history.append(entry)
                    write_json(work/'history.json',history)
                    with (work/'metrics.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(entry,ensure_ascii=False)+'\n')
                    save_atomic(work/'last.pt',snapshot(epoch,position));save_atomic(work/f'step_{step:04d}.pt',snapshot(epoch,position))
                    print('EPOCH',epoch,'UPDATE',step,'BEST',best_step,best_arm,flush=True)
            assert all(v==1 for v in encountered.values()) and sum(encountered.values())==4*sampling['training_crops']
            assert sum(v for (k,i),v in encountered.items() if k=='yolo')==sampling['training_crops']
            terms=Counter(mix.datasets[k].rows[i]['text'] for k,i in encountered)
            required=sampling['plans'][epoch-1]['exact_term_counts'];assert all(terms[t]>=1 for t in required)
            audit=dict(epoch=epoch,records=sum(encountered.values()),yolo_presentations=sampling['training_crops'],all_selected_records_once=True,
                       exact_term_encounters={t:terms[t] for t in required})
            epoch_audits=[a for a in epoch_audits if a['epoch']!=epoch]+[audit]
            write_json(work/'epoch_audits.json',epoch_audits);save_atomic(work/f'epoch_{epoch:02d}.pt',snapshot(epoch,len(plan)));epoch_offset=end
        check_bn();assert step==total and len(epoch_audits)==4 and sha(START)==source_sha
        save_atomic(work/'final_raw.pt',snapshot(4,len(plans[-1]),model));save_atomic(work/'final_ema.pt',snapshot(4,len(plans[-1]),ema,'ema'))
        result=dict(state='completed',updates=step,epochs=4,presentations=sampling['total_presentations'],yolo_presentations=sampling['yolo_presentations'],
            best_update=best_step,best_arm=best_arm,best_score=best_score,bn147_buffers_fixed=True,bn_affine_trainable=True,source_unchanged=True,production_promoted=False)
        write_json(work/'report.json',result)
        lines=['# V38: four YOLO passes at25%','',json.dumps(result,indent=2),'',
               'Each eligible train crop once per pass; menu-disjoint holdout, exact-hash exclusion; VLM labels not independently verified. '
               'OOV labels excluded without rewriting. No additional image distortion. 521 exact synthetic terms per pass. '
               'best.pt is retention-selected; final_raw.pt/last.pt and final_ema.pt preserve final weights. Initial V36 is fallback.', '',
               '| Epoch | Update | Arm | YOLO train CER | YOLO holdout CER | FP16 failures | BF16 failures |','|---:|---:|---|---:|---:|---|---|']
        for h in history:
            for arm,d in h['arms'].items():
                full=load(work/f"step_{h['update']:04d}_{arm}_fp16.json")['validation']
                lines.append(f"| {h['epoch']} | {h['update']} | {arm} | {full['yolo_current_train']['0.0']['all']['cer']:.2%} | {full['yolo_current_holdout']['0.0']['all']['cer']:.2%} | {d['fp16_retention']['failed']} | {d['bf16_retention']['failed']} |")
        (work/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');status(**result)
    except Exception:status(state='failed',error=traceback.format_exc());raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=WORK);p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true');run(p.parse_args())
