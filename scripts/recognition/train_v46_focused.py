"""Time-bounded, frozen-CNN recurrent adaptation from promotedV43."""
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import random
import time
import traceback
import zipfile
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.recognition.prepare_v46 import prepare,WORK,SEED,START
from scripts.recognition.prepare_v44 import expanded_state
from scripts.recognition.train_v45_latin_head import latin_metrics,rows,V40
from scripts.recognition.prepare_v43 import normalize_dataset,digit_metrics
from scripts.recognition.prepare_v44 import adaptation_metrics
from scripts.recognition.train_v38_yolo25 import BucketCrops
from outputs.recognition.v37_long512 import make_views,evaluate_any,LongLines,WORK as V37
from outputs.recognition.v36_yolo_fullpasses import load
from outputs.recognition.application_mixture_pilot import cpu_copy
from outputs.recognition.evaluate_bn_swap import gates as legacy_gates,V34
from outputs.recognition.evaluate_fp16_retention import evaluate_fp16
from src.recognition.current import REGISTRY,current_checkpoint
from src.recognition.retention import apply_retention_policy
from src.recognition.vocabulary import split_existing_classifier,fused_state
from src.recognition.retention_distillation import teacher_kl
from src.recognition.model import MenuRecognizer
from src.recognition.font_training import evaluate
from src.recognition.data import collate
from src.recognition.checkpointing import save_atomic
from src.recognition.utils import sha,write_json,training_device,seed_all


def run(args):
    work=args.output;work.mkdir(parents=True,exist_ok=args.resume or args.smoke or args.prepare_only)
    started=time.time();prior_elapsed=0.
    def elapsed():return prior_elapsed+time.time()-started
    def status(**kw):write_json(work/'status.json',{'elapsed_seconds':elapsed(),**kw})
    try:
        torch.set_num_threads(4);seed_all(SEED)
        source_hash=sha(START);registry_hash=sha(REGISTRY);active=current_checkpoint();active_hash=sha(active)
        assert load(REGISTRY)['version']=='V43'
        source=torch.load(START,map_location='cpu',weights_only=False)
        assert source['updates']==2000 and source['checkpoint_arm']=='raw'
        status(state='preparing');vocab,added,mix,plan,extra,sampling=prepare(work,source)
        mapping=vocab['char_to_id'];old_classes=len(source['vocab']['char_to_id'])+1
        cfg=dict(source['config'],classes=len(mapping)+1)
        rates=dict(recurrent=1e-5,old_classifier=3e-6,new_classifier=1e-3)
        config=dict(source=str(START),source_sha256=source_hash,max_updates=4000,max_presentations=128000,
            batch=32,microbatch=4,seed=SEED,rates=rates,warmup_updates=200,new_rows_only_updates=200,
            min_lr_factor=.1,ema_decay=.998,teacher_kl_weight=1.,temperature=2.,training_deadline_seconds=args.train_seconds,
            frozen='Encoder,projection,orientationconditioning,temporalblocks,BNaffineand147buffers in evalmode; only residualBiLSTM switches to training mode after200 for cuDNNbackward.',
            trainable='After200updates:complete residualBiLSTM branch+oldclassifier+33newrows.',
            teacher='FrozenexpandedV43; fullsoftmax validframe KL onold-only ordinary/application/YOLOreplay,notLatin/parentheses/digit/basic targets.',
            selection='Weightedtargetscore;source-relativeguardsonYOLO,historical,shorttext,digits;reportabsolute17gatesbothprecisions;noautopromotion.',
            sampling_sha256=sha(work/'sampling.json'))
        if (work/'run_config.json').exists():assert load(work/'run_config.json')==config,'Resume settings changed'
        write_json(work/'run_config.json',config)
        if args.prepare_only:status(state='prepared');return
        device=training_device('cuda:0','RTX 3070');initial=expanded_state(source,vocab)
        model=MenuRecognizer(len(mapping)+1,use_bilstm=True);model.load_state_dict(initial)
        model.classifier=split_existing_classifier(model.classifier,old_classes)
        model.to(device).eval();model.vocabulary=mapping;model.recurrent.lstm.flatten_parameters()
        model.requires_grad_(False);model.classifier.new.requires_grad_(True)
        ema=copy.deepcopy(model).eval().requires_grad_(False);ema.recurrent.lstm.flatten_parameters()
        teacher=MenuRecognizer(len(mapping)+1,use_bilstm=True).to(device)
        teacher.load_state_dict(initial);teacher.eval().requires_grad_(False);teacher.recurrent.lstm.flatten_parameters()
        frozen={k:v.clone() for k,v in model.state_dict().items() if not k.startswith(('recurrent.','classifier.'))}
        assert sum(k.endswith(('running_mean','running_var','num_batches_tracked')) for k in frozen)==147
        def audit():
            for candidate in (model,ema):
                assert not candidate.training
                assert all(torch.equal(candidate.state_dict()[k],v) for k,v in frozen.items())
                assert all(not m.training for m in candidate.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm))
            assert all(not p.requires_grad and p.grad is None for n,p in model.named_parameters() if not n.startswith(('recurrent.','classifier.')))
        groups=[]
        for role,module in [('recurrent',model.recurrent),('old_classifier',model.classifier.old),('new_classifier',model.classifier.new)]:
            for decay in (True,False):
                pp=[p for p in module.parameters() if (p.ndim>1)==decay]
                if pp:groups.append(dict(params=pp,lr=rates[role],base_lr=rates[role],role=role,weight_decay=.01 if decay else 0.))
        optimizer=torch.optim.AdamW(groups);criterion=nn.CTCLoss(blank=0,zero_infinity=False)
        replay={'yolo','original','generated_clean','generated_noisy'}
        def train_batch(batch,step,kind):
            model.recurrent.requires_grad_(step>200);model.classifier.old.requires_grad_(step>200)
            # cuDNN saves RNN backward intermediates only in training mode. This
            # single-layer LSTM has no dropout; backbone/temporalBN remain eval.
            model.recurrent.train(step>200)
            factor=min(1.,step/200)*(.1+.9*(1+math.cos(math.pi*max(0,step-200)/3800))/2)
            for g in optimizer.param_groups:g['lr']=g['base_lr']*factor
            optimizer.zero_grad(set_to_none=True);offset=0;cv=kv=0.;n=len(batch['texts'])
            for start in range(0,n,4):
                b={k:v[start:start+4] for k,v in batch.items() if k!='targets'}
                count=int(b['lengths'].sum());targets=batch['targets'][offset:offset+count];offset+=count
                images=b['images'].to(device);ori=b['orientation'].to(device)
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(images,ori,b['input_lengths'])
                ctc=criterion(logits.float().log_softmax(-1),targets.to(device),b['input_lengths'],b['lengths'])
                eligible=torch.tensor([kind in replay and not any(c in added for c in t) for t in b['texts']],dtype=torch.bool)
                if eligible.any():
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):ref=teacher(images,ori,b['input_lengths'])
                    kl=teacher_kl(logits[:,eligible],ref[:,eligible],b['input_lengths'][eligible],2.)*float(eligible.float().mean())
                else:kl=logits.float().sum()*0
                loss=ctc+kl
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite CTC/KL')
                w=len(b['texts'])/n;(loss*w).backward();cv+=float(ctc.detach())*w;kv+=float(kl.detach())*w
            if step<=200:
                assert all(torch.equal(v,initial[k]) for k,v in fused_state(model).items() if not k.startswith('classifier.'))
                assert torch.equal(model.classifier.old.weight.detach().cpu(),source['model']['classifier.weight'])
                assert torch.equal(model.classifier.old.bias.detach().cpu(),source['model']['classifier.bias'])
            else:
                assert all(p.grad is not None for p in model.recurrent.parameters())
                assert model.classifier.old.weight.grad is not None
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5,error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for p,q in zip(ema.parameters(),model.parameters()):p.lerp_(q,.002)
            return dict(ctc=cv,teacher_kl=kv,lr_factor=factor)
        if args.smoke:
            details=[];t0=time.time()
            for step in (1,201):
                for kind in ('latin','parentheses','digits','basic','yolo','original','generated_clean','generated_noisy'):
                    ids=next(b for b in plan if b[0][0]==kind);batch=collate([mix[key] for key in ids])
                    details.append(dict(kind=kind,step=step,**train_batch(batch,step,kind)));audit()
            write_json(work/'smoke.json',dict(passed=True,batches=details,seconds=time.time()-t0,frozen_state_verified=True,
                bn147_buffers_fixed=True,recurrent_and_old_head_gradients_after_warmup=True));status(state='smoke_passed');return
        _,views=make_views(cfg,mapping);views.update(extra)
        views['new_character_probe']=BucketCrops([r for r in rows(V40/'targeted/labels.jsonl') if r['split']=='probe'],V40/'targeted',cfg,mapping)
        long=rows(V37/'long_labels.jsonl')
        for view in ('clean','noisy'):views['long512_'+view]=LongLines([r for r in long if r['split']=='holdout' and r['view']==view],V37,cfg,mapping)
        for ds in views.values():
            if hasattr(ds,'mapping'):ds.mapping=mapping
            normalize_dataset(ds)
        baselines={'fp16':load('outputs/recognition/evaluation/v35_paired_short_inspection/fp16_full_source.json'),'bf16':load(V34/'initial.json')}
        def validate(candidate,step,arm):
            deployed=MenuRecognizer(len(mapping)+1,use_bilstm=True).to(device)
            deployed.load_state_dict(fused_state(candidate));deployed.vocabulary=mapping;deployed.eval();deployed.recurrent.lstm.flatten_parameters()
            output={}
            for precision in ('fp16','bf16'):
                status(state='evaluating',update=step,arm=arm,precision=precision);r={}
                for name,ds in views.items():
                    fn=evaluate_fp16 if precision=='fp16' else evaluate
                    r[name]=evaluate_any(deployed,ds,device,precision) if name in extra or name.startswith(('new_character_','long512_')) else fn(deployed,ds,device)
                # Attach digit-template identities for the shared digit evaluator.
                for name in ('digits_known','digits_heldout'):
                    for record in r[name]['0.0']['predictions']:
                        row=views[name].rows[record['index']];record.update(kind=row['kind'],fullwidth_rendered=row['fullwidth_rendered'])
                result=dict(validation=r,retention=apply_retention_policy(legacy_gates(r,baselines[precision])),latin=latin_metrics(r,added),
                    adaptation_metrics=adaptation_metrics(r),digit_metrics=digit_metrics(r),
                    false_new_latin_emissions={name:sum(any(c in added for c in row['prediction']) for row in value['0.0']['predictions'] if not any(c in added for c in row['truth'])) for name,value in r.items() if not name.startswith('latin')})
                write_json(work/(f'initial_{precision}.json' if step==0 else f'step_{step:04d}_{arm}_{precision}.json'),result);output[precision]=result
            del deployed;audit()
            assert all(torch.equal(v.cpu(),initial[k]) for k,v in teacher.state_dict().items())
            return output
        step=0;streams=Counter();chars=Counter();unique={k:set() for k in mix.datasets};history=[];best_step=None;best_arm=None;best_score=None
        def snapshot(candidate=None,arm='raw'):
            return dict(model=fused_state(candidate or model),raw_training_model=cpu_copy(model.state_dict()),ema_training_model=cpu_copy(ema.state_dict()),
                optimizer=cpu_copy(optimizer.state_dict()),vocab=vocab,config=dict(cfg,v46_training=config),updates=step,checkpoint_arm=arm,
                best_update=best_step,best_arm=best_arm,best_score=best_score,stream_encounters=dict(streams),character_encounters=dict(chars),
                unique_records={k:sorted(v) for k,v in unique.items()},history=history,budget_elapsed=elapsed(),
                rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all())
        if not (work/'source.zip').exists():
            with zipfile.ZipFile(work/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in [Path(__file__),Path('scripts/recognition/prepare_v46.py'),Path('scripts/recognition/run_v46.ps1'),Path('scripts/recognition/cleanup_after_v46.py'),*Path('src/recognition').glob('*.py')]:z.write(p,p.as_posix())
        if args.resume and (work/'last.pt').exists():
            ck=torch.load(work/'last.pt',map_location='cpu',weights_only=False)
            assert ck['vocab']==vocab
            model.load_state_dict(ck['raw_training_model']);ema.load_state_dict(ck['ema_training_model']);optimizer.load_state_dict(ck['optimizer'])
            step=ck['updates'];best_step=ck['best_update'];best_arm=ck['best_arm'];best_score=ck['best_score'];prior_elapsed=ck['budget_elapsed']
            streams.update(ck['stream_encounters']);chars.update(ck['character_encounters']);history=ck['history']
            for k,v in ck['unique_records'].items():unique[k].update(v)
            random.setstate(ck['rng_python']);np.random.set_state(ck['rng_numpy']);torch.set_rng_state(ck['rng_torch']);torch.cuda.set_rng_state_all(ck['rng_cuda'])
            assert load(work/'initial_parity.json')['passed']
        else:
            output=validate(model,0,'initial');parity={}
            for precision,result in output.items():
                expected=load(START.parent/f'step_2000_raw_{precision}.json')['validation'];parity[precision]={}
                for name,value in expected.items():
                    if name not in result['validation']:continue
                    a=result['validation'][name]['0.0']['predictions'];b=value['0.0']['predictions']
                    assert len(a)==len(b) and all(x['truth']==y['truth'] for x,y in zip(a,b))
                    parity[precision][name]=sum(x['prediction']!=y['prediction'] for x,y in zip(a,b))
                assert not any(parity[precision].values())
                assert not any(any(c in added for c in row['prediction']) for v in result['validation'].values() for row in v['0.0']['predictions'])
            write_json(work/'initial_parity.json',dict(passed=True,prediction_disagreements=parity,old_ids_unchanged=True,new_rows_below_blank=8))
            save_atomic(work/'initial.pt',snapshot());save_atomic(work/'last.pt',snapshot())
        initial_results={p:load(work/f'initial_{p}.json') for p in ('fp16','bf16')}
        def selection(result,precision):
            base=initial_results[precision];r=result['validation'];b=base['validation']
            checks={name:r[name]['0.0']['all']['cer']<=b[name]['0.0']['all']['cer']+tol+1e-12 for name,tol in [('yolo_current_holdout',.01),('historical_bucket',.01),('fresh_short_clean',.02),('fresh_short_noisy',.02)]}
            checks['digit_recall']=result['digit_metrics']['digits_heldout']['macro_recall']>=base['digit_metrics']['digits_heldout']['macro_recall']-.02
            checks['no_added_absolute_gate_failures']=set(result['retention']['failed'])<=set(base['retention']['failed'])
            score=sum(w*r[name]['0.0']['all']['cer'] for w,name in [(.4,'latin_heldout'),(.2,'parentheses_heldout'),(.15,'digits_heldout'),(.1,'basic_heldout'),(.15,'yolo_current_holdout')])
            return dict(eligible=all(checks.values()),checks=checks,score=score)
        def checkpoint_evaluation():
            nonlocal best_step,best_arm,best_score
            arms={}
            for arm,candidate in [('raw',model),('ema',ema)]:
                result=validate(candidate,step,arm);selected={p:selection(r,p) for p,r in result.items()}
                arms[arm]={p:dict(retention=r['retention'],selection=selected[p],latin=r['latin'],
                    parentheses=r['adaptation_metrics']['parentheses'],digits=r['digit_metrics'],
                    basic_cer=r['validation']['basic_heldout']['0.0']['all']['cer'],yolo_cer=r['validation']['yolo_current_holdout']['0.0']['all']['cer']) for p,r in result.items()}
                score=selected['fp16']['score']
                if all(v['eligible'] for v in selected.values()) and (best_score is None or score<best_score):
                    best_step=step;best_arm=arm;best_score=score;save_atomic(work/'best.pt',snapshot(candidate,arm))
            history.append(dict(update=step,elapsed_seconds=elapsed(),arms=arms))
            (work/'metrics.jsonl').write_text(''.join(json.dumps(h,ensure_ascii=False)+'\n' for h in history),encoding='utf-8')
            save_atomic(work/f'step_{step:04d}.pt',snapshot());save_atomic(work/'last.pt',snapshot())
        loader=DataLoader(mix,batch_sampler=plan[step:],num_workers=0,collate_fn=collate,generator=torch.Generator().manual_seed(SEED))
        time_limited=False
        for batch in loader:
            if elapsed()>=args.train_seconds:time_limited=True;break
            kind=plan[step][0][0];loss=train_batch(batch,step+1,kind)
            for k,i in plan[step]:streams[k]+=1;chars.update(mix.datasets[k].rows[i]['text']);unique[k].add(i)
            step+=1
            if step==1 or step%100==0:
                audit();status(state='training',update=step,max_updates=4000,**loss)
                if step not in (500,1000,2000,3000,4000):save_atomic(work/'last.pt',snapshot())
                print('TRAIN',step,loss,flush=True)
            if step in (500,1000,2000,3000,4000):checkpoint_evaluation()
        if step and (not history or history[-1]['update']!=step):checkpoint_evaluation()
        audit();assert sum(streams.values())==step*32
        assert sha(START)==source_hash and sha(REGISTRY)==registry_hash and sha(active)==active_hash
        consumed=plan[:step];actual_terms=Counter(mix.datasets[k].rows[i]['text'] for b in consumed for k,i in b)
        new_delta=(model.classifier.new.weight.detach().cpu()-initial['classifier.weight'][old_classes:]).flatten(1).norm(dim=1)
        report=dict(completed=True,updates=step,presentations=step*32,time_limited=time_limited,elapsed_seconds=elapsed(),stream_encounters=dict(streams),
            actual_ratios={k:v/(step*32) for k,v in streams.items()},unique_records={k:len(v) for k,v in unique.items()},
            actual_character_encounters=dict(chars),required_term_encounters={t:actual_terms[t] for t in sampling['required_term_encounters']},
            new_row_delta_norms={c:float(v) for c,v in zip(added,new_delta)},
            bn147_fixed=True,cnn_projection_condition_temporal_and_all_bn_affine_fixed=True,new_rows_only_first200_verified=True,
            teacher_unchanged=True,source_and_registry_unchanged=True,best_update=best_step,best_arm=best_arm,best_score=best_score,
            full_4000_plan_completed=step==4000,production_promoted=False)
        save_atomic(work/'final_raw.pt',snapshot());save_atomic(work/'final_ema.pt',snapshot(ema,'ema'));write_json(work/'report.json',report)
        lines=['# V46 focused recurrent adaptation','',json.dumps(report,ensure_ascii=False,indent=2),'',
            'Promoted V43 remains unchanged. No automatic promotion; source-relative selection is not an absolute retention pass. Time-limited run reports actual coverage, not full-pass claims.','',
            '|Update|Arm|Latin train CER|Latin heldout recall|Parenthesis heldout CER|Digit heldout CER|Basic heldout CER|YOLO CER|FP16 failed gates|','|---:|---|---:|---:|---:|---:|---:|---:|---|']
        for h in history:
            for arm,p in h['arms'].items():
                r=p['fp16'];update=h['update']
                paren=load(work/f'step_{update:04d}_{arm}_fp16.json')['validation']['parentheses_heldout']['0.0']['all']['cer']
                lines.append(f"|{update}|{arm}|{r['latin']['latin_train']['cer']:.2%}|{r['latin']['latin_heldout']['groups']['all52']['macro_recall']:.2%}|{paren:.2%}|{r['digits']['digits_heldout']['cer']:.2%}|{r['basic_cer']:.2%}|{r['yolo_cer']:.2%}|{r['retention']['failed']}|")
        (work/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');status(state='completed',**report)
    except Exception:status(state='failed',error=traceback.format_exc());raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=WORK);p.add_argument('--resume',action='store_true')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--train-seconds',type=int,default=6000)
    run(p.parse_args())
