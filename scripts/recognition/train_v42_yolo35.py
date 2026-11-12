"""2000-update V41 continuation:35%YOLOjitter,30%true512,BNstats frozen."""
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
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.recognition.prepare_v42 import prepare,WORK,START,SEED,V41
from scripts.recognition.train_v40_new_characters import new_character_metrics,make_optimizer
from scripts.recognition.review_v38_yolo25 import cluster_bootstrap
from outputs.recognition.v37_long512 import make_views,evaluate_any
from outputs.recognition.v36_yolo_fullpasses import load,factor
from outputs.recognition.application_mixture_pilot import cpu_copy
from outputs.recognition.evaluate_bn_swap import gates as legacy_gates,V34
from outputs.recognition.evaluate_fp16_retention import evaluate_fp16
from src.recognition.current import REGISTRY
from src.recognition.retention import apply_retention_policy,POLICY_PATH
from src.recognition.vocabulary import split_existing_classifier,fused_state
from src.recognition.retention_distillation import teacher_kl
from src.recognition.model import MenuRecognizer
from src.recognition.font_training import evaluate
from src.recognition.data import collate
from src.recognition.checkpointing import save_atomic
from src.recognition.utils import sha,write_json,training_device,seed_all


def run(args):
    work=args.output;work.mkdir(parents=True,exist_ok=args.resume or args.prepare_only);started=time.time()
    def status(**kw):write_json(work/'status.json',dict(elapsed_seconds=time.time()-started,**kw))
    try:
        torch.set_num_threads(4);seed_all(SEED);source_hash=sha(START);registry_hash=sha(REGISTRY)
        source=torch.load(START,map_location='cpu',weights_only=False);assert source['updates']==1000 and source['checkpoint_arm']=='ema'
        status(state='preparing');vocab,added,mix,plan,extra,sampling=prepare(work,source)
        mapping=vocab['char_to_id'];cfg=source['config'];rates={'encoder':5e-7,'head':3e-6,'new_classifier':3e-5}
        config=dict(source=str(START),source_sha256=source_hash,source_arm='V41selectedEMA1000',updates=2000,presentations=64000,
            seed=SEED,batch=32,microbatch=4,trainable='Full recognizer,including BNaffine; runningstats frozen',learning_rates=rates,
            warmup_updates=200,min_lr_factor=.1,ema_decay=.998,teacher_consistency_weight=.5,teacher_temperature=2.,
            teacher='Frozen startingV41; full-vocabulary valid-frame KL only on old synthetic/long replay, not YOLO or new-character targets.',
            retention_policy=load(POLICY_PATH)['id'],sampling_sha256=sha(work/'sampling.json'),
            selection='0.6YOLOholdoutCER+0.2heldoutfont long512meanCER+0.2*(1-newcharheldoutfontmacrorecall),all17revisedgates bothprecisions; source fallback.',
            interpretation='Combined recipe change,not matched single-factor ablation. Clean heldout geometry remains unchanged.')
        if (work/'run_config.json').exists():assert load(work/'run_config.json')==config,'Resume configuration changed'
        write_json(work/'run_config.json',config)
        if args.prepare_only:status(state='prepared',updates=2000,physical512_fraction=sampling['actual_512_fraction']);return
        device=training_device('cuda:0','RTX 3070');model=MenuRecognizer(len(mapping)+1,use_bilstm=True)
        model.load_state_dict(source['model']);model.classifier=split_existing_classifier(model.classifier,1723)
        assert all(torch.equal(v,source['model'][k]) for k,v in fused_state(model).items())
        model.vocabulary=mapping;model.to(device);model.recurrent.lstm.flatten_parameters()
        ema=copy.deepcopy(model).eval();ema.requires_grad_(False);ema.recurrent.lstm.flatten_parameters()
        teacher=MenuRecognizer(len(mapping)+1,use_bilstm=True).to(device);teacher.load_state_dict(source['model']);teacher.eval();teacher.requires_grad_(False);teacher.recurrent.lstm.flatten_parameters()
        bn=[m for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)]
        buffers={k:v for k,v in source['model'].items() if k.endswith(('running_mean','running_var','num_batches_tracked'))};assert len(buffers)==147
        def check_bn():
            for candidate in (model,ema,teacher):assert all(torch.equal(candidate.state_dict()[k].cpu(),v) for k,v in buffers.items())
        def check_teacher():assert all(torch.equal(v.cpu(),source['model'][k]) for k,v in teacher.state_dict().items())
        optimizer=make_optimizer(model)
        for g in optimizer.param_groups:g['lr']=g['base_lr']=rates[g['role']]
        criterion=nn.CTCLoss(blank=0,zero_infinity=False);_,views=make_views(cfg,mapping);views.update(extra)
        for ds in views.values():
            if hasattr(ds,'mapping'):ds.mapping=mapping
        baselines={'fp16':load('outputs/recognition/evaluation/v35_paired_short_inspection/fp16_full_source.json'),'bf16':load(V34/'initial.json')}
        generic=set(extra)
        def validation(candidate,precision):
            deployed=MenuRecognizer(len(mapping)+1,use_bilstm=True).to(device);deployed.load_state_dict(fused_state(candidate));deployed.vocabulary=mapping
            deployed.eval();deployed.recurrent.lstm.flatten_parameters();r={}
            for name,ds in views.items():
                fn=evaluate_fp16 if precision=='fp16' else evaluate
                r[name]=evaluate_any(deployed,ds,device,precision) if name in generic else fn(deployed,ds,device)
            del deployed;return r
        def long_cer(r):return sum(r['long512_'+v]['0.0']['by_cohort']['heldout']['cer'] for v in ('clean','noisy'))/2
        def score(r):return .6*r['yolo_current_holdout']['0.0']['all']['cer']+.2*long_cer(r)+.2*(1-new_character_metrics(r,added)['heldout']['macro_recall'])
        step=0;best_step=0;best_arm='initial_V41EMA1000';best_score=None;history=[];streams=Counter();characters=Counter();canvases=Counter();yolo_seen=Counter()
        def snapshot(candidate=None,arm='raw'):
            return dict(model=fused_state(candidate or model),raw_training_model=cpu_copy(model.state_dict()),ema_training_model=cpu_copy(ema.state_dict()),
                optimizer=cpu_copy(optimizer.state_dict()),vocab=vocab,config=dict(cfg,v42_training=config),updates=step,checkpoint_arm=arm,
                best_update=best_step,best_arm=best_arm,best_score=best_score,stream_encounters=dict(streams),character_encounters=dict(characters),
                canvas_encounters=dict(canvases),yolo_base_encounters=dict(yolo_seen),rng_python=random.getstate(),rng_numpy=np.random.get_state(),
                rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all())
        if not (work/'source.zip').exists():
            with zipfile.ZipFile(work/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in [Path(__file__),Path('scripts/recognition/prepare_v42.py'),*Path('src/recognition').glob('*.py')]:z.write(p,p.as_posix())
        if args.resume and (work/'last.pt').exists():
            saved=torch.load(work/'last.pt',map_location='cpu',weights_only=False);assert saved['vocab']==vocab
            model.load_state_dict(saved['raw_training_model']);ema.load_state_dict(saved['ema_training_model']);optimizer.load_state_dict(saved['optimizer'])
            step=saved['updates'];best_step=saved['best_update'];best_arm=saved['best_arm'];best_score=saved['best_score'];history=load(work/'history.json')
            streams.update(saved['stream_encounters']);characters.update(saved['character_encounters']);canvases.update(saved['canvas_encounters']);yolo_seen.update(saved['yolo_base_encounters'])
            random.setstate(saved['rng_python']);np.random.set_state(saved['rng_numpy']);torch.set_rng_state(saved['rng_torch']);torch.cuda.set_rng_state_all(saved['rng_cuda'])
            assert load(work/'initial_parity.json')['passed']
        else:
            initial={};parity={}
            for precision in ('fp16','bf16'):
                status(state='initial_evaluation',precision=precision);r=validation(model,precision);initial[precision]=r
                expected=load(V41/f'step_1000_ema_{precision}.json')['validation'];parity[precision]={}
                for name,old in expected.items():
                    a=r[name]['0.0']['predictions'];b=old['0.0']['predictions'];assert len(a)==len(b)
                    parity[precision][name]=sum(x['prediction']!=y['prediction'] for x,y in zip(a,b))
                g=apply_retention_policy(legacy_gates(r,baselines[precision]));assert not any(parity[precision].values()) and g['eligible']
                write_json(work/f'initial_{precision}.json',dict(validation=r,retention=g,new_character_metrics=new_character_metrics(r,added)))
            write_json(work/'initial_parity.json',dict(passed=True,source='V41selectedEMA1000',prediction_disagreements=parity))
            best_score=score(initial['fp16']);save_atomic(work/'best.pt',snapshot(model,best_arm));write_json(work/'history.json',history)
        check_bn();check_teacher();ctc_sum=0.;teacher_sum=0.;batches_seen=0
        loader=DataLoader(mix,batch_sampler=plan[step:],num_workers=0,collate_fn=collate,generator=torch.Generator().manual_seed(SEED))
        for batch in loader:
            kind=plan[step][0][0];assert all(k==kind for k,i in plan[step]);model.train()
            for m in bn:m.eval()
            teacher.eval();f=factor(step+1,2000)
            for g in optimizer.param_groups:g['lr']=g['base_lr']*f
            optimizer.zero_grad(set_to_none=True);offset=0;ctc_value=0.;teacher_value=0.;n=len(batch['texts'])
            for start in range(0,n,4):
                micro={k:v[start:start+4] for k,v in batch.items() if k!='targets'};length=int(micro['lengths'].sum())
                micro['targets']=batch['targets'][offset:offset+length];offset+=length
                images=micro['images'].to(device);orientation=micro['orientation'].to(device)
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(images,orientation,micro['input_lengths'])
                with torch.autocast('cuda',enabled=False):ctc=criterion(logits.float().log_softmax(-1),micro['targets'].to(device),micro['input_lengths'],micro['lengths'])
                if kind in ('original','generated','long512'):
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):reference=teacher(images,orientation,micro['input_lengths'])
                    with torch.autocast('cuda',enabled=False):kd=teacher_kl(logits,reference,micro['input_lengths'],2.)
                else:kd=logits.float().sum()*0
                loss=ctc+.5*kd
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite CTC/teacher loss')
                weight=len(micro['texts'])/n;(loss*weight).backward();ctc_value+=float(ctc.detach())*weight;teacher_value+=float(kd.detach())*weight
            assert all(m.weight.grad is not None and m.bias.grad is not None for m in bn)
            nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();step+=1
            with torch.no_grad():
                for p,q in zip(ema.parameters(),model.parameters()):p.lerp_(q,.002)
            for k,i in plan[step-1]:
                r=mix.datasets[k].rows[i];streams[k]+=1;characters.update(c for c in r['text'] if c in added)
                width=batch['images'].shape[-1];canvases[width]+=1
                if k=='yolo':yolo_seen[mix.datasets[k].views[i]['base_index']]+=1
            ctc_sum+=ctc_value;teacher_sum+=teacher_value;batches_seen+=1
            if step==1 or step%100==0:
                check_bn();status(state='training',update=step,total_updates=2000,mean_ctc=ctc_sum/batches_seen,mean_teacher=teacher_sum/batches_seen,best_update=best_step)
            if step%500==0:
                entry=dict(update=step,lr_factor=f,mean_ctc=ctc_sum/batches_seen,mean_teacher=teacher_sum/batches_seen,arms={})
                for arm,candidate in (('raw',model),('ema',ema)):
                    details={}
                    for precision in ('fp16','bf16'):
                        status(state='evaluating',update=step,arm=arm,precision=precision);r=validation(candidate,precision)
                        g=apply_retention_policy(legacy_gates(r,baselines[precision]));m=new_character_metrics(r,added)
                        write_json(work/f'step_{step:04d}_{arm}_{precision}.json',dict(validation=r,retention=g,new_character_metrics=m,score=score(r)))
                        details[precision]=dict(retention=g,score=score(r),new_character_macro_recall=m['heldout']['macro_recall'],
                            yolo_holdout_cer=r['yolo_current_holdout']['0.0']['all']['cer'],long_heldoutfont_mean_cer=long_cer(r))
                    if all(d['retention']['eligible'] for d in details.values()) and details['fp16']['score']<best_score:
                        best_score=details['fp16']['score'];best_step=step;best_arm=arm;save_atomic(work/'best.pt',snapshot(candidate,arm))
                    entry['arms'][arm]=details
                check_bn();check_teacher();entry.update(best_update=best_step,best_arm=best_arm);history.append(entry);write_json(work/'history.json',history)
                with (work/'metrics.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(entry,ensure_ascii=False)+'\n')
                save_atomic(work/f'step_{step:04d}.pt',snapshot());print('EVAL',step,'best',best_step,best_arm,flush=True)
            if step%100==0:save_atomic(work/'last.pt',snapshot())
        check_bn();check_teacher();assert step==2000 and streams==sampling['streams'] and sum(streams.values())==64000
        assert dict(canvases)=={int(k):v for k,v in sampling['physical_width_presentations'].items()}
        assert len(yolo_seen)==1561 and sum(yolo_seen.values())==22400
        assert all(characters[c]==sampling['planned_new_character_counts'][c] and characters[c]>=100 for c in added)
        assert sha(START)==source_hash and sha(REGISTRY)==registry_hash
        save_atomic(work/'final_raw.pt',snapshot());save_atomic(work/'final_ema.pt',snapshot(ema,'ema'))
        report=dict(completed=True,updates=2000,presentations=64000,stream_encounters=dict(streams),physical_width_encounters=dict(canvases),
            actual_yolo_fraction=.35,actual_long512_fraction=.30,actual_physical512_fraction=canvases[512]/64000,
            yolo_base_encounters=dict(yolo_seen),new_character_encounters={c:characters[c] for c in added},minimum_new_character_encounters=min(characters[c] for c in added),
            bn147_buffers_fixed=True,teacher_unchanged=True,source_and_registry_unchanged=True,best_update=best_step,best_arm=best_arm,best_score=best_score,production_promoted=False)
        write_json(work/'report.json',report)
        paired={}
        for precision in ('fp16','bf16'):
            initial=load(work/f'initial_{precision}.json')['validation']['yolo_current_holdout']['0.0']['predictions'];paired[precision]={}
            for arm in ('raw','ema'):
                final=load(work/f'step_2000_{arm}_{precision}.json')['validation']['yolo_current_holdout']['0.0']['predictions']
                effect=cluster_bootstrap(initial,final,extra['yolo_current_holdout'].rows);effect['v42_minus_source_V41_pp']=effect.pop('v38_minus_v36_pp');paired[precision][arm]=effect
        write_json(work/'paired_analysis.json',paired)
        lines=['# V42: YOLO35% jitter +true long51230%','',json.dumps(report,ensure_ascii=False,indent=2),'',
            'Full fine-tune from gate-passing V41EMA1000; frozenBNstats,trainableaffine. NativeYOLOcrop edgepadding2–5%requested per side,quantizedto pixels; horizontalstretch.95–1.05; no added optical noise or glyph clipping. '
            'All1561traincrops used,156holdoutgeometryunchanged. Genuine11–14char512replay is30%; actual512share also includesYOLOviews. '
            'TeacherKL protects oldsynthetic/longreplay,not YOLO adaptation or novel-character targets. No automatic promotion.', '',
            '| Update | Arm | YOLOholdoutCER | Heldoutfont long512CER | Newcharrecall | FP16failures | BF16failures |','|---:|---|---:|---:|---:|---|---|']
        for h in history:
            for arm,d in h['arms'].items():lines.append(f"| {h['update']} | {arm} | {d['fp16']['yolo_holdout_cer']:.2%} | {d['fp16']['long_heldoutfont_mean_cer']:.2%} | {d['fp16']['new_character_macro_recall']:.2%} | {d['fp16']['retention']['failed']} | {d['bf16']['retention']['failed']} |")
        lines+=['','See initial/final fullreports for paired512versus384longprobes, cleanYOLOtrain/holdout and matchedclean/augmentedtrainingprobe. '
            'This combined package is not an isolated augmentation ablation. V38 remains promoted; sourceV41/registry preserved.']
        (work/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');status(state='completed',**report)
    except Exception:status(state='failed',error=traceback.format_exc());raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=WORK);p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true');run(p.parse_args())
