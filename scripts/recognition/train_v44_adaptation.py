"""V44: 2000-update V43 continuation with complete Latin coverage, parentheses and extra-long lines."""
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

from scripts.recognition.prepare_v44 import prepare,WORK,START,SEED,V43,digit_metrics,normalize_dataset,expanded_state,adaptation_metrics,evaluate_windows
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
        source=torch.load(START,map_location='cpu',weights_only=False);assert source['updates']==2000 and source['checkpoint_arm']=='raw'
        status(state='preparing');vocab,added,mix,plan,extra,sampling=prepare(work,source)
        mapping=vocab['char_to_id'];cfg=dict(source['config'],classes=len(mapping)+1);rates={'encoder':2.5e-7,'head':1e-6,'new_classifier':1e-3}
        config=dict(source=str(START),source_sha256=source_hash,source_arm='V43finalRAW2000',updates=2000,presentations=64000,
            seed=SEED,batch=32,microbatch=4,trainable='Full recognizer,including BNaffine; runningstats frozen',learning_rates=rates,
            warmup_updates=200,new_rows_only_updates=200,min_lr_factor=.1,ema_decay=.998,teacher_consistency_weight=1.,teacher_temperature=2.,
            teacher='Frozen expanded startingV43; valid-frame KL on ordinary/digit/existing11-14char replay only, not new extra-long/YOLO/Latin/parentheses/new-character targets.',
            retention_policy=load(POLICY_PATH)['id'],sampling_sha256=sha(work/'sampling.json'),
            selection='0.4YOLOholdoutCER+0.2LatinheldoutCER+0.1parenthesesheldoutCER+0.2selectedgeometrylongheldoutCER+0.1digitsheldoutCER; both-precision retention required. Source fallback not eligible.',
            interpretation='Combined recipe change,not matched single-factor ablation. Clean heldout geometry remains unchanged.')
        if (work/'run_config.json').exists():assert load(work/'run_config.json')==config,'Resume configuration changed'
        write_json(work/'run_config.json',config)
        if args.prepare_only:status(state='prepared',updates=2000,physical512_fraction=sampling['actual_512_fraction']);return
        device=training_device('cuda:0','RTX 3070');model=MenuRecognizer(len(mapping)+1,use_bilstm=True)
        initial_state=expanded_state(source,vocab);model.load_state_dict(initial_state);model.classifier=split_existing_classifier(model.classifier,len(source['vocab']['char_to_id'])+1)
        assert all(torch.equal(v,initial_state[k]) for k,v in fused_state(model).items())
        model.vocabulary=mapping;model.to(device);model.recurrent.lstm.flatten_parameters()
        ema=copy.deepcopy(model).eval();ema.requires_grad_(False);ema.recurrent.lstm.flatten_parameters()
        teacher=MenuRecognizer(len(mapping)+1,use_bilstm=True).to(device);teacher.load_state_dict(initial_state);teacher.eval();teacher.requires_grad_(False);teacher.recurrent.lstm.flatten_parameters()
        bn=[m for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)]
        buffers={k:v for k,v in source['model'].items() if k.endswith(('running_mean','running_var','num_batches_tracked'))};assert len(buffers)==147
        def check_bn():
            for candidate in (model,ema,teacher):assert all(torch.equal(candidate.state_dict()[k].cpu(),v) for k,v in buffers.items())
        def check_teacher():assert all(torch.equal(v.cpu(),initial_state[k]) for k,v in teacher.state_dict().items())
        optimizer=make_optimizer(model)
        for g in optimizer.param_groups:g['lr']=g['base_lr']=rates[g['role']]
        criterion=nn.CTCLoss(blank=0,zero_infinity=False);_,views=make_views(cfg,mapping);views.update(extra)
        for ds in views.values():
            if hasattr(ds,'mapping'):ds.mapping=mapping
            normalize_dataset(ds)
        baselines={'fp16':load('outputs/recognition/evaluation/v35_paired_short_inspection/fp16_full_source.json'),'bf16':load(V34/'initial.json')}
        generic=set(extra)
        def validation(candidate,precision):
            deployed=MenuRecognizer(len(mapping)+1,use_bilstm=True).to(device);deployed.load_state_dict(fused_state(candidate));deployed.vocabulary=mapping
            deployed.eval();deployed.recurrent.lstm.flatten_parameters();r={}
            for name,ds in views.items():
                fn=evaluate_fp16 if precision=='fp16' else evaluate
                r[name]=evaluate_any(deployed,ds,device,precision) if name in generic else fn(deployed,ds,device)
                if name.startswith('digits_'):
                    for record in r[name]['0.0']['predictions']:
                        row=ds.rows[record['index']]
                        record.update(kind=row['kind'],fullwidth_rendered=row['fullwidth_rendered'])
            for cohort in ('known','heldout'):
                r['longwide_overlap_'+cohort]=evaluate_windows(deployed,extra['longwide_'+cohort],cfg,mapping,device,precision)
            del deployed;return r
        def long_cer(r):return sum(r['long512_'+v]['0.0']['by_cohort']['heldout']['cer'] for v in ('clean','noisy'))/2
        def score(r):
            longkey='longwide_overlap_heldout' if sampling['geometry']=='overlap512' else 'longwide_heldout'
            return sum(weight*r[name]['0.0']['all']['cer'] for weight,name in ((.4,'yolo_current_holdout'),(.2,'latin_heldout'),(.1,'parentheses_heldout'),(.2,longkey),(.1,'digits_heldout')))
        step=0;best_step=0;best_arm='initial_V43RAW2000';best_score=None;history=[];streams=Counter();characters=Counter();canvases=Counter();yolo_seen=Counter()
        digit_encounters=Counter();digit_anchors=Counter();digit_images=Counter()
        def snapshot(candidate=None,arm='raw'):
            return dict(model=fused_state(candidate or model),raw_training_model=cpu_copy(model.state_dict()),ema_training_model=cpu_copy(ema.state_dict()),
                optimizer=cpu_copy(optimizer.state_dict()),vocab=vocab,config=dict(cfg,v44_training=config),updates=step,checkpoint_arm=arm,
                best_update=best_step,best_arm=best_arm,best_score=best_score,stream_encounters=dict(streams),character_encounters=dict(characters),
                canvas_encounters=dict(canvases),yolo_base_encounters=dict(yolo_seen),digit_encounters=dict(digit_encounters),
                digit_anchors=dict(digit_anchors),digit_images=dict(digit_images),rng_python=random.getstate(),rng_numpy=np.random.get_state(),
                rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all())
        if not (work/'source.zip').exists():
            with zipfile.ZipFile(work/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in [Path(__file__),Path('scripts/recognition/prepare_v44.py'),*Path('src/recognition').glob('*.py')]:z.write(p,p.as_posix())
        if args.resume and (work/'last.pt').exists():
            saved=torch.load(work/'last.pt',map_location='cpu',weights_only=False);assert saved['vocab']==vocab
            model.load_state_dict(saved['raw_training_model']);ema.load_state_dict(saved['ema_training_model']);optimizer.load_state_dict(saved['optimizer'])
            step=saved['updates'];best_step=saved['best_update'];best_arm=saved['best_arm'];best_score=saved['best_score'];history=load(work/'history.json')
            streams.update(saved['stream_encounters']);characters.update(saved['character_encounters']);canvases.update(saved['canvas_encounters']);yolo_seen.update(saved['yolo_base_encounters'])
            digit_encounters.update(saved['digit_encounters']);digit_anchors.update(saved['digit_anchors']);digit_images.update(saved['digit_images'])
            random.setstate(saved['rng_python']);np.random.set_state(saved['rng_numpy']);torch.set_rng_state(saved['rng_torch']);torch.cuda.set_rng_state_all(saved['rng_cuda'])
            assert load(work/'initial_parity.json')['passed']
        else:
            initial={};parity={};new_emissions={}
            for precision in ('fp16','bf16'):
                status(state='initial_evaluation',precision=precision);r=validation(model,precision);initial[precision]=r
                expected=load(V43/f'step_2000_raw_{precision}.json')['validation'];parity[precision]={}
                for name,old in expected.items():
                    a=r[name]['0.0']['predictions'];b=old['0.0']['predictions'];assert len(a)==len(b)
                    assert all(x['truth']==y['truth'] for x,y in zip(a,b)), 'Existing parity probes changed under digit normalization'
                    parity[precision][name]=sum(x['prediction']!=y['prediction'] for x,y in zip(a,b))
                g=apply_retention_policy(legacy_gates(r,baselines[precision]));assert not any(parity[precision].values())
                new_emissions[precision]=sum(any(c in sampling['latin_added'] for c in row['prediction']) for value in r.values() for row in value['0.0']['predictions'])
                assert new_emissions[precision]==0,'Untrained Latin rows emitted before updates'
                write_json(work/f'initial_{precision}.json',dict(validation=r,retention=g,new_character_metrics=new_character_metrics(r,added),digit_metrics=digit_metrics(r),adaptation_metrics=adaptation_metrics(r)))
            write_json(work/'initial_parity.json',dict(passed=True,source='V43finalRAW2000',prediction_disagreements=parity,new_token_emission_lines=new_emissions,old_ids_preserved=True,new_letters=sampling['latin_added'],new_rows_below_blank_by=8))
            best_score=score(initial['fp16']);save_atomic(work/'best.pt',snapshot(model,best_arm));write_json(work/'history.json',history)
        check_bn();check_teacher();ctc_sum=0.;teacher_sum=0.;batches_seen=0
        loader=DataLoader(mix,batch_sampler=plan[step:],num_workers=0,collate_fn=collate,generator=torch.Generator().manual_seed(SEED))
        for batch in loader:
            kind=plan[step][0][0];assert all(k==kind for k,i in plan[step]);model.train()
            for m in bn:m.eval()
            for name,param in model.named_parameters():param.requires_grad_(step>=200 or name.startswith('classifier.new.'))
            teacher.eval();f=factor(step+1,2000)
            for g in optimizer.param_groups:g['lr']=g['base_lr']*f
            optimizer.zero_grad(set_to_none=True);offset=0;ctc_value=0.;teacher_value=0.;n=len(batch['texts'])
            for start in range(0,n,4):
                micro={k:v[start:start+4] for k,v in batch.items() if k!='targets'};length=int(micro['lengths'].sum())
                micro['targets']=batch['targets'][offset:offset+length];offset+=length
                images=micro['images'].to(device);orientation=micro['orientation'].to(device)
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(images,orientation,micro['input_lengths'])
                with torch.autocast('cuda',enabled=False):ctc=criterion(logits.float().log_softmax(-1),micro['targets'].to(device),micro['input_lengths'],micro['lengths'])
                if kind in ('original','generated','long512','digits'):
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):reference=teacher(images,orientation,micro['input_lengths'])
                    with torch.autocast('cuda',enabled=False):kd=teacher_kl(logits,reference,micro['input_lengths'],2.)
                else:kd=logits.float().sum()*0
                loss=ctc+kd
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite CTC/teacher loss')
                weight=len(micro['texts'])/n;(loss*weight).backward();ctc_value+=float(ctc.detach())*weight;teacher_value+=float(kd.detach())*weight
            assert step<200 or all(m.weight.grad is not None and m.bias.grad is not None for m in bn)
            if step<200:
                assert all(torch.equal(v,source['model'][k]) for k,v in fused_state(model).items() if not k.startswith('classifier.'))
                assert torch.equal(model.classifier.old.weight.detach().cpu(),source['model']['classifier.weight'])
                assert torch.equal(model.classifier.old.bias.detach().cpu(),source['model']['classifier.bias'])
            nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();step+=1
            with torch.no_grad():
                for p,q in zip(ema.parameters(),model.parameters()):p.lerp_(q,.002)
            for k,i in plan[step-1]:
                r=mix.datasets[k].rows[i];streams[k]+=1;characters.update(c for c in r['text'] if c in sampling['planned_new_character_counts'])
                digit_encounters.update(c for c in r['text'] if c in '0123456789')
                if k=='digits':digit_anchors[r['target_digit']]+=1;digit_images[i]+=1
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
                        write_json(work/f'step_{step:04d}_{arm}_{precision}.json',dict(validation=r,retention=g,new_character_metrics=m,digit_metrics=digit_metrics(r),adaptation_metrics=adaptation_metrics(r),score=score(r)))
                        details[precision]=dict(retention=g,score=score(r),new_character_macro_recall=m['heldout']['macro_recall'],
                            yolo_holdout_cer=r['yolo_current_holdout']['0.0']['all']['cer'],long_heldoutfont_mean_cer=long_cer(r),digit_metrics=digit_metrics(r),adaptation_metrics=adaptation_metrics(r))
                    if all(d['retention']['eligible'] for d in details.values()) and details['fp16']['score']<best_score:
                        best_score=details['fp16']['score'];best_step=step;best_arm=arm;save_atomic(work/'best.pt',snapshot(candidate,arm))
                    entry['arms'][arm]=details
                check_bn();check_teacher();entry.update(best_update=best_step,best_arm=best_arm);history.append(entry);write_json(work/'history.json',history)
                with (work/'metrics.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(entry,ensure_ascii=False)+'\n')
                save_atomic(work/f'step_{step:04d}.pt',snapshot());print('EVAL',step,'best',best_step,best_arm,flush=True)
            if step%100==0:save_atomic(work/'last.pt',snapshot())
        check_bn();check_teacher();assert step==2000 and streams==sampling['streams'] and sum(streams.values())==64000
        assert dict(canvases)=={int(k):v for k,v in sampling['physical_width_presentations'].items()}
        assert len(yolo_seen)==1561 and sum(yolo_seen.values())==19200
        assert len(digit_images)==6000
        assert all(characters[c]==sampling['planned_new_character_counts'][c] and characters[c]>=25 for c in sampling['planned_new_character_counts'])
        new_delta=(model.classifier.new.weight.detach().cpu()-initial_state['classifier.weight'][-len(sampling['latin_added']):]).flatten(1).norm(dim=1)
        assert bool((new_delta>0).all())
        parentheses_seen=sum(any(c in mix.datasets[k].rows[i]['text'] for c in '()（）') for batch in plan for k,i in batch)
        assert parentheses_seen==sampling['parentheses_presentations'] and parentheses_seen>=3200
        assert sha(START)==source_hash and sha(REGISTRY)==registry_hash
        save_atomic(work/'final_raw.pt',snapshot());save_atomic(work/'final_ema.pt',snapshot(ema,'ema'))
        report=dict(completed=True,updates=2000,presentations=64000,stream_encounters=dict(streams),physical_width_encounters=dict(canvases),
            actual_yolo_fraction=.30,actual_long512_fraction=.05,extra_long_fraction=.15,digit_fraction=.15,latin_fraction=.15,dedicated_parentheses_fraction=.05,actual_physical512_fraction=canvases[512]/64000,
            yolo_base_encounters=dict(yolo_seen),new_character_encounters=dict(characters),minimum_new_character_encounters=min(characters.values()),latin_added=sampling['latin_added'],geometry=sampling['geometry'],
            actual_digit_token_encounters=dict(digit_encounters),actual_digit_anchor_encounters=dict(digit_anchors),unique_digit_training_images=len(digit_images),
            appended_row_delta_norms={c:float(d) for c,d in zip(sampling['latin_added'],new_delta)},all_appended_rows_updated=True,
            parentheses_presentations=parentheses_seen,new_rows_only_first200_verified=True,
            label_normalization='U+FF10..U+FF19 -> ASCII in every training stream; original label files untouched.',
            bn147_buffers_fixed=True,teacher_unchanged=True,source_and_registry_unchanged=True,best_update=best_step,best_arm=best_arm,best_score=best_score,production_promoted=False)
        write_json(work/'report.json',report)
        paired={}
        for precision in ('fp16','bf16'):
            initial=load(work/f'initial_{precision}.json')['validation']['yolo_current_holdout']['0.0']['predictions'];paired[precision]={}
            for arm in ('raw','ema'):
                final=load(work/f'step_2000_{arm}_{precision}.json')['validation']['yolo_current_holdout']['0.0']['predictions']
                effect=cluster_bootstrap(initial,final,extra['yolo_current_holdout'].rows);effect['v44_minus_source_V43_pp']=effect.pop('v38_minus_v36_pp');paired[precision][arm]=effect
        write_json(work/'paired_analysis.json',paired)
        lines=['# V44: Latin, parentheses and long-line adaptation','',json.dumps(report,ensure_ascii=False,indent=2),'',
            'Source V43finalraw2000. Append-only33missingLatinletters; allASCIIA-Z/a-z covered. '
            '30%YOLO/15%Latin/5%parentheses/15%digits/15%extra-long/5%oldlong/5%newchar/10%ordinary replay. '
            'First200updates only appended rows; then lower-LRfullfine-tune. Frozen147BNbuffers. '
            'Source fallback is not retention eligible. No promotion. See geometry_preflight.json and detailed adaptation_metrics in evaluation files.', '',
            '| Update | Arm | YOLOholdoutCER | Heldoutfont long512CER | Newcharrecall | FP16failures | BF16failures |','|---:|---|---:|---:|---:|---|---|']
        for h in history:
            for arm,d in h['arms'].items():lines.append(f"| {h['update']} | {arm} | {d['fp16']['yolo_holdout_cer']:.2%} | {d['fp16']['long_heldoutfont_mean_cer']:.2%} | {d['fp16']['new_character_macro_recall']:.2%} | {d['fp16']['retention']['failed']} | {d['bf16']['retention']['failed']} |")
        lines+=['','See initial/final fullreports for paired512versus384longprobes, cleanYOLOtrain/holdout and matchedclean/augmentedtrainingprobe. '
            'Combined adaptation experiment, not isolated ablation. V38 remains promoted; sourceV43/registry preserved. best.pt at update0 is not a gate-passing checkpoint.']
        (work/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');status(state='completed',**report)
    except Exception:status(state='failed',error=traceback.format_exc());raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=WORK);p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true');run(p.parse_args())
