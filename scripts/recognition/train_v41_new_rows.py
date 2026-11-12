"""V41: corrected initialization;2000updates of appended classifier rows only."""
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

from scripts.recognition.prepare_v41 import prepare,WORK,SEED,V40
from scripts.recognition.train_v40_new_characters import new_character_metrics
from scripts.recognition.train_v38_yolo25 import WORK as V38
from scripts.recognition.review_v38_yolo25 import cluster_bootstrap
from outputs.recognition.v37_long512 import make_views,evaluate_any
from outputs.recognition.v36_yolo_fullpasses import load,factor
from outputs.recognition.application_mixture_pilot import cpu_copy
from outputs.recognition.evaluate_bn_swap import gates as legacy_gates,V34
from outputs.recognition.evaluate_fp16_retention import evaluate_fp16
from src.recognition.current import current_checkpoint,REGISTRY
from src.recognition.retention import apply_retention_policy,POLICY_PATH
from src.recognition.vocabulary import ExtendedCTCHead,fused_state
from src.recognition.retention_distillation import old_class_mass_penalty
from src.recognition.model import MenuRecognizer
from src.recognition.font_training import evaluate
from src.recognition.data import collate
from src.recognition.checkpointing import save_atomic
from src.recognition.utils import sha,write_json,training_device,seed_all


def run(args):
    work=args.output;work.mkdir(parents=True,exist_ok=args.resume or args.prepare_only);started=time.time()
    def status(**kw):write_json(work/'status.json',dict(elapsed_seconds=time.time()-started,**kw))
    try:
        torch.set_num_threads(4);seed_all(SEED)
        source_path=current_checkpoint();source_hash=sha(source_path);registry_hash=sha(REGISTRY)
        source=torch.load(source_path,map_location='cpu',weights_only=False);assert source['updates']==686
        status(state='preparing');vocab,added,mix,plan,probe,hold,sampling=prepare(work,source)
        mapping=vocab['char_to_id'];old_classes=len(source['vocab']['char_to_id'])+1;cfg=dict(source['config'],classes=len(mapping)+1)
        config=dict(source=str(source_path),source_sha256=source_hash,updates=2000,presentations=64000,batch=32,microbatch=4,
            seed=SEED,trainable_parameters='classifier.new.weight and classifier.new.bias only for all2000updates',
            new_classifier_lr=1e-3,warmup_updates=200,min_lr_factor=.1,ema_decay=.998,
            initializer='Copy blank row weights; new bias=blank bias-8. Must pass zero-new-emission and zero-prediction-difference preflight before updates.',
            blank_margin=8,teacher_consistency_weight=1.,teacher='Frozen V38 old logits; full-softmax KL on valid frames of old-only labeled examples. No teacher suppression on new-character labels.',
            old_features_mode='eval throughout, no stochastic-layer changes',frozen_bn_buffers=147,pretrained_affine_and_classifier_frozen=True,
            retention_policy=load(POLICY_PATH)['id'],sampling_sha256=sha(work/'sampling.json'),vocab_sha256=sha(work/'vocab.json'),
            selection='Same V40score:0.5*(1-heldoutfont newchar macro recall)+0.5*full156YOLOholdoutCER, all revised gates in both precisions; no auto-promotion.')
        if (work/'run_config.json').exists():assert load(work/'run_config.json')==config,'Resume settings changed'
        write_json(work/'run_config.json',config)
        if args.prepare_only:status(state='prepared',updates=2000,minimum_new_character_encounters=sampling['minimum_total_new_character_encounters']);return
        device=training_device('cuda:0','RTX 3070')
        model=MenuRecognizer(old_classes,use_bilstm=True);model.load_state_dict(source['model'])
        model.classifier=ExtendedCTCHead(model.classifier,len(added),initialization='below_blank',blank_margin=8)
        model.requires_grad_(False);model.classifier.new.requires_grad_(True);model.vocabulary=mapping;model.to(device).eval();model.recurrent.lstm.flatten_parameters()
        ema=copy.deepcopy(model).eval();ema.requires_grad_(False);ema.recurrent.lstm.flatten_parameters()
        frozen={k:v.detach().clone() for k,v in model.state_dict().items() if not k.startswith('classifier.new.')}
        initial_new={k:v.detach().cpu().clone() for k,v in model.classifier.new.state_dict().items()}
        def check_frozen():
            for candidate in (model,ema):
                state=candidate.state_dict();assert all(torch.equal(state[k],v) for k,v in frozen.items())
            assert all(p.grad is None and not p.requires_grad for n,p in model.named_parameters() if not n.startswith('classifier.new.'))
        check_frozen();assert sum(k.endswith(('running_mean','running_var','num_batches_tracked')) for k in frozen)==147
        optimizer=torch.optim.AdamW([dict(params=[model.classifier.new.weight],lr=1e-3,base_lr=1e-3,weight_decay=.01),
                                     dict(params=[model.classifier.new.bias],lr=1e-3,base_lr=1e-3,weight_decay=0.)])
        criterion=nn.CTCLoss(blank=0,zero_infinity=False)
        _,views=make_views(cfg,mapping);views['yolo_current_train']=mix.datasets['yolo'];views['yolo_current_holdout']=hold;views['new_character_probe']=probe
        for ds in views.values():
            if hasattr(ds,'mapping'):ds.mapping=mapping
        baselines={'fp16':load('outputs/recognition/evaluation/v35_paired_short_inspection/fp16_full_source.json'),'bf16':load(V34/'initial.json')}
        def validation(candidate,precision):
            deployed=MenuRecognizer(len(mapping)+1,use_bilstm=True).to(device);deployed.load_state_dict(fused_state(candidate));deployed.vocabulary=mapping
            deployed.eval();deployed.recurrent.lstm.flatten_parameters();result={}
            for name,ds in views.items():
                fn=evaluate_fp16 if precision=='fp16' else evaluate
                result[name]=evaluate_any(deployed,ds,device,precision) if name.startswith(('yolo_current_','new_character_')) else fn(deployed,ds,device)
            del deployed;return result
        def score(r):return .5*(1-new_character_metrics(r,added)['heldout']['macro_recall'])+.5*r['yolo_current_holdout']['0.0']['all']['cer']
        step=0;best_step=0;best_arm='initial_extended_V38';best_score=None;history=[];encounters=Counter();streams=Counter()
        def snapshot(candidate=None,arm='raw'):
            return dict(model=fused_state(candidate or model),raw_training_model=cpu_copy(model.state_dict()),ema_training_model=cpu_copy(ema.state_dict()),
                optimizer=cpu_copy(optimizer.state_dict()),vocab=vocab,config=dict(cfg,v41_training=config),updates=step,
                checkpoint_arm=arm,best_update=best_step,best_arm=best_arm,best_score=best_score,
                new_character_encounters=dict(encounters),stream_encounters=dict(streams),rng_python=random.getstate(),rng_numpy=np.random.get_state(),
                rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all())
        if not (work/'source.zip').exists():
            with zipfile.ZipFile(work/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in [Path(__file__),Path('scripts/recognition/prepare_v41.py'),*Path('src/recognition').glob('*.py')]:z.write(p,p.as_posix())
        if args.resume and (work/'last.pt').exists():
            saved=torch.load(work/'last.pt',map_location='cpu',weights_only=False);assert saved['vocab']==vocab
            model.load_state_dict(saved['raw_training_model']);ema.load_state_dict(saved['ema_training_model']);optimizer.load_state_dict(saved['optimizer'])
            step=saved['updates'];best_step=saved['best_update'];best_arm=saved['best_arm'];best_score=saved['best_score'];history=load(work/'history.json')
            encounters.update(saved['new_character_encounters']);streams.update(saved['stream_encounters'])
            random.setstate(saved['rng_python']);np.random.set_state(saved['rng_numpy']);torch.set_rng_state(saved['rng_torch']);torch.cuda.set_rng_state_all(saved['rng_cuda'])
            assert load(work/'initial_parity.json')['passed']
        else:
            initial={};parity={};new_emissions={}
            for precision in ('fp16','bf16'):
                status(state='initial_parity_evaluation',precision=precision);r=validation(model,precision);initial[precision]=r
                reference=load(V38/f'step_0686_raw_{precision}.json')['validation']
                reference={k:v for k,v in reference.items() if not k.startswith('yolo_current_')}
                reference.update(load(V40/f'review_original_v38_{precision}.json'))
                parity[precision]={}
                for name,old in reference.items():
                    a=r[name]['0.0']['predictions'];b=old['0.0']['predictions'];assert len(a)==len(b)
                    parity[precision][name]=sum(x['prediction']!=y['prediction'] for x,y in zip(a,b))
                new_emissions[precision]=sum(any(c in added for c in row['prediction']) for view in r.values() for row in view['0.0']['predictions'])
                g=apply_retention_policy(legacy_gates(r,baselines[precision]))
                write_json(work/f'initial_{precision}.json',dict(validation=r,retention=g,new_character_metrics=new_character_metrics(r,added)))
                write_json(work/'initial_parity.json',dict(passed=False,zero_updates=True,prediction_disagreements=parity,new_token_emission_lines=new_emissions))
                assert not any(parity[precision].values()),'Initialization changed source predictions; do not train'
                assert new_emissions[precision]==0 and g['eligible'],'Initialization must preserve source retention and emit no new characters'
            write_json(work/'initial_parity.json',dict(passed=True,zero_updates=True,prediction_disagreements=parity,new_token_emission_lines=new_emissions,
                old_ids_preserved=True,blank_id=0,old_parameters_bit_identical=True,new_logits_below_blank_by=8))
            best_score=score(initial['fp16']);save_atomic(work/'best.pt',snapshot(model,best_arm));write_json(work/'history.json',history)
        check_frozen();total_ctc=0.;total_teacher=0.;batches_seen=0
        loader=DataLoader(mix,batch_sampler=plan[step:],num_workers=0,collate_fn=collate,generator=torch.Generator().manual_seed(SEED))
        for batch in loader:
            model.eval();f=factor(step+1,2000)
            for group in optimizer.param_groups:group['lr']=group['base_lr']*f
            optimizer.zero_grad(set_to_none=True);target_offset=0;ctc_value=0.;teacher_value=0.;n=len(batch['texts'])
            for start in range(0,n,4):
                micro={k:v[start:start+4] for k,v in batch.items() if k!='targets'};count=int(micro['lengths'].sum())
                micro['targets']=batch['targets'][target_offset:target_offset+count];target_offset+=count
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(micro['images'].to(device),micro['orientation'].to(device),micro['input_lengths'])
                with torch.autocast('cuda',enabled=False):
                    ctc=criterion(logits.float().log_softmax(-1),micro['targets'].to(device),micro['input_lengths'],micro['lengths'])
                    teacher=old_class_mass_penalty(logits,old_classes,micro['input_lengths'],[not any(c in added for c in t) for t in micro['texts']])
                    loss=ctc+teacher
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                weight=len(micro['texts'])/n;(loss*weight).backward();ctc_value+=float(ctc.detach())*weight;teacher_value+=float(teacher.detach())*weight
            nn.utils.clip_grad_norm_(model.classifier.new.parameters(),5,error_if_nonfinite=True);optimizer.step();step+=1
            with torch.no_grad():
                for p,q in zip(ema.classifier.new.parameters(),model.classifier.new.parameters()):p.lerp_(q,.002)
            streams.update(k for k,i in plan[step-1])
            for text in batch['texts']:encounters.update(c for c in text if c in added)
            total_ctc+=ctc_value;total_teacher+=teacher_value;batches_seen+=1
            if step==1 or step%100==0:
                check_frozen();status(state='training',update=step,total_updates=2000,mean_ctc=total_ctc/batches_seen,mean_teacher=total_teacher/batches_seen,
                    minimum_new_character_encounters=min(encounters[c] for c in added),best_update=best_step)
            if step%500==0:
                entry=dict(update=step,lr_factor=f,mean_ctc=total_ctc/batches_seen,mean_teacher=total_teacher/batches_seen,arms={})
                for arm,candidate in (('raw',model),('ema',ema)):
                    details={}
                    for precision in ('fp16','bf16'):
                        status(state='evaluating',update=step,arm=arm,precision=precision);r=validation(candidate,precision)
                        g=apply_retention_policy(legacy_gates(r,baselines[precision]));metrics=new_character_metrics(r,added)
                        write_json(work/f'step_{step:04d}_{arm}_{precision}.json',dict(validation=r,retention=g,new_character_metrics=metrics,score=score(r)))
                        details[precision]=dict(retention=g,score=score(r),new_character_macro_recall=metrics['heldout']['macro_recall'],
                            yolo_train_cer=r['yolo_current_train']['0.0']['all']['cer'],yolo_holdout_cer=r['yolo_current_holdout']['0.0']['all']['cer'])
                    if all(d['retention']['eligible'] for d in details.values()) and details['fp16']['score']<best_score:
                        best_score=details['fp16']['score'];best_step=step;best_arm=arm;save_atomic(work/'best.pt',snapshot(candidate,arm))
                    entry['arms'][arm]=details
                check_frozen();entry.update(best_update=best_step,best_arm=best_arm);history.append(entry);write_json(work/'history.json',history)
                with (work/'metrics.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(entry,ensure_ascii=False)+'\n')
                save_atomic(work/f'step_{step:04d}.pt',snapshot());print('EVAL',step,'best',best_step,best_arm,flush=True)
            if step%100==0:save_atomic(work/'last.pt',snapshot())
        check_frozen();assert step==2000 and streams==sampling['streams'] and sum(streams.values())==64000
        assert all(encounters[c]==sampling['total_new_character_counts'][c] and encounters[c]>=100 for c in added)
        delta=(model.classifier.new.weight.detach().cpu()-initial_new['weight']).flatten(1).norm(dim=1).tolist()
        assert all(v>0 for v in delta)
        assert sha(source_path)==source_hash and sha(REGISTRY)==registry_hash
        save_atomic(work/'final_raw.pt',snapshot());save_atomic(work/'final_ema.pt',snapshot(ema,'ema'))
        report=dict(completed=True,updates=2000,presentations=64000,stream_encounters=dict(streams),
            new_character_encounters={c:encounters[c] for c in added},minimum_new_character_encounters=min(encounters[c] for c in added),
            all57_new_rows_changed_from_blank_initialization=True,new_row_weight_delta_norms={c:delta[i] for i,c in enumerate(added)},
            all_pretrained_parameters_and_buffers_bit_identical=True,bn147_buffers_fixed=True,source_and_registry_unchanged=True,
            best_update=best_step,best_arm=best_arm,best_score=best_score,production_promoted=False)
        write_json(work/'report.json',report)
        write_json(work/'new_character_coverage.json',[dict(character=c,codepoint=f'U+{ord(c):04X}',planned=sampling['total_new_character_counts'][c],
            actual=encounters[c],row_weight_delta=delta[i]) for i,c in enumerate(added)])
        paired={}
        for precision in ('fp16','bf16'):
            original=load(work/f'initial_{precision}.json')['validation']['yolo_current_holdout']['0.0']['predictions'];paired[precision]={}
            for arm in ('raw','ema'):
                final=load(work/f'step_2000_{arm}_{precision}.json')['validation']['yolo_current_holdout']['0.0']['predictions']
                effect=cluster_bootstrap(original,final,hold.rows);effect['v41_minus_V38_pp']=effect.pop('v38_minus_v36_pp');paired[precision][arm]=effect
        write_json(work/'paired_analysis.json',paired)
        lines=['# V41: new rows only /2000updates','',json.dumps(report,ensure_ascii=False,indent=2),'',
            'Corrected blank-minus8initialization passed strict zero-update prediction parity. Old features/rows/BNaffine and buffers stayed bit-identical for entire run. '
            '25%controlledone-newchar/25%YOLO/50%synthetic replay. Full-softmax teacher consistency only on old-only labels and valid frames. '
            'Old IDs/blank0unchanged; no automatic promotion.', '',
            '| Update | Arm | Heldout-font new-char recall | Full156YOLOholdout CER | FP16 failures | BF16 failures |','|---:|---|---:|---:|---|---|']
        for h in history:
            for arm,d in h['arms'].items():lines.append(f"| {h['update']} | {arm} | {d['fp16']['new_character_macro_recall']:.2%} | {d['fp16']['yolo_holdout_cer']:.2%} | {d['fp16']['retention']['failed']} | {d['bf16']['retention']['failed']} |")
        lines+=['','Compare to V40onidentical684newfont probes and156YOLOholdout; this changes initialization,freezing,replayandcontexts together, so it tests the revised package, not isolated causal effects. '
            'Best is retention-selected; finalraw/EMA kept separately. Promoted V38 unchanged.']
        (work/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');status(state='completed',**report)
    except Exception:status(state='failed',error=traceback.format_exc());raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=WORK);p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true');run(p.parse_args())
