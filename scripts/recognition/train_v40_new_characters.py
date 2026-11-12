"""2000 updates: append57CTCcharacters, explicitly train each >=100 encounters."""
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

from scripts.recognition.prepare_v40 import prepare,WORK,SEED,INPUT
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
from src.recognition.model import MenuRecognizer
from src.recognition.font_training import evaluate
from src.recognition.data import collate
from src.recognition.metrics import align_errors
from src.recognition.checkpointing import save_atomic
from src.recognition.utils import sha,write_json,training_device,loss_for,seed_all


def new_character_metrics(report,added):
    predictions=report['new_character_probe']['0.0']['predictions'];groups={}
    for cohort in ('all','known','heldout'):
        chosen=[r for r in predictions if cohort=='all' or r['cohort']==cohort]
        refs=Counter();correct=Counter();hyp=Counter()
        for r in chosen:
            refs.update(c for c in r['truth'] if c in added);hyp.update(c for c in r['prediction'] if c in added)
            correct.update(op['truth'] for op in align_errors(r['truth'],r['prediction']) if op['type']=='match' and op['truth'] in added)
        per={c:dict(reference=refs[c],correct=correct[c],predicted=hyp[c],recall=correct[c]/refs[c] if refs[c] else None) for c in added}
        assert all(refs[c]>0 for c in added)
        groups[cohort]=dict(macro_recall=sum(v['recall'] for v in per.values())/len(added),per_character=per,
            characters_with_no_correct_predictions=[c for c in added if not correct[c]])
    return groups


def make_optimizer(model):
    buckets={}
    for name,p in model.named_parameters():
        role='new_classifier' if name.startswith('classifier.new.') else 'encoder' if name.startswith('encoder.') else 'head'
        decay=p.ndim>1 and not name.endswith('.bias');buckets.setdefault((role,decay),[]).append(p)
    rates={'encoder':5e-7,'head':3e-6,'new_classifier':1e-3}
    return torch.optim.AdamW([dict(params=p,role=role,base_lr=rates[role],lr=rates[role],weight_decay=.01 if decay else 0.) for (role,decay),p in buckets.items()])


def run(args):
    work=args.output;work.mkdir(parents=True,exist_ok=args.resume or args.prepare_only);started=time.time()
    def status(**kw):write_json(work/'status.json',dict(elapsed_seconds=time.time()-started,**kw))
    try:
        torch.set_num_threads(4);seed_all(SEED)
        source_path=current_checkpoint();source_hash=sha(source_path);registry_hash=sha(REGISTRY)
        source=torch.load(source_path,map_location='cpu',weights_only=False);assert source['updates']==686
        status(state='preparing');vocab,added,mix,plan,probe,hold,sampling=prepare(work,source,status)
        mapping=vocab['char_to_id'];old_count=len(source['vocab']['char_to_id']);cfg=dict(source['config'],classes=len(mapping)+1)
        config=dict(source=str(source_path),source_sha256=source_hash,updates=2000,presentations=64000,
            pretrained_characters=old_count,added_characters=added,total_characters=len(mapping),classes=len(mapping)+1,
            batch=32,microbatch=4,seed=SEED,new_classifier_lr=1e-3,encoder_lr=5e-7,head_lr=3e-6,
            warmup_updates=200,new_classifier_only_updates=200,min_lr_factor=.1,ema_decay=.998,
            frozen_bn_buffers=147,bn_affine_trainable_after_update200=True,retention_policy=load(POLICY_PATH)['id'],
            sampling_sha256=sha(work/'sampling.json'),vocabulary_sha256=sha(work/'vocab.json'),
            selection='Minimize0.5*(1-heldout-font new-character macro recall)+0.5*full-YOLOholdoutCER subject to revised17gates in both precisions; retain initialized extended V38 if none qualifies.',
            export='Concatenate old/new classifier rows into standard MenuRecognizer classifier tensors; preserve all old IDs and blank0.')
        if (work/'run_config.json').exists():assert load(work/'run_config.json')==config,'Run configuration changed'
        write_json(work/'run_config.json',config)
        if args.prepare_only:status(state='prepared',updates=2000,minimum_new_character_encounters=sampling['minimum_new_character_encounters']);return
        device=training_device('cuda:0','RTX 3070')
        model=MenuRecognizer(old_count+1,use_bilstm=True);model.load_state_dict(source['model'])
        model.classifier=ExtendedCTCHead(model.classifier,len(added));model.vocabulary=mapping;model.to(device)
        initial_state=fused_state(model)
        for k,v in source['model'].items():
            assert torch.equal(initial_state[k][:old_count+1] if k.startswith('classifier.') else initial_state[k],v)
        ema=copy.deepcopy(model).eval();ema.requires_grad_(False)
        model.recurrent.lstm.flatten_parameters();ema.recurrent.lstm.flatten_parameters()
        optimizer=make_optimizer(model);criterion=nn.CTCLoss(blank=0,zero_infinity=False)
        bn=[m for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)]
        buffers={k:v for k,v in source['model'].items() if k.endswith(('running_mean','running_var','num_batches_tracked'))};assert len(buffers)==147
        def check_bn():
            for m in (model,ema):assert all(torch.equal(m.state_dict()[k].cpu(),v) for k,v in buffers.items())
        _,views=make_views(cfg,mapping)
        views['yolo_current_train']=mix.datasets['yolo'];views['yolo_current_holdout']=hold;views['new_character_probe']=probe
        for ds in views.values():
            if hasattr(ds,'mapping'):ds.mapping=mapping
        baselines={'fp16':load('outputs/recognition/evaluation/v35_paired_short_inspection/fp16_full_source.json'),'bf16':load(V34/'initial.json')}
        def validation(candidate,precision):
            # Validate the fused standard architecture that saved checkpoints use.
            fused=MenuRecognizer(len(mapping)+1,use_bilstm=True).to(device);fused.vocabulary=mapping
            fused.load_state_dict(fused_state(candidate));fused.eval();fused.recurrent.lstm.flatten_parameters()
            results={}
            for name,ds in views.items():
                fn=evaluate_fp16 if precision=='fp16' else evaluate
                results[name]=evaluate_any(fused,ds,device,precision) if name.startswith(('yolo_current_','new_character_')) else fn(fused,ds,device)
            del fused
            return results
        def score(r):return .5*(1-new_character_metrics(r,added)['heldout']['macro_recall'])+.5*r['yolo_current_holdout']['0.0']['all']['cer']
        step=0;best_step=0;best_arm='initial_extended_V38';best_score=None;history=[];encounters=Counter();streams=Counter()
        def snapshot(candidate=None,arm='raw'):
            return dict(model=fused_state(candidate or model),raw_training_model=cpu_copy(model.state_dict()),ema_training_model=cpu_copy(ema.state_dict()),
                optimizer=cpu_copy(optimizer.state_dict()),vocab=vocab,config=dict(cfg,v40_training=config),updates=step,
                best_update=best_step,best_arm=best_arm,best_score=best_score,checkpoint_arm=arm,
                new_character_encounters=dict(encounters),stream_encounters=dict(streams),
                rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all())
        if not (work/'source.zip').exists():
            with zipfile.ZipFile(work/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in [Path(__file__),Path('scripts/recognition/prepare_v40.py'),*Path('src/recognition').glob('*.py')]:z.write(p,p.as_posix())
        if args.resume and (work/'last.pt').exists():
            saved=torch.load(work/'last.pt',map_location='cpu',weights_only=False);assert saved['vocab']==vocab
            model.load_state_dict(saved['raw_training_model']);ema.load_state_dict(saved['ema_training_model']);optimizer.load_state_dict(saved['optimizer'])
            step=saved['updates'];best_step=saved['best_update'];best_arm=saved['best_arm'];best_score=saved['best_score']
            encounters.update(saved['new_character_encounters']);streams.update(saved['stream_encounters']);history=load(work/'history.json')
            random.setstate(saved['rng_python']);np.random.set_state(saved['rng_numpy']);torch.set_rng_state(saved['rng_torch']);torch.cuda.set_rng_state_all(saved['rng_cuda'])
        else:
            initial={};parity={}
            for precision in ('fp16','bf16'):
                status(state='initial_evaluation',precision=precision);initial[precision]=validation(model,precision)
                old=load(V38/f'step_0686_raw_{precision}.json')['validation'];parity[precision]={}
                # CurrentYOLOsets are expanded to include formerly OOV labels.
                for name in old:
                    if name.startswith('yolo_current_'):continue
                    a=initial[precision][name]['0.0']['predictions'];b=old[name]['0.0']['predictions'];assert len(a)==len(b)
                    parity[precision][name]=sum(x['prediction']!=y['prediction'] for x,y in zip(a,b))
                write_json(work/f'initial_{precision}.json',dict(validation=initial[precision],new_character_metrics=new_character_metrics(initial[precision],added)))
            write_json(work/'initial_parity.json',dict(old_classifier_rows_bit_identical=True,old_ids_preserved=True,
                prediction_disagreements=parity,note='Fused classifier output dimension changes; numerical operation order may differ. New logits initiallyconstant-12.'))
            best_score=score(initial['fp16']);save_atomic(work/'best.pt',snapshot(model,best_arm))
            write_json(work/'history.json',history)
        check_bn();loss_sum=0.;loss_n=0
        loader=DataLoader(mix,batch_sampler=plan[step:],num_workers=0,collate_fn=collate,generator=torch.Generator().manual_seed(SEED))
        for batch in loader:
            for name,p in model.named_parameters():p.requires_grad_(step>=200 or name.startswith('classifier.new.'))
            model.train()
            for m in bn:m.eval()
            f=factor(step+1,2000)
            for g in optimizer.param_groups:g['lr']=g['base_lr']*f
            optimizer.zero_grad(set_to_none=True);target_offset=0;batch_loss=0.;n=len(batch['texts'])
            for start in range(0,n,4):
                micro={k:v[start:start+4] for k,v in batch.items() if k!='targets'};count=int(micro['lengths'].sum())
                micro['targets']=batch['targets'][target_offset:target_offset+count];target_offset+=count
                loss=loss_for(model,micro,device,criterion);weight=len(micro['texts'])/n
                (loss*weight).backward();batch_loss+=float(loss.detach())*weight
            nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();step+=1
            with torch.no_grad():
                for p,q in zip(ema.parameters(),model.parameters()):p.lerp_(q,.002)
            for text in batch['texts']:encounters.update(c for c in text if c in added)
            streams.update(k for k,i in plan[step-1]);loss_sum+=batch_loss;loss_n+=1
            if step==1 or step%100==0:
                check_bn();status(state='training',update=step,total_updates=2000,new_rows_only=step<=200,
                    mean_loss=loss_sum/loss_n,minimum_new_character_encounters=min(encounters[c] for c in added),best_update=best_step)
            if step==200:
                frozen=fused_state(model)
                assert all(torch.equal(frozen[k][:old_count+1] if k.startswith('classifier.') else frozen[k],v) for k,v in source['model'].items())
                write_json(work/'new_row_warmup_audit.json',dict(update=200,all_pretrained_parameters_and_buffers_bit_identical=True,
                    new_rows_with_nonzero_weights=int((model.classifier.new.weight.detach().flatten(1).abs().sum(1)>0).sum())))
            if step%500==0:
                entry=dict(update=step,lr_factor=f,mean_loss=loss_sum/loss_n,arms={})
                for arm,candidate in (('raw',model),('ema',ema)):
                    details={}
                    for precision in ('fp16','bf16'):
                        status(state='evaluating',update=step,arm=arm,precision=precision)
                        r=validation(candidate,precision);g=apply_retention_policy(legacy_gates(r,baselines[precision]));metrics=new_character_metrics(r,added)
                        result=dict(validation=r,retention=g,new_character_metrics=metrics,score=score(r))
                        write_json(work/f'step_{step:04d}_{arm}_{precision}.json',result)
                        details[precision]=dict(retention=g,score=result['score'],new_character_macro_recall=metrics['heldout']['macro_recall'],
                            yolo_train_cer=r['yolo_current_train']['0.0']['all']['cer'],yolo_holdout_cer=r['yolo_current_holdout']['0.0']['all']['cer'])
                    if all(d['retention']['eligible'] for d in details.values()) and details['fp16']['score']<best_score:
                        best_score=details['fp16']['score'];best_step=step;best_arm=arm;save_atomic(work/'best.pt',snapshot(candidate,arm))
                    entry['arms'][arm]=details
                check_bn();entry.update(best_update=best_step,best_arm=best_arm);history.append(entry);write_json(work/'history.json',history)
                with (work/'metrics.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(entry,ensure_ascii=False)+'\n')
                save_atomic(work/f'step_{step:04d}.pt',snapshot());print('EVAL',step,'best',best_step,best_arm,flush=True)
            if step%100==0:save_atomic(work/'last.pt',snapshot())
        check_bn();assert step==2000 and sum(streams.values())==64000 and streams==sampling['streams']
        assert all(encounters[c]==sampling['planned_new_character_encounters'][c] and encounters[c]>=100 for c in added)
        row_norms=model.classifier.new.weight.detach().float().flatten(1).norm(dim=1).cpu().tolist()
        assert all(v>0 for v in row_norms),'Every appended classifier row must have learned nonzero weights'
        assert sha(source_path)==source_hash and sha(REGISTRY)==registry_hash
        save_atomic(work/'final_raw.pt',snapshot());save_atomic(work/'final_ema.pt',snapshot(ema,'ema'))
        report=dict(completed=True,updates=2000,presentations=64000,old_characters=1722,added_characters=57,total_characters=1779,
            stream_encounters=dict(streams),new_character_encounters={c:encounters[c] for c in added},minimum_new_character_encounters=min(encounters[c] for c in added),
            all_new_characters_trained_at_least100times=True,bn147_buffers_fixed=True,source_and_registry_unchanged=True,
            new_classifier_row_norms={c:row_norms[i] for i,c in enumerate(added)},all57_new_classifier_rows_updated=True,
            best_update=best_step,best_arm=best_arm,best_score=best_score,production_promoted=False)
        write_json(work/'report.json',report)
        write_json(work/'new_character_coverage.json',[dict(character=c,codepoint=f'U+{ord(c):04X}',
            planned=sampling['planned_new_character_encounters'][c],actual=encounters[c],new_row_weight_norm=row_norms[i],
            supported_training_fonts=len(sampling['font_support'][c]['known'])) for i,c in enumerate(added)])
        paired={}
        for precision in ('fp16','bf16'):
            initial=load(work/f'initial_{precision}.json')['validation']['yolo_current_holdout']['0.0']['predictions']
            paired[precision]={}
            for arm in ('raw','ema'):
                final=load(work/f'step_2000_{arm}_{precision}.json')['validation']['yolo_current_holdout']['0.0']['predictions']
                result=cluster_bootstrap(initial,final,hold.rows);result['v40_minus_initialized_v38_pp']=result.pop('v38_minus_v36_pp');paired[precision][arm]=result
        write_json(work/'paired_analysis.json',paired)
        lines=['# V40:57newCTCcharacters /2000updates','',json.dumps(report,ensure_ascii=False,indent=2),'',
            'Append-only vocab1779characters+blank0; old IDs/rows copied exactly. First200updates train only57new classifier rows; then full fine-tuning with frozen147BNbuffers. '
            '50%targeted/25%YOLO/25%oldsynthetic, same native buckets. Non-OK YOLO images never trained. All57newcharacters have >=100actual supervised encounters. '
            'Spaces are internal separators, not blank-only crops. Old V38/source vocabulary remain intact; expanded version is vocab_v40.json.', '',
            '| Update | Arm | Heldout-font new-char recall | FullYOLOholdout CER | FP16 failures | BF16 failures |','|---:|---|---:|---:|---|---|']
        for h in history:
            for arm,d in h['arms'].items():lines.append(f"| {h['update']} | {arm} | {d['fp16']['new_character_macro_recall']:.2%} | {d['fp16']['yolo_holdout_cer']:.2%} | {d['fp16']['retention']['failed']} | {d['bf16']['retention']['failed']} |")
        lines+=['','Compare against initial evaluation on the SAME expanded156-crop YOLOholdout,not V38s old135-crop subset. '
            'Newcharacterprobe has684independent renders including heldoutfont families; not unseen-label/character evidence. '
            'best.pt is retention-selected; final_raw/final_ema preserve actual trained weights; no automatic production promotion.']
        (work/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');status(state='completed',**report)
    except Exception:status(state='failed',error=traceback.format_exc());raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=WORK);p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true');run(p.parse_args())
