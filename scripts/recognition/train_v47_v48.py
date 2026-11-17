"""Independent tiny Latin fitting and bracket-only adaptation experiments."""
import argparse
from collections import Counter
import copy,json,math,random,string,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.recognition.prepare_v47_v48 import prepare,SEED,START
from scripts.recognition.prepare_v44 import expanded_state
from scripts.recognition.prepare_v43 import normalize_dataset,digit_metrics
from scripts.recognition.train_v45_latin_head import rows,V40
from scripts.recognition.train_v38_yolo25 import BucketCrops
from outputs.recognition.v37_long512 import make_views,evaluate_any
from outputs.recognition.v36_yolo_fullpasses import load
from outputs.recognition.application_mixture_pilot import cpu_copy
from outputs.recognition.evaluate_bn_swap import gates as legacy_gates,V34
from outputs.recognition.evaluate_fp16_retention import evaluate_fp16
from src.recognition.font_training import evaluate
from src.recognition.metrics import align_errors,summarise
from src.recognition.model import MenuRecognizer
from src.recognition.partial_classifier import SelectedCTCRows,GroupedCTCRows,export_state
from src.recognition.retention_distillation import teacher_kl
from src.recognition.retention import apply_retention_policy
from src.recognition.data import collate
from src.recognition.checkpointing import save_atomic
from src.recognition.current import REGISTRY,current_checkpoint
from src.recognition.utils import sha,write_json,training_device,seed_all
from src.recognition.canonical_ctc import CanonicalTrainingLabels,canonical_ctc_inputs,canonical_teacher_kl,evaluate_parenthesis_modes
from src.recognition.label_normalization import normalize_parentheses


def character_metrics(report,name,characters):
    refs=Counter();hits=Counter()
    for r in report[name]['0.0']['predictions']:
        for op in align_errors(r['truth'],r['prediction']):
            if op['truth'] and op['truth'] in characters:
                refs[op['truth']]+=1;hits[op['truth']]+=op['type']=='match'
    per={c:dict(reference=refs[c],correct=hits[c],recall=hits[c]/refs[c] if refs[c] else None) for c in characters}
    values=[r['recall'] for r in per.values() if r['recall'] is not None]
    return dict(macro_recall=sum(values)/len(values) if values else None,per_character=per)


def run(args):
    selective=args.arm=='selective';latin=args.arm in ('latin','selective');updates=1000 if latin and not selective else 2000
    normalized=args.normalize_parentheses
    recurrent_trainable=not selective or args.unfreeze_bilstm
    if normalized or args.unfreeze_bilstm:assert selective,'V50flagsarevalidonlywithselectivearm'
    if normalized:assert args.output is not None,'Normalizedmatchedarmsmustuseexplicitnewoutputdirectories'
    if args.unfreeze_bilstm:assert normalized,'Useanexplicitnormalizedmatchedarmforrecurrentunfreezing'
    work=args.output or Path('outputs/recognition/pilots')/('v49_selective_head_2000' if selective else 'v47_clean_latin_fit_1000' if latin else 'v48_parentheses_2000')
    work.mkdir(parents=True,exist_ok=args.resume or args.smoke);started=time.time()
    def status(**kw):write_json(work/'status.json',dict(elapsed_seconds=time.time()-started,**kw))
    try:
        torch.set_num_threads(4);seed_all(SEED+int(not latin));registry_hash=sha(REGISTRY);active_hash=sha(current_checkpoint())
        source_hash=sha(START);source=torch.load(START,map_location='cpu',weights_only=False)
        assert load(REGISTRY)['version']=='V43' and source['updates']==2000 and source['checkpoint_arm']=='raw'
        prepare_fn=prepare
        if selective:
            from scripts.recognition.prepare_v49 import prepare as prepare_fn
        status(state='preparing');vocab,added,mix,plan,extra,sampling=prepare_fn(work,source,args.arm)
        if normalized:
            reference=Path('outputs/recognition/pilots/v49_selective_head_2000/batches.json')
            assert sha(work/'batches.json')==sha(reference),'MatchedV49presentationplanmustbeidentical'
        mapping=vocab['char_to_id'];cfg=dict(source['config'],classes=len(mapping)+1)
        rates=dict(latin=1e-3,digits_brackets=1e-5) if selective else dict(recurrent=1e-4 if latin else 1e-5,classifier=1e-3 if latin else 3e-6)
        config=dict(arm=args.arm,source=str(START),source_sha256=source_hash,updates=updates,presentations=updates*32,
            batch=32,microbatch=4,rates=rates,warmup=50 if latin else 100,min_lr_factor=.1,
            trainable='ResidualBiLSTM+all52Latinrows; allnonLatinrowsfrozen' if latin else 'ResidualBiLSTM+fullclassifier',
            frozen='CNN/projection/orientation/temporal/allBNaffine/147buffers; no additional unfreezing',
            teacher_kl_weight=0 if latin else 1.,teacher_temperature=2.,ema_decay=None if latin else .998,
            interpretation='Unregularized tiny-set fitting diagnostic; no replay or retention protection; NEVER autopromote' if latin else 'Independent V43 parentheses adaptation,30target/45synthetic/25YOLO; fullsoftmaxKL onreplayonly',
            sampling_sha256=sha(work/'sampling.json'))
        if selective:
            config.update(warmup=100,trainable='Only52LatinrowsLR1e-3and10digit+4bracketrowsLR1e-5',
                frozen='AllsharedfeaturesINCLUDINGBiLSTM,allremainingclassifierrows,blank,BNaffineand147buffersbit-identical',
                teacher_kl_weight=1.,ema_decay=.998,
                interpretation='Controlledhead-onlypilot;fullsoftmaxKLweight1T2onnonLatin-labeledreplay;noKLondedicatedtargets;source-relativegatedselection;noautopromotion')
        if normalized:
            if args.unfreeze_bilstm:rates['recurrent']=1e-5
            config.update(parentheses_normalized=True,matched_v49_plan_sha256=sha(reference),
                recurrent_trainable=args.unfreeze_bilstm,
                trainable='V49same66rowgroups'+(';residualBiLSTM1e-5' if args.unfreeze_bilstm else ';BiLSTMfrozencontrol'),
                frozen='CNN/projection/orientation/temporal/BNaffine147buffers/allunselectedclassifierrows'+('' if args.unfreeze_bilstm else '/BiLSTM'),
                canonical_ctc='Maplabels（→(and）→);logaddexpwidth-pairlogitsbeforeCTCloss/greedycollapse;allIDsretained',
                teacher='SameV43fullsoftmaxvalidnonLatinreplay;poolbothstudent/teacherwidth-pairmassatT2;nootherclassmassdiscarded',
                selection='SameV49rawretention/YOLO/digit/historical/shortguards;canonicalCERweightedranking. Literal/textnormalized/pooledreportsseparate.')
        if (work/'run_config.json').exists():assert load(work/'run_config.json')==config
        write_json(work/'run_config.json',config)
        device=training_device('cuda:0','RTX 3070');initial=expanded_state(source,vocab) if latin else source['model']
        model=MenuRecognizer(len(mapping)+1,use_bilstm=True);model.load_state_dict(initial);model.requires_grad_(False)
        if selective:model.classifier=GroupedCTCRows(model.classifier,{'latin':[mapping[c] for c in string.ascii_letters],'digits_brackets':[mapping[c] for c in string.digits+'()（）']})
        elif latin:model.classifier=SelectedCTCRows(model.classifier,[mapping[c] for c in string.ascii_letters])
        model.recurrent.requires_grad_(recurrent_trainable)
        if selective:model.classifier.heads.requires_grad_(True)
        elif latin:model.classifier.rows.requires_grad_(True)
        else:model.classifier.requires_grad_(True)
        model.to(device).eval();model.vocabulary=mapping;model.recurrent.lstm.flatten_parameters()
        ema=None if latin and not selective else copy.deepcopy(model).eval().requires_grad_(False)
        if ema is not None:ema.recurrent.lstm.flatten_parameters()
        teacher=None
        if not latin or selective:
            teacher=MenuRecognizer(len(mapping)+1,use_bilstm=True).to(device).eval().requires_grad_(False)
            teacher.load_state_dict(initial);teacher.recurrent.lstm.flatten_parameters()
        frozen={k:v.clone() for k,v in model.state_dict().items() if not k.startswith(('classifier.',) if selective and not recurrent_trainable else ('recurrent.','classifier.'))}
        def audit():
            for candidate in [model]+([ema] if ema is not None else []):
                assert all(torch.equal(candidate.state_dict()[k],v) for k,v in frozen.items())
                assert all(not m.training for m in candidate.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm))
            assert sum(k.endswith(('running_mean','running_var','num_batches_tracked')) for k in frozen)==147
            if latin:
                for candidate in [model]+([ema] if ema is not None else []):
                    assert torch.equal(candidate.classifier.base.weight.detach().cpu(),initial['classifier.weight'])
                    assert torch.equal(candidate.classifier.base.bias.detach().cpu(),initial['classifier.bias'])
            if selective:
                assert all(not p.requires_grad and p.grad is None for n,p in model.named_parameters() if not n.startswith(('classifier.heads.','recurrent.') if recurrent_trainable else ('classifier.heads.',)))
        groups=[];classifier=model.classifier.heads if selective else model.classifier.rows if latin else model.classifier
        modules=list(classifier.items()) if selective else [('recurrent',model.recurrent),('classifier',classifier)]
        if selective and recurrent_trainable:modules.append(('recurrent',model.recurrent))
        for role,module in modules:
            for decay in (True,False):
                params=[p for p in module.parameters() if (p.ndim>1)==decay]
                if params:groups.append(dict(params=params,lr=rates[role],base_lr=rates[role],role=role,weight_decay=0. if latin or not decay else .01))
        optimizer=torch.optim.AdamW(groups);criterion=nn.CTCLoss(blank=0,zero_infinity=False)
        def train_batch(batch,step,kind):
            model.recurrent.train(recurrent_trainable);warm=100 if selective else 50 if latin else 100
            factor=min(1.,step/warm)*(.1+.9*(1+math.cos(math.pi*max(0,step-warm)/(updates-warm)))/2)
            for g in optimizer.param_groups:g['lr']=g['base_lr']*factor
            optimizer.zero_grad(set_to_none=True);offset=0;cv=kv=0.;n=len(batch['texts'])
            for start in range(0,n,4):
                b={k:v[start:start+4] for k,v in batch.items() if k!='targets'}
                size=int(b['lengths'].sum());targets=batch['targets'][offset:offset+size];offset+=size
                images=b['images'].to(device);orientation=b['orientation'].to(device)
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(images,orientation,b['input_lengths'])
                if normalized:ctc_logp,loss_targets=canonical_ctc_inputs(logits,targets,mapping)
                else:ctc_logp,loss_targets=logits.float().log_softmax(-1),targets
                ctc=criterion(ctc_logp,loss_targets.to(device),b['input_lengths'],b['lengths'])
                use_teacher=teacher is not None and (kind in ('original','generated','yolo') if selective else kind!='target')
                eligible=torch.tensor([not any(c in string.ascii_letters for c in text) for text in b['texts']],dtype=torch.bool) if selective else torch.ones(len(b['texts']),dtype=torch.bool)
                if use_teacher and eligible.any():
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):reference=teacher(images,orientation,b['input_lengths'])
                    if normalized:kd=canonical_teacher_kl(logits[:,eligible],reference[:,eligible],b['input_lengths'][eligible],mapping,2.)*float(eligible.float().mean())
                    else:kd=teacher_kl(logits[:,eligible],reference[:,eligible],b['input_lengths'][eligible],2.)*float(eligible.float().mean())
                else:kd=logits.float().sum()*0
                loss=ctc+kd
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite CTC/KL')
                weight=len(b['texts'])/n;(loss*weight).backward();cv+=float(ctc.detach())*weight;kv+=float(kd.detach())*weight
            if recurrent_trainable:assert all(p.grad is not None for p in model.recurrent.parameters())
            if selective:assert all(p.grad is not None for p in model.classifier.heads.parameters())
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5,error_if_nonfinite=True);optimizer.step()
            if ema is not None:
                with torch.no_grad():
                    for p,q in zip(ema.parameters(),model.parameters()):p.lerp_(q,.002)
            return dict(ctc=cv,teacher_kl=kv,lr_factor=factor)
        if args.smoke:
            evidence=[]
            train_mix=CanonicalTrainingLabels(mix,mapping) if normalized else mix
            for kind in (['target','latin_multi','digits','brackets','original','generated','yolo'] if selective else ['target'] if latin else ['target','original','generated','yolo']):
                ids=next(b for b in plan if b[0][0]==kind);batch=collate([train_mix[k] for k in ids])
                evidence.append(dict(kind=kind,**train_batch(batch,1,kind)));audit()
            write_json(work/'smoke.json',dict(passed=True,evidence=evidence,frozen_weights_and_bn147_verified=True));status(state='smoke_passed');return
        if normalized and not (work/'source.zip').exists():
            import zipfile
            with zipfile.ZipFile(work/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for p in [Path(__file__),Path('scripts/recognition/prepare_v49.py'),Path('scripts/recognition/run_v50.ps1'),*Path('src/recognition').glob('*.py')]:archive.write(p,p.as_posix())
        _,views=make_views(cfg,mapping);views.update(extra)
        views['new_character_probe']=BucketCrops([r for r in rows(V40/'targeted/labels.jsonl') if r['split']=='probe'],V40/'targeted',cfg,mapping)
        for ds in views.values():
            if hasattr(ds,'mapping'):ds.mapping=mapping
            normalize_dataset(ds)
        baselines={'fp16':load('outputs/recognition/evaluation/v35_paired_short_inspection/fp16_full_source.json'),'bf16':load(V34/'initial.json')}
        def validate(candidate,step,arm,full=True):
            deployed=MenuRecognizer(len(mapping)+1,use_bilstm=True).to(device).eval();deployed.load_state_dict(export_state(candidate))
            deployed.vocabulary=mapping;deployed.recurrent.lstm.flatten_parameters();out={}
            precisions=('fp16','bf16') if full else ('fp16',)
            for precision in precisions:
                status(state='evaluating',update=step,arm=arm,precision=precision);r={};canonical={};textnormalized={}
                for name,ds in views.items():
                    if not full and not name.startswith('target_'):continue
                    fn=evaluate_fp16 if precision=='fp16' else evaluate
                    if normalized:
                        modes=evaluate_parenthesis_modes(deployed,ds,device,precision)
                        r[name]=modes['literal'];canonical[name]=modes['canonical'];textnormalized[name]=modes['text_normalized']
                    else:r[name]=evaluate_any(deployed,ds,device,precision) if name in extra or name=='new_character_probe' else fn(deployed,ds,device)
                    if name.startswith('target_'):
                        for record in r[name]['0.0']['predictions']:
                            meta=ds.rows[record['index']]
                            for key in ('anchor','kind','style','position'):
                                if key in meta:record[key]=meta[key]
                result=dict(validation=r,target_metrics={name:character_metrics(r,name,string.ascii_letters if latin else '()（）') for name in r if name.startswith('target_')})
                if normalized:
                    result.update(canonical_validation=canonical,text_normalized_validation=textnormalized,
                        canonical_symbol_metrics={name:character_metrics(canonical,name,'()') for name in ('parentheses_known','parentheses_heldout','brackets_known','brackets_heldout','yolo_current_holdout')},
                        canonical_target_metrics={name:character_metrics(canonical,name,string.ascii_letters) for name in canonical if name.startswith('target_')})
                result['target_breakdown']={}
                for name,value in r.items():
                    if not name.startswith('target_'):continue
                    rr=value['0.0']['predictions'];result['target_breakdown'][name]={}
                    for key in (('kind',) if latin else ('style','position')):
                        result['target_breakdown'][name][key]={v:summarise([x for x in rr if x[key]==v]) for v in sorted({x[key] for x in rr})}
                    if latin:
                        result['target_breakdown'][name]['new33']=character_metrics(r,name,added)
                        result['target_breakdown'][name]['existing19']=character_metrics(r,name,sorted(set(string.ascii_letters)-set(added)))
                if full:
                    result['retention']=apply_retention_policy(legacy_gates(r,baselines[precision]))
                    for name in ('digits_known','digits_heldout'):
                        for row in r[name]['0.0']['predictions']:
                            meta=views[name].rows[row['index']];row.update(kind=meta['kind'],fullwidth_rendered=meta['fullwidth_rendered'])
                    result['digit_metrics']=digit_metrics(r)
                    result['parentheses_metrics']={name:character_metrics(r,name,'()（）') for name in ('parentheses_known','parentheses_heldout','yolo_current_holdout')}
                write_json(work/(f'initial_{precision}.json' if step==0 else f'step_{step:04d}_{arm}_{precision}.json'),result);out[precision]=result
            del deployed;audit();return out
        step=0;history=[];streams=Counter();chars=Counter();best_step=None;best_score=None;best_arm=None
        def snapshot(candidate=None,arm='raw'):
            return dict(model=export_state(candidate or model),raw_training_model=cpu_copy(model.state_dict()),
                ema_training_model=cpu_copy(ema.state_dict()) if ema is not None else None,optimizer=cpu_copy(optimizer.state_dict()),
                vocab=vocab,config=dict(cfg,isolated_arm=config),updates=step,checkpoint_arm=arm,history=history,
                streams=dict(streams),characters=dict(chars),best_update=best_step,best_arm=best_arm,best_score=best_score,
                rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all())
        if args.resume and (work/'last.pt').exists():
            saved=torch.load(work/'last.pt',map_location='cpu',weights_only=False);assert saved['vocab']==vocab
            model.load_state_dict(saved['raw_training_model']);optimizer.load_state_dict(saved['optimizer'])
            if ema is not None:ema.load_state_dict(saved['ema_training_model'])
            step=saved['updates'];history=saved['history'];streams.update(saved['streams']);chars.update(saved['characters'])
            best_step=saved['best_update'];best_arm=saved['best_arm'];best_score=saved['best_score']
            random.setstate(saved['rng_python']);np.random.set_state(saved['rng_numpy']);torch.set_rng_state(saved['rng_torch']);torch.cuda.set_rng_state_all(saved['rng_cuda'])
            assert load(work/'initial_parity.json')['passed']
        else:
            result=validate(model,0,'initial');parity={}
            for precision,d in result.items():
                expected=load(START.parent/f'step_2000_raw_{precision}.json')['validation'];parity[precision]={}
                for name,a in d['validation'].items():
                    if name not in expected:continue
                    x=a['0.0']['predictions'];y=expected[name]['0.0']['predictions']
                    assert len(x)==len(y) and all(p['truth']==q['truth'] for p,q in zip(x,y))
                    parity[precision][name]=sum(p['prediction']!=q['prediction'] for p,q in zip(x,y))
                assert not any(parity[precision].values())
                if normalized:
                    expected_v49=load(Path('outputs/recognition/pilots/v49_selective_head_2000')/f'initial_{precision}.json')['validation']
                    for name,value in expected_v49.items():
                        if name not in d['validation']:continue
                        assert [x['prediction'] for x in value['0.0']['predictions']]==[x['prediction'] for x in d['validation'][name]['0.0']['predictions']]
            write_json(work/'initial_parity.json',dict(passed=True,disagreements=parity));save_atomic(work/'initial.pt',snapshot());save_atomic(work/'last.pt',snapshot())
        initial_results={p:load(work/f'initial_{p}.json') for p in ('fp16','bf16')}
        checkpoints={100,250,500,1000} if latin and not selective else {500,1000,1500,2000}
        def evaluate_checkpoint():
            nonlocal best_step,best_score,best_arm
            arms={}
            for arm,candidate in [('raw',model)]+([('ema',ema)] if ema is not None else []):
                full=not latin or selective or step==updates
                result=validate(candidate,step,arm,full);score=result['fp16']['validation']['target_train' if latin else 'target_heldout']['0.0']['all']['cer']
                if selective:
                    v=result['fp16']['canonical_validation'] if normalized else result['fp16']['validation']
                    score=sum(w*v[name]['0.0']['all']['cer'] for w,name in [(.25,'target_multifont_heldout'),(.25,'target_heldout_fonts'),(.2,'parentheses_heldout'),(.15,'digits_heldout'),(.15,'yolo_current_holdout')])
                eligible=True
                if not latin or selective:
                    for precision,d in result.items():
                        base=initial_results[precision];v=d['validation'];b=base['validation']
                        eligible &= set(d['retention']['failed'])<=set(base['retention']['failed'])
                        eligible &= v['yolo_current_holdout']['0.0']['all']['cer']<=b['yolo_current_holdout']['0.0']['all']['cer']+.01
                        eligible &= d['digit_metrics']['digits_heldout']['macro_recall']>=base['digit_metrics']['digits_heldout']['macro_recall']-.02
                        if selective:
                            eligible &= v['historical_bucket']['0.0']['all']['cer']<=b['historical_bucket']['0.0']['all']['cer']+.01
                            for n in ('pooled_original_1char','pooled_original_2char'):
                                eligible &= d['retention']['current'][n]<=base['retention']['current'][n]+.02
                if eligible and (best_score is None or score<best_score):
                    best_step=step;best_arm=arm;best_score=score;save_atomic(work/'best.pt',snapshot(candidate,arm))
                arms[arm]={p:dict(target_cer={k:v['0.0']['all']['cer'] for k,v in d['validation'].items() if k.startswith('target_')},
                    target_metrics=d['target_metrics'],retention=d.get('retention'),digit_metrics=d.get('digit_metrics'),
                    parentheses_metrics=d.get('parentheses_metrics'),yolo_cer=d['validation'].get('yolo_current_holdout',{}).get('0.0',{}).get('all',{}).get('cer'),selection_eligible=eligible) for p,d in result.items()}
                if normalized:
                    for p,d in result.items():
                        arms[arm][p].update(canonical_all={k:v['0.0']['all'] for k,v in d['canonical_validation'].items()},
                            canonical_symbols=d['canonical_symbol_metrics'],canonical_target_metrics=d['canonical_target_metrics'])
            history.append(dict(update=step,arms=arms));(work/'metrics.jsonl').write_text(''.join(json.dumps(h,ensure_ascii=False)+'\n' for h in history),encoding='utf-8')
            save_atomic(work/f'step_{step:04d}.pt',snapshot());save_atomic(work/'last.pt',snapshot())
        train_mix=CanonicalTrainingLabels(mix,mapping) if normalized else mix
        loader=DataLoader(train_mix,batch_sampler=plan[step:],num_workers=0,collate_fn=collate,generator=torch.Generator().manual_seed(SEED))
        for batch in loader:
            kind=plan[step][0][0];loss=train_batch(batch,step+1,kind)
            for k,i in plan[step]:streams[k]+=1;chars.update(mix.datasets[k].rows[i]['text'])
            step+=1
            if step==1 or step%100==0:
                audit();status(state='training',update=step,total_updates=updates,**loss);print('TRAIN',step,loss,flush=True)
                if step not in checkpoints:save_atomic(work/'last.pt',snapshot())
            if step in checkpoints:evaluate_checkpoint()
        audit();assert dict(streams)==sampling['streams'] and step==updates
        if teacher is not None:assert all(torch.equal(v.cpu(),initial[k]) for k,v in teacher.state_dict().items())
        assert sha(START)==source_hash and sha(REGISTRY)==registry_hash and sha(current_checkpoint())==active_hash
        save_atomic(work/'final_raw.pt',snapshot())
        if ema is not None:save_atomic(work/'final_ema.pt',snapshot(ema,'ema'))
        report=dict(completed=True,arm=args.arm,updates=step,presentations=step*32,elapsed_seconds=time.time()-started,
            streams=dict(streams),character_encounters=dict(chars),best_update=best_step,best_arm=best_arm,best_score=best_score,
            selection='MinimumtrainingCERfittingcheckpoint; notretentionqualified' if latin else 'MinimumheldouttargetCERsubjecttosource-relativegate/YOLO/digitguardsbothprecisions',
            frozen_cnn_projection_condition_temporal_bn_affine_and147buffers=True,
            all_nonlatin_classifier_rows_frozen=latin,source_and_registry_unchanged=True,production_promoted=False)
        if selective:
            report.update(selection='WeightedLatin/paren/digit/YOLOscorewithsource-relativeguardsbothprecisions;notabsoluteeligibility',
                bilstm_frozen=not recurrent_trainable,all_non_selected_classifier_rows_frozen=True,all_nonlatin_classifier_rows_frozen=False,
                row_groups={'latin':52,'digits_brackets':14},blank_row_frozen=True)
        if normalized:
            canonical_counts=Counter();changed=0
            for batch in plan:
                for k,i in batch:
                    text=mix.datasets[k].rows[i]['text'];new=normalize_parentheses(text);canonical_counts.update(new);changed+=new!=text
            report.update(parentheses_normalized=True,canonical_character_encounters=dict(canonical_counts),normalization_changed_presentations=changed,
                matched_v49_plan_sha256=sha(reference),recurrent_lr=1e-5 if recurrent_trainable else 0,
                comparison='BothnormalizedarmschangeparenthesissemanticsversusV49;onlyrecurrentadaptationdiffersbetweenthetwoV50arms.')
        write_json(work/'report.json',report)
        lines=['# '+('V50 normalized '+('BiLSTM' if recurrent_trainable else 'frozen control') if normalized else 'V49 selective head-only' if selective else 'V47 clean Latin fitting' if latin else 'V48 isolated parentheses'),'',json.dumps(report,ensure_ascii=False,indent=2),'',
               'Independent V43 source; no automatic promotion. V49/V48best is source-relative gated; V47best is fitting-only. See per-step JSON for raw/EMA and exact-symbol metrics.','',
               '|Update|Arm|Target train CER|Target heldout/new-font CER|YOLO CER|','|---:|---|---:|---:|---:|']
        for h in history:
            for arm,r in h['arms'].items():
                d=r['fp16'];key='target_heldout_fonts' if latin else 'target_heldout';yc=d['yolo_cer']
                lines.append(f"|{h['update']}|{arm}|{d['target_cer']['target_train']:.2%}|{d['target_cer'][key]:.2%}|{format(yc,'.2%') if yc is not None else 'not evaluated'}|")
        (work/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');status(state='completed',updates=step)
    except Exception:status(state='failed',error=traceback.format_exc());raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['latin','parentheses','selective'],required=True)
    p.add_argument('--normalize-parentheses',action='store_true');p.add_argument('--unfreeze-bilstm',action='store_true')
    p.add_argument('--output',type=Path);p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true');run(p.parse_args())
