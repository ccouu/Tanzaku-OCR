"""Stage1A: frozen V43 features, only33 appended Latin rows trainable."""
import argparse
from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
import random
import string
import time
import traceback
import zipfile

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.recognition.prepare_v44 import START, WORK as V44, expanded_state
from scripts.recognition.prepare_v43 import normalize_dataset
from scripts.recognition.train_v38_yolo25 import BucketCrops, INPUT, HOLDOUT
from outputs.recognition.v36_yolo_fullpasses import Combined, Subset, load, dump_rows
from outputs.recognition.application_mixture_pilot import ROOT, RenderedLines, cpu_copy
from outputs.recognition.v37_long512 import make_views, evaluate_any
from outputs.recognition.evaluate_bn_swap import gates as legacy_gates, V34
from outputs.recognition.evaluate_fp16_retention import evaluate_fp16
from src.recognition.font_training import evaluate, row_canvas
from src.recognition.datasets import FontLines
from src.recognition.model import MenuRecognizer
from src.recognition.vocabulary import extend_vocabulary, split_existing_classifier, fused_state
from src.recognition.retention_distillation import teacher_kl
from src.recognition.retention import apply_retention_policy
from src.recognition.current import REGISTRY
from src.recognition.data import collate
from src.recognition.metrics import align_errors
from src.recognition.checkpointing import save_atomic
from src.recognition.utils import sha, write_json, seed_all, training_device

WORK = Path('outputs/recognition/pilots/v45_latin_head_only_1000')
SEED = 20261145
V40 = Path('outputs/recognition/training/v40_new_characters_2000')


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines()]


def prepare(work, source):
    vocab, added = extend_vocabulary(source['vocab'], string.ascii_letters)
    vocab['extension']['source_extension'] = copy.deepcopy(source['vocab'].get('extension'))
    assert len(added) == 33
    mapping = vocab['char_to_id']; cfg = source['config']
    latin = rows(V44/'targeted/labels.jsonl')
    # V44 used appearance augmentation exactly at global render index %5 ==0.
    clean = [r for r in latin if r['kind']=='latin' and r['split']=='train'
             and int(Path(r['image']).stem)%5 != 0 and 1 <= len(r['text']) <= 6 and r['canvas'] in (128,256)]
    assert set(r['anchor'] for r in clean) == set(string.ascii_letters)
    yolo = rows(V40/'yolo_labels.jsonl')
    train = [r for r in yolo if r['split']=='train']
    hold = [r for r in yolo if r['split']=='val_holdout']
    assert len(train)==1561 and len(hold)==156
    assert not {r['source_menu'] for r in train} & {r['source_menu'] for r in hold}
    assert {r['source_menu'] for r in hold}==HOLDOUT
    assert not {r['sha256'] for r in train} & {r['sha256'] for r in hold}
    original = FontLines(ROOT,'train',cfg['mean'],cfg['std'])
    generated = rows(V34/'corpus/labels.jsonl')
    corpus = load(V34/'corpus/manifest.json')
    assert sha(ROOT/'labels.jsonl')==corpus['original_labels_sha256']
    assert sha(V34/'corpus/labels.jsonl')==corpus['labels_sha256']
    mix = Combined(original, RenderedLines(generated,V34/'corpus',cfg,mapping),BucketCrops(train,INPUT,cfg,mapping))
    mix.datasets['latin'] = BucketCrops(clean,V44/'targeted',cfg,mapping)
    for ds in mix.datasets.values():
        ds.mapping=mapping; normalize_dataset(ds)
    def width(k,i):
        r=mix.datasets[k].rows[i]
        return row_canvas(r) if k=='original' else r['canvas']
    rng=random.Random(SEED); plan=[]
    def schedule(k,pool,batches,mandatory=()):
        groups=defaultdict(list); must=defaultdict(list)
        for i in pool: groups[width(k,i)].append(i)
        for i in mandatory: must[width(k,i)].append(i)
        order=[w for w,ids in sorted(must.items()) for _ in range(math.ceil(len(ids)/32))]
        assert len(order)<=batches
        ws=sorted(groups)
        order+=rng.choices(ws,weights=[len(groups[w]) for w in ws],k=batches-len(order))
        for w in order:
            ids=must[w][:32];del must[w][:len(ids)]
            ids+=rng.choices(groups[w],k=32-len(ids));rng.shuffle(ids)
            plan.append([(k,i) for i in ids])
        assert not any(must.values())
    for half in range(2):
        start=len(plan)
        schedule('latin',range(len(clean)),200,range(len(clean)))
        schedule('yolo',range(len(train)),100,range(len(train)))
        schedule('original',range(len(original)),100)
        for view in ('clean_control','texture_color_noise'):
            schedule('generated',[i for i,r in enumerate(generated) if r['view']==view],50)
        segment=plan[start:];rng.shuffle(segment);plan[start:]=segment
        assert Counter(k for batch in segment for k,i in batch)==dict(latin=6400,yolo=3200,original=3200,generated=3200)
        assert {i for batch in segment for k,i in batch if k=='yolo'}==set(range(1561))
    streams=Counter(); chars=Counter(); kd=Counter(); canvases=Counter()
    for batch in plan:
        assert len(batch)==32 and len({width(k,i) for k,i in batch})==1
        for k,i in batch:
            text=mix.datasets[k].rows[i]['text'];assert all(c in mapping for c in text)
            streams[k]+=1;chars.update(text);canvases[width(k,i)]+=1
            kd['excluded_new_label' if any(c in added for c in text) else 'old_only']+=k!='latin'
    assert streams==dict(latin=12800,yolo=6400,original=6400,generated=6400)
    extra={'latin_train':mix.datasets['latin'], 'yolo_current_train':mix.datasets['yolo'],
           'yolo_current_holdout':BucketCrops(hold,INPUT,cfg,mapping)}
    # A fixed, evenly anchored fitting diagnostic independent of sampled counts.
    fitting=[]
    for c in string.ascii_letters:
        fitting += [('latin',i) for i,r in enumerate(clean) if r['anchor']==c][:4]
    extra['latin_fit208']=Subset(mix,fitting)
    for cohort in ('known','heldout'):
        extra['latin_'+cohort]=BucketCrops([r for r in latin if r['kind']=='latin' and r['split']=='probe' and r['cohort']==cohort],V44/'targeted',cfg,mapping)
        digits=rows(START.parent/'digits/labels.jsonl')
        extra['digits_'+cohort]=BucketCrops([r for r in digits if r['split']=='probe' and r['cohort']==cohort],START.parent/'digits',cfg,mapping)
    sampling=dict(updates=1000,presentations=32000,streams=dict(streams),half_pass_streams=dict(latin=6400,yolo=3200,original=3200,generated=3200),
        latin_train_images=len(clean),latin_train_widths=sorted({r['canvas'] for r in clean}),latin_max_length=6,
        latin_characters={c:chars[c] for c in string.ascii_letters},latin_added=added,
        replay_teacher_eligibility=dict(kd),physical_width_presentations=dict(canvases),
        yolo_train=1561,yolo_holdout=156,holdout_menus=sorted(HOLDOUT),
        source_sha256=sha(START),latin_labels_sha256=sha(V44/'targeted/labels.jsonl'),
        yolo_labels_sha256=sha(V40/'yolo_labels.jsonl'),normalization='Full-width decimal digits to ASCII; source labels untouched',
        note='Only Latin targets restricted to128/256px; replay retains its native buckets. No new optical distortion. VLM YOLO labels not independently verified.')
    write_json(work/'vocab.json',vocab);write_json(work/'batches.json',plan)
    sampling['plan_sha256']=sha(work/'batches.json');write_json(work/'sampling.json',sampling)
    dump_rows(work/'latin_train_labels.jsonl',clean)
    return vocab,added,mix,plan,extra,sampling


def latin_metrics(report, added):
    result={}
    for name in ('latin_train','latin_fit208','latin_known','latin_heldout'):
        rr=report[name]['0.0']['predictions']; per={}
        for group,characters in (('new33',added),('existing19',set(string.ascii_letters)-set(added)),('all52',string.ascii_letters)):
            refs=Counter();hits=Counter();err=Counter()
            for r in rr:
                refs.update(c for c in r['truth'] if c in characters)
                for op in align_errors(r['truth'],r['prediction']):
                    if op['truth'] in characters:
                        if op['type']=='match':hits[op['truth']]+=1
                        else:err[op['type']]+=1
            recalls={c:hits[c]/refs[c] for c in sorted(characters) if refs[c]}
            per[group]=dict(macro_recall=sum(recalls.values())/len(recalls) if recalls else None,per_character=recalls,errors=dict(err))
        result[name]=dict(cer=report[name]['0.0']['all']['cer'],groups=per,
            by_kind={kind:dict(records=len(selected),exact=sum(r['truth']==r['prediction'] for r in selected))
                     for kind in ('single','repeated','word')
                     for selected in [[r for r in rr if ('single' if len(r['truth'])==1 else 'repeated' if len(set(r['truth']))==1 else 'word')==kind]]})
    return result


def run(args):
    work=args.output;work.mkdir(parents=True,exist_ok=args.resume or args.smoke or args.prepare_only)
    started=time.time()
    def status(**kw):write_json(work/'status.json',dict(elapsed_seconds=time.time()-started,**kw))
    try:
        torch.set_num_threads(4);seed_all(SEED)
        source=torch.load(START,map_location='cpu',weights_only=False)
        source_hash=sha(START);registry_hash=sha(REGISTRY)
        assert source['updates']==2000 and source['checkpoint_arm']=='raw'
        status(state='preparing');vocab,added,mix,plan,extra,sampling=prepare(work,source)
        mapping=vocab['char_to_id'];old_classes=len(source['vocab']['char_to_id'])+1
        cfg=dict(source['config'],classes=len(mapping)+1)
        config=dict(source=str(START),source_sha256=source_hash,updates=1000,batch=32,microbatch=4,
            new_row_lr=.001,warmup_updates=100,min_lr_factor=.1,blank_margin=8,teacher_kl_weight=1.,temperature=2.,
            trainable='Only33newLatinclassifierrows; alloldweights,BNaffine,and147BNbuffersfrozen; evalmode throughout',
            distillation='Full-softmax teacher KL on valid frames of all old-only synthetic/YOLO replay; no KL on labels containing new33letters',
            early_stop='At500ifLatintrainingCER<10%andall52heldoutmacrorecall>70%andnoadditionalretentionfailuresvsV43bothprecisions; not automaticpromotion',
            sampling_sha256=sha(work/'sampling.json'))
        if (work/'run_config.json').exists():assert load(work/'run_config.json')==config
        write_json(work/'run_config.json',config)
        if args.prepare_only:status(state='prepared');return
        device=training_device('cuda:0','RTX 3070')
        initial=expanded_state(source,vocab)
        model=MenuRecognizer(len(mapping)+1,use_bilstm=True);model.load_state_dict(initial)
        model.classifier=split_existing_classifier(model.classifier,old_classes)
        model.requires_grad_(False);model.classifier.new.requires_grad_(True)
        model.to(device).eval();model.vocabulary=mapping;model.recurrent.lstm.flatten_parameters()
        teacher=MenuRecognizer(len(mapping)+1,use_bilstm=True).to(device)
        teacher.load_state_dict(initial);teacher.eval().requires_grad_(False);teacher.recurrent.lstm.flatten_parameters()
        frozen={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('classifier.new.')}
        def audit():
            assert not model.training
            assert all(torch.equal(model.state_dict()[k],v) for k,v in frozen.items())
            assert all(not p.requires_grad and p.grad is None for n,p in model.named_parameters() if not n.startswith('classifier.new.'))
            assert sum(k.endswith(('running_mean','running_var','num_batches_tracked')) for k in frozen)==147
        def audit_teacher():
            assert all(torch.equal(v.cpu(),initial[k]) for k,v in teacher.state_dict().items())
        optimizer=torch.optim.AdamW([dict(params=[model.classifier.new.weight],weight_decay=.01),dict(params=[model.classifier.new.bias],weight_decay=0)],lr=.001)
        criterion=nn.CTCLoss(blank=0,zero_infinity=False)
        def train_batch(batch,step):
            f=min(1.,step/100)*(.1+.9*(1+math.cos(math.pi*max(0,step-100)/900))/2)
            for g in optimizer.param_groups:g['lr']=.001*f
            optimizer.zero_grad(set_to_none=True);offset=0;cv=kv=0.;n=len(batch['texts'])
            for start in range(0,n,4):
                b={k:v[start:start+4] for k,v in batch.items() if k!='targets'}
                count=int(b['lengths'].sum());targets=batch['targets'][offset:offset+count];offset+=count
                images=b['images'].to(device);ori=b['orientation'].to(device)
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(images,ori,b['input_lengths'])
                with torch.autocast('cuda',enabled=False):ctc=criterion(logits.float().log_softmax(-1),targets.to(device),b['input_lengths'],b['lengths'])
                eligible=torch.tensor([not any(c in added for c in t) for t in b['texts']],dtype=torch.bool)
                if plan[step-1][0][0]=='latin':eligible[:]=False
                if eligible.any():
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):ref=teacher(images,ori,b['input_lengths'])
                    kl=teacher_kl(logits[:,eligible],ref[:,eligible],b['input_lengths'][eligible],2.)*float(eligible.float().mean())
                else:kl=logits.sum()*0
                loss=ctc+kl
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                weight=len(b['texts'])/n;(loss*weight).backward();cv+=float(ctc.detach())*weight;kv+=float(kl.detach())*weight
            assert model.classifier.new.weight.grad is not None
            nn.utils.clip_grad_norm_(model.classifier.new.parameters(),5,error_if_nonfinite=True)
            optimizer.step();audit();return dict(ctc=cv,teacher_kl=kv,lr=.001*f)
        if args.smoke:
            batches=[]
            for kind in ('latin','original','generated','yolo'):
                j=next(i for i,b in enumerate(plan) if b[0][0]==kind)
                batch=collate([mix[key] for key in plan[j]])
                batches.append(dict(kind=kind,**train_batch(batch,j+1)))
            write_json(work/'smoke.json',dict(passed=True,batches=batches,all_pretrained_weights_fixed=True,bn147_fixed=True))
            status(state='smoke_passed');return
        _,views=make_views(cfg,mapping);views.update(extra)
        for ds in views.values():
            if hasattr(ds,'mapping'):ds.mapping=mapping
            normalize_dataset(ds)
        baselines={'fp16':load('outputs/recognition/evaluation/v35_paired_short_inspection/fp16_full_source.json'),'bf16':load(V34/'initial.json')}
        def validate(step):
            output={}
            deployed=MenuRecognizer(len(mapping)+1,use_bilstm=True).to(device)
            deployed.load_state_dict(fused_state(model));deployed.vocabulary=mapping;deployed.eval();deployed.recurrent.lstm.flatten_parameters()
            for precision in ('fp16','bf16'):
                status(state='evaluating',update=step,precision=precision)
                report={}
                for name,ds in views.items():
                    fn=evaluate_fp16 if precision=='fp16' else evaluate
                    report[name]=evaluate_any(deployed,ds,device,precision) if name in extra else fn(deployed,ds,device)
                gates=apply_retention_policy(legacy_gates(report,baselines[precision]))
                false={name:sum(any(c in added for c in r['prediction']) for r in value['0.0']['predictions'] if not any(c in added for c in r['truth'])) for name,value in report.items() if not name.startswith('latin')}
                result=dict(validation=report,retention=gates,latin=latin_metrics(report,added),false_new_latin_emissions=false)
                write_json(work/(f'initial_{precision}.json' if step==0 else f'step_{step:04d}_raw_{precision}.json'),result)
                output[precision]=result
            del deployed;audit();audit_teacher();return output
        step=0;streams=Counter();chars=Counter();history=[];best_step=0;best_cer=None
        def snapshot():
            return dict(model=fused_state(model),raw_training_model=cpu_copy(model.state_dict()),optimizer=cpu_copy(optimizer.state_dict()),
                vocab=vocab,config=dict(cfg,v45_training=config),updates=step,checkpoint_arm='raw',best_update=best_step,best_cer=best_cer,
                stream_encounters=dict(streams),character_encounters=dict(chars),history=history,
                rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all())
        if not (work/'source.zip').exists():
            with zipfile.ZipFile(work/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in [Path(__file__),Path('scripts/recognition/run_v45.ps1'),*Path('src/recognition').glob('*.py')]:z.write(p,p.as_posix())
        if args.resume and (work/'last.pt').exists():
            ck=torch.load(work/'last.pt',map_location='cpu',weights_only=False);assert ck['vocab']==vocab
            model.load_state_dict(ck['raw_training_model']);optimizer.load_state_dict(ck['optimizer']);step=ck['updates'];best_step=ck['best_update'];best_cer=ck['best_cer']
            streams.update(ck['stream_encounters']);chars.update(ck['character_encounters']);history=ck['history']
            random.setstate(ck['rng_python']);np.random.set_state(ck['rng_numpy']);torch.set_rng_state(ck['rng_torch']);torch.cuda.set_rng_state_all(ck['rng_cuda'])
            assert load(work/'initial_parity.json')['passed']
        else:
            results=validate(0);parity={}
            for precision,r in results.items():
                expected=load(START.parent/f'step_2000_raw_{precision}.json')['validation'];parity[precision]={}
                for name,value in r['validation'].items():
                    if name not in expected:continue
                    a=value['0.0']['predictions'];b=expected[name]['0.0']['predictions']
                    assert len(a)==len(b) and all(x['truth']==y['truth'] for x,y in zip(a,b))
                    parity[precision][name]=sum(x['prediction']!=y['prediction'] for x,y in zip(a,b))
                assert not any(parity[precision].values()) and not any(r['false_new_latin_emissions'].values())
                assert not any(any(c in added for c in row['prediction']) for value in r['validation'].values() for row in value['0.0']['predictions'])
            write_json(work/'initial_parity.json',dict(passed=True,disagreements=parity,blank_margin=8,old_ids_unchanged=True))
            save_atomic(work/'initial.pt',snapshot());save_atomic(work/'last.pt',snapshot())
        initial_fail={p:set(load(work/f'initial_{p}.json')['retention']['failed']) for p in ('fp16','bf16')}
        def acceptable(results):return all(set(r['retention']['failed'])<=initial_fail[p] for p,r in results.items())
        stopped=False
        if step in (500,1000):
            results={p:load(work/f'step_{step:04d}_raw_{p}.json') for p in ('fp16','bf16')}
            stopped=step==500 and results['fp16']['latin']['latin_train']['cer']<.10 and results['fp16']['latin']['latin_heldout']['groups']['all52']['macro_recall']>.70 and acceptable(results)
        remaining=[] if stopped else plan[step:]
        loader=DataLoader(mix,batch_sampler=remaining,num_workers=0,collate_fn=collate,generator=torch.Generator().manual_seed(SEED))
        for batch in loader:
            loss=train_batch(batch,step+1)
            for k,i in plan[step]:streams[k]+=1;chars.update(mix.datasets[k].rows[i]['text'])
            step+=1
            if step==1 or step%100==0:
                status(state='training',update=step,total_updates=1000,**loss)
                # An evaluation-boundary checkpoint is committed only after its
                # reports/history exist, so failure resumes from the prior100.
                if step%500!=0:save_atomic(work/'last.pt',snapshot())
                print('TRAIN',step,loss,flush=True)
            if step%500==0:
                results=validate(step);cer=results['fp16']['latin']['latin_train']['cer']
                entry=dict(update=step,loss=loss,precisions={p:dict(retention=r['retention'],latin=r['latin'],false_new_latin_emissions=r['false_new_latin_emissions'],yolo_holdout_cer=r['validation']['yolo_current_holdout']['0.0']['all']['cer']) for p,r in results.items()})
                history.append(entry)
                if acceptable(results) and (best_cer is None or cer<best_cer):
                    best_cer=cer;best_step=step;save_atomic(work/'best.pt',snapshot())
                save_atomic(work/f'step_{step:04d}.pt',snapshot());save_atomic(work/'last.pt',snapshot())
                (work/'metrics.jsonl').write_text(''.join(json.dumps(h,ensure_ascii=False)+'\n' for h in history),encoding='utf-8')
                stopped=step==500 and cer<.10 and results['fp16']['latin']['latin_heldout']['groups']['all52']['macro_recall']>.70 and acceptable(results)
                if stopped:break
        audit();assert sum(streams.values())==step*32 and streams==dict(latin=step*32*.4,yolo=step*32*.2,original=step*32*.2,generated=step*32*.2)
        assert sha(START)==source_hash and sha(REGISTRY)==registry_hash
        delta=(model.classifier.new.weight.detach().cpu()-initial['classifier.weight'][old_classes:]).flatten(1).norm(dim=1)
        assert bool((delta>0).all())
        save_atomic(work/'final_raw.pt',snapshot())
        report=dict(completed=True,updates=step,presentations=step*32,early_stopped=stopped,streams=dict(streams),
            actual_latin_encounters={c:chars[c] for c in string.ascii_letters},new_row_delta_norms={c:float(v) for c,v in zip(added,delta)},
            all_pretrained_weights_and_affine_bit_identical=True,bn147_buffers_fixed=True,source_and_registry_unchanged=True,
            best_update=best_step,best_selection='LowestLatintrainCERamongcheckpointswithnoadditionalretentionfailuresversusV43,bothprecisions;notgatepassingguarantee',
            absolute_retention_pass=all(r['retention']['eligible'] for r in results.values()),production_promoted=False)
        write_json(work/'report.json',report)
        text=['# V45 Stage1A Latin head-only feasibility','',json.dumps(report,ensure_ascii=False,indent=2),'',
              'Only33newLatinrows trained from V43finalraw; no automatic Stage1B or promotion. best.pt exists only if a trained checkpoint meets the source-relative retention rule. V43 already fails absolute gates.', '',
              '|Update|Precision|Train Latin CER|Heldout macro recall|YOLO CER|Failed retention gates|','|---:|---|---:|---:|---:|---|']
        for h in history:
            for p,r in h['precisions'].items():text.append(f"|{h['update']}|{p}|{r['latin']['latin_train']['cer']:.2%}|{r['latin']['latin_heldout']['groups']['all52']['macro_recall']:.2%}|{r['yolo_holdout_cer']:.2%}|{r['retention']['failed']}|")
        (work/'analysis.md').write_text('\n'.join(text)+'\n',encoding='utf-8');status(state='completed',**report)
    except Exception:
        status(state='failed',error=traceback.format_exc());raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=WORK);p.add_argument('--resume',action='store_true')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--smoke',action='store_true');run(p.parse_args())
