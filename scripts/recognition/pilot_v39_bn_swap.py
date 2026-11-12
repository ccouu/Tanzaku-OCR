"""V38 forward-only YOLO-train AdaBN pilot; no optimizer, no promotion."""
from collections import Counter,defaultdict
import copy
import json
from pathlib import Path
import random
import time
import traceback
import zipfile
import torch
from torch.utils.data import DataLoader

from src.recognition.current import current_checkpoint,load_current_recognizer,REGISTRY
from src.recognition.bn_calibration import recalibrate_bn
from src.recognition.data import collate
from src.recognition.utils import sha,write_json,training_device
from src.recognition.retention import apply_retention_policy
from src.recognition.checkpointing import save_atomic
from src.recognition.font_training import evaluate
from outputs.recognition.v37_long512 import make_views,evaluate_any
from outputs.recognition.v36_yolo_fullpasses import load
from outputs.recognition.tiered_short_pilot import rows
from outputs.recognition.evaluate_fp16_retention import evaluate_fp16
from outputs.recognition.evaluate_bn_swap import gates as legacy_gates,V34
from scripts.recognition.train_v38_yolo25 import WORK as V38,BucketCrops,HOLDOUT
from scripts.recognition.review_v38_yolo25 import cluster_bootstrap

WORK=Path('outputs/recognition/pilots/v39_v38_yolo_bn_swap')


def run():
    WORK.mkdir(parents=True,exist_ok=False);started=time.time()
    def status(**kw):write_json(WORK/'status.json',dict(elapsed_seconds=time.time()-started,**kw))
    try:
        torch.set_num_threads(4);torch.manual_seed(39);device=training_device('cuda:0','RTX 3070')
        path=current_checkpoint();source_hash=sha(path);registry_hash=sha(REGISTRY)
        source_model,saved=load_current_recognizer('cpu');assert saved['updates']==686
        cfg=saved['config'];mapping=saved['vocab']['char_to_id']
        delivered=rows(V38/'yolo/labels.jsonl');train=[r for r in delivered if r['split']=='train'];hold=[r for r in delivered if r['split']=='val_holdout']
        assert len(train)==1336 and len(hold)==135 and not {r['source_menu'] for r in train}&HOLDOUT
        assert {r['source_menu'] for r in hold}==HOLDOUT
        assert not {r['sha256'] for r in train}&{r['sha256'] for r in hold}
        ds=BucketCrops(train,V38/'yolo',cfg,mapping);held=BucketCrops(hold,V38/'yolo',cfg,mapping)
        for r in delivered:assert sha(V38/'yolo'/r['image'])==r['sha256']
        groups=defaultdict(list)
        for i,r in enumerate(train):groups[r['canvas']].append(i)
        rng=random.Random(39);plan=[]
        for width,indices in sorted(groups.items()):
            rng.shuffle(indices);plan.extend(indices[j:j+16] for j in range(0,len(indices),16))
        rng.shuffle(plan)
        assert Counter(i for b in plan for i in b)==Counter(range(1336))
        write_json(WORK/'calibration_plan.json',plan)
        write_json(WORK/'run_config.json',dict(source=str(path),source_sha256=source_hash,training_updates=0,
            calibration_crops=1336,holdout_crops=135,calibration_batch=16,calibration_forward_batches=len(plan),
            precision='FP32 calibration; FP16/BF16 evaluation',seed=39,
            normalization='All49BNlayers pooled activation-count-weighted input moments; unbiased running variance. No EMA blending with old stats.',
            mode='Non-BN modules eval; BN batch mode only; one train-only pass; no optimizer/backward; affine and all weights unchanged.',
            physical_padding='Delivered128/256/384/512 widths; homogeneous-width batches; includes padded background in BN statistics.',
            limits='Deeper-layer calibration inputs use upstream batch-mode BN. No iterative inference-stat fixed point. VLM labels not independently verified.',
            selection='Diagnostic only; no automatic promotion; use menu_adaptation_v2 gates in both precisions.'))
        model=copy.deepcopy(source_model).to(device);model.requires_grad_(False)
        before={k:v.cpu().clone() for k,v in model.state_dict().items()}
        bnkeys={k for k in before if k.endswith(('running_mean','running_var','num_batches_tracked'))};assert len(bnkeys)==147
        seen=[]
        def forward(candidate,batch):
            return candidate(batch['images'].to(device),batch['orientation'].to(device),batch['input_lengths'])
        def batches():
            loader=DataLoader(ds,batch_sampler=plan,num_workers=0,collate_fn=collate)
            for n,batch in enumerate(loader):
                seen.extend(plan[n])
                if n%20==0:status(state='calibrating',batch=n+1,total_batches=len(plan),crops_seen=len(seen))
                yield batch
        status(state='calibrating');layer_audit=recalibrate_bn(model,batches(),forward)
        model.eval()
        assert Counter(seen)==Counter(range(1336)) and len(layer_audit)==49
        state={k:v.cpu().clone() for k,v in model.state_dict().items()}
        assert all(torch.equal(state[k],before[k]) for k in state if k not in bnkeys)
        assert all(p.grad is None for p in model.parameters())
        changed=[k for k in state if not torch.equal(state[k],before[k])]
        assert set(changed)<=bnkeys
        for k in state:
            if k.endswith('running_var'):assert torch.isfinite(state[k]).all() and (state[k]>=0).all()
        calibration=dict(crops_seen=len(seen),unique_crops=len(set(seen)),all_once=True,holdout_used=False,
            bn_layers=49,targeted_buffers=147,changed_buffers=changed,all_other_state_bit_identical=True,
            gradients_created=False,optimizer_updates=0,layer_activation_counts=layer_audit)
        write_json(WORK/'calibration_audit.json',calibration)
        save_atomic(WORK/'candidate.pt',dict(model=state,vocab=saved['vocab'],config=cfg,
            experiment='V39 YOLO-only forward BN swap',training_updates=0,source=str(path),source_sha256=source_hash,
            calibration=calibration,promoted=False))
        _,views=make_views(cfg,mapping)
        views['yolo_current_train']=ds;views['yolo_current_holdout']=held
        baselines={'fp16':load('outputs/recognition/evaluation/v35_paired_short_inspection/fp16_full_source.json'),'bf16':load(V34/'initial.json')}
        results={};parity={}
        # Re-run unchanged V38 to confirm current input/precision parity, rather
        # than trusting cached predictions from a different evaluation context.
        for arm,candidate in (('v38',source_model.to(device)),('bn_yolo',model)):
            results[arm]={};candidate.eval()
            fixed={k:v.cpu().clone() for k,v in candidate.state_dict().items()}
            for precision in ('fp16','bf16'):
                result={}
                for name,view in views.items():
                    status(state='evaluating',arm=arm,precision=precision,dataset=name)
                    fn=evaluate_fp16 if precision=='fp16' else evaluate
                    result[name]=evaluate_any(candidate,view,device,precision) if name.startswith('yolo_current_') else fn(candidate,view,device)
                g=apply_retention_policy(legacy_gates(result,baselines[precision]))
                results[arm][precision]=dict(validation=result,retention=g)
                write_json(WORK/f'{arm}_{precision}.json',results[arm][precision])
                if arm=='v38':
                    old=load(V38/f'step_0686_raw_{precision}.json')['validation']
                    parity[precision]={}
                    for name in old:
                        a=result[name]['0.0']['predictions'];b=old[name]['0.0']['predictions']
                        assert len(a)==len(b)
                        parity[precision][name]=sum(x['prediction']!=y['prediction'] for x,y in zip(a,b))
                    assert not any(parity[precision].values())
            assert all(torch.equal(candidate.state_dict()[k].cpu(),v) for k,v in fixed.items())
        write_json(WORK/'initial_parity.json',parity)
        paired={}
        for precision in ('fp16','bf16'):
            a=results['v38'][precision]['validation'];b=results['bn_yolo'][precision]['validation']
            paired[precision]={}
            for name,metadata in (('yolo_current_train',train),('yolo_current_holdout',hold)):
                comparison=cluster_bootstrap(a[name]['0.0']['predictions'],b[name]['0.0']['predictions'],metadata)
                comparison['bn_swap_minus_v38_pp']=comparison.pop('v38_minus_v36_pp')
                paired[precision][name]=comparison
        assert sha(path)==source_hash and sha(REGISTRY)==registry_hash
        write_json(WORK/'paired_analysis.json',paired)
        summary=dict(training_updates=0,calibration_crops=1336,holdout_used_for_calibration=False,
            only_bn_buffers_changed=True,changed_bn_buffers=len(changed),all_weights_and_affine_unchanged=True,
            source_and_registry_unchanged=True,production_promoted=False,
            retention={arm:{p:r['retention'] for p,r in prec.items()} for arm,prec in results.items()},paired=paired)
        write_json(WORK/'report.json',summary)
        lines=['# V39: V38 YOLO-only forward BN swap','',
            'One FP32 forward pass through1336YOLOtrainingcrops; all49BNlayers recalibrated. No gradient/optimizer updates. '
            'Pooled activation-count-weighted means/variances include ordinary padding. All weights/affine bit-identical. '
            'Holdout135crops/13menus never used for calibration. Source V38/registry unchanged; no promotion.', '',
            '| Dataset | V38 FP16 CER | YOLO-BN FP16 CER | V38 BF16 CER | YOLO-BN BF16 CER |','|---|---:|---:|---:|---:|']
        for name in views:
            values=[results[arm][p]['validation'][name]['0.0']['all']['cer'] for p in ('fp16','bf16') for arm in ('v38','bn_yolo')]
            lines.append('| '+name+' | '+' | '.join(f'{v:.2%}' for v in values)+' |')
        lines+=['','Revised retention failures:']
        for arm,prec in results.items():
            for p,r in prec.items():lines.append(f"- {arm}/{p}: {r['retention']['failed']}")
        lines+=['', 'See paired_analysis.json for train/holdout menu-cluster bootstrap intervals. '
            'Calibration uses BN batch-mode upstream activations, not an iterative inference-mode fixed point. '
            'A negative result would not rule out weaker momentum adaptation or partial-layer calibration, which are not run here.']
        (WORK/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        with zipfile.ZipFile(WORK/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in (Path(__file__),Path('src/recognition/bn_calibration.py')):z.write(p,p.as_posix())
        status(state='completed',training_updates=0,calibration_crops=1336,production_promoted=False)
        print('\n'.join(lines),flush=True)
    except Exception:status(state='failed',error=traceback.format_exc());raise


if __name__=='__main__':run()
