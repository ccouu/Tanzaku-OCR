"""User-authorized V38 raw promotion under menu_adaptation_v2; no training."""
from datetime import datetime, timezone
import json
from pathlib import Path
import torch

from scripts.recognition.train_v38_yolo25 import WORK, START, SOURCE
from outputs.recognition.v36_yolo_fullpasses import load
from src.recognition.retention import apply_retention_policy, POLICY_PATH
from src.recognition.checkpointing import save_atomic
from src.recognition.current import REGISTRY, load_current_recognizer
from src.recognition.utils import sha, write_json


def main():
    torch.set_num_threads(4)
    source_path=WORK/'final_raw.pt';source_hash=sha(source_path);v36_hash=sha(START);old_best_hash=sha(WORK/'best.pt')
    legacy=load(WORK/'report.json');assert legacy['updates']==686
    evaluations={}
    candidates=[]
    for step in (173,344,515,686):
        for arm in ('raw','ema'):
            results={p:load(WORK/f'step_{step:04d}_{arm}_{p}.json') for p in ('fp16','bf16')}
            adjusted={p:apply_retention_policy(r['retention']) for p,r in results.items()}
            candidates.append(dict(update=step,arm=arm,score=results['fp16']['score'],
                                   eligible=all(g['eligible'] for g in adjusted.values()),retention=adjusted))
            if step==686 and arm=='raw':evaluations=results
    chosen=min((r for r in candidates if r['eligible']),key=lambda r:r['score'])
    assert (chosen['update'],chosen['arm'])==(686,'raw')
    saved=torch.load(source_path,map_location='cpu',weights_only=False)
    bnsource=torch.load(SOURCE,map_location='cpu',weights_only=False)['model']
    keys=[k for k in bnsource if k.endswith(('running_mean','running_var','num_batches_tracked'))]
    assert len(keys)==147 and all(torch.equal(saved['model'][k],bnsource[k]) for k in keys)
    target=Path('checkpoints/menu_recognizer_v38.pt')
    decision=dict(version='V38',arm='raw',update=686,epoch=4,source=str(source_path).replace('\\','/'),source_sha256=source_hash,
        retention_policy=load(POLICY_PATH),old_policy_failures={p:r['retention']['failed'] for p,r in evaluations.items()},
        new_policy_retention=chosen['retention'],user_authorized=True,
        reason='User prioritizes significant YOLO-crop gain for intended menu use over marginal synthetic loss.',
        evaluation_caveat='Policy adopted after observing V38. VLM labels not independently verified; menu holdout used for selection, not independent final testing.',
        fp16_metrics={name:r['0.0']['all'] for name,r in evaluations['fp16']['validation'].items()},
        frozen_bn_buffers=147,predecessor=dict(version='V36',checkpoint=str(START).replace('\\','/'),sha256=v36_hash),
        promoted_utc=datetime.now(timezone.utc).isoformat())
    if not target.exists():
        # Inference/continuation weights, without stale legacy-best metadata or
        # duplicate raw/EMA/optimizer recovery payloads. Source remains intact.
        artifact=dict(model=saved['model'],vocab=saved['vocab'],config=saved['config'],
                      updates=686,epoch=4,checkpoint_arm='raw',promotion=decision)
        save_atomic(target,artifact)
    artifact=torch.load(target,map_location='cpu',weights_only=False)
    assert artifact['vocab']==saved['vocab'] and artifact['updates']==686
    assert all(torch.equal(artifact['model'][k],v) for k,v in saved['model'].items())
    metadata=dict(version='V38',checkpoint=target.as_posix(),sha256=sha(target),source=decision['source'],
                  source_sha256=source_hash,update=686,arm='raw',retention_policy='menu_adaptation_v2',
                  normalization=dict(mean=list(saved['config']['mean']),std=list(saved['config']['std'])),
                  inference=dict(height=64,channels='RGB',width_buckets=[128,256,384,512],
                                 padding='replicated edge',valid_lengths='ceil(valid_width/4)',blank_logit_penalty=0,
                                 use_bilstm=True,orientation_flag='0 horizontal;1 vertical rotated CCW'),
                  predecessor=decision['predecessor'],promotion_record=(WORK/'promotion.json').as_posix())
    if REGISTRY.exists():
        previous=load(REGISTRY)
        assert previous==metadata,'Refusing to replace an unexpected active model registry'
    write_json(REGISTRY,metadata)
    model,loaded=load_current_recognizer('cpu')
    with torch.inference_mode():
        logits=model(torch.zeros(1,3,64,128),torch.tensor([0]),torch.tensor([32]))
    assert tuple(logits.shape)==(32,1,len(saved['vocab']['char_to_id'])+1)
    assert sha(source_path)==source_hash and sha(START)==v36_hash and sha(WORK/'best.pt')==old_best_hash
    decision.update(promoted_checkpoint=target.as_posix(),promoted_sha256=metadata['sha256'],
                    original_checkpoints_unchanged=True,legacy_best_preserved_as_V36=True,
                    registry=str(REGISTRY),strict_load_and_cpu_forward_pass=True)
    write_json(WORK/'promotion.json',decision)
    write_json(WORK/'retention_policy_v2_candidates.json',candidates)
    write_json(target.with_suffix('.metadata.json'),metadata)
    print(json.dumps(dict(promoted=metadata,original_checkpoints_unchanged=True),indent=2))


if __name__=='__main__':main()
