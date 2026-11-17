"""Explicit user-authorized menu-domain promotion; preserve V38 rollback."""
from datetime import datetime, timezone
import copy
import json
from pathlib import Path
import torch

from src.recognition.current import REGISTRY, load_current_recognizer
from src.recognition.checkpointing import save_atomic
from src.recognition.model import MenuRecognizer
from src.recognition.utils import sha, write_json

WORK=Path('outputs/recognition/pilots/v43_digits_prices_2000')
SOURCE=WORK/'final_raw.pt'
TARGET=Path('checkpoints/menu_recognizer_v43.pt')
SOURCE_HASH='ec27f47bf0c68c6cafca08777c23eb4387947e2cf2a3286008412d9099e2f49b'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    torch.set_num_threads(4)
    previous=read(REGISTRY)
    assert previous['version'] in ('V38','V43'),'Unexpected active model; refusing replacement'
    assert sha(SOURCE)==SOURCE_HASH
    saved=torch.load(SOURCE,map_location='cpu',weights_only=False)
    assert saved['updates']==2000 and saved['checkpoint_arm']=='raw'
    mapping=saved['vocab']['char_to_id'];assert len(mapping)==1779 and saved['vocab'].get('blank_id',0)==0
    v38=Path('checkpoints/menu_recognizer_v38.pt');v38_hash=sha(v38)
    old=torch.load(v38,map_location='cpu',weights_only=False)
    assert len(old['vocab']['char_to_id'])==1722
    assert all(mapping[c]==i for c,i in old['vocab']['char_to_id'].items())
    bnkeys=[k for k in saved['model'] if k.endswith(('running_mean','running_var','num_batches_tracked'))]
    assert len(bnkeys)==147 and all(torch.equal(saved['model'][k],old['model'][k]) for k in bnkeys)
    evals={p:read(WORK/f'step_2000_raw_{p}.json') for p in ('fp16','bf16')}
    rollback=copy.deepcopy(previous) if previous['version']=='V38' else read(WORK/'promotion_previous_registry.json')
    assert rollback['version']=='V38' and rollback['sha256']==v38_hash
    decision=dict(version='V43',source=SOURCE.as_posix(),source_sha256=SOURCE_HASH,arm='raw',update=2000,
        user_authorized=True,promoted_utc=datetime.now(timezone.utc).isoformat(),
        reason='User explicitly selected V43 after matched V38/V43 comparison; prioritize YOLO-menu crops, digits and long lines over synthetic short-text retention.',
        retention_policy='menu_adaptation_v2',retention_override=True,
        retention_override_scope='This specific V43 final raw promotion only; gate thresholds and future checkpoint selection are unchanged.',
        retention_failed={p:r['retention']['failed'] for p,r in evals.items()},
        evaluation_caveat='Reused development menus and VLM-generated labels, not independently verified production evidence.',
        comparison='outputs/recognition/evaluation/v38_vs_v43_promotion/comparison.json',
        rollback_registry=rollback,bn147_buffers_equal_v38=True)
    if not TARGET.exists():
        save_atomic(TARGET,dict(model=saved['model'],vocab=saved['vocab'],config=saved['config'],updates=2000,
                               checkpoint_arm='raw',promotion=decision))
    artifact=torch.load(TARGET,map_location='cpu',weights_only=False)
    assert artifact['vocab']==saved['vocab'] and artifact['updates']==2000 and artifact['checkpoint_arm']=='raw'
    assert artifact['model'].keys()==saved['model'].keys()
    assert all(torch.equal(artifact['model'][k],v) for k,v in saved['model'].items())
    # Validate the complete inference artifact before changing the active registry.
    model=MenuRecognizer(len(mapping)+1,use_bilstm=True).eval()
    model.load_state_dict(artifact['model'],strict=True);model.vocabulary=mapping
    shapes={}
    with torch.inference_mode():
        for width in (128,256,384,512):
            logits=model(torch.zeros(1,3,64,width),torch.tensor([0]),torch.tensor([width//4]))
            assert logits.shape==(width//4,1,1780) and torch.isfinite(logits).all()
            shapes[str(width)]=list(logits.shape)
    metadata=dict(version='V43',checkpoint=TARGET.as_posix(),sha256=sha(TARGET),source=SOURCE.as_posix(),
        source_sha256=SOURCE_HASH,update=2000,arm='raw',characters=1779,blank_id=0,
        retention_policy='menu_adaptation_v2',retention_override=True,
        normalization=dict(mean=list(saved['config']['mean']),std=list(saved['config']['std'])),
        label_normalization='Full-width decimal digits U+FF10..U+FF19 to ASCII; no other compatibility normalization.',
        inference=copy.deepcopy(rollback['inference']),
        predecessor=dict(version='V38',checkpoint=v38.as_posix(),sha256=v38_hash),
        promotion_record=(WORK/'promotion.json').as_posix())
    if previous['version']=='V43':assert previous==metadata
    else:write_json(WORK/'promotion_previous_registry.json',rollback)
    decision.update(promoted_checkpoint=TARGET.as_posix(),promoted_sha256=metadata['sha256'],
                    strict_load_and_cpu_bucket_forward_pass=True,output_shapes=shapes)
    write_json(TARGET.with_suffix('.metadata.json'),metadata)
    write_json(WORK/'promotion.json',decision)
    write_json(REGISTRY,metadata)
    try:
        loaded,checkpoint=load_current_recognizer('cpu')
        assert checkpoint['vocab']==saved['vocab'] and not loaded.training
        assert all(torch.equal(loaded.state_dict()[k],v) for k,v in saved['model'].items())
        assert sha(SOURCE)==SOURCE_HASH and sha(v38)==v38_hash
    except Exception:
        write_json(REGISTRY,previous)
        raise
    decision.update(source_and_v38_unchanged=True,registry_loader_verified=True)
    write_json(WORK/'promotion.json',decision)
    print(json.dumps(dict(promoted=metadata,verification=decision['output_shapes'],source_and_v38_unchanged=True),indent=2))


if __name__=='__main__':main()
