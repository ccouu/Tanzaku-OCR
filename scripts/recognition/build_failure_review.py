"""Build versioned training-only failure queues from verified saved predictions."""
import argparse
from collections import Counter
from pathlib import Path
import json

from src.common import dump_json,digest
from src.recognition.metrics import align_errors

PROJECT=Path(__file__).resolve().parents[2]
RUN=PROJECT/'outputs/recognition/training/v38_yolo25_fullpasses4'
INPUT=PROJECT/'data/processed/gemini_banana_mobile_menus_50_crops'
OUT=PROJECT/'outputs/recognition/review/v38_training_failures'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def build(version='v38',output=None):
    if version=='v38':
        run=RUN;out=output or OUT;step=686;tag='V38';baseline_name='V36'
        image_root=run/'yolo';labels=run/'yolo/labels.jsonl';hold_labels=None
        checkpoint=PROJECT/'checkpoints/menu_recognizer_v38.pt'
    elif version=='v42':
        run=PROJECT/'outputs/recognition/pilots/v42_yolo35_jitter_long512_2000'
        out=output or PROJECT/'outputs/recognition/review/v42_training_failures'
        step=2000;tag='V42';baseline_name='V41 EMA1000'
        image_root=INPUT;labels=run/'yolo_base_labels.jsonl';hold_labels=run/'yolo_holdout_labels.jsonl'
        checkpoint=run/'final_raw.pt'
    else:raise ValueError('Unknown model version')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    import torch
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['updates']==step and saved.get('checkpoint_arm','raw')=='raw'
    checkpoint_sha=digest(checkpoint)
    if (out/'index.json').exists():
        assert read(out/'index.json')['metadata']['checkpoint_sha256']==checkpoint_sha,'Use a new output folder for another checkpoint'
    prediction_file=run/f'step_{step:04d}_raw_fp16.json'
    predictions=read(prediction_file)['validation']['yolo_current_train']['0.0']
    before=read(run/'initial_fp16.json');before=before.get('validation',before)
    baseline=before['yolo_current_train']['0.0']['predictions']
    source=read(INPUT/'crops_manifest.json');menu_root=Path(source['input_dir'])
    yolo=[json.loads(s) for s in labels.read_text(encoding='utf-8').splitlines() if s.strip()]
    train=[r for r in yolo if r['split']=='train'];hold={r['source_menu'] for r in yolo if r['split']=='val_holdout'}
    if hold_labels:hold={json.loads(s)['source_menu'] for s in hold_labels.read_text(encoding='utf-8').splitlines() if s.strip()}
    assert len(train)==len(predictions['predictions'])==len(baseline)==predictions['all']['lines']
    assert not {r['source_menu'] for r in train}&hold
    assets={};failures=[]
    for i,(r,p,b) in enumerate(zip(train,predictions['predictions'],baseline)):
        assert p['index']==b['index']==i and p['truth']==b['truth']==r['text']
        image=image_root/r['image'];assert digest(image)==r['sha256']
        if p['truth']==p['prediction']:continue
        ident=r['crop_id'];native=INPUT/r['display_crop'];menu=menu_root/r['source_image']
        assert native.resolve().is_relative_to(INPUT.resolve()) and menu.resolve().is_relative_to(menu_root.resolve())
        assert native.is_file() and menu.is_file()
        alignment=align_errors(p['truth'],p['prediction'])
        counts=Counter(op['type'] for op in alignment if op['type']!='match')
        assert all(counts[k]==p['error_counts'][k] for k in ('deletion','insertion','substitution'))
        for key,path in ((f'input/{ident}',image),(f'native/{ident}',native),(f'menu/{r["source_menu"]}',menu)):
            assets[key]=path.resolve().relative_to(PROJECT).as_posix()
        edits=sum(p['error_counts'].values());old_edits=sum(b['error_counts'].values())
        failures.append(dict(id=ident,source_menu=r['source_menu'],orientation=r['orientation_flag'],
            label=p['truth'],prediction=p['prediction'],initial_prediction=b['prediction'],
            errors=p['error_counts'],edits=edits,cer=edits/len(p['truth']),delta_edits=edits-old_edits,
            alignment=alignment,confidence=r['confidence'],review_required=r.get('review_required',False),
            source_ids=r.get('source_ids',[]),canvas=r['canvas'],valid_width=r['valid_width'],
            core=r['obb_polygon'],padded=r['padded_polygon'],input_sha256=r['sha256'],
            input_url=f'/assets/input/{ident}',native_url=f'/assets/native/{ident}',menu_url=f'/assets/menu/{r["source_menu"]}'))
    failures.sort(key=lambda r:(-r['edits'],-r['cer'],r['id']))
    assert len(failures)==len(train)-predictions['all']['exact']
    meta=dict(version=tag,baseline_name=baseline_name,model=f'{tag} final raw · update{step} · FP16 greedy CTC β=0',checkpoint=checkpoint.relative_to(PROJECT).as_posix(),
        checkpoint_sha256=checkpoint_sha,prediction_file=str(prediction_file.relative_to(PROJECT)),
        predictions_sha256=digest(prediction_file),total_training_crops=len(train),correct=predictions['all']['exact'],failures=len(failures),
        training_cer=predictions['all']['cer'],error_counts=predictions['all']['error_counts'],
        empty_predictions=sum(not r['prediction'] for r in failures),holdout_included=False,
        note='These are training crops and VLM-provided labels, not independently verified ground truth. Review decisions are saved separately; no training labels or model weights are modified.')
    vocabulary=saved['vocab']['char_to_id'];assert digest(checkpoint)==checkpoint_sha
    dump_json(out/'index.json',dict(metadata=meta,rows=failures,vocab_chars=list(vocabulary)))
    dump_json(out/'assets.json',dict(allowed_roots=sorted({str(p.relative_to(PROJECT)).replace('\\','/') for p in (INPUT,image_root,menu_root)}),assets=assets))
    dump_json(out/'build_audit.json',dict(training_only=True,training_inputs_hash_verified=len(train),failure_rows=len(failures),
        source_checkpoint_unchanged=True,no_new_inference=True,no_visual_model_calls=True,asset_count=len(assets)))
    print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--version',choices=['v38','v42'],default='v38');parser.add_argument('--output',type=Path)
    args=parser.parse_args();build(args.version,args.output)
