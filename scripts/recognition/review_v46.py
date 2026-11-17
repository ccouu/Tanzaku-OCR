"""Independent checkpoint, consumed-plan and cleanup verification for V46."""
from collections import Counter,defaultdict
from pathlib import Path
import json
import torch

from scripts.recognition.prepare_v46 import WORK,START
from scripts.recognition.train_v45_latin_head import rows,V44,V40,ROOT,V34
from scripts.recognition.train_v38_yolo25 import INPUT,HOLDOUT
from scripts.recognition.review_v38_yolo25 import cluster_bootstrap
from outputs.recognition.v36_yolo_fullpasses import load
from src.recognition.current import REGISTRY,load_current_recognizer,current_checkpoint
from src.recognition.utils import sha,write_json


def main():
    torch.set_num_threads(4)
    s=load(WORK/'sampling.json');r=load(WORK/'report.json');plan=load(WORK/'batches.json')
    assert r['completed'] and r['updates']==4000 and r['presentations']==128000 and not r['time_limited']
    assert sha(WORK/'batches.json')==s['plan_sha256']
    registry=load(REGISTRY);assert registry['version']=='V43'
    assert sha(START)==registry['source_sha256']
    source=torch.load(START,map_location='cpu',weights_only=False)
    vocab=load(WORK/'vocab.json');old=source['vocab']['char_to_id'];mapping=vocab['char_to_id']
    assert len(old)==1779 and len(mapping)==1812 and vocab.get('blank_id',0)==0
    assert all(mapping[c]==i for c,i in old.items())
    targeted=rows(V44/'targeted/labels.jsonl');yolo=rows(V40/'yolo_labels.jsonl')
    train=[x for x in yolo if x['split']=='train'];hold=[x for x in yolo if x['split']=='val_holdout']
    assert len(train)==1561 and len(hold)==156 and not {x['source_menu'] for x in train}&HOLDOUT
    assert {x['source_menu'] for x in hold}==HOLDOUT
    assert not {x['sha256'] for x in train}&{x['sha256'] for x in hold}
    original=[x for x in rows(ROOT/'labels.jsonl') if x['split']=='train'];generated=rows(V34/'corpus/labels.jsonl')
    data=dict(original=(original,ROOT),generated_clean=(generated,V34/'corpus'),generated_noisy=(generated,V34/'corpus'),
        yolo=(train,INPUT),latin=(rows(WORK/'replay/latin_train_labels.jsonl'),V44/'targeted'),
        basic=([x for x in rows(WORK/'basic_japanese/labels.jsonl') if x['split']=='train'],WORK/'basic_japanese'),
        parentheses=([x for x in targeted if x['kind']=='parentheses' and x['split']=='train' and int(Path(x['image']).stem)%5!=0],V44/'targeted'),
        digits=([x for x in rows(START.parent/'digits/labels.jsonl') if x['split']=='train'],START.parent/'digits'))
    counts=Counter();chars=Counter();terms=Counter();unique=defaultdict(set);fonts=Counter();inputs={}
    for batch in plan:
        assert len(batch)==32 and len({k for k,i in batch})==1
        for k,i in batch:
            row,root=data[k][0][i],data[k][1];counts[k]+=1;chars.update(row['text']);terms[row['text']]+=1;unique[k].add(i)
            assert not any('\uff10'<=c<='\uff19' for c in row['text'])
            font=row.get('font_path',row.get('preferred_font',row.get('font','unrecorded')));fonts[str(font)]+=1
            p=root/row['image'];inputs[p]=row['sha256']
    assert dict(counts)==r['stream_encounters']==s['streams'] and dict(chars)==r['actual_character_encounters']
    assert all(terms[t]==r['required_term_encounters'][t]>=4 for t in s['required_term_encounters'])
    assert len(unique['yolo'])==1561 and len(unique['digits'])==6000 and len(unique['basic'])==1056
    for p,h in inputs.items():assert p.is_file() and sha(p)==h,str(p)
    for row in hold:assert sha(INPUT/row['image'])==row['sha256']
    manifest=load(WORK/'basic_japanese/manifest.json')
    assert len(manifest['anchors'])==176 and all(chars[c]>0 for c in manifest['anchors'])
    # Per-stream emitted font metadata is reported without assuming basename and
    # full-path forms represent different font families.
    font_names=Counter()
    for f,n in fonts.items():font_names[Path(f).name]+=n
    assert not set(manifest['heldout_fonts'])&set(manifest['known_fonts'])
    for path in manifest['known_fonts']+manifest['heldout_fonts']:assert Path(path).is_file()
    bn=[k for k in source['model'] if k.endswith(('running_mean','running_var','num_batches_tracked'))];assert len(bn)==147
    checkpoints={};optimizer_audit={}
    for arm in ('raw','ema'):
        path=WORK/f'final_{arm}.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['updates']==4000 and ck['checkpoint_arm']==arm and ck['vocab']==vocab
        for field in ('model','raw_training_model','ema_training_model'):
            for k,v in source['model'].items():
                if k.startswith(('recurrent.','classifier.')):continue
                assert torch.equal(ck[field][k],v),(arm,field,k)
        assert all(torch.equal(ck['model'][k],source['model'][k]) for k in bn)
        assert any(not torch.equal(ck['model'][k],v) for k,v in source['model'].items() if k.startswith('recurrent.'))
        assert not torch.equal(ck['model']['classifier.weight'][:1780],source['model']['classifier.weight'])
        assert all(ck['model']['classifier.weight'][i].ne(ck['model']['classifier.weight'][0]).any() for i in range(1780,1813))
        steps=defaultdict(set)
        for group in ck['optimizer']['param_groups']:
            for p in group['params']:
                state=ck['optimizer']['state'][p]
                steps[group['role']].add(int(state['step']))
        assert steps==dict(recurrent={3800},old_classifier={3800},new_classifier={4000})
        optimizer_audit[arm]={k:sorted(v) for k,v in steps.items()};checkpoints[arm]=dict(path=str(path),sha256=sha(path))
    assert r['best_update'] is None and not (WORK/'best.pt').exists()
    parity=load(WORK/'initial_parity.json')
    assert parity['passed'] and all(n==0 for p in parity['prediction_disagreements'].values() for n in p.values())
    paired={}
    for precision in ('fp16','bf16'):
        baseline=load(WORK/f'initial_{precision}.json')['validation']['yolo_current_holdout']['0.0']['predictions'];paired[precision]={}
        for arm in ('raw','ema'):
            final=load(WORK/f'step_4000_{arm}_{precision}.json')['validation']['yolo_current_holdout']['0.0']['predictions']
            effect=cluster_bootstrap(baseline,final,hold);effect['v46_minus_v43_pp']=effect.pop('v38_minus_v36_pp');paired[precision][arm]=effect
    write_json(WORK/'paired_analysis.json',paired)
    cleanup=Path('outputs/recognition/cleanup/v46_post_training')
    deletion=load(cleanup/'deletion_manifest.json');cr=load(cleanup/'report.json');journal=rows(cleanup/'deleted.jsonl')
    assert cr['completed'] and len(journal)==cr['deleted_files']==26015
    assert sum(x['bytes'] for x in journal)==cr['deleted_bytes']==13431880400
    allowed=[Path(x).resolve() for x in deletion['allowed_roots']];root=Path.cwd().resolve()
    for row in journal:
        p=(root/row['path']).resolve();assert p.is_relative_to(root) and any(p.is_relative_to(a) for a in allowed)
        assert not p.exists(),str(p)
        assert not p.is_relative_to(root/'checkpoints') and not p.is_relative_to(root/'data/raw')
        if p.suffix.lower() in ('.pt','.bin'):assert p.is_relative_to(root/'.cache')
    assert Path('data/processed/gemini_recognition').is_dir(),'Non-proven duplicate must be preserved'
    model,ck=load_current_recognizer('cpu');assert len(ck['vocab']['char_to_id'])==1779
    with torch.inference_mode():assert model(torch.zeros(1,3,64,128),torch.tensor([0]),torch.tensor([32])).shape==(32,1,1780)
    verify=dict(passed=True,updates=4000,presentations=128000,minutes=r['elapsed_seconds']/60,time_limited=False,
        exact_stream_counts=dict(counts),unique_records={k:len(v) for k,v in unique.items()},required_exact_terms=521,
        minimum_term_encounters=min(terms[t] for t in s['required_term_encounters']),
        fonts_presentations=dict(font_names),basic_anchors=176,minimum_basic_character_encounters=min(chars[c] for c in manifest['anchors']),
        unique_training_image_hashes_verified=len(inputs),heldout_image_hashes_verified=156,checkpoint_audit=checkpoints,
        frozen_cnn_projection_condition_temporal_bn_affine_and147buffers_verified=True,recurrent_and_old_classifier_changed=True,
        optimizer_steps=optimizer_audit,warmup_evidence='Runtime equality assertions plus optimizer counters; no step200 checkpoint independently available.',
        teacher_evidence='Runtime full-state equality assertion at each evaluation; teacher not stored as separate final checkpoint.',
        no_best_checkpoint=True,cleanup_deleted_files=cr['deleted_files'],cleanup_deleted_bytes=cr['deleted_bytes'],cleanup_deleted_GiB=cr['deleted_GiB'],
        deletion_journal_allowlist_and_absence_verified=True,active_registry='V43',active_sha256=sha(current_checkpoint()),
        v38_rollback_sha256=sha(Path('checkpoints/menu_recognizer_v38.pt')),source_and_registry_consistent=True)
    write_json(WORK/'completion_verification.json',verify)
    print(json.dumps(verify,ensure_ascii=False,indent=2))
    print('PAIRED',json.dumps(paired))


if __name__=='__main__':main()
