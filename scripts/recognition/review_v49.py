"""Audit V49 consumed data, selective row updates, and matched menu effects."""
from collections import Counter,defaultdict
from pathlib import Path
import json,string
import torch
from scripts.recognition.prepare_v49 import WORK,START,ROOT,V34,V40,V44,INPUT
from scripts.recognition.train_v45_latin_head import rows
from scripts.recognition.prepare_v44 import expanded_state
from scripts.recognition.review_v38_yolo25 import cluster_bootstrap
from outputs.recognition.v36_yolo_fullpasses import load
from src.recognition.current import REGISTRY,current_checkpoint,load_current_recognizer
from src.recognition.utils import sha,write_json


def main():
    torch.set_num_threads(4);report=load(WORK/'report.json');sampling=load(WORK/'sampling.json');plan=load(WORK/'batches.json')
    assert report['completed'] and report['updates']==2000 and report['presentations']==64000 and len(plan)==2000
    assert sha(WORK/'batches.json')==sampling['plan_sha256']
    source=torch.load(START,map_location='cpu',weights_only=False);vocab=load(WORK/'vocab.json');mapping=vocab['char_to_id']
    assert len(source['vocab']['char_to_id'])==1779 and len(mapping)==1812 and vocab.get('blank_id',0)==0
    assert all(mapping[c]==i for c,i in source['vocab']['char_to_id'].items())
    initial=expanded_state(source,vocab);letters=string.ascii_letters;symbols=string.digits+'()（）'
    selected={mapping[c] for c in letters+symbols};assert len(selected)==66
    keep=[i for i in range(1813) if i not in selected]
    checkpoint_audit={};row_deltas={}
    for arm in ('raw','ema'):
        path=WORK/f'final_{arm}.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['updates']==2000 and ck['checkpoint_arm']==arm and ck['vocab']==vocab
        for field in ('model','raw_training_model','ema_training_model'):
            state=ck[field]
            for k,v in initial.items():
                if k.startswith('classifier.'):continue
                assert torch.equal(state[k],v),(arm,field,k)
            if field=='model':
                for suffix in ('weight','bias'):assert torch.equal(state['classifier.'+suffix][keep],initial['classifier.'+suffix][keep])
            else:
                for suffix in ('weight','bias'):assert torch.equal(state['classifier.base.'+suffix],initial['classifier.'+suffix])
        buffers=[k for k in initial if k.endswith(('running_mean','running_var','num_batches_tracked'))];assert len(buffers)==147
        rates=defaultdict(set);steps=defaultdict(set)
        for group in ck['optimizer']['param_groups']:
            rates[group['role']].add(group['base_lr'])
            for p in group['params']:steps[group['role']].add(int(ck['optimizer']['state'][p]['step']))
        assert rates==dict(latin={.001},digits_brackets={.00001}) and steps==dict(latin={2000},digits_brackets={2000})
        delta={c:float((ck['model']['classifier.weight'][mapping[c]]-initial['classifier.weight'][mapping[c]]).norm()) for c in letters+symbols}
        assert all(v>0 for v in delta.values());row_deltas[arm]=delta
        checkpoint_audit[arm]=dict(sha256=sha(path),frozen_shared_features_including_bilstm=True,
            frozen_bn_buffers=147,blank_and_all_other_classifier_rows_bit_identical=True,
            row_groups={k:dict(lr=next(iter(rates[k])),optimizer_steps=next(iter(steps[k]))) for k in rates},all66_rows_changed=True)
    best=torch.load(WORK/'best.pt',map_location='cpu',weights_only=False);raw1500=torch.load(WORK/'step_1500.pt',map_location='cpu',weights_only=False)
    assert report['best_update']==best['updates']==1500 and report['best_arm']==best['checkpoint_arm']=='raw'
    assert all(torch.equal(v,raw1500['model'][k]) for k,v in best['model'].items())
    p47=Path('outputs/recognition/pilots/v47_clean_latin_fit_1000/targets');p48=Path('outputs/recognition/pilots/v48_parentheses_2000/targets')
    multi=rows(V44/'targeted/labels.jsonl');yc=rows(V40/'yolo_labels.jsonl');yt=[r for r in yc if r['split']=='train'];hold=[r for r in yc if r['split']=='val_holdout']
    assert len(yt)==1561 and len(hold)==156 and not {r['source_menu'] for r in yt}&{r['source_menu'] for r in hold}
    assert not {r['sha256'] for r in yt}&{r['sha256'] for r in hold}
    dataset=dict(original=([r for r in rows(ROOT/'labels.jsonl') if r['split']=='train'],ROOT),
        generated=(rows(V34/'corpus/labels.jsonl'),V34/'corpus'),yolo=(yt,INPUT),
        target=([r for r in rows(p47/'labels.jsonl') if r['split']=='train'],p47),
        latin_multi=([r for r in multi if r['kind']=='latin' and r['split']=='train' and int(Path(r['image']).stem)%5!=0 and 1<=len(r['text'])<=6 and r['canvas'] in (128,256)],V44/'targeted'),
        brackets=([r for r in rows(p48/'labels.jsonl') if r['split']=='train'],p48),
        digits=([r for r in rows(START.parent/'digits/labels.jsonl') if r['split']=='train'],START.parent/'digits'))
    streams=Counter();chars=Counter();terms=Counter();unique=defaultdict(set);styles=Counter();teacher=Counter();images={};fonts=Counter()
    for batch in plan:
        assert len(batch)==32 and len({k for k,i in batch})==1
        for k,i in batch:
            rr,root=dataset[k];r=rr[i];t=r['text'];streams[k]+=1;chars.update(t);terms[t]+=1;unique[k].add(i)
            assert not any('\uff10'<=c<='\uff19' for c in t)
            images[root/r['image']]=r['sha256'];fonts[Path(r.get('font_path',r.get('preferred_font',r.get('font','unrecorded')))).name]+=1
            if k=='brackets':styles[r['style']]+=1
            if k in ('original','generated','yolo'):teacher['excluded_latin_label' if any(c in string.ascii_letters for c in t) else 'eligible']+=1
    assert dict(streams)==report['streams']==sampling['streams'] and dict(chars)==report['character_encounters']==sampling['character_encounters']
    assert styles==sampling['parentheses_styles']==dict(ascii=3200,fullwidth=3200)
    assert teacher==sampling['teacher_replay']==dict(eligible=25529,excluded_latin_label=71)
    assert len(unique['yolo'])==1561 and len(unique['digits'])==6000
    assert len(sampling['required_term_encounters'])==521 and all(terms[t]==n>=2 for t,n in sampling['required_term_encounters'].items())
    for p,h in images.items():assert p.is_file() and sha(p)==h,str(p)
    for r in hold:assert sha(INPUT/r['image'])==r['sha256']
    parity=load(WORK/'initial_parity.json');assert parity['passed'] and all(n==0 for p in parity['disagreements'].values() for n in p.values())
    effects={};false_latin={}
    for precision in ('fp16','bf16'):
        initial_eval=load(WORK/f'initial_{precision}.json');effects[precision]={};false_latin[precision]={}
        for step,arm in [(1500,'raw'),(2000,'raw'),(2000,'ema')]:
            d=load(WORK/f'step_{step:04d}_{arm}_{precision}.json')
            a=initial_eval['validation']['yolo_current_holdout']['0.0']['predictions'];b=d['validation']['yolo_current_holdout']['0.0']['predictions']
            e=cluster_bootstrap(a,b,hold);e['v49_minus_v43_pp']=e.pop('v38_minus_v36_pp');effects[precision][f'{step}_{arm}']=e
            added=sampling['added_letters']
            false_latin[precision][f'{step}_{arm}']={name:sum(any(c in added for c in r['prediction']) for r in d['validation'][name]['0.0']['predictions'] if not any(c in added for c in r['truth'])) for name in ('yolo_current_holdout','seen','historical_bucket','fresh_short_clean')}
    write_json(WORK/'paired_analysis.json',effects)
    model,current=load_current_recognizer('cpu');assert len(current['vocab']['char_to_id'])==1779 and load(REGISTRY)['version']=='V43'
    with torch.inference_mode():assert model(torch.zeros(1,3,64,128),torch.tensor([0]),torch.tensor([32])).shape==(32,1,1780)
    audit=dict(passed=True,updates=2000,presentations=64000,minutes=report['elapsed_seconds']/60,
        stream_counts=dict(streams),unique_records={k:len(v) for k,v in unique.items()},required_terms=521,minimum_term_encounters=2,
        synthetic_font_presentations=dict(fonts),parentheses_presentations=dict(styles),teacher_eligibility=dict(teacher),
        unique_consumed_image_hashes_verified=len(images),heldout_image_hashes_verified=156,
        checkpoint_audit=checkpoint_audit,selected_checkpoint=dict(update=1500,arm='raw',sha256=sha(WORK/'best.pt')),
        row_delta_norms=row_deltas,false_new_latin_emission_lines=false_latin,
        teacher_evidence='FullsoftmaxKLweight1T2validframesnonLatinreplay policy inspected; teacher full-state identity asserted at training completion.',
        active_registry='V43',active_sha256=sha(current_checkpoint()),source_sha256=sha(START),
        v38_rollback_sha256=sha(Path('checkpoints/menu_recognizer_v38.pt')),production_unchanged=True)
    write_json(WORK/'completion_verification.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('row_delta_norms','synthetic_font_presentations')},indent=2));print('PAIRED',json.dumps(effects))


if __name__=='__main__':main()
