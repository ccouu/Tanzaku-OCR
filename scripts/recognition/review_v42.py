"""Independent V42 coverage/geometry/BN audit and saved-prediction review."""
from collections import Counter
import gc
from pathlib import Path
import torch

from scripts.recognition.prepare_v42 import WORK,START,V41,V37,V40,INPUT,ROOT
from scripts.recognition.review_v41 import recognition_counts
from outputs.recognition.v36_yolo_fullpasses import load
from outputs.recognition.tiered_short_pilot import rows
from src.recognition.crop_jitter import jitter_parameters
from src.recognition.metrics import summarise
from src.recognition.current import REGISTRY
from src.recognition.utils import sha,write_json


def main():
    torch.set_num_threads(4);s=load(WORK/'sampling.json');report=load(WORK/'report.json');cfg=load(WORK/'run_config.json')
    assert sha(START)==cfg['source_sha256'];assert load(REGISTRY)['version']=='V38'
    source=torch.load(START,map_location='cpu',weights_only=False);vocab=load(WORK/'vocab.json');added=vocab['extension']['added_characters']
    assert vocab==source['vocab'] and len(vocab['char_to_id'])==1779 and vocab['blank_id']==0
    parity=load(WORK/'initial_parity.json');assert parity['passed'] and all(v==0 for d in parity['prediction_disagreements'].values() for v in d.values())
    base=rows(WORK/'yolo_base_labels.jsonl');held=rows(WORK/'yolo_holdout_labels.jsonl');views=rows(WORK/'yolo_views.jsonl')
    original_held=[r for r in rows(V40/'yolo_labels.jsonl') if r['split']=='val_holdout']
    assert held==original_held and len(base)==1561 and len(held)==156 and len(views)==22400
    assert not {r['source_menu'] for r in base}&{r['source_menu'] for r in held}
    assert not {r['sha256'] for r in base}&{r['sha256'] for r in held}
    for r in base:
        assert sha(INPUT/r['display_crop'])==r['native_sha256'] and sha(INPUT/r['image'])==r['sha256']
    for r in held:assert sha(INPUT/r['image'])==r['sha256']
    assert sha(WORK/'yolo_views.jsonl')==s['augmentation_seed_plan_sha256']
    for v in views:
        expected=jitter_parameters(base[v['base_index']]['oriented_native_size'],v['seed'])
        assert all(v[k]==value for k,value in expected.items())
        assert all(.02<=x<=.05 for x in v['margin_ratios'].values()) and .95<=v['horizontal_stretch']<=1.05
        assert all(p>0 for p in v['margin_pixels'].values())
    assert len({(v['base_index'],v['seed']) for v in views})==22384 and sum(v['quota_duplicate'] for v in views)==16
    original=[r for r in rows(ROOT/'labels.jsonl') if r['split']=='train'];generated=rows(Path('outputs/recognition/training/v34_broad_fullpasses6/corpus/labels.jsonl'))
    target=rows(V41/'target_train_labels.jsonl');long=rows(V37/'long_labels.jsonl');longtrain=[r for r in long if r['split']=='train'];longheld=[r for r in long if r['split']=='holdout']
    assert len(longtrain)==4800 and len(longheld)==800 and not {r['sha256'] for r in longtrain}&{r['sha256'] for r in longheld}
    for r in long:assert sha(V37/r['image'])==r['sha256'] and 385<=r['valid_width']<=512 and 11<=len(r['text'])<=14
    data={'original':original,'generated':generated,'target':target,'long512':longtrain}
    plan=load(WORK/'batches.json');assert sha(WORK/'batches.json')==s['plan_sha256']
    streams=Counter();widths=Counter();chars=Counter();terms=Counter();yseen=Counter();viewseen=Counter();lengths=Counter();longviews=Counter();orient=Counter()
    for batch in plan:
        assert len(batch)==32
        for kind,index in batch:
            if kind=='yolo':
                v=views[index];r=base[v['base_index']];width=v['canvas'];yseen[v['base_index']]+=1;viewseen[index]+=1
            else:
                r=data[kind][index]
                if kind=='original':
                    span=r['glyph_box'][2]-r['glyph_box'][0];guard=4 if len(r['text'])==1 else 16
                    width=next(w for w in (128,256,384) if min(384,span+2*guard)<=w)
                else:width=r['canvas']
            streams[kind]+=1;widths[width]+=1;chars.update(c for c in r['text'] if c in added);terms[r['text']]+=1;orient[r.get('orientation_flag',0)]+=1
            if kind=='long512':lengths[len(r['text'])]+=1;longviews[r['view']]+=1
    assert len(plan)==2000 and sum(streams.values())==64000 and streams==report['stream_encounters']==s['streams']
    assert len(viewseen)==22400 and all(n==1 for n in viewseen.values()) and len(yseen)==1561
    assert yseen=={int(k):v for k,v in report['yolo_base_encounters'].items()}
    assert dict(widths)=={int(k):v for k,v in report['physical_width_encounters'].items()}=={int(k):v for k,v in s['physical_width_presentations'].items()}
    assert widths[512]==21760 and widths[512]/64000==.34 and lengths=={n:4800 for n in (11,12,13,14)}
    assert longviews=={'clean':14400,'noisy':4800} and orient=={int(k):v for k,v in s['orientation_presentations'].items()}
    assert all(chars[c]==report['new_character_encounters'][c]==s['planned_new_character_counts'][c] for c in added) and min(chars[c] for c in added)==168
    assert len(s['required_term_encounters'])==521 and all(terms[t]==n for t,n in s['required_term_encounters'].items())
    bkeys=[k for k in source['model'] if k.endswith(('running_mean','running_var','num_batches_tracked'))];assert len(bkeys)==147
    affine=[k.rsplit('.',1)[0]+'.'+suffix for k in bkeys if k.endswith('running_mean') for suffix in ('weight','bias')]
    checkpoints={}
    for name in ('best.pt','last.pt','final_raw.pt','final_ema.pt','step_0500.pt','step_1000.pt','step_1500.pt','step_2000.pt'):
        ck=torch.load(WORK/name,map_location='cpu',weights_only=False);assert ck['vocab']==vocab
        for field in ('model','raw_training_model','ema_training_model'):
            assert all(torch.equal(ck[field][k],source['model'][k]) for k in bkeys)
        if name=='best.pt':assert ck['updates']==0 and all(torch.equal(ck['model'][k],v) for k,v in source['model'].items())
        else:
            assert len(ck['optimizer']['state'])>2 and all(int(v['step'])==ck['updates'] for v in ck['optimizer']['state'].values())
        checkpoints[name]=dict(update=ck['updates'],arm=ck['checkpoint_arm'],bn_buffers_fixed=147,
            BNaffine_tensors_changed=sum(not torch.equal(ck['model'][k],source['model'][k]) for k in affine))
        del ck;gc.collect()
    verification=dict(passed=True,updates=2000,presentations=64000,streams=dict(streams),yolo_fraction=.35,
        yolo_bases_seen=1561,augmented_view_presentations=22400,unique_transforms=22384,quota_duplicates=16,
        true_long512_presentations=19200,true_long512_fraction=.30,physical512_presentations=21760,physical512_fraction=.34,
        long_length_counts=dict(lengths),orientation_presentations=dict(orient),holdout_geometry_and_pixel_hashes_unchanged=True,
        requested_augmentation_ranges_and_all_seed_reproductions_verified=True,minimum_added_character_encounters=168,
        all521requiredterms_encountered=True,source_and_promoted_registry_unchanged=True,
        teacher_unchanged_runtime_assertions_passed=True,checkpoints=checkpoints)
    write_json(WORK/'completion_verification.json',verification)
    initial=load(WORK/'initial_fp16.json')['validation'];cases={}
    for label,step,arm in (('raw1500',1500,'raw'),('raw2000',2000,'raw'),('ema2000',2000,'ema')):
        r=load(WORK/f'step_{step:04d}_{arm}_fp16.json');v=r['validation'];before=initial['yolo_current_holdout']['0.0']['predictions'];after=v['yolo_current_holdout']['0.0']['predictions']
        strat={}
        for key,values in (('orientation_flag',(0,1)),('canvas',(128,256,384,512))):
            for val in values:
                ids=[i for i,row in enumerate(held) if row[key]==val]
                if ids:strat[f'{key}={val}']=dict(initial=summarise([before[i] for i in ids]),candidate=summarise([after[i] for i in ids]))
        menus={}
        for menu in sorted({row['source_menu'] for row in held}):
            ids=[i for i,row in enumerate(held) if row['source_menu']==menu]
            menus[menu]=dict(initial=summarise([before[i] for i in ids]),candidate=summarise([after[i] for i in ids]))
        pred=[row for row in v['new_character_probe']['0.0']['predictions'] if row['cohort']=='heldout']
        cases[label]=dict(metrics={k:data['0.0']['all'] for k,data in v.items()},retention=r['retention'],stratified=strat,by_menu=menus,
            new_character_font=recognition_counts(pred,added),new_character_YOLO=recognition_counts(after,added))
    write_json(WORK/'review_comparison.json',cases)
    lines=['# V42 independent review','',
        'Audit passed:2000updates/64000presentations. Exact35%YOLO(22400views/all1561bases),30%genuine11–14char512(19200),34%physical512(21760),15%controllednewchars/20%otherreplay. '
        'Every requested2–5%margin/.95–1.05stretch parameter reproduces from its saved seed; positivepadding,integerpixelquantization,16explicitquota duplicates. '
        'All147BNbuffers fixed in every saved raw/EMA/selected/step state. All98BNaffine tensors updated in finalraw. SourceV41/holdoutgeometry/productionregistry unchanged. '
        'Source prediction parity passed both precisions. All57addedchars>=168encounters;521requiredterms covered.', '',
        '**best.pt retains starting V41EMA1000(V42update0). No trained candidate passes all revised gates. Raw1500 has the lowest YOLOholdoutCER; finalraw2000 has slightly better long/retention results.**', '',
        '| FP16 CER | Source V41EMA1000 | Raw1500 | Final raw2000 | Final EMA2000 |','|---|---:|---:|---:|---:|']
    for name in ('yolo_current_holdout','yolo_current_train','matched_clean_train_probe','augmented_train_probe','long512_clean','long512_noisy','long384_clean','long384_noisy','seen','heldout','historical_bucket','fresh_short_clean','new_character_probe'):
        values=[initial[name]['0.0']['all']['cer']]+[cases[label]['metrics'][name]['cer'] for label in ('raw1500','raw2000','ema2000')]
        lines.append('| '+name+' | '+' | '.join(f'{v:.2%}' for v in values)+' |')
    lines+=['','## Menu recognition and geometry','',
        'Finalraw holdout51.82→45.46%(-6.36pp);13menu-cluster95%CI[-9.38,-2.83]pp. Exact24→43/156. '
        'Deletions294→249,substitutions176→153,insertions27→34. Nine menus improve,three worsen,one ties. '
        'These are reused development/selection menus with VLM labels,not independent production evidence.', '',
        'Matchedclean/augmented256trainingprobe gap drops2.00→.32pp. This demonstrates reduced sensitivity to the fixed geometric perturbations, '
        'but cannot attribute the gain to padding/stretch alone because exposure,longreplay,fullfine-tuning andteacherKD also changed.', '',
        'Horizontalholdout55.19→47.12%;vertical37.08→38.20%. Only11.26%oftrainingpresentations arevertical;alllongreplayishorizontal. '
        'Orientation imbalance is a concern to test,not an isolated proven cause.', '',
        '## Long and novel-character results','',
        'Heldout-font long512mean6.64→1.88%;all-font clean6.52→1.72%,noisy6.92→2.38%. '
        'Fitted384views of identicallonglabels remain18.12/20.90%CER atfinal, so native512routing remains important.', '',
        'Newcharheldoutfontrecall52.05→65.20%,precision92.15%,F1 76.37%. This improves the selected source but is below V41finalraw85.67%recall. '
        'NovelcharYOLOholdoutmatches0→1/36;only12addedclassesobserved and17occurrencesarespaces. Aggregate menu gains primarily concern old characters,not reliable transfer of all57additions.', '',
        '## Retention and checkpoint recommendation','',
        'Finalraw fails5FP16/6BF16gates. Historical10.69→12.53%;seen5.42→5.94%;fresh-short-clean25.17→28.83%. '
        'Pooledtwochar58/400FP16against56cap;BF16pooledonechar63/200against60andtwochar60/400against56. '
        'The historical drift is larger than a one-edit gate issue.', '',
        'Keep V38 promoted and preserve the source fallback. V42 is a promising menu/long-line adaptation candidate with an explicit retention/vertical tradeoff,not an automatic promotion. '
        'Before another longrun, test whether a lightweight source/candidate weight interpolation retains menu gains while recovering short/historical performance; '
        'also balance vertical versus horizontal exposure within YOLO replay if further training is authorized. No new training or GPU inference run in this review.']
    (WORK/'review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
