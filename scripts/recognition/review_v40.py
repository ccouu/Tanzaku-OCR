"""Independent coverage/state audit and V40 saved-prediction analysis."""
from collections import Counter
import gc
import json
from pathlib import Path
import torch

from scripts.recognition.prepare_v40 import WORK,ROOT,INPUT
from scripts.recognition.train_v38_yolo25 import WORK as V38
from scripts.recognition.review_v38_yolo25 import cluster_bootstrap
from outputs.recognition.v36_yolo_fullpasses import load
from outputs.recognition.tiered_short_pilot import rows
from src.recognition.current import current_checkpoint
from src.recognition.metrics import align_errors,summarise
from src.recognition.utils import sha,write_json


def recalls(predictions,added):
    refs=Counter();correct=Counter()
    for r in predictions:
        refs.update(c for c in r['truth'] if c in added)
        correct.update(op['truth'] for op in align_errors(r['truth'],r['prediction']) if op['type']=='match' and op['truth'] in added)
    observed=[c for c in added if refs[c]]
    return dict(reference_occurrences=sum(refs.values()),correct_occurrences=sum(correct.values()),
        micro_recall=sum(correct.values())/sum(refs.values()) if refs else None,
        macro_recall_observed=sum(correct[c]/refs[c] for c in observed)/len(observed) if observed else None,
        per_character={c:dict(reference=refs[c],correct=correct[c],recall=correct[c]/refs[c]) for c in observed})


def main():
    torch.set_num_threads(4);cfg=load(WORK/'run_config.json');s=load(WORK/'sampling.json');report=load(WORK/'report.json')
    path=current_checkpoint();assert sha(path)==cfg['source_sha256']
    source=torch.load(path,map_location='cpu',weights_only=False);old=source['vocab'];vocab=load(WORK/'vocab.json');added=s['added']
    assert len(old['char_to_id'])==1722 and len(vocab['char_to_id'])==1779 and len(added)==57
    assert vocab['characters'][:1722]==old['characters'] and vocab['blank_id']==0
    assert all(vocab['char_to_id'][c]==i for c,i in old['char_to_id'].items())
    assert {vocab['char_to_id'][c] for c in added}==set(range(1723,1780))
    assert load(Path('data/processed/recognition/vocab_v40.json'))==vocab
    assert load(ROOT/'vocab.json')['char_to_id']==old['char_to_id']
    assert sha(INPUT/'labeled_crops.jsonl')==s['input_labels_sha256'] and sha(WORK/'batches.json')==s['plan_sha256']
    original=[r for r in rows(ROOT/'labels.jsonl') if r['split']=='train'];generated=rows(Path('outputs/recognition/training/v34_broad_fullpasses6/corpus/labels.jsonl'))
    targets=rows(WORK/'targeted/labels.jsonl');train_target=[r for r in targets if r['split']=='train'];probe=[r for r in targets if r['split']=='probe']
    yolo=rows(WORK/'yolo_labels.jsonl');train_yolo=[r for r in yolo if r['split']=='train'];held=[r for r in yolo if r['split']=='val_holdout']
    assert len(train_yolo)==1561 and len(held)==156 and not {r['source_menu'] for r in train_yolo}&{r['source_menu'] for r in held}
    assert not {r['sha256'] for r in train_yolo}&{r['sha256'] for r in held}
    assert not {r['sha256'] for r in train_target}&{r['sha256'] for r in probe}
    for r in yolo:assert sha(INPUT/r['image'])==r['sha256']
    for r in targets:assert sha(WORK/'targeted'/r['image'])==r['sha256']
    data={'original':original,'generated':generated,'target':train_target,'yolo':train_yolo}
    plan=load(WORK/'batches.json');counts=Counter();streams=Counter();seen_yolo=set();terms=Counter()
    for batch in plan:
        assert len(batch)==32
        for kind,index in batch:
            streams[kind]+=1;text=data[kind][index]['text'];terms[text]+=1;counts.update(c for c in text if c in added)
            if kind=='yolo':seen_yolo.add(index)
    assert len(plan)==2000 and sum(streams.values())==64000 and streams==report['stream_encounters']==s['streams']
    assert all(counts[c]==s['planned_new_character_encounters'][c]==report['new_character_encounters'][c] for c in added)
    assert min(counts[c] for c in added)==532 and len(seen_yolo)==1561
    assert all(terms[t]>=1 for t in s['required_term_encounters']) and len(s['required_term_encounters'])==521
    buffers=[k for k in source['model'] if k.endswith(('running_mean','running_var','num_batches_tracked'))];assert len(buffers)==147
    checkpoints={}
    for name in ('best.pt','last.pt','final_raw.pt','final_ema.pt','step_0500.pt','step_1000.pt','step_1500.pt','step_2000.pt'):
        saved=torch.load(WORK/name,map_location='cpu',weights_only=False)
        assert saved['vocab']==vocab and saved['model']['classifier.weight'].shape[0]==1780
        for field in ('model','raw_training_model','ema_training_model'):
            assert all(torch.equal(saved[field][k],source['model'][k]) for k in buffers)
        if name=='best.pt':
            assert saved['updates']==0
            assert all(torch.equal(saved['model'][k][:1723] if k.startswith('classifier.') else saved['model'][k],v) for k,v in source['model'].items())
            assert not saved['model']['classifier.weight'][1723:].count_nonzero()
        else:assert (saved['model']['classifier.weight'][1723:].flatten(1).norm(dim=1)>0).all()
        if name in ('last.pt','final_raw.pt','final_ema.pt'):
            assert saved['updates']==2000
            for group in saved['optimizer']['param_groups']:
                expected=2000 if group['role']=='new_classifier' else 1800
                assert all(int(saved['optimizer']['state'][i]['step'])==expected for i in group['params'])
        checkpoints[name]=dict(update=saved['updates'],arm=saved['checkpoint_arm'],vocab_size=1779,bn147_buffers_fixed=True)
        del saved;gc.collect()
    warmup=load(WORK/'new_row_warmup_audit.json');assert warmup['update']==200 and warmup['all_pretrained_parameters_and_buffers_bit_identical'] and warmup['new_rows_with_nonzero_weights']==57
    verification=dict(passed=True,append_only_ids=True,blank_id=0,new_id_range=[1723,1779],updates=2000,presentations=64000,
        stream_counts=dict(streams),minimum_actual_new_character_encounters=532,all57newrows_updated=True,
        all147_bn_buffers_fixed=True,all521terms_encountered=True,all1561YOLOtraincrops_used=True,
        all_data_hashes_verified=True,menu_and_exact_pixel_disjoint=True,
        warmup_runtime_assertion_passed=True,optimizer_steps_new2000_old1800=True,
        source_and_original_vocabulary_unchanged=True,checkpoints=checkpoints)
    write_json(WORK/'completion_verification.json',verification)
    initial={p:load(WORK/f'initial_{p}.json') for p in ('fp16','bf16')}
    final={f'{arm}_{p}':load(WORK/f'step_2000_{arm}_{p}.json') for p in ('fp16','bf16') for arm in ('raw','ema')}
    v38={p:load(WORK/f'review_original_v38_{p}.json') for p in ('fp16','bf16')}
    old_hold={r['crop_id'] for r in rows(V38/'yolo/labels.jsonl') if r['split']=='val_holdout'}
    subsets={};effects={}
    for precision in ('fp16','bf16'):
        effects[precision]={};subsets[precision]={}
        base=v38[precision]['yolo_current_holdout']['0.0']['predictions']
        for arm in ('raw','ema'):
            pred=final[f'{arm}_{precision}']['validation']['yolo_current_holdout']['0.0']['predictions']
            effect=cluster_bootstrap(base,pred,held);effect['v40_minus_original_v38_pp']=effect.pop('v38_minus_v36_pp');effects[precision][arm]=effect
            subsets[precision][arm]={}
            for name,keep in (('old135',[i for i,r in enumerate(held) if r['crop_id'] in old_hold]),('newly_covered21',[i for i,r in enumerate(held) if r['crop_id'] not in old_hold])):
                subsets[precision][arm][name]=dict(v38=summarise([base[i] for i in keep]),v40=summarise([pred[i] for i in keep]))
    recall_real={name:recalls(value['validation']['yolo_current_holdout']['0.0']['predictions'],added) for name,value in final.items()}
    old_source=load(V38/'step_0686_raw_fp16.json')['validation'];initial_differences={}
    for name,old_result in old_source.items():
        if name.startswith('yolo_current_'):continue
        current=initial['fp16']['validation'][name]['0.0']['predictions'];before=old_result['0.0']['predictions']
        changed=[(a,b) for a,b in zip(current,before) if a['prediction']!=b['prediction']]
        novel=Counter(c for a,b in changed for c in a['prediction'] if c in added)
        initial_differences[name]=dict(changed_lines=len(changed),new_characters_in_changed_predictions=dict(novel))
    write_json(WORK/'review_paired_analysis.json',dict(original_V38_comparison=effects,holdout_subsets=subsets,new_character_recall_YOLO=recall_real,initialization_disagreements=initial_differences))
    lines=['# V40 independent review','',
        'Audit passed:2000updates/64000presentations,50%targeted/25%YOLO/25%synthetic. All57IDs appended1723–1779; original1722IDs andblank0 unchanged. All57new rows learned nonzero weights. Actual encounter floor532,maximum15388(space). All147BNbuffers fixed, all521requiredterms covered, all1561eligible YOLOtraincrops used. First200updates old-parameter preservation runtime assertion passed; optimizer counters independently confirm2000new-row/1800pretrained updates. Source/promotedV38 and original vocabulary unchanged.', '',
        '**Keep V38 promoted. best.pt is untrained expanded V38 at update0, not a trained V40 nor an exact prediction-equivalent V38.**', '',
        '| Same evaluation set, FP16 | Original V38 | Initialized V40 | Final raw | Final EMA |','|---|---:|---:|---:|---:|']
    for name in ('yolo_current_holdout','new_character_probe'):
        values=[v38['fp16'][name]['0.0']['all']['cer'],initial['fp16']['validation'][name]['0.0']['all']['cer'],final['raw_fp16']['validation'][name]['0.0']['all']['cer'],final['ema_fp16']['validation'][name]['0.0']['all']['cer']]
        lines.append('| '+name+' | '+' | '.join(f'{x:.2%}' for x in values)+' |')
    for name in ('seen','heldout','historical_bucket','application_clean_control','application_texture_color_noise','menu_clean','menu_noisy'):
        values=[old_source[name]['0.0']['all']['cer'],initial['fp16']['validation'][name]['0.0']['all']['cer'],final['raw_fp16']['validation'][name]['0.0']['all']['cer'],final['ema_fp16']['validation'][name]['0.0']['all']['cer']]
        lines.append('| '+name+' | '+' | '.join(f'{x:.2%}' for x in values)+' |')
    lines+=['','## What learned','',
        'Heldout-font new-character macro recall0→53.51%raw/46.78%EMA. Six heldoutfont-render instances per character; individual character percentages are coarse. '
        'Zero heldoutfont successes for space, dot, slash, square brackets, lowercaseg/y. Hundreds of exposures did not guarantee reliable recognition. '
        'Space15388/E9588/S7500encounters dwarf the least-seen newcharacter申532: anchor balance did not balance all tokens in realistic contexts.', '',
        '## Initialization defect','',
        'The new rows were zero weights with constant-12bias. All57new logits tie, and on weak frames -12 can exceed every old logit. '
        'The first appended ID is space, so untrained space emissions changed initial predictions (11seen-font lines and45historical lines in FP16). '
        'The original initial_parity.json note about possible rounding is insufficient: changes contain new spaces and are not prediction-preserving initialization. '
        'Old IDs and weight rows were preserved, but greedy decoding was not. A future extension should copy the blank classifier row with a negative bias offset, guaranteeing new logits remain below blank initially.', '',
        '## Holdout evidence','']
    for arm in ('raw','ema'):
        p=effects['fp16'][arm];lines.append(f"{arm}: actual-original-V38 holdout change {p['v40_minus_original_v38_pp']:+.2f}pp,13menu bootstrap95%CI [{p['ci95_pp'][0]:+.2f},{p['ci95_pp'][1]:+.2f}].")
    lines+=['','The full156-crop holdout includes21formerly OOV crops; compare identical156inputs, not the old135-crop CER. '
        'review_paired_analysis.json separates common135 and newlycovered21. YOLO-menu new-character recall is reported separately from syntheticfont probes.', '',
        '## Retention and next recommendation','',
        f"Final raw fails{len(final['raw_fp16']['retention']['failed'])}FP16/{len(final['raw_bf16']['retention']['failed'])}BF16revisedgates; EMA fails{len(final['ema_fp16']['retention']['failed'])}in each precision. "
        'These losses are materially larger than V38s marginal tradeoff. No further gate loosening or automatic promotion.',
        '', 'Keep the expanded vocabulary but restart from V38 for the next pilot. Fix prediction-preserving initialization first. '
        'Initially freeze the encoder, temporal head, recurrent layers and original classifier rows; train new rows with stronger old-data replay/teacher consistency. '
        'Use controlled short contexts containing one new target to prevent accidental Latin/space token oversampling. '
        'Only consider limited old-head unfreezing if new-row-only learning plateaus. This review did not launch further training.']
    lines+=['','## Matched YOLO subsets and novel characters','',
        '| Subset, FP16 | Original V38 CER | V40 final raw CER |','|---|---:|---:|']
    for subset,result in subsets['fp16']['raw'].items():
        lines.append(f"| {subset} | {result['v38']['cer']:.2%} | {result['v40']['cer']:.2%} |")
    actual=recall_real['raw_fp16']
    lines+=['',f"On the YOLO holdout, new-character matches are {actual['correct_occurrences']}/{actual['reference_occurrences']} ({actual['micro_recall']:.2%}). "
        'Only12of57added characters appear there, with17of36occurrences being spaces. This limited punctuation-heavy sample does not measure all57classes on menu crops; nevertheless it shows weak transfer for the observed additions.',
        '', 'The synthetic heldout-font macro53.51% is over57classes with six instances/class, so it is not directly comparable to this differently composed YOLO micro recall.']
    (WORK/'review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)
    print('REAL_NEW_RECALL',json.dumps(recall_real['raw_fp16'],ensure_ascii=False))
    print('SUBSETS',json.dumps(subsets['fp16']['raw'],ensure_ascii=False))


if __name__=='__main__':main()
