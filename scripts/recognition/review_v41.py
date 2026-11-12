"""Independent frozen-model, appended-row coverage and saved-prediction review."""
from collections import Counter
import gc
from pathlib import Path
import torch

from scripts.recognition.prepare_v41 import WORK,V40,ROOT,INPUT
from outputs.recognition.v36_yolo_fullpasses import load
from outputs.recognition.tiered_short_pilot import rows
from src.recognition.current import current_checkpoint
from src.recognition.metrics import align_errors
from src.recognition.utils import sha,write_json


def recognition_counts(predictions,added):
    refs=Counter();correct=Counter();predicted=Counter()
    for r in predictions:
        refs.update(c for c in r['truth'] if c in added);predicted.update(c for c in r['prediction'] if c in added)
        correct.update(o['truth'] for o in align_errors(r['truth'],r['prediction']) if o['type']=='match' and o['truth'] in added)
    tp=sum(correct.values());ref=sum(refs.values());hyp=sum(predicted.values())
    return dict(correct=tp,reference=ref,predicted=hyp,precision=tp/hyp if hyp else 0,
        recall=tp/ref if ref else 0,f1=2*tp/(ref+hyp) if ref+hyp else 0,
        by_character={c:dict(reference=refs[c],correct=correct[c],predicted=predicted[c]) for c in added})


def main():
    torch.set_num_threads(4);sampling=load(WORK/'sampling.json');report=load(WORK/'report.json');config=load(WORK/'run_config.json')
    source_path=current_checkpoint();assert sha(source_path)==config['source_sha256'];source=torch.load(source_path,map_location='cpu',weights_only=False)
    vocab=load(WORK/'vocab.json');added=sampling['added_characters'];assert len(added)==57 and len(vocab['char_to_id'])==1779
    assert vocab==load(V40/'vocab.json') and vocab['blank_id']==0 and vocab['characters'][:1722]==source['vocab']['characters']
    assert all(vocab['char_to_id'][c]==i for c,i in source['vocab']['char_to_id'].items())
    parity=load(WORK/'initial_parity.json');assert parity['passed'] and all(v==0 for d in parity['prediction_disagreements'].values() for v in d.values())
    assert parity['new_token_emission_lines']=={'fp16':0,'bf16':0}
    target=rows(WORK/'target_train_labels.jsonl');assert len(target)==2668
    assert all(sum(c in added for c in r['text'])==1 and len(r['text'])<=5 and r['canvas']<=256 for r in target)
    original=[r for r in rows(ROOT/'labels.jsonl') if r['split']=='train'];generated=rows(Path('outputs/recognition/training/v34_broad_fullpasses6/corpus/labels.jsonl'))
    yolo=rows(WORK/'yolo_labels.jsonl');train=[r for r in yolo if r['split']=='train'];hold=[r for r in yolo if r['split']=='val_holdout']
    assert len(train)==1561 and len(hold)==156 and not {r['source_menu'] for r in train}&{r['source_menu'] for r in hold}
    assert not {r['sha256'] for r in train}&{r['sha256'] for r in hold}
    for r in yolo:assert sha(INPUT/r['image'])==r['sha256']
    for r in target:assert sha(V40/'targeted'/r['image'])==r['sha256']
    probe=[r for r in rows(V40/'targeted/labels.jsonl') if r['split']=='probe'];assert len(probe)==684
    assert not {r['sha256'] for r in target}&{r['sha256'] for r in probe}
    for r in probe:assert sha(V40/'targeted'/r['image'])==r['sha256']
    data={'original':original,'generated':generated,'target':target,'yolo':train};plan=load(WORK/'batches.json')
    assert sha(WORK/'batches.json')==sampling['plan_sha256']
    streams=Counter();encounters=Counter();terms=Counter();target_counts=Counter();yolo_seen=set();teacher_eligible=0
    for batch in plan:
        assert len(batch)==32
        for kind,index in batch:
            r=data[kind][index];text=r['text'];streams[kind]+=1;terms[text]+=1
            encounters.update(c for c in text if c in added)
            teacher_eligible+=not any(c in added for c in text)
            if kind=='target':target_counts.update(c for c in text if c in added)
            if kind=='yolo':yolo_seen.add(index)
    assert len(plan)==2000 and sum(streams.values())==64000 and streams==sampling['streams']==report['stream_encounters']
    assert len(yolo_seen)==1561 and len(sampling['required_term_encounters'])==521
    assert all(terms[t]==n for t,n in sampling['required_term_encounters'].items())
    assert all(encounters[c]==report['new_character_encounters'][c]==sampling['total_new_character_counts'][c] for c in added)
    assert min(encounters[c] for c in added)==280
    assert all(target_counts[c]==sampling['target_character_counts'][c] for c in added)
    assert min(target_counts.values())==280 and max(target_counts.values())==281
    bn=[k for k in source['model'] if k.endswith(('running_mean','running_var','num_batches_tracked'))];assert len(bn)==147
    checkpoints={}
    for name in ('best.pt','last.pt','final_raw.pt','final_ema.pt','step_0500.pt','step_1000.pt','step_1500.pt','step_2000.pt'):
        ck=torch.load(WORK/name,map_location='cpu',weights_only=False);assert ck['vocab']==vocab
        for field in ('model','raw_training_model','ema_training_model'):
            for key,value in source['model'].items():
                if key.startswith('classifier.'):
                    candidate=ck[field][key][:1723] if field=='model' else ck[field][key.replace('classifier.','classifier.old.')]
                else:candidate=ck[field][key]
                assert torch.equal(candidate,value),(name,field,key)
        initial=source['model']['classifier.weight'][0:1].expand(57,-1,-1)
        assert ((ck['model']['classifier.weight'][1723:]-initial).flatten(1).norm(dim=1)>0).all()
        assert len(ck['optimizer']['state'])==2 and all(int(v['step'])==ck['updates'] for v in ck['optimizer']['state'].values())
        if name=='best.pt':assert ck['updates']==1000 and ck['checkpoint_arm']=='ema'
        elif name in ('last.pt','final_raw.pt','final_ema.pt'):assert ck['updates']==2000
        checkpoints[name]=dict(update=ck['updates'],arm=ck['checkpoint_arm'],all_pretrained_parameters_and_buffers_bit_identical=True)
        del ck;gc.collect()
    verification=dict(passed=True,updates=2000,presentations=64000,stream_counts=dict(streams),
        append_only_ids=True,blank0_unchanged=True,strict_initial_parity_passed_both_precisions=True,all57newrows_updated=True,
        minimum_actual_newchar_encounters=280,target_character_count_range=[280,281],
        all_pretrained_parameters_old_classifier_BNaffine_and147buffers_bit_identical=True,
        only_two_new_classifier_parameter_tensors_in_optimizer=True,all521terms_and1561YOLOtraincrops_encountered=True,
        data_hashes_and_menu_pixel_disjointness_verified=True,teacher_eligible_training_examples=teacher_eligible,
        teacher_policy='Valid frames only on old-only labels, as checked in unit tests and archived training code.',
        source_V38_unchanged=True,checkpoints=checkpoints)
    write_json(WORK/'completion_verification.json',verification)
    cases={'selected_ema1000':(WORK,'step_1000_ema'),'final_raw':(WORK,'step_2000_raw'),'final_ema':(WORK,'step_2000_ema'),'V40_final_raw':(V40,'step_2000_raw')}
    comparisons={}
    for precision in ('fp16','bf16'):
        comparisons[precision]={}
        for case,(folder,stem) in cases.items():
            r=load(folder/f'{stem}_{precision}.json');pred=r['validation']['new_character_probe']['0.0']['predictions']
            heldfont=[p for p in pred if p['cohort']=='heldout'];oldmiss=Counter();newmiss=Counter()
            for row in pred:
                for op in align_errors(row['truth'],row['prediction']):
                    if op['type'] in ('deletion','substitution'):(newmiss if op['truth'] in added else oldmiss).update([op['type']])
            comparisons[precision][case]=dict(retention=r['retention'],metrics={k:v['0.0']['all'] for k,v in r['validation'].items()},
                heldoutfont_newchar=recognition_counts(heldfont,added),YOLO_newchar=recognition_counts(r['validation']['yolo_current_holdout']['0.0']['predictions'],added),
                old_character_errors_in_font_probe=dict(oldmiss),new_character_errors_in_font_probe=dict(newmiss))
    write_json(WORK/'review_comparison.json',comparisons)
    before=load(WORK/'initial_bf16.json')['validation'];after=load(WORK/'step_2000_raw_bf16.json')['validation'];short_changes=[]
    for name in ('seen','heldout','original_bucket'):
        for a,b in zip(before[name]['0.0']['predictions'],after[name]['0.0']['predictions']):
            if len(a['truth'])==1 and a['prediction']!=b['prediction']:
                short_changes.append(dict(dataset=name,truth=a['truth'],initial=a['prediction'],final=b['prediction'],
                                          font=a.get('font'),initial_edits=sum(a['error_counts'].values()),final_edits=sum(b['error_counts'].values())))
    write_json(WORK/'short_text_differences.json',short_changes)
    initial=load(WORK/'initial_fp16.json')['validation'];lines=['# V41 independent review','',
        'Audit passed:2000updates/64000presentations,25%controllednewchar/25%YOLO/50%synthetic replay. All57IDs appended; old IDs/blank0unchanged. '
        'Actual added-character floor280; targeted counts280–281perclass. All57rows changed from blank-row initialization. '
        'Every pretrained parameter,oldclassifier row,BNaffine and147BNbuffers bit-identical in selected/final/step checkpoints including raw/EMA training states. '
        'Only two appended classifier parameter tensors exist in optimizer state. Initial FP16/BF16 predictions exactly match original V38; no new emissions.', '',
        '**best.pt is trained EMA at1000, passing all17revisedgates in both precisions. Finalraw/EMA2000 are separate candidates, not the selected model. V38 stays promoted.**', '',
        '| FP16 metric | V38 | V41selected EMA1000 | V41final raw | V41final EMA | V40final raw |','|---|---:|---:|---:|---:|---:|']
    for name in ('yolo_current_holdout','yolo_current_train','seen','heldout','historical_bucket','application_texture_color_noise','fresh_short_clean','new_character_probe'):
        values=[initial[name]['0.0']['all']['cer']]+[comparisons['fp16'][case]['metrics'][name]['cer'] for case in cases]
        lines.append('| '+name+' CER | '+' | '.join(f'{v:.2%}' for v in values)+' |')
    lines+=['','## Novel-character precision and recall','',
        '| Heldout-font newchars | Precision | Recall | F1 |','|---|---:|---:|---:|']
    for case,r in comparisons['fp16'].items():
        m=r['heldoutfont_newchar'];lines.append(f"| {case} | {m['precision']:.2%} | {m['recall']:.2%} | {m['f1']:.2%} |")
    lines+=['','Finalraw correctly aligns293/342heldoutfont newchar occurrences and emits322newcharacters; all57have at least one heldoutfont success. '
        'V40correct183/342with214emissions. HighV41recall is accompanied by~91%precision, not simply more random new-token output.', '',
        '## Menu transfer and remaining failures','',
        'The156-crop YOLOholdout remains51.82%CER for selected and both final candidates. Selected recognizes0/36novel-character occurrences; '
        'finalraw/EMA recognize1/36,with10novel emissions each. Only12added classes appear there,17of36references are spaces; this sparse VLM-labeled sample is not all57classes.', '',
        'Finalraw/EMA pass allFP16gates but fail BF16pooled_original_1char:62/200edits31%,cap60/20030%. '
        'All old model tensors are unchanged, so retained-input regressions arise from competition of new outputs, not encoder/BN/old-head weight drift.', '',
        'The two extra BF16one-char errors are 火→久火 (new久insertion,Meiryoheldoutfont) and 株→抹 (new抹substitution,original_bucket). '
        'Both were correct at initialization; neither is evidence of changed old classifier weights. See short_text_differences.json.', '',
        'Full synthetic new-character probe CER remains52.09%raw despite85.67%newchar recall because old surrounding characters are still poorly recognized on this rendering distribution. '
        'Do not equate newchar recall with complete line accuracy. V40s full-model updates reduced that probe CER much more but damaged broad retention.', '',
        '## Recommendation','',
        'Retain V38 as promoted. V41EMA1000 is a gate-passing expanded-vocabulary development candidate; finalraw is the stronger synthetic newchar recognizer with a two-edit BF16short-gate failure. '
        'Do not extend clean-font repetition blindly: next evidence should focus on verified menu examples containing the additions and an appearance-matched new-character probe. '
        'A cheap new-row logit/weight interpolation diagnostic could assess the late synthetic recall/one-character tradeoff, but would not itself establish menu-domain improvement. '
        'No new training, inference or promotion performed by this review.']
    (WORK/'review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
