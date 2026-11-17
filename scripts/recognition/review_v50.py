"""CPU-only V50 matched checkpoint/data audit and saved-prediction comparison."""
from collections import Counter, defaultdict
from pathlib import Path
import gc
import json
import string
import numpy as np
import torch
from scripts.recognition.review_v49 import START, ROOT, V34, V40, V44, INPUT, rows, expanded_state
from src.recognition.label_normalization import normalize_parentheses
from src.recognition.utils import sha, write_json
from src.recognition.current import REGISTRY, current_checkpoint

WORK = Path('outputs/recognition/pilots/v50_normalized_matched_2000')


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def paired(a, b, menus=None):
    groups = defaultdict(lambda: np.zeros(3))
    for i, (x, y) in enumerate(zip(a, b)):
        assert x['truth'] == y['truth'] and x['index'] == y['index']
        groups[menus[i] if menus else i] += [sum(x['error_counts'].values()),
            sum(y['error_counts'].values()), len(x['truth'])]
    assert len(a) == len(b)
    values = np.array(list(groups.values()))
    rng = np.random.default_rng(50)
    samples = values[rng.integers(len(values), size=(5000, len(values)))].sum(1)
    delta = (samples[:, 1] - samples[:, 0]) / samples[:, 2] * 100
    return dict(delta_pp=float((values[:, 1].sum()-values[:, 0].sum())/values[:, 2].sum()*100),
        ci95_pp=np.percentile(delta, [2.5, 97.5]).tolist(),
        unit='menu' if menus else 'line', units=len(values))


def main():
    torch.set_num_threads(4)
    source = torch.load(START, map_location='cpu', weights_only=False)
    vocab = load(WORK/'control/vocab.json'); mapping = vocab['char_to_id']
    initial = expanded_state(source, vocab)
    assert len(mapping) == 1812 and all(mapping[c] == i for c, i in source['vocab']['char_to_id'].items())
    selected = {mapping[c] for c in string.ascii_letters+string.digits+'()（）'}
    keep = [i for i in range(1813) if i not in selected]
    bn = [k for k in initial if k.endswith(('running_mean','running_var','num_batches_tracked'))]
    assert len(bn) == 147 and len(selected) == 66
    audits = {}
    for name in ('control', 'bilstm'):
        p = WORK/name; r = load(p/'report.json'); s = load(p/'sampling.json')
        assert r['completed'] and r['updates'] == 2000 and r['presentations'] == 64000
        assert sha(p/'batches.json') == s['plan_sha256'] == sha(Path('outputs/recognition/pilots/v49_selective_head_2000/batches.json'))
        assert load(p/'vocab.json') == vocab and load(p/'initial_parity.json')['passed']
        assert all(n == 0 for counts in load(p/'initial_parity.json')['disagreements'].values() for n in counts.values())
        zero = torch.load(p/'initial.pt', map_location='cpu', weights_only=False)
        assert all(torch.equal(zero['model'][k],v) for k,v in initial.items())
        del zero
        states = {}
        for arm in ('raw','ema'):
            ck = torch.load(p/f'final_{arm}.pt', map_location='cpu', weights_only=False)
            assert ck['updates'] == 2000 and ck['vocab'] == vocab and ck['streams'] == r['streams']
            changed_recurrent = []
            for field in ('model','raw_training_model','ema_training_model'):
                state = ck[field]
                for k, v in initial.items():
                    if k.startswith('classifier.'): continue
                    if name == 'bilstm' and k.startswith('recurrent.'):
                        if field == 'model' and not torch.equal(state[k],v): changed_recurrent.append(k)
                    else: assert torch.equal(state[k],v), (name,arm,field,k)
                for suffix in ('weight','bias'):
                    if field == 'model': assert torch.equal(state['classifier.'+suffix][keep], initial['classifier.'+suffix][keep])
                    else: assert torch.equal(state['classifier.base.'+suffix], initial['classifier.'+suffix])
                assert all(torch.equal(state[k],initial[k]) for k in bn)
            changed_rows = sum(not torch.equal(ck['model']['classifier.weight'][i], initial['classifier.weight'][i]) for i in selected)
            assert changed_rows == 66
            assert bool(changed_recurrent) == (name == 'bilstm')
            groups = {}
            for g in ck['optimizer']['param_groups']:
                assert all(int(ck['optimizer']['state'][i]['step']) == 2000 for i in g['params'])
                groups[g['role']] = g['base_lr']
            assert groups == dict(latin=.001,digits_brackets=.00001,**({'recurrent':.00001} if name=='bilstm' else {}))
            states[arm] = dict(frozen_bn_buffers=147,all_other_features_and_rows_exact=True,
                changed_selected_rows=changed_rows,changed_recurrent_tensors=changed_recurrent,optimizer_base_rates=groups)
            del ck; gc.collect()
        if name == 'control':
            best = torch.load(p/'best.pt',map_location='cpu',weights_only=False)
            step = torch.load(p/'step_1500.pt',map_location='cpu',weights_only=False)
            assert best['updates']==1500 and best['checkpoint_arm']=='raw'
            assert all(torch.equal(v,step['model'][k]) for k,v in best['model'].items())
            del best,step
        else: assert not (p/'best.pt').exists() and r['best_update'] is None
        audits[name] = dict(minutes=r['elapsed_seconds']/60,checkpoints=states,
            selected=r['best_update'],normalization_changed_presentations=r['normalization_changed_presentations'])

    # Recount the identical consumed plan from source metadata, preserving originals.
    tiny = Path('outputs/recognition/pilots/v47_clean_latin_fit_1000/targets')
    brackets = Path('outputs/recognition/pilots/v48_parentheses_2000/targets')
    multi = rows(V44/'targeted/labels.jsonl'); yolo = rows(V40/'yolo_labels.jsonl')
    train = [r for r in yolo if r['split']=='train']; hold = [r for r in yolo if r['split']=='val_holdout']
    assert len(train)==1561 and len(hold)==156
    assert not {r['source_menu'] for r in train}&{r['source_menu'] for r in hold}
    assert not {r['sha256'] for r in train}&{r['sha256'] for r in hold}
    ds = dict(original=([r for r in rows(ROOT/'labels.jsonl') if r['split']=='train'],ROOT),
        generated=(rows(V34/'corpus/labels.jsonl'),V34/'corpus'),yolo=(train,INPUT),
        target=([r for r in rows(tiny/'labels.jsonl') if r['split']=='train'],tiny),
        latin_multi=([r for r in multi if r['kind']=='latin' and r['split']=='train' and int(Path(r['image']).stem)%5!=0 and 1<=len(r['text'])<=6 and r['canvas'] in (128,256)],V44/'targeted'),
        brackets=([r for r in rows(brackets/'labels.jsonl') if r['split']=='train'],brackets),
        digits=([r for r in rows(START.parent/'digits/labels.jsonl') if r['split']=='train'],START.parent/'digits'))
    streams=Counter(); chars=Counter(); canonical=Counter(); terms=Counter(); unique=defaultdict(set); images={}; changed=0
    for batch in load(WORK/'control/batches.json'):
        assert len(batch)==32
        for k,i in batch:
            rr,base=ds[k]; row=rr[i]; text=row['text']; normalized=normalize_parentheses(text)
            streams[k]+=1; chars.update(text); canonical.update(normalized); terms[text]+=1; unique[k].add(i)
            changed+=text!=normalized; images[base/row['image']]=row['sha256']
            assert all(c in mapping for c in normalized) and not any('０'<=c<='９' for c in text)
    for p,h in images.items(): assert sha(p)==h,str(p)
    for row in hold: assert sha(INPUT/row['image'])==row['sha256']
    for name in ('control','bilstm'):
        r=load(WORK/name/'report.json'); s=load(WORK/name/'sampling.json')
        assert r['streams']==streams and r['character_encounters']==chars and r['canonical_character_encounters']==canonical
        assert r['normalization_changed_presentations']==changed==3426
        assert len(s['required_term_encounters'])==521 and all(terms[t]==n>=2 for t,n in s['required_term_encounters'].items())
        assert s['unique_records']=={k:len(v) for k,v in unique.items()}
    assert len(unique['digits'])==6000 and len(unique['yolo'])==1561

    effects={}; summaries={}; guard_reasons={}
    for precision in ('fp16','bf16'):
        assert load(WORK/'control'/f'initial_{precision}.json') == load(WORK/'bilstm'/f'initial_{precision}.json')
        effects[precision]={}; summaries[precision]={}; guard_reasons[precision]={}
        for step,arm in [(1500,'raw'),(2000,'raw'),(2000,'ema')]:
            evaluations={n:load(WORK/n/f'step_{step:04d}_{arm}_{precision}.json') for n in ('control','bilstm')}
            key=f'{step}_{arm}'; effects[precision][key]={}; summaries[precision][key]={}
            for probe in ('target_multifont_heldout','brackets_heldout','yolo_current_holdout'):
                a,b=[evaluations[n]['canonical_validation'][probe]['0.0']['predictions'] for n in ('control','bilstm')]
                effects[precision][key][probe]=paired(a,b,[r['source_menu'] for r in hold] if probe=='yolo_current_holdout' else None)
            for name,e in evaluations.items():
                can=e['canonical_validation']; lit=e['validation']; symbols=e['canonical_symbol_metrics']
                pc=e['canonical_target_metrics']['target_multifont_heldout']['per_character']
                added=load(WORK/name/'sampling.json')['added_letters']
                latin_recall={label:sum(pc[c]['recall'] for c in letters)/len(letters) for label,letters in [('new33',added),('old19',sorted(set(string.ascii_letters)-set(added)))]}
                summaries[precision][key][name]=dict(tiny_train_cer=can['target_train']['0.0']['all']['cer'],
                    multifont_heldout_cer=can['target_multifont_heldout']['0.0']['all']['cer'],latin_recall=latin_recall,
                    clean_brackets=can['brackets_heldout']['0.0']['all'],canonical_symbols=symbols,
                    yolo_canonical=can['yolo_current_holdout']['0.0']['all'],yolo_literal=lit['yolo_current_holdout']['0.0']['all'],
                    digit_recall=e['digit_metrics']['digits_heldout']['macro_recall'],retention=e['retention'])
                base=load(WORK/name/f'initial_{precision}.json'); reasons=[]
                extra=set(e['retention']['failed'])-set(base['retention']['failed'])
                if extra: reasons.append(dict(extra_literal_gate_failures=sorted(extra)))
                for metric in ('pooled_original_1char','pooled_original_2char'):
                    if e['retention']['current'][metric]>base['retention']['current'][metric]+.02: reasons.append(metric)
                if e['digit_metrics']['digits_heldout']['macro_recall']<base['digit_metrics']['digits_heldout']['macro_recall']-.02: reasons.append('digit_recall')
                guard_reasons[precision].setdefault(key,{})[name]=reasons
    assert load(REGISTRY)['version']=='V43'
    assert sha(current_checkpoint())=='41823c3d508967e50273834fabe2f27a85ea88137b7df3d2539cce735d9986a1'
    assert sha(START)==load(WORK/'control/run_config.json')['source_sha256']
    audit=dict(passed=True,arms=audits,streams=dict(streams),consumed_image_hashes_verified=len(images),
        holdout_hashes_verified=156,required_exact_terms=521,minimum_term_encounters=2,
        canonical_targets_recounted=True,source_labels_preserved=True,active_registry='V43',unchanged_source_and_production=True)
    write_json(WORK/'completion_verification.json',audit)
    write_json(WORK/'paired_analysis.json',dict(direction='bilstm_minus_control;negative_CER_delta_is_better',effects=effects,summaries=summaries,guard_failures=guard_reasons))
    print(json.dumps(audit)); print('PAIRED',json.dumps(effects)); print('GUARDS',json.dumps(guard_reasons))
    for p in ('fp16','bf16'):
        for key in ('1500_raw','2000_raw','2000_ema'):
            for name,s in summaries[p][key].items():
                print('SUMMARY',p,key,name,json.dumps({k:v for k,v in s.items() if k not in ('canonical_symbols','retention')}))


if __name__=='__main__': main()
