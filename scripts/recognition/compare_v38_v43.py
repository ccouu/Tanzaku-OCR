"""Forward-only comparison; never modifies promoted weights or registry."""
from pathlib import Path
import json
import torch
from scripts.recognition.train_v38_yolo25 import BucketCrops
from scripts.recognition.review_v38_yolo25 import cluster_bootstrap
from scripts.recognition.prepare_v43 import normalize_dataset
from outputs.recognition.v37_long512 import LongLines, evaluate_any
from outputs.recognition.v36_yolo_fullpasses import load
from src.recognition.model import MenuRecognizer
from src.recognition.current import current_checkpoint, REGISTRY
from src.recognition.metrics import align_errors
from src.recognition.utils import sha, write_json

OUT=Path('outputs/recognition/evaluation/v38_vs_v43_promotion')
V43=Path('outputs/recognition/pilots/v43_digits_prices_2000')
V40=Path('outputs/recognition/training/v40_new_characters_2000')
V41=Path('outputs/recognition/pilots/v41_new_rows_only_2000')
V37=Path('outputs/recognition/pilots/v37_long512_2000')


def rows(path):return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines()]


def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    path=Path('checkpoints/menu_recognizer_v38.pt');registry_hash=sha(REGISTRY)
    assert sha(path)=='59d04ddcc44ec6f3388cce17c221ae4ced2b6c4fd2a39e1397752492fc10ddad'
    ck=torch.load(path,map_location='cpu',weights_only=False)
    mapping=ck['vocab']['char_to_id'];cfg=ck['config']
    assert len(mapping)==1722
    # Dataset target IDs are unused by this forward-only evaluator. Use V43's
    # label mapping to represent the yen sign absent from V38; decoding still
    # uses V38's original vocabulary, so its OOV errors remain counted.
    label_mapping=load(V43/'vocab.json')['char_to_id']
    model=MenuRecognizer(len(mapping)+1,use_bilstm=True).cuda().eval()
    model.load_state_dict(ck['model']);model.vocabulary=mapping;model.recurrent.lstm.flatten_parameters()
    digits=rows(V43/'digits/labels.jsonl');long=rows(V37/'long_labels.jsonl');extra={}
    for cohort in ('known','heldout'):
        extra['digits_'+cohort]=BucketCrops([r for r in digits if r['split']=='probe' and r['cohort']==cohort],V43/'digits',cfg,mapping)
    for view in ('clean','noisy'):
        extra['long512_'+view]=LongLines([r for r in long if r['split']=='holdout' and r['view']==view],V37,cfg,mapping)
    evaluated={}
    for name,ds in extra.items():
        normalize_dataset(ds)
        ds.mapping=label_mapping
        assert all(c in label_mapping for r in ds.rows for c in r['text'])
        evaluated[name]=evaluate_any(model,ds,'cuda:0','fp16')
        print(name,evaluated[name]['0.0']['all']['cer'],flush=True)
        write_json(OUT/'v38_additional_fp16.json',evaluated)
    comparisons={}
    meta=[r for r in rows(V40/'yolo_labels.jsonl') if r['split']=='val_holdout']
    for precision in ('fp16','bf16'):
        # V41initial is prediction-equivalent to promotedV38, verified by its preflight.
        old=load(V41/f'initial_{precision}.json')['validation']
        new=load(V43/f'step_2000_raw_{precision}.json')['validation']
        original=load(V40/f'review_original_v38_{precision}.json')
        assert [r['prediction'] for r in old['yolo_current_holdout']['0.0']['predictions']]==[r['prediction'] for r in original['yolo_current_holdout']['0.0']['predictions']]
        if precision=='fp16':old.update(evaluated)
        values={}
        for name in old.keys() & new.keys():
            a=old[name]['0.0']['predictions'];b=new[name]['0.0']['predictions']
            assert len(a)==len(b) and all(x['truth']==y['truth'] for x,y in zip(a,b)),name
            values[name]=dict(v38=old[name]['0.0']['all'],v43=new[name]['0.0']['all'])
            if 'by_length' in old[name]['0.0'] and 'by_length' in new[name]['0.0']:
                values[name]['by_length']={version:report[name]['0.0']['by_length'] for version,report in [('v38',old),('v43',new)]}
        a=old['yolo_current_holdout']['0.0']['predictions'];b=new['yolo_current_holdout']['0.0']['predictions']
        boot=cluster_bootstrap(a,b,meta);boot['v43_minus_v38_pp']=boot.pop('v38_minus_v36_pp')
        paired=dict(improved=0,worsened=0,unchanged=0)
        for x,y in zip(a,b):
            d=sum(y['error_counts'].values())-sum(x['error_counts'].values())
            paired['improved' if d<0 else 'worsened' if d>0 else 'unchanged']+=1
        digit={}
        for version,report in [('v38',old),('v43',new)]:
            digit[version]={}
            for name in ['yolo_current_holdout']+(['digits_heldout'] if precision=='fp16' else []):
                refs={c:0 for c in '0123456789'};correct=dict(refs)
                for r in report[name]['0.0']['predictions']:
                    for op in align_errors(r['truth'],r['prediction']):
                        if op['truth'] in refs:
                            refs[op['truth']]+=1;correct[op['truth']]+=op['type']=='match'
                per={c:correct[c]/refs[c] for c in refs if refs[c]}
                digit[version][name]=dict(macro_recall=sum(per.values())/len(per),references=refs,correct=correct)
        comparisons[precision]=dict(metrics=values,yolo_menu_cluster_bootstrap=boot,yolo_paired_lines=paired,digit_recall=digit)
    assert sha(REGISTRY)==registry_hash
    result=dict(v38=str(path),v38_sha256=sha(path),v43=str(V43/'final_raw.pt'),v43_sha256=sha(V43/'final_raw.pt'),
        comparisons=comparisons,registry_unchanged=True,
        caveats=['Reused development menus; VLM labels not independently human verified.','Digits normalized identically; no fuzzy lexicon or blank bias.','Yen sign is OOV for V38; digit-line CER includes that coverage disadvantage. Digit-only recall is reported separately.','V38legacy results from verified prediction-equivalentV41initial; added digit/long probes freshly evaluated with promotedV38.'])
    write_json(OUT/'comparison.json',result)
    lines=['# V38 versus V43 final raw','', '|FP16 metric|V38|V43|','|---|---:|---:|']
    for name in ('yolo_current_holdout','seen','heldout','historical_bucket','fresh_short_clean','application_clean_control','application_texture_color_noise','digits_heldout','long512_clean','long512_noisy','yolo_provisional','strong_appearance'):
        m=comparisons['fp16']['metrics'][name];lines.append(f"|{name} CER|{m['v38']['cer']:.2%}|{m['v43']['cer']:.2%}|")
    lines+=['',json.dumps({p:{k:v for k,v in r.items() if k!='metrics'} for p,r in comparisons.items()},ensure_ascii=False,indent=2), '',*result['caveats']]
    (OUT/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({p:r['yolo_menu_cluster_bootstrap'] for p,r in comparisons.items()}),flush=True)


if __name__=='__main__':main()
