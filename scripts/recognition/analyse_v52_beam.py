"""Check decoder invariants and quantify the optimistic beam25 candidate ceiling."""
from pathlib import Path
import json
import numpy as np
import torch
from src.recognition.prefix_beam import decode_beams
from src.recognition.data import edit_distance
from src.recognition.utils import sha,write_json

WORK=Path('outputs/recognition/evaluation/v52_beam_lexicon')


def main():
    torch.set_num_threads(4)
    report=json.loads((WORK/'report.json').read_text(encoding='utf-8'));assert report['completed']
    assert sha(Path('data/processed/lexicon/culinary_menu.db'))==report['lexicon_sha256']
    audit=dict(passed=True,training=False,production_unchanged=True,models={},lexicon_digit_lock_verified=True)
    for tag,model in report['models'].items():
        assert sha(Path(model['checkpoint']))==model['sha256']
        source=torch.load(model['checkpoint'],map_location='cpu',weights_only=False);mapping=source['vocab']['char_to_id']
        source_eval=Path('outputs/recognition/pilots')/('v43_digits_prices_2000' if tag=='V43' else 'v48_parentheses_2000')/'step_2000_raw_fp16.json'
        baseline=json.loads(source_eval.read_text(encoding='utf-8'))['validation']
        parity={};digit_checks=0
        for probe in model['probes']:
            d=json.loads((WORK/f'{tag}_{probe}.json').read_text(encoding='utf-8'))
            assert len({len(rs) for rs in d['predictions'].values()})==1
            if probe in baseline:
                a=baseline[probe]['0.0']['predictions'];b=d['predictions']['greedy'];assert len(a)==len(b)
                parity[probe]=sum(x['prediction']!=y['literal_prediction'] for x,y in zip(a,b))
            for weight in (.1,.3):
                for a,b in zip(d['predictions']['beam25'],d['predictions'][f'beam25_lex{weight}']):
                    assert ''.join(c for c in a['prediction'] if c.isdigit())==''.join(c for c in b['prediction'] if c.isdigit())
                    digit_checks+=1
        name='yolo_current_holdout';cache=json.loads((WORK/f'{tag}_{name}_cache.json').read_text(encoding='utf-8'));arrays=np.load(WORK/f'{tag}_{name}_cache.npz')
        exact=0;edits=0;chars=0;rows=[]
        for i,r in enumerate(cache['records']):
            nb=decode_beams(arrays[f'b{i}'],arrays[f'i{i}'],arrays[f'p{i}'],25,mapping)
            oracle=min(nb,key=lambda x:edit_distance(r['truth'],x[0]));e=edit_distance(r['truth'],oracle[0])
            exact+=e==0;edits+=e;chars+=len(r['truth'])
            rows.append(dict(crop_id=r['crop_id'],truth=r['truth'],oracle=oracle[0],oracle_edits=e,acoustic_best=nb[0][0]))
        audit['models'][tag]=dict(record_count_per_decoder=sum(p['records'] for p in model['probes'].values()),
            literal_baseline_prediction_disagreements=parity,lexicon_digit_invariants_checked=digit_checks,
            oracle_beam25=dict(exact=exact,lines=len(rows),cer=edits/chars,note='Optimisticreference-selectedcandidate;notadeployabledecoder'),
            yolo_oracle_rows=rows)
    assert sha(Path('checkpoints/menu_recognizer_v43.pt'))=='41823c3d508967e50273834fabe2f27a85ea88137b7df3d2539cce735d9986a1'
    assert json.loads(Path('src/recognition/current_model.json').read_text())['version']=='V43'
    write_json(WORK/'completion_verification.json',audit)
    for tag,d in audit['models'].items():print(tag,{k:v for k,v in d.items() if k!='yolo_oracle_rows'})


if __name__=='__main__':main()
