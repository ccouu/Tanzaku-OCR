"""V30 single-character menu terms versus frequency-matched Kanji controls."""
from collections import Counter
import html
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image,ImageFilter
from fontTools.ttLib import TTFont
import torch

from outputs.recognition.single_character_font_audit import mask_for,DATA,CHECKPOINT
from outputs.recognition.application_pilot import appearance,scaled_bucket,state_hash
from src.recognition.data import preprocess,collate,decode
from src.recognition.metrics import align_errors,summarise
from src.recognition.model import MenuRecognizer
from src.recognition.utils import sha,write_json

OUT=Path('outputs/recognition/evaluation/v30_meaningful_onechar_terms')
V29=Path('outputs/recognition/pilots/v29_application_mixture_2000')
V30=Path('outputs/recognition/pilots/v30_short_retention_2000')
SEED=20261205
TERMS=[('円','yen'),('茶','tea'),('酒','alcohol/sake'),('水','water'),('湯','hot water'),('氷','ice'),
    ('米','rice'),('肉','meat'),('魚','fish'),('塩','salt'),('酢','vinegar'),('卵','egg'),
    ('貝','shellfish'),('牛','cattle/beef context'),('豚','pig/pork context'),('鶏','chicken')]
SOURCES=[dict(title='Daijisen / Kotobank: 円',url='https://kotobank.jp/word/%E5%86%86-37934'),
    dict(title='Gunma restaurant association multilingual food/menu guide',url='https://gunma-insyoku.or.jp/documents/itadakimasu.pdf'),
    dict(title='Japan Sushi Association official textbook: tea/rice/water/vinegar',url='https://sushiken.or.jp/textbook/official/ch06'),
    dict(title='National Diet Library authority: tea',url='https://id.ndl.go.jp/auth/ndlsh/%E8%8C%B6'),
    dict(title='Kotobank: 卵',url='https://kotobank.jp/word/%E5%8D%B5-94296'),
    dict(title='Kotobank: 鶏',url='https://kotobank.jp/word/%E9%B6%8F-460974'),
    dict(title='Wiktionary Japanese: 湯',url='https://ja.wiktionary.org/wiki/%E6%B9%AF'),
    dict(title='Wiktionary Japanese: 氷',url='https://ja.wiktionary.org/wiki/%E6%B0%B7'),
    dict(title='Salt Industry Center: cooking seasonings',url='https://www.shiojigyo.com/recipe/tips/other/')]


def main():
    OUT.mkdir(parents=True,exist_ok=False);started=time.perf_counter();torch.set_num_threads(4)
    settings=json.loads(Path('outputs/recognition/evaluation/v30_onechar_fonts_1000/manifest.json').read_text(encoding='utf-8'))['settings']
    mapping=json.loads((DATA/'vocab.json').read_text(encoding='utf-8'))['char_to_id'];common=set(mapping)
    for s in settings:
        path=Path(s['path'])
        with TTFont(path,fontNumber=0 if path.suffix.lower()=='.ttc' else -1) as f:common&={chr(n) for n in f.getBestCmap()}
    original=[r for r in map(json.loads,(DATA/'labels.jsonl').read_text(encoding='utf-8').splitlines()) if r['split']=='train']
    counts=Counter(c for r in original for c in r['text']);single=Counter(r['text'] for r in original if len(r['text'])==1)
    generated=[json.loads(s) for s in (V29/'labels.jsonl').read_text(encoding='utf-8').splitlines()]
    replay=Counter();replay_single=Counter()
    for file in (V29/'batches.json',V30/'control_batches.json'):
        for batch in json.loads(file.read_text()):
            for kind,i in batch:
                text=(original[i] if kind=='original' else generated[i])['text'];replay.update(text)
                if len(text)==1:replay_single.update(text)
    terms=[c for c,m in TERMS];assert all(c in common for c in terms)
    candidates={c for c in common if '\u4e00'<=c<='\u9fff' and c not in terms and counts[c]>0}
    pairs=[]
    for term in sorted(terms,key=lambda c:counts[c],reverse=True):
        # Frequency only: visual complexity and standalone lexical status remain uncontrolled.
        def cost(c):return abs(math.log((counts[c]+1)/(counts[term]+1)))+.2*abs(math.log((replay[c]+1)/(replay[term]+1)))+.1*abs(single[c]-single[term])
        control=min(candidates,key=lambda c:(cost(c),c));candidates.remove(control)
        pairs.append(dict(term=term,control=control,original_counts=[counts[term],counts[control]],
            original_standalone_counts=[single[term],single[control]],
            v29_v30_replay_counts=[replay[term],replay[control]],
            v29_v30_replay_standalone_counts=[replay_single[term],replay_single[control]],matching_cost=cost(control)))
    pairs=sorted(pairs,key=lambda p:terms.index(p['term']));rows=[];(OUT/'images').mkdir()
    for pi,pair in enumerate(pairs):
        for si,setting in enumerate(settings):
            for layout in range(2):
                for group in ('term','control'):
                    char=pair[group];rng=random.Random(SEED+pi*101+layout*10007);nrng=np.random.default_rng(SEED+pi*101+layout*10007)
                    size=rng.randint(24,120);mask=mask_for(char,setting['path'],size,setting['weight'],rng)
                    texture=('paper','wood','chalkboard','cloth')[(pi+layout)%4];contrast=('low','medium','high')[(pi+layout)%3]
                    color,meta=appearance(mask,texture,contrast,rng,nrng)
                    a=np.asarray(mask,dtype=float)/255;clean=Image.fromarray(np.rint(242-222*a).astype('uint8')).convert('RGB')
                    sigma=rng.uniform(1.5,3.5);blur=rng.uniform(.25,.55);pixels=np.asarray(color.filter(ImageFilter.GaussianBlur(blur)),dtype=float)
                    noisy=Image.fromarray(np.rint(pixels+nrng.normal(0,sigma,pixels.shape)).clip(0,255).astype('uint8'))
                    for view,image in (('clean',clean),('noisy',noisy)):
                        image,width,canvas=scaled_bucket(image);assert canvas==128
                        dest=OUT/'images'/f'{pi:02d}_{si:02d}_{layout}_{group}_{view}.png';image.save(dest)
                        rows.append(dict(index=len(rows),pair=pi,truth=char,group=group,view=view,layout=layout,
                            setting=setting['label'],font=Path(setting['path']).name,cohort=setting['cohort'],
                            native_font_size=size,valid_width=width,canvas=canvas,image=str(dest.relative_to(OUT)),sha256=sha(dest),
                            original_frequency=counts[char],original_standalone_frequency=single[char],
                            replay_frequency=replay[char],replay_standalone_frequency=replay_single[char],
                            texture=texture,contrast=contrast,**meta))
    assert len(rows)==2688
    (OUT/'labels.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    write_json(OUT/'manifest.json',dict(seed=SEED,images=2688,meaningful_term_images=1344,control_images=1344,
        terms=[dict(character=c,meaning=m) for c,m in TERMS],sources=SOURCES,settings=settings,matching=pairs,
        design='16menu-relevant single-character terms,16other-Kanji controls,21font settings,2layouts,clean/noisy pairs',
        limits='Controls also have meanings; they are not meaningless glyphs. Matching uses original and replay occurrence counts,not stroke complexity or independently verified lexical frequency. Original V26/pretraining exposure is unknown. No causal semantic-understanding conclusion is possible.',
        script_sha256=sha(Path(__file__))))
    before=sha(CHECKPOINT);saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    model=MenuRecognizer(len(mapping)+1,use_bilstm=True).cuda();model.load_state_dict(saved['model']);model.eval();before_state=state_hash(model)
    results=[]
    with torch.inference_mode():
        for start in range(0,len(rows),32):
            chunk=rows[start:start+32];samples=[]
            for r in chunk:
                with Image.open(OUT/r['image']) as im:t,w,compressed=preprocess(im,saved['config']['mean'],saved['config']['std'],'batch')
                assert w==128 and not compressed;samples.append((t,torch.tensor([mapping[r['truth']]]),0,r['valid_width'],r['truth'],False))
            batch=collate(samples)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(batch['images'].cuda(),batch['orientation'].cuda(),batch['input_lengths'])
            for row,pred in zip(chunk,decode(logits,mapping,batch['input_lengths'])):
                c=Counter(op['type'] for op in align_errors(row['truth'],pred) if op['type']!='match')
                results.append(dict(row,prediction=pred,error_counts={k:c[k] for k in ('deletion','insertion','substitution')}))
    assert sha(CHECKPOINT)==before and state_hash(model)==before_state
    (OUT/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results),encoding='utf-8')
    def metric(group,view,**conditions):return summarise([r for r in results if r['group']==group and r['view']==view and all(r[k]==v for k,v in conditions.items())])
    aggregate={g:{v:metric(g,v) for v in ('clean','noisy')} for g in ('term','control')}
    by_term={c:{v:metric('term',v,truth=c) for v in ('clean','noisy')} for c in terms}
    by_font={s['label']:{g:{v:metric(g,v,setting=s['label']) for v in ('clean','noisy')} for g in ('term','control')} for s in settings}
    # Resample matched character-pair identities; all fonts/layouts/views remain grouped.
    rng=np.random.default_rng(SEED);differences={}
    for view in ('clean','noisy'):
        deltas=np.array([metric('term',view,pair=i)['cer']-metric('control',view,pair=i)['cer'] for i in range(16)])
        boot=deltas[rng.integers(0,16,(5000,16))].mean(1)
        differences[view]=dict(term_minus_control_cer=float(deltas.mean()),pair_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),clusters=16,draws=5000)
    write_json(OUT/'report.json',dict(checkpoint=str(CHECKPOINT),checkpoint_sha256=before,model_state_unchanged=True,
        training_updates=0,blank_bias=0,aggregate=aggregate,by_term=by_term,by_font=by_font,paired_differences=differences,
        elapsed_seconds=time.perf_counter()-started))
    lines=['# V30 meaningful single-character menu terms','',
        '16valid menu-relevant single-character terms,all19 fonts plus2 Noto weight variants,2native layouts,paired clean/noisy views.1,344 term images and1,344 frequency-matched other-Kanji control images. No training,lexicon or decoder change.',
        '', '| Group | Clean CER | Clean exact /672 | Noisy CER | Noisy exact /672 |','|---|---:|---:|---:|---:|']
    for g in ('term','control'):
        a,b=aggregate[g]['clean'],aggregate[g]['noisy'];lines.append(f"| {g} | {a['cer']:.2%} | {a['exact']} | {b['cer']:.2%} | {b['exact']} |")
    lines.extend(['','## Meaningful terms','', '| Term | Meaning | Clean CER | Noisy CER | Original char /standalone count | V29+V30 replay char /standalone count |','|---|---|---:|---:|---:|---:|'])
    for c,m in TERMS:lines.append(f"| {c} | {m} | {by_term[c]['clean']['cer']:.2%} | {by_term[c]['noisy']['cer']:.2%} | {counts[c]} /{single[c]} | {replay[c]} /{replay_single[c]} |")
    lines.extend(['','## Fonts on meaningful terms','', '| Font setting | Clean CER | Noisy CER |','|---|---:|---:|'])
    for s in sorted(by_font,key=lambda s:by_font[s]['term']['clean']['cer'],reverse=True):lines.append(f"| {s} | {by_font[s]['term']['clean']['cer']:.2%} | {by_font[s]['term']['noisy']['cer']:.2%} |")
    lines.extend(['','## Interpretation limits','', 'This tests recognition of menu-relevant standalone characters,not semantic understanding. Controls also have meanings. Training frequency is approximately matched,while glyph shape and visual complexity differ. Textures and renderer come from the same generator family used during V29/V30 training; results are a development diagnostic. The previous random test included small kana and rare Kanji,so direct improvement against it mixes several changes.',
        '', 'Original occurrence counts refer to the64,000-record V27 corpus. Replay counts are the actual V29 plus V30-control presentations; they omit V26/pretraining and earlier experiments. Font/layout repetitions do not create independent term identities.',
        '', '[Images and predictions](review.html) · [Complete report](report.json) · [Matching and sources](manifest.json)','', '## Lexical verification sources',''])
    lines.extend(f"- [{s['title']}]({s['url']})" for s in SOURCES)
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    review=['<!doctype html><meta charset="utf-8"><title>Meaningful Japanese single-character terms</title><style>body{font-family:Meiryo,sans-serif}table{border-collapse:collapse}td{padding:6px;border:1px solid #aaa}img{width:128px}</style>']
    for c,m in TERMS:
        review.append(f'<h2>{html.escape(c)} — {html.escape(m)}</h2><table><tr><th>Font</th><th>Layout</th><th>Clean</th><th>Noisy</th></tr>')
        for s in settings:
            for layout in range(2):
                review.append(f'<tr><td>{html.escape(s["label"])}</td><td>{layout}</td>')
                for view in ('clean','noisy'):
                    r=next(r for r in results if r['group']=='term' and r['truth']==c and r['setting']==s['label'] and r['layout']==layout and r['view']==view)
                    review.append(f'<td><img src="{r["image"]}"><br>{html.escape(r["prediction"] or "[blank]")}</td>')
                review.append('</tr>')
        review.append('</table>')
    (OUT/'review.html').write_text(''.join(review),encoding='utf-8')
    print(json.dumps(dict(aggregate=aggregate,paired_differences=differences),ensure_ascii=False),flush=True)
    print('COMPLETE',time.perf_counter()-started,flush=True)


if __name__=='__main__':main()
