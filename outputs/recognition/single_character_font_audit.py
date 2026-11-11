"""Exactly1,000 fresh images: all19 fonts plus regular-weight Noto diagnostics."""
from collections import Counter
from functools import lru_cache
import html
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from fontTools.ttLib import TTFont
import torch

from outputs.recognition.application_pilot import appearance, scaled_bucket, state_hash
from src.recognition.data import preprocess, collate, decode
from src.recognition.metrics import align_errors, summarise
from src.recognition.model import MenuRecognizer
from src.recognition.utils import sha, write_json

OUT=Path('outputs/recognition/evaluation/v30_onechar_fonts_1000')
DATA=Path('data/datasets/recognition/v27_fonts')
CHECKPOINT=Path('outputs/recognition/pilots/v30_short_retention_2000/control/best.pt')
SEED=20261204


@lru_cache(maxsize=256)
def font_at(path,size,weight):
    f=ImageFont.truetype(path,size,index=0)
    if weight:
        axes=f.get_variation_axes()
        f.set_variation_by_axes([min(a['maximum'],max(a['minimum'],weight)) if a['name']==b'Weight' else a['default'] for a in axes])
    return f


def mask_for(char,path,size,weight,rng):
    f=font_at(str(path),size,weight);bbox=f.getbbox(char);w,h=bbox[2]-bbox[0],bbox[3]-bbox[1]
    l,r=[max(2,round(size*rng.uniform(.06,.32))) for _ in range(2)]
    t,b=[max(2,round(size*rng.uniform(.12,.5))) for _ in range(2)]
    mask=Image.new('L',(l+w+r,t+h+b),0)
    ImageDraw.Draw(mask).text((l-bbox[0],t-bbox[1]),char,font=f,fill=255)
    ink=mask.getbbox();assert ink and min(ink[:2])>0 and ink[2]<mask.width and ink[3]<mask.height
    return mask


def main():
    OUT.mkdir(parents=True,exist_ok=False);started=time.perf_counter();torch.set_num_threads(4)
    manifest=json.loads((DATA/'manifest.json').read_text());mapping=json.loads((DATA/'vocab.json').read_text(encoding='utf-8'))['char_to_id']
    settings=[];cmaps=[]
    for cohort in ('old','new','heldout'):
        for path in manifest['paths'][cohort]:
            path=Path(path)
            with TTFont(path,fontNumber=0 if path.suffix.lower()=='.ttc' else -1) as f:
                cmaps.append(set(f.getBestCmap()));axes={a.axisTag:a.defaultValue for a in f['fvar'].axes} if 'fvar' in f else {}
            label=path.stem+('_default'+str(int(axes['wght'])) if 'wght' in axes else '')
            settings.append(dict(label=label,path=str(path),cohort=cohort,weight=0,default_axes=axes,sha256=sha(path)))
    for s in list(settings):
        if 'wght' in s['default_axes']:settings.append(dict(s,label=Path(s['path']).stem+'_weight400',weight=400))
    assert len(settings)==21
    common=set.intersection(*cmaps);characters=[c for c in mapping if ord(c) in common]
    counts=Counter(c for line in (DATA/'labels.jsonl').read_text(encoding='utf-8').splitlines() if (r:=json.loads(line))['split']=='train' for c in r['text'])
    rng=random.Random(SEED)
    hira=[c for c in characters if '\u3041'<=c<='\u3096'];kata=[c for c in characters if '\u30a1'<=c<='\u30fa']
    kanji=sorted((c for c in characters if '\u4e00'<=c<='\u9fff'),key=lambda c:(counts[c],c))
    assert min(len(hira),len(kata))>=4 and len(kanji)>=15
    chosen=rng.sample(hira,4)+rng.sample(kata,4)
    thirds=[kanji[:len(kanji)//3],kanji[len(kanji)//3:2*len(kanji)//3],kanji[2*len(kanji)//3:]]
    for pool in thirds:chosen+=rng.sample(pool,5)
    assert len(chosen)==23
    extra=rng.choice([c for c in kanji if c not in chosen]);rows=[]
    folder=OUT/'images';folder.mkdir()
    for sindex,setting in enumerate(settings):
        for cindex,char in enumerate(chosen+([extra] if sindex<17 else [])):
            # Same seed, native font size and margin fractions per character across font settings.
            seed=SEED+101*cindex;crng=random.Random(seed);nrng=np.random.default_rng(seed)
            size=crng.randint(24,120);mask=mask_for(char,setting['path'],size,setting['weight'],crng)
            texture=('paper','wood','chalkboard','cloth')[cindex%4];contrast=('low','medium','high')[cindex%3]
            colored,appearance_meta=appearance(mask,texture,contrast,crng,nrng)
            a=np.asarray(mask,dtype=float)/255;clean=Image.fromarray(np.rint(242-222*a).astype('uint8')).convert('RGB')
            sigma=crng.uniform(1.5,3.5);blur=crng.uniform(.25,.55)
            arr=np.asarray(colored.filter(ImageFilter.GaussianBlur(blur)),dtype=float)
            noisy=Image.fromarray(np.rint(arr+nrng.normal(0,sigma,arr.shape)).clip(0,255).astype('uint8'))
            box=mask.getbbox()
            for view,image in (('clean',clean),('noisy',noisy)):
                padded,width,canvas=scaled_bucket(image);assert canvas==128
                dest=folder/f'{sindex:02d}_{cindex:02d}_{view}.png';padded.save(dest)
                rows.append(dict(index=len(rows),truth=char,setting=setting['label'],font=Path(setting['path']).name,
                    cohort=setting['cohort'],weight=setting['weight'] or setting['default_axes'].get('wght'),
                    core=cindex<23,view=view,native_font_size=size,valid_width=width,canvas=canvas,
                    scaled_ink_height=(box[3]-box[1])*64/mask.height,
                    scaled_ink_width=(box[2]-box[0])*64/mask.height,training_character_count=counts[char],
                    image=str(dest.relative_to(OUT)),sha256=sha(dest),texture=texture,contrast=contrast,**appearance_meta))
    assert len(rows)==1000 and sum(r['core'] for r in rows)==966
    (OUT/'labels.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    write_json(OUT/'manifest.json',dict(seed=SEED,images=1000,paired_examples=500,unique_fonts=19,font_settings=21,
        core_characters=chosen,extra_character=extra,settings=settings,
        core_design='23 matched characters per setting,clean/noisy paired;966 images. Seventeen extra pairs complete1000 and are excluded from font rankings.',
        character_design='4hiragana,4katakana,15kanji:5from each third of training frequency among glyphs supported by every font. No fallback fonts.',
        normalization='Native font24–120px; random source margins; proportional H64 resize;edge-replicate to128; valid lengths exclude bucket padding.',
        limits='Only23 independent character identities per font in primary ranking; conditional on these characters, native crops and procedural appearance. This is not a broad population font benchmark.',
        script_sha256=sha(Path(__file__))))
    before=sha(CHECKPOINT);saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);assert saved['updates']==2000 and saved['arm']=='control'
    model=MenuRecognizer(len(mapping)+1,use_bilstm=True).cuda();model.load_state_dict(saved['model']);model.eval();before_state=state_hash(model)
    results=[]
    with torch.inference_mode():
        for start in range(0,len(rows),32):
            chunk=rows[start:start+32];samples=[]
            for r in chunk:
                with Image.open(OUT/r['image']) as im:t,w,compressed=preprocess(im,saved['config']['mean'],saved['config']['std'],'batch')
                assert w==128 and not compressed
                samples.append((t,torch.tensor([mapping[r['truth']]]),0,r['valid_width'],r['truth'],False))
            batch=collate(samples)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(batch['images'].cuda(),batch['orientation'].cuda(),batch['input_lengths'])
            for row,prediction in zip(chunk,decode(logits,mapping,batch['input_lengths'])):
                errors=Counter(op['type'] for op in align_errors(row['truth'],prediction) if op['type']!='match')
                results.append(dict(row,prediction=prediction,error_counts={k:errors[k] for k in ('deletion','insertion','substitution')}))
    assert sha(CHECKPOINT)==before and state_hash(model)==before_state
    (OUT/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results),encoding='utf-8')
    core=[r for r in results if r['core']]
    fonts={s['label']:{view:summarise([r for r in core if r['setting']==s['label'] and r['view']==view]) for view in ('clean','noisy')} for s in settings}
    confusions={s['label']:Counter((r['truth'],r['prediction']) for r in core if r['setting']==s['label'] and r['truth']!=r['prediction']) for s in settings}
    chars={c:{view:summarise([r for r in core if r['truth']==c and r['view']==view]) for view in ('clean','noisy')} for c in chosen}
    write_json(OUT/'report.json',dict(checkpoint=str(CHECKPOINT),checkpoint_sha256=before,model_state_unchanged=True,
        training_updates=0,blank_bias=0,all=summarise(results),core=summarise(core),by_setting=fonts,by_character=chars,
        top_confusions={s:[dict(truth=a,prediction=b,count=n) for (a,b),n in v.most_common(5)] for s,v in confusions.items()},
        elapsed_seconds=time.perf_counter()-started))
    ordered=sorted(fonts,key=lambda s:(fonts[s]['clean']['cer'],fonts[s]['noisy']['cer']),reverse=True)
    lines=['# Single-character font audit:1,000 images','',
        'V30 control update2000; raw greedy CTC,blank penalty0. Primary font comparison uses the same23 characters per setting,each with clean/noisy views.19 fonts plus2 regular-weight Noto settings.17 extra pairs complete1000 and are excluded from the ranking.',
        '', '| Font setting | Clean CER | Clean exact /23 | Noisy CER | Noisy exact /23 | Clean D/I/S | Noisy D/I/S |',
        '|---|---:|---:|---:|---:|---|---|']
    for label in ordered:
        clean,noisy=fonts[label]['clean'],fonts[label]['noisy']
        c='/'.join(str(clean['error_counts'][k]) for k in ('deletion','insertion','substitution'));n='/'.join(str(noisy['error_counts'][k]) for k in ('deletion','insertion','substitution'))
        lines.append(f"| {label} | {clean['cer']:.2%} | {clean['exact']} | {noisy['cer']:.2%} | {noisy['exact']} | {c} | {n} |")
    lines.extend(['','Only23 shared characters per font: one edit changes CER4.35 points. Rankings are descriptive,not a reliable ordering of close fonts. All fonts explicitly support every character; no tofu/fallback substitutions were rendered. Native size and margin fractions are matched per character,but actual glyph proportions vary by font.',
        '', 'Noto default weights match the application renderer; weight400 matches the original V27 renderer. Paired weight results help distinguish font-family issues from stroke-weight mismatch.',
        '', '[Image/prediction review](review.html) · [Detailed metrics](report.json) · [Manifest](manifest.json)'])
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    review=['<!doctype html><meta charset="utf-8"><title>Single-character font audit</title><style>body{font-family:Meiryo,sans-serif}td{padding:6px;border:1px solid #aaa}table{border-collapse:collapse}img{width:128px}</style><h1>Font audit</h1>']
    preview=Image.new('RGB',(1100,21*105),'#dddddd');draw=ImageDraw.Draw(preview)
    for si,label in enumerate(ordered):
        review.append(f'<h2>{html.escape(label)}</h2><table><tr><th>Truth</th><th>Clean</th><th>Noisy</th></tr>')
        for c in chosen:
            review.append('<tr><td>'+html.escape(c)+'</td>')
            for view in ('clean','noisy'):
                r=next(r for r in core if r['setting']==label and r['truth']==c and r['view']==view)
                review.append(f'<td><img src="{r["image"]}"><br>{html.escape(r["prediction"] or "[blank]")}</td>')
            review.append('</tr>')
        review.append('</table>')
        draw.text((4,si*105+3),label,fill='black')
        candidates=[r for r in core if r['setting']==label and r['view']=='clean' and r['truth']!=r['prediction']][:4]
        candidates += [r for r in core if r['setting']==label and r['view']=='noisy' and r['truth']!=r['prediction']][:4]
        for col,r in enumerate(candidates):
            with Image.open(OUT/r['image']) as im:preview.paste(im,(col*136,si*105+24))
            draw.text((col*136,si*105+88),r['view'],fill='black')
    preview.save(OUT/'error_preview.png');(OUT/'review.html').write_text(''.join(review),encoding='utf-8')
    print((OUT/'summary.md').read_text(encoding='utf-8'),flush=True)
    print('COMPLETE',time.perf_counter()-started,flush=True)


if __name__=='__main__':main()
