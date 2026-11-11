"""V27 zero-update application-like synthetic crop evaluation. Run from project root."""
from collections import Counter
from functools import lru_cache
import hashlib
import html
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from fontTools.ttLib import TTFont
import torch

from src.recognition.data import preprocess, collate, decode, minimum_ctc_steps
from src.recognition.datasets import canvas_for
from src.recognition.metrics import align_errors, summarise
from src.recognition.model import MenuRecognizer
from src.recognition.utils import sha

OUT = Path('outputs/recognition/evaluation/v27_application_synthetic')
DATA = Path('data/datasets/recognition/v27_fonts')
CHECKPOINT = Path('outputs/recognition/training/v27_font_full20/epoch_20.pt')
SEED = 20261130
VIEWS = ('clean_control', 'texture_color', 'texture_color_noise')
LUMA = np.array([.299, .587, .114])


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


@lru_cache(maxsize=1024)
def font_at(path, size):
    return ImageFont.truetype(path, size)


def state_hash(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode()); h.update(value.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def glyph_mask(text, font_path, size, rng):
    font = font_at(str(font_path), size)
    bbox = font.getbbox(text)
    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    left, right = (max(2, round(size*rng.uniform(.06, .32))) for _ in range(2))
    top, bottom = (max(2, round(size*rng.uniform(.12, .5))) for _ in range(2))
    mask = Image.new('L', (left+w+right, top+h+bottom), 0)
    ImageDraw.Draw(mask).text((left-bbox[0], top-bbox[1]), text, font=font, fill=255)
    box = mask.getbbox()
    assert box and box[0]>0 and box[1]>0 and box[2]<mask.width and box[3]<mask.height
    return mask


def appearance(mask, texture, contrast_band, rng, nrng):
    w, h = mask.size
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    x, y = xx/max(1,w-1), yy/max(1,h-1)
    smooth = np.asarray(Image.fromarray(nrng.integers(0,256,(5,7),dtype=np.uint8)).resize((w,h),Image.Resampling.BICUBIC), dtype=float)/255-.5
    if texture == 'paper': pattern = 7*smooth + nrng.normal(0,1.5,(h,w))
    elif texture == 'wood': pattern = 4*np.sin(y*50+2*np.sin(x*7))+3*np.sin(y*133+x*9)+5*smooth
    elif texture == 'chalkboard': pattern = 9*smooth + 2*np.sin(x*13+y*19)+nrng.normal(0,1,(h,w))
    else: pattern = 2*np.sin(xx*2.5)+2*np.sin(yy*2.3)+7*smooth
    pattern += rng.uniform(-5,5)*(x-.5)+rng.uniform(-5,5)*(y-.5)
    delta = rng.uniform(*{'low':(45,70),'medium':(85,115),'high':(150,180)}[contrast_band])
    bright_ink = rng.choice((False,True))
    base = rng.uniform(28,62) if bright_ink else rng.uniform(193,227)
    ink = base+delta if bright_ink else base-delta
    def tint():
        c=nrng.uniform(-16,16,3); return c-np.dot(c,LUMA)
    bg=np.clip(base+pattern[:,:,None]+tint(),0,255)
    color=np.clip(ink+tint(),0,255)
    alpha=np.asarray(mask,dtype=float)/255
    arr=bg*(1-alpha[:,:,None])+color*alpha[:,:,None]
    core=alpha>.8
    local_delta=np.abs(np.dot(bg-color,LUMA))[core]
    assert len(local_delta) and local_delta.min()>25, 'Text must remain locally distinguishable'
    return Image.fromarray(np.rint(arr).clip(0,255).astype('uint8')), dict(
        bright_ink=bright_ink, background_luma=base, font_rgb=color.tolist(),
        nominal_luma_difference=delta, minimum_core_luma_difference=float(local_delta.min()),
        median_core_luma_difference=float(np.median(local_delta)))


def scaled_bucket(image):
    width=round(image.width*64/image.height)
    assert 0<width<=384
    canvas=canvas_for(width)
    scaled=image.resize((width,64),Image.Resampling.BICUBIC)
    padded=Image.fromarray(np.pad(np.asarray(scaled),((0,0),(0,canvas-width),(0,0)),mode='edge'))
    assert padded.size==(canvas,64)
    return padded,width,canvas


def build():
    OUT.mkdir(parents=True, exist_ok=False)
    manifest=json.loads((DATA/'manifest.json').read_text())
    fonts=[Path(p) for cohort in ('old','new') for p in manifest['paths'][cohort]]
    rows=[json.loads(s) for s in (DATA/'labels.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=[r for r in rows if r['split'] in ('seen','heldout')]
    mapping=json.loads((DATA/'vocab.json').read_text(encoding='utf-8'))['char_to_id']
    rng=random.Random(SEED); nrng=np.random.default_rng(SEED)
    records=[]; rejected=0; selected=[]
    for font_index,path in enumerate(fonts):
        with TTFont(path) as font: cmap=set(font.getBestCmap())
        for length in range(1,11):
            pool=[r for r in rows if len(r['text'])==length and all(ord(c) in cmap and c in mapping for c in r['text'])]
            unique=list(dict.fromkeys(r['text'] for r in pool))
            assert len(unique)>=2, (path,length)
            for text in rng.sample(unique,2):
                selected.append((font_index,path,text))
    rng.shuffle(selected)
    for index,(font_index,path,text) in enumerate(selected):
        # Resample the native layout only; never squeeze/crop overflowing glyphs.
        size=rng.randint(24,120)
        for attempt in range(500):
            mask=glyph_mask(text,path,size,rng)
            if round(mask.width*64/mask.height)<=384: break
            rejected+=1
        else: raise ValueError('Cannot route source crop without distortion')
        texture=('paper','wood','chalkboard','cloth')[index%4]
        band=('low','medium','high')[index%3]
        colored, meta=appearance(mask,texture,band,rng,nrng)
        alpha=np.asarray(mask,dtype=float)/255
        control=Image.fromarray(np.rint(242-222*alpha).astype('uint8')).convert('RGB')
        sigma=rng.uniform(1.5,3.5); blur=rng.uniform(.25,.55)
        noisy=np.asarray(colored.filter(ImageFilter.GaussianBlur(blur)),dtype=float)
        noisy=Image.fromarray(np.rint(noisy+nrng.normal(0,sigma,noisy.shape)).clip(0,255).astype('uint8'))
        native_dir=OUT/'native_crops';native_dir.mkdir(exist_ok=True)
        native_path=native_dir/f'{index:03d}.png';noisy.save(native_path)
        for view,image in zip(VIEWS,(control,colored,noisy)):
            padded,width,canvas=scaled_bucket(image)
            assert (width+3)//4>=minimum_ctc_steps(text)
            folder=OUT/'images'/view;folder.mkdir(parents=True,exist_ok=True)
            dest=folder/f'{index:03d}.png';padded.save(dest)
            records.append(dict(index=index,truth=text,view=view,font=path.name,font_index=font_index,
                cohort='old' if font_index<7 else 'new',orientation_flag=0,
                native_font_size=size,native_size=list(mask.size),valid_width=width,canvas=canvas,
                texture=texture,contrast=band,noise_sigma=sigma,blur_radius=blur,
                image=str(dest.relative_to(OUT)),sha256=sha(dest),native_crop=str(native_path.relative_to(OUT)),
                **meta))
    (OUT/'labels.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    dump(OUT/'manifest.json',dict(seed=SEED,examples=300,images=len(records),views=VIEWS,
        source_labels_sha256=sha(DATA/'labels.jsonl'),script_sha256=sha(Path(__file__)),
        fonts=[dict(path=str(p),sha256=sha(p)) for p in fonts],
        layout_rejections=rejected,source_font_size=[24,120],lengths=list(range(1,11)),
        source_margin_fraction_left_right=[.06,.32],source_margin_fraction_top_bottom=[.12,.5],
        routing='Aspect-preserving bicubic resize to H64; nearest128/256/384; right BORDER_REPLICATE',
        lengths_policy='ceil(unpadded resized crop width/4); artificial bucket fill excluded',
        overflow='Reject native layouts wider than384 after height scaling; no clipping or anisotropic squeeze',
        limits='Horizontal rendered crops only; known fonts/reused validation transcriptions; procedural textures, not photographs or detector evaluation',
        training_updates=0))
    return records


def score(model, records, config, mapping, grayscale=False):
    results=[]
    with torch.inference_mode():
        for canvas in (128,256,384):
            selected=[r for r in records if r['canvas']==canvas]
            for start in range(0,len(selected),32):
                chunk=selected[start:start+32];samples=[]
                for row in chunk:
                    with Image.open(OUT/row['image']) as im:
                        image=im.convert('L').convert('RGB') if grayscale else im.convert('RGB')
                    tensor,w,compressed=preprocess(image,config['mean'],config['std'],'batch')
                    assert w==canvas and not compressed
                    samples.append((tensor,torch.tensor([mapping[c] for c in row['truth']]),0,row['valid_width'],row['truth'],False))
                batch=collate(samples)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=model(batch['images'].cuda(),batch['orientation'].cuda(),batch['input_lengths'])
                predictions=decode(logits,mapping,batch['input_lengths'])
                for row,prediction in zip(chunk,predictions):
                    counts=Counter(o['type'] for o in align_errors(row['truth'],prediction) if o['type']!='match')
                    results.append(dict(**row,prediction=prediction,error_counts={k:counts[k] for k in ('deletion','insertion','substitution')}))
    result={'all':summarise(results)}
    for field in ('texture','contrast','canvas','font','cohort'):
        result['by_'+field]={str(k):summarise([r for r in results if r[field]==k]) for k in sorted({r[field] for r in results})}
    result['by_length']={str(n):summarise([r for r in results if len(r['truth'])==n]) for n in range(1,11)}
    result['by_source_size']={f'{a}-{b}':summarise([r for r in results if a<=r['native_font_size']<=b]) for a,b in ((24,47),(48,79),(80,120))}
    result['predictions']=results
    return result


def main():
    started=time.perf_counter();torch.set_num_threads(4)
    records=build();print('Generated',len(records),'images',flush=True)
    before=sha(CHECKPOINT);saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    model=MenuRecognizer(len(saved['vocab']['char_to_id'])+1,use_bilstm=True).cuda()
    model.load_state_dict(saved['model'],strict=True);model.eval();original_state=state_hash(model)
    results={}
    for view in VIEWS:
        results[view]=score(model,[r for r in records if r['view']==view],saved['config'],saved['vocab']['char_to_id'])
        print(view,results[view]['all'],flush=True)
    results['noise_grayscale']=score(model,[r for r in records if r['view']==VIEWS[-1]],saved['config'],saved['vocab']['char_to_id'],True)
    print('noise_grayscale',results['noise_grayscale']['all'],flush=True)
    assert state_hash(model)==original_state and sha(CHECKPOINT)==before
    dump(OUT/'report.json',dict(results=results,checkpoint=str(CHECKPOINT),checkpoint_sha256=before,
        model_state_unchanged=True,training_updates=0,blank_bias=0,decoder='greedy CTC',elapsed_seconds=time.perf_counter()-started))
    lines=['# V27 application-like synthetic pilot','',
        '300 paired examples, 15 fonts, 1–10 characters, 24–120px native font size. No training. V27 epoch20; raw greedy CTC, blank bias0.',
        '', '| View | CER | Exact /300 | Deletions | Insertions | Substitutions |','|---|---:|---:|---:|---:|---:|']
    for key,result in results.items():
        m=result['all'];c=m['error_counts'];lines.append(f"| {key} | {m['cer']:.2%} | {m['exact']} | {c['deletion']} | {c['insertion']} | {c['substitution']} |")
    for field in ('contrast','texture','length','canvas','source_size'):
        lines.extend(['',f'## Noisy RGB results by {field}','', '| Group | Lines | CER | Exact |','|---|---:|---:|---:|'])
        for key,m in results['texture_color_noise']['by_'+field].items():
            lines.append(f"| {key} | {m['lines']} | {m['cer']:.2%} | {m['exact']} |")
    lines.extend(['','Crops are resized proportionally to H64 and edge-extended to128/256/384. Lengths exclude artificial padding. Overflowing layouts are rejected, not squeezed. Mild native-scale Gaussian blur/noise precedes resizing.',
        '', 'These are known-font, reused-transcription procedural examples, not independent real-menu or YOLO accuracy. Contrast groups are an engineered stress test. Different groups contain different texts; paired views share exact geometry and labels. Native text sizes are randomly sampled, and long lines favor taller crops to fit384.',
        '', '[Visual review](review.html) · [Full report](report.json) · [Generation manifest](manifest.json)'])
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lookup={key:{r['index']:r for r in value['predictions']} for key,value in results.items()}
    review=['<!doctype html><meta charset="utf-8"><title>V27 application pilot</title><style>body{font-family:Meiryo,sans-serif}td{border:1px solid #aaa;padding:8px}img{max-width:384px}table{border-collapse:collapse}</style><h1>V27 application-like crops</h1><p>300 paired examples; predictions shown below the images. Beta0, no lexicon.</p><table><tr><th>Truth / conditions</th>']
    review.extend(f'<th>{key}</th>' for key in results);review.append('</tr>')
    preview=Image.new('RGB',(1200,24*90),'#dddddd');draw=ImageDraw.Draw(preview)
    for index in range(300):
        row=lookup['clean_control'][index]
        review.append(f'<tr><td>{html.escape(row["truth"])}<br>{html.escape(row["font"])}<br>{row["native_font_size"]}px → H64<br>{row["texture"]} / {row["contrast"]}<br>valid{row["valid_width"]} / bucket{row["canvas"]}</td>')
        for key in results:
            r=lookup[key][index]
            review.append(f'<td><img src="{r["image"]}"'+(' style="filter:grayscale(1)"' if key=='noise_grayscale' else '')+f'><br>{html.escape(r["prediction"])}<br>{html.escape(str(r["error_counts"]))}</td>')
        review.append('</tr>')
    for k,index in enumerate(range(0,300,13)):
        if k>=24:break
        for col,view in enumerate(VIEWS):
            r=lookup[view][index]
            with Image.open(OUT/r['image']) as im:preview.paste(im,(col*400,k*90+23))
            draw.text((col*400,k*90+3),f'{index} {view} {r["contrast"]}',fill='black')
    preview.save(OUT/'preview.png')
    (OUT/'review.html').write_text(''.join(review)+'</table>',encoding='utf-8')
    print('COMPLETE',time.perf_counter()-started,flush=True)


if __name__=='__main__': main()
