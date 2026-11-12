"""Expanded-vocabulary corpus and audited 2000-update rare-character plan."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image,ImageFilter
from fontTools.ttLib import TTFont

from scripts.recognition.train_v38_yolo25 import INPUT,HOLDOUT,BucketCrops
from outputs.recognition.application_mixture_pilot import ROOT,RenderedLines
from outputs.recognition.application_pilot import glyph_mask,appearance
from outputs.recognition.v36_yolo_fullpasses import Combined,load,dump_rows
from outputs.recognition.evaluate_bn_swap import V34
from outputs.recognition.tiered_short_pilot import rows
from src.recognition.datasets import FontLines
from src.recognition.font_training import row_canvas
from src.detection.line_crops import recognition_input
from src.recognition.data import minimum_ctc_steps
from src.recognition.vocabulary import extend_vocabulary
from src.recognition.utils import sha,write_json

SEED=20261120
WORK=Path('outputs/recognition/training/v40_new_characters_2000')


def prepare(work,source,status):
    cfg=source['config'];old=source['vocab'];delivered=rows(INPUT/'labeled_crops.jsonl')
    vocab,added=extend_vocabulary(old,[c for r in delivered for c in r['text']]);mapping=vocab['char_to_id']
    assert len(old['char_to_id'])==1722 and len(added)==57 and len(mapping)==1779
    assert all(mapping[c]==i for c,i in old['char_to_id'].items())
    write_json(work/'vocab.json',vocab)
    versioned=Path('data/processed/recognition/vocab_v40.json');versioned.parent.mkdir(parents=True,exist_ok=True)
    if versioned.exists():assert load(versioned)==vocab
    else:write_json(versioned,vocab)
    eligible=[]
    for r in delivered:
        if r['status']!='OK':continue
        assert r['text'] and all(c in mapping for c in r['text'])
        assert (r['valid_width']+3)//4>=minimum_ctc_steps(r['text'])
        p=INPUT/r['ctc_crop'];assert p.resolve().is_relative_to(INPUT.resolve()) and sha(p)==r['ctc_sha256']
        with Image.open(p) as im:assert im.mode=='RGB' and im.size==(r['canvas'],64)
        eligible.append(dict(r,image=r['ctc_crop'],sha256=r['ctc_sha256'],split='val_holdout' if r['source_menu'] in HOLDOUT else 'train'))
    ytrain=[r for r in eligible if r['split']=='train'];yhold=[r for r in eligible if r['split']=='val_holdout']
    assert len(ytrain)==1561 and len(yhold)==156
    assert not {r['source_menu'] for r in ytrain}&HOLDOUT
    assert not {r['sha256'] for r in ytrain}&{r['sha256'] for r in yhold}
    dump_rows(work/'yolo_labels.jsonl',eligible)
    manifest=load(ROOT/'manifest.json');font_paths=manifest['paths'];cmaps={};font_hashes={}
    for cohort in ('old','new','heldout'):
        for path in font_paths[cohort]:
            with TTFont(path,fontNumber=0) as font:cmaps[path]=set(font.getBestCmap())
            font_hashes[path]=sha(Path(path))
    known=font_paths['old']+font_paths['new'];held=font_paths['heldout']
    assert len(known)==15 and len(held)==4
    target=work/'targeted';target.mkdir(exist_ok=True)
    target_rows=[];support={}
    for ci,c in enumerate(added):
        support[c]={cohort:[p for p in paths if ord(c) in cmaps[p]] for cohort,paths in (('known',known),('heldout',held))}
        assert support[c]['known'] and support[c]['heldout'],repr(c)
        contexts=list(dict.fromkeys(r['text'] for r in ytrain if c in r['text']))
        fallback=('魚'+c+'円')
        for split,count in (('train',80),('probe',12)):
            for i in range(count):
                cohort='heldout' if split=='probe' and i>=6 else 'known'
                paths=support[c][cohort];path=paths[i%len(paths)]
                rng=random.Random(SEED+ci*10000+i+(1000000 if split=='probe' else 0))
                nrng=np.random.default_rng(SEED+ci*10000+i+(1000000 if split=='probe' else 0))
                if split=='train' and contexts and i%2==0:
                    choices=[t for t in contexts if all(ord(x) in cmaps[path] for x in t)]
                    text=rng.choice(choices) if choices else fallback
                elif '\u4e00'<=c<='\u9fff' and i%3==0:text=c
                else:text=fallback
                assert c in text and all(x in mapping and ord(x) in cmaps[path] for x in text),(repr(c),path,text)
                seed=SEED+ci*10000+i+(1000000 if split=='probe' else 0)
                out=target/'images'/split/f'{ci:02d}_{i:03d}.png';out.parent.mkdir(parents=True,exist_ok=True)
                if not out.exists():
                    mask=glyph_mask(text,Path(path),rng.randint(32,96),rng)
                    if i%2:
                        image,_=appearance(mask,('paper','wood','chalkboard','cloth')[i%4],('medium','high')[i%2],rng,nrng)
                        image=image.filter(ImageFilter.GaussianBlur(rng.uniform(.15,.35)))
                        pixels=np.asarray(image,dtype=float);image=Image.fromarray(np.rint(pixels+nrng.normal(0,1.5,pixels.shape)).clip(0,255).astype('uint8'))
                    else:image=Image.fromarray(242-np.rint(np.asarray(mask,dtype=float)*222/255).astype('uint8')).convert('RGB')
                    normalized,meta=recognition_input(image);normalized.save(out,compress_level=1)
                    write_json(out.with_suffix('.json'),meta)
                meta=load(out.with_suffix('.json'))
                assert (meta['valid_width']+3)//4>=minimum_ctc_steps(text)
                target_rows.append(dict(image=str(out.relative_to(target)),sha256=sha(out),text=text,split=split,
                    target_character=c,preferred_font=Path(path).name,cohort=cohort,font_path=path,
                    view='texture_color_noise' if i%2 else 'clean_control',orientation_flag=0,seed=seed,**meta))
        status(state='preparing_targeted_characters',completed_characters=ci+1,total_characters=len(added))
    dump_rows(target/'labels.jsonl',target_rows)
    training=[r for r in target_rows if r['split']=='train'];probe=[r for r in target_rows if r['split']=='probe']
    assert len(training)==4560 and len(probe)==684 and not {r['sha256'] for r in training}&{r['sha256'] for r in probe}
    original=FontLines(ROOT,'train',cfg['mean'],cfg['std']);generated=rows(V34/'corpus/labels.jsonl')
    corpus=load(V34/'corpus/manifest.json')
    assert sha(ROOT/'labels.jsonl')==corpus['original_labels_sha256'] and sha(V34/'corpus/labels.jsonl')==corpus['labels_sha256']
    mix=Combined(original,RenderedLines(generated,V34/'corpus',cfg,mapping),BucketCrops(ytrain,INPUT,cfg,mapping))
    mix.datasets['target']=BucketCrops(training,target,cfg,mapping)
    required=corpus['required_terms'];assert len(required)==521
    mandatory_o=set();mandatory_g={'clean_control':set(),'texture_color_noise':set()}
    for t in required:
        hits=[i for i,r in enumerate(original.rows) if r['text']==t]
        if hits:mandatory_o.add(hits[0])
        else:
            for view in mandatory_g:mandatory_g[view].add(next(i for i,r in enumerate(generated) if r['text']==t and r['view']==view))
    rng=random.Random(SEED);plan=[]
    def width(kind,index):
        row=mix.datasets[kind].rows[index]
        return row_canvas(row) if kind=='original' else row['canvas']
    def ordinary(kind,pool,count,mandatory=()):
        groups=defaultdict(list);must=defaultdict(list)
        for i in pool:groups[width(kind,i)].append(i)
        for i in sorted(mandatory):must[width(kind,i)].append(i)
        widths=[]
        for w,indices in sorted(must.items()):widths.extend([w]*((len(indices)+31)//32))
        choices=sorted(groups);widths+=rng.choices(choices,weights=[len(groups[w]) for w in choices],k=count-len(widths))
        queues={w:[] for w in choices}
        def take(w,n):
            result=[]
            while len(result)<n:
                if not queues[w]:queues[w]=rng.sample(groups[w],len(groups[w]))
                need=min(n-len(result),len(queues[w]));result+=queues[w][:need];del queues[w][:need]
            return result
        for w in widths:
            forced=must[w][:32];del must[w][:len(forced)]
            indices=forced+take(w,32-len(forced));rng.shuffle(indices);plan.append([(kind,i) for i in indices])
        assert not any(must.values())
    ordinary('yolo',range(len(ytrain)),500,range(len(ytrain)))
    ordinary('original',range(len(original)),250,mandatory_o)
    for view in ('clean_control','texture_color_noise'):
        ordinary('generated',[i for i,r in enumerate(generated) if r['view']==view],125,mandatory_g[view])
    by_char=defaultdict(lambda:defaultdict(list));by_width=defaultdict(list);queues={}
    for i,r in enumerate(training):by_char[r['target_character']][r['canvas']].append(i);by_width[r['canvas']].append(i)
    for j in range(1000):
        c=added[j%len(added)];choices=sorted(w for w in by_char[c] if len(by_char[c][w])>=8)
        assert choices,'Each target character needs a diverse same-width anchor pool'
        w=rng.choices(choices,weights=[len(by_char[c][w]) for w in choices])[0]
        key=(c,w)
        while len(queues.get(key,[]))<8:queues[key]=queues.get(key,[])+rng.sample(by_char[c][w],len(by_char[c][w]))
        anchors=queues[key][:8];del queues[key][:8]
        ids=anchors+rng.choices(by_width[w],k=24);rng.shuffle(ids);plan.append([('target',i) for i in ids])
    rng.shuffle(plan);assert len(plan)==2000 and all(len(b)==32 for b in plan)
    encounters=Counter();new_counts=Counter();term_counts=Counter();unique=defaultdict(set)
    for batch in plan:
        assert len({width(k,i) for k,i in batch})==1
        for k,i in batch:
            encounters[k]+=1;unique[k].add(i);text=mix.datasets[k].rows[i]['text'];term_counts[text]+=1;new_counts.update(c for c in text if c in added)
    assert encounters=={'target':32000,'yolo':16000,'original':8000,'generated':8000}
    assert all(new_counts[c]>=100 for c in added) and all(term_counts[t]>=1 for t in required)
    assert len(unique['yolo'])==1561
    write_json(work/'batches.json',plan)
    sampling=dict(updates=2000,presentations=64000,streams=dict(encounters),yolo_fraction=.25,target_fraction=.5,
        old_characters=1722,new_characters=57,total_characters=1779,added=added,
        minimum_new_character_encounters=min(new_counts[c] for c in added),planned_new_character_encounters={c:new_counts[c] for c in added},
        unique_records={k:len(v) for k,v in unique.items()},target_unique_training_images=4560,target_probe_images=684,
        yolo_train=1561,yolo_holdout=156,holdout_menus=sorted(HOLDOUT),newly_included_OK_labels=246,
        required_terms=521,required_term_encounters={t:term_counts[t] for t in required},font_support=support,font_sha256=font_hashes,
        vocabulary_sha256=sha(work/'vocab.json'),input_labels_sha256=sha(INPUT/'labeled_crops.jsonl'),plan_sha256=sha(work/'batches.json'),
        target_labels_sha256=sha(target/'labels.jsonl'),yolo_labels_sha256=sha(work/'yolo_labels.jsonl'),
        labels_policy='Only OK YOLO labels; full original text including spaces/punctuation. New characters from all VLM labels, including two valid characters otherwise only in non-OK rows. Non-OK images never trained.',
        probe_policy='New render seeds; 15known and4heldoutfont families. Not unseen-character or fully unseen-label evaluation. Training contexts use train menus only; no heldout menu images or transcriptions used to build contexts.',
        whitespace_policy='Spaces rendered inside lines, never blank-only images. Case/fullwidth/punctuation preserved; no glyph remapping or tofu fallback.')
    write_json(work/'sampling.json',sampling)
    return vocab,added,mix,plan,BucketCrops(probe,target,cfg,mapping),BucketCrops(yhold,INPUT,cfg,mapping),sampling
