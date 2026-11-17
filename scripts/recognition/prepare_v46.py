"""Focused Latin/parentheses/digits/basic-Japanese adaptation corpus."""
from collections import Counter,defaultdict
from pathlib import Path
import math
import random
import numpy as np
from PIL import Image,ImageFont
from fontTools.ttLib import TTFont

from scripts.recognition.train_v45_latin_head import prepare as prepare45,rows,V44,START,V34
from scripts.recognition.prepare_v44 import mask_with_boxes
from scripts.recognition.train_v38_yolo25 import BucketCrops
from src.recognition.long_line_inference import bucket_image
from src.recognition.font_training import row_canvas
from src.recognition.utils import write_json,sha
from outputs.recognition.v36_yolo_fullpasses import load,dump_rows

SEED=20261146
WORK=Path('outputs/recognition/training/v46_focused_recurrent_4000')


def basic_corpus(work,mapping):
    root=work/'basic_japanese';root.mkdir(exist_ok=True)
    manifest=load(V44/'targeted/manifest.json');known=manifest['training_fonts'];held=manifest['heldout_fonts']
    requested=[chr(i) for a,b in ((0x3041,0x3096),(0x30a1,0x30fa)) for i in range(a,b+1) if chr(i).isalpha()]
    requested+=list('日月火水木金土上下大小山川口人一二三十百千円')
    chars=sorted(set(requested)&set(mapping));excluded=sorted(set(requested)-set(mapping))
    cmaps={}
    for path in known+held:
        with TTFont(path,fontNumber=0) as font:cmaps[path]=set(font.getBestCmap())
    records=[]
    for split,count in (('train',6),('probe',4)):
        for ci,c in enumerate(chars):
            for j in range(count):
                rng=random.Random(SEED+ci*100+j+(100000 if split=='probe' else 0))
                # Small kana also appear in context, preserving relative glyph size.
                text=c if j%3==0 else c*2 if j%3==1 else c+rng.choice(chars)
                cohort='heldout' if split=='probe' and j>=2 else 'known'
                supported=[p for p in (held if cohort=='heldout' else known) if all(ord(x) in cmaps[p] for x in text)]
                assert supported,(c,text,cohort)
                path=supported[(ci+j)%len(supported)];dest=root/'images'/f'{split}_{ci:03d}_{j}.png'
                dest.parent.mkdir(exist_ok=True)
                if not dest.exists():
                    mask,_=mask_with_boxes(text,path,rng.randint(38,82),rng)
                    bg,ink=(242,20) if j%2 else (20,235)
                    image=Image.fromarray(np.rint(bg+(ink-bg)*np.asarray(mask,dtype=float)/255).clip(0,255).astype('uint8')).convert('RGB')
                    image,meta=bucket_image(image,256);assert meta['scaled_height']==64
                    image.save(dest,compress_level=1);write_json(dest.with_suffix('.json'),meta)
                meta=load(dest.with_suffix('.json'))
                records.append(dict(text=text,anchor=c,split=split,cohort=cohort,image=str(dest.relative_to(root)),sha256=sha(dest),
                    orientation_flag=0,preferred_font=Path(path).name,font_path=path,**meta))
    train=[r for r in records if r['split']=='train'];probe=[r for r in records if r['split']=='probe']
    assert not {r['sha256'] for r in train}&{r['sha256'] for r in probe}
    dump_rows(root/'labels.jsonl',records);write_json(root/'manifest.json',dict(anchors=chars,excluded_oov=excluded,train=len(train),probe=len(probe),known_fonts=known,heldout_fonts=held))
    return root,train,probe


def prepare(work,source):
    replay=work/'replay';replay.mkdir(exist_ok=True)
    vocab,added,mix,_,extra,_=prepare45(replay,source);mapping=vocab['char_to_id'];cfg=source['config']
    root,train,probe=basic_corpus(work,mapping)
    mix.datasets['basic']=BucketCrops(train,root,cfg,mapping)
    targeted=rows(V44/'targeted/labels.jsonl')
    mix.datasets['parentheses']=BucketCrops([r for r in targeted if r['kind']=='parentheses' and r['split']=='train' and int(Path(r['image']).stem)%5!=0],V44/'targeted',cfg,mapping)
    for cohort in ('known','heldout'):
        extra['parentheses_'+cohort]=BucketCrops([r for r in targeted if r['kind']=='parentheses' and r['split']=='probe' and r['cohort']==cohort],V44/'targeted',cfg,mapping)
        extra['basic_'+cohort]=BucketCrops([r for r in probe if r['cohort']==cohort],root,cfg,mapping)
    mix.datasets['digits']=BucketCrops([r for r in rows(START.parent/'digits/labels.jsonl') if r['split']=='train'],START.parent/'digits',cfg,mapping)
    mix.datasets['generated_clean']=mix.datasets['generated']
    mix.datasets['generated_noisy']=mix.datasets['generated']
    required=load(V34/'corpus/manifest.json')['required_terms'];mo=set();mg=set()
    for t in required:
        i=next((i for i,r in enumerate(mix.datasets['original'].rows) if r['text']==t),None)
        if i is not None:mo.add(i)
        else:mg.add(next(i for i,r in enumerate(mix.datasets['generated'].rows) if r['text']==t and r['view']=='clean_control'))
    rng=random.Random(SEED);plan=[]
    def width(k,i):
        r=mix.datasets[k].rows[i];return row_canvas(r) if k=='original' else r['canvas']
    def schedule(k,pool,batches,mandatory=()):
        groups=defaultdict(list);must=defaultdict(list)
        for i in pool:groups[width(k,i)].append(i)
        for i in sorted(mandatory):must[width(k,i)].append(i)
        order=[w for w,ids in sorted(must.items()) for _ in range(math.ceil(len(ids)/32))]
        assert len(order)<=batches,(k,len(order),batches)
        ws=sorted(groups);order+=rng.choices(ws,weights=[len(groups[w]) for w in ws],k=batches-len(order))
        for w in order:
            ids=must[w][:32];del must[w][:len(ids)];ids+=rng.choices(groups[w],k=32-len(ids));rng.shuffle(ids)
            plan.append([(k,i) for i in ids])
        assert not any(must.values())
    proportions=dict(latin=300,parentheses=100,digits=150,basic=50,yolo=200,original=100,generated_clean=50,generated_noisy=50)
    digit_ids=list(range(6000));rng.shuffle(digit_ids)
    for block in range(4):
        start=len(plan)
        for k,n in proportions.items():
            pool=list(range(len(mix.datasets[k])));mandatory=[]
            if k in ('latin','parentheses','basic','yolo'):mandatory=pool
            if k=='digits':mandatory=digit_ids[block*1500:(block+1)*1500]
            if k=='original':mandatory=mo
            if k.startswith('generated_'):
                view='clean_control' if k.endswith('clean') else 'texture_color_noise'
                pool=[i for i,r in enumerate(mix.datasets[k].rows) if r['view']==view]
                if k.endswith('clean'):mandatory=mg
            schedule(k,pool,n,mandatory)
        segment=plan[start:];rng.shuffle(segment);plan[start:]=segment
        assert Counter(k for b in segment for k,i in b)=={k:v*32 for k,v in proportions.items()}
    streams=Counter();chars=Counter();terms=Counter();canvases=Counter();kd=Counter()
    for batch in plan:
        assert len(batch)==32 and len({width(k,i) for k,i in batch})==1
        for k,i in batch:
            t=mix.datasets[k].rows[i]['text'];assert all(c in mapping for c in t)
            streams[k]+=1;chars.update(t);terms[t]+=1;canvases[width(k,i)]+=1
            if k in ('yolo','original','generated_clean','generated_noisy'):kd['excluded_new_letter' if any(c in added for c in t) else 'eligible']+=1
    assert len(plan)==4000 and sum(streams.values())==128000 and all(terms[t]>=4 for t in required)
    sampling=dict(updates=4000,presentations=128000,streams=dict(streams),ratios={k:v/1000 for k,v in proportions.items()},
        physical_width_presentations=dict(canvases),latin_added=added,all_latin_characters={c:chars[c] for c in __import__('string').ascii_letters},
        basic_anchor_characters=load(root/'manifest.json')['anchors'],basic_characters={c:chars[c] for c in load(root/'manifest.json')['anchors']},
        parentheses_token_encounters={c:chars[c] for c in '()（）'},digit_encounters={c:chars[c] for c in '0123456789'},
        required_term_encounters={t:terms[t] for t in required},yolo_train=1561,yolo_holdout=156,teacher_replay_counts=dict(kd),
        time_budget='At most4000updates; stop training at100minutes elapsed including initial evaluation, then final evaluation/report/cleanup.',
        source_sha256=sha(START),interpretation='Multi-target adaptation; not a matched single-factor comparison withV45. Presentations,not full corpus passes.')
    write_json(work/'vocab.json',vocab);write_json(work/'batches.json',plan);sampling['plan_sha256']=sha(work/'batches.json');write_json(work/'sampling.json',sampling)
    return vocab,added,mix,plan,extra,sampling
