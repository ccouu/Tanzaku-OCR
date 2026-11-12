"""V43 digit/price adaptation, with immutable source labels and normalized copies."""
from collections import Counter, defaultdict
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from fontTools.ttLib import TTFont

from scripts.recognition import prepare_v42 as prior
from scripts.recognition.train_v38_yolo25 import BucketCrops
from outputs.recognition.application_pilot import glyph_mask, appearance
from outputs.recognition.v36_yolo_fullpasses import load, dump_rows
from src.detection.line_crops import recognition_input
from src.recognition.label_normalization import normalize_digits
from src.recognition.metrics import align_errors, summarise
from src.recognition.font_training import row_canvas
from src.recognition.utils import sha, write_json

V42=prior.WORK
WORK=Path('outputs/recognition/pilots/v43_digits_prices_2000')
START=V42/'final_raw.pt'
SEED=20261123


def normalize_dataset(ds):
    """Normalize run-local row objects, including the native-crop lookup table."""
    changes=[]
    for attr in ('rows','base'):
        for i,r in enumerate(getattr(ds,attr,[])):
            if 'text' not in r:continue
            original=r['text'];text=normalize_digits(original)
            if original!=text:
                changes.append(dict(index=i,attribute=attr,original=original,normalized=text))
                r.setdefault('original_text',original);r['text']=text
    return changes


def digit_metrics(validation):
    result={}
    for name in ('digits_known','digits_heldout','yolo_current_train','yolo_current_holdout'):
        a=validation[name]['0.0'];stats={c:Counter() for c in '0123456789'};conf=Counter()
        for r in a['predictions']:
            for op in align_errors(r['truth'],r['prediction']):
                c=op['truth'];h=op['prediction']
                if c in stats:
                    stats[c]['occurrences']+=1;stats[c][op['type']]+=1
                    if op['type']!='match':conf[c,h or '<deleted>']+=1
                elif op['type']=='insertion' and h in stats:stats[h]['insertion']+=1
        per={c:dict(s,recall=s['match']/s['occurrences'] if s['occurrences'] else None) for c,s in stats.items()}
        recalls=[s['recall'] for s in per.values() if s['recall'] is not None]
        result[name]=dict(cer=a['all']['cer'],per_digit=per,macro_recall=sum(recalls)/len(recalls) if recalls else None,
                          confusions=[[c,h,n] for (c,h),n in conf.most_common(30)])
        if name.startswith('digits_'):
            for key in ('kind','fullwidth_rendered'):
                values=sorted({r[key] for r in a['predictions']})
                result[name]['by_'+key]={str(v):summarise([r for r in a['predictions'] if r[key]==v]) for v in values}
    return result


def render_digits(work,cfg,mapping):
    root=work/'digits';root.mkdir(exist_ok=True)
    paths=load(prior.ROOT/'manifest.json')['paths'];known=paths['old']+paths['new'];held=paths['heldout']
    cmaps={}
    for path in known+held:
        with TTFont(path,fontNumber=0) as f:cmaps[path]=set(f.getBestCmap())
    records=[]
    for split,count in (('train',600),('probe',100)):
        for digit in '0123456789':
            for i in range(count):
                seed=SEED+int(digit)*100000+i+(2000000 if split=='probe' else 0)
                rng=random.Random(seed);nrng=np.random.default_rng(seed)
                cohort='heldout' if split=='probe' and i>=50 else 'known';fonts=held if cohort=='heldout' else known
                path=fonts[(i//6+int(digit))%len(fonts)];kind=('single','repeat2','repeat3','number','yen','price')[i%6]
                if kind=='single':text=digit
                elif kind=='repeat2':text=digit*2
                elif kind=='repeat3':text=digit*3
                elif kind=='number':text=digit+str((int(digit)+3)%10)
                else:
                    # Realistic prices, including repeated zeros; equal digit anchors, not equal token frequencies.
                    amount=('100' if digit=='0' else digit+rng.choice(('00','50','80','800')))
                    text=('¥'+amount) if kind=='yen' else amount+'円'
                requested=text.translate(str.maketrans('0123456789','０１２３４５６７８９')) if i%2 else text
                rendered=requested if all(ord(c) in cmaps[path] for c in requested) else text
                assert normalize_digits(rendered)==text and all(c in mapping for c in text)
                assert all(ord(c) in cmaps[path] for c in rendered),(path,rendered)
                out=root/'images'/split/f'{digit}_{i:04d}.png';out.parent.mkdir(parents=True,exist_ok=True)
                if not out.exists():
                    mask=glyph_mask(rendered,Path(path),rng.randint(30,96),rng)
                    # Reduce perfect alignment while retaining safe margins and every glyph pixel.
                    mask=ImageOps.expand(mask,border=(rng.randint(0,18),rng.randint(0,4),rng.randint(0,18),rng.randint(0,4)),fill=0)
                    if i%5==0:
                        image,_=appearance(mask,('paper','wood','chalkboard','cloth')[i%4],'high',rng,nrng)
                        image=image.filter(ImageFilter.GaussianBlur(rng.uniform(.1,.3)))
                    else:
                        bg=rng.randint(225,250);ink=rng.randint(10,45)
                        if i%4==0:bg,ink=ink,bg
                        image=Image.fromarray(np.rint(bg+(ink-bg)*np.asarray(mask,dtype=float)/255).clip(0,255).astype('uint8')).convert('RGB')
                    normalized,meta=recognition_input(image);normalized.save(out,compress_level=1);write_json(out.with_suffix('.json'),meta)
                meta=load(out.with_suffix('.json'))
                records.append(dict(image=str(out.relative_to(root)),sha256=sha(out),text=text,original_text=rendered,
                    split=split,target_digit=digit,kind=kind,cohort=cohort,preferred_font=Path(path).name,font_path=path,
                    fullwidth_rendered=rendered!=text,orientation_flag=0,seed=seed,**meta))
    train=[r for r in records if r['split']=='train'];probe=[r for r in records if r['split']=='probe']
    assert len(train)==6000 and len(probe)==1000 and not {r['sha256'] for r in train}&{r['sha256'] for r in probe}
    dump_rows(root/'labels.jsonl',records)
    write_json(root/'manifest.json',dict(train=6000,probe=1000,training_fonts=known,heldout_fonts=held,
        normalization='Only U+FF10..U+FF19 -> ASCII 0..9; original rendered text retained.',
        training_anchors=dict(Counter(r['target_digit'] for r in train)),font_sha256={p:sha(Path(p)) for p in known+held}))
    return BucketCrops(train,root,cfg,mapping),{ 'digits_'+c:BucketCrops([r for r in probe if r['cohort']==c],root,cfg,mapping) for c in ('known','heldout')}


def prepare(work,source):
    # Reuse audited source geometry/holdout and augmentation generation in a private subdirectory.
    base=work/'replay';base.mkdir(exist_ok=True)
    vocab,added,mix,oldplan,extra,_=prior.prepare(base,source)
    cfg=source['config'];mapping=vocab['char_to_id'];write_json(work/'vocab.json',vocab)
    digits,probes=render_digits(work,cfg,mapping);mix.datasets['digits']=digits;extra.update(probes)
    normalization={}
    for k,ds in mix.datasets.items():
        normalization[k]=normalize_dataset(ds);ds.mapping=mapping
        dump_rows(work/f'normalized_{k}_labels.jsonl',ds.rows)
    for k,ds in extra.items():normalize_dataset(ds)
    rng=random.Random(SEED);plan=[]
    def width(k,i):
        r=mix.datasets[k].rows[i];return row_canvas(r) if k=='original' else r['canvas']
    def ordinary(k,count,mandatory=()):
        groups=defaultdict(list);must=defaultdict(list)
        for i in range(len(mix.datasets[k].rows)):groups[width(k,i)].append(i)
        for i in sorted(mandatory):must[width(k,i)].append(i)
        schedule=[w for w,ids in sorted(must.items()) for _ in range((len(ids)+31)//32)]
        assert len(schedule)<=count
        ws=sorted(groups);schedule+=rng.choices(ws,weights=[len(groups[w]) for w in ws],k=count-len(schedule))
        for w in schedule:
            ids=must[w][:32];del must[w][:len(ids)];ids+=rng.choices(groups[w],k=32-len(ids));rng.shuffle(ids)
            plan.append([(k,i) for i in ids])
        assert not any(must.values())
    first_view={}
    for i,v in enumerate(mix.datasets['yolo'].views):first_view.setdefault(v['base_index'],i)
    assert len(first_view)==1561
    ordinary('yolo',500,first_view.values())
    # Keep full coverage of the 521 audited terms, plus all 57 new-character target groups.
    required=load(prior.V34/'corpus/manifest.json')['required_terms'];mandatory_o=set();mandatory_g=set()
    for term in required:
        hit=next((i for i,r in enumerate(mix.datasets['original'].rows) if r['text']==term),None)
        if hit is not None:mandatory_o.add(hit)
        else:mandatory_g.add(next(i for i,r in enumerate(mix.datasets['generated'].rows) if r['text']==term))
    ordinary('original',150,mandatory_o);ordinary('generated',150,mandatory_g)
    ordinary('target',200,range(len(mix.datasets['target'].rows)))
    # 400*32 = 12,800 presentations, so select per length equally, with 75/25 clean/noisy.
    long_rows=mix.datasets['long512'].rows
    for n in (11,12,13,14):
        for view,count in (('clean',2400),('noisy',800)):
            pool=[i for i,r in enumerate(long_rows) if len(r['text'])==n and r['view']==view]
            ids=rng.choices(pool,k=count)
            plan.extend([[('long512',i) for i in ids[j:j+32]] for j in range(0,count,32)])
    # 600 homogeneous digit batches, exactly 60 batches per anchor digit, covering all 6000 renders.
    for c in '0123456789':
        mandatory=[i for i,r in enumerate(digits.rows) if r['target_digit']==c]
        groups=defaultdict(list)
        for i in mandatory:groups[width('digits',i)].append(i)
        schedule=[w for w,ids in sorted(groups.items()) for _ in range((len(ids)+31)//32)]
        ws=sorted(groups);schedule+=rng.choices(ws,weights=[len(groups[w]) for w in ws],k=60-len(schedule))
        queues={w:rng.sample(ids,len(ids)) for w,ids in groups.items()}
        for w in schedule:
            ids=queues[w][:32];del queues[w][:len(ids)];ids+=rng.choices(groups[w],k=32-len(ids))
            plan.append([('digits',i) for i in ids])
        assert not any(queues.values())
    rng.shuffle(plan);streams=Counter();canvases=Counter();chars=Counter();terms=Counter();unique=defaultdict(set);digit_counts=Counter()
    for batch in plan:
        assert len(batch)==32 and len({width(k,i) for k,i in batch})==1
        for k,i in batch:
            r=mix.datasets[k].rows[i];assert normalize_digits(r['text'])==r['text']
            streams[k]+=1;canvases[width(k,i)]+=1;chars.update(c for c in r['text'] if c in added);terms[r['text']]+=1;unique[k].add(i)
            if k=='digits':digit_counts[r['target_digit']]+=1
    assert len(plan)==2000 and streams==dict(digits=19200,yolo=16000,long512=12800,target=6400,original=4800,generated=4800),(len(plan),dict(streams))
    yolo_bases={mix.datasets['yolo'].views[i]['base_index'] for i in unique['yolo']}
    assert len(unique['digits'])==6000 and len(yolo_bases)==1561 and all(terms[t] for t in required)
    assert min(chars[c] for c in added)>=50 and all(n==1920 for n in digit_counts.values())
    write_json(work/'batches.json',plan)
    sampling=dict(updates=2000,presentations=64000,streams=dict(streams),physical_width_presentations=dict(canvases),
        actual_512_fraction=canvases[512]/64000,planned_new_character_counts={c:chars[c] for c in added},
        digit_anchor_counts=dict(digit_counts),unique_records={k:len(v) for k,v in unique.items()},unique_yolo_base_crops=len(yolo_bases),
        required_term_encounters={t:terms[t] for t in required},normalization_changes=normalization,
        label_normalization='Full-width decimal digits only -> ASCII; source files untouched.',
        holdout_menus=sorted(prior.HOLDOUT),plan_sha256=sha(work/'batches.json'),source_checkpoint_sha256=sha(START))
    write_json(work/'sampling.json',sampling)
    return vocab,added,mix,plan,extra,sampling
