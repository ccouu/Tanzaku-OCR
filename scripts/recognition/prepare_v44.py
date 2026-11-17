"""V44: complete ASCII Latin coverage, parentheses and extra-long lines."""
from collections import Counter,defaultdict
import copy
import math
from pathlib import Path
import random
import string
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from fontTools.ttLib import TTFont
import torch

from scripts.recognition import prepare_v42 as base
from scripts.recognition.prepare_v43 import normalize_dataset,digit_metrics
from scripts.recognition.train_v38_yolo25 import BucketCrops
from outputs.recognition.v36_yolo_fullpasses import load,dump_rows
from outputs.recognition.application_pilot import appearance
from src.recognition.vocabulary import extend_vocabulary
from src.recognition.long_line_inference import bucket_image,native_strip,predict_long
from src.recognition.label_normalization import normalize_digits
from src.recognition.font_training import row_canvas
from src.recognition.metrics import align_errors,summarise
from src.recognition.model import MenuRecognizer
from src.recognition.utils import sha,write_json

V43=Path('outputs/recognition/pilots/v43_digits_prices_2000')
WORK=Path('outputs/recognition/pilots/v44_latin_parentheses_long_2000')
START=V43/'final_raw.pt';SEED=20261124


def adaptation_metrics(validation):
    metrics={}
    for kind,characters in (('latin',string.ascii_letters),('parentheses','()（）')):
        metrics[kind]={}
        for cohort in ('known','heldout'):
            rr=validation[kind+'_'+cohort]['0.0']['predictions'];refs=Counter();hits=Counter();errors=Counter()
            for row in rr:
                refs.update(c for c in row['truth'] if c in characters)
                for op in align_errors(row['truth'],row['prediction']):
                    if op['truth'] in characters:
                        if op['type']=='match':hits[op['truth']]+=1
                        else:errors[op['type']]+=1
            per={c:dict(reference=refs[c],correct=hits[c],recall=hits[c]/refs[c] if refs[c] else None) for c in characters}
            values=[r['recall'] for r in per.values() if r['recall'] is not None]
            metrics[kind][cohort]=dict(per_character=per,macro_recall=sum(values)/len(values),error_counts=dict(errors))
    return metrics


def evaluate_windows(model,ds,cfg,mapping,device,precision):
    predictions=[]
    for i,r in enumerate(ds.rows):
        with Image.open(ds.root/r['image']) as im:image=im.crop((0,0,r['valid_width'],64)).convert('RGB')
        pred=predict_long(model,image,cfg,mapping,device,precision,mode='overlap512')
        err=Counter(o['type'] for o in align_errors(r['text'],pred) if o['type']!='match')
        predictions.append(dict(index=i,truth=r['text'],prediction=pred,font=r['preferred_font'],cohort=r['cohort'],error_counts={k:err[k] for k in ('deletion','insertion','substitution')}))
    return {'0.0':dict(all=summarise(predictions),predictions=predictions)}


def expanded_state(source,vocab):
    state=copy.deepcopy(source['model']);old=len(source['vocab']['char_to_id'])+1
    n=len(vocab['char_to_id'])+1-old
    state['classifier.weight']=torch.cat((state['classifier.weight'],state['classifier.weight'][:1].expand(n,-1,-1).clone()))
    state['classifier.bias']=torch.cat((state['classifier.bias'],state['classifier.bias'][:1].expand(n).clone()-8))
    assert all(vocab['char_to_id'][c]==i for c,i in source['vocab']['char_to_id'].items())
    return state


def mask_with_boxes(text,path,size,rng):
    """Render on one baseline, record every glyph for safe synthetic window labels."""
    font=ImageFont.truetype(path,size);boxes=[font.getbbox(c,anchor='ls') for c in text]
    gap=rng.uniform(0,.08)*size;left=rng.randint(3,18);right=rng.randint(3,18)
    top=rng.randint(3,max(4,size//3));bottom=rng.randint(3,max(4,size//3))
    upper=min(b[1] for b in boxes);lower=max(b[3] for b in boxes);baseline=top-upper
    origins=[];x=left-min(0,boxes[0][0])
    for c,b in zip(text,boxes):origins.append(x);x+=font.getlength(c)+gap
    w=math.ceil(max(x,origins[-1]+boxes[-1][2])+right)
    image=Image.new('L',(w,top+lower-upper+bottom),0);draw=ImageDraw.Draw(image)
    positioned=[]
    for c,x,b in zip(text,origins,boxes):
        draw.text((x,baseline),c,font=font,anchor='ls',fill=255)
        positioned.append([x+b[0],baseline+b[1],x+b[2],baseline+b[3]])
    return image,positioned


def make_corpus(work,mapping):
    root=work/'targeted';root.mkdir(exist_ok=True)
    roster=load(base.ROOT/'manifest.json')['paths'];known=roster['old']+roster['new'];held=roster['heldout'];cmaps={}
    for path in known+held:
        with TTFont(path,fontNumber=0) as f:cmaps[path]=set(f.getBestCmap())
    terms=load(base.V34/'corpus/manifest.json')['required_terms']
    terms=[t for t in terms if all(c in mapping for c in t) and not any(c in string.ascii_letters for c in t)]
    specs=[]
    words=['MENU','RAMEN','BEER','SAKE','COFFEE','TAX','ICE','HOT','SPICY','LUNCH','DINNER','Set','Menu','Beer','Hot','Ice','Takeout','Coffee','Whisky','Juice','Pizza','S size','M size','L size']
    for split,count in (('train',40),('probe',16)):
        for ci,c in enumerate(string.ascii_letters):
            for i in range(count):
                rng=random.Random(SEED+ci*10000+i+(1000000 if split=='probe' else 0))
                choices=[w for w in words if c in w]
                text=(c if i%4==0 else c*2 if i%4==1 else rng.choice(choices) if choices else 'Menu '+c)
                specs.append(dict(kind='latin',text=text,anchor=c,split=split,cohort='heldout' if split=='probe' and i>=8 else 'known',seed=rng.randrange(2**32)))
    paren_texts=['定食（税込）','ビール（中）','ランチ(数量限定)','刺身（本日入荷）','ご飯(大盛)','酒（冷）','お茶（温）','セット(980円)','(限定)','（小）','(大)','（並）']
    paren_texts=[t for t in paren_texts if all(c in mapping for c in t)]
    assert len(paren_texts)>=8
    for split,count in (('train',640),('probe',160)):
        for i in range(count):
            specs.append(dict(kind='parentheses',text=paren_texts[i%len(paren_texts)],anchor='',split=split,
                cohort='heldout' if split=='probe' and i>=80 else 'known',seed=SEED+3000000+i+(100000 if split=='probe' else 0)))
    for split,count in (('train',180),('probe',40)):
        for n in (16,20,24):
            for i in range(count):
                rng=random.Random(SEED+n*10000+i+(4000000 if split=='probe' else 5000000));text=''
                while len(text)<n:text+=('・' if text else '')+rng.choice(terms)
                specs.append(dict(kind='longwide',text=text[:n],anchor='',split=split,
                    cohort='heldout' if split=='probe' and i>=20 else 'known',seed=rng.randrange(2**32)))
    records=[];windows=[];used=Counter()
    for j,s in enumerate(specs):
        rng=random.Random(s['seed']);nrng=np.random.default_rng(s['seed']);text=normalize_digits(s['text'])
        supported=[p for p in (held if s['cohort']=='heldout' else known) if all(ord(c) in cmaps[p] for c in text)]
        assert supported,('Unsupported rendered text',text,s['cohort'])
        path=supported[j%len(supported)];out=root/'images'/s['kind']/s['split']/f'{j:05d}.png';out.parent.mkdir(parents=True,exist_ok=True)
        if not out.exists():
            mask,boxes=mask_with_boxes(text,path,rng.randint(32,90),rng)
            # Extra-long synthetic probes match H64 widths513..768; record added background explicitly.
            added_vertical=0
            if s['kind']=='longwide':
                wanted=rng.randint(590,760)
                if mask.width*64/mask.height>wanted:
                    added_vertical=max(0,math.ceil(mask.width*64/wanted)-mask.height)
                    mask=Image.fromarray(np.pad(np.asarray(mask),((added_vertical//2,added_vertical-added_vertical//2),(0,0)),constant_values=0))
                # Shorter natural cases get background width only; glyphs never stretch or clip.
                if mask.width*64/mask.height<=512:
                    pad=math.ceil(560*mask.height/64)-mask.width
                    mask=Image.fromarray(np.pad(np.asarray(mask),((0,0),(0,pad)),constant_values=0))
            if j%5==0:
                image,_=appearance(mask,('paper','wood','chalkboard','cloth')[j%4],'high',rng,nrng)
                image=image.filter(ImageFilter.GaussianBlur(rng.uniform(.1,.3)))
            else:
                bg,ink=(rng.randint(225,250),rng.randint(10,40))
                if j%4==0:bg,ink=ink,bg
                image=Image.fromarray(np.rint(bg+(ink-bg)*np.asarray(mask,dtype=float)/255).clip(0,255).astype('uint8')).convert('RGB')
            normalized,meta=bucket_image(image,768 if s['kind']=='longwide' else 512)
            normalized.save(out,compress_level=1)
            scale=64/image.height
            meta.update(glyph_boxes_x=[[b[0]*scale,b[2]*scale] for b in boxes],added_vertical_background=added_vertical)
            write_json(out.with_suffix('.json'),meta)
        meta=load(out.with_suffix('.json'))
        r=dict(s,image=str(out.relative_to(root)),sha256=sha(out),text=text,original_text=s['text'],preferred_font=Path(path).name,font_path=path,orientation_flag=0,**meta)
        if s['kind']=='longwide':assert r['canvas']==768 and 512<r['valid_width']<=768 and r['scaled_height']==64
        records.append(r);used[path]+=1
        if s['kind']=='longwide' and s['split']=='train':
            # Ground-truth segment labels from rendered glyph boxes; never infer labels from OCR.
            mid=len(text)//2
            for wi,(a,b) in enumerate(((0,mid+2),(mid-2,len(text)))):
                left=0 if a==0 else math.floor((meta['glyph_boxes_x'][a-1][1]+meta['glyph_boxes_x'][a][0])/2)
                right=r['valid_width'] if b==len(text) else math.ceil((meta['glyph_boxes_x'][b-1][1]+meta['glyph_boxes_x'][b][0])/2)
                assert 0<=left<right<=r['valid_width'] and right-left<=512
                part=root/'windows'/f'{j:05d}_{wi}.png';part.parent.mkdir(exist_ok=True)
                if not part.exists():
                    with Image.open(out) as im:cropped=im.crop((left,0,right,64))
                    fitted,wmeta=bucket_image(cropped,512);fitted.save(part,compress_level=1);write_json(part.with_suffix('.json'),wmeta)
                windows.append(dict(r,image=str(part.relative_to(root)),sha256=sha(part),text=text[a:b],segment=[a,b],**load(part.with_suffix('.json'))))
    train=[r for r in records if r['split']=='train'];probe=[r for r in records if r['split']=='probe']
    assert not {r['sha256'] for r in train+windows}&{r['sha256'] for r in probe}
    dump_rows(root/'labels.jsonl',records);dump_rows(root/'window_labels.jsonl',windows)
    write_json(root/'manifest.json',dict(train=len(train),probe=len(probe),kinds=dict(Counter(r['kind'] for r in records)),
        training_fonts=known,heldout_fonts=held,font_sha256={p:sha(Path(p)) for p in known+held},used_fonts=dict(used),
        long_probe_limitation='Known templates/characters; generated long strips include audited vertical background additions to match width513..768, not independent real-menu proof.'))
    return root,train,probe,windows


def geometry_check(work,source,root,probe,extra):
    path=work/'geometry_preflight.json'
    if path.exists():
        saved=load(path);assert saved['source_sha256']==sha(START);return saved['chosen_training_geometry']
    model=MenuRecognizer(len(source['vocab']['char_to_id'])+1,use_bilstm=True).cuda().eval();model.load_state_dict(source['model'])
    mapping=source['vocab']['char_to_id'];cfg=source['config'];tests=[]
    for r in probe:
        if r['kind']=='longwide':tests.append((root/r['image'],r['text'],r['cohort'],0))
    # Actual menu-disjoint holdout native crops; no holdout pixels enter training.
    for r in extra['yolo_current_holdout'].rows:
        native=base.INPUT/r['display_crop']
        from PIL import Image as PILImage
        with PILImage.open(native) as im:
            w,h=im.size if not r['orientation_flag'] else im.size[::-1]
        if len(r['text'])>=16 and w*64/h>512:tests.append((native,r['text'],'real_holdout',r['orientation_flag']))
    outcomes={mode:[] for mode in ('fit512','wide768','overlap512')}
    for image,text,cohort,orientation in tests:
        with Image.open(image) as im:strip=im.convert('RGB')
        # Synthetic probes are padded; use their recorded valid width, not physical768.
        synthetic=next((r for r in probe if root/r['image']==image),None)
        if synthetic:strip=strip.crop((0,0,synthetic['valid_width'],64))
        if orientation:strip=strip.transpose(Image.Transpose.ROTATE_90)
        for mode in outcomes:
            prediction=predict_long(model,strip,cfg,mapping,'cuda:0',mode=mode,orientation=orientation)
            errors=Counter(o['type'] for o in align_errors(text,prediction) if o['type']!='match')
            outcomes[mode].append(dict(truth=text,prediction=prediction,cohort=cohort,image=str(image),error_counts={k:errors[k] for k in ('deletion','insertion','substitution')}))
    summary={mode:{c:summarise([r for r in rr if r['cohort']==c]) for c in sorted({r['cohort'] for r in rr})} for mode,rr in outcomes.items()}
    # Prefer one-pass768 unless overlapping windows improve heldout-font CER by >1pp.
    choice='overlap512' if summary['overlap512']['heldout']['cer']+.01<summary['wide768']['heldout']['cer'] else 'wide768'
    write_json(path,dict(source_sha256=sha(START),precision='fp16',overlap_h64_pixels=100,summary=summary,predictions=outcomes,
        chosen_training_geometry=choice,selection='Heldout-font synthetic CER; require >1pp overlap advantage, otherwise prefer one-pass768. Real holdout is reported separately, not used for geometry choice.'))
    del model;torch.cuda.empty_cache();return choice


def prepare(work,source):
    replay=work/'replay';replay.mkdir(exist_ok=True)
    oldvocab,oldadded,mix,_,extra,_=base.prepare(replay,source)
    vocab,latin_added=extend_vocabulary(oldvocab,string.ascii_letters)
    vocab['extension']['source_extension']=oldvocab['extension'];mapping=vocab['char_to_id'];cfg=source['config']
    assert set(string.ascii_letters)<=set(mapping) and len(latin_added)==33
    write_json(work/'vocab.json',vocab)
    root,training,probe,windows=make_corpus(work,mapping)
    geometry=geometry_check(work,source,root,probe,extra)
    for kind in ('latin','parentheses','longwide'):
        values=[r for r in training if r['kind']==kind]
        if kind=='longwide' and geometry=='overlap512':values=windows
        mix.datasets[kind]=BucketCrops(values,root,cfg,mapping)
        for cohort in ('known','heldout'):
            extra[kind+'_'+cohort]=BucketCrops([r for r in probe if r['kind']==kind and r['cohort']==cohort],root,cfg,mapping)
    drows=[__import__('json').loads(s) for s in (V43/'digits/labels.jsonl').read_text(encoding='utf-8').splitlines()]
    mix.datasets['digits']=BucketCrops([r for r in drows if r['split']=='train'],V43/'digits',cfg,mapping)
    for cohort in ('known','heldout'):extra['digits_'+cohort]=BucketCrops([r for r in drows if r['split']=='probe' and r['cohort']==cohort],V43/'digits',cfg,mapping)
    for k,ds in mix.datasets.items():
        normalize_dataset(ds);ds.mapping=mapping;dump_rows(work/f'normalized_{k}_labels.jsonl',ds.rows)
    for ds in extra.values():normalize_dataset(ds);ds.mapping=mapping
    plan=[];rng=random.Random(SEED)
    def width(k,i):
        r=mix.datasets[k].rows[i];return row_canvas(r) if k=='original' else r['canvas']
    def schedule(k,batches,mandatory=()):
        groups=defaultdict(list);must=defaultdict(list)
        for i,r in enumerate(mix.datasets[k].rows):groups[width(k,i)].append(i)
        for i in sorted(mandatory):must[width(k,i)].append(i)
        ws=sorted(groups);order=[w for w,ids in sorted(must.items()) for _ in range((len(ids)+31)//32)]
        assert len(order)<=batches,(k,len(order),batches)
        order+=rng.choices(ws,weights=[len(groups[w]) for w in ws],k=batches-len(order))
        for w in order:
            ids=must[w][:32];del must[w][:len(ids)];ids+=rng.choices(groups[w],k=32-len(ids));rng.shuffle(ids);plan.append([(k,i) for i in ids])
        assert not any(must.values())
    first={}
    for i,v in enumerate(mix.datasets['yolo'].views):first.setdefault(v['base_index'],i)
    schedule('yolo',600,first.values());schedule('latin',300,range(len(mix.datasets['latin'].rows)))
    schedule('parentheses',100,range(len(mix.datasets['parentheses'].rows)))
    schedule('longwide',300,range(len(mix.datasets['longwide'].rows)))
    schedule('digits',300,range(len(mix.datasets['digits'].rows)))
    schedule('target',100,range(len(mix.datasets['target'].rows)));schedule('long512',100)
    required=load(base.V34/'corpus/manifest.json')['required_terms'];mo=set();mg=set()
    for t in required:
        i=next((i for i,r in enumerate(mix.datasets['original'].rows) if r['text']==t),None)
        if i is not None:mo.add(i)
        else:mg.add(next(i for i,r in enumerate(mix.datasets['generated'].rows) if r['text']==t))
    schedule('original',100,mo);schedule('generated',100,mg);rng.shuffle(plan)
    streams=Counter();canvases=Counter();chars=Counter();terms=Counter();paren=0;ys=set();unique=defaultdict(set)
    for batch in plan:
        assert len(batch)==32 and len({width(k,i) for k,i in batch})==1
        for k,i in batch:
            r=mix.datasets[k].rows[i];t=r['text'];assert normalize_digits(t)==t and all(c in mapping for c in t)
            streams[k]+=1;canvases[width(k,i)]+=1;chars.update(t);terms[t]+=1;paren+=any(c in t for c in '()（）');unique[k].add(i)
            if k=='yolo':ys.add(mix.datasets[k].views[i]['base_index'])
    assert len(plan)==2000 and sum(streams.values())==64000 and streams['yolo']==19200
    assert len(ys)==1561 and all(terms[t] for t in required) and paren>=3200
    assert min(chars[c] for c in latin_added)>=40 and min(chars[c] for c in oldadded)>=25
    write_json(work/'batches.json',plan)
    sampling=dict(updates=2000,presentations=64000,streams=dict(streams),physical_width_presentations=dict(canvases),actual_512_fraction=canvases[512]/64000,
        source_characters=1779,total_characters=len(mapping),latin_added=latin_added,all_ascii_latin_encounters={c:chars[c] for c in string.ascii_letters},
        planned_new_character_counts={c:chars[c] for c in oldadded+latin_added},parentheses_presentations=paren,parentheses_fraction=paren/64000,
        dedicated_parentheses_presentations=streams['parentheses'],required_term_encounters={t:terms[t] for t in required},
        unique_yolo_base_crops=len(ys),unique_records={k:len(v) for k,v in unique.items()},geometry=geometry,source_checkpoint_sha256=sha(START),
        plan_sha256=sha(work/'batches.json'),holdout_menus=sorted(base.HOLDOUT),normalization='Full-width decimal digits -> ASCII only; source labels preserved.')
    write_json(work/'sampling.json',sampling)
    return vocab,oldadded,mix,plan,extra,sampling
