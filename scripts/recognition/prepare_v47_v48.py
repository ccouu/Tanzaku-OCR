"""Tiny whole-string Latin fitting corpus and separate balanced bracket curriculum."""
from collections import Counter,defaultdict
from pathlib import Path
import math,random,string
from PIL import Image,ImageDraw,ImageFont
from fontTools.ttLib import TTFont

from scripts.recognition.train_v45_latin_head import rows,V44,V40,ROOT,V34,START
from scripts.recognition.train_v38_yolo25 import BucketCrops,INPUT,HOLDOUT
from scripts.recognition.prepare_v43 import normalize_dataset
from outputs.recognition.v36_yolo_fullpasses import Combined,load,dump_rows
from outputs.recognition.application_mixture_pilot import RenderedLines
from src.recognition.datasets import FontLines
from src.recognition.font_training import row_canvas
from src.recognition.long_line_inference import bucket_image
from src.recognition.vocabulary import extend_vocabulary
from src.recognition.utils import write_json,sha

SEED=20261147


def render(root,key,text,fontpath,size,left=8,right=8):
    font=ImageFont.truetype(fontpath,size);box=font.getbbox(text,anchor='ls');baseline=48
    assert baseline+box[1]>=1 and baseline+box[3]<64,(text,fontpath,size,box)
    x=left-min(0,box[0]);width=math.ceil(max(font.getlength(text)+x,box[2]+x)+right)
    im=Image.new('RGB',(width,64),(242,242,242))
    ImageDraw.Draw(im).text((x,baseline),text,font=font,anchor='ls',fill=(20,20,20))
    im,meta=bucket_image(im,512);assert meta['scaled_height']==64
    out=root/'images'/(key+'.png');out.parent.mkdir(parents=True,exist_ok=True)
    if not out.exists():im.save(out,compress_level=1)
    return dict(text=text,image=out.relative_to(root).as_posix(),sha256=sha(out),preferred_font=Path(fontpath).name,
                font_path=fontpath,orientation_flag=0,**meta)


def prepare(work,source,arm):
    cfg=source['config'];manifest=load(V44/'targeted/manifest.json')
    targets=work/'targets';targets.mkdir(exist_ok=True)
    if arm=='latin':
        vocab,added=extend_vocabulary(source['vocab'],string.ascii_letters);assert len(added)==33
    else:vocab=source['vocab'];added=[]
    mapping=vocab['char_to_id'];records=[]
    if arm=='latin':
        fitfont=next(f for f in manifest['training_fonts'] if 'NotoSansJP' in f)
        for path in [fitfont,*manifest['heldout_fonts']]:
            with TTFont(path,fontNumber=0) as f:
                assert all(ord(c) in f.getBestCmap() for c in string.ascii_letters+' '),(path,'MissingLatinGlyph')
        words=['MENU','RAMEN','BEER','SAKE','COFFEE','TAX','ICE','HOT','SPICY','LUNCH','DINNER','Set','Menu','Beer','Hot','Ice','Coffee','Whisky','Juice','Pizza']
        for ci,c in enumerate(string.ascii_letters):
            word=next((w for w in words if c in w),'Menu '+c)
            for j,text in enumerate((c,c*2,c*3,word)):
                r=render(targets,f'train_{ci:02d}_{j}',text,fitfont,48);assert r['canvas'] in (128,256)
                records.append(dict(r,anchor=c,kind=('single','repeat2','repeat3','word')[j],split='train',cohort='fit_font'))
            for j,text in enumerate((c,c*2,c+'e','Ice '+c)):
                r=render(targets,f'same_{ci:02d}_{j}',text,fitfont,44,left=12,right=12)
                records.append(dict(r,anchor=c,kind=('single','repeat2','pair','word')[j],split='probe',cohort='samefont_new_layout'))
                font=manifest['heldout_fonts'][j]
                r=render(targets,f'held_{ci:02d}_{j}',text,font,44,left=12,right=12)
                records.append(dict(r,anchor=c,kind=('single','repeat2','pair','word')[j],split='probe',cohort='heldout_fonts'))
    else:
        terms=['税込','大','塩・タレ','数量限定','本日入荷','冷','温','中','小','並','セット','持帰り']
        cmaps={}
        for path in manifest['training_fonts']+manifest['heldout_fonts']:
            with TTFont(path,fontNumber=0) as f:cmaps[path]=set(f.getBestCmap())
        for split,count,cohorts in [('train',100,['known']),('probe',40,['known','heldout'])]:
            for style,(opening,closing) in enumerate([('(',')'),('（','）')]):
                for position in ('start','middle','after_space'):
                    for j in range(count):
                        term=terms[j%len(terms)];text=opening+term+closing
                        if position=='middle':text='定食'+text
                        if position=='after_space':text='定食 '+text
                        assert all(c in mapping for c in text)
                        cohort=cohorts[j%len(cohorts)];pool=manifest['heldout_fonts'] if cohort=='heldout' else manifest['training_fonts']
                        supported=[p for p in pool if all(ord(c) in cmaps[p] for c in text)]
                        path=supported[(j+style)%len(supported)]
                        size=(32,34,36)[j%3] if split=='train' else (31,33,35)[j%3]
                        r=render(targets,f'{split}_{style}_{position}_{j:03d}',text,path,size,left=8 if split=='train' else 12,right=8)
                        records.append(dict(r,style='ascii' if style==0 else 'fullwidth',position=position,split=split,cohort=cohort))
    train=[r for r in records if r['split']=='train'];probe=[r for r in records if r['split']=='probe']
    assert not {r['sha256'] for r in train}&{r['sha256'] for r in probe}
    dump_rows(targets/'labels.jsonl',records);write_json(work/'vocab.json',vocab)
    target=BucketCrops(train,targets,cfg,mapping);extra={'target_train':target}
    for cohort in sorted({r['cohort'] for r in probe}):extra['target_'+cohort]=BucketCrops([r for r in probe if r['cohort']==cohort],targets,cfg,mapping)
    yolo=rows(V40/'yolo_labels.jsonl');yt=[r for r in yolo if r['split']=='train'];yh=[r for r in yolo if r['split']=='val_holdout']
    assert len(yt)==1561 and len(yh)==156 and not {r['source_menu'] for r in yt}&HOLDOUT
    mix=Combined(FontLines(ROOT,'train',cfg['mean'],cfg['std']),RenderedLines(rows(V34/'corpus/labels.jsonl'),V34/'corpus',cfg,mapping),BucketCrops(yt,INPUT,cfg,mapping))
    mix.datasets['target']=target
    extra.update(yolo_current_train=mix.datasets['yolo'],yolo_current_holdout=BucketCrops(yh,INPUT,cfg,mapping))
    digitrows=rows(START.parent/'digits/labels.jsonl')
    for cohort in ('known','heldout'):extra['digits_'+cohort]=BucketCrops([r for r in digitrows if r['split']=='probe' and r['cohort']==cohort],START.parent/'digits',cfg,mapping)
    for cohort in ('known','heldout'):
        old=[r for r in rows(V44/'targeted/labels.jsonl') if r['kind']=='parentheses' and r['split']=='probe' and r['cohort']==cohort]
        extra['parentheses_'+cohort]=BucketCrops(old,V44/'targeted',cfg,mapping)
    for ds in list(mix.datasets.values())+list(extra.values()):
        ds.mapping=mapping;normalize_dataset(ds)
    rng=random.Random(SEED+(arm=='parentheses'));plan=[]
    def width(k,i):return row_canvas(mix.datasets[k].rows[i]) if k=='original' else mix.datasets[k].rows[i]['canvas']
    def schedule(k,pool,n,mandatory=()):
        groups=defaultdict(list);must=defaultdict(list)
        for i in pool:groups[width(k,i)].append(i)
        for i in mandatory:must[width(k,i)].append(i)
        order=[w for w,ids in sorted(must.items()) for _ in range(math.ceil(len(ids)/32))]
        assert len(order)<=n
        ws=sorted(groups);order+=rng.choices(ws,weights=[len(groups[w]) for w in ws],k=n-len(order))
        for w in order:
            ids=must[w][:32];del must[w][:len(ids)];ids+=rng.choices(groups[w],k=32-len(ids));rng.shuffle(ids);plan.append([(k,i) for i in ids])
        assert not any(must.values())
    if arm=='latin':
        # Repeated tiny-set presentations: ensure every anchor occurs in every
        # 100-update block rather than trusting random draws to cover the tail.
        for block in range(10):schedule('target',range(len(train)),100,range(len(train)))
    else:
        # Exact50/50 style quota across homogeneous-width batches. The wider
        # full-width glyphs need not share each ASCII sample's physical bucket.
        for block in range(2):
            start=len(plan)
            for style in ('ascii','fullwidth'):
                pool=[i for i,r in enumerate(train) if r['style']==style]
                schedule('target',pool,150,pool)
            schedule('yolo',range(len(yt)),250,range(len(yt)))
            schedule('original',range(len(mix.datasets['original'])),225)
            for view,n in [('clean_control',113 if block==0 else 112),('texture_color_noise',112 if block==0 else 113)]:
                schedule('generated',[i for i,r in enumerate(mix.datasets['generated'].rows) if r['view']==view],n)
            segment=plan[start:];rng.shuffle(segment);plan[start:]=segment
            assert len(segment)==1000
    counts=Counter();characters=Counter();unique=defaultdict(set);styles=Counter();positions=Counter()
    for batch in plan:
        assert len(batch)==32 and len({width(k,i) for k,i in batch})==1
        for k,i in batch:
            r=mix.datasets[k].rows[i];counts[k]+=1;characters.update(r['text']);unique[k].add(i)
            if k=='target' and arm=='parentheses':styles[r['style']]+=1;positions[r['position']]+=1
    assert len(plan)==(1000 if arm=='latin' else 2000)
    if arm=='parentheses':assert counts==dict(target=19200,yolo=16000,original=14400,generated=14400) and styles==dict(ascii=9600,fullwidth=9600)
    sampling=dict(arm=arm,updates=len(plan),presentations=len(plan)*32,streams=dict(counts),unique_records={k:len(v) for k,v in unique.items()},
        character_encounters=dict(characters),target_styles=dict(styles),target_positions=dict(positions),target_train=len(train),target_probe=len(probe),
        added_letters=added,source_sha256=sha(START),description='Whole-string clean rendering; no noise or opticalaugmentation. Latin is a fitting diagnostic without retention protection, not a production candidate.' if arm=='latin' else 'Independent V43 bracket arm; cleanbalancedASCII/fullwidth start/middle/afterspace; 30target/45synthetic/25YOLO.')
    write_json(work/'batches.json',plan);sampling['plan_sha256']=sha(work/'batches.json');write_json(work/'sampling.json',sampling)
    return vocab,added,mix,plan,extra,sampling
