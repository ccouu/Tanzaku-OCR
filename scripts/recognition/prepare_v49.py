"""Head-only Latin/digit/bracket adaptation with frozen shared features."""
from collections import Counter,defaultdict
from pathlib import Path
import math,random,string
from scripts.recognition.prepare_v47_v48 import START,rows,V44,V40,ROOT,V34
from scripts.recognition.train_v38_yolo25 import BucketCrops,INPUT,HOLDOUT
from scripts.recognition.prepare_v43 import normalize_dataset
from outputs.recognition.v36_yolo_fullpasses import Combined,load
from outputs.recognition.application_mixture_pilot import RenderedLines
from src.recognition.datasets import FontLines
from src.recognition.font_training import row_canvas
from src.recognition.vocabulary import extend_vocabulary
from src.recognition.utils import sha,write_json

WORK=Path('outputs/recognition/pilots/v49_selective_head_2000')


def prepare(work,source,arm=None):
    vocab,added=extend_vocabulary(source['vocab'],string.ascii_letters);assert len(added)==33
    mapping=vocab['char_to_id'];cfg=source['config']
    v47=Path('outputs/recognition/pilots/v47_clean_latin_fit_1000/targets')
    v48=Path('outputs/recognition/pilots/v48_parentheses_2000/targets')
    tiny=rows(v47/'labels.jsonl');brackets=rows(v48/'labels.jsonl');multi=rows(V44/'targeted/labels.jsonl')
    yt=[r for r in rows(V40/'yolo_labels.jsonl') if r['split']=='train'];yh=[r for r in rows(V40/'yolo_labels.jsonl') if r['split']=='val_holdout']
    assert len(yt)==1561 and len(yh)==156 and not {r['source_menu'] for r in yt}&HOLDOUT
    assert {r['source_menu'] for r in yh}==HOLDOUT and not {r['sha256'] for r in yt}&{r['sha256'] for r in yh}
    original=FontLines(ROOT,'train',cfg['mean'],cfg['std']);generated=rows(V34/'corpus/labels.jsonl')
    mix=Combined(original,RenderedLines(generated,V34/'corpus',cfg,mapping),BucketCrops(yt,INPUT,cfg,mapping))
    mix.datasets['target']=BucketCrops([r for r in tiny if r['split']=='train'],v47,cfg,mapping)
    mix.datasets['latin_multi']=BucketCrops([r for r in multi if r['kind']=='latin' and r['split']=='train' and int(Path(r['image']).stem)%5!=0 and 1<=len(r['text'])<=6 and r['canvas'] in (128,256)],V44/'targeted',cfg,mapping)
    mix.datasets['brackets']=BucketCrops([r for r in brackets if r['split']=='train'],v48,cfg,mapping)
    digitrows=rows(START.parent/'digits/labels.jsonl')
    mix.datasets['digits']=BucketCrops([r for r in digitrows if r['split']=='train'],START.parent/'digits',cfg,mapping)
    extra={'target_train':mix.datasets['target'],'target_multifont_train':mix.datasets['latin_multi'],
        'yolo_current_train':mix.datasets['yolo'],'yolo_current_holdout':BucketCrops(yh,INPUT,cfg,mapping)}
    for cohort in ('samefont_new_layout','heldout_fonts'):extra['target_'+cohort]=BucketCrops([r for r in tiny if r['split']=='probe' and r['cohort']==cohort],v47,cfg,mapping)
    for cohort in ('known','heldout'):
        extra['target_multifont_'+cohort]=BucketCrops([r for r in multi if r['kind']=='latin' and r['split']=='probe' and r['cohort']==cohort],V44/'targeted',cfg,mapping)
        extra['parentheses_'+cohort]=BucketCrops([r for r in multi if r['kind']=='parentheses' and r['split']=='probe' and r['cohort']==cohort],V44/'targeted',cfg,mapping)
        extra['brackets_'+cohort]=BucketCrops([r for r in brackets if r['split']=='probe' and r['cohort']==cohort],v48,cfg,mapping)
        extra['digits_'+cohort]=BucketCrops([r for r in digitrows if r['split']=='probe' and r['cohort']==cohort],START.parent/'digits',cfg,mapping)
    for ds in list(mix.datasets.values())+list(extra.values()):ds.mapping=mapping;normalize_dataset(ds)
    required=load(V34/'corpus/manifest.json')['required_terms'];mo=set();mg=set()
    for term in required:
        i=next((i for i,r in enumerate(original.rows) if r['text']==term),None)
        if i is not None:mo.add(i)
        else:mg.add(next(i for i,r in enumerate(generated) if r['text']==term and r['view']=='clean_control'))
    rng=random.Random(20261149);plan=[]
    def width(k,i):return row_canvas(mix.datasets[k].rows[i]) if k=='original' else mix.datasets[k].rows[i]['canvas']
    def schedule(k,pool,n,mandatory=()):
        groups=defaultdict(list);must=defaultdict(list)
        for i in pool:groups[width(k,i)].append(i)
        for i in sorted(mandatory):must[width(k,i)].append(i)
        order=[w for w,ids in sorted(must.items()) for _ in range(math.ceil(len(ids)/32))];assert len(order)<=n,(k,n,len(order))
        ws=sorted(groups);order+=rng.choices(ws,weights=[len(groups[w]) for w in ws],k=n-len(order))
        for w in order:
            ids=must[w][:32];del must[w][:len(ids)];ids+=rng.choices(groups[w],k=32-len(ids));rng.shuffle(ids);plan.append([(k,i) for i in ids])
        assert not any(must.values())
    digit_ids=list(range(6000));rng.shuffle(digit_ids)
    for block in range(2):
        start=len(plan)
        for k,n in [('target',150),('latin_multi',150),('digits',200),('yolo',200),('original',100)]:
            mandatory=range(len(mix.datasets[k])) if k in ('target','latin_multi','yolo') else mo if k=='original' else digit_ids[block*3000:(block+1)*3000]
            schedule(k,range(len(mix.datasets[k])),n,mandatory)
        for style in ('ascii','fullwidth'):
            pool=[i for i,r in enumerate(mix.datasets['brackets'].rows) if r['style']==style];schedule('brackets',pool,50,pool)
        for view in ('clean_control','texture_color_noise'):
            pool=[i for i,r in enumerate(generated) if r['view']==view];schedule('generated',pool,50,mg if view=='clean_control' else ())
        segment=plan[start:];rng.shuffle(segment);plan[start:]=segment;assert len(segment)==1000
    streams=Counter();chars=Counter();terms=Counter();styles=Counter();seen=defaultdict(set);widths=Counter();teacher=Counter()
    for batch in plan:
        assert len(batch)==32 and len({width(k,i) for k,i in batch})==1
        for k,i in batch:
            row=mix.datasets[k].rows[i];t=row['text'];streams[k]+=1;chars.update(t);terms[t]+=1;seen[k].add(i);widths[width(k,i)]+=1
            assert all(c in mapping for c in t) and not any('\uff10'<=c<='\uff19' for c in t)
            if k=='brackets':styles[row['style']]+=1
            if k in ('original','generated','yolo'):teacher['excluded_latin_label' if any(c in string.ascii_letters for c in t) else 'eligible']+=1
    assert streams==dict(target=9600,latin_multi=9600,digits=12800,brackets=6400,yolo=12800,original=6400,generated=6400)
    assert styles==dict(ascii=3200,fullwidth=3200) and len(seen['digits'])==6000 and len(seen['yolo'])==1561
    assert all(terms[t]>=2 for t in required)
    sampling=dict(updates=2000,presentations=64000,streams=dict(streams),unique_records={k:len(v) for k,v in seen.items()},
        character_encounters=dict(chars),parentheses_styles=dict(styles),teacher_replay=dict(teacher),physical_width_presentations=dict(widths),
        required_term_encounters={t:terms[t] for t in required},added_letters=added,source_sha256=sha(START),
        recipe='30%Latin(15%tiny/15%multi-font),20%digit,10%brackets,20%YOLO,20%synthetic(10original/5appclean/5appnoisy). Allsharedfeaturesfrozen.')
    write_json(work/'vocab.json',vocab);write_json(work/'batches.json',plan);sampling['plan_sha256']=sha(work/'batches.json');write_json(work/'sampling.json',sampling)
    return vocab,added,mix,plan,extra,sampling
