"""35% native YOLO crop jitter,30% genuine long512,15%target,20%other replay."""
from collections import Counter,defaultdict
from pathlib import Path
import random
import numpy as np
from PIL import Image,ImageDraw
import torch

from scripts.recognition.prepare_v41 import V40,ROOT,INPUT
from scripts.recognition.train_v38_yolo25 import BucketCrops,HOLDOUT
from outputs.recognition.v36_yolo_fullpasses import Combined,load,dump_rows
from outputs.recognition.tiered_short_pilot import rows
from outputs.recognition.application_mixture_pilot import RenderedLines
from outputs.recognition.evaluate_bn_swap import V34
from outputs.recognition.v37_long512 import LongLines
from src.recognition.datasets import FontLines
from src.recognition.font_training import row_canvas
from src.recognition.crop_jitter import jitter_parameters,apply_crop_jitter
from src.recognition.data import preprocess,minimum_ctc_steps
from src.recognition.utils import sha,write_json

V41=Path('outputs/recognition/pilots/v41_new_rows_only_2000')
V37=Path('outputs/recognition/pilots/v37_long512_2000')
WORK=Path('outputs/recognition/pilots/v42_yolo35_jitter_long512_2000')
START=V41/'best.pt';SEED=20261122


class AugmentedYolo(RenderedLines):
    def __init__(self,base,views,cfg,mapping):
        self.base=base;self.views=views
        records=[dict(base[v['base_index']],canvas=v['canvas'],valid_width=v['valid_width'],augmentation=v) for v in views]
        super().__init__(records,INPUT,cfg,mapping)

    def __getitem__(self,i):
        v=self.views[i];r=self.base[v['base_index']];p=INPUT/r['display_crop']
        if v['base_index'] not in self.verified:assert sha(p)==r['native_sha256'];self.verified.add(v['base_index'])
        with Image.open(p) as im:strip=im.convert('RGB')
        if r['orientation_flag']:strip=strip.transpose(Image.Transpose.ROTATE_90)
        image,meta=apply_crop_jitter(strip,v)
        tensor,width,compressed=preprocess(image,self.cfg['mean'],self.cfg['std'],'batch',max_width=meta['canvas'])
        assert not compressed and width==meta['canvas'] and (meta['valid_width']+3)//4>=minimum_ctc_steps(r['text'])
        return tensor,torch.tensor([self.mapping[c] for c in r['text']]),r['orientation_flag'],meta['valid_width'],r['text'],False


def prepare(work,source):
    cfg=source['config'];vocab=source['vocab'];mapping=vocab['char_to_id'];added=vocab['extension']['added_characters']
    assert len(mapping)==1779 and len(added)==57 and vocab==load(V41/'vocab.json')
    write_json(work/'vocab.json',vocab)
    yolo=rows(V40/'yolo_labels.jsonl');train=[r for r in yolo if r['split']=='train'];hold=[r for r in yolo if r['split']=='val_holdout']
    assert len(train)==1561 and len(hold)==156 and not {r['source_menu'] for r in train}&HOLDOUT
    assert {r['source_menu'] for r in hold}==HOLDOUT
    assert not {r['sha256'] for r in train}&{r['sha256'] for r in hold}
    for r in yolo:assert sha(INPUT/r['image'])==r['sha256']
    for r in train:
        native=INPUT/r['display_crop'];r['native_sha256']=sha(native)
        with Image.open(native) as im:size=im.size
        r['oriented_native_size']=list(reversed(size)) if r['orientation_flag'] else list(size)
    dump_rows(work/'yolo_base_labels.jsonl',train);dump_rows(work/'yolo_holdout_labels.jsonl',hold)
    rng=random.Random(SEED)
    order=list(range(len(train)))*(22400//len(train))+rng.sample(range(len(train)),22400%len(train));rng.shuffle(order)
    raw=[];first={}
    for i,index in enumerate(order):
        params=jitter_parameters(train[index]['oriented_native_size'],SEED*100000+i)
        assert (params['valid_width']+3)//4>=minimum_ctc_steps(train[index]['text'])
        raw.append(dict(params,base_index=index,view_id=i,quota_duplicate=False));first.setdefault(index,i)
    groups=defaultdict(list)
    for i,v in enumerate(raw):groups[v['canvas']].append(i)
    quotas={w:len(v)//32*32 for w,v in groups.items()}
    remainder=22400-sum(quotas.values())
    for w in sorted(groups,key=lambda w:(len(groups[w])%32,w),reverse=True)[:remainder//32]:quotas[w]+=32
    assert sum(quotas.values())==22400
    protected=set(first.values());views=[]
    for w,indices in sorted(groups.items()):
        forced=[i for i in indices if i in protected];optional=[i for i in indices if i not in protected]
        assert quotas[w]>=len(forced)
        selected=forced+rng.sample(optional,min(len(optional),quotas[w]-len(forced)))
        chosen=[raw[i] for i in selected]
        while len(chosen)<quotas[w]:
            chosen.append(dict(raw[rng.choice(indices)],view_id=22400+len(views)+len(chosen),quota_duplicate=True))
        rng.shuffle(chosen);views.extend(chosen)
    assert len(views)==22400 and {r['base_index'] for r in views}==set(range(1561))
    dump_rows(work/'yolo_views.jsonl',views)
    controlled=rows(V41/'target_train_labels.jsonl');assert len(controlled)==2668
    assert all(sum(c in added for c in r['text'])==1 for r in controlled)
    for r in controlled:assert sha(V40/'targeted'/r['image'])==r['sha256']
    long_rows=rows(V37/'long_labels.jsonl');long_train=[r for r in long_rows if r['split']=='train'];long_test=[r for r in long_rows if r['split']=='holdout']
    assert len(long_train)==4800 and len(long_test)==800
    assert not {r['sha256'] for r in long_train}&{r['sha256'] for r in long_test}
    for r in long_rows:assert r['canvas']==512 and 385<=r['valid_width']<=512 and 11<=len(r['text'])<=14 and all(c in mapping for c in r['text'])
    for r in long_rows:assert sha(V37/r['image'])==r['sha256']
    original=FontLines(ROOT,'train',cfg['mean'],cfg['std']);generated=rows(V34/'corpus/labels.jsonl');corpus=load(V34/'corpus/manifest.json')
    assert sha(ROOT/'labels.jsonl')==corpus['original_labels_sha256'] and sha(V34/'corpus/labels.jsonl')==corpus['labels_sha256']
    aug=AugmentedYolo(train,views,cfg,mapping)
    mix=Combined(original,RenderedLines(generated,V34/'corpus',cfg,mapping),aug)
    mix.datasets['target']=BucketCrops(controlled,V40/'targeted',cfg,mapping);mix.datasets['long512']=LongLines(long_train,V37,cfg,mapping)
    required=corpus['required_terms'];mandatory_o=set();mandatory_g={'clean_control':set(),'texture_color_noise':set()}
    for t in required:
        hits=[i for i,r in enumerate(original.rows) if r['text']==t]
        if hits:mandatory_o.add(hits[0])
        else:
            for view in mandatory_g:mandatory_g[view].add(next(i for i,r in enumerate(generated) if r['text']==t and r['view']==view))
    plan=[]
    for w in sorted(quotas):
        ids=[i for i,v in enumerate(views) if v['canvas']==w];rng.shuffle(ids)
        assert len(ids)%32==0;plan.extend([[('yolo',i) for i in ids[j:j+32]] for j in range(0,len(ids),32)])
    long_ids=[i for i,r in enumerate(long_train) for _ in range(6 if r['view']=='clean' else 2)];rng.shuffle(long_ids)
    assert len(long_ids)==19200;plan.extend([[('long512',i) for i in long_ids[j:j+32]] for j in range(0,len(long_ids),32)])
    def width(kind,index):
        r=mix.datasets[kind].rows[index];return row_canvas(r) if kind=='original' else r['canvas']
    def ordinary(kind,pool,count,mandatory=()):
        buckets=defaultdict(list);must=defaultdict(list)
        for i in pool:buckets[width(kind,i)].append(i)
        for i in sorted(mandatory):must[width(kind,i)].append(i)
        schedule=[]
        for w,ids in sorted(must.items()):schedule.extend([w]*((len(ids)+31)//32))
        choices=sorted(buckets);schedule+=rng.choices(choices,weights=[len(buckets[w]) for w in choices],k=count-len(schedule))
        queues={w:[] for w in choices}
        for w in schedule:
            ids=must[w][:32];del must[w][:len(ids)]
            while len(ids)<32:
                if not queues[w]:queues[w]=rng.sample(buckets[w],len(buckets[w]))
                need=min(32-len(ids),len(queues[w]));ids+=queues[w][:need];del queues[w][:need]
            rng.shuffle(ids);plan.append([(kind,i) for i in ids])
        assert not any(must.values())
    ordinary('original',range(len(original)),200,mandatory_o)
    for view in mandatory_g:ordinary('generated',[i for i,r in enumerate(generated) if r['view']==view],100,mandatory_g[view])
    pools=defaultdict(dict);queues={};target_counts=Counter()
    for i,r in enumerate(controlled):pools[r['canvas']].setdefault(r['target_character'],[]).append(i)
    def target_take(c,w):
        if not queues.get((c,w)):queues[c,w]=rng.sample(pools[w][c],len(pools[w][c]))
        return queues[c,w].pop()
    for j in range(300):
        anchor=added[j%57];choices=sorted(w for w,p in pools.items() if anchor in p)
        w=rng.choices(choices,weights=[len(pools[w][anchor]) for w in choices])[0];batch=[('target',target_take(anchor,w))];target_counts[anchor]+=1
        for slot in range(31):
            least=min(target_counts[c] for c in pools[w]);c=rng.choice(sorted(c for c in pools[w] if target_counts[c]==least))
            batch.append(('target',target_take(c,w)));target_counts[c]+=1
        rng.shuffle(batch);plan.append(batch)
    rng.shuffle(plan);streams=Counter();new_counts=Counter();terms=Counter();canvases=Counter();long_lengths=Counter();long_views=Counter();orientations=Counter()
    for batch in plan:
        assert len(batch)==32 and len({width(k,i) for k,i in batch})==1
        for k,i in batch:
            r=mix.datasets[k].rows[i];streams[k]+=1;new_counts.update(c for c in r['text'] if c in added);terms[r['text']]+=1;canvases[width(k,i)]+=1;orientations[r.get('orientation_flag',0)]+=1
            if k=='long512':long_lengths[len(r['text'])]+=1;long_views[r['view']]+=1
    assert len(plan)==2000 and streams=={'yolo':22400,'long512':19200,'target':9600,'original':6400,'generated':6400}
    assert long_lengths=={n:4800 for n in (11,12,13,14)} and long_views=={'clean':14400,'noisy':4800}
    assert all(new_counts[c]>=100 for c in added) and all(terms[t]>=1 for t in required)
    write_json(work/'batches.json',plan)
    probe_indices=rng.sample(range(len(views)),256)
    aug_probe=AugmentedYolo(train,[views[i] for i in probe_indices],cfg,mapping)
    clean_probe=BucketCrops([train[views[i]['base_index']] for i in probe_indices],INPUT,cfg,mapping)
    write_json(work/'augmented_train_probe_indices.json',probe_indices)
    preview=Image.new('RGB',(1024,8*94),'#e2e8ed');draw=ImageDraw.Draw(preview)
    for i in range(16):
        r=aug_probe.views[i];base=train[r['base_index']]
        with Image.open(INPUT/base['display_crop']) as im:strip=im.convert('RGB')
        if base['orientation_flag']:strip=strip.transpose(Image.Transpose.ROTATE_90)
        image,_=apply_crop_jitter(strip,r);x=i%2*512;y=i//2*94
        preview.paste(image,(x,y+20));draw.text((x+3,y+3),f"{base['crop_id']} | {r['canvas']}px | x{r['horizontal_stretch']:.3f}",fill='#102b40')
    preview.save(work/'augmentation_preview.png')
    sampling=dict(updates=2000,presentations=64000,streams=dict(streams),yolo_fraction=.35,true_long512_fraction=.30,
        physical_width_presentations=dict(canvases),actual_512_fraction=canvases[512]/64000,orientation_presentations=dict(orientations),
        yolo_width_counts=quotas,yolo_training_base_crops=1561,yolo_holdout_crops=156,holdout_menus=sorted(HOLDOUT),
        unique_augmented_transforms=len({(v['base_index'],v['seed']) for v in views}),quota_duplicate_views=sum(v['quota_duplicate'] for v in views),
        margin_ratio_requested=[.02,.05],horizontal_stretch_range=[.95,1.05],padding='Edge replicate; no source pixel cut. Ratios per side independently sampled; native pixel margins rounded,minimum1pixel.',
        no_added_photometric_augmentation=True,holdout_geometry_unchanged=True,augmentation_seed_plan_sha256=sha(work/'yolo_views.jsonl'),
        native_size_min=[min(r['oriented_native_size'][a] for r in train) for a in (0,1)],
        long_char_length_counts=dict(long_lengths),long_view_counts=dict(long_views),long_train_unique_images=4800,long_holdout_images=800,
        long_source_labels_sha256=sha(V37/'long_labels.jsonl'),target_character_counts={c:target_counts[c] for c in added},
        planned_new_character_counts={c:new_counts[c] for c in added},minimum_new_character_encounters=min(new_counts[c] for c in added),
        required_term_count=521,required_term_encounters={t:terms[t] for t in required},plan_sha256=sha(work/'batches.json'),
        source_yolo_labels_sha256=sha(V40/'yolo_labels.jsonl'),source_target_labels_sha256=sha(V41/'target_train_labels.jsonl'),
        vocab_sha256=sha(work/'vocab.json'),labels_verified='VLM OK labels,not independently re-transcribed')
    write_json(work/'sampling.json',sampling)
    extra={'yolo_current_train':BucketCrops(train,INPUT,cfg,mapping),'yolo_current_holdout':BucketCrops(hold,INPUT,cfg,mapping),
           'augmented_train_probe':aug_probe,'matched_clean_train_probe':clean_probe,
           'new_character_probe':BucketCrops([r for r in rows(V40/'targeted/labels.jsonl') if r['split']=='probe'],V40/'targeted',cfg,mapping)}
    for view in ('clean','noisy'):
        values=[r for r in long_test if r['view']==view]
        extra['long512_'+view]=LongLines(values,V37,cfg,mapping);extra['long384_'+view]=LongLines(values,V37,cfg,mapping,fit384=True)
    return vocab,added,mix,plan,extra,sampling
