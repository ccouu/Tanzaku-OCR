"""Controlled one-new-character inputs, more replay, exact2000-update plan."""
from collections import Counter,defaultdict
from pathlib import Path
import random

from scripts.recognition.prepare_v40 import INPUT,ROOT
from scripts.recognition.train_v38_yolo25 import BucketCrops,HOLDOUT
from outputs.recognition.v36_yolo_fullpasses import Combined,load,dump_rows
from outputs.recognition.application_mixture_pilot import RenderedLines
from outputs.recognition.tiered_short_pilot import rows
from outputs.recognition.evaluate_bn_swap import V34
from src.recognition.datasets import FontLines
from src.recognition.font_training import row_canvas
from src.recognition.utils import sha,write_json

V40=Path('outputs/recognition/training/v40_new_characters_2000')
WORK=Path('outputs/recognition/pilots/v41_new_rows_only_2000')
SEED=20261121


def prepare(work,source):
    cfg=source['config'];vocab=load(V40/'vocab.json');mapping=vocab['char_to_id'];added=vocab['extension']['added_characters'];new=set(added)
    assert len(added)==57 and len(mapping)==1779 and vocab['blank_id']==0
    assert all(mapping[c]==i for c,i in source['vocab']['char_to_id'].items())
    assert vocab['characters'][:1722]==source['vocab']['characters']
    write_json(work/'vocab.json',vocab)
    all_target=rows(V40/'targeted/labels.jsonl')
    target=[r for r in all_target if r['split']=='train' and sum(c in new for c in r['text'])==1 and len(r['text'])<=5 and r['canvas']<=256]
    probe=[r for r in all_target if r['split']=='probe']
    assert len(target)==2668 and len(probe)==684
    assert set(r['target_character'] for r in target)==new
    assert all(sum(c in new for c in r['text'])==1 for r in target+probe)
    assert not {r['sha256'] for r in target}&{r['sha256'] for r in probe}
    for r in target+probe:assert sha(V40/'targeted'/r['image'])==r['sha256']
    yolo=rows(V40/'yolo_labels.jsonl');ytrain=[r for r in yolo if r['split']=='train'];yhold=[r for r in yolo if r['split']=='val_holdout']
    assert len(ytrain)==1561 and len(yhold)==156
    assert not {r['source_menu'] for r in ytrain}&HOLDOUT and {r['source_menu'] for r in yhold}==HOLDOUT
    assert not {r['sha256'] for r in ytrain}&{r['sha256'] for r in yhold}
    for r in yolo:assert sha(INPUT/r['image'])==r['sha256']
    dump_rows(work/'target_train_labels.jsonl',target);dump_rows(work/'yolo_labels.jsonl',yolo)
    original=FontLines(ROOT,'train',cfg['mean'],cfg['std']);generated=rows(V34/'corpus/labels.jsonl');corpus=load(V34/'corpus/manifest.json')
    assert sha(ROOT/'labels.jsonl')==corpus['original_labels_sha256'] and sha(V34/'corpus/labels.jsonl')==corpus['labels_sha256']
    mix=Combined(original,RenderedLines(generated,V34/'corpus',cfg,mapping),BucketCrops(ytrain,INPUT,cfg,mapping))
    mix.datasets['target']=BucketCrops(target,V40/'targeted',cfg,mapping)
    required=corpus['required_terms'];mandatory_o=set();mandatory_g={'clean_control':set(),'texture_color_noise':set()}
    for t in required:
        hits=[i for i,r in enumerate(original.rows) if r['text']==t]
        if hits:mandatory_o.add(hits[0])
        else:
            for view in mandatory_g:mandatory_g[view].add(next(i for i,r in enumerate(generated) if r['text']==t and r['view']==view))
    rng=random.Random(SEED);plan=[]
    def width(kind,index):
        r=mix.datasets[kind].rows[index];return row_canvas(r) if kind=='original' else r['canvas']
    def ordinary(kind,pool,count,mandatory=()):
        groups=defaultdict(list);must=defaultdict(list)
        for i in pool:groups[width(kind,i)].append(i)
        for i in sorted(mandatory):must[width(kind,i)].append(i)
        schedule=[]
        for w,ids in sorted(must.items()):schedule.extend([w]*((len(ids)+31)//32))
        choices=sorted(groups);schedule+=rng.choices(choices,weights=[len(groups[w]) for w in choices],k=count-len(schedule))
        queues={w:[] for w in choices}
        def take(w,n):
            result=[]
            while len(result)<n:
                if not queues[w]:queues[w]=rng.sample(groups[w],len(groups[w]))
                need=min(n-len(result),len(queues[w]));result+=queues[w][:need];del queues[w][:need]
            return result
        for w in schedule:
            forced=must[w][:32];del must[w][:len(forced)]
            indices=forced+take(w,32-len(forced));rng.shuffle(indices);plan.append([(kind,i) for i in indices])
        assert not any(must.values())
    ordinary('yolo',range(len(ytrain)),500,range(len(ytrain)))
    ordinary('original',range(len(original)),500,mandatory_o)
    for view in ('clean_control','texture_color_noise'):
        ordinary('generated',[i for i,r in enumerate(generated) if r['view']==view],250,mandatory_g[view])
    pools=defaultdict(dict);queues={};target_counts=Counter()
    for i,r in enumerate(target):pools[r['canvas']].setdefault(r['target_character'],[]).append(i)
    def target_take(c,w):
        key=(c,w)
        if not queues.get(key):queues[key]=rng.sample(pools[w][c],len(pools[w][c]))
        return queues[key].pop()
    for j in range(500):
        anchor=added[j%len(added)];choices=sorted(w for w,p in pools.items() if anchor in p)
        w=rng.choices(choices,weights=[len(pools[w][anchor]) for w in choices])[0]
        batch=[('target',target_take(anchor,w))];target_counts[anchor]+=1
        for slot in range(31):
            minimum=min(target_counts[c] for c in pools[w]);c=rng.choice(sorted(c for c in pools[w] if target_counts[c]==minimum))
            batch.append(('target',target_take(c,w)));target_counts[c]+=1
        rng.shuffle(batch);plan.append(batch)
    rng.shuffle(plan);assert len(plan)==2000 and all(len(b)==32 for b in plan)
    counts=Counter();streams=Counter();terms=Counter();seen_yolo=set()
    for batch in plan:
        assert len({width(k,i) for k,i in batch})==1
        for k,i in batch:
            text=mix.datasets[k].rows[i]['text'];streams[k]+=1;terms[text]+=1;counts.update(c for c in text if c in new)
            if k=='yolo':seen_yolo.add(i)
    assert streams=={'target':16000,'yolo':16000,'original':16000,'generated':16000}
    assert min(target_counts[c] for c in added)>=100 and len(seen_yolo)==1561
    assert all(terms[t]>=1 for t in required)
    write_json(work/'batches.json',plan)
    sampling=dict(updates=2000,presentations=64000,streams=dict(streams),target_fraction=.25,yolo_fraction=.25,old_synthetic_fraction=.5,
        target_records=2668,probe_records=684,target_text_policy='Exactly one added character per line, <=5characters, native128/256width; fixed subset of V40rendered images, no new augmentation.',
        target_character_counts={c:target_counts[c] for c in added},minimum_target_character_encounters=min(target_counts[c] for c in added),
        total_new_character_counts={c:counts[c] for c in added},minimum_total_new_character_encounters=min(counts[c] for c in added),
        yolo_train=1561,yolo_holdout=156,holdout_menus=sorted(HOLDOUT),required_terms=521,required_term_encounters={t:terms[t] for t in required},
        source_vocabulary_sha256=sha(V40/'vocab.json'),source_target_labels_sha256=sha(V40/'targeted/labels.jsonl'),
        source_yolo_labels_sha256=sha(V40/'yolo_labels.jsonl'),plan_sha256=sha(work/'batches.json'),
        vocab_size=1779,added_characters=added,training_contexts_exclude_holdout_menus=True)
    write_json(work/'sampling.json',sampling)
    return vocab,added,mix,plan,BucketCrops(probe,V40/'targeted',cfg,mapping),BucketCrops(yhold,INPUT,cfg,mapping),sampling
