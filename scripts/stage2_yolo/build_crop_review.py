"""Generate a before/after crop review dataset from the all130 selected detector."""
from collections import Counter,defaultdict
from pathlib import Path
import shutil
from PIL import Image
from src.common import digest,dump_json
from src.detection.common import read_json,validate_shapes
from src.detection.line_crops import process,overlap,box,rectify,recognition_input,area
from scripts.stage2_yolo.finetune_all130 import DATA,WORK
from scripts.stage2_yolo.finetune_banana_latest import runtime

OUT=Path('outputs/detection/crop_review/banana130_v5').resolve()


def quality(predictions,truth,threshold):
    pairs=[]
    for i,p in enumerate(predictions):
        for j,g in enumerate(truth):
            if p['label']==g['label']:
                score=overlap(p['points'],g['points'])['iou']
                if score>=threshold:pairs.append((score,i,j))
    used_p=set();used_g=set();matches=[]
    for score,i,j in sorted(pairs,reverse=True):
        if i not in used_p and j not in used_g:used_p.add(i);used_g.add(j);matches.append(dict(prediction=i,truth=j,iou=score))
    return dict(tp=len(matches),predicted=len(predictions),truth=len(truth),matches=matches)


def overlap_pairs(polygons):
    return [dict(a=i,b=j,**overlap(a,b)) for i,a in enumerate(polygons) for j,b in enumerate(polygons) if j>i and overlap(a,b)['iom']>.02]


def summary_quality(values):
    tp=sum(x['tp'] for x in values);pred=sum(x['predicted'] for x in values);gt=sum(x['truth'] for x in values)
    precision=tp/max(1,pred);recall=tp/max(1,gt)
    return dict(tp=tp,predictions=pred,truth=gt,precision=precision,recall=recall,f1=2*precision*recall/max(1e-12,precision+recall))


def build():
    OUT.mkdir(parents=True,exist_ok=True);training=read_json(WORK/'evaluation_report.json');manifest=read_json(DATA/'training_manifest.json')
    weights=Path(training['selected_checkpoint']);assert digest(weights)==training['selected_sha256']
    torch,YOLO=runtime();torch.set_num_threads(4);model=YOLO(str(weights));menus=[];totals=Counter();events=Counter();bad=[];measurements=[]
    for index,r in enumerate(sorted(manifest['records'],key=lambda x:x['id']),1):
        ident=r['id'];source=DATA/'images'/r['split']/(ident+'.jpg');assert digest(source)==r['image_sha256']
        image_path=OUT/'pages'/(ident+'.jpg');image_path.parent.mkdir(parents=True,exist_ok=True)
        if image_path.exists():assert digest(image_path)==r['image_sha256']
        else:shutil.copy2(source,image_path)
        with Image.open(source) as im:image=im.convert('RGB')
        width,height=image.size;gt=read_json(DATA/'annotations'/r['split']/(ident+'.json'))['shapes']
        cached=OUT/'experimental/menus'/(ident+'.json')
        cache=read_json(cached) if cached.exists() else None
        result=None if cache else model.predict(str(source),imgsz=640,conf=.25,iou=.7,max_det=300,device=0,verbose=False)[0]
        raw=cache['raw'] if cache else [];invalid=cache.get('invalid_detections',[]) if cache else []
        if result is not None and result.obb is not None:
            for j,(q,cls,conf) in enumerate(zip(result.obb.xyxyxyxy.cpu().tolist(),result.obb.cls.cpu().tolist(),result.obb.conf.cpu().tolist())):
                entry=dict(id=f'det{j:03d}',label=model.names[int(cls)],confidence=float(conf),points=[[max(0.,min(width,float(x))),max(0.,min(height,float(y)))] for x,y in q])
                try:validate_shapes([entry],width,height)
                except ValueError as exc:invalid.append(dict(proposal=j,error=str(exc)));continue
                b=box(entry['points'],entry['label'],entry['confidence'],[entry['id']]);q=b['core'];u=b['u'];v=b['v'];along=.04*b['span'];across=.06*b['height']
                pad=q.copy();pad[0]-=u*along+v*across;pad[1]+=u*along-v*across;pad[2]+=u*along+v*across;pad[3]+=-u*along+v*across
                pad[:,0]=pad[:,0].clip(0,width);pad[:,1]=pad[:,1].clip(0,height);entry['legacy_crop_polygon']=pad.tolist();raw.append(entry)
        processed=process(raw,width,height);events.update(d['action'] for d in processed['decisions'])
        for crop in processed['crops']:
            strip=rectify(image,crop['crop_polygon']);normalized,meta=recognition_input(strip)
            native=OUT/'crops/native'/f"{ident}_{crop['id']}.png";normal=OUT/'crops/recognition'/native.name
            native.parent.mkdir(parents=True,exist_ok=True);normal.parent.mkdir(parents=True,exist_ok=True);strip.save(native,compress_level=1);normalized.save(normal,compress_level=1)
            crop.update(native_image=str(native.relative_to(OUT)).replace('\\','/'),recognition_image=str(normal.relative_to(OUT)).replace('\\','/'),
                native_size=list(strip.size),native_sha256=digest(native),recognition_sha256=digest(normal),**meta)
            # Every original member's vertices must remain inside the merged core.
            from src.detection.line_crops import inside
            for member in raw:
                if member['id'] in crop['source_ids']:
                    if not all(inside(p,crop['core']) for p in member['points']):
                        # Suppressed near-duplicates may extend a few pixels beyond
                        # the chosen confidencewinner:report it,never hide it.
                        bad.append(dict(menu=ident,crop=crop['id'],source=member['id'],reason='suppressed_duplicate_extends_outside_selected_core'))
        before_core=overlap_pairs([x['points'] for x in raw]);before_pad=overlap_pairs([x['legacy_crop_polygon'] for x in raw]);after_pad=overlap_pairs([x['crop_polygon'] for x in processed['crops']])
        quality_report={}
        for name,preds in (('raw',raw),('processed',[dict(label=c['label'],points=c['core']) for c in processed['crops']])):
            quality_report[name]={str(t):quality(preds,gt,t) for t in (.5,.75)}
        record=dict(id=ident,split=r['split'],group=r['group'],image=str(image_path.relative_to(OUT)).replace('\\','/'),width=width,height=height,
            raw=raw,processed=processed['crops'],decisions=processed['decisions'],conflicts=processed['conflicts'],truth=gt,
            before_core_overlaps=before_core,before_padding_overlaps=before_pad,after_padding_overlaps=after_pad,quality=quality_report,
            invalid_detections=invalid,policy=processed['policy'])
        if cache:
            preview=cache['processed']
            for c in preview:
                c['native_image']='experimental/'+c['native_image'];c['recognition_image']='experimental/'+c['recognition_image']
            record['adjacent_merge_preview']=dict(crops=preview,decisions=cache['decisions'],conflicts=cache['conflicts'])
        dump_json(OUT/'menus'/(ident+'.json'),record)
        menu=dict(id=ident,split=r['split'],group=r['group'],raw=len(raw),processed=len(processed['crops']),conflicts=len(processed['conflicts']),
            before_padding_overlaps=len(before_pad),after_padding_overlaps=len(after_pad),merged=sum(d['action']=='merge_line_fragments' for d in processed['decisions']),
            suppressed=sum(d['action']=='suppress_duplicate' for d in processed['decisions']),padding_reduced=sum(d['action']=='reduce_padding' for d in processed['decisions']))
        menus.append(menu);measurements.append(record)
        totals.update(dict(raw=len(raw),processed=len(processed['crops']),core_conflicts=len(processed['conflicts']),old_padding_overlap_pairs=len(before_pad),new_padding_overlap_pairs=len(after_pad)))
        if index%10==0:print('BUILD_REVIEW',index,130,dict(totals),flush=True)
    aggregate={}
    for split in ('all','train','val'):
        cases=[r for r in measurements if split=='all' or r['split']==split]
        aggregate[split]={name:{str(t):summary_quality([r['quality'][name][str(t)] for r in cases]) for t in (.5,.75)} for name in ('raw','processed')}
    assert digest(weights)==training['selected_sha256']
    report=dict(pages=130,weights=str(weights),weights_sha256=training['selected_sha256'],using_finetuned=training['selected_is_finetuned'],
        totals=dict(totals),decisions=dict(events),fixed_threshold_quality=aggregate,coverage_warnings=bad,
        note='RawmeansYOLOoutputafterinternalNMS. Processedrequires20percentintersectionoversmallerareaPLUSalignmentforautomaticmerging. Adjacentmergingisanexperimentalpreviewonly. Unresolvedcoreoverlapispreserved/flagged. '
             'GTmatchesarefixedthresholdprecision/recall,notAP.117pagestrained/13valusedforselection. Croptranscriptionsnotverified;noOCRCERclaim.')
    dump_json(OUT/'report.json',report);dump_json(OUT/'index.json',dict(menus=menus,report=report,training=training))
    print('REVIEW_READY',OUT,dict(totals),flush=True)


if __name__=='__main__':build()
