"""Paired V3/V5 raw and production-crop geometry; heuristic foreground loss.

Human boxes are geometric references, not pixel-perfect stroke masks. Never call
the threshold-derived foreground estimate a verified clipped-stroke count.
"""
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.common import digest, dump_json
from src.detection.common import read_json, yolo_text
from src.detection.line_crops import area, intersection, overlap, frame, process
from scripts.stage2_yolo.build_crop_review import quality
from scripts.stage2_yolo.finetune_all130 import DATA, START, WORK
from scripts.stage2_yolo.finetune_banana_latest import runtime, ROOT


def geometry(pred, gt):
    pa, ga = area(pred), area(gt)
    common = area(intersection(pred, gt))
    u, v = frame(gt)
    p, g = np.asarray(pred), np.asarray(gt)
    h = max(1e-9, np.ptp(g @ v))
    # Positive: extra margin; negative: inward edge relative to annotation.
    sides = [((g@u).min()-(p@u).min())/h, ((p@u).max()-(g@u).max())/h,
             ((g@v).min()-(p@v).min())/h, ((p@v).max()-(g@v).max())/h]
    return dict(iou=common/max(1e-9, pa+ga-common),
                excess_fraction=(pa-common)/max(1e-9, pa),
                missed_fraction=(ga-common)/max(1e-9, ga),
                outside_area=max(0., pa-common), missed_area=max(0., ga-common),
                pred_area=pa, gt_area=ga, edge_offsets_line_heights=[float(x) for x in sides])


def mask_polygon(points, origin, size):
    im = Image.new('1', size)
    ImageDraw.Draw(im).polygon([(float(x-origin[0]), float(y-origin[1])) for x,y in points], fill=1)
    return np.asarray(im, dtype=bool)


def foreground(image, gt):
    """Conservative Otsu polarity proxy, rejecting weak/ambiguous separation."""
    q = np.asarray(gt)
    x0, y0 = np.maximum(0, np.floor(q.min(0))).astype(int)
    x1, y1 = np.minimum(image.size, np.ceil(q.max(0))+1).astype(int)
    gray = np.asarray(image.crop((x0,y0,x1,y1)).convert('L'))
    valid = mask_polygon(gt, (x0,y0), (x1-x0,y1-y0))
    values = gray[valid]
    if len(values)<20: return None
    hist = np.bincount(values, minlength=256).astype(float)
    prob=hist/hist.sum(); w=np.cumsum(prob); mu=np.cumsum(prob*np.arange(256))
    score=(mu[-1]*w-mu)**2/np.maximum(1e-12,w*(1-w))
    t=int(score.argmax()); low=valid & (gray<=t); high=valid & (gray>t)
    if not low.any() or not high.any(): return None
    contrast=float(gray[high].mean()-gray[low].mean())
    # Border band chooses background polarity; interiors may contain dense ink.
    yy,xx=np.indices(gray.shape); pts=np.stack((xx+x0,yy+y0),axis=-1)
    u,v=frame(gt); along=pts@u; across=pts@v
    a=q@u;b=q@v
    border=valid & ((along<a.min()+.08*np.ptp(a)) | (along>a.max()-.08*np.ptp(a)) |
                    (across<b.min()+.12*np.ptp(b)) | (across>b.max()-.12*np.ptp(b)))
    if not border.any(): return None
    bg_high=float((gray[border]>t).mean())
    if .35<bg_high<.65: return None
    ink=low if bg_high>=.65 else high
    occupancy=float(ink.sum()/valid.sum())
    if contrast<35 or not .005<=occupancy<=.55: return None
    return dict(mask=ink, origin=(int(x0),int(y0)), size=(int(x1-x0),int(y1-y0)),
                contrast=contrast, occupancy=occupancy)


def ink_loss(proxy, pred):
    if proxy is None: return None
    inside=mask_polygon(pred, proxy['origin'], proxy['size'])
    return float((proxy['mask'] & ~inside).sum()/max(1,proxy['mask'].sum()))


def summarize(rows, paired=False):
    use=[r for r in rows if all(r['models'][m] is not None for m in ('v3','v5'))] if paired else rows
    result=dict(truth=len(rows), paired_lines=sum(all(r['models'][m] is not None for m in ('v3','v5')) for r in rows))
    for m in ('v3','v5'):
        matched=[r['models'][m] for r in use if r['models'][m] is not None]
        ink=[r['ink_loss'][m] for r in use if r['models'][m] is not None and r['ink_loss'][m] is not None]
        result[m]=dict(matched=len(matched), missed=len(use)-len(matched),
            mean_excess_pct=float(np.mean([v['excess_fraction'] for v in matched])*100) if matched else None,
            mean_missed_pct=float(np.mean([v['missed_fraction'] for v in matched])*100) if matched else None,
            excess_gt5pct=sum(v['excess_fraction']>.05 for v in matched),
            missed_gt5pct=sum(v['missed_fraction']>.05 for v in matched),
            foreground_eligible=len(ink), mean_foreground_loss_pct=float(np.mean(ink)*100) if ink else None,
            foreground_loss_gt1pct=sum(x>.01 for x in ink))
    return result


def make_overlay(image, truth, predictions, dest, bounds=None):
    if bounds:
        x0,y0,x1,y1=bounds; im=image.crop(bounds)
    else:
        x0=y0=0;im=image.copy()
    scale=min(1., (1000 if bounds else 1500)/max(im.size))
    im=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))))
    draw=ImageDraw.Draw(im)
    for polygons,color in ((truth,'#12d466'),(predictions['v3'],'#ff3c50'),(predictions['v5'],'#00baff')):
        for poly in polygons:
            pts=[((x-x0)*scale,(y-y0)*scale) for x,y in poly]
            draw.line(pts+[pts[0]],fill=color,width=2)
    dest.parent.mkdir(parents=True,exist_ok=True);im.save(dest,quality=90)


def main():
    out=ROOT/'outputs/detection/evaluation'/('boundary_v3_v5_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True,exist_ok=False)
    manifest=read_json(DATA/'training_manifest.json');records=sorted(manifest['records'],key=lambda r:r['id'])
    assert len(records)==130
    weights={'v3':START,'v5':WORK/'yolo11n_menu_obb_banana130_best.pt'}
    hashes={m:digest(p) for m,p in weights.items()}
    torch,YOLO=runtime();torch.set_num_threads(4)
    settings=dict(imgsz=640,conf=.25,iou=.7,max_det=300,device=0,half=False,verbose=False,augment=False)
    predictions={m:{} for m in weights}
    for m,path in weights.items():
        model=YOLO(str(path))
        for i,r in enumerate(records):
            source=DATA/'images'/r['split']/(r['id']+'.jpg')
            assert digest(source)==r['image_sha256']
            result=model.predict(str(source),**settings)[0]
            detections=[]
            if result.obb is not None:
                for j,(points,cls,conf) in enumerate(zip(result.obb.xyxyxyxy.cpu().tolist(),result.obb.cls.cpu().tolist(),result.obb.conf.cpu().tolist())):
                    # Keep true predicted quadrilaterals; raster masks naturally clip to image.
                    detections.append(dict(id=f'det{j:03d}',label=model.names[int(cls)],confidence=float(conf),points=points))
            predictions[m][r['id']]=detections
            if (i+1)%25==0: print('PREDICT',m,i+1,flush=True)
        assert digest(path)==hashes[m]
        dump_json(out/f'predictions_{m}.json',predictions[m])
    allrows=[];pages=[]
    for i,r in enumerate(records):
        ident=r['id'];path=DATA/'images'/r['split']/(ident+'.jpg')
        assert digest(DATA/'labels'/r['split']/(ident+'.txt'))==r['label_sha256']
        source_doc=read_json(Path(r['reviewed_annotation']))
        assert digest(Path(r['reviewed_annotation']))==r['annotation_sha256'], 'Reviewed annotations changed since V5 snapshot'
        doc=read_json(DATA/'annotations'/r['split']/(ident+'.json'));truth=doc['shapes']
        assert yolo_text(truth,doc['imageWidth'],doc['imageHeight'])==(DATA/'labels'/r['split']/(ident+'.txt')).read_text(encoding='utf-8')
        with Image.open(path) as source:image=source.convert('RGB')
        proxies=[foreground(image,g['points']) for g in truth]
        modes={}
        for mode in ('raw','production_crop'):
            ps={}
            for m in weights:
                raw=predictions[m][ident]
                ps[m]=raw if mode=='raw' else [dict(points=c['crop_polygon'],label=c['label']) for c in process(raw,*image.size)['crops']]
            matches={m:quality(ps[m],truth,.5) for m in weights}
            maps={m:{v['truth']:v['prediction'] for v in matches[m]['matches']} for m in weights}
            rows=[]
            for j,g in enumerate(truth):
                models={m:geometry(ps[m][maps[m][j]]['points'],g['points']) if j in maps[m] else None for m in weights}
                losses={m:ink_loss(proxies[j],ps[m][maps[m][j]]['points']) if j in maps[m] else None for m in weights}
                rows.append(dict(menu=ident,split=r['split'],group=r['group'],mode=mode,truth_index=j,label=g['label'],
                                 models=models,ink_loss=losses,foreground_eligible=proxies[j] is not None))
            modes[mode]=dict(rows=rows,predictions={m:len(ps[m]) for m in weights},
                             unmatched_predictions={m:len(ps[m])-len(maps[m]) for m in weights},
                             summary=summarize(rows),paired=summarize(rows,True))
            allrows.extend(rows)
        overview=f'overlays/{ident}.jpg'
        make_overlay(image,[g['points'] for g in truth],{m:[p['points'] for p in predictions[m][ident]] for m in weights},out/overview)
        # Three largest paired foreground/geometric differences per page, no visual inference.
        ranked=sorted(modes['raw']['rows'],key=lambda row:max([abs((row['models']['v3'] or {}).get(k,1.)-(row['models']['v5'] or {}).get(k,1.)) for k in ('missed_fraction','excess_fraction')]),reverse=True)[:3]
        details=[]
        for row in ranked:
            j=row['truth_index'];q=np.asarray(truth[j]['points']);margin=max(12,int(np.ptp(q[:,1])*.15));lo=np.maximum(0,np.floor(q.min(0))-margin).astype(int);hi=np.minimum(image.size,np.ceil(q.max(0))+margin).astype(int)
            imagefile=f'details/{ident}_{j:02d}.jpg'
            nearby={m:[p['points'] for p in predictions[m][ident] if overlap(p['points'],q)['iou']>.1] for m in weights}
            make_overlay(image,[q.tolist()],nearby,out/imagefile,(*lo,*hi))
            details.append(dict(image=imagefile,truth_index=j,models=row['models'],ink_loss=row['ink_loss']))
        page=dict(id=ident,split=r['split'],group=r['group'],overview=overview,details=details,modes=modes)
        pages.append(page);dump_json(out/'menus'/(ident+'.json'),page)
        if (i+1)%20==0:print('MEASURE',i+1,flush=True)
    aggregates={}
    for mode in ('raw','production_crop'):
        aggregates[mode]={}
        for split in ('all','train','val','first50','second50','new30'):
            rows=[r for r in allrows if r['mode']==mode and (split=='all' or r['split']==split or r['group']==split)]
            selected=[p for p in pages if split=='all' or p['split']==split or p['group']==split]
            aggregates[mode][split]=dict(all_matched=summarize(rows),common_matched=summarize(rows,True),
                predictions={m:sum(p['modes'][mode]['predictions'][m] for p in selected) for m in weights},
                unmatched_predictions={m:sum(p['modes'][mode]['unmatched_predictions'][m] for p in selected) for m in weights})
    report=dict(state='completed',pages=130,truth_boxes=sum(r['boxes'] for r in records),settings=settings,
                weights={m:dict(path=str(weights[m]),sha256=hashes[m]) for m in weights},
                manifest_sha256=digest(DATA/'training_manifest.json'),aggregates=aggregates,
                method=dict(matching='Class-aware greedy rotated IoU >=0.5, one prediction per GT. Common-matched comparisons use the same GT lines.',
                    excess='Area of prediction outside reviewed box / predicted area.',
                    missed='Area of reviewed box outside prediction / reviewed area.',
                    foreground='Otsu threshold with boundary-band background polarity, contrast >=35/255, occupancy .005–.55; ambiguous crops excluded. Pixel-loss proxy, NOT verified strokes.',
                    production='Identical conservative deduplication/20% overlap aligned merges/height-relative collision-bounded margins for both models.',
                    caveats=['Reviewed boxes are not pixel-perfect stroke masks.','All130 includes V5 training pages; validation13 reported separately.',
                             'Merged boxes may span multiple GT lines; one-to-one matching can penalize these.',
                             'Matching at IoU .5 excludes severe partial detections from margin statistics; unmatched GT reported separately.',
                             'Heuristic foreground may include texture or miss faint/colored strokes. No manual visual confirmation performed.']))
    for m in weights:assert digest(weights[m])==hashes[m]
    dump_json(out/'report.json',report)
    dump_json(out/'line_measurements.json',allrows)
    write_review(out,report,pages)
    audit_proposal_bias(out)
    print('BOUNDARY_REPORT',out,flush=True)
    print(json.dumps(aggregates,ensure_ascii=False),flush=True)


def write_review(out,report,pages):
    lines=['# V3 / V5 crop-boundary comparison','',
           'Same confidence 0.25, IoU NMS 0.7, 640px inference. Latest reviewed labels verified against frozen V5 snapshot.',
           'Lower excess area and missed annotated area are better. Foreground loss is an unverified segmentation proxy, not a count of clipped strokes.',
           'Common-matched lines avoid comparing different subsets; missed detections are reported separately.', '']
    for mode in ('raw','production_crop'):
        lines += [f'## {mode}', '', '| Set | Shared lines | V3 excess | V5 excess | V3 missed area | V5 missed area | V3 unmatched GT | V5 unmatched GT |', '|---|---:|---:|---:|---:|---:|---:|---:|']
        for split in ('all','train','val','first50','second50','new30'):
            a=report['aggregates'][mode][split];c=a['common_matched'];s=a['all_matched']
            lines.append(f"| {split} | {c['paired_lines']} | {c['v3']['mean_excess_pct']:.2f}% | {c['v5']['mean_excess_pct']:.2f}% | {c['v3']['mean_missed_pct']:.2f}% | {c['v5']['mean_missed_pct']:.2f}% | {s['v3']['missed']} | {s['v5']['missed']} |")
        lines += ['']
    lines += ['All130: 117 V5 training pages and 13 selection-validation pages. Training-page improvements do not establish generalization.', '',
              'Green = reviewed box; red = V3; blue = V5. Review index.html for all130 menus and worst-difference detail overlays.']
    (out/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    cards=[]
    for p in pages:
        c=p['modes']['raw']['paired'];a=p['modes']['raw']['summary']
        text=' | '.join(f"{m.upper()}: excess {c[m]['mean_excess_pct']:.2f}%, missed area {c[m]['mean_missed_pct']:.2f}%, unmatched GT {a[m]['missed']}" for m in ('v3','v5')) if c['paired_lines'] else 'No shared matched lines'
        details=''.join(f'<figure><img loading="lazy" src="{d["image"]}"><figcaption>GT line {d["truth_index"]+1}</figcaption></figure>' for d in p['details'])
        cards.append(f'<section data-split="{p["split"]}" id="{p["id"]}"><h2>{p["id"]} · {p["split"]} · {p["group"]}</h2><p>{html.escape(text)}</p><a href="{p["overview"]}" target="_blank"><img class="page" loading="lazy" src="{p["overview"]}"></a><div class="details">{details}</div><a href="menus/{p["id"]}.json">Per-line measurements</a></section>')
    doc='''<!doctype html><meta charset="utf-8"><title>V3 vs V5 boundary review</title>
<style>body{font:16px system-ui;background:#151922;color:#eee;margin:24px}header{position:sticky;top:0;background:#151922;padding:12px;z-index:2}a{color:#8dd8ff}section{border-top:1px solid #556;padding:18px 0}.page{max-width:95%;max-height:900px}.details{display:flex;flex-wrap:wrap;gap:12px}.details figure{margin:0;max-width:48%}.details img{max-width:100%;max-height:240px}select,input{font:inherit;padding:5px}</style>
<header><b>V3 / V5 · 130-menu boundary comparison</b><p><span style="color:#12d466">Green: reviewed GT</span> · <span style="color:#ff3c50">Red: V3</span> · <span style="color:#00baff">Blue: V5</span></p>
<p>Raw boxes at identical confidence .25. Details show largest differences, not verified clipped strokes. Training pages are not independent evaluation.</p>
<a href="analysis.md">Summary</a> · <a href="report.json">Full report</a> · Filter <select id="split"><option value="all">All130</option><option value="val">Validation13</option><option value="train">Training117</option></select> <input id="find" placeholder="menu-109"></header>'''+''.join(cards)+'''<script>function filter(){let s=document.getElementById('split').value,q=document.getElementById('find').value.toLowerCase();document.querySelectorAll('section').forEach(e=>e.hidden=(s!=='all'&&e.dataset.split!==s)||!e.id.includes(q))}document.getElementById('split').onchange=filter;document.getElementById('find').oninput=filter;</script>'''
    (out/'index.html').write_text(doc,encoding='utf-8')


def audit_proposal_bias(out):
    """Expose proposal-derived label bias; add menu-cluster uncertainty."""
    out=Path(out)
    report=read_json(out/'report.json');rows=read_json(out/'line_measurements.json')
    raw=[r for r in rows if r['mode']=='raw']
    provenance={}
    for folder in ('gemini_banana_menus_001_100_rereview','gemini_banana_menus_101_130'):
        p=ROOT/'data/annotations'/folder/'prediction_report.json'
        provenance[folder]=read_json(p)
    counts={m:sum(r['models'][m] is not None and r['models'][m]['iou']>=.999 for r in raw) for m in ('v3','v5')}
    audit=dict(nearly_exact_iou_threshold=.999, nearly_exact_gt_matches=counts,
               total_truth=len(raw), proposal_provenance=provenance,
               warning='Both annotation queues originate from V3 proposals. Near-exact matches favor V3 by construction; agreement with these boxes is not independent proof of stroke-tightness.',
               nonidentical_subset={})
    for split in ('all','val'):
        subset=[r for r in raw if (split=='all' or r['split']==split) and r['models']['v3'] is not None and r['models']['v3']['iou']<.999]
        audit['nonidentical_subset'][split]=summarize(subset,True)
    rng=np.random.default_rng(20261112)
    audit['paired_menu_bootstrap']={}
    for mode in ('raw','production_crop'):
        audit['paired_menu_bootstrap'][mode]={}
        for split in ('all','val'):
            subset=[r for r in rows if r['mode']==mode and (split=='all' or r['split']==split) and all(r['models'][m] is not None for m in ('v3','v5'))]
            menus=sorted({r['menu'] for r in subset});effects={}
            for metric in ('excess_fraction','missed_fraction','foreground_loss'):
                differences={menu:[] for menu in menus}
                for r in subset:
                    if metric=='foreground_loss':
                        if any(r['ink_loss'][m] is None for m in ('v3','v5')):continue
                        a,b=r['ink_loss']['v3'],r['ink_loss']['v5']
                    else:a,b=r['models']['v3'][metric],r['models']['v5'][metric]
                    differences[r['menu']].append((b-a)*100)
                sums=np.array([sum(differences[m]) for m in menus]);ns=np.array([len(differences[m]) for m in menus])
                samples=rng.integers(0,len(menus),size=(3000,len(menus)))
                vals=sums[samples].sum(1)/np.maximum(1,ns[samples].sum(1))
                effects[metric]=dict(v5_minus_v3_pp=float(sums.sum()/max(1,ns.sum())),ci95_pp=np.percentile(vals,[2.5,97.5]).tolist(),menus=len(menus),lines=int(ns.sum()))
            audit['paired_menu_bootstrap'][mode][split]=effects
    report['proposal_bias_audit']=audit
    dump_json(out/'report.json',report);dump_json(out/'proposal_bias_audit.json',audit)
    warning=f"IMPORTANT: V3 generated the annotation proposals. {counts['v3']}/{len(raw)} GT boxes match V3 at IoU >= .999, versus {counts['v5']} for V5. Geometric agreement is biased toward V3 and cannot independently prove tighter text crops."
    analysis=out/'analysis.md';text=analysis.read_text(encoding='utf-8')
    if '## Proposal bias audit' not in text:
        text+='\n## Proposal bias audit\n\n'+warning+'\n\n'
        for split,s in audit['nonidentical_subset'].items():
            text+=f"Non-identical V3 subset, {split}: {s['paired_lines']} common matched lines; V3/V5 excess {s['v3']['mean_excess_pct']:.2f}/{s['v5']['mean_excess_pct']:.2f}%, missed area {s['v3']['mean_missed_pct']:.2f}/{s['v5']['mean_missed_pct']:.2f}%. This selected subset is also not unbiased.\n\n"
        text+='Foreground-loss differences and menu-cluster bootstrap intervals are in proposal_bias_audit.json. Neither heuristics nor these labels establish actual clipped-stroke counts.\n'
        analysis.write_text(text,encoding='utf-8')
    page=out/'index.html';text=page.read_text(encoding='utf-8')
    if 'id="bias-warning"' not in text:
        text=text.replace('<header>','<header><p id="bias-warning" style="color:#ffce70">'+html.escape(warning)+'</p>',1)
        page.write_text(text,encoding='utf-8')
    return audit


if __name__=='__main__':main()
