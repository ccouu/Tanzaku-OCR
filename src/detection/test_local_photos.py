"""Run local OBB inference on a fresh, unlabelled scraper collection."""
import argparse
from collections import Counter
import os
from pathlib import Path
import time

import numpy as np
from src.common import digest, dump_json
from src.common.paths import DATASET_V2, INFERENCE_OUTPUT, RUNTIME_ROOT, ROOT, resolve_path
from .common import bootstrap_packages, read_json


def run(root, output, weights, expected=100, confidence=.25):
    root,output,weights=resolve_path(root),resolve_path(output),resolve_path(weights)
    collection=read_json(root/'manifest.json');rows=collection['images']
    if len(rows)!=expected or collection['status']!='completed':raise ValueError('Collection is incomplete')
    if len({r['restaurant_id'] for r in rows})!=expected or len({r['sha256'] for r in rows})!=expected:
        raise ValueError('Duplicate test images or restaurant identities')
    training=read_json(DATASET_V2/'training_manifest.json')
    known_restaurants={g.split(':',1)[1] for r in training['records'] for g in r['groups'] if g.startswith('restaurant:')}
    known_hashes={r['original_sha256'] for r in training['records']}
    if known_restaurants & {r['restaurant_id'] for r in rows}:raise ValueError('Restaurant overlap with model development dataset')
    if known_hashes & {r['sha256'] for r in rows}:raise ValueError('Image overlap with model development dataset')
    bootstrap_packages()
    os.environ.setdefault('YOLO_CONFIG_DIR',str(RUNTIME_ROOT))
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    model=YOLO(str(weights));model_sha=digest(weights)
    output.mkdir(parents=True,exist_ok=True)
    classes=Counter();timings=[];entries=[]
    for index,row in enumerate(rows):
        image=root/row['image']
        if digest(image)!=row['sha256']:raise ValueError('Photo checksum changed')
        start=time.perf_counter()
        result=model.predict(str(image),imgsz=640,conf=confidence,device='cpu',verbose=False)[0]
        elapsed=(time.perf_counter()-start)*1000;timings.append(elapsed)
        shapes=[];lines=[]
        if result.obb is not None:
            for points,normal,cls,score in zip(result.obb.xyxyxyxy.cpu().numpy(),result.obb.xyxyxyxyn.cpu().numpy(),
                                              result.obb.cls.cpu().numpy(),result.obb.conf.cpu().numpy()):
                label=model.names[int(cls)];classes[label]+=1
                shapes.append(dict(label=label,class_id=int(cls),confidence=float(score),points=points.tolist(),
                                   shape_type='polygon',flags={'reviewed':False}))
                lines.append(str(int(cls))+' '+' '.join(f'{n:.8f}' for n in normal.flatten()))
        doc=dict(imagePath=str(image.resolve()),imageWidth=row['width'],imageHeight=row['height'],
                 image_sha256=row['sha256'],restaurant_id=row['restaurant_id'],source_info=str(image.with_suffix('.json').resolve()),
                 model_sha256=model_sha,flags={'reviewed':False},shapes=shapes,inference_ms=elapsed)
        dump_json(output/'predictions'/(image.stem+'.json'),doc)
        (output/'predictions'/(image.stem+'.txt')).write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')
        result.save(filename=str(output/(image.stem+'_prediction.jpg')))
        entries.append(dict(image=row['image'],restaurant_id=row['restaurant_id'],sha256=row['sha256'],boxes=len(shapes)))
        if (index+1)%20==0:print(f'Local inference: {index+1}/{expected}',flush=True)
    report=dict(status='completed',images=len(rows),boxes=sum(classes.values()),classes=dict(classes),
                zero_detection_images=sum(r['boxes']==0 for r in entries),device='cpu',confidence=confidence,imgsz=640,
                checkpoint=str(weights),checkpoint_sha256=model_sha,source_collection=str(root),
                random_seed=collection.get('random_seed'),sampling_policy=collection.get('sampling_policy'),
                development_restaurant_overlap=0,development_image_overlap=0,
                inference_ms=dict(mean=float(np.mean(timings)),p50=float(np.median(timings)),p95=float(np.percentile(timings,95))),
                visually_inspected=False,ground_truth_available=False,accuracy_metrics=None,
                note='Fresh unlabelled menu-photo inference; no detection accuracy claim. First-image time includes warm-up.',records=entries)
    dump_json(output/'report.json',report)
    print({k:v for k,v in report.items() if k!='records'},flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('data/raw/tabelog_test_100'))
    p.add_argument('--output',type=Path,default=INFERENCE_OUTPUT)
    p.add_argument('--weights',type=Path,default=Path('checkpoints/yolo11n_menu_obb_best.pt'))
    p.add_argument('--expected',type=int,default=100)
    p.add_argument('--confidence',type=float,default=.25)
    a=p.parse_args();run(a.input,a.output,a.weights,a.expected,a.confidence)
