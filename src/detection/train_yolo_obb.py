"""Standalone Colab worker: verify reviewed snapshot, train YOLO11n-OBB, evaluate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
import zipfile
import math


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def publish_recovery(trainer, work, state, epochs, prior):
    """Publish immutable, complete checkpoint copies for concurrent local retrieval."""
    epoch=int(trainer.epoch)+1
    recovery=prior
    if epoch==1 or epoch%5==0 or epoch==epochs:
        recovery=dict(epoch=epoch,files={})
        directory=work/'recovery';directory.mkdir(exist_ok=True)
        for kind,source in [('last',trainer.last),('best',trainer.best)]:
            source=Path(source)
            if source.is_file():
                target=directory/f'epoch_{epoch:03d}_{kind}.pt'
                temporary=target.with_suffix('.pt.tmp');shutil.copy2(source,temporary);temporary.replace(target)
                recovery['files'][kind]=dict(path=str(target),sha256=sha(target))
    write_json(state,dict(state='training',epoch=epoch,epochs=epochs,recovery=recovery,
                          metrics={k:float(v) for k,v in trainer.metrics.items()}))
    return recovery


def orientation_metrics(result):
    matrix=result.confusion_matrix.matrix
    matched=matrix[:2,:2]
    total=float(matched.sum())
    return dict(confusion_matrix=matrix.tolist(),matched_boxes=int(total),
                matched_orientation_accuracy=float(matched.trace()/total) if total else None,
                note='Conditional on matched detections; missed/spurious detections remain in the full matrix.')


def crop_audit(best, dataset, manifest, work):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    out=work/'evaluation/crop_audit';out.mkdir(parents=True,exist_ok=True)
    audit_split='test' if manifest['splits'].get('test') else 'val'
    images=[dataset/'images'/audit_split/(r['id']+'.jpg') for r in manifest['records'] if r['split']==audit_split and r['real']][:12]
    crops=[];records=[]
    for path in images:
        result=best.predict(str(path),imgsz=640,conf=.25,device=0,verbose=False)[0]
        result.save(filename=str(out/(path.stem+'_detections.jpg')))
        if result.obb is None:continue
        pixels=np.asarray(Image.open(path).convert('RGB'))
        for index,(points,cls) in enumerate(zip(result.obb.xyxyxyxy.cpu().numpy(),result.obb.cls.cpu().numpy())):
            if len(crops)>=64:break
            width=max(2,round(float(np.linalg.norm(points[1]-points[0]))))
            height=max(2,round(float(np.linalg.norm(points[2]-points[1]))))
            destination=np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],dtype=np.float32)
            transform=cv2.getPerspectiveTransform(points.astype(np.float32),destination)
            crop=Image.fromarray(cv2.warpPerspective(pixels,transform,(width,height)))
            if crop.height>crop.width:crop=crop.rotate(90,expand=True)
            filename=f'{path.stem}_{index:03d}.jpg';crop.save(out/filename)
            crops.append(crop);records.append(dict(image=path.name,crop=filename,cls=int(cls),points=points.tolist()))
    if crops:
        sheet=Image.new('RGB',(8*200,math.ceil(len(crops)/8)*100),'#eeeeee');draw=ImageDraw.Draw(sheet)
        for index,crop in enumerate(crops):
            thumb=crop.copy();thumb.thumbnail((194,74));x=(index%8)*200;y=(index//8)*100
            sheet.paste(thumb,(x+3,y+3));draw.text((x+3,y+80),f'{index+1}: class {records[index]["cls"]}',fill='black')
        sheet.save(work/'evaluation/crop_audit_grid.jpg')
    write_json(out/'index.json',dict(crops=records,reviewed=False,
        note='Inspect full crops for clipping. These are predicted samples, not verified CTC labels.'))


def run(archive, expected_sha, work, epochs=100):
    work=Path(work); work.mkdir(parents=True,exist_ok=True)
    state=work/'status.json'
    write_json(state,dict(state='verifying',epochs=epochs))
    if sha(archive)!=expected_sha: raise ValueError('Uploaded archive checksum mismatch')
    dataset=work/'dataset'
    if dataset.exists(): raise ValueError('Use a fresh work directory; refusing to overwrite a run')
    dataset.mkdir()
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target=(dataset/item.filename).resolve()
            if not target.is_relative_to(dataset.resolve()): raise ValueError('Unsafe archive member')
        z.extractall(dataset)
    manifest=json.loads((dataset/'training_manifest.json').read_text())
    if manifest.get('reviewed_only') is not True or manifest['classes'] != ['dish_horizontal','dish_vertical']:
        raise ValueError('Expected reviewed-only, two-class snapshot')
    for row in manifest['records']:
        image=dataset/'images'/row['split']/(row['id']+'.jpg')
        label=dataset/'labels'/row['split']/(row['id']+'.txt')
        annotation=dataset/'annotations'/row['split']/(row['id']+'.json')
        doc=json.loads(annotation.read_text())
        if doc.get('flags',{}).get('reviewed') is not True: raise ValueError('Unreviewed image in upload')
        if sha(image)!=row['training_sha256'] or sha(label)!=row['label_sha256']: raise ValueError('Snapshot checksum mismatch')
        for line in label.read_text().splitlines():
            parts=line.split()
            if len(parts)!=9 or parts[0] not in ['0','1']: raise ValueError('Expected two-class OBB labels')
    import torch
    import ultralytics
    from ultralytics import YOLO
    if not torch.cuda.is_available(): raise RuntimeError('GPU is required; refusing slow CPU training')
    import yaml
    config=yaml.safe_load((dataset/'dataset.yaml').read_text())
    config['path']=str(dataset.resolve())
    (dataset/'dataset.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    settings=dict(data=str(dataset/'dataset.yaml'),epochs=epochs,imgsz=640,batch=16,
                  optimizer='AdamW',lr0=.001,cos_lr=True,degrees=15.0,
                  perspective=.0005,mosaic=.5,fliplr=0.0,flipud=0.0,
                  seed=manifest['seed'],device=0,workers=2,amp=True,patience=100,
                  project=str(work/'runs'),name='menu_obb',exist_ok=False,
                  save=True,save_period=10,plots=True,verbose=False)
    # OBB orientation classes must never be silently turned through 90 degrees.
    settings['erasing']=0.0
    write_json(work/'run_config.json',dict(model='yolo11n-obb.pt',classes=manifest['classes'],
        ultralytics=ultralytics.__version__,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
        archive_sha256=expected_sha,settings=settings))
    write_json(state,dict(state='training',epochs=epochs,gpu=torch.cuda.get_device_name(0)))
    model=YOLO('yolo11n-obb.pt')
    recovery={}
    def save_progress(trainer):
        nonlocal recovery
        recovery=publish_recovery(trainer,work,state,epochs,recovery)
    model.add_callback('on_model_save',save_progress)
    # Ultralytics AMP self-check loads an unrelated detector on some versions.
    # Use standard AMP support, letting Ultralytics perform its documented check.
    model.train(**settings)
    weights=work/'runs/menu_obb/weights/best.pt'
    if not weights.exists(): raise RuntimeError('Best checkpoint not produced')
    checkpoint=work/'yolo11n_menu_obb_best.pt'; shutil.copy2(weights,checkpoint)
    write_json(state,dict(state='evaluating',epochs=epochs,recovery=recovery))
    best=YOLO(str(checkpoint))
    metrics={}; orientations={}
    for split in [s for s in ['val','test'] if manifest['splits'].get(s)]:
        result=best.val(data=str(dataset/'dataset.yaml'),split=split,imgsz=640,batch=16,
                        device=0,workers=2,plots=True,project=str(work/'evaluation'),name=split)
        metrics[split]={k:float(v) for k,v in result.results_dict.items()}
        orientations[split]=orientation_metrics(result)
    if manifest.get('real_test_images',0):
        real_config=dict(config,test='real_test.txt')
        (dataset/'real_dataset.yaml').write_text(yaml.safe_dump(real_config,sort_keys=False))
        result=best.val(data=str(dataset/'real_dataset.yaml'),split='test',imgsz=640,batch=16,
                        device=0,workers=2,plots=True,project=str(work/'evaluation'),name='real_test')
        metrics['real_test']={k:float(v) for k,v in result.results_dict.items()}
        orientations['real_test']=orientation_metrics(result)
    if manifest.get('real_val_images',0):
        real_config=dict(config,val='real_val.txt')
        (dataset/'real_validation.yaml').write_text(yaml.safe_dump(real_config,sort_keys=False))
        result=best.val(data=str(dataset/'real_validation.yaml'),split='val',imgsz=640,batch=16,
                        device=0,workers=2,plots=True,project=str(work/'evaluation'),name='real_val')
        metrics['real_val']={k:float(v) for k,v in result.results_dict.items()}
        orientations['real_val']=orientation_metrics(result)
    crop_audit(best,dataset,manifest,work)
    evaluation_source='real_test' if 'real_test' in metrics else 'real_val'
    real=metrics.get(evaluation_source,{})
    report=dict(metrics=metrics,orientation=orientations,accuracy_gate=dict(
        map50_target=.90,map50_95_target=.75,
        matched_orientation_target=.98,
        evaluation_source=evaluation_source,
        real_orientation_passed=(orientations.get(evaluation_source,{}).get('matched_orientation_accuracy') or 0)>=.98,
        real_detection_targets_passed=real.get('metrics/mAP50(B)',0)>=.90 and real.get('metrics/mAP50-95(B)',0)>=.75),
        checkpoint_sha256=sha(checkpoint),reviewed_images=manifest['images'],
        note='Two orientation classes. Validation is used for model selection; it is not an independent test holdout. Crop clipping still requires visual audit.')
    write_json(work/'evaluation_report.json',report)
    # Preserve training provenance and all diagnostics without duplicating dataset pixels.
    shutil.copy2(dataset/'training_manifest.json',work/'training_manifest.json')
    completed=dict(state='completed',checkpoint=str(checkpoint),metrics=metrics)
    with zipfile.ZipFile(work/'training_results.zip','w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for path in sorted(work.rglob('*')):
            if (path.is_file() and 'dataset' not in path.relative_to(work).parts
                and path.name not in ['training_results.zip','status.json'] and 'recovery' not in path.relative_to(work).parts and not path.name.startswith('epoch')):
                z.write(path,path.relative_to(work).as_posix())
        z.writestr('status.json',json.dumps(completed,indent=2))
    write_json(state,completed)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--work',default='/content/menu_obb_training')
    parser.add_argument('--epochs',type=int,default=100)
    args=parser.parse_args()
    try:
        run(args.archive,args.sha256,args.work,args.epochs)
    except Exception:
        work=Path(args.work);work.mkdir(parents=True,exist_ok=True)
        write_json(work/'status.json',dict(state='failed',error=traceback.format_exc()))
        raise
