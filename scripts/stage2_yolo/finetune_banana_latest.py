"""Latest50human-reviewed menus only;preserve5menuholdout;localYOLO11nOBBfine-tuning."""
import argparse
from collections import Counter,defaultdict
import copy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

from PIL import Image
from src.common import digest,dump_json
from src.detection.common import annotation,read_json,validate_shapes,yolo_text
from src.detection.prepare_training import orientation,NAMES

ROOT=Path(__file__).resolve().parents[2]
REVIEW=ROOT/'data/annotations/gemini_banana_mobile_menus_50'
DATA=ROOT/'data/datasets/detection/banana50_latest_v3'
WORK=ROOT/'outputs/detection/training/banana50_finetune_v3'
WEIGHTS=ROOT/'checkpoints/yolo11n_menu_obb_best.pt'


def snapshot():
    if DATA.exists():raise ValueError('Refusing to overwrite existing latest-only snapshot')
    manifest=read_json(REVIEW/'manifest.json');eligible=[];frozen={};source_hashes={};annotation_hashes={}
    for r in manifest['records']:
        p=REVIEW/'labels'/(r['id']+'.json')
        if not p.exists():continue
        doc=read_json(p)
        if doc.get('flags',{}).get('reviewed') is not True:continue
        assert r['role']=='page' and doc['source_id']==r['id'] and doc['image_sha256']==r['sha256']
        assert (doc['imageWidth'],doc['imageHeight'])==(r['width'],r['height'])
        validate_shapes(doc['shapes'],r['width'],r['height'])
        assert doc['shapes'] or doc['flags'].get('confirmed_empty') is True
        assert p.with_suffix('.txt').read_text(encoding='utf-8')==yolo_text(doc['shapes'],r['width'],r['height'])
        assert digest(REVIEW/'images'/r['image'])==r['sha256']
        frozen[r['id']]=copy.deepcopy(doc);annotation_hashes[r['id']]=digest(p);source_hashes[r['id']]=r['sha256'];eligible.append(r)
    assert len(eligible)==50,'Expected all50latestreviewedpages;do not silently substitute olderannotations'
    holdout={'menu-010','menu-020','menu-030','menu-040','menu-050'}
    assert holdout<={r['id'] for r in eligible}
    images_by_split=defaultdict(set);entries=[];classes=defaultdict(Counter)
    DATA.mkdir(parents=True)
    for r in eligible:
        ident=r['id'];split='val' if ident in holdout else 'train';doc=frozen[ident]
        converted=[]
        for s in doc['shapes']:
            if s['label']=='price_tag':
                old=s['label'];s['label']=orientation(s['points'])[0];converted.append(dict(old=old,new=s['label']))
            assert s['label'] in NAMES
        image=DATA/'images'/split/(ident+'.jpg');image.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(REVIEW/'images'/r['image'],image)
        label=DATA/'labels'/split/(ident+'.txt');label.parent.mkdir(parents=True,exist_ok=True)
        label.write_text(yolo_text(doc['shapes'],r['width'],r['height']),encoding='utf-8')
        doc['imagePath']=f'../../images/{split}/{ident}.jpg';dump_json(DATA/'annotations'/split/(ident+'.json'),doc)
        dump_json(DATA/'provenance'/(ident+'.json'),r)
        assert digest(image)==r['sha256'];assert digest(REVIEW/'labels'/(ident+'.json'))==annotation_hashes[ident]
        images_by_split[split].add(r['sha256']);classes[split].update(s['label'] for s in doc['shapes'])
        entries.append(dict(id=ident,split=split,source_menu=ident,source_path=r['source_path'],image_sha256=digest(image),
                            reviewed_annotation_sha256=annotation_hashes[ident],label_sha256=digest(label),boxes=len(doc['shapes']),price_conversions=converted))
    assert not images_by_split['train']&images_by_split['val']
    (DATA/'dataset.yaml').write_text(f'path: {DATA.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: dish_horizontal\n  1: dish_vertical\n',encoding='utf-8')
    report=dict(reviewed_only=True,latest_annotation_root=str(REVIEW),classes=NAMES,images=50,
        splits=dict(Counter(r['split'] for r in entries)),class_counts={k:dict(v) for k,v in classes.items()},
        records=entries,source_menu_holdout=sorted(holdout),source_checkpoint=str(WEIGHTS),source_checkpoint_sha256=digest(WEIGHTS),
        source_annotation_sha256=annotation_hashes,created_utc=datetime.now(timezone.utc).isoformat(),
        older_dataset_images_included=0,proposals_used_as_ground_truth=False,duplicate_hash_overlap=0,
        note='All50pages fromlatesthumanreviewonly.45train/5val,sameheldoutmenusasStage3. Val selects checkpoints;not an independenttest. Humanannotationfiles are nevermodified.')
    dump_json(DATA/'training_manifest.json',report)
    return report


def runtime():
    # Use compatibleCUDA torch/torchvision from recognitionvenv;append missing
    # localdetectorpackages rather than shadowing torch with the old package tree.
    cache=ROOT/'.cache/ultralytics_latest_finetune';cache.mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(cache);os.environ['YOLO_OFFLINE']='true';os.environ['YOLO_AUTOINSTALL']='false'
    os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib_latest_finetune')
    font=Path('C:/Windows/Fonts/arial.ttf')
    if font.exists() and not (cache/'Arial.ttf').exists():shutil.copy2(font,cache/'Arial.ttf')
    import torch,torchvision
    sys.path.append(str(ROOT/'.stage2_packages'))
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS,USER_CONFIG_DIR
    if font.exists() and not (USER_CONFIG_DIR/'Arial.ttf').exists():shutil.copy2(font,USER_CONFIG_DIR/'Arial.ttf')
    SETTINGS.update({k:False for k in ('sync','clearml','comet','dvc','mlflow','raytune','tensorboard','wandb')})
    assert torch.cuda.is_available(),'GPUrequired;do not silently fall back toCPU'
    return torch,YOLO


def metrics(result):return {k:float(v) for k,v in result.results_dict.items()}


def train():
    WORK.mkdir(parents=True,exist_ok=True);state=WORK/'status.json'
    try:
        manifest=read_json(DATA/'training_manifest.json');assert manifest['splits']=={'train':45,'val':5} or manifest['splits']=={'val':5,'train':45}
        assert manifest['older_dataset_images_included']==0 and manifest['reviewed_only'] is True
        for r in manifest['records']:
            assert digest(DATA/'images'/r['split']/(r['id']+'.jpg'))==r['image_sha256']
            assert digest(DATA/'labels'/r['split']/(r['id']+'.txt'))==r['label_sha256']
        source_sha=digest(WEIGHTS);assert source_sha==manifest['source_checkpoint_sha256']
        torch,YOLO=runtime();torch.set_num_threads(4)
        model=YOLO(str(WEIGHTS));assert list(model.names.values())==NAMES
        dump_json(state,dict(state='baseline_validation',images=50,train=45,val=5))
        baseline=model.val(data=str(DATA/'dataset.yaml'),imgsz=640,batch=8,device=0,workers=0,
                           plots=False,project=str(WORK/'evaluation'),name='baseline',verbose=False)
        dump_json(WORK/'baseline_metrics.json',metrics(baseline))
        settings=dict(data=str(DATA/'dataset.yaml'),epochs=100,patience=20,imgsz=640,batch=8,nbs=16,
            optimizer='AdamW',lr0=.0001,lrf=.1,cos_lr=True,warmup_epochs=3.,warmup_bias_lr=.0001,
            degrees=5.,translate=.03,scale=.15,perspective=.0002,mosaic=.25,close_mosaic=10,
            fliplr=0.,flipud=0.,mixup=0.,copy_paste=0.,erasing=0.,hsv_h=.01,hsv_s=.15,hsv_v=.2,
            seed=20261115,device=0,workers=0,amp=False,save=True,save_period=10,plots=False,
            project=str(WORK/'runs'),name='menu_obb',exist_ok=False,verbose=False)
        dump_json(WORK/'run_config.json',dict(source_checkpoint=str(WEIGHTS),source_sha256=source_sha,
            dataset=str(DATA),dataset_manifest_sha256=digest(DATA/'training_manifest.json'),settings=settings,
            torch=torch.__version__,gpu=torch.cuda.get_device_name(0),latest_only=True,
            note='Freshfine-tuningoptimizerfromexistingYOLOweights;nottrainingfromscratch. FP32avoids unrelated AMPcheckmodeldownloads. Allruntime cacheslocal,offline,noinstall.'))
        def progress(trainer):
            dump_json(state,dict(state='training',epoch=int(trainer.epoch)+1,maximum_epochs=100,
                                metrics={k:float(v) for k,v in trainer.metrics.items()},latest_only=True,train_images=45,val_images=5))
        model.add_callback('on_fit_epoch_end',progress)
        model.train(**settings)
        best=WORK/'runs/menu_obb/weights/best.pt';last=WORK/'runs/menu_obb/weights/last.pt'
        assert best.exists() and last.exists()
        candidate=WORK/'yolo11n_menu_obb_banana50_best.pt';shutil.copy2(best,candidate)
        dump_json(state,dict(state='final_validation'))
        result=YOLO(str(candidate)).val(data=str(DATA/'dataset.yaml'),imgsz=640,batch=8,device=0,workers=0,
                                      plots=False,project=str(WORK/'evaluation'),name='finetuned',verbose=False)
        assert digest(WEIGHTS)==source_sha
        report=dict(state='completed',latest_only=True,train_images=45,val_images=5,older_dataset_images=0,
            baseline=metrics(baseline),finetuned=metrics(result),source_checkpoint_unchanged=True,source_sha256=source_sha,
            candidate=str(candidate),candidate_sha256=digest(candidate),best=str(best),last=str(last),
            note='Validationmenuusedforselection;not independenttest. No automatic replacement ofcheckpoints/yolo11n_menu_obb_best.pt. RecognitionV37data/checkpointsunchanged.')
        dump_json(WORK/'evaluation_report.json',report);dump_json(state,report)
        print(json.dumps(report),flush=True)
    except Exception:
        dump_json(state,dict(state='failed',error=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    if a.prepare_only:print(json.dumps(snapshot(),ensure_ascii=False))
    else:train()
