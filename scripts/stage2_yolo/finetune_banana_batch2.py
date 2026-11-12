"""Fine-tune on latest051-100annotations;validate jointly on new andpreviousmenus."""
import argparse
from collections import Counter,defaultdict
import copy
from datetime import datetime,timezone
from pathlib import Path
import shutil
import traceback

from src.common import dump_json,digest
from src.detection.common import read_json,validate_shapes,yolo_text
from scripts.stage2_yolo.finetune_banana_latest import runtime,metrics,NAMES,orientation,ROOT

REVIEW=ROOT/'data/annotations/gemini_banana_menus_051_100'
OLD=ROOT/'data/datasets/detection/banana50_latest_v3'
DATA=ROOT/'data/datasets/detection/banana51_100_v4'
WORK=ROOT/'outputs/detection/training/banana51_100_finetune_v4'
START=ROOT/'outputs/detection/training/banana50_finetune_v3/yolo11n_menu_obb_banana50_best.pt'


def snapshot():
    assert not DATA.exists(),'Freshdatasetdirectoryrequired'
    manifest=read_json(REVIEW/'manifest.json');old=read_json(OLD/'training_manifest.json');docs={};revisions={}
    assert len(manifest['records'])==50
    for r in manifest['records']:
        ident=r['id'];p=REVIEW/'labels'/(ident+'.json');doc=read_json(p)
        assert doc.get('flags',{}).get('reviewed') is True and r['role']=='page'
        assert doc['source_id']==ident and doc['image_sha256']==r['sha256']
        assert (doc['imageWidth'],doc['imageHeight'])==(r['width'],r['height'])
        validate_shapes(doc['shapes'],r['width'],r['height'])
        assert doc['shapes'] or doc['flags'].get('confirmed_empty') is True
        assert p.with_suffix('.txt').read_text()==yolo_text(doc['shapes'],r['width'],r['height'])
        assert digest(REVIEW/'images'/r['image'])==r['sha256']
        docs[ident]=copy.deepcopy(doc);revisions[ident]=digest(p)
    assert set(docs)=={f'menu-{i:03d}' for i in range(51,101)}
    # Evenlyspaced60/70/80/90/100containnovERTICALboxes. Reserve86instead90
    # so the new-onlyvalidation diagnoses both orientationclasses.
    newval={f'menu-{i:03d}' for i in (60,70,80,86,100)};DATA.mkdir(parents=True)
    records=[];classes=defaultdict(Counter);group_ids=defaultdict(list)
    def emit(ident,split,group,image,doc,annotation_sha,provenance):
        target=DATA/'images'/split/(ident+'.jpg');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(image,target)
        for s in doc['shapes']:
            if s['label']=='price_tag':s['label']=orientation(s['points'])[0]
            assert s['label'] in NAMES
        text=yolo_text(doc['shapes'],doc['imageWidth'],doc['imageHeight'])
        label=DATA/'labels'/split/(ident+'.txt');label.parent.mkdir(parents=True,exist_ok=True);label.write_text(text,encoding='utf-8')
        doc['imagePath']=f'../../images/{split}/{ident}.jpg';dump_json(DATA/'annotations'/split/(ident+'.json'),doc)
        dump_json(DATA/'provenance'/(ident+'.json'),provenance)
        records.append(dict(id=ident,split=split,validation_group=group if split=='val' else None,annotation_source=group,
            source_menu=ident,image_sha256=digest(target),label_sha256=digest(label),reviewed_annotation_sha256=annotation_sha,boxes=len(doc['shapes'])))
        classes[split].update(s['label'] for s in doc['shapes'])
        if split=='val':group_ids[group].append(ident)
    for r in manifest['records']:
        ident=r['id'];emit(ident,'val' if ident in newval else 'train','latest',REVIEW/'images'/r['image'],docs[ident],revisions[ident],r)
        assert digest(REVIEW/'labels'/(ident+'.json'))==revisions[ident]
    previous=[r for r in old['records'] if r['split']=='val'];assert len(previous)==5
    for r in previous:
        ident=r['id'];im=OLD/'images/val'/(ident+'.jpg');label=OLD/'labels/val'/(ident+'.txt');doc=read_json(OLD/'annotations/val'/(ident+'.json'))
        assert digest(im)==r['image_sha256'] and digest(label)==r['label_sha256'] and doc['flags']['reviewed'] is True
        emit(ident,'val','previous',im,doc,r['reviewed_annotation_sha256'],r)
    assert Counter(r['split'] for r in records)=={'train':45,'val':10}
    assert {r['id'] for r in records if r['annotation_source']=='previous'}=={'menu-010','menu-020','menu-030','menu-040','menu-050'}
    assert not {r['image_sha256'] for r in records if r['split']=='train'}&{r['image_sha256'] for r in records if r['split']=='val'}
    header=f'path: {DATA.as_posix()}\ntrain: images/train\n'
    names='names:\n  0: dish_horizontal\n  1: dish_vertical\n'
    (DATA/'dataset.yaml').write_text(header+'val: images/val\n'+names,encoding='utf-8')
    for group,ids in group_ids.items():
        (DATA/(group+'_val.txt')).write_text(''.join(f'./images/val/{i}.jpg\n' for i in sorted(ids)),encoding='utf-8')
        (DATA/('dataset_'+group+'.yaml')).write_text(header+f'val: {group}_val.txt\n'+names,encoding='utf-8')
    result=dict(reviewed_only=True,classes=NAMES,latest_review_root=str(REVIEW),images=55,latest_reviewed_pages=50,
        splits={'train':45,'val':10},validation_groups={k:sorted(v) for k,v in group_ids.items()},records=records,
        class_counts={k:dict(v) for k,v in classes.items()},previous_training_images_included=0,previous_validation_images=5,
        starting_checkpoint=str(START),starting_sha256=digest(START),created_utc=datetime.now(timezone.utc).isoformat(),
        note='Trainonlylatest45menus051-100;fiveheldoutlatestplusfivepreviousvalmenusformjointvalidation. Groupmetricsreportedseparately. Valusedforselection,notindependenttest.')
    dump_json(DATA/'training_manifest.json',result);print(result['splits'],result['class_counts'],flush=True)


def train():
    WORK.mkdir(parents=True,exist_ok=True);state=WORK/'status.json'
    try:
        m=read_json(DATA/'training_manifest.json');assert m['previous_training_images_included']==0 and m['splits']=={'train':45,'val':10}
        for r in m['records']:
            assert digest(DATA/'images'/r['split']/(r['id']+'.jpg'))==r['image_sha256']
            assert digest(DATA/'labels'/r['split']/(r['id']+'.txt'))==r['label_sha256']
        source_sha=digest(START);assert source_sha==m['starting_sha256'];torch,YOLO=runtime();torch.set_num_threads(4)
        def evaluate_groups(path,phase):
            model=YOLO(str(path));out={}
            for group in ('combined','latest','previous'):
                data=DATA/('dataset.yaml' if group=='combined' else f'dataset_{group}.yaml')
                result=model.val(data=str(data),imgsz=640,batch=8,device=0,workers=0,plots=False,verbose=False,
                    project=str(WORK/'evaluation'),name=phase+'_'+group)
                out[group]=metrics(result)
            dump_json(WORK/(phase+'_metrics.json'),out);return out
        dump_json(state,dict(state='baseline_validation',train_images=45,val_images=10))
        baseline=evaluate_groups(START,'baseline')
        settings=read_json(ROOT/'outputs/detection/training/banana50_finetune_v3/run_config.json')['settings']
        settings.update(data=str(DATA/'dataset.yaml'),project=str(WORK/'runs'),name='menu_obb',exist_ok=False,seed=20261116)
        dump_json(WORK/'run_config.json',dict(start=str(START),source_sha256=source_sha,dataset_manifest_sha256=digest(DATA/'training_manifest.json'),
            settings=settings,checkpoint_selection='Combined10menuvalidationmAP50-95;separatelatestandpreviousbaseline/finalmetrics',
            older_training_images=0,previous_validation_images=5))
        model=YOLO(str(START));assert list(model.names.values())==NAMES
        def progress(t):dump_json(state,dict(state='training',epoch=int(t.epoch)+1,maximum_epochs=100,train_images=45,
            val_images=10,previous_validation_images=5,metrics={k:float(v) for k,v in t.metrics.items()}))
        model.add_callback('on_fit_epoch_end',progress);model.train(**settings)
        best=WORK/'runs/menu_obb/weights/best.pt';assert best.exists()
        candidate=WORK/'yolo11n_menu_obb_banana100_best.pt';shutil.copy2(best,candidate)
        dump_json(state,dict(state='final_validation'));final=evaluate_groups(candidate,'finetuned')
        assert digest(START)==source_sha
        report=dict(state='completed',train_images=45,val_images=10,latest_training_only=True,previous_validation_included=True,
            baseline=baseline,finetuned=final,candidate=str(candidate),candidate_sha256=digest(candidate),best=str(best),
            last=str(WORK/'runs/menu_obb/weights/last.pt'),source_unchanged=True,source_sha256=source_sha,
            note='Latest45menustraining;new5andprevious5validation. Combinedvalidationselectsbest;groupsnotindependenttests. No existingcheckpointoverwritten.')
        dump_json(WORK/'evaluation_report.json',report);dump_json(state,report);print(report,flush=True)
    except Exception:dump_json(state,dict(state='failed',error=traceback.format_exc()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    if args.prepare_only:snapshot()
    else:train()
