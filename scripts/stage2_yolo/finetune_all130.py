"""All130latestreviewedmenus,retainedvalidation,best-including-baseline selection."""
import argparse
from collections import Counter,defaultdict
import copy
from pathlib import Path
import shutil
import traceback
from src.common import digest,dump_json
from src.detection.common import read_json,validate_shapes,yolo_text
from scripts.stage2_yolo.finetune_banana_latest import runtime,metrics,NAMES,ROOT

DATA=ROOT/'data/datasets/detection/banana130_v5'
WORK=ROOT/'outputs/detection/training/banana130_finetune_v5'
START=ROOT/'outputs/detection/training/banana50_finetune_v3/yolo11n_menu_obb_banana50_best.pt'
HOLDOUT={10,20,30,40,50,60,70,80,86,100,109,120,130}


def prepare():
    assert not DATA.exists(),'Preserveexistingdatasets'
    inputs=[]
    for folder,expected in (('gemini_banana_menus_001_100_rereview',range(1,101)),('gemini_banana_menus_101_130',range(101,131))):
        root=ROOT/'data/annotations'/folder;m=read_json(root/'manifest.json')
        assert {r['id'] for r in m['records']}=={f'menu-{n:03d}' for n in expected}
        for r in m['records']:
            path=root/'labels'/(r['id']+'.json');doc=read_json(path)
            assert doc['flags'].get('reviewed') is True and doc['source_id']==r['id'] and r['role']=='page'
            assert doc['image_sha256']==r['sha256'] and (doc['imageWidth'],doc['imageHeight'])==(r['width'],r['height'])
            validate_shapes(doc['shapes'],r['width'],r['height'])
            assert doc['shapes'] or doc['flags'].get('confirmed_empty') is True
            assert path.with_suffix('.txt').read_text()==yolo_text(doc['shapes'],r['width'],r['height'])
            assert digest(root/'images'/r['image'])==r['sha256']
            assert all(s['label'] in NAMES for s in doc['shapes'])
            inputs.append((root,r,doc,digest(path)))
    assert len(inputs)==130;DATA.mkdir(parents=True);records=[];groups=defaultdict(list);classes=defaultdict(Counter)
    for root,r,original,annotation_sha in inputs:
        ident=r['id'];n=int(ident.split('-')[1]);split='val' if n in HOLDOUT else 'train'
        group='first50' if n<=50 else 'second50' if n<=100 else 'new30'
        image=DATA/'images'/split/(ident+'.jpg');image.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/'images'/r['image'],image)
        doc=copy.deepcopy(original);doc['imagePath']=f'../../images/{split}/{ident}.jpg'
        label=DATA/'labels'/split/(ident+'.txt');label.parent.mkdir(parents=True,exist_ok=True);label.write_text(yolo_text(doc['shapes'],r['width'],r['height']),encoding='utf-8')
        dump_json(DATA/'annotations'/split/(ident+'.json'),doc);dump_json(DATA/'provenance'/(ident+'.json'),r)
        assert digest(root/'labels'/(ident+'.json'))==annotation_sha
        records.append(dict(id=ident,split=split,group=group,image_sha256=digest(image),label_sha256=digest(label),
            reviewed_annotation=str(root/'labels'/(ident+'.json')),annotation_sha256=annotation_sha,boxes=len(doc['shapes'])))
        classes[split].update(s['label'] for s in doc['shapes'])
        if split=='val':groups[group].append(ident)
    assert Counter(r['split'] for r in records)=={'train':117,'val':13}
    assert not {r['image_sha256'] for r in records if r['split']=='train'}&{r['image_sha256'] for r in records if r['split']=='val'}
    header=f'path: {DATA.as_posix()}\ntrain: images/train\n';names='names:\n  0: dish_horizontal\n  1: dish_vertical\n'
    (DATA/'dataset.yaml').write_text(header+'val: images/val\n'+names,encoding='utf-8')
    for group,ids in groups.items():
        (DATA/f'{group}.txt').write_text(''.join(f'./images/val/{i}.jpg\n' for i in sorted(ids)),encoding='utf-8')
        (DATA/f'dataset_{group}.yaml').write_text(header+f'val: {group}.txt\n'+names,encoding='utf-8')
    report=dict(reviewed_only=True,images=130,splits={'train':117,'val':13},classes=NAMES,records=records,validation_groups=dict(groups),
        class_counts={k:dict(v) for k,v in classes.items()},starting_checkpoint=str(START),source_sha256=digest(START),
        note='Latestre-reviewed1-100plus101-130. Existing10holdoutsretained,109/120/130newholdouts. All130inexperiment,117actualtraining. Valusedforselection,notindependenttest.')
    dump_json(DATA/'training_manifest.json',report);print(report['splits'],report['class_counts'],flush=True)


def train():
    WORK.mkdir(parents=True,exist_ok=True);state=WORK/'status.json'
    try:
        manifest=read_json(DATA/'training_manifest.json');source_sha=digest(START);assert source_sha==manifest['source_sha256']
        for r in manifest['records']:
            assert digest(DATA/'images'/r['split']/(r['id']+'.jpg'))==r['image_sha256']
            assert digest(DATA/'labels'/r['split']/(r['id']+'.txt'))==r['label_sha256']
        torch,YOLO=runtime();torch.set_num_threads(4)
        def evaluate(path,phase):
            model=YOLO(str(path));out={}
            for group in ('combined','first50','second50','new30'):
                yaml=DATA/('dataset.yaml' if group=='combined' else f'dataset_{group}.yaml')
                out[group]=metrics(model.val(data=str(yaml),imgsz=640,batch=8,device=0,workers=0,plots=False,verbose=False,
                    project=str(WORK/'evaluation'),name=phase+'_'+group))
            dump_json(WORK/f'{phase}_metrics.json',out);return out
        dump_json(state,dict(state='baseline_validation'));baseline=evaluate(START,'baseline')
        settings=read_json(ROOT/'outputs/detection/training/banana50_finetune_v3/run_config.json')['settings']
        settings.update(data=str(DATA/'dataset.yaml'),project=str(WORK/'runs'),name='menu_obb',seed=20261117,lr0=5e-5,warmup_bias_lr=5e-5)
        dump_json(WORK/'run_config.json',dict(settings=settings,source=str(START),source_sha256=source_sha,
            dataset_manifest_sha256=digest(DATA/'training_manifest.json'),selection='Besttrainedvsstartingv3oncombinedvalidationmAP50-95;baselinefallback'))
        model=YOLO(str(START))
        def progress(t):dump_json(state,dict(state='training',epoch=int(t.epoch)+1,maximum_epochs=100,train_pages=117,val_pages=13,
            metrics={k:float(v) for k,v in t.metrics.items()}))
        model.add_callback('on_fit_epoch_end',progress);model.train(**settings)
        best=WORK/'runs/menu_obb/weights/best.pt';candidate=WORK/'yolo11n_menu_obb_banana130_best.pt';shutil.copy2(best,candidate)
        dump_json(state,dict(state='final_validation'));finetuned=evaluate(candidate,'finetuned')
        improved=finetuned['combined']['metrics/mAP50-95(B)']>baseline['combined']['metrics/mAP50-95(B)']
        selected=candidate if improved else START
        assert digest(START)==source_sha
        report=dict(state='completed',baseline=baseline,finetuned=finetuned,candidate=str(candidate),candidate_sha256=digest(candidate),
            selected_checkpoint=str(selected),selected_sha256=digest(selected),selected_is_finetuned=improved,source_unchanged=True,
            train_pages=117,val_pages=13,holdout_ids=sorted(HOLDOUT),note='All130latestreviewedpages,baselinefallback. No rootcheckpointsreplaced. Validationforselectionnotindependenttest.')
        dump_json(WORK/'evaluation_report.json',report);dump_json(state,report);print(report,flush=True)
    except Exception:dump_json(state,dict(state='failed',error=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');args=p.parse_args();prepare() if args.prepare_only else train()
