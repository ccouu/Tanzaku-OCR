"""Prepare a separate human OBB review queue for an inclusive menu-number range."""
import argparse
from collections import Counter
from pathlib import Path
import shutil
from PIL import Image
from src.common import digest,dump_json
from src.detection.common import annotation,save_pair,validate_shapes,read_json
from scripts.stage2_yolo.finetune_banana_latest import runtime,NAMES


def run(args):
    source=args.source.resolve();root=args.output.resolve();weights=args.weights.resolve()
    assert 1<=args.first<=args.last
    checksum=digest(weights);paths=[source/f'menu-{n:03d}.jpg' for n in range(args.first,args.last+1)]
    assert all(p.is_file() for p in paths),'Missingrequestedmenuimages'
    Torch,YOLO=runtime();Torch.set_num_threads(4);model=YOLO(str(weights));assert list(model.names.values())==NAMES
    provenance=dict(backend='YOLO11n-OBB',weights=str(weights),weights_sha256=checksum,imgsz=640,confidence=.25,nms_iou=.7,
        note='Recommendedv3detector;predictionsrequirehumanreview. Existingannotationcollectionsarenotmodified.')
    records=[];counts=Counter();invalid=[]
    for n,path in enumerate(paths,1):
        ident=path.stem;target=root/'images'/path.name;target.parent.mkdir(parents=True,exist_ok=True);image_sha=digest(path)
        if target.exists():assert digest(target)==image_sha
        else:shutil.copy2(path,target)
        with Image.open(target) as im:width,height=im.size
        side=path.with_suffix('.json')
        record=dict(id=ident,image=path.name,source_path=str(path),source_root=str(source),sha256=image_sha,width=width,height=height,
            role='page',source_kind='generated',source_menu=ident,source_info_path=str(side) if side.exists() else None,
            source_info_sha256=digest(side) if side.exists() else None,source_info=read_json(side) if side.exists() else None)
        out=root/'predictions'/(ident+'.json')
        if out.exists():
            doc=read_json(out);assert doc['image_sha256']==image_sha and doc['provenance']==provenance
        else:
            result=model.predict(str(path),imgsz=640,conf=.25,iou=.7,max_det=300,device=0,verbose=False)[0];shapes=[]
            if result.obb is not None:
                for j,(q,cls,conf) in enumerate(zip(result.obb.xyxyxyxy.cpu().tolist(),result.obb.cls.cpu().tolist(),result.obb.conf.cpu().tolist())):
                    shape=dict(label=NAMES[int(cls)],points=[[max(0.,min(width,float(x))),max(0.,min(height,float(y)))] for x,y in q],
                        group_id=None,shape_type='polygon',flags={'reviewed':False},confidence=float(conf),proposal_id=f'{ident}__box{j:03d}')
                    try:validate_shapes([shape],width,height)
                    except ValueError as exc:invalid.append(dict(id=ident,proposal=j,error=str(exc)));continue
                    shapes.append(shape)
            doc=annotation(record,shapes,provenance);save_pair(root/'predictions',record,doc)
        dump_json(root/'provenance'/(ident+'.json'),record);records.append(record);counts.update(s['label'] for s in doc['shapes'])
        print('PRELABEL',n,len(paths),ident,len(doc['shapes']),flush=True)
    assert digest(weights)==checksum
    dump_json(root/'manifest.json',dict(schema_version=1,copy_policy='Byte-identicalsourcepages,separatereviewqueue,priorannotationspreserved',records=records,counts={'page':len(paths)}))
    dump_json(root/'prediction_report.json',dict(images=len(paths),first_menu=args.first,last_menu=args.last,boxes=sum(counts.values()),classes=dict(counts),
        provenance=provenance,rejected_invalid_polygons=invalid,source_checkpoint_unchanged=True,training_started=False))
    print('READY',root,sum(counts.values()),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--first',type=int,required=True);p.add_argument('--last',type=int,required=True)
    p.add_argument('--source',type=Path,default=Path('data/raw/gemini_banana_mobile_menus_50'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--weights',type=Path,default=Path('outputs/detection/training/banana50_finetune_v3/yolo11n_menu_obb_banana50_best.pt'));run(p.parse_args())
