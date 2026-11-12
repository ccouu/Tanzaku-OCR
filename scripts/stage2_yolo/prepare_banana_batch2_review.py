"""Prelabel exactlymenus051-100withlatestfine-tunedYOLO;separatemanualreviewroot."""
from collections import Counter
from pathlib import Path
import shutil
from PIL import Image
from src.common import digest,dump_json
from src.detection.common import annotation,save_pair,validate_shapes,read_json
from scripts.stage2_yolo.finetune_banana_latest import runtime,NAMES


def main():
    source=Path('data/raw/gemini_banana_mobile_menus_50').resolve()
    root=Path('data/annotations/gemini_banana_menus_051_100').resolve()
    weights=Path('outputs/detection/training/banana50_finetune_v3/yolo11n_menu_obb_banana50_best.pt').resolve()
    expected=read_json(weights.parent/'evaluation_report.json')['candidate_sha256'];assert digest(weights)==expected
    paths=[source/f'menu-{n:03d}.jpg' for n in range(51,101)]
    assert all(p.is_file() for p in paths),'All50requestedmenuimagesmustexist'
    Torch,YOLO=runtime();Torch.set_num_threads(4);model=YOLO(str(weights));assert list(model.names.values())==NAMES
    records=[];counts=Counter();invalid=[]
    provenance=dict(backend='YOLO11n-OBB',weights=str(weights),weights_sha256=expected,imgsz=640,confidence=.25,nms_iou=.7,
                    device='cuda:0',note='LatestBanana50fine-tunedbest,epoch36. Onlymenus051-100;predictionsunrevieweduntilhumanexplicitlysaves.')
    for index,path in enumerate(paths,1):
        ident=path.stem;target=root/'images'/path.name;target.parent.mkdir(parents=True,exist_ok=True)
        checksum=digest(path)
        if target.exists():assert digest(target)==checksum
        else:shutil.copy2(path,target)
        with Image.open(target) as image:width,height=image.size
        side=path.with_suffix('.json');metadata=read_json(side) if side.exists() else None
        record=dict(id=ident,image=path.name,source_path=str(path),source_root=str(source),sha256=checksum,width=width,height=height,
            role='page',source_kind='generated',source_menu=ident,source_info_path=str(side) if side.exists() else None,
            source_info_sha256=digest(side) if side.exists() else None,source_info=metadata)
        out=root/'predictions'/(ident+'.json')
        if out.exists():
            doc=read_json(out);assert doc['image_sha256']==checksum and doc['provenance']==provenance
        else:
            result=model.predict(str(path),imgsz=640,conf=.25,iou=.7,max_det=300,device=0,verbose=False)[0];shapes=[]
            if result.obb is not None:
                for j,(quad,cls,conf) in enumerate(zip(result.obb.xyxyxyxy.cpu().tolist(),result.obb.cls.cpu().tolist(),result.obb.conf.cpu().tolist())):
                    shape=dict(label=NAMES[int(cls)],points=[[max(0.,min(width,float(x))),max(0.,min(height,float(y)))] for x,y in quad],
                        group_id=None,shape_type='polygon',flags={'reviewed':False},confidence=float(conf),proposal_id=f'{ident}__box{j:03d}')
                    try:validate_shapes([shape],width,height)
                    except ValueError as exc:invalid.append(dict(id=ident,proposal=j,error=str(exc)));continue
                    shapes.append(shape)
            doc=annotation(record,shapes,provenance);save_pair(root/'predictions',record,doc)
        dump_json(root/'provenance'/(ident+'.json'),record);records.append(record);counts.update(s['label'] for s in doc['shapes'])
        print('PRELABEL',index,50,ident,len(doc['shapes']),flush=True)
    assert digest(weights)==expected and [r['id'] for r in records]==[f'menu-{n:03d}' for n in range(51,101)]
    dump_json(root/'manifest.json',dict(schema_version=1,copy_policy='Byte-identicalcopies;originalsmetadataandpriorreviewcollectionpreserved',records=records,counts={'page':50}))
    dump_json(root/'prediction_report.json',dict(images=50,first_menu=51,last_menu=100,boxes=sum(counts.values()),classes=dict(counts),
        provenance=provenance,rejected_invalid_polygons=invalid,source_checkpoint_unchanged=True,reviewed=False))
    print('READY',root,dict(counts),flush=True)


if __name__=='__main__':main()
