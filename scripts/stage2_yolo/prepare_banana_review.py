"""Load checkpoint-matched cached YOLO OBBs into the existing manual editor."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import shutil
from PIL import Image
from src.common import dump_json,digest
from src.detection.common import annotation,save_pair,validate_shapes


def main():
    source=Path('data/raw/gemini_banana_mobile_menus_50').resolve()
    root=Path('data/annotations/gemini_banana_mobile_menus_50').resolve()
    handoff=json.loads((source/'yolo_crops/handoff_manifest.json').read_text(encoding='utf-8'))
    cache=Path(handoff['paths']['crop_root_directory'])/'crops_manifest.json'
    predictions=json.loads(cache.read_text(encoding='utf-8'))
    weights=Path('checkpoints/yolo11n_menu_obb_best.pt').resolve()
    assert digest(weights)==predictions['weights_sha256']==handoff['provenance']['detector_weights_sha256']
    assert Path(predictions['input_dir']).resolve()==source and predictions['total_images']==50
    hashes=json.loads(Path('outputs/recognition/training/v36_yolo_fullpasses2/sampling.json').read_text(encoding='utf-8'))['source_menu_sha256']
    grouped=defaultdict(list)
    for crop in predictions['crops']:grouped[crop['source_menu']].append(crop)
    assert len(grouped)==50 and sum(map(len,grouped.values()))==678
    records=[];total=0
    for ident,crops in sorted(grouped.items()):
        path=source/(ident+'.jpg');checksum=digest(path);assert checksum==hashes[ident]
        target=root/'images'/path.name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert digest(target)==checksum
        else:shutil.copy2(path,target)
        with Image.open(target) as image:width,height=image.size
        side=path.with_suffix('.json');metadata=json.loads(side.read_text(encoding='utf-8')) if side.exists() else None
        record=dict(id=ident,image=path.name,source_path=str(path),source_root=str(source),sha256=checksum,width=width,height=height,
                    role='page',source_kind='generated',source_info_path=str(side) if side.exists() else None,
                    source_info_sha256=digest(side) if side.exists() else None,source_info=metadata,
                    source_menu=ident,recognition_holdout=ident in handoff['split_policy']['holdout_menus'])
        shapes=[]
        for crop in crops:
            points=[[max(0.,min(float(width),float(x))),max(0.,min(float(height),float(y)))] for x,y in crop['obb_polygon']]
            shapes.append(dict(label=crop['detector_class'],points=points,group_id=None,shape_type='polygon',
                               flags={'reviewed':False},confidence=crop['confidence'],crop_id=crop['crop_id']))
        validate_shapes(shapes,width,height)
        provenance=dict(backend='YOLO11n-OBB',weights=str(weights),weights_sha256=digest(weights),
                        cached_predictions=str(cache),cached_predictions_sha256=digest(cache),
                        confidence=predictions['confidence'],imgsz=predictions['imgsz'],iou_dedup=predictions['iou_dedup'],
                        note='Original unpadded detector OBBs;all678 proposals retained,not onlyOCR-eligible boxes. Explicit human saves required.')
        prediction_path=root/'predictions'/(ident+'.json')
        if prediction_path.exists():
            previous=json.loads(prediction_path.read_text(encoding='utf-8'))
            assert previous['image_sha256']==checksum and previous['provenance']==provenance
        else:save_pair(root/'predictions',record,annotation(record,shapes,provenance))
        dump_json(root/'provenance'/(ident+'.json'),record);records.append(record);total+=len(shapes)
    dump_json(root/'manifest.json',dict(schema_version=1,records=records,counts=dict(Counter(r['role'] for r in records)),
                                      copy_policy='Byte-identical source pages;existing reviewed labels preserved'))
    dump_json(root/'prediction_report.json',dict(images=len(records),boxes=total,checkpoint=str(weights),checkpoint_sha256=digest(weights),
        all_proposals_unreviewed=True,reused_checkpoint_matched_cache=True,recognition_holdout_menus=handoff['split_policy']['holdout_menus'],
        note='Stage2manual review only. Do not merge detector fine-tuning splits without preserving source-menu holdout identities.'))
    print(json.dumps(dict(root=str(root),images=len(records),boxes=total)),flush=True)


if __name__=='__main__':main()
