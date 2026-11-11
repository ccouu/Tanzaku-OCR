"""Offline validation for a downloaded real menu/interior photo collection."""
import argparse
import json
from pathlib import Path

from PIL import Image

from src.common import digest, dump_json, safe_path
from src.data.tabelog_collection import OUTPUT_DIR,PHOTO_KINDS,is_photo_url,profile_url


def verify(root=OUTPUT_DIR, target=200, report_path=None):
    root = Path(root).resolve()
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    kind = manifest.get('photo_kind','interior')
    if kind not in PHOTO_KINDS:
        raise ValueError('Unrecognized photo kind')
    rows = manifest['images']
    if len(rows)!=target or manifest['downloaded_images']!=target or manifest['unique_restaurants']!=target:
        raise ValueError('Collection has not reached the requested target')
    ids, hashes = set(), set()
    for row in rows:
        restaurant = profile_url(row['restaurant_url'])
        if not restaurant or restaurant[1]!=row['restaurant_id'] or not is_photo_url(row['image_url']):
            raise ValueError('Invalid restaurant or image provenance')
        if not row['gallery_url'].endswith(PHOTO_KINDS[kind]['subpath']) or row['category']!=PHOTO_KINDS[kind]['category']:
            raise ValueError('Photo came from the wrong gallery category')
        if kind=='menu' and (row.get('transcription_verified') is not False or row.get('text') is not None or row.get('annotations')!=[]):
            raise ValueError('Unreviewed real menus must not be assigned fabricated OCR labels')
        if row['restaurant_id'] in ids or row['sha256'] in hashes:
            raise ValueError('Duplicate restaurant or image content')
        path = safe_path(root,row['image'])
        sidecar = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if sidecar!=row or digest(path)!=row['sha256']:
            raise ValueError('Image checksum or sidecar/manifest mismatch')
        with Image.open(path) as image:
            if image.size!=(row['width'],row['height']) or min(image.size)<256:
                raise ValueError('Invalid photo dimensions')
            image.verify()
        ids.add(row['restaurant_id'])
        hashes.add(row['sha256'])
    sheets = json.loads((root/'contact_sheets.json').read_text())
    if sheets['images']!=target or len(sheets['pages'])!=(target+19)//20:
        raise ValueError('Contact sheet index does not match collection')
    for filename in sheets['pages']:
        with Image.open(safe_path(root,filename)) as image:
            image.verify()
    report = dict(status='passed',images=target,unique_restaurants=len(ids),unique_image_hashes=len(hashes),
        sidecars_verified=target,contact_sheet_pages=len(sheets['pages']),
        visually_reviewed_interiors=sum(r.get('interior_visual_review')=='passed' for r in rows),
        visually_reviewed_menus=sum(r.get('menu_visual_review')=='passed' for r in rows),
        photo_kind=kind,transcriptions_verified=sum(bool(r.get('transcription_verified')) for r in rows),
        negative_training_labels_assigned=False,source='tabelog')
    dump_json(report_path or root/'verification_report.json',report)
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=OUTPUT_DIR)
    parser.add_argument('--target',type=int,default=200)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    verify(args.data_root,args.target,args.report)
