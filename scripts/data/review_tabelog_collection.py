"""Quarantine visually rejected Tabelog photos and persist venue exclusions."""
import argparse
import json
from pathlib import Path

from src.common import ROOT, dump_json, safe_path
from src.data.tabelog_collection import OUTPUT_DIR


def reject(root, rejections):
    root = Path(root).resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError('Review operations must remain inside the workspace')
    manifest_path = root/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    kind = manifest.get('photo_kind','interior')
    if kind not in ('interior','menu'):
        raise ValueError('Unrecognized collection category')
    records = {row['restaurant_id']:row for row in manifest['images']}
    if not set(rejections)<=records.keys():
        raise ValueError('Rejection references a restaurant not in the selected collection')
    for rid,reason in rejections.items():
        if not isinstance(reason,str) or not reason.strip():
            raise ValueError('Every rejected photo needs a review reason')
        row = records[rid]
        source = safe_path(root,row['image'])
        target = safe_path(root,'rejected/'+source.name)
        if not source.is_file() or target.exists():
            raise ValueError('Rejected image is missing or quarantine destination exists')
    for rid,reason in rejections.items():
        row = records[rid]
        source = safe_path(root,row['image'])
        target = safe_path(root,'rejected/'+source.name)
        target.parent.mkdir(parents=True,exist_ok=True)
        # Both absolute paths have been checked against the explicit dataset root.
        source.replace(target)
        source.with_suffix('.json').replace(target.with_suffix('.json'))
        row.update(reviewed=True,review_status='rejected_not_'+kind,rejection_reason=reason)
        row[kind+'_visual_review'] = 'failed'
        dump_json(target.with_suffix('.json'),row)
    selected = [r for r in manifest['images'] if r['restaurant_id'] not in rejections]
    manifest.update(images=selected,downloaded_images=len(selected),unique_restaurants=len(selected),
        status='partial',excluded_restaurants=sorted(set(manifest.get('excluded_restaurants',[]))|set(rejections)))
    dump_json(manifest_path,manifest)
    print(json.dumps({'quarantined':len(rejections),'retained':len(selected)}))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=OUTPUT_DIR)
    parser.add_argument('--rejections-json',type=Path,required=True,help='JSON object mapping restaurant IDs to reasons')
    args = parser.parse_args()
    reject(args.data_root,json.loads(args.rejections_json.read_text(encoding='utf-8')))
