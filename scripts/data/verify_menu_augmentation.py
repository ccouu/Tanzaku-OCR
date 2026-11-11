"""Audit transformed geometry, preserved text, paired files and split isolation."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.common import ROOT, digest, dump_json, safe_path
from src.common.paths import resolve_path
from src.data.menu_augmentation import corners, transform_points, validate_polygon, project_menu_plane


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root=ROOT/'data/processed/augmentation/run_1100', report_path=None):
    root = resolve_path(root).resolve()
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    config = json.loads((root/'config.json').read_text(encoding='utf-8'))
    rows = manifest['samples']
    require(len(rows)==manifest['menu_images']==manifest['composite_images'], 'Augmentation counts differ')
    if config.get('requested_samples') is not None:
        require(len(rows)==config['requested_samples']==manifest['requested_samples'], 'Requested exact sample count was not met')
    require(manifest['yolo_labels_assigned'] is False, 'YOLO labels must remain deferred')
    groups, source_hashes, background_groups, background_hashes = [defaultdict(set) for _ in range(4)]
    info_cache, page_indexes, hashes, composite_cache = {}, {}, {}, {}
    expected_ctc = 0
    generated = 0
    annotation_count = 0
    poses,scales,used_backgrounds = [],[],set()
    def sha(path):
        path = Path(path)
        if path not in hashes:
            hashes[path] = digest(path)
        return hashes[path]
    for row in rows:
        split = row['split']
        groups[split].add(row['source_group'])
        background_groups[split].add(row['background_group'])
        pairs = {}
        for kind in ('menu','composite'):
            pair = row[kind]
            image_path, label_path = safe_path(root,pair['image']), safe_path(root,pair['label'])
            label = json.loads(label_path.read_text(encoding='utf-8'))
            pairs[kind] = label
            require(label['image']==pair['image'] and label['id']==row['id'] and label['split']==split, 'Image/label pairing mismatch')
            require(sha(image_path)==label['sha256'], 'Augmented image checksum mismatch')
            require(label['yolo_labels_assigned'] is False, 'Unexpected YOLO class assignment')
            with Image.open(image_path) as image:
                require(image.size==(config['size'],config['size']), 'Incorrect output dimensions')
                require(image.mode==('RGBA' if kind=='menu' else 'RGB'), 'Incorrect image channels')
                if kind=='menu':
                    alpha = np.asarray(image.getchannel('A'))
                    require(alpha.min()==0 and alpha.max()==255, 'Menu canvas must preserve transparent margins and opaque content')
                image.verify() if kind!='menu' else None
            source_path = resolve_path(ROOT/label['source_image']).resolve()
            info_path = resolve_path(ROOT/label['source_info']).resolve()
            require(sha(source_path)==label['source_image_sha256'], 'Source image changed')
            require(sha(info_path)==label['source_info_sha256'], 'Source metadata changed')
            source_hashes[split].add(label['source_image_sha256'])
            if info_path not in info_cache:
                info_cache[info_path] = json.loads(info_path.read_text(encoding='utf-8'))
                if isinstance(info_cache[info_path],list):
                    page_indexes[info_path] = {safe_path(info_path.parent.parent,p['image']):p for p in info_cache[info_path]}
            source_info = info_cache[info_path]
            if label['source_kind']=='known_synthetic_text':
                original = page_indexes[info_path][source_path]
                original_annotations = original['items']
                require(label['text_verified'] is True, 'Lost known synthetic transcription status')
                require(label['text']=='\n'.join(a['text'] for a in original_annotations), 'Source text changed')
            else:
                original_annotations = []
                require(label['intended_text']==source_info['intended_text'] and label['text_verified'] is False,
                        'Generated prompt text changed or was promoted to verified ground truth')
            require(len(label['annotations'])==len(original_annotations), 'Transformed line count changed')
            with Image.open(source_path) as image:
                source_size = image.size
            matrix = np.array(label['source_to_image'])
            expected_boundary = transform_points(corners(*source_size),matrix)
            require(np.allclose(expected_boundary,label['menu_boundary']), 'Menu boundary does not match transform')
            validate_polygon(expected_boundary,(config['size'],config['size']))
            for original, transformed in zip(original_annotations,label['annotations']):
                require(original['text']==transformed['text'], 'Transcription was altered')
                require(original['class_id']==transformed['source_class_id'], 'Source class provenance changed')
                require(transformed['orientation_flag']==int(original['class_id']==1), 'Vertical orientation changed')
                expected = transform_points(original['corners'], matrix)
                require(np.allclose(expected,transformed['corners'],atol=1e-5), 'Line corners were not transformed with pixels')
                validate_polygon(expected,(config['size'],config['size']))
        menu, scene = pairs['menu'], pairs['composite']
        if 'max_pitch' in config:
            pose = menu['pose']
            require(scene['pose']==pose, 'Menu/composite pose differs')
            require(0<=pose['pitch_magnitude_degrees']<=config['max_pitch'] and
                    abs(pose['pitch_degrees'])==pose['pitch_magnitude_degrees'], 'Pitch outside configured magnitude')
            require(abs(pose['roll_degrees'])<=config['max_roll'] and abs(pose['yaw_degrees'])<=config['max_yaw'], 'Roll or yaw outside configured range')
            direction = 'forward' if pose['pitch_degrees']>0 else 'backward' if pose['pitch_degrees']<0 else 'none'
            require(pose['pitch_direction']==direction, 'Pitch direction mismatch')
            require(pose['projection']=='pinhole_3d_plane' and pose['rotation_order']==['pitch_x','yaw_y','roll_z'], 'Incorrect pose projection convention')
            with Image.open(resolve_path(ROOT/menu['source_image']).resolve()) as original_image:
                original_size = original_image.size
            expected = project_menu_plane(original_size,pose['pitch_degrees'],pose['roll_degrees'],pose['yaw_degrees'])
            fit_fraction = min(.86,.96/config['max_scene_scale'])
            expected *= config['size']*fit_fraction/max(np.ptp(expected,axis=0))
            expected += config['size']/2-(expected.min(axis=0)+expected.max(axis=0))/2
            require(np.allclose(expected,menu['menu_boundary']), 'Recorded 3D pose does not reproduce menu geometry')
            actual_scale = scene['menu_scale']
            require(config['min_scene_scale']<=actual_scale<=config['max_scene_scale'], 'Menu scale outside configured range')
            placement = np.asarray(scene['menu_to_image'])
            require(np.allclose(placement[:2,:2],np.eye(2)*actual_scale), 'Menu scale was clamped or recorded incorrectly')
            require(scene['background_selection']=='seeded_uniform_within_split', 'Incorrect background selection policy')
            scales.append(actual_scale)
            poses.append(pose)
        require(np.allclose(np.array(scene['menu_to_image'])@np.array(menu['source_to_image']),scene['source_to_image']), 'Composite transform chain mismatch')
        require(scene['augmented_menu_image']==row['menu']['image'], 'Composite does not reference its augmented menu')
        bg = resolve_path(ROOT/scene['background_image']).resolve()
        require(sha(bg)==scene['background_sha256'], 'Background changed')
        require(manifest['background_splits'][scene['background_group']]==split, 'Background assigned to wrong split')
        background_hashes[split].add(scene['background_sha256'])
        used_backgrounds.add(scene['background_group'])
        composite_cache[scene['image']] = scene
        annotation_count += len(scene['annotations'])
        expected_ctc += len(scene['annotations']) if config['export_ctc'] else 0
        generated += not scene['text_verified']
    for collection in (groups,source_hashes,background_groups,background_hashes):
        splits = list(collection)
        for i,left in enumerate(splits):
            for right in splits[i+1:]:
                require(not collection[left]&collection[right], 'Source menu or restaurant background leaks across splits')
    lines = [json.loads(line) for line in (root/'ctc_lines/labels.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(lines)==expected_ctc==manifest['ctc_lines'], 'CTC line count mismatch')
    seen = set()
    for row in lines:
        require(row['image'] not in seen, 'Duplicate CTC path')
        seen.add(row['image'])
        label = json.loads(safe_path(root,row['label']).read_text(encoding='utf-8'))
        require(label==row, 'CTC JSONL and individual sidecar differ')
        require(row['text_verified'] is True and row['orientation_flag'] in (0,1), 'Unverified CTC label')
        scene = composite_cache[row['composite_image']]
        require(scene['text_verified'] is True and row['split']==scene['split'], 'Unverified or cross-split CTC source')
        require(any(a['text']==row['text'] and a['orientation_flag']==row['orientation_flag']
            and np.allclose(a['corners'],row['composite_polygon']) for a in scene['annotations']), 'CTC text/polygon not in source scene')
        path = safe_path(root,row['image'])
        require(sha(path)==row['sha256'], 'CTC checksum mismatch')
        with Image.open(path) as image:
            require(image.size==(384,64) and image.mode=='L', 'Incorrect CTC image shape or channels')
            require(np.asarray(image).std()>1, 'Blank CTC crop')
    require(not list(root.rglob('*.txt')), 'Unexpected YOLO text labels were created')
    report = dict(status='passed', menu_images=len(rows), composite_images=len(rows),
        transformed_text_regions=annotation_count, ctc_lines=len(lines),
        generated_menu_images_with_unverified_text=generated,
        splits=dict(Counter(r['split'] for r in rows)),
        split_disjoint_menu_groups=True, split_disjoint_restaurant_backgrounds=True,
        source_text_preserved=True, yolo_labels_assigned=False,
        background_menu_or_signage_review_pending=True)
    if poses:
        report.update(pitch_degrees=[min(p['pitch_degrees'] for p in poses),max(p['pitch_degrees'] for p in poses)],
            roll_degrees=[min(p['roll_degrees'] for p in poses),max(p['roll_degrees'] for p in poses)],
            yaw_degrees=[min(p['yaw_degrees'] for p in poses),max(p['yaw_degrees'] for p in poses)],
            menu_scale=[min(scales),max(scales)],samples_scaled_above_one=sum(s>1 for s in scales),
            unique_backgrounds_used=len(used_backgrounds))
    dump_json(report_path or root/'verification_report.json',report)
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=ROOT/'data/processed/augmentation/run_1100')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    verify(args.data_root, args.report)
