"""Build a reviewed-only two-class snapshot, with transitive provenance grouping."""
import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import shutil
import zipfile

from PIL import Image
from src.common.paths import DATASET_V1, resolve_path, source_kind
from src.common import digest, dump_json
from .common import DEFAULT_ROOT, read_json, save_pair, validate_shapes, yolo_text

NAMES = ['dish_horizontal', 'dish_vertical']


def orientation(points):
    """Compare opposite edge means, preserving a marked rectangle's local axes."""
    lengths = [math.dist(points[i], points[(i+1) % 4]) for i in range(4)]
    width, height = (lengths[0]+lengths[2])/2, (lengths[1]+lengths[3])/2
    return NAMES[int(height > width)], width, height


def group_keys(record):
    info = record.get('source_info') or {}
    keys = {'pixels:'+record['sha256']}
    for field in ['source_group', 'source_menu_id']:
        if info.get(field):
            keys.add(field+':'+str(info[field]))
    for field in ['restaurant_id', 'background_group']:
        if info.get(field):
            keys.add('restaurant:'+str(info[field]))
    for field in ['source_image_sha256', 'background_sha256']:
        if info.get(field):
            keys.add('pixels:'+info[field])
    return keys


def connected_groups(records):
    parent = {r['id']:r['id'] for r in records}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    owners = {}
    for record in records:
        for key in group_keys(record):
            ident = record['id']
            if key in owners:
                parent[find(ident)] = find(owners[key])
            else:
                owners[key] = ident
    components = defaultdict(list)
    for record in records:
        components[find(record['id'])].append(record)
    return list(components.values())


def grouped_split(records, seed=20261108):
    components = connected_groups(records)
    if len(components) < 3:
        raise ValueError('Fewer than three independent source components; cannot make a leakage-free split')
    groups = sorted(components, key=lambda g: (-len(g),hashlib.sha256((str(seed)+min(r['id'] for r in g)).encode()).hexdigest()))
    targets = {'train':.8*len(records),'val':.1*len(records),'test':.1*len(records)}
    counts = Counter()
    assignments = {}
    for group in groups:
        # Largest remaining absolute deficit; deterministic and keeps connected groups intact.
        split = max(targets, key=lambda s:targets[s]-counts[s])
        for record in group:
            assignments[record['id']] = split
        counts[split] += len(group)
    if any(counts[s] == 0 for s in targets):
        raise ValueError('Provenance groups are too large to form all three splits')
    return assignments, [sorted(r['id'] for r in group) for group in groups]


def source_ratio_split(records, seed=20261108):
    """Solve exact per-source quotas while keeping whole connected groups together."""
    import numpy as np
    from scipy.optimize import milp, Bounds, LinearConstraint
    groups=sorted(connected_groups(records),key=lambda g:min(r['id'] for r in g))
    kinds=['real','generated']
    matrix=np.array([[sum((source_kind(r)=='generated')==(kind=='generated') for r in group) for group in groups] for kind in kinds],dtype=float)
    totals=matrix.sum(axis=1)
    targets=np.array([math.floor(n*(.1 if kind=='generated' else .5)+.5) for kind,n in zip(kinds,totals)])
    costs=np.array([int(hashlib.sha256((str(seed)+min(r['id'] for r in g)).encode()).hexdigest()[:8],16)/2**32 for g in groups])
    result=milp(costs,integrality=np.ones(len(groups)),bounds=Bounds(0,1),
                constraints=LinearConstraint(matrix,targets,targets),options={'time_limit':60})
    if result.x is None:
        raise ValueError('Requested source quotas cannot be met with disjoint source groups: '+result.message)
    chosen=np.rint(result.x).astype(int)
    if not np.array_equal(matrix@chosen,targets):raise ValueError('Source quota solver returned invalid counts')
    assignments={r['id']:('val' if selected else 'train') for group,selected in zip(groups,chosen) for r in group}
    if set(assignments.values())!={'train','val'}:raise ValueError('Both training and validation must be nonempty')
    return assignments,[sorted(r['id'] for r in group) for group in groups]


def prepare(root=DEFAULT_ROOT, output=DATASET_V1, seed=20261108, split_policy='grouped_80_10_10'):
    root, output = resolve_path(root), resolve_path(output)
    if output.exists():
        raise ValueError('Use a fresh output directory to preserve existing training snapshots')
    manifest = read_json(root/'manifest.json')
    selected, documents, conversions, pending = [], {}, [], []
    for record in manifest['records']:
        path = root/'labels'/(record['id']+'.json')
        if not path.exists():
            continue  # Predictions are NEVER considered training labels.
        doc = read_json(path)
        if doc.get('flags',{}).get('reviewed') is not True:
            continue
        if record['role'] != 'page':
            raise ValueError(f"Reviewed non-page: {record['id']}")
        if doc.get('image_sha256') != record['sha256'] or doc.get('source_id') != record['id']:
            raise ValueError(f'Annotation identity mismatch: {path}')
        if (doc.get('imageWidth'),doc.get('imageHeight')) != (record['width'],record['height']):
            raise ValueError(f'Annotation dimensions mismatch: {path}')
        validate_shapes(doc['shapes'],record['width'],record['height'])
        if not doc['shapes'] and doc.get('flags',{}).get('confirmed_empty') is not True:
            raise ValueError(f'Unconfirmed empty negative: {path}')
        if path.with_suffix('.txt').read_text() != yolo_text(doc['shapes'],record['width'],record['height']):
            raise ValueError(f'JSON/text mismatch: {path}')
        if digest(root/'images'/record['image']) != record['sha256']:
            raise ValueError(f'Image checksum mismatch: {path}')
        converted = copy.deepcopy(doc)
        for index, shape in enumerate(converted['shapes']):
            if shape['label'].lower() in ('price','price_tag'):
                label, width, height = orientation(shape['points'])
                conversions.append(dict(id=record['id'],shape=index,old_label=shape['label'],new_label=label,width=width,height=height))
                shape['label'] = label
                shape['reclassification'] = dict(previous_label='price_tag',rule='mean local height > mean local width => vertical; otherwise horizontal')
        if any(s['label'] not in NAMES for s in converted['shapes']):
            raise ValueError('Unsupported training class')
        if converted != doc:
            converted['price_relabelled_at_utc'] = datetime.now(timezone.utc).isoformat()
            pending.append((record,path,converted))
        selected.append(record)
        documents[record['id']] = converted
    if not selected:
        raise ValueError('No explicitly reviewed page labels')
    assignments, groups = (source_ratio_split if split_policy=='source_ratios' else grouped_split)(selected, seed)
    # Validate the complete snapshot before changing any human annotation files.
    output.mkdir(parents=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = root/'label_backups'/('before_price_relabel_'+stamp)
    for record,path,doc in pending:
        backup.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,backup/path.name)
        shutil.copy2(path.with_suffix('.txt'),backup/path.with_suffix('.txt').name)
        save_pair(root/'labels',record,doc)
    entries, counts, class_counts = [], Counter(), defaultdict(Counter)
    for record in selected:
        ident = record['id']; split = assignments[ident]; doc = documents[ident]
        destination = output/'images'/split/(ident+'.jpg')
        destination.parent.mkdir(parents=True,exist_ok=True)
        with Image.open(root/'images'/record['image']) as image:
            rgba = image.convert('RGBA'); white = Image.new('RGBA',rgba.size,'white'); white.alpha_composite(rgba)
            white.convert('RGB').save(destination,quality=95,subsampling=0)
        labels = output/'labels'/split/(ident+'.txt')
        labels.parent.mkdir(parents=True,exist_ok=True)
        labels.write_text(yolo_text(doc['shapes'],record['width'],record['height']),encoding='utf-8')
        snapshot = copy.deepcopy(doc)
        snapshot['imagePath'] = f'../../images/{split}/{ident}.jpg'
        snapshot['original_image_sha256']=record['sha256']
        snapshot['image_sha256']=digest(destination)
        dump_json(output/'annotations'/split/(ident+'.json'),snapshot)
        info_path = output/'provenance'/(ident+'.json')
        dump_json(info_path,record)
        entry = dict(id=ident,split=split,original_sha256=record['sha256'],
                     training_sha256=digest(destination),source_path=record['source_path'],
                     reviewed_annotation_sha256=digest(root/'labels'/(ident+'.json')),
                     label_sha256=digest(labels),groups=sorted(group_keys(record)),
                     source_kind=source_kind(record),real=source_kind(record) != 'generated',boxes=len(doc['shapes']))
        entries.append(entry); counts[split]+=1
        class_counts[split].update(s['label'] for s in doc['shapes'])
    split_keys = {s:set().union(*(set(e['groups']) for e in entries if e['split']==s)) for s in ['train','val','test']}
    if any(split_keys[a]&split_keys[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Source leakage detected')
    (output/'dataset.yaml').write_text('path: .\ntrain: images/train\nval: images/val\n'+('test: images/test\n' if counts['test'] else '')+'names:\n  0: dish_horizontal\n  1: dish_vertical\n',encoding='utf-8')
    real_test = [f'./images/test/{e["id"]}.jpg' for e in entries if e['split']=='test' and e['real']]
    (output/'real_test.txt').write_text('\n'.join(real_test)+'\n',encoding='utf-8')
    real_val=[f'./images/val/{e["id"]}.jpg' for e in entries if e['split']=='val' and e['real']]
    (output/'real_val.txt').write_text('\n'.join(real_val)+'\n',encoding='utf-8')
    source_counts={kind:dict(Counter(e['split'] for e in entries if e['source_kind']==kind)) for kind in ['tabelog_menu','tabelog_interior','generated']}
    report = dict(schema_version=1,reviewed_only=True,classes=NAMES,seed=seed,
                  images=len(entries),excluded_unreviewed=len(manifest['records'])-len(entries),
                  splits=dict(counts),class_counts={s:dict(c) for s,c in class_counts.items()},
                  price_conversions=conversions,price_conversion_count=len(conversions),
                  backup_path=str(backup) if pending else None,
                  independent_components=len(groups),largest_component=max(map(len,groups)),
                  groups=groups,records=entries,real_test_images=len(real_test),real_val_images=len(real_val),
                  split_policy=split_policy,source_counts=source_counts,leakage_check='passed')
    dump_json(output/'training_manifest.json',report)
    archive = output/'dataset.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path != archive: z.write(path,path.relative_to(output).as_posix())
    dump_json(output/'archive_info.json',dict(path=str(archive),sha256=digest(archive),bytes=archive.stat().st_size))
    print({k:v for k,v in report.items() if k not in ['records','groups','price_conversions']},flush=True)
    print('Archive:',archive,'SHA256:',digest(archive),flush=True)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--output',type=Path,default=DATASET_V1)
    parser.add_argument('--seed',type=int,default=20261108)
    parser.add_argument('--split-policy',choices=['grouped_80_10_10','source_ratios'],default='grouped_80_10_10')
    args=parser.parse_args(); prepare(args.root,args.output,args.seed,args.split_policy)
