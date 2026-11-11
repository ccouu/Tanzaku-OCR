"""Text-preserving menu geometry, restaurant compositing, and optional CTC crops.

Produces JSON text sidecars, not new YOLO class/label assignments. All projective
transforms are also applied to existing source line polygons.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from .common import ROOT, digest, dump_json, safe_path
from src.common.paths import resolve_path

SPLITS = ('train', 'validation', 'test')
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}


@dataclass(frozen=True)
class AugmentationConfig:
    size: int = 640
    variants: int = 1
    seed: int = 20261107
    max_pitch: float = 30.0
    max_roll: float = 10.0
    max_yaw: float = 10.0
    min_scene_scale: float = 0.50
    max_scene_scale: float = 1.20
    export_ctc: bool = True
    workers: int = 4

    def validate(self):
        if (self.size < 256 or self.variants < 1 or self.workers < 1
                or not 0 <= self.max_pitch <= 60 or not 0 <= self.max_roll <= 45
                or not 0 <= self.max_yaw <= 60
                or not .2 <= self.min_scene_scale <= self.max_scene_scale <= 2):
            raise ValueError('Invalid augmentation dimensions, angle, perspective, scale, or worker count')


def corners(width, height):
    return np.array([[0., 0.], [width, 0.], [width, height], [0., height]])


def homography(source, destination):
    """Return a 3x3 projective matrix mapping four source corners to destination."""
    source, destination = np.asarray(source, dtype=float), np.asarray(destination, dtype=float)
    if source.shape != (4, 2) or destination.shape != (4, 2):
        raise ValueError('Homography requires two four-corner polygons')
    rows, values = [], []
    for (x, y), (u, v) in zip(source, destination):
        rows.extend([[x,y,1,0,0,0,-u*x,-u*y], [0,0,0,x,y,1,-v*x,-v*y]])
        values.extend([u, v])
    matrix = np.append(np.linalg.solve(np.array(rows), np.array(values)), 1).reshape(3, 3)
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError('Degenerate perspective transform')
    return matrix


def transform_points(points, matrix):
    points = np.asarray(points, dtype=float)
    projected = np.column_stack((points, np.ones(len(points)))) @ np.asarray(matrix).T
    if np.any(np.abs(projected[:, 2]) < 1e-9):
        raise ValueError('Polygon intersects the projective horizon')
    result = projected[:, :2] / projected[:, 2:]
    if not np.isfinite(result).all():
        raise ValueError('Invalid transformed coordinates')
    return result


def validate_polygon(points, size):
    points = np.asarray(points, dtype=float)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError('Invalid four-corner polygon')
    width, height = size
    if np.any(points < -1e-5) or np.any(points > np.array([width, height]) + 1e-5):
        raise ValueError('Transform would clip source content')
    edges = np.roll(points, -1, axis=0) - points
    next_edges = np.roll(edges, -1, axis=0)
    cross = edges[:,0]*next_edges[:,1] - edges[:,1]*next_edges[:,0]
    if not (np.all(cross > 1e-6) or np.all(cross < -1e-6)):
        raise ValueError('Degenerate or self-intersecting polygon')


def warp_image(image, matrix, output_size):
    inverse = np.linalg.inv(matrix)
    inverse /= inverse[2,2]
    fill = (0,0,0,0) if image.mode == 'RGBA' else 255 if image.mode == 'L' else (255,255,255)
    return image.transform(output_size, Image.Transform.PERSPECTIVE, tuple(inverse.flat[:8]),
                           resample=Image.Resampling.BICUBIC, fillcolor=fill)


def project_menu_plane(source_size, pitch, roll, yaw, camera_distance=2.):
    """Pinhole-project a centered plane: pitch X, yaw Y, then roll Z.

    Image X points right and Y points down; positive Z points away from camera.
    Positive pitch brings the top edge forward; positive yaw brings the right
    edge forward; positive roll turns clockwise in image coordinates.
    """
    width, height = source_size
    x,y,z = map(math.radians,(pitch,yaw,roll))
    rx = np.array([[1,0,0],[0,math.cos(x),-math.sin(x)],[0,math.sin(x),math.cos(x)]])
    ry = np.array([[math.cos(y),0,math.sin(y)],[0,1,0],[-math.sin(y),0,math.cos(y)]])
    rz = np.array([[math.cos(z),-math.sin(z),0],[math.sin(z),math.cos(z),0],[0,0,1]])
    plane = np.column_stack(((corners(width,height)-[width/2,height/2])/max(width,height),np.zeros(4)))
    rotated = plane @ (rz@ry@rx).T
    depth = camera_distance+rotated[:,2]
    if np.any(depth<=0):
        raise ValueError('Menu plane is behind the camera')
    return rotated[:,:2]*camera_distance/depth[:,None]


def menu_transform(source_size, canvas_size, rng, max_pitch=30, max_roll=10, max_yaw=10,
                   max_scene_scale=1.2):
    """Use explicit 3D angles and reserve space for actual scene enlargement."""
    width,height = source_size
    magnitude = rng.uniform(0,max_pitch)
    pitch = magnitude*rng.choice((-1,1))
    roll,yaw = rng.uniform(-max_roll,max_roll),rng.uniform(-max_yaw,max_yaw)
    quad = project_menu_plane(source_size,pitch,roll,yaw)
    # A 1.2x scale applies to the visible menu, not just transparent padding.
    # Reserve enough margin up front rather than clamping the requested scale.
    fit_fraction = min(.86,.96/max_scene_scale)
    quad *= canvas_size*fit_fraction/max(np.ptp(quad,axis=0))
    quad += canvas_size/2-(quad.min(axis=0)+quad.max(axis=0))/2
    validate_polygon(quad, (canvas_size, canvas_size))
    pose = dict(pitch_degrees=pitch,pitch_magnitude_degrees=magnitude,
        pitch_direction='forward' if pitch>0 else 'backward' if pitch<0 else 'none',
        roll_degrees=roll,yaw_degrees=yaw,projection='pinhole_3d_plane',
        rotation_order=['pitch_x','yaw_y','roll_z'],camera_distance=2.,fit_fraction=fit_fraction)
    return homography(corners(width,height),quad),pose


def scene_transform(size, rng, min_scale=.5, max_scale=1.2, menu_boundary=None):
    scale = rng.uniform(min_scale, max_scale)
    boundary = np.asarray(menu_boundary if menu_boundary is not None else corners(size,size))
    lower,upper = (boundary*scale).min(axis=0),(boundary*scale).max(axis=0)
    margin = .01*size
    low,high = margin-lower,size-margin-upper
    if np.any(high<low-1e-6):
        raise ValueError('Menu does not fit the scene at the requested actual scale')
    left,top = [rng.uniform(float(a),float(b)) for a,b in zip(low,high)]
    # Only transparent margins may leave the frame; no menu content is cropped.
    return np.array([[scale,0,left],[0,scale,top],[0,0,1.]])


def transform_annotations(annotations, matrix, output_size):
    result = []
    for annotation in annotations:
        quad = transform_points(annotation['corners'], matrix)
        validate_polygon(quad, output_size)
        result.append({**annotation, 'corners': quad.tolist()})
    return result


def extract_ctc_line(image, polygon, orientation_flag):
    if orientation_flag not in (0, 1):
        raise ValueError('Invalid CTC orientation')
    size = (64, 384) if orientation_flag else (384, 64)
    matrix = homography(polygon, corners(*size))
    crop = warp_image(image.convert('L'), matrix, size)
    if orientation_flag:
        crop = crop.transpose(Image.Transpose.ROTATE_90)
    if crop.size != (384, 64):
        raise ValueError('CTC line normalization failed')
    return crop


def grouped_split(groups, seed):
    groups = sorted(set(groups))
    if len(groups) < 3:
        raise ValueError('At least three independent groups are needed for disjoint splits')
    random.Random(seed).shuffle(groups)
    heldout = max(1, round(len(groups)*.1))
    train_end, val_end = len(groups)-2*heldout, len(groups)-heldout
    return {group: 'train' if i < train_end else 'validation' if i < val_end else 'test'
            for i, group in enumerate(groups)}


def load_sources(menu_dir, yolo_root, seed):
    sources, skipped = [], []
    menu_dir, yolo_root = resolve_path(menu_dir).resolve(), resolve_path(yolo_root).resolve()
    generated = []
    for path in sorted(menu_dir.glob('menu-*.json')):
        info = json.loads(path.read_text(encoding='utf-8'))
        if info.get('status') != 'completed':
            skipped.append({'info': str(path), 'reason': 'Generation is not completed'})
            continue
        image = safe_path(menu_dir.parent, info['image'])
        if digest(image) != info['sha256']:
            raise ValueError(f'Source checksum mismatch: {image.name}')
        # Identical intended menu content stays together, across handwriting variants.
        group = hashlib.sha256(info['intended_text'].encode('utf-8')).hexdigest()
        generated.append(dict(id='genai-'+path.stem, image=image, info=path, group='genai-content-'+group,
                              annotations=[], intended_text=info['intended_text'], text_verified=False,
                              source_kind='generated_menu', info_sha256=digest(path)))
    if generated:
        assignments = grouped_split([s['group'] for s in generated], seed)
        for source in generated:
            source['split'] = assignments[source['group']]
        sources.extend(generated)
    pages_path = yolo_root/'pages.json'
    pages = json.loads(pages_path.read_text(encoding='utf-8'))
    pages_sha256 = digest(pages_path)
    for page in pages:
        if page['split'] not in SPLITS:
            raise ValueError('Unrecognized source split')
        image = safe_path(yolo_root.parent, page['image'])
        with Image.open(image) as im:
            source_size = im.size
        annotations = []
        for item in page['items']:
            if not isinstance(item.get('text'), str) or not item['text']:
                raise ValueError('Synthetic text annotation is missing its transcription')
            validate_polygon(item['corners'], source_size)
            annotations.append(dict(text=item['text'], corners=item['corners'],
                source_class_id=item['class_id'], orientation_flag=int(item['class_id'] == 1)))
        label = safe_path(yolo_root.parent, page['label'])
        original_rows = [line.split() for line in label.read_text().splitlines() if line.strip()]
        if len(original_rows) != len(annotations):
            raise ValueError('Source label and page-info annotation counts differ')
        for row, annotation in zip(original_rows, annotations):
            if (len(row) != 9 or int(row[0]) != annotation['source_class_id'] or
                    not np.allclose(np.array(row[1:], dtype=float).reshape(4,2)*source_size,
                                    annotation['corners'], atol=.001)):
                raise ValueError('Source labels do not agree with page-info geometry')
        sources.append(dict(id=f'synthetic-{page["split"]}-{image.stem}', image=image, info=pages_path,
            label=label, group='synthetic-'+page['split']+'-'+image.stem, split=page['split'],
            annotations=annotations, intended_text=None, text_verified=True, source_kind='known_synthetic_text',
            info_sha256=pages_sha256))
    return sources, skipped


def load_backgrounds(directory, seed):
    directory = resolve_path(directory).resolve()
    manifest = directory/'manifest.json'
    if manifest.exists():
        records = json.loads(manifest.read_text(encoding='utf-8'))['images']
    else:
        records = [dict(image=p.name, restaurant_id=p.stem) for p in sorted(directory.iterdir())
                   if p.suffix.lower() in IMAGE_EXTENSIONS and not p.stem.startswith('contact_sheet')]
    backgrounds = []
    seen_hashes = set()
    for record in records:
        path = safe_path(directory, record['image'])
        sha = digest(path)
        if sha in seen_hashes or (record.get('sha256') and record['sha256'] != sha):
            raise ValueError('Duplicate background or checksum mismatch')
        seen_hashes.add(sha)
        backgrounds.append(dict(image=path, sha256=sha, group=str(record.get('restaurant_id') or path.stem),
            negative_training_approved=record.get('negative_training_approved', False)))
    assignments = grouped_split([b['group'] for b in backgrounds], seed+1)
    for background in backgrounds:
        background['split'] = assignments[background['group']]
    return backgrounds


def relative_input(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def render_sample(source, background, variant, root, config):
    name = source['id'] + f'__v{variant:03d}'
    sample_seed = int.from_bytes(hashlib.sha256(f'{config.seed}:{name}'.encode()).digest()[:8], 'big')
    rng = random.Random(sample_seed)
    size = config.size
    with Image.open(source['image']) as original:
        original = original.convert('RGBA')
        matrix, pose = menu_transform(original.size,size,rng,config.max_pitch,config.max_roll,
                                      config.max_yaw,config.max_scene_scale)
        menu = warp_image(original, matrix, (size, size))
        menu_boundary = transform_points(corners(*original.size), matrix)
    menu_annotations = transform_annotations(source['annotations'], matrix, (size, size))
    with Image.open(background['image']) as original_background:
        scene = ImageOps.fit(ImageOps.exif_transpose(original_background).convert('RGB'), (size,size),
                             method=Image.Resampling.LANCZOS).convert('RGBA')
    placement = scene_transform(size,rng,config.min_scene_scale,config.max_scene_scale,menu_boundary)
    # Composite the actual augmented RGBA menu, preserving transparency outside it.
    placed = warp_image(menu, placement, (size,size))
    placed = ImageEnhance.Brightness(placed).enhance(rng.uniform(.9,1.05))
    shadow_alpha = placed.getchannel('A').filter(ImageFilter.GaussianBlur(3)).point(lambda v: int(v*.32))
    shadow = Image.new('RGBA', (size,size), (0,0,0,0))
    shadow.paste(Image.new('RGBA', (size,size), (0,0,0,255)), (4,6), shadow_alpha)
    scene = Image.alpha_composite(Image.alpha_composite(scene, shadow), placed).convert('RGB')
    composite_annotations = transform_annotations(menu_annotations, placement, (size,size))
    composite_boundary = transform_points(menu_boundary, placement)
    validate_polygon(composite_boundary, (size,size))
    split = source['split']
    paths = {kind: dict(image=f'{kind}/images/{split}/{name}'+('.png' if kind=='menu' else '.jpg'),
                        label=f'{kind}/labels/{split}/{name}.json') for kind in ('menu','composites')}
    for pair in paths.values():
        for relative in pair.values():
            (root/relative).parent.mkdir(parents=True, exist_ok=True)
    menu.save(root/paths['menu']['image'], compress_level=1)
    scene.save(root/paths['composites']['image'], quality=94)
    common = dict(id=name, split=split, source_menu_id=source['id'], source_group=source['group'],
        source_image=relative_input(source['image']), source_info=relative_input(source['info']),
        source_image_sha256=digest(source['image']), source_info_sha256=source['info_sha256'],
        source_kind=source['source_kind'], intended_text=source['intended_text'],
        pose=pose,
        text_verified=source['text_verified'], text='\n'.join(a['text'] for a in source['annotations']) or None,
        yolo_labels_assigned=False, seed=sample_seed, width=size, height=size)
    dump_json(root/paths['menu']['label'], dict(**common, image=paths['menu']['image'],
        sha256=digest(root/paths['menu']['image']), angle_degrees=pose['roll_degrees'],
        source_to_image=matrix.tolist(), menu_boundary=menu_boundary.tolist(), annotations=menu_annotations))
    dump_json(root/paths['composites']['label'], dict(**common, image=paths['composites']['image'],
        sha256=digest(root/paths['composites']['image']), augmented_menu_image=paths['menu']['image'],
        background_image=relative_input(background['image']), background_sha256=background['sha256'],
        background_group=background['group'], background_review_pending=not background['negative_training_approved'],
        menu_scale=float(placement[0,0]),background_selection='seeded_uniform_within_split',
        menu_to_image=placement.tolist(), source_to_image=(placement@matrix).tolist(),
        menu_boundary=composite_boundary.tolist(), annotations=composite_annotations))
    ctc_rows = []
    if config.export_ctc:
        for index, annotation in enumerate(composite_annotations):
            crop = extract_ctc_line(scene, annotation['corners'], annotation['orientation_flag'])
            crop_image = f'ctc_lines/images/{split}/{name}__line{index:02d}.png'
            crop_label = f'ctc_lines/labels/{split}/{name}__line{index:02d}.json'
            (root/crop_image).parent.mkdir(parents=True, exist_ok=True)
            crop.save(root/crop_image, compress_level=1)
            row = dict(image=crop_image, label=crop_label, text=annotation['text'],
                orientation_flag=annotation['orientation_flag'], split=split, width=384, height=64,
                text_verified=True, source_menu_id=source['id'], source_group=source['group'],
                background_group=background['group'], composite_image=paths['composites']['image'],
                composite_polygon=annotation['corners'], sha256=digest(root/crop_image))
            dump_json(root/crop_label, row)
            ctc_rows.append(row)
    return dict(id=name, split=split, source_group=source['group'], background_group=background['group'],
                menu=paths['menu'], composite=paths['composites'], ctc_rows=ctc_rows,
                text_verified=source['text_verified'])


def preview(root, rows):
    selected = [r for r in rows if r['id'].startswith('genai-')][:3] + [r for r in rows if r['id'].startswith('synthetic-')][:3]
    selected = list({row['id']: row for row in selected}.values())
    sheet = Image.new('RGB', (960, len(selected)*335), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(selected):
        for col, kind in enumerate(('menu', 'composite')):
            with Image.open(root/row[kind]['image']) as im:
                im = im.convert('RGBA')
                bg = Image.new('RGBA', im.size, (225,225,225,255))
                im = ImageOps.contain(Image.alpha_composite(bg,im).convert('RGB'), (470,305))
                sheet.paste(im, (col*480+(480-im.width)//2, i*335))
            draw.text((col*480+8,i*335+310), row['id']+' / '+kind, fill='black')
    sheet.save(root/'preview.jpg', quality=90)
    labels_path = root/'ctc_lines/labels.jsonl'
    if labels_path.exists():
        with labels_path.open(encoding='utf-8') as stream:
            lines = []
            for line in stream:
                lines.append(json.loads(line))
                if len(lines)==12:
                    break
        if lines:
            ctc_sheet = Image.new('RGB',(800,len(lines)*160),'white')
            ctc_draw = ImageDraw.Draw(ctc_sheet)
            font_path = ROOT/'data/resources/fonts/KleeOne-SemiBold.ttf'
            font = ImageFont.truetype(str(font_path),18) if font_path.exists() else ImageFont.load_default()
            for i,line in enumerate(lines):
                with Image.open(root/line['image']) as crop:
                    ctc_sheet.paste(crop.resize((768,128),Image.Resampling.NEAREST).convert('RGB'),(16,i*160))
                caption = f'flag={line["orientation_flag"]}  '+line['text']
                if not font_path.exists():
                    caption = caption.encode('ascii','backslashreplace').decode()
                ctc_draw.text((16,i*160+132),caption,font=font,fill='black')
            ctc_sheet.save(root/'ctc_preview.jpg',quality=92)


def build(output_dir=ROOT/'data/processed/augmentation/run_1100', menu_dir=ROOT/'data/raw/genai_seeds/menu_images',
          yolo_root=ROOT/'data/processed/synthetic_base_v1/yolo_obb', background_dir=ROOT/'data/raw/tabelog_interiors',
          config=None, max_menus=None, count=None):
    config = config or AugmentationConfig()
    config.validate()
    if count is not None and count<1:
        raise ValueError('Requested sample count must be positive')
    root = resolve_path(output_dir).resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError('Output must stay inside the workspace')
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Choose a new empty output directory; existing augmentation is never overwritten')
    sources, skipped = load_sources(menu_dir, yolo_root, config.seed)
    backgrounds = load_backgrounds(background_dir, config.seed)
    if max_menus is not None:
        if max_menus < 1:
            raise ValueError('max_menus must be positive')
        sources = sources[:max_menus]
    if not sources:
        raise ValueError('No source menus found')
    pools = {split: [b for b in backgrounds if b['split']==split] for split in SPLITS}
    tasks = []
    variants = max(config.variants,math.ceil(count/len(sources))) if count is not None else config.variants
    for source in sources:
        for variant in range(variants):
            key = int.from_bytes(hashlib.sha256(f'{config.seed}:{source["id"]}:{variant}:background'.encode()).digest()[:8], 'big')
            pool = pools[source['split']]
            tasks.append((source,random.Random(key).choice(pool),variant))
    if count is not None:
        random.Random(config.seed+2).shuffle(tasks)
        tasks = tasks[:count]
    used_sources = {source['id']:source for source,_,_ in tasks}
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    dump_json(root/'config.json', {**asdict(config), 'requested_samples':count,'menu_dir':relative_input(menu_dir),
        'yolo_root':relative_input(yolo_root), 'background_dir':relative_input(background_dir)})
    def run(task):
        return render_sample(*task, root, config)
    rows = []
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        for row in pool.map(run, tasks):
            rows.append(row)
            if len(rows)%100==0 or len(rows)==len(tasks):
                print(f'Augmented {len(rows)}/{len(tasks)} menus and composites', flush=True)
    ctc = [line for row in rows for line in row.pop('ctc_rows')]
    (root/'ctc_lines').mkdir(exist_ok=True)
    (root/'ctc_lines/labels.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in ctc), encoding='utf-8')
    manifest = dict(status='built', menu_images=len(rows), composite_images=len(rows), ctc_lines=len(ctc),
        source_menus=len(used_sources),available_source_menus=len(sources),requested_samples=count,
        generated_menus_without_verified_line_labels=sum(not s['text_verified'] for s in used_sources.values()),
        splits=dict(Counter(r['split'] for r in rows)), yolo_labels_assigned=False,
        annotation_policy='Preserve source text and transform existing polygons; YOLO label assignment deferred to Stage 2.',
        generated_text_policy='Intended prompt text retained as unverified; no CTC crops fabricated for unannotated generated menus.',
        background_policy='Restaurant groups remain split-disjoint; visible background menus/text still require review.',
        background_splits={b['group']: b['split'] for b in backgrounds}, skipped_sources=skipped,
        seconds=round(time.perf_counter()-started,3), samples=rows)
    dump_json(root/'manifest.json', manifest)
    preview(root, rows)
    from scripts.data.verify_menu_augmentation import verify
    verify(root)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'data/processed/augmentation/run_1100')
    parser.add_argument('--menu-dir', type=Path, default=ROOT/'data/raw/genai_seeds/menu_images')
    parser.add_argument('--yolo-root', type=Path, default=ROOT/'data/processed/synthetic_base_v1/yolo_obb')
    parser.add_argument('--background-dir', type=Path, default=ROOT/'data/raw/tabelog_interiors')
    parser.add_argument('--variants', type=int, default=1)
    parser.add_argument('--size', type=int, default=640)
    parser.add_argument('--seed', type=int, default=20261107)
    parser.add_argument('--max-pitch',type=float,default=30,help='Maximum forward/backward pitch magnitude in degrees')
    parser.add_argument('--max-roll','--max-angle',dest='max_roll',type=float,default=10)
    parser.add_argument('--max-yaw',type=float,default=10)
    parser.add_argument('--min-scene-scale', type=float, default=.5)
    parser.add_argument('--max-scene-scale', type=float, default=1.2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--no-ctc', action='store_true')
    parser.add_argument('--max-menus', type=int)
    parser.add_argument('--count',type=int,help='Exact number of augmented menu/composite pairs; seeded sampling across source sets')
    args = parser.parse_args()
    config = AugmentationConfig(size=args.size,variants=args.variants,seed=args.seed,
        max_pitch=args.max_pitch,max_roll=args.max_roll,max_yaw=args.max_yaw,
        min_scene_scale=args.min_scene_scale,max_scene_scale=args.max_scene_scale,
        export_ctc=not args.no_ctc,workers=args.workers)
    build(args.output_dir,args.menu_dir,args.yolo_root,args.background_dir,config,args.max_menus,args.count)
