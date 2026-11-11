import json
import math
from pathlib import Path

from src.common import dump_json, digest
from src.common.paths import ANNOTATION_ROOT, ROOT, resolve_path

CLASSES = ['dish_horizontal', 'dish_vertical', 'price_tag']
DEFAULT_ROOT = ANNOTATION_ROOT


def read_json(path):
    return json.loads(resolve_path(path).read_text(encoding='utf-8'))


def validate_shapes(shapes, width, height):
    if not isinstance(shapes, list) or len(shapes) > 10000:
        raise ValueError('Expected a list of at most 10000 shapes')
    for shape in shapes:
        if shape.get('label') not in CLASSES:
            raise ValueError('Unknown detector class')
        points = shape.get('points')
        if not isinstance(points, list) or len(points) != 4:
            raise ValueError('Each OBB must have four ordered corners')
        for point in points:
            if len(point) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in point):
                raise ValueError('Coordinates must be finite numbers')
            if not 0 <= point[0] <= width or not 0 <= point[1] <= height:
                raise ValueError('Box extends outside the image')
        cross = []
        for i in range(4):
            a, b, c = points[i], points[(i+1)%4], points[(i+2)%4]
            cross.append((b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0]))
        if not (all(v > 1e-6 for v in cross) or all(v < -1e-6 for v in cross)):
            raise ValueError('Box must be convex, ordered, and nondegenerate')
    return shapes


def yolo_text(shapes, width, height):
    validate_shapes(shapes, width, height)
    return ''.join(str(CLASSES.index(s['label'])) + ' ' + ' '.join(
        f'{v / (width if i % 2 == 0 else height):.8f}'
        for i, v in enumerate(sum(s['points'], []))) + '\n' for s in shapes)


def annotation(record, shapes, provenance):
    return dict(version='2.5.0', flags={'reviewed': False}, shapes=shapes,
                imagePath='../images/' + record['image'], imageData=None,
                imageWidth=record['width'], imageHeight=record['height'],
                image_sha256=record['sha256'], source_id=record['id'],
                provenance=provenance)


def save_pair(directory, record, doc):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    content = yolo_text(doc['shapes'], record['width'], record['height'])
    # JSON is the authoritative record; write it last as the completion marker.
    txt = directory / (record['id'] + '.txt')
    tmp = txt.with_suffix('.txt.tmp')
    tmp.write_text(content, encoding='utf-8')
    tmp.replace(txt)
    dump_json(directory / (record['id'] + '.json'), doc)


def bootstrap_packages():
    """Support the workspace-only dependency installation used on this machine."""
    import sys
    local = ROOT / '.stage2_packages'
    if local.is_dir():
        sys.path.insert(0, str(local))
