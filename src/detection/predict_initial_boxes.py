"""Run local PP-OCRv4 DBNet over consolidated images; never approve labels."""
import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np
from PIL import Image
from src.common import dump_json, digest
from src.common.paths import resolve_path
from .common import (CLASSES, DEFAULT_ROOT, annotation, bootstrap_packages,
                     read_json, save_pair)


def class_hint(points, width, height, text=''):
    p = np.asarray(points)
    w, h = np.ptp(p, axis=0)
    short = max(w / width, h / height) < .3
    if short and (any(c in text for c in ('円', '¥', '￥')) or
                  p[:, 0].mean() / width > .82 or p[:, 1].mean() / height > .85):
        return CLASSES[2], 'short region with currency or bottom/right margin'
    return (CLASSES[1], 'H/W > 1.5') if h / max(w, 1e-6) > 1.5 else (CLASSES[0], 'default horizontal')


def detector(weights=None, side=960):
    bootstrap_packages()
    import rapidocr_onnxruntime
    import yaml
    from rapidocr_onnxruntime.ch_ppocr_det import TextDetector
    package = Path(rapidocr_onnxruntime.__file__).parent
    config = yaml.safe_load((package / 'config.yaml').read_text())['Det']
    model = Path(weights) if weights else package / 'models/ch_PP-OCRv4_det_infer.onnx'
    if not model.is_file():
        raise ValueError('Local PP-OCRv4 ONNX weights are required')
    config.update(model_path=str(model.resolve()), limit_type='max',
                  limit_side_len=side, intra_op_num_threads=2, inter_op_num_threads=1)
    engine = TextDetector(config)
    # RapidOCR's default max mode overrides limit_side_len. Fix resolution explicitly.
    from rapidocr_onnxruntime.ch_ppocr_det.utils import DetPreProcess
    engine.get_preprocess = lambda _: DetPreProcess(side, 'max', config['mean'], config['std'])
    return engine, dict(backend='PP-OCRv4 DBNet ONNX / RapidOCR', weights=str(model),
                       weights_sha256=digest(model), input_max_side=side,
                       detection_only=True, transcription_verified=False, config=config)


def predict(root=DEFAULT_ROOT, weights=None, side=960, limit=None):
    root = resolve_path(root)
    manifest = read_json(root / 'manifest.json')
    engine, provenance = detector(weights, side)
    start = time.monotonic()
    done = 0
    failures = []
    for record in manifest['records']:
        target = root / 'predictions' / (record['id'] + '.json')
        if target.exists():
            prior = read_json(target)
            if prior.get('image_sha256') != record['sha256'] or prior.get('provenance') != provenance:
                raise ValueError(f'Stale prediction: {target}; use a fresh output collection')
            continue
        try:
            image_path = root / 'images' / record['image']
            if digest(image_path) != record['sha256']:
                raise ValueError('Image checksum changed')
            with Image.open(image_path) as image:
                rgba = image.convert('RGBA')
                # Transparent menu pixels must have the same white background as the UI.
                rgb = Image.new('RGBA', rgba.size, 'white')
                rgb.alpha_composite(rgba)
                pixels = np.asarray(rgb.convert('RGB'))[:, :, ::-1].copy()
            boxes, _ = engine(pixels)
            shapes = []
            for points in ([] if boxes is None else boxes):
                label, reason = class_hint(points, record['width'], record['height'])
                shapes.append(dict(label=label, points=np.asarray(points, dtype=float).tolist(),
                                   group_id=None, shape_type='polygon', flags={'reviewed': False},
                                   description='Unreviewed heuristic: ' + reason))
            doc = annotation(record, shapes, provenance)
            save_pair(root / 'predictions', record, doc)
            done += 1
            if done % 100 == 0:
                print(f'{done} new predictions; {time.monotonic()-start:.1f}s', flush=True)
            if limit and done >= limit:
                break
        except Exception as exc:
            failures.append(dict(id=record['id'], error=str(exc)))
            print(f"FAILED {record['id']}: {exc}", flush=True)
    counts = Counter()
    total = boxes_count = 0
    for record in manifest['records']:
        path = root / 'predictions' / (record['id'] + '.json')
        if path.exists():
            doc = read_json(path)
            total += 1
            boxes_count += len(doc['shapes'])
            counts.update(s['label'] for s in doc['shapes'])
    report = dict(images=len(manifest['records']), predicted=total, boxes=boxes_count,
                  classes=dict(counts), failures=failures, complete=total == len(manifest['records']) and not failures,
                  provenance=provenance, manually_reviewed=0)
    dump_json(root / 'prediction_report.json', report)
    print({k: v for k, v in report.items() if k != 'provenance'}, flush=True)
    if failures:
        raise RuntimeError('Some images failed; see prediction_report.json and rerun to resume')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--side', type=int, default=960)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    predict(args.root, args.weights, args.side, args.limit)


if __name__ == '__main__':
    main()
