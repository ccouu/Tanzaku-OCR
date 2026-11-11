"""Pretrained inference only; export X-AnyLabeling polygons for manual review.

Use locally downloaded ogkalu/comic-text-segmenter-yolov8m weights with
--backend yolo --weights model.pt, or an installed PaddleOCR DBNet detector
with --backend paddle --weights /path/to/local/detection/model.
No model training or implicit weight download is performed here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.common import dump_json


def export_annotation(image_path, polygons, output_path):
    image_path = Path(image_path)
    with Image.open(image_path) as image:
        width, height = image.size
    shapes = []
    for polygon in polygons:
        points = np.asarray(polygon, dtype=float).reshape(-1, 2)
        if len(points) < 4 or not np.isfinite(points).all():
            raise ValueError('Invalid detector polygon')
        points[:, 0] = points[:, 0].clip(0, width)
        points[:, 1] = points[:, 1].clip(0, height)
        vertical = np.ptp(points[:, 1]) > np.ptp(points[:, 0])
        shapes.append(dict(label='', points=points.tolist(), group_id=None, shape_type='polygon',
                           flags={'reviewed': False, 'vertical': bool(vertical)},
                           description='Replace label with reviewed transcription; confirm orientation and price class.'))
    import os
    dump_json(output_path, dict(version='2.5.0', flags={'reviewed': False}, shapes=shapes,
        imagePath=os.path.relpath(image_path.resolve(), Path(output_path).parent.resolve()),
        imageData=None, imageHeight=height, imageWidth=width))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['yolo', 'paddle'], required=True)
    parser.add_argument('--weights', type=Path, required=True)
    args = parser.parse_args()
    if not args.weights.exists():
        parser.error('Local pretrained weights are required; no automatic downloads')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.backend == 'yolo':
        from ultralytics import YOLO
        detector = YOLO(str(args.weights))
    else:
        # PaddleOCR 2.x local DBNet API. Install this optional backend separately.
        from paddleocr import PaddleOCR
        detector = PaddleOCR(det_model_dir=str(args.weights), use_angle_cls=False, lang='japan', show_log=False)
    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
    if not paths:
        parser.error('No real menu photographs found')
    for path in paths:
        if args.backend == 'yolo':
            result = detector.predict(str(path), verbose=False)[0]
            if result.obb is not None:
                polygons = result.obb.xyxyxyxy.cpu().numpy()
            elif result.masks is not None:
                polygons = []
                for polygon in result.masks.xy:
                    x1, y1 = polygon.min(axis=0)
                    x2, y2 = polygon.max(axis=0)
                    polygons.append([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
            else:
                polygons = [[[x1,y1],[x2,y1],[x2,y2],[x1,y2]] for x1,y1,x2,y2 in result.boxes.xyxy.cpu().numpy()]
        else:
            polygons = detector.ocr(str(path), det=True, rec=False, cls=False)[0] or []
        export_annotation(path, polygons, args.output / (path.stem + '.json'))
    print('Prelabels exported. Review in X-AnyLabeling before crop export and merge.')


if __name__ == '__main__':
    main()
