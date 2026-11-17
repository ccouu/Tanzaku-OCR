"""YOLO11n-OBB line cropping with H=64 (>512px, 100px overlap) splitting BEFORE VLM canvas packing."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.yolo_crop_canvas_pipeline import (
    DEFAULT_WEIGHTS,
    dump_json,
    pack_canvases,
    run_build,
    sha256_file,
)
from src.recognition.long_line_inference import bucket_image, native_strip, window_ranges


def split_manifest_crops_h64(
    output_dir: Path,
    max_width: int = 512,
    overlap: int = 100,
    items_per_canvas: int = 25,
    separate_vertical: bool = True,
    include_hints: bool = True,
) -> Dict:
    manifest_path = output_dir / "crops_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    orig_crops: List[Dict] = manifest["crops"]

    new_crops: List[Dict] = []
    long_split_parents = 0
    total_windows_created = 0

    for crop in orig_crops:
        display_path = output_dir / crop["display_crop"]
        with Image.open(display_path) as im:
            rgb = im.convert("RGB")
            oriented = rgb.transpose(Image.Transpose.ROTATE_90) if crop["is_vertical"] else rgb

        strip = native_strip(oriented)
        w_h64 = strip.width

        if w_h64 <= max_width:
            ctc_im, bmeta = bucket_image(strip, max_width=max_width)
            ctc_path = output_dir / crop["ctc_crop"]
            ctc_im.save(ctc_path, compress_level=1)
            updated = dict(crop)
            updated.update(bmeta)
            updated["w_h64_original"] = w_h64
            updated["window_index"] = 1
            updated["window_count"] = 1
            updated["window_range_h64"] = [0, w_h64]
            updated["parent_crop_id"] = crop["crop_id"]
            updated["ctc_sha256"] = sha256_file(ctc_path)
            new_crops.append(updated)
            continue

        long_split_parents += 1
        ranges = window_ranges(w_h64, max_width=max_width, overlap=overlap)
        total_windows_created += len(ranges)

        for win_idx, (x0, x1) in enumerate(ranges, start=1):
            win_strip = strip.crop((x0, 0, x1, 64))
            win_display = (
                win_strip.transpose(Image.Transpose.ROTATE_270)
                if crop["is_vertical"]
                else win_strip
            )
            win_crop_id = f"{crop['crop_id']}__w{win_idx:02d}"
            win_display_rel = f"crops_display/{win_crop_id}.png"
            win_ctc_rel = f"crops_ctc/{win_crop_id}.png"

            win_display_path = output_dir / win_display_rel
            win_ctc_path = output_dir / win_ctc_rel
            win_display_path.parent.mkdir(parents=True, exist_ok=True)
            win_ctc_path.parent.mkdir(parents=True, exist_ok=True)

            win_display.save(win_display_path, compress_level=1)
            ctc_im, bmeta = bucket_image(win_strip, max_width=max_width)
            ctc_im.save(win_ctc_path, compress_level=1)

            win_entry = dict(crop)
            win_entry.update(bmeta)
            win_entry["crop_id"] = win_crop_id
            win_entry["parent_crop_id"] = crop["crop_id"]
            win_entry["window_index"] = win_idx
            win_entry["window_count"] = len(ranges)
            win_entry["window_range_h64"] = [int(x0), int(x1)]
            win_entry["w_h64_original"] = w_h64
            win_entry["display_crop"] = win_display_rel
            win_entry["display_size"] = list(win_display.size)
            win_entry["ctc_crop"] = win_ctc_rel
            win_entry["image"] = win_ctc_rel
            win_entry["ctc_sha256"] = sha256_file(win_ctc_path)
            new_crops.append(win_entry)

    manifest["parent_crops_count"] = len(orig_crops)
    manifest["long_crops_split_count"] = long_split_parents
    manifest["long_split_windows_created"] = total_windows_created
    manifest["split_max_width_h64"] = max_width
    manifest["split_overlap_h64"] = overlap
    manifest["total_crops"] = len(new_crops)
    manifest["crops"] = new_crops
    dump_json(manifest_path, manifest)

    canvas_manifest = pack_canvases(
        crops=new_crops,
        output_root=output_dir,
        items_per_canvas=items_per_canvas,
        separate_vertical=separate_vertical,
        include_hints=include_hints,
    )
    print(
        f"[H64 Split + Pack] parent_crops={len(orig_crops)} long_split={long_split_parents} "
        f"split_windows={total_windows_created} final_slots={len(new_crops)} "
        f"canvases={canvas_manifest['total_canvases']}",
        flush=True,
    )
    return canvas_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO11n-OBB crop + H64 >512px split (100px overlap) + VLM canvas pack")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--iou-dedup", type=float, default=0.65)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--items-per-canvas", type=int, default=25)
    parser.add_argument("--max-width", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--no-separate-vertical", action="store_true")
    parser.add_argument("--no-hints", action="store_true")
    args = parser.parse_args()

    run_build(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        weights=args.weights,
        confidence=args.confidence,
        iou_dedup=args.iou_dedup,
        imgsz=args.imgsz,
        device=args.device,
        items_per_canvas=args.items_per_canvas,
        separate_vertical=not args.no_separate_vertical,
        include_hints=not args.no_hints,
        safe_line_crops=True,
        merge_iom=0.55,
    )
    split_manifest_crops_h64(
        output_dir=args.output_dir,
        max_width=args.max_width,
        overlap=args.overlap,
        items_per_canvas=args.items_per_canvas,
        separate_vertical=not args.no_separate_vertical,
        include_hints=not args.no_hints,
    )


if __name__ == "__main__":
    main()
