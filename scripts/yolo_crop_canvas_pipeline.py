"""End-to-end YOLO11n-OBB line cropping, token-efficient VLM contact-sheet builder, and label parser.

Workflow stages:
  1. `build` (default): Run the selected V5 YOLO11n-OBB checkpoint on menu photos,
     rectify oriented line crops, and pack them into numbered contact-sheet canvases (`canvas_001.png`, ...)
     with ready-to-paste VLM prompts (`canvas_001_prompt.txt`) and a reversible `canvas_manifest.json`.
  2. `pack`: Rebuild contact-sheet canvases from an existing `crops_manifest.json` without re-running YOLO.
  3. `parse`: Parse VLM transcription outputs (`[01] | <text> | <OK|TRUNCATED|NOISE>`) back into
     `labeled_crops.jsonl` and Stage-3 recognizer `ctc_labels.txt` (`<crop_path>\\t<text>`).
     This prepares images/prompts and parses supplied responses; it does not call a VLM API.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOCAL_PKGS = ROOT / ".stage2_packages"
DEFAULT_WEIGHTS = ROOT / "outputs/detection/training/banana130_finetune_v5/yolo11n_menu_obb_banana130_best.pt"
DEFAULT_WEIGHTS_REASON = "V5: selected by combined latest-label 13-menu validation mAP50-95; V3 has higher recall. Not proof of superior stroke-tightness."
RUNTIME_ROOT = ROOT / ".cache/ultralytics"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def corners(width: float, height: float) -> np.ndarray:
    return np.array([[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]], dtype=float)


def homography(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    destination = np.asarray(destination, dtype=float)
    if source.shape != (4, 2) or destination.shape != (4, 2):
        raise ValueError("Homography requires two (4, 2) corner arrays")
    rows, values = [], []
    for (x, y), (u, v) in zip(source, destination):
        rows.extend([[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]])
        values.extend([u, v])
    matrix = np.append(np.linalg.solve(np.array(rows, dtype=float), np.array(values, dtype=float)), 1.0).reshape(3, 3)
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("Degenerate perspective transform")
    return matrix


def warp_polygon(image: Image.Image, quad: np.ndarray, output_size: Tuple[int, int]) -> Image.Image:
    w, h = int(round(output_size[0])), int(round(output_size[1]))
    w, h = max(8, w), max(8, h)
    matrix = homography(quad, corners(w, h))
    inverse = np.linalg.inv(matrix)
    inverse /= inverse[2, 2]
    fill = (255, 255, 255) if image.mode == "RGB" else 255
    return image.transform(
        (w, h),
        Image.Transform.PERSPECTIVE,
        tuple(inverse.flat[:8]),
        resample=Image.Resampling.BICUBIC,
        fillcolor=fill,
    )


def canonicalize_obb_corners(pts: np.ndarray, label: str = "") -> Tuple[np.ndarray, float, float, bool]:
    """Order 4 OBB corners as [top-left, top-right, bottom-right, bottom-left] aligned with the box's principal axes.

    For horizontal lines, the long edge is the top/bottom edge (width >= height).
    For vertical lines (`縦書き`), the short edge is the top/bottom edge and the long edge runs top-to-bottom (height > width).
    """
    pts = np.asarray(pts, dtype=float).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]  # clockwise perimeter order in image coordinates (x-right, y-down)

    # Opposite edge pairs: pair A = (p0->p1, p3->p2), pair B = (p1->p2, p0->p3)
    len_a = 0.5 * (float(np.linalg.norm(pts[1] - pts[0])) + float(np.linalg.norm(pts[2] - pts[3])))
    len_b = 0.5 * (float(np.linalg.norm(pts[2] - pts[1])) + float(np.linalg.norm(pts[3] - pts[0])))
    vec_a = 0.5 * ((pts[1] - pts[0]) + (pts[2] - pts[3]))
    vec_b = 0.5 * ((pts[2] - pts[1]) + (pts[3] - pts[0]))

    if len_a >= len_b:
        long_vec, long_len, short_len = vec_a, len_a, len_b
        long_is_a = True
    else:
        long_vec, long_len, short_len = vec_b, len_b, len_a
        long_is_a = False

    # Decide if this text line's reading axis is vertical (top-to-bottom) or horizontal (left-to-right)
    long_angle_deg = abs(math.degrees(math.atan2(abs(long_vec[1]), abs(long_vec[0]))))
    if label == "dish_vertical" and long_len > short_len * 1.15:
        is_vertical = True
    elif label in ("dish_horizontal", "price_tag") and long_angle_deg < 55.0:
        is_vertical = False
    else:
        is_vertical = bool(long_angle_deg >= 48.0 and long_len > short_len * 1.25)

    # Candidate cyclic shifts where edge (p0 -> p1) is the desired top edge:
    # - If vertical: top edge (p0 -> p1) must be the SHORT edge
    # - If horizontal: top edge (p0 -> p1) must be the LONG edge
    want_a_as_top = (not long_is_a) if is_vertical else long_is_a
    valid_shifts = [0, 2] if want_a_as_top else [1, 3]

    # Among the two opposite candidate edges, pick the one higher up in the image (smaller mean y; tie-break by x)
    best_shift = min(
        valid_shifts,
        key=lambda s: (
            0.5 * (pts[s % 4, 1] + pts[(s + 1) % 4, 1]),
            0.5 * (pts[s % 4, 0] + pts[(s + 1) % 4, 0]),
        ),
    )
    ordered = np.roll(pts, -best_shift, axis=0)
    width = max(8.0, 0.5 * (float(np.linalg.norm(ordered[1] - ordered[0])) + float(np.linalg.norm(ordered[2] - ordered[3]))))
    height = max(8.0, 0.5 * (float(np.linalg.norm(ordered[3] - ordered[0])) + float(np.linalg.norm(ordered[2] - ordered[1]))))
    return ordered, width, height, is_vertical


def expand_quad(quad: np.ndarray, pad_x_ratio: float, pad_y_ratio: float, img_w: int, img_h: int) -> Tuple[np.ndarray, float, float]:
    """Expand a canonical [TL, TR, BR, BL] quad outward along its local axes and clip to image bounds."""
    tl, tr, br, bl = quad
    u_top = tr - tl
    u_bot = br - bl
    v_left = bl - tl
    v_right = br - tr
    out = np.array(
        [
            tl - pad_x_ratio * u_top - pad_y_ratio * v_left,
            tr + pad_x_ratio * u_top - pad_y_ratio * v_right,
            br + pad_x_ratio * u_bot + pad_y_ratio * v_right,
            bl - pad_x_ratio * u_bot + pad_y_ratio * v_left,
        ],
        dtype=float,
    )
    out[:, 0] = np.clip(out[:, 0], 0.0, float(img_w))
    out[:, 1] = np.clip(out[:, 1], 0.0, float(img_h))
    w = max(8.0, 0.5 * (float(np.linalg.norm(out[1] - out[0])) + float(np.linalg.norm(out[2] - out[3]))))
    h = max(8.0, 0.5 * (float(np.linalg.norm(out[3] - out[0])) + float(np.linalg.norm(out[2] - out[1]))))
    return out, w, h


def aabb_iou(q1: np.ndarray, q2: np.ndarray) -> float:
    x1_min, y1_min = q1.min(axis=0)
    x1_max, y1_max = q1.max(axis=0)
    x2_min, y2_min = q2.min(axis=0)
    x2_max, y2_max = q2.max(axis=0)
    ix = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
    iy = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0
    a1 = max(1.0, (x1_max - x1_min) * (y1_max - y1_min))
    a2 = max(1.0, (x2_max - x2_min) * (y2_max - y2_min))
    return float(inter / (a1 + a2 - inter))


def deduplicate_detections(detections: List[Dict], iou_threshold: float = 0.65) -> List[Dict]:
    """Suppress duplicate cross-class OBB detections on the exact same text line."""
    ranked = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept: List[Dict] = []
    for cand in ranked:
        q_cand = np.asarray(cand["obb_polygon"], dtype=float)
        if any(aabb_iou(q_cand, np.asarray(k["obb_polygon"], dtype=float)) >= iou_threshold for k in kept):
            continue
        kept.append(cand)
    return kept


def sort_detections_reading_order(detections: List[Dict]) -> List[Dict]:
    """Sort horizontal boxes top-to-bottom/left-to-right, and vertical boxes right-to-left/top-to-bottom."""
    horiz = [d for d in detections if not d["is_vertical"]]
    vert = [d for d in detections if d["is_vertical"]]
    horiz.sort(key=lambda d: (round(d["center"][1] / 24.0), d["center"][0]))
    vert.sort(key=lambda d: (-round(d["center"][0] / 24.0), d["center"][1]))
    return horiz + vert


def normalize_menu_id(image_path: Path) -> str:
    stem = image_path.stem
    m = re.match(r"^menu[_-]0*(\d+)(?:_\d+)?$", stem, re.IGNORECASE)
    if m:
        return f"menu-{int(m.group(1)):03d}"
    return stem


def load_sidecar_hints(image_path: Path) -> List[str]:
    norm_id = normalize_menu_id(image_path)
    candidates = [image_path.with_suffix(".json"), image_path.parent / f"{norm_id}.json"]
    sidecar = next((c for c in candidates if c.exists()), None)
    if sidecar is None:
        return []
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data.get("gt_lines"), list):
        return [str(x).strip() for x in data["gt_lines"] if str(x).strip()]
    if isinstance(data.get("intended_text"), str) and data["intended_text"].strip():
        return [s.strip() for s in re.split(r"[、/]", data["intended_text"]) if s.strip()]
    return []


def load_font(size: int = 20) -> ImageFont.ImageFont:
    for candidate in (
        "C:/Windows/Fonts/consolab.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/msgothic.ttc",
    ):
        p = Path(candidate)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_horizontal_canvas(
    items: Sequence[Dict],
    output_root: Path,
    canvas_path: Path,
    target_row_height: int = 56,
    max_crop_width: int = 860,
) -> Tuple[int, int]:
    """Stack horizontal crops top-to-bottom with a left `[ID]` index gutter."""
    font_id = load_font(22)
    font_meta = load_font(13)
    gutter_w = 78
    pad_y = 14
    header_h = 44

    scaled_crops: List[Tuple[str, Dict, Image.Image]] = []
    max_w = 480
    total_h = header_h + pad_y

    for slot_idx, item in enumerate(items, start=1):
        slot_tag = f"[{slot_idx:02d}]"
        crop_path = output_root / item["display_crop"]
        with Image.open(crop_path) as im:
            im_rgb = im.convert("RGB")
            cw, ch = im_rgb.size
            if item["is_vertical"] and ch > cw * 1.35:
                scale = min(2.2, max(1.0, 180.0 / max(1, ch)))
                nw, nh = int(round(cw * scale)), int(round(ch * scale))
            else:
                scale = target_row_height / max(1.0, float(ch))
                nw = int(round(cw * scale))
                nh = target_row_height
                if nw > max_crop_width:
                    shrink = max_crop_width / float(nw)
                    nw = max_crop_width
                    nh = max(32, int(round(nh * shrink)))
            resized = im_rgb.resize((max(12, nw), max(12, nh)), Image.Resampling.BICUBIC)
        scaled_crops.append((slot_tag, item, resized))
        max_w = max(max_w, resized.size[0])
        total_h += max(resized.size[1], 42) + pad_y

    canvas_w = gutter_w + max_w + 32
    canvas = Image.new("RGB", (canvas_w, total_h), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, canvas_w, header_h], fill=(30, 41, 59))
    draw.text(
        (12, 12),
        f"{canvas_path.stem}  |  Read [01]..[{len(items):02d}] verbatim in Japanese",
        fill=(255, 255, 255),
        font=font_meta,
    )

    y = header_h + pad_y // 2
    for slot_tag, item, crop_im in scaled_crops:
        row_h = max(crop_im.size[1], 42)
        row_bg = (255, 255, 255) if int(slot_tag[1:-1]) % 2 == 1 else (244, 246, 250)
        draw.rectangle([0, y, canvas_w, y + row_h + pad_y - 2], fill=row_bg)
        draw.rectangle([0, y, gutter_w - 6, y + row_h + pad_y - 2], fill=(238, 242, 255))
        draw.line([(gutter_w - 6, y), (gutter_w - 6, y + row_h + pad_y - 2)], fill=(79, 70, 229), width=2)
        draw.text((10, y + (row_h - 22) // 2 + 2), slot_tag, fill=(185, 28, 28), font=font_id)

        paste_y = y + (row_h + pad_y - 2 - crop_im.size[1]) // 2
        canvas.paste(crop_im, (gutter_w + 8, paste_y))
        draw.rectangle(
            [gutter_w + 7, paste_y - 1, gutter_w + 8 + crop_im.size[0], paste_y + crop_im.size[1]],
            outline=(203, 213, 225),
            width=1,
        )
        draw.line([(0, y + row_h + pad_y - 2), (canvas_w, y + row_h + pad_y - 2)], fill=(203, 213, 225), width=1)
        y += row_h + pad_y

    canvas_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(canvas_path, compress_level=3)
    return canvas.size


def render_vertical_columns_canvas(
    items: Sequence[Dict],
    output_root: Path,
    canvas_path: Path,
    target_col_width: int = 58,
    max_crop_height: int = 760,
) -> Tuple[int, int]:
    """Pack upright vertical Japanese text lines (`縦書き`) side-by-side in numbered columns `[01]..[N]`."""
    font_id = load_font(18)
    font_meta = load_font(13)
    header_h = 44
    top_badge_h = 38
    pad_x = 14
    pad_y = 16

    scaled_cols: List[Tuple[str, Dict, Image.Image, int]] = []
    max_h = 260
    total_w = pad_x

    for slot_idx, item in enumerate(items, start=1):
        slot_tag = f"[{slot_idx:02d}]"
        crop_path = output_root / item["display_crop"]
        with Image.open(crop_path) as im:
            im_rgb = im.convert("RGB")
            cw, ch = im_rgb.size
            scale = target_col_width / max(1.0, float(cw))
            nw = target_col_width
            nh = int(round(ch * scale))
            if nh > max_crop_height:
                shrink = max_crop_height / float(nh)
                nh = max_crop_height
                nw = max(32, int(round(nw * shrink)))
            resized = im_rgb.resize((max(16, nw), max(16, nh)), Image.Resampling.BICUBIC)
        col_w = max(resized.size[0], 56)
        scaled_cols.append((slot_tag, item, resized, col_w))
        max_h = max(max_h, resized.size[1])
        total_w += col_w + pad_x

    canvas_w = max(480, total_w)
    canvas_h = header_h + top_badge_h + max_h + 2 * pad_y
    canvas = Image.new("RGB", (canvas_w, canvas_h), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, canvas_w, header_h], fill=(30, 41, 59))
    draw.text(
        (12, 12),
        f"{canvas_path.stem} (Vertical Columns)  |  Read [01]..[{len(items):02d}] top-to-bottom",
        fill=(255, 255, 255),
        font=font_meta,
    )

    x = pad_x // 2
    for slot_tag, item, crop_im, col_w in scaled_cols:
        col_bg = (255, 255, 255) if int(slot_tag[1:-1]) % 2 == 1 else (244, 246, 250)
        draw.rectangle([x, header_h, x + col_w + pad_x - 2, canvas_h], fill=col_bg)
        draw.rectangle([x, header_h, x + col_w + pad_x - 2, header_h + top_badge_h], fill=(238, 242, 255))
        draw.line([(x, header_h + top_badge_h), (x + col_w + pad_x - 2, header_h + top_badge_h)], fill=(79, 70, 229), width=2)
        draw.text((x + 6, header_h + 9), slot_tag, fill=(185, 28, 28), font=font_id)

        paste_x = x + (col_w + pad_x - 2 - crop_im.size[0]) // 2
        paste_y = header_h + top_badge_h + pad_y
        canvas.paste(crop_im, (paste_x, paste_y))
        draw.rectangle(
            [paste_x - 1, paste_y - 1, paste_x + crop_im.size[0], paste_y + crop_im.size[1]],
            outline=(203, 213, 225),
            width=1,
        )
        draw.line([(x + col_w + pad_x - 2, header_h), (x + col_w + pad_x - 2, canvas_h)], fill=(203, 213, 225), width=1)
        x += col_w + pad_x

    canvas_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(canvas_path, compress_level=3)
    return canvas.size


def build_vlm_prompt(canvas_name: str, items: Sequence[Dict], include_hints: bool = True) -> str:
    n = len(items)
    hint_lines: List[str] = []
    if include_hints:
        seen_sources = []
        for it in items:
            src = it.get("source_image", "")
            hints = it.get("source_hints") or []
            if hints and src not in seen_sources:
                seen_sources.append(src)
                hint_lines.append(f"- Candidate strings from {src}: " + " / ".join(hints))

    prompt = [
        f"Inspect every numbered YOLO crop proposal [01] through [{n:02d}] in `{canvas_name}` and transcribe only text actually visible inside it.",
        "",
        "Rules:",
        f"1. Output exactly {n} lines, one for each ID from [01] to [{n:02d}] in numerical order.",
        "2. Use this exact pipe-delimited format per line:",
        "   [ID] | <exact Japanese transcription> | <OK | TRUNCATED | NOISE | UNCERTAIN>",
        "3. Preserve exact orthographic surface forms (kanji vs. kana, full/half-width numerals, ¥ vs. 円). Do not translate or explain.",
        "4. If a crop has a cut-off character at the edge, transcribe the visible characters and mark status `TRUNCATED`.",
        "5. A detected box is NOT guaranteed to contain text. Background, wood grain, shadows, tableware, bottles, decorations and other objects may be false positives. If no text is visible, write `[ID] | | NOISE`; never invent a label to fill a slot.",
        "6. Read only image pixels inside each numbered crop, not the ID/header. Never complete cut-off words from menu knowledge or reference hints.",
        "7. Vertical-column crops read top-to-bottom. Report the ID printed beside the crop, not its spatial position on the sheet.",
        "8. If text is visibly present but unreadable/ambiguous, or overlapping lines cannot be separated confidently, use status `UNCERTAIN`; transcribe only confidently readable characters or leave text empty. Do not guess, and do not call unreadable text a blank background.",
        "9. Mark `OK` only when the full visible line is confidently readable without cut-off glyphs. Class labels and detection confidence are not evidence that text exists; never transcribe contact-sheet badges or headers.",
    ]
    flagged = [f"[{i:02d}]" for i, item in enumerate(items, 1) if item.get('review_required')]
    if flagged:
        prompt.append("Detector core-overlap review flags: " + ", ".join(flagged) + ". Inspect for neighboring text; use UNCERTAIN if the target line is ambiguous.")
    if hint_lines:
        prompt.append("")
        prompt.append("Optional source-sidecar candidates, NOT ground truth: generation hints can differ from the image. Never use them to infer text presence, replace unreadable glyphs, or complete a cut-off word. Pixels take precedence:")
        prompt.extend(hint_lines)
    return "\n".join(prompt) + "\n"


def pack_canvases(
    crops: Sequence[Dict],
    output_root: Path,
    items_per_canvas: int = 25,
    separate_vertical: bool = True,
    include_hints: bool = True,
) -> Dict:
    if items_per_canvas < 1:
        raise ValueError("items_per_canvas must be positive")
    canvases_dir = output_root / "canvases"
    canvases_dir.mkdir(parents=True, exist_ok=True)

    if separate_vertical:
        groups = [
            ("horiz", [c for c in crops if not c["is_vertical"]], False),
            ("vert", [c for c in crops if c["is_vertical"]], True),
        ]
    else:
        groups = [("mixed", list(crops), False)]

    canvas_entries = []
    canvas_counter = 0

    for prefix, group_items, use_vertical_columns in groups:
        for start in range(0, len(group_items), items_per_canvas):
            batch = group_items[start : start + items_per_canvas]
            if not batch:
                continue
            canvas_counter += 1
            canvas_id = f"canvas_{canvas_counter:03d}"
            canvas_rel = f"canvases/{canvas_id}.png"
            prompt_rel = f"canvases/{canvas_id}_prompt.txt"
            canvas_path = output_root / canvas_rel
            prompt_path = output_root / prompt_rel

            if use_vertical_columns:
                size = render_vertical_columns_canvas(batch, output_root, canvas_path)
            else:
                size = render_horizontal_canvas(batch, output_root, canvas_path)

            prompt_text = build_vlm_prompt(f"{canvas_id}.png", batch, include_hints=include_hints)
            prompt_path.write_text(prompt_text, encoding="utf-8")

            slots = []
            for slot_idx, item in enumerate(batch, start=1):
                slot_tag = f"[{slot_idx:02d}]"
                slots.append(
                    {
                        "slot": slot_tag,
                        "slot_index": slot_idx,
                        "crop_id": item["crop_id"],
                        "source_menu": item["source_menu"],
                        "source_image": item["source_image"],
                        "detector_class": item["detector_class"],
                        "confidence": item["confidence"],
                        "is_vertical": item["is_vertical"],
                        "orientation_flag": item["orientation_flag"],
                        "display_crop": item["display_crop"],
                        "ctc_crop": item["ctc_crop"],
                        "image": item["ctc_crop"],
                        "obb_polygon": item["obb_polygon"],
                        "padded_polygon": item.get("padded_polygon", item["obb_polygon"]),
                        **{k:item[k] for k in ('valid_width','canvas','input_length','scaled_height','ctc_sha256','source_ids','padding_scale','review_required','overlap_source_ids') if k in item},
                    }
                )
            canvas_entries.append(
                {
                    "canvas_id": canvas_id,
                    "layout": "vertical_columns" if use_vertical_columns else "horizontal_rows",
                    "image": canvas_rel,
                    "prompt_file": prompt_rel,
                    "size": list(size),
                    "sha256": sha256_file(canvas_path),
                    "num_items": len(batch),
                    "slots": slots,
                }
            )

    manifest = {
        "total_crops": len(crops),
        "total_canvases": len(canvas_entries),
        "items_per_canvas": items_per_canvas,
        "separate_vertical": separate_vertical,
        "canvases": canvas_entries,
    }
    dump_json(output_root / "canvas_manifest.json", manifest)
    return manifest


def run_build(
    input_dir: Path,
    output_dir: Path,
    weights: Path = DEFAULT_WEIGHTS,
    confidence: float = 0.25,
    iou_dedup: float = 0.65,
    imgsz: int = 640,
    device: str = "cpu",
    pad_x: float = 0.04,
    pad_y: float = 0.06,
    items_per_canvas: int = 25,
    separate_vertical: bool = True,
    include_hints: bool = True,
    safe_line_crops: bool = True,
    merge_iom: float = 0.20,
) -> Dict:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    weights = weights.resolve()
    if not weights.exists():
        raise FileNotFoundError(f"YOLO OBB checkpoint not found: {weights}")
    if not 0 <= confidence <= 1 or not 0 < iou_dedup <= 1 or not 0.20 <= merge_iom <= 1:
        raise ValueError("confidence must be 0..1, duplicate IoU 0..1, merge IoM .20..1")
    if items_per_canvas < 1:
        raise ValueError("items_per_canvas must be positive")
    if (output_dir / 'crops_manifest.json').exists() or (output_dir / 'canvas_manifest.json').exists():
        raise FileExistsError("Use a new build output directory to preserve existing crops/labels; use pack to rebuild canvases.")

    raw_image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.stem.endswith("_prediction")
    )
    canonical_stems = {p.stem for p in raw_image_paths if re.match(r"^menu-\d{3}$", p.stem, re.IGNORECASE)}
    image_paths = [
        p for p in raw_image_paths
        if not (re.match(r"^menu_\d{3}_\d+$", p.stem, re.IGNORECASE) and normalize_menu_id(p) in canonical_stems)
    ]
    if not image_paths:
        raise ValueError(f"No menu images found in {input_dir}")
    ids = [normalize_menu_id(p) for p in image_paths]
    if len(ids) != len(set(ids)):
        raise ValueError("Multiple input images map to the same menu ID; remove duplicate inputs before building")

    os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNTIME_ROOT))
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    # Load this environment's compatible torch/torchvision before the supplemental
    # detector packages, which may contain older copies of numerical libraries.
    import torch, torchvision
    if LOCAL_PKGS.is_dir() and str(LOCAL_PKGS) not in sys.path:
        sys.path.append(str(LOCAL_PKGS))
    from ultralytics import YOLO

    torch.set_num_threads(4)
    model = YOLO(str(weights))
    model_sha = sha256_file(weights)

    output_dir.mkdir(parents=True, exist_ok=True)
    crops: List[Dict] = []
    line_processing=[]
    class_counts: Counter = Counter()
    timings_ms: List[float] = []

    for img_idx, img_path in enumerate(image_paths, start=1):
        t0 = time.perf_counter()
        result = model.predict(str(img_path), imgsz=imgsz, conf=confidence, iou=0.7,
                               max_det=300, device=device, half=False, augment=False, verbose=False)[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        timings_ms.append(elapsed_ms)

        with Image.open(img_path) as raw_im:
            rgb_im = raw_im.convert("RGB")
            gray_im = raw_im.convert("L")
            img_w, img_h = rgb_im.size

        hints = load_sidecar_hints(img_path)
        raw_dets: List[Dict] = []
        if result.obb is not None:
            for points, cls_id, score in zip(
                result.obb.xyxyxyxy.cpu().numpy(),
                result.obb.cls.cpu().numpy(),
                result.obb.conf.cpu().numpy(),
            ):
                label = str(model.names[int(cls_id)])
                quad, box_w, box_h, vert = canonicalize_obb_corners(points, label=label)
                padded_quad, pad_w, pad_h = expand_quad(quad, pad_x, pad_y, img_w, img_h)
                raw_dets.append(
                    {
                        "label": label,
                        "class_id": int(cls_id),
                        "confidence": float(score),
                        "obb_polygon": quad.tolist(),
                        "padded_polygon": [[round(float(x), 2), round(float(y), 2)] for x, y in padded_quad.tolist()],
                        "box_size": [round(pad_w, 1), round(pad_h, 1)],
                        "center": quad.mean(axis=0).tolist(),
                        "is_vertical": vert,
                    }
                )

        if safe_line_crops:
            from src.detection.line_crops import process as process_lines
            processed=process_lines([dict(id=str(i),points=d['obb_polygon'],label=d['label'],confidence=d['confidence']) for i,d in enumerate(raw_dets)],img_w,img_h,duplicate_iou=iou_dedup,min_merge_iom=merge_iom)
            line_processing.append(dict(source_image=img_path.name,**processed))
            deduped_dets=[]
            for item in processed['crops']:
                overlaps=sorted({s for conflict in processed['conflicts']
                    if set(item['source_ids']) & set(conflict['source_a']+conflict['source_b'])
                    for s in conflict['source_a']+conflict['source_b']} - set(item['source_ids']))
                vert=bool(item['orientation_flag']);core=np.asarray(item['core']);q=np.asarray(item['crop_polygon'])
                if vert:core=np.roll(core,1,axis=0);q=np.roll(q,1,axis=0)
                w=float((np.linalg.norm(q[1]-q[0])+np.linalg.norm(q[2]-q[3]))/2)
                h=float((np.linalg.norm(q[3]-q[0])+np.linalg.norm(q[2]-q[1]))/2)
                deduped_dets.append(dict(label=item['label'],class_id=1 if vert else 0,confidence=item['confidence'],
                    obb_polygon=core.tolist(),padded_polygon=q.tolist(),box_size=[w,h],center=core.mean(0).tolist(),is_vertical=vert,
                    source_ids=item['source_ids'],padding_scale=item['padding_scale'],
                    review_required=bool(overlaps),overlap_source_ids=overlaps))
        else:
            deduped_dets = deduplicate_detections(raw_dets, iou_threshold=iou_dedup)
        ordered_dets = sort_detections_reading_order(deduped_dets)
        menu_id = normalize_menu_id(img_path)
        for line_idx, det in enumerate(ordered_dets, start=1):
            crop_id = f"{menu_id}__line{line_idx:02d}"
            padded_quad = np.asarray(det["padded_polygon"], dtype=float)
            pad_w, pad_h = det["box_size"]

            # 1. Native-resolution upright RGB crop for VLM contact sheet
            display_crop_im = warp_polygon(rgb_im, padded_quad, (pad_w, pad_h))
            display_rel = f"crops_display/{crop_id}.png"
            display_path = output_dir / display_rel
            display_path.parent.mkdir(parents=True, exist_ok=True)
            display_crop_im.save(display_path, compress_level=1)

            # 2. RGB aspect-preserving H64 bucket export (legacy mode: stretched grayscale).
            normalization={}
            if safe_line_crops:
                from src.detection.line_crops import recognition_input
                strip=display_crop_im.transpose(Image.Transpose.ROTATE_90) if det['is_vertical'] else display_crop_im
                ctc_crop_im,normalization=recognition_input(strip)
            else:
                ctc_size = (64, 384) if det["is_vertical"] else (384, 64)
                ctc_crop_im = warp_polygon(gray_im, padded_quad, ctc_size)
                if det["is_vertical"]:
                    ctc_crop_im = ctc_crop_im.transpose(Image.Transpose.ROTATE_90)
            ctc_rel = f"crops_ctc/{crop_id}.png"
            ctc_path = output_dir / ctc_rel
            ctc_path.parent.mkdir(parents=True, exist_ok=True)
            ctc_crop_im.save(ctc_path, compress_level=1)

            class_counts[det["label"]] += 1
            crops.append(
                {
                    "crop_id": crop_id,
                    "source_menu": menu_id,
                    "source_image": img_path.name,
                    "source_line_index": line_idx,
                    "detector_class": det["label"],
                    "class_id": det["class_id"],
                    "confidence": round(det["confidence"], 4),
                    "is_vertical": det["is_vertical"],
                    "orientation_flag": 1 if det["is_vertical"] else 0,
                    "obb_polygon": det["obb_polygon"],
                    "padded_polygon": det["padded_polygon"],
                    "display_crop": display_rel,
                    "display_size": list(display_crop_im.size),
                    "ctc_crop": ctc_rel,
                    "image": ctc_rel,
                    "ctc_sha256": sha256_file(ctc_path),
                    "source_hints": hints,
                    "source_ids":det.get('source_ids',[]),
                    "padding_scale":det.get('padding_scale'),
                    "review_required":det.get('review_required',False),
                    "overlap_source_ids":det.get('overlap_source_ids',[]),
                    **normalization,
                }
            )

        if img_idx % 10 == 0 or img_idx == len(image_paths):
            print(f"[YOLO-OBB] Processed {img_idx}/{len(image_paths)} images ({len(crops)} crops so far)", flush=True)

    crops_manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "weights": str(weights),
        "weights_sha256": model_sha,
        "weights_selection": DEFAULT_WEIGHTS_REASON if weights == DEFAULT_WEIGHTS.resolve() else "Explicit checkpoint override",
        "confidence": confidence,
        "iou_dedup": iou_dedup,
        "imgsz": imgsz,
        "detector_nms_iou": 0.7,
        "merge_iom": merge_iom if safe_line_crops else None,
        "review_required_crops":sum(c['review_required'] for c in crops),
        "crop_policy_parameters":line_processing[0]['policy'] if line_processing else None,
        "total_images": len(image_paths),
        "total_crops": len(crops),
        "class_counts": dict(class_counts),
        "mean_inference_ms": round(float(np.mean(timings_ms)), 2) if timings_ms else 0.0,
        "crops": crops,
        "crop_policy":"safe_oriented_lines_rgb_buckets" if safe_line_crops else "legacy_aabb_stretched_grayscale",
    }
    if sha256_file(weights) != model_sha:
        raise RuntimeError("Checkpoint changed during build; discard output and retry")
    dump_json(output_dir / "crops_manifest.json", crops_manifest)
    if safe_line_crops:dump_json(output_dir/'line_processing.json',line_processing)

    canvas_manifest = pack_canvases(
        crops=crops,
        output_root=output_dir,
        items_per_canvas=items_per_canvas,
        separate_vertical=separate_vertical,
        include_hints=include_hints,
    )
    print(
        f"[Canvas Pack] Built {canvas_manifest['total_canvases']} canvases for {len(crops)} crops in {output_dir / 'canvases'}",
        flush=True,
    )
    return canvas_manifest


def parse_vlm_transcriptions(
    output_dir: Path,
    responses_path: Path,
    keep_statuses: Sequence[str] = ("OK",),
) -> Dict:
    """Parse VLM responses and join with `canvas_manifest.json` to emit `labeled_crops.jsonl` and `ctc_labels.txt`."""
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "canvas_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"canvas_manifest.json not found in {output_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    current_canvas_id = manifest["canvases"][0]["canvas_id"] if len(manifest["canvases"]) == 1 else None
    parsed_by_canvas_slot: Dict[Tuple[str, int], Tuple[str, str]] = {}
    raw_text = responses_path.read_text(encoding="utf-8")

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m_canvas = re.search(r"(canvas_\d{3})", line, re.IGNORECASE)
        if m_canvas and "|" not in line and not line.startswith("{"):
            current_canvas_id = m_canvas.group(1).lower()
            continue

        if line.startswith("{") and line.endswith("}"):
            obj = json.loads(line)
            cid = str(obj.get("canvas_id") or current_canvas_id or "").lower()
            slot_str = str(obj.get("slot") or obj.get("id") or "")
            m_slot = re.search(r"(\d+)", slot_str)
            if cid and m_slot:
                parsed_by_canvas_slot[(cid, int(m_slot.group(1)))] = (
                    str(obj.get("text", "")).strip(),
                    str(obj.get("status", "OK")).strip().upper(),
                )
            continue

        m_pipe = re.match(
            r"^(?:(canvas_\d{3})\s*)?\[?0*(\d+)\]?\s*\|\s*(.*?)(?:\s*\|\s*(OK|TRUNCATED|NOISE|UNCERTAIN))?\s*$",
            line,
            re.IGNORECASE,
        )
        if m_pipe:
            cid = (m_pipe.group(1) or current_canvas_id or "").lower()
            if not cid:
                raise ValueError(
                    f"Line '{line}' does not specify a canvas_id (`canvas_001`) and multiple canvases exist. "
                    "Add a `# canvas_001` header before each canvas block."
                )
            slot_idx = int(m_pipe.group(2))
            text_val = m_pipe.group(3).strip()
            status_val = (m_pipe.group(4) or ("NOISE" if not text_val else "OK")).upper()
            parsed_by_canvas_slot[(cid, slot_idx)] = (text_val, status_val)

    keep_set = {s.upper() for s in keep_statuses}
    labeled_records = []
    ctc_lines = []
    status_counts: Counter = Counter()

    for canvas in manifest["canvases"]:
        cid = canvas["canvas_id"]
        for slot in canvas["slots"]:
            s_idx = int(slot["slot_index"])
            text_val, status_val = parsed_by_canvas_slot.get((cid, s_idx), ("", "UNLABELED"))
            status_counts[status_val] += 1
            rec = {
                **slot,
                "canvas_id": cid,
                "text": text_val,
                "status": status_val,
            }
            labeled_records.append(rec)
            if status_val in keep_set and text_val:
                ctc_lines.append(f"{slot['ctc_crop']}\t{text_val}")

    (output_dir / "labeled_crops.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in labeled_records) + ("\n" if labeled_records else ""),
        encoding="utf-8",
    )
    (output_dir / "ctc_labels.txt").write_text(
        "\n".join(ctc_lines) + ("\n" if ctc_lines else ""),
        encoding="utf-8",
    )
    summary = {
        "total_slots": len(labeled_records),
        "exported_ctc_lines": len(ctc_lines),
        "status_counts": dict(status_counts),
        "labeled_crops_jsonl": str(output_dir / "labeled_crops.jsonl"),
        "ctc_labels_txt": str(output_dir / "ctc_labels.txt"),
    }
    dump_json(output_dir / "parse_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    p_build = sub.add_parser("build", help="Run YOLO11n-OBB cropping and build VLM contact-sheet canvases")
    p_build.add_argument("--input", type=Path, required=True, help="Directory of menu photos")
    p_build.add_argument("--output", type=Path, required=True, help="Output directory for crops and canvases")
    p_build.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="YOLO OBB checkpoint path (default: selected V5 all130 candidate)")
    p_build.add_argument("--confidence", type=float, default=0.25, help="YOLO confidence threshold")
    p_build.add_argument("--iou-dedup", type=float, default=0.65, help="Rotated duplicate IoU threshold (AABB only in legacy mode)")
    p_build.add_argument('--legacy-crops',action='store_true',help='Reproduce old AABB suppression and stretched384x64grayscale exports')
    p_build.add_argument('--merge-iom',type=float,default=.20,help='Minimum core intersection/smaller area for aligned fragment merging (.20..1); adjacent-only boxes stay separate')
    p_build.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    p_build.add_argument("--device", type=str, default="cpu", help="Inference device (cpu or 0)")
    p_build.add_argument("--pad-x", type=float, default=0.04, help="Legacy mode only: horizontal OBB padding ratio")
    p_build.add_argument("--pad-y", type=float, default=0.06, help="Legacy mode only: vertical OBB padding ratio")
    p_build.add_argument("--items-per-canvas", type=int, default=25, help="Max crops packed per contact sheet")
    p_build.add_argument(
        "--mixed-orientation",
        action="store_true",
        help="Pack vertical and horizontal crops on the same row-based canvas instead of separating vertical columns",
    )
    p_build.add_argument("--no-hints", action="store_true", help="Do not include sidecar gt_lines in prompt files")

    p_pack = sub.add_parser("pack", help="Repack existing crops_manifest.json into contact-sheet canvases")
    p_pack.add_argument("--output", type=Path, required=True, help="Pipeline output directory containing crops_manifest.json")
    p_pack.add_argument("--items-per-canvas", type=int, default=25)
    p_pack.add_argument("--mixed-orientation", action="store_true")
    p_pack.add_argument("--no-hints", action="store_true")

    p_parse = sub.add_parser("parse", help="Parse VLM responses into labeled_crops.jsonl and ctc_labels.txt")
    p_parse.add_argument("--output", type=Path, required=True, help="Pipeline output directory containing canvas_manifest.json")
    p_parse.add_argument("--responses", type=Path, required=True, help="Text or JSONL file of VLM transcriptions")
    p_parse.add_argument(
        "--keep-status",
        nargs="+",
        default=["OK"],
        help="Statuses to include in ctc_labels.txt (default: OK)",
    )

    args = parser.parse_args()
    if args.command == "build" or args.command is None:
        if getattr(args, "input", None) is None or getattr(args, "output", None) is None:
            parser.error("`build` requires --input and --output")
        run_build(
            input_dir=args.input,
            output_dir=args.output,
            weights=args.weights,
            confidence=args.confidence,
            iou_dedup=args.iou_dedup,
            imgsz=args.imgsz,
            device=args.device,
            pad_x=args.pad_x,
            pad_y=args.pad_y,
            items_per_canvas=args.items_per_canvas,
            separate_vertical=not args.mixed_orientation,
            include_hints=not args.no_hints,
            safe_line_crops=not args.legacy_crops,
            merge_iom=args.merge_iom,
        )
    elif args.command == "pack":
        crops_data = json.loads((args.output / "crops_manifest.json").read_text(encoding="utf-8"))
        pack_canvases(
            crops=crops_data["crops"],
            output_root=args.output,
            items_per_canvas=args.items_per_canvas,
            separate_vertical=not args.mixed_orientation,
            include_hints=not args.no_hints,
        )
    elif args.command == "parse":
        parse_vlm_transcriptions(
            output_dir=args.output,
            responses_path=args.responses,
            keep_statuses=args.keep_status,
        )


if __name__ == "__main__":
    main()
