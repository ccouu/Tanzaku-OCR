"""Modular YOLO V5 native-resolution quality filtering and static review gallery for Tabelog menus.

Measures physical source-resolution core OBB geometry BEFORE any recognizer H64 normalization.
Never upscales or lets padded output thickness inflate core short-side measurements.
"""
from __future__ import annotations

import hashlib
import html
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.common import ROOT, dump_json, safe_path
from src.detection.line_crops import area as poly_area, process as process_lines, rectify

DEFAULT_YOLO_WEIGHTS = ROOT / "outputs/detection/training/banana130_finetune_v5/yolo11n_menu_obb_banana130_best.pt"
EXPECTED_YOLO_SHA_PREFIX = "0cab3e12"
LOCAL_PKGS = ROOT / ".stage2_packages"
YOLO_RUNTIME_ROOT = ROOT / ".cache/ultralytics"

CONFIDENCE_DEFAULT = 0.35
ACCEPTED_MIN_SHORT_PX = 24.0
ACCEPTED_MIN_LONG_PX = 24.0
BORDERLINE_MIN_SHORT_PX = 16.0
CLOSEUP_MIN_SHORT_PX = 32.0
MIN_ACCEPTED_FOR_PHOTO = 3
MIN_CLOSEUP_FOR_PHOTO = 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_exif_transposed_rgb(blob: bytes) -> tuple[Image.Image, dict[str, Any]]:
    """Decode image bytes, preserve EXIF provenance, and apply EXIF transpose before inference/cropping."""
    with Image.open(io.BytesIO(blob)) as raw_im:
        raw_im.verify()
    with Image.open(io.BytesIO(blob)) as raw_im:
        fmt = raw_im.format
        raw_w, raw_h = raw_im.size
        exif_orientation = 1
        try:
            exif = raw_im.getexif()
            if exif and 274 in exif:
                exif_orientation = int(exif[274])
        except Exception:
            exif_orientation = 1
        transposed = ImageOps.exif_transpose(raw_im)
        rgb = transposed.convert("RGB")
        w, h = rgb.size
    return rgb, {
        "format": fmt,
        "raw_width": raw_w,
        "raw_height": raw_h,
        "exif_orientation": exif_orientation,
        "exif_transposed": exif_orientation not in (1, 0),
        "width": w,
        "height": h,
    }


def measure_quad_native(quad: Any) -> dict[str, Any]:
    """Measure physical native-pixel geometry of a 4-point quad before any resizing."""
    try:
        q = np.asarray(quad, dtype=float)
    except Exception:
        return {"valid": False, "reason": "non_array_quad", "short_px": 0.0, "long_px": 0.0, "span_px": 0.0, "thickness_px": 0.0, "area_px2": 0.0}
    if q.shape != (4, 2) or not np.all(np.isfinite(q)):
        return {"valid": False, "reason": "non_finite_or_bad_shape", "short_px": 0.0, "long_px": 0.0, "span_px": 0.0, "thickness_px": 0.0, "area_px2": 0.0}

    span_px = 0.5 * (float(np.linalg.norm(q[1] - q[0])) + float(np.linalg.norm(q[2] - q[3])))
    thickness_px = 0.5 * (float(np.linalg.norm(q[3] - q[0])) + float(np.linalg.norm(q[2] - q[1])))
    area_px2 = float(poly_area(q))
    short_px = min(span_px, thickness_px)
    long_px = max(span_px, thickness_px)
    valid = (
        math.isfinite(span_px)
        and math.isfinite(thickness_px)
        and math.isfinite(area_px2)
        and short_px > 1e-6
        and long_px > 1e-6
        and area_px2 > 1e-6
    )
    return {
        "valid": valid,
        "reason": "ok" if valid else "degenerate_geometry",
        "short_px": round(short_px, 3),
        "long_px": round(long_px, 3),
        "span_px": round(span_px, 3),
        "thickness_px": round(thickness_px, 3),
        "area_px2": round(area_px2, 2),
    }


def classify_crop_resolution(
    core_quad: Any,
    padded_quad: Any | None = None,
    *,
    accepted_min_short_px: float = ACCEPTED_MIN_SHORT_PX,
    accepted_min_long_px: float = ACCEPTED_MIN_LONG_PX,
    borderline_min_short_px: float = BORDERLINE_MIN_SHORT_PX,
    closeup_min_short_px: float = CLOSEUP_MIN_SHORT_PX,
) -> dict[str, Any]:
    """Classify a single line crop strictly from its native core geometry (never padded thickness)."""
    core_m = measure_quad_native(core_quad)
    pad_m = measure_quad_native(padded_quad) if padded_quad is not None else core_m
    if not core_m["valid"]:
        return {
            "category": "invalid_geometry",
            "is_closeup_32px": False,
            "core_short_px": core_m["short_px"],
            "core_long_px": core_m["long_px"],
            "core_span_px": core_m["span_px"],
            "core_thickness_px": core_m["thickness_px"],
            "core_area_px2": core_m["area_px2"],
            "padded_short_px": pad_m["short_px"],
            "padded_long_px": pad_m["long_px"],
            "geometry_reason": core_m["reason"],
        }

    s = core_m["short_px"]
    l = core_m["long_px"]
    if s >= accepted_min_short_px and l >= accepted_min_long_px:
        cat = "accepted"
        closeup = s >= closeup_min_short_px
    elif s >= borderline_min_short_px:
        cat = "borderline"
        closeup = False
    else:
        cat = "too_small"
        closeup = False

    return {
        "category": cat,
        "is_closeup_32px": closeup,
        "core_short_px": s,
        "core_long_px": l,
        "core_span_px": core_m["span_px"],
        "core_thickness_px": core_m["thickness_px"],
        "core_area_px2": core_m["area_px2"],
        "padded_short_px": pad_m["short_px"],
        "padded_long_px": pad_m["long_px"],
        "geometry_reason": "ok",
    }


def _crop_diagnostics(crop_rgb: Image.Image) -> dict[str, float]:
    """Record contrast & gradient sharpness diagnostics only (never used for automatic rejection)."""
    arr = np.asarray(crop_rgb.convert("L"), dtype=np.float32)
    contrast_std = float(arr.std()) if arr.size else 0.0
    if arr.shape[0] >= 3 and arr.shape[1] >= 3:
        lap = (
            -4.0 * arr[1:-1, 1:-1]
            + arr[:-2, 1:-1]
            + arr[2:, 1:-1]
            + arr[1:-1, :-2]
            + arr[1:-1, 2:]
        )
        sharp_var = float(lap.var())
    else:
        sharp_var = 0.0
    return {"diag_contrast_std": round(contrast_std, 2), "diag_sharpness_var": round(sharp_var, 2)}


def evaluate_detections_for_photo(
    rgb_image: Image.Image,
    raw_detections: Sequence[dict[str, Any]],
    *,
    accepted_min_short_px: float = ACCEPTED_MIN_SHORT_PX,
    accepted_min_long_px: float = ACCEPTED_MIN_LONG_PX,
    borderline_min_short_px: float = BORDERLINE_MIN_SHORT_PX,
    closeup_min_short_px: float = CLOSEUP_MIN_SHORT_PX,
    min_accepted_for_photo: int = MIN_ACCEPTED_FOR_PHOTO,
    min_closeup_for_photo: int = MIN_CLOSEUP_FOR_PHOTO,
    rectify_crops: bool = True,
) -> dict[str, Any]:
    """Process OBB detections through shared `line_crops.process`, classify native core geometry, and decide photo acceptance."""
    width, height = rgb_image.size
    valid_raw: list[dict[str, Any]] = []
    invalid_raw_records: list[dict[str, Any]] = []

    for idx, d in enumerate(raw_detections):
        m = measure_quad_native(d.get("points"))
        if not m["valid"]:
            invalid_raw_records.append(
                {
                    "crop_id": f"invalid_raw_{idx:03d}",
                    "label": str(d.get("label", "unknown")),
                    "confidence": float(d.get("confidence", 0.0)),
                    "orientation_flag": int(str(d.get("label", "")) == "dish_vertical"),
                    "category": "invalid_geometry",
                    "is_closeup_32px": False,
                    "core_short_px": m["short_px"],
                    "core_long_px": m["long_px"],
                    "padded_short_px": 0.0,
                    "padded_long_px": 0.0,
                    "geometry_reason": m["reason"],
                    "core_polygon": d.get("points"),
                    "crop_polygon": None,
                    "saved_crop_relpath": None,
                }
            )
            continue
        valid_raw.append(d)

    processed = process_lines(valid_raw, width, height) if valid_raw else {"crops": [], "decisions": [], "conflicts": [], "raw_count": 0, "processed_count": 0}

    crop_records: list[dict[str, Any]] = []
    crop_images: dict[str, Image.Image] = {}
    counts = {"accepted": 0, "closeup_32px": 0, "borderline": 0, "too_small": 0, "invalid_geometry": len(invalid_raw_records)}

    for c in processed["crops"]:
        cls = classify_crop_resolution(
            c["core"],
            c["crop_polygon"],
            accepted_min_short_px=accepted_min_short_px,
            accepted_min_long_px=accepted_min_long_px,
            borderline_min_short_px=borderline_min_short_px,
            closeup_min_short_px=closeup_min_short_px,
        )
        cat = cls["category"]
        counts[cat] = counts.get(cat, 0) + 1
        if cls["is_closeup_32px"]:
            counts["closeup_32px"] += 1

        rec: dict[str, Any] = {
            "crop_id": c["id"],
            "label": c["label"],
            "confidence": round(float(c["confidence"]), 4),
            "orientation_flag": int(c["orientation_flag"]),
            "source_ids": c["source_ids"],
            "padding_scale": round(float(c["padding_scale"]), 4),
            "overlap_severity": c.get("overlap_severity"),
            "review_required": bool(c.get("review_required", False)),
            "core_polygon": [[round(float(x), 2), round(float(y), 2)] for x, y in c["core"]],
            "crop_polygon": [[round(float(x), 2), round(float(y), 2)] for x, y in c["crop_polygon"]],
            **cls,
            "native_rectified_width": None,
            "native_rectified_height": None,
            "saved_crop_relpath": None,
        }

        if rectify_crops and cat in ("accepted", "borderline"):
            try:
                strip = rectify(rgb_image, c["crop_polygon"])
                rec["native_rectified_width"] = strip.width
                rec["native_rectified_height"] = strip.height
                rec.update(_crop_diagnostics(strip))
                crop_images[c["id"]] = strip
            except Exception as exc:
                rec["rectify_error"] = type(exc).__name__

        crop_records.append(rec)

    crop_records.extend(invalid_raw_records)

    has_acc = counts["accepted"] >= min_accepted_for_photo
    has_cu = counts["closeup_32px"] >= min_closeup_for_photo
    photo_accepted = has_acc or has_cu
    if has_acc and has_cu:
        decision_reason = "accepted_count_ge_3_and_closeup_32px_ge_1"
    elif has_acc:
        decision_reason = "accepted_count_ge_3"
    elif has_cu:
        decision_reason = "closeup_32px_ge_1"
    elif len(raw_detections) == 0:
        decision_reason = "no_yolo_detections"
    elif counts["borderline"] > 0:
        decision_reason = "borderline_only_below_threshold"
    else:
        decision_reason = "too_small_or_invalid_only"

    return {
        "photo_accepted": photo_accepted,
        "decision_reason": decision_reason,
        "raw_detection_count": len(raw_detections),
        "processed_line_count": len(processed["crops"]),
        "counts": counts,
        "crop_records": crop_records,
        "crop_images": crop_images,
    }


class YoloMenuQualityEvaluator:
    """Reusable YOLO11n-OBB V5 native-resolution menu quality evaluator."""

    def __init__(
        self,
        weights_path: Path = DEFAULT_YOLO_WEIGHTS,
        *,
        expected_sha_prefix: str | None = EXPECTED_YOLO_SHA_PREFIX,
        confidence: float = CONFIDENCE_DEFAULT,
        imgsz: int = 1024,
        device: str | None = None,
        detector_fn: Callable[[Image.Image], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.weights_path = Path(weights_path).resolve()
        self.confidence = float(confidence)
        self.imgsz = int(imgsz)
        self._detector_fn = detector_fn
        self._model = None
        self.device = device

        if detector_fn is None:
            if not self.weights_path.is_file():
                raise FileNotFoundError(f"YOLO V5 checkpoint not found: {self.weights_path}")
            self.weights_sha256 = sha256_file(self.weights_path)
            if expected_sha_prefix and not self.weights_sha256.startswith(expected_sha_prefix):
                raise ValueError(
                    f"YOLO weights SHA mismatch: got {self.weights_sha256[:16]}, expected prefix {expected_sha_prefix}"
                )
            self._init_yolo()
        else:
            self.weights_sha256 = "mock_detector"

    def _init_yolo(self) -> None:
        os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_RUNTIME_ROOT))
        os.environ.setdefault("YOLO_OFFLINE", "true")
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        import torch  # noqa: F401
        import torchvision  # noqa: F401

        if LOCAL_PKGS.is_dir() and str(LOCAL_PKGS) not in sys.path:
            sys.path.append(str(LOCAL_PKGS))
        from ultralytics import YOLO

        if self.device is None:
            self.device = "0" if torch.cuda.is_available() else "cpu"
        self._model = YOLO(str(self.weights_path))

    def policy_dict(self) -> dict[str, Any]:
        return {
            "weights_path": str(self.weights_path),
            "weights_sha256": self.weights_sha256,
            "confidence_threshold": self.confidence,
            "imgsz": self.imgsz,
            "device": str(self.device),
            "measurement_stage": "native_exif_transposed_core_obb_before_h64_resize",
            "accepted_min_core_short_px": ACCEPTED_MIN_SHORT_PX,
            "accepted_min_core_long_px": ACCEPTED_MIN_LONG_PX,
            "borderline_min_core_short_px": BORDERLINE_MIN_SHORT_PX,
            "closeup_min_core_short_px": CLOSEUP_MIN_SHORT_PX,
            "photo_accept_rule": f"accepted_count >= {MIN_ACCEPTED_FOR_PHOTO} OR closeup_32px_count >= {MIN_CLOSEUP_FOR_PHOTO}",
            "automatic_blur_rejection": False,
        }

    def detect_rgb(self, rgb_image: Image.Image) -> list[dict[str, Any]]:
        if self._detector_fn is not None:
            return list(self._detector_fn(rgb_image))
        assert self._model is not None
        res = self._model.predict(
            rgb_image,
            imgsz=self.imgsz,
            conf=self.confidence,
            iou=0.7,
            max_det=300,
            device=self.device,
            half=False,
            augment=False,
            verbose=False,
        )[0]
        raw_dets: list[dict[str, Any]] = []
        if res.obb is not None:
            for idx, (pts, cls_id, score) in enumerate(
                zip(
                    res.obb.xyxyxyxy.cpu().numpy(),
                    res.obb.cls.cpu().numpy(),
                    res.obb.conf.cpu().numpy(),
                )
            ):
                label = str(self._model.names[int(cls_id)])
                raw_dets.append(
                    {
                        "id": f"det_{idx:03d}",
                        "label": label,
                        "confidence": float(score),
                        "points": np.asarray(pts, dtype=float).tolist(),
                    }
                )
        return raw_dets

    def evaluate_bytes(self, blob: bytes) -> dict[str, Any]:
        rgb, img_meta = load_exif_transposed_rgb(blob)
        raw_dets = self.detect_rgb(rgb)
        ev = evaluate_detections_for_photo(rgb, raw_dets, rectify_crops=True)
        ev["image_meta"] = img_meta
        return ev


def save_accepted_photo_crops(
    dataset_root: Path,
    restaurant_id: str,
    photo_filename: str,
    photo_sha256: str,
    source_meta: dict[str, Any],
    eval_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Save native-resolution accepted and borderline crops; return enriched crop JSONL records for ALL detections."""
    acc_dir = dataset_root / "crops/accepted"
    bor_dir = dataset_root / "crops/borderline"
    acc_dir.mkdir(parents=True, exist_ok=True)
    bor_dir.mkdir(parents=True, exist_ok=True)

    crop_images: dict[str, Image.Image] = eval_result.get("crop_images", {})
    enriched: list[dict[str, Any]] = []
    for rec in eval_result["crop_records"]:
        row = dict(rec)
        row["restaurant_id"] = restaurant_id
        row["source_image"] = photo_filename
        row["source_sha256"] = photo_sha256
        row["restaurant_url"] = source_meta.get("restaurant_url")
        row["gallery_url"] = source_meta.get("gallery_url")
        row["image_url"] = source_meta.get("image_url")
        cat = row["category"]
        cid = row["crop_id"]
        if cat in ("accepted", "borderline") and cid in crop_images:
            sub = "accepted" if cat == "accepted" else "borderline"
            fname = f"menu_{restaurant_id}_{cid}.png"
            out_path = dataset_root / "crops" / sub / fname
            crop_images[cid].save(out_path, format="PNG")
            row["saved_crop_relpath"] = f"crops/{sub}/{fname}"
        enriched.append(row)
    return enriched


def _build_crop_contact_sheets(
    dataset_root: Path,
    crops: Sequence[dict[str, Any]],
    sheet_prefix: str,
    out_dir: Path,
    per_page: int = 36,
) -> list[str]:
    if not crops:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    cols, cw, ch = 4, 300, 110
    pages: list[str] = []
    for offset in range(0, len(crops), per_page):
        subset = crops[offset : offset + per_page]
        rows = (len(subset) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cw, rows * ch), "white")
        draw = ImageDraw.Draw(sheet)
        for idx, c in enumerate(subset):
            rel = c.get("saved_crop_relpath")
            if not rel:
                continue
            p = safe_path(dataset_root, rel)
            if not p.is_file():
                continue
            gx, gy = (idx % cols) * cw, (idx // cols) * ch
            with Image.open(p) as im:
                thumb = ImageOps.contain(im.convert("RGB"), (cw - 14, ch - 30))
                sheet.paste(thumb, (gx + 7, gy + 4))
            lbl = f"{c['restaurant_id']} {c['crop_id']} {c['core_short_px']:.0f}x{c['core_long_px']:.0f}px c={c['confidence']:.2f}"
            draw.text((gx + 7, gy + ch - 22), lbl[:44], fill="black")
        fname = f"{sheet_prefix}_{offset // per_page + 1:03d}.jpg"
        sheet.save(out_dir / fname, quality=88)
        pages.append(f"contact_sheets/{fname}")
    return pages


def build_review_site(dataset_root: Path, manifest: dict[str, Any]) -> Path:
    """Build static review/index.html with thumbnails, crop previews, contact sheets, and relative links."""
    review_dir = dataset_root / "review"
    thumbs_dir = review_dir / "thumbs"
    sheets_dir = review_dir / "contact_sheets"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    crops_jsonl = dataset_root / "crops_manifest.jsonl"
    all_crops: list[dict[str, Any]] = []
    if crops_jsonl.is_file():
        for line in crops_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                all_crops.append(json.loads(line))

    by_rest: dict[str, list[dict[str, Any]]] = {}
    for c in all_crops:
        by_rest.setdefault(str(c["restaurant_id"]), []).append(c)

    accepted_crops = [c for c in all_crops if c.get("category") == "accepted" and c.get("saved_crop_relpath")]
    borderline_crops = [c for c in all_crops if c.get("category") == "borderline" and c.get("saved_crop_relpath")]
    acc_sheets = _build_crop_contact_sheets(dataset_root, accepted_crops, "accepted", sheets_dir)
    bor_sheets = _build_crop_contact_sheets(dataset_root, borderline_crops, "borderline", sheets_dir)

    records = manifest.get("images", [])
    rows_html: list[str] = []
    for idx, r in enumerate(records, start=1):
        rid = str(r["restaurant_id"])
        img_rel = str(r["image"])
        src_p = safe_path(dataset_root, img_rel)
        thumb_name = f"thumb_{rid}.jpg"
        thumb_p = thumbs_dir / thumb_name
        if src_p.is_file() and not thumb_p.is_file():
            with Image.open(src_p) as im:
                t = ImageOps.contain(ImageOps.exif_transpose(im).convert("RGB"), (320, 260))
                t.save(thumb_p, quality=85)

        q = r.get("quality", {})
        cnts = q.get("counts", {})
        rcrops = [c for c in by_rest.get(rid, []) if c.get("saved_crop_relpath")]
        crop_chips: list[str] = []
        for c in rcrops[:12]:
            c_rel = c["saved_crop_relpath"]
            badge = "acc" if c["category"] == "accepted" else "bor"
            cu = " 32+" if c.get("is_closeup_32px") else ""
            crop_chips.append(
                f'<a class="chip {badge}" href="../{html.escape(c_rel)}" target="_blank">'
                f'<img loading="lazy" src="../{html.escape(c_rel)}" alt="{html.escape(c["crop_id"])}">'
                f'<span>{html.escape(c["crop_id"])}: {c["core_short_px"]:.0f}x{c["core_long_px"]:.0f}px ({c["category"]}{cu})</span></a>'
            )
        more_note = f"<div class='more'>+{len(rcrops)-12} more saved crops in crops_manifest.jsonl</div>" if len(rcrops) > 12 else ""

        rows_html.append(
            f"<tr>"
            f"<td>{idx}</td>"
            f"<td><a href='../{html.escape(img_rel)}' target='_blank'><img loading='lazy' class='src-thumb' src='thumbs/{html.escape(thumb_name)}' alt='{html.escape(rid)}'></a>"
            f"<div class='meta'>{r.get('width')}x{r.get('height')} (EXIF {r.get('exif_orientation', 1)})</div></td>"
            f"<td><b>{html.escape(str(r.get('restaurant_name', rid)))}</b><br>"
            f"ID: <code>{html.escape(rid)}</code><br>"
            f"<a href='{html.escape(str(r.get('restaurant_url', '#')))}' target='_blank'>Tabelog Profile</a> | "
            f"<a href='{html.escape(str(r.get('gallery_url', '#')))}' target='_blank'>Menu Gallery</a> | "
            f"<a href='../{html.escape(img_rel)}' target='_blank'>Full Image</a><br>"
            f"<div class='meta'>Accepted: <b>{cnts.get('accepted', 0)}</b> (32px+: {cnts.get('closeup_32px', 0)}) | "
            f"Borderline: {cnts.get('borderline', 0)} | Small: {cnts.get('too_small', 0)} | "
            f"Rule: {html.escape(str(q.get('decision_reason', '')))}</div></td>"
            f"<td><div class='chips'>{''.join(crop_chips)}</div>{more_note}</td>"
            f"</tr>"
        )

    acc_sheet_links = " ".join(
        f"<a href='{html.escape(p)}' target='_blank'>Page {i+1}</a>" for i, p in enumerate(acc_sheets)
    ) or "None"
    bor_sheet_links = " ".join(
        f"<a href='{html.escape(p)}' target='_blank'>Page {i+1}</a>" for i, p in enumerate(bor_sheets)
    ) or "None"

    pol = manifest.get("quality_policy", {})
    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Tabelog YOLO Quality-Filtered Review ({len(records)} Restaurants)</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 20px; background: #f8f9fa; color: #1f2328; }}
.summary {{ background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 14px 18px; margin-bottom: 18px; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; }}
th, td {{ border: 1px solid #d0d7de; padding: 8px; vertical-align: top; text-align: left; }}
th {{ background: #f0f3f6; position: sticky; top: 0; }}
.src-thumb {{ max-width: 240px; max-height: 200px; display: block; border: 1px solid #ccc; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.chip {{ display: inline-flex; flex-direction: column; border: 1px solid #ccc; padding: 4px; border-radius: 4px; text-decoration: none; color: #222; font-size: 11px; background: #fafafa; max-width: 220px; }}
.chip.acc {{ border-color: #2da44e; background: #f0fff4; }}
.chip.bor {{ border-color: #d29922; background: #fff8c5; }}
.chip img {{ max-height: 44px; max-width: 200px; object-fit: contain; }}
.meta {{ font-size: 12px; color: #57606a; margin-top: 4px; }}
.more {{ font-size: 11px; color: #57606a; margin-top: 4px; }}
</style>
</head>
<body>
<h1>Tabelog YOLO V5 Quality-Filtered Menu Review</h1>
<div class="summary">
  <b>Status:</b> {html.escape(str(manifest.get('status', 'unknown')))} |
  <b>Accepted Restaurants:</b> {len(records)} / {manifest.get('requested_restaurants')} |
  <b>Accepted Crops:</b> {len(accepted_crops)} |
  <b>Borderline Crops:</b> {len(borderline_crops)}<br>
  <b>Policy:</b> core short side &gt;= {pol.get('accepted_min_core_short_px', 24)}px (&gt;= {pol.get('accepted_min_core_long_px', 24)}px long) accepted;
  {pol.get('borderline_min_core_short_px', 16)}..{pol.get('accepted_min_core_short_px', 24)}px borderline;
  photo accepted if &gt;=3 accepted OR &gt;=1 &gt;=32px closeup; conf &gt;= {pol.get('confidence_threshold', 0.35)}.<br>
  <b>YOLO V5 SHA-256:</b> <code>{html.escape(str(pol.get('weights_sha256', 'unknown')))}</code><br>
  <b>Accepted Crop Contact Sheets:</b> {acc_sheet_links}<br>
  <b>Borderline Crop Contact Sheets:</b> {bor_sheet_links}
</div>
<table>
<thead><tr><th>#</th><th>Source Photo</th><th>Restaurant &amp; Quality Summary</th><th>Native Crops (Accepted / Borderline)</th></tr></thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>
</body>
</html>
"""
    index_path = review_dir / "index.html"
    index_path.write_text(doc, encoding="utf-8")
    write_json_atomic(
        review_dir / "review_manifest.json",
        {
            "accepted_restaurants": len(records),
            "accepted_crops": len(accepted_crops),
            "borderline_crops": len(borderline_crops),
            "accepted_contact_sheets": acc_sheets,
            "borderline_contact_sheets": bor_sheets,
            "index_html": "review/index.html",
        },
    )
    return index_path
