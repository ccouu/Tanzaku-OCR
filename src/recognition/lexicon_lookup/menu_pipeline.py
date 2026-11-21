"""Reusable full-menu photo -> V5 YOLO11n-OBB -> refined crops -> OCR/trie/translation pipeline.

Completely decoupled from HTTP/UI:
- Decodes JPEG/PNG/WebP with EXIF transpose before detection so source pixel
  coordinates match the displayed photo 1:1.
- Gracefully rejects unsupported formats (including HEIC/HEIF when unavailable)
  and oversized uploads/pixel dimensions with actionable errors.
- Reuses `src/detection/line_crops.py` (`process` and `rectify`) directly — zero
  duplicate crop math, zero legacy 512-height-letterbox shrinking.
- Preserves vertical orientation contract: `line_crops.rectify` rotates vertical
  crops CCW once; `lookup_image(..., already_rectified_strip=True)` never double-rotates.
- Isolates per-crop exceptions so one bad crop never fails the entire menu.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image, ImageOps, UnidentifiedImageError

from ...detection.line_crops import process as process_lines, rectify
from ..long_line_inference import native_strip
from .contracts import LookupConfig
from .index import sha256_file
from .pipeline import LexiconPipeline

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_V5_YOLO_WEIGHTS = ROOT / "outputs/detection/training/banana130_finetune_v5/yolo11n_menu_obb_banana130_best.pt"
LOCAL_YOLO_PKGS = ROOT / ".stage2_packages"

MAX_MENU_UPLOAD_BYTES = 16 * 1024 * 1024  # 16 MB
MAX_MENU_PIXELS = 24_000_000  # e.g. 6000 x 4000
MAX_CROPS_PER_MENU = 120
ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})


def _is_heic_header(data: bytes) -> bool:
    if len(data) < 12:
        return False
    if data[4:8] != b"ftyp":
        return False
    brand = data[8:12].lower()
    return brand in (b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"heim", b"heis")


def _pil_to_data_url(im: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        im.convert("RGB").save(buf, format="JPEG", quality=88)
        mime = "image/jpeg"
    else:
        im.save(buf, format="PNG")
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def decode_uploaded_image_bytes(
    image_bytes: bytes,
    *,
    max_bytes: int = MAX_MENU_UPLOAD_BYTES,
    max_pixels: int = MAX_MENU_PIXELS,
) -> tuple[Image.Image, dict[str, Any]]:
    """Decode JPEG/PNG/WebP bytes with EXIF transpose and strict size/pixel bounds."""
    if not image_bytes:
        raise ValueError("Uploaded image is empty")
    if len(image_bytes) > max_bytes:
        raise OverflowError(f"Uploaded image exceeds {max_bytes // (1024 * 1024)} MB limit")
    if _is_heic_header(image_bytes):
        raise ValueError(
            "HEIC/HEIF image detected, but no HEIC decoder is enabled in this local Python environment. "
            "Please convert/export the photo as JPEG, PNG, or WebP and re-upload."
        )

    prev_max = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        try:
            raw_im = Image.open(io.BytesIO(image_bytes))
            fmt = (raw_im.format or "").upper()
            if fmt not in ALLOWED_FORMATS:
                raise ValueError(
                    f"Unsupported image format {fmt or 'unknown'!r}. Supported formats: JPEG, PNG, WebP."
                )
            raw_w, raw_h = raw_im.size
            if raw_w < 2 or raw_h < 2:
                raise ValueError(f"Image dimensions too small: {raw_w}x{raw_h}")
            if raw_w * raw_h > max_pixels:
                raise ValueError(
                    f"Decoded image has {raw_w * raw_h:,} pixels, exceeding the {max_pixels:,}-pixel limit."
                )
            exif_ori = None
            try:
                exif = raw_im.getexif()
                exif_ori = int(exif.get(274)) if exif and 274 in exif else None
            except Exception:
                exif_ori = None
            transposed = ImageOps.exif_transpose(raw_im)
            rgb = transposed.convert("RGB")
        except Image.DecompressionBombError as exc:
            raise ValueError(f"Image exceeds decoded-pixel safety limit ({max_pixels:,} px): {exc}") from exc
        except UnidentifiedImageError as exc:
            raise ValueError(
                "Could not decode image file. Please upload a valid JPEG, PNG, or WebP photo."
            ) from exc
    finally:
        Image.MAX_IMAGE_PIXELS = prev_max

    meta = {
        "format": fmt,
        "raw_width": int(raw_w),
        "raw_height": int(raw_h),
        "exif_orientation": exif_ori,
        "exif_transposed": exif_ori not in (None, 1),
        "width": int(rgb.width),
        "height": int(rgb.height),
        "bytes": len(image_bytes),
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
    }
    return rgb, meta


class V5YoloDetector:
    """Lazy, thread-safe wrapper around local V5 YOLO11n-OBB detector."""

    def __init__(
        self,
        weights_path: str | Path = DEFAULT_V5_YOLO_WEIGHTS,
        *,
        device: str = "cpu",
        imgsz: int = 640,
        iou: float = 0.70,
        max_det: int = 300,
    ) -> None:
        self._weights_path = Path(weights_path).resolve()
        if not self._weights_path.is_file():
            raise FileNotFoundError(f"V5 YOLO weights not found: {self._weights_path}")
        self._weights_sha256 = sha256_file(self._weights_path)
        self._device = str(device)
        self._imgsz = int(imgsz)
        self._iou = float(iou)
        self._max_det = int(max_det)
        self._lock = threading.Lock()
        self._yolo: Any = None

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "weights_path": str(self._weights_path),
            "weights_name": self._weights_path.name,
            "weights_sha256": self._weights_sha256,
            "imgsz": self._imgsz,
            "iou": self._iou,
            "max_det": self._max_det,
            "device": self._device,
        }

    def _ensure_loaded(self) -> Any:
        if self._yolo is not None:
            return self._yolo
        os.environ.setdefault("YOLO_OFFLINE", "true")
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        if LOCAL_YOLO_PKGS.is_dir() and str(LOCAL_YOLO_PKGS) not in sys.path:
            sys.path.append(str(LOCAL_YOLO_PKGS))
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not available in this Python environment and was not found in "
                f"{LOCAL_YOLO_PKGS}. Full-menu detection requires ultralytics."
            ) from exc
        self._yolo = YOLO(str(self._weights_path))
        return self._yolo

    def detect(self, rgb_image: Image.Image, *, confidence: float = 0.35) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Run V5 YOLO11n-OBB on an EXIF-transposed RGB PIL image."""
        if not (0.01 <= confidence <= 0.99):
            raise ValueError(f"confidence must be in [0.01, 0.99], got {confidence}")
        t0 = time.perf_counter()
        with self._lock:
            yolo = self._ensure_loaded()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                tmp_path = Path(tf.name)
                rgb_image.save(tmp_path)
            try:
                res = yolo.predict(
                    str(tmp_path),
                    imgsz=self._imgsz,
                    conf=float(confidence),
                    iou=self._iou,
                    max_det=self._max_det,
                    device=self._device,
                    half=False,
                    augment=False,
                    verbose=False,
                )[0]
            finally:
                tmp_path.unlink(missing_ok=True)

            raw_dets: list[dict[str, Any]] = []
            if res.obb is not None:
                pts_arr = res.obb.xyxyxyxy.cpu().numpy()
                cls_arr = res.obb.cls.cpu().numpy()
                conf_arr = res.obb.conf.cpu().numpy()
                for idx, (pts, cls_id, score) in enumerate(zip(pts_arr, cls_arr, conf_arr)):
                    raw_dets.append(
                        {
                            "id": f"raw_{idx:03d}",
                            "label": str(yolo.names[int(cls_id)]),
                            "confidence": round(float(score), 4),
                            "points": [[round(float(x), 2), round(float(y), 2)] for x, y in pts.tolist()],
                        }
                    )
        det_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        prov = {
            **self.provenance,
            "confidence_threshold": float(confidence),
            "detection_ms": det_ms,
        }
        return raw_dets, prov


def extract_menu_crops(
    rgb_image: Image.Image,
    raw_detections: list[dict[str, Any]],
    *,
    duplicate_iou: float = 0.65,
    min_merge_iom: float = 0.20,
    max_crops: int = MAX_CROPS_PER_MENU,
) -> dict[str, Any]:
    """Run refined `line_crops.process` + `rectify` and build per-crop strips & previews."""
    t0 = time.perf_counter()
    img_w, img_h = rgb_image.size
    processed = process_lines(
        raw_detections,
        img_w,
        img_h,
        duplicate_iou=duplicate_iou,
        min_merge_iom=min_merge_iom,
    )
    decisions = processed["decisions"]
    conflicts = processed["conflicts"]
    crops_raw = processed["crops"]
    truncated_crops = len(crops_raw) > max_crops
    crops_limited = crops_raw[:max_crops]

    # Index merge/dedup decisions by source_id for per-crop provenance trace
    crops_prepared: list[dict[str, Any]] = []
    for idx, item in enumerate(crops_limited):
        src_set = set(item["source_ids"])
        def touches_crop(value: Any) -> bool:
            # Merge decisions contain nested source groups, whereas suppression
            # and padding decisions contain flat IDs. Never hash a group itself.
            if isinstance(value, (list, tuple)):
                return any(touches_crop(child) for child in value)
            return isinstance(value, (str, int)) and value in src_set

        related_decisions = [
            d for d in decisions
            if any(touches_crop(d.get(key, ())) for key in
                   ("sources", "kept", "suppressed", "result", "source_a", "source_b"))
        ]
        ori_flag = int(item["orientation_flag"])
        crop_poly = [[round(float(x), 2), round(float(y), 2)] for x, y in item["crop_polygon"]]
        core_poly = [[round(float(x), 2), round(float(y), 2)] for x, y in item["core"]]

        # `rectify` produces the reading-axis strip (vertical glyphs rotated CCW once!)
        rectified_strip = rectify(rgb_image, item["crop_polygon"])
        # Upright preview for human display (undo CCW rotation for vertical crops)
        upright_im = (
            rectified_strip.transpose(Image.Transpose.ROTATE_270) if ori_flag == 1 else rectified_strip
        )
        # Native H=64 OCR strip preview (no legacy 512-letterbox shrinking!)
        ocr_strip_im = native_strip(rectified_strip)

        crops_prepared.append(
            {
                "crop_id": str(item["id"]),
                "crop_index": idx + 1,
                "label": str(item["label"]),
                "confidence": round(float(item["confidence"]), 4),
                "orientation_flag": ori_flag,
                "orientation_label": "vertical" if ori_flag == 1 else "horizontal",
                "source_ids": list(item["source_ids"]),
                "padding_scale": round(float(item["padding_scale"]), 4),
                "overlap_severity": str(item.get("overlap_severity", "clear")),
                "minor_overlap_allowed": bool(item.get("minor_overlap_allowed", False)),
                "review_required": bool(item.get("review_required", False)),
                "legacy_review_required": bool(item.get("legacy_review_required", False)),
                "minor_overlap_policy_version": item.get("minor_overlap_policy_version"),
                "core_polygon": core_poly,
                "crop_polygon": crop_poly,
                "merge_decisions": related_decisions,
                "rectified_strip": rectified_strip,
                "upright_crop": upright_im,
                "ocr_strip_preview": ocr_strip_im,
            }
        )

    return {
        "crops": crops_prepared,
        "raw_count": int(processed["raw_count"]),
        "processed_count": int(processed["processed_count"]),
        "returned_count": len(crops_prepared),
        "crops_truncated": truncated_crops,
        "decisions": decisions,
        "conflicts": conflicts,
        "policy": processed["policy"],
        "postprocess_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }


def make_config_signature(
    *,
    decode_mode: str,
    lookup_config: LookupConfig,
    confidence: float,
) -> str:
    payload = {
        "decode_mode": decode_mode,
        "lookup_config": lookup_config.to_dict(),
        "confidence": round(float(confidence), 4),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def run_full_menu_pipeline(
    *,
    image_bytes: bytes,
    filename: str,
    detector: Any,  # V5YoloDetector or duck-typed object with .detect(rgb_im, confidence=...)
    pipeline: LexiconPipeline,
    lookup_config: LookupConfig,
    decode_mode: str = "greedy",
    confidence: float = 0.35,
    max_crops: int = MAX_CROPS_PER_MENU,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict[str, Any]:
    """Execute full menu photo -> detection -> crop rectification -> OCR/LM/trie/translation."""
    t_total0 = time.perf_counter()

    def _emit(stage: str, done: int, total: int) -> None:
        if progress_callback is not None:
            progress_callback(stage, done, total)

    def _check_cancel() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Menu job cancelled")

    _emit("decoding", 0, 0)
    t_dec0 = time.perf_counter()
    rgb_im, img_meta = decode_uploaded_image_bytes(image_bytes)
    decode_ms = round((time.perf_counter() - t_dec0) * 1000.0, 2)
    _check_cancel()

    _emit("detecting", 0, 0)
    raw_dets, det_prov = detector.detect(rgb_im, confidence=confidence)
    detection_ms = float(det_prov.get("detection_ms", 0.0))
    _check_cancel()

    _emit("postprocessing", 0, len(raw_dets))
    extracted = extract_menu_crops(rgb_im, raw_dets, max_crops=max_crops)
    postprocess_ms = float(extracted["postprocess_ms"])
    prepared_crops = extracted["crops"]
    total_crops = len(prepared_crops)
    _check_cancel()

    config_sig = make_config_signature(
        decode_mode=decode_mode,
        lookup_config=lookup_config,
        confidence=confidence,
    )
    sanitized_name = Path(str(filename or "menu.jpg")).name
    menu_id = f"menu_{img_meta['sha256'][:12]}_{config_sig[:8]}"

    t_rec0 = time.perf_counter()
    crop_details: dict[str, dict[str, Any]] = {}
    crop_summaries: list[dict[str, Any]] = []
    ok_count = 0
    empty_count = 0
    error_count = 0

    for idx, c in enumerate(prepared_crops):
        _check_cancel()
        _emit("recognizing", idx, total_crops)
        cid = c["crop_id"]
        t_c0 = time.perf_counter()
        upright_url = _pil_to_data_url(c["upright_crop"], fmt="PNG")
        ocr_strip_url = _pil_to_data_url(c["ocr_strip_preview"], fmt="PNG")

        geom_info = {
            "crop_id": cid,
            "crop_index": c["crop_index"],
            "label": c["label"],
            "confidence": c["confidence"],
            "orientation_flag": c["orientation_flag"],
            "orientation_label": c["orientation_label"],
            "source_ids": c["source_ids"],
            "padding_scale": c["padding_scale"],
            "core_polygon": c["core_polygon"],
            "crop_polygon": c["crop_polygon"],
            "merge_decisions": c["merge_decisions"],
            "upright_size": [int(c["upright_crop"].width), int(c["upright_crop"].height)],
            "ocr_strip_size": [int(c["ocr_strip_preview"].width), int(c["ocr_strip_preview"].height)],
        }

        try:
            res = pipeline.lookup_image(
                c["rectified_strip"],
                orientation=c["orientation_flag"],
                already_rectified_strip=True,
                config=lookup_config,
                decode_mode=decode_mode,
                crop_provenance={
                    "menu_id": menu_id,
                    "crop_id": cid,
                    "source_ids": c["source_ids"],
                    "detector_label": c["label"],
                    "detector_confidence": c["confidence"],
                },
            )
            res_dict = res.to_dict()
            elapsed_crop_ms = round((time.perf_counter() - t_c0) * 1000.0, 2)
            is_empty = not res.canonical_ocr.strip()
            crop_status = "empty" if is_empty else "ok"
            if is_empty:
                empty_count += 1
            else:
                ok_count += 1
            detail = {
                "crop_id": cid,
                "crop_index": c["crop_index"],
                "status": crop_status,
                "error": None,
                "elapsed_ms": elapsed_crop_ms,
                "geometry": geom_info,
                "upright_crop_data_url": upright_url,
                "ocr_strip_data_url": ocr_strip_url,
                "result": res_dict,
                "menu_provenance": {
                    "menu_id": menu_id,
                    "filename": sanitized_name,
                    "image_sha256": img_meta["sha256"],
                    "config_signature": config_sig,
                    "detector": det_prov,
                },
            }
            summary = {
                "crop_id": cid,
                "crop_index": c["crop_index"],
                "status": crop_status,
                "error": None,
                "elapsed_ms": elapsed_crop_ms,
                "label": c["label"],
                "confidence": c["confidence"],
                "orientation_flag": c["orientation_flag"],
                "orientation_label": c["orientation_label"],
                "crop_polygon": c["crop_polygon"],
                "core_polygon": c["core_polygon"],
                "source_ids": c["source_ids"],
                "canonical_ocr": res.canonical_ocr,
                "lm_alternative": res.lm_alternative,
                "suggested_line": res.suggested_line,
                "visual_reranked_line": res.visual_reranked_line,
                "long_line_route": res_dict["provenance"]["crop"].get("long_line_route"),
                "warning_count": len(res.warnings),
                "upright_crop_data_url": upright_url,
            }
        except Exception as exc:
            error_count += 1
            elapsed_crop_ms = round((time.perf_counter() - t_c0) * 1000.0, 2)
            err_msg = f"{type(exc).__name__}: {exc}"
            detail = {
                "crop_id": cid,
                "crop_index": c["crop_index"],
                "status": "error",
                "error": err_msg,
                "elapsed_ms": elapsed_crop_ms,
                "geometry": geom_info,
                "upright_crop_data_url": upright_url,
                "ocr_strip_data_url": ocr_strip_url,
                "result": None,
                "menu_provenance": {
                    "menu_id": menu_id,
                    "filename": sanitized_name,
                    "image_sha256": img_meta["sha256"],
                    "config_signature": config_sig,
                    "detector": det_prov,
                },
            }
            summary = {
                "crop_id": cid,
                "crop_index": c["crop_index"],
                "status": "error",
                "error": err_msg,
                "elapsed_ms": elapsed_crop_ms,
                "label": c["label"],
                "confidence": c["confidence"],
                "orientation_flag": c["orientation_flag"],
                "orientation_label": c["orientation_label"],
                "crop_polygon": c["crop_polygon"],
                "core_polygon": c["core_polygon"],
                "source_ids": c["source_ids"],
                "canonical_ocr": "",
                "lm_alternative": None,
                "suggested_line": "",
                "visual_reranked_line": None,
                "long_line_route": None,
                "warning_count": 1,
                "upright_crop_data_url": upright_url,
            }

        crop_details[cid] = detail
        crop_summaries.append(summary)

    _emit("recognizing", total_crops, total_crops)
    recognition_lookup_ms = round((time.perf_counter() - t_rec0) * 1000.0, 2)
    total_ms = round((time.perf_counter() - t_total0) * 1000.0, 2)

    source_photo_url = _pil_to_data_url(rgb_im, fmt="JPEG")

    overview = {
        "menu_id": menu_id,
        "filename": sanitized_name,
        "image": img_meta,
        "source_photo_data_url": source_photo_url,
        "config_signature": config_sig,
        "decode_mode": decode_mode,
        "lookup_config": lookup_config.to_dict(),
        "detector": det_prov,
        "postprocess": {
            "raw_count": extracted["raw_count"],
            "processed_count": extracted["processed_count"],
            "returned_count": extracted["returned_count"],
            "crops_truncated": extracted["crops_truncated"],
            "conflicts_count": len(extracted["conflicts"]),
            "decisions_count": len(extracted["decisions"]),
            "policy": extracted["policy"],
        },
        "raw_detections": raw_dets,
        "crops": crop_summaries,
        "counts": {
            "total": total_crops,
            "ok": ok_count,
            "empty": empty_count,
            "error": error_count,
        },
        "timings_ms": {
            "decode_ms": decode_ms,
            "detection_ms": detection_ms,
            "postprocess_ms": postprocess_ms,
            "recognition_lookup_ms": recognition_lookup_ms,
            "total_ms": total_ms,
        },
        "auto_corrected": False,
    }
    return {"overview": overview, "crop_details": crop_details}
