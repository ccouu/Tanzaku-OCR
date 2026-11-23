"""Stage 4 Desktop Reference Backend — loopback-only Python server reusing shared
V5 YOLO11n-OBB, Step1000 RAW (2,105 vocab), CharLM V2 HP1010, Lexicon V3, and
read-only `culinary_menu.db`.

Security & integrity:
  - Binds strictly to loopback (`127.0.0.1`); refuses non-loopback hosts.
  - Requires per-launch random session token (`X-Stage4-Token`) on all mutation/read endpoints
    (with same-origin loopback `/api/bootstrap` when serving the local Flutter web demo).
  - Strict CORS allowlist, bounded uploads (16 MB / 24 MP), bounded LRU cache (max 4 pages),
    zero arbitrary filesystem path reads, zero runtime cloud/LLM API calls.
  - Verifies all 6 pinned Stage 4 asset SHA-256 hashes on startup; 147/147 BN buffers frozen.
"""
from __future__ import annotations

import argparse
import base64
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import sys
import tempfile
import threading
import time
from typing import Any, Optional
from urllib.parse import urlparse

from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recognition.bn_calibration import snapshot_bn_buffers, verify_bn_buffers_identical
from src.recognition.canonical_ctc import fold_parenthesis_logits
from src.recognition.char_lm_beam import DecodeConfig, ImmutableCharNGramLM, NUM_SPAN_RE, decode_with_config
from src.recognition.checkpointing import resolve_checkpoint_state
from src.recognition.data import IMAGENET_MEAN, IMAGENET_STD, preprocess
from src.recognition.label_normalization import normalize_digits, normalize_parentheses
from src.recognition.lexicon_lookup.contracts import LookupConfig
from src.recognition.lexicon_lookup.index import LexiconIndex
from src.recognition.lexicon_lookup.menu_pipeline import (
    MAX_CROPS_PER_MENU,
    MAX_MENU_PIXELS,
    MAX_MENU_UPLOAD_BYTES,
    V5YoloDetector,
    decode_uploaded_image_bytes,
    extract_menu_crops,
)
from src.recognition.lexicon_lookup.particle_guard import probable_particles
from src.recognition.lexicon_lookup.pipeline import LexiconPipeline
from src.recognition.lexicon_lookup.translation_repo import ReadOnlyTranslationRepo
from src.recognition.long_line_inference import bucket_image, native_strip, owned_tokens, window_ranges
from src.recognition.model import ARCH_SPECS, MenuRecognizer, load_recognizer_state
from src.stage4.bundle_manifest import verify_pinned_stage4_bundle

PARTICLE_CHARS = frozenset("でのをにとはがもや")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
VALID_ROTATION_DEGREES = frozenset({0, 90, 180, 270})
EXIF_AXIS_SWAP_ORIENTATIONS = frozenset({5, 6, 7, 8})


def validate_capture_resolution_parity(
    *,
    stream_width: Optional[int],
    stream_height: Optional[int],
    saved_width: int,
    saved_height: int,
    stream_rotation: int = 0,
    exif_orientation: Optional[int] = None,
) -> dict[str, Any]:
    """Validate ordered capture stream WxH against saved WxH.

    Requires exact (stream_w == saved_w and stream_h == saved_h), OR allows a swapped
    (stream_w == saved_h and stream_h == saved_w) ONLY when a known 90/270 rotation
    or EXIF axis-swapping orientation (5, 6, 7, 8) justifies the swap.
    Rejects misreported stream dimensions with ValueError.
    """
    if saved_width <= 0 or saved_height <= 0:
        raise ValueError(f"Invalid saved image dimensions: {saved_width}x{saved_height}")
    rot = int(stream_rotation) % 360
    if rot not in VALID_ROTATION_DEGREES:
        raise ValueError(f"stream_rotation must be one of 0, 90, 180, 270; got {stream_rotation}")

    if stream_width is None and stream_height is None:
        return {
            "stream_width": saved_width,
            "stream_height": saved_height,
            "stream_rotation": rot,
            "exif_orientation": exif_orientation,
            "stream_saved_parity": True,
            "stream_saved_parity_mode": "inferred_from_saved",
        }

    if stream_width is None or stream_height is None or int(stream_width) <= 0 or int(stream_height) <= 0:
        raise ValueError(f"Invalid stream dimensions: {stream_width}x{stream_height}")

    sw, sh = int(stream_width), int(stream_height)
    rotation_swaps_axes = (rot in (90, 270)) or (exif_orientation in EXIF_AXIS_SWAP_ORIENTATIONS)

    if sw == saved_width and sh == saved_height:
        if rotation_swaps_axes and sw != sh:
            raise ValueError(
                f"Misreported stream dimensions: rotation={rot}/exif={exif_orientation} swaps axes, "
                f"but stream ({sw}x{sh}) matched saved ({saved_width}x{saved_height}) unswapped"
            )
        return {
            "stream_width": sw,
            "stream_height": sh,
            "stream_rotation": rot,
            "exif_orientation": exif_orientation,
            "stream_saved_parity": True,
            "stream_saved_parity_mode": "exact",
        }

    if sw == saved_height and sh == saved_width:
        if rotation_swaps_axes:
            return {
                "stream_width": sw,
                "stream_height": sh,
                "stream_rotation": rot,
                "exif_orientation": exif_orientation,
                "stream_saved_parity": True,
                "stream_saved_parity_mode": "rotated_90_270_swap",
            }
        raise ValueError(
            f"Misreported stream dimensions: stream ({sw}x{sh}) is swapped vs saved ({saved_width}x{saved_height}) "
            f"without a justified 90/270 rotation or EXIF orientation swap"
        )

    raise ValueError(
        f"Misreported stream dimensions: stream ({sw}x{sh}) does not match saved ({saved_width}x{saved_height})"
    )


def canon_text(text: str) -> str:
    return normalize_digits(normalize_parentheses(text)).strip()


def _pil_to_png_b64(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def format_crop_translation(
    decoded_ocr: str,
    lex_res: Any,
    repo: ReadOnlyTranslationRepo,
    particles: tuple[dict, ...],
) -> dict[str, Any]:
    """Build structured component + reviewed whole-phrase English payload without inventing translations."""
    if not decoded_ocr:
        return {
            "primary_english": "No readable text",
            "reviewed_whole_phrase_en": None,
            "translation_status": "no_readable_text",
            "components": [],
            "suggestions": [],
        }

    part_pos = {p["start"] for p in particles}
    components: list[dict[str, Any]] = []
    suggestions: list[dict[str, Any]] = []
    comp_en_parts: list[str] = []
    has_exact_en = has_exact_await = has_fuzzy = has_unmatched = has_protected = False

    for seg in lex_res.segments:
        if seg.kind == "whitespace":
            continue
        if seg.kind == "protected":
            has_protected = True
            components.append({"text": seg.text, "kind": "number_or_price", "english": seg.text, "start": seg.start, "end": seg.end})
            comp_en_parts.append(seg.text)
            continue
        if seg.kind == "exact" and seg.selected_candidate is not None:
            c = seg.selected_candidate
            tr = c.translation
            en_v = (tr.name_en or "").strip() if tr is not None else ""
            if en_v:
                has_exact_en = True
                components.append({
                    "text": c.term,
                    "kind": "exact",
                    "english": en_v,
                    "reading": tr.reading_hiragana if tr else "",
                    "db_id": tr.db_id if tr else c.db_id,
                    "start": seg.start,
                    "end": seg.end,
                })
                comp_en_parts.append(en_v)
            else:
                has_exact_await = True
                components.append({
                    "text": c.term,
                    "kind": "await_for_translation",
                    "english": "Await for translation",
                    "reading": tr.reading_hiragana if tr else "",
                    "db_id": tr.db_id if tr else c.db_id,
                    "start": seg.start,
                    "end": seg.end,
                })
                comp_en_parts.append("Await for translation")
            continue
        if seg.kind == "fuzzy_suggestion" and seg.selected_candidate is not None:
            has_fuzzy = True
            c = seg.selected_candidate
            tr = c.translation
            en_v = (tr.name_en or "").strip() if tr is not None else ""
            sugg_en = en_v if en_v else "Await for translation"
            suggestions.append({
                "source_span": seg.text,
                "suggested_term": c.term,
                "suggested_english": sugg_en,
                "match_type": c.match_type,
                "hard_edits": c.hard_edits,
            })
            components.append({
                "text": seg.text,
                "kind": "unmatched_with_suggestion",
                "english": "No lexicon match",
                "start": seg.start,
                "end": seg.end,
            })
            comp_en_parts.append(f"[{seg.text}]")
            continue

        if len(seg.text) == 1 and (seg.start in part_pos or seg.text in PARTICLE_CHARS):
            components.append({"text": seg.text, "kind": "particle", "english": "", "start": seg.start, "end": seg.end})
            continue
        if not re.search(r"[^\s0-9.,¥円()（）/\-~〜・&＆+%!！?？:：]", seg.text):
            components.append({"text": seg.text, "kind": "symbol", "english": seg.text, "start": seg.start, "end": seg.end})
            comp_en_parts.append(seg.text)
            continue
        has_unmatched = True
        components.append({"text": seg.text, "kind": "unmatched", "english": "No lexicon match", "start": seg.start, "end": seg.end})
        comp_en_parts.append(f"[{seg.text}]")

    phrase_plan = repo.get_phrase_component_plan(decoded_ocr)
    reviewed_phrase_en = phrase_plan["phrase_en"] if phrase_plan else None

    if has_exact_en and not has_exact_await and not has_unmatched and not has_fuzzy:
        status = "exact_db_english"
        fallback_en = " / ".join(comp_en_parts)
    elif has_exact_await and not has_exact_en and not has_unmatched and not has_fuzzy:
        status = "await_for_translation"
        fallback_en = "Await for translation"
    elif has_exact_en or has_exact_await:
        status = "partial_segment_english"
        fallback_en = " · ".join(comp_en_parts)
    elif has_fuzzy:
        status = "fuzzy_suggestion_only"
        fallback_en = "No lexicon match"
    elif has_unmatched:
        status = "no_lexicon_match"
        fallback_en = "No lexicon match"
    elif has_protected:
        status = "numeric_or_symbol_only"
        fallback_en = " ".join(comp_en_parts)
    else:
        status = "no_lexicon_match"
        fallback_en = "No lexicon match"

    return {
        "primary_english": reviewed_phrase_en if reviewed_phrase_en else fallback_en,
        "reviewed_whole_phrase_en": reviewed_phrase_en,
        "reviewed_phrase_provenance": phrase_plan["provenance_source"] if phrase_plan else None,
        "translation_status": status,
        "components": components,
        "suggestions": suggestions,
    }


class Stage4Engine:
    """Thread-safe local reference engine owning pinned models, bounded page cache, and active job."""

    def __init__(self, *, root: Path = ROOT, cache_dir: Path | None = None, max_cached_pages: int = 4) -> None:
        self.root = Path(root).resolve()
        self.bundle = verify_pinned_stage4_bundle(self.root, strict=True)
        self.cache_dir = Path(cache_dir) if cache_dir else (Path.home() / ".cache" / "stage4_capture_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cached_pages = int(max_cached_pages)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
        self.backend_id = f"desktop_python_reference_{self.device.type}"

        self._load_lock = threading.Lock()
        self._loaded = False
        self._cold_load_ms: float | None = None
        self._model: Optional[MenuRecognizer] = None
        self._bn_ref: dict[str, torch.Tensor] = {}
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: dict[int, str] = {}
        self._detector: Optional[V5YoloDetector] = None
        self._lm: Optional[ImmutableCharNGramLM] = None
        self._lex_index: Optional[LexiconIndex] = None
        self._repo: Optional[ReadOnlyTranslationRepo] = None
        self._lex_pipe: Optional[LexiconPipeline] = None

        self._state_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._queue_cond = threading.Condition(self._state_lock)
        self._captures: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._runs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._active_cancel: Optional[threading.Event] = None
        self._active_run_id: Optional[str] = None
        self._pending_job: Optional[tuple[dict[str, Any], dict[str, Any], dict[str, Any], threading.Event]] = None
        self._shutdown = False
        self._run_seq = 0
        self._inflight_model_runs = 0
        self._max_concurrent_model_runs = 0
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="stage4-serialized-worker",
            daemon=True,
        )
        self._worker_thread.start()

    def close(self) -> None:
        """Stop the serialized worker cleanly."""
        with self._queue_cond:
            self._shutdown = True
            if self._active_cancel is not None:
                self._active_cancel.set()
            if self._pending_job is not None:
                stale_run, _, _, stale_ev = self._pending_job
                self._pending_job = None
                stale_ev.set()
                self._mark_cancelled_before_start_locked(stale_run)
            self._queue_cond.notify_all()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

    def ensure_loaded(self) -> float:
        with self._load_lock:
            if self._loaded:
                return 0.0
            t0 = time.perf_counter()
            assets = self.bundle["assets"]
            vocab = json.loads(Path(assets["vocab_2105"]["path"]).read_text(encoding="utf-8"))
            chars = vocab["characters"]
            assert len(chars) == 2105
            self._char_to_id = {c: i + 1 for i, c in enumerate(chars)}
            self._id_to_char = {v: k for k, v in self._char_to_id.items()}

            sd, _, _, _ = resolve_checkpoint_state(
                Path(assets["ocr_checkpoint_raw"]["path"]),
                arm_key="raw",
                expected_sha256=assets["ocr_checkpoint_raw"]["expected_sha256"],
            )
            model = MenuRecognizer(len(chars) + 1, use_bilstm=True, **ARCH_SPECS["h4_k41"])
            load_recognizer_state(model, sd)
            self._bn_ref = snapshot_bn_buffers(model)
            assert len(self._bn_ref) == 147
            model.to(self.device).eval()
            self._model = model

            self._detector = V5YoloDetector(
                Path(assets["detector_v5_obb"]["path"]),
                device="0" if self.device.type == "cuda" else "cpu",
                imgsz=640,
                iou=0.70,
            )
            self._lm = ImmutableCharNGramLM.load_gz(Path(assets["char_lm_v2"]["path"]), allow_legacy_v1=False)
            self._lex_index = LexiconIndex.load_gz(Path(assets["lexicon_index_v3"]["path"]))
            self._repo = ReadOnlyTranslationRepo(Path(assets["culinary_menu_db"]["path"]))
            self._lex_pipe = LexiconPipeline(self._lex_index, self._repo, immutable_lm=self._lm)
            _sync_cuda(self.device)
            self._cold_load_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            self._loaded = True
            return self._cold_load_ms

    def health_summary(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "status": "ok",
                "backend_id": self.backend_id,
                "device": str(self.device),
                "device_name": self.device_name,
                "precision": "fp32",
                "models_loaded": self._loaded,
                "cold_load_ms": self._cold_load_ms,
                "bundle_id": self.bundle["bundle_id"],
                "bundle_verified": self.bundle["all_verified"],
                "cached_pages": len(self._captures),
                "active_run_id": self._active_run_id,
                "inflight_model_runs": self._inflight_model_runs,
                "max_concurrent_model_runs": self._max_concurrent_model_runs,
            }

    def _evict_old_captures(self) -> None:
        while len(self._captures) > self.max_cached_pages:
            _, evicted = self._captures.popitem(last=False)
            p = evicted.get("saved_path")
            if p:
                Path(p).unlink(missing_ok=True)
        while len(self._runs) > 32:
            self._runs.popitem(last=False)

    def _validate_settings(self, confidence: float, alpha_lm: float, decode_mode: str) -> dict[str, Any]:
        conf = round(float(confidence), 2)
        alpha = round(float(alpha_lm), 2)
        mode = str(decode_mode).strip().lower()
        if not (0.10 <= conf <= 0.90):
            raise ValueError(f"confidence must be in [0.10, 0.90], got {conf}")
        if not (0.00 <= alpha <= 0.60):
            raise ValueError(f"alpha_lm must be in [0.00, 0.60], got {alpha}")
        if mode not in ("charlm_r1", "greedy"):
            raise ValueError(f"decode_mode must be 'charlm_r1' or 'greedy', got {mode!r}")
        cfg_sig = hashlib.sha256(f"{conf:.2f}|{alpha:.2f}|{mode}".encode("utf-8")).hexdigest()[:12]
        return {"confidence": conf, "alpha_lm": alpha, "decode_mode": mode, "config_id": cfg_sig}

    def start_capture_job(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        capture_id: Optional[str] = None,
        stream_width: Optional[int] = None,
        stream_height: Optional[int] = None,
        stream_rotation: int = 0,
        capture_source: str = "import",
        confidence: float = 0.35,
        alpha_lm: float = 0.30,
        decode_mode: str = "charlm_r1",
        request_received_mono: Optional[float] = None,
        request_read_ms: float = 0.0,
    ) -> dict[str, Any]:
        req_mono = time.perf_counter() if request_received_mono is None else float(request_received_mono)
        settings = self._validate_settings(confidence, alpha_lm, decode_mode)
        t_dec0 = time.perf_counter()
        rgb, img_meta = decode_uploaded_image_bytes(
            image_bytes, max_bytes=MAX_MENU_UPLOAD_BYTES, max_pixels=MAX_MENU_PIXELS
        )
        decode_normalize_ms = round((time.perf_counter() - t_dec0) * 1000.0, 2)

        parity = validate_capture_resolution_parity(
            stream_width=stream_width,
            stream_height=stream_height,
            saved_width=img_meta["width"],
            saved_height=img_meta["height"],
            stream_rotation=stream_rotation,
            exif_orientation=img_meta.get("exif_orientation"),
        )

        cid = capture_id if (capture_id and SAFE_ID_RE.match(capture_id)) else f"cap_{secrets.token_hex(6)}"
        t_per0 = time.perf_counter()
        saved_path = self.cache_dir / f"{cid}.png"
        rgb.save(saved_path, format="PNG")
        persist_ms = round((time.perf_counter() - t_per0) * 1000.0, 2)
        request_ingest_ms = round((time.perf_counter() - req_mono) * 1000.0, 2)

        cap_entry = {
            "capture_id": cid,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "filename": Path(filename or "capture.png").name,
            "capture_source": capture_source,
            "stream_width": parity["stream_width"],
            "stream_height": parity["stream_height"],
            "stream_rotation": parity["stream_rotation"],
            "exif_orientation": parity["exif_orientation"],
            "stream_saved_parity": parity["stream_saved_parity"],
            "stream_saved_parity_mode": parity["stream_saved_parity_mode"],
            "saved_width": img_meta["width"],
            "saved_height": img_meta["height"],
            "image_sha256": img_meta["sha256"],
            "saved_path": str(saved_path),
            "rgb": rgb,
            "request_read_ms": round(float(request_read_ms), 2),
            "decode_normalize_ms": decode_normalize_ms,
            "persist_ms": persist_ms,
            "request_ingest_ms": request_ingest_ms,
            # Immutable settings-keyed caches (never mutated in-place; committed only by non-cancelled runs)
            "raw_detections_by_conf": {},  # float conf -> {"detections": tuple[dict,...], "source_run_seq": int}
            "ocr_cache_by_conf": {},       # float conf -> {"crops_meta": tuple, "logits": tuple, "raw_count": int, "crops_truncated": bool, "source_run_seq": int, "source_run_id": str}
        }
        with self._state_lock:
            self._captures[cid] = cap_entry
            self._captures.move_to_end(cid)
            self._evict_old_captures()

        return self._launch_run(
            cap_entry,
            settings,
            is_reprocess=False,
            request_received_mono=req_mono,
            request_read_ms=round(float(request_read_ms), 2),
            request_ingest_ms=request_ingest_ms,
        )

    def start_reprocess_job(
        self,
        *,
        capture_id: str,
        confidence: float = 0.35,
        alpha_lm: float = 0.30,
        decode_mode: str = "charlm_r1",
        request_received_mono: Optional[float] = None,
        request_read_ms: float = 0.0,
    ) -> dict[str, Any]:
        req_mono = time.perf_counter() if request_received_mono is None else float(request_received_mono)
        settings = self._validate_settings(confidence, alpha_lm, decode_mode)
        with self._state_lock:
            cap_entry = self._captures.get(capture_id)
        if cap_entry is None:
            raise KeyError(f"Capture {capture_id!r} not found in bounded cache (evicted or unknown)")
        request_ingest_ms = round((time.perf_counter() - req_mono) * 1000.0, 2)
        return self._launch_run(
            cap_entry,
            settings,
            is_reprocess=True,
            request_received_mono=req_mono,
            request_read_ms=round(float(request_read_ms), 2),
            request_ingest_ms=request_ingest_ms,
        )

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        with self._queue_cond:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(f"Run {run_id!r} not found")
            ev = run.get("_cancel_event")
            if ev is not None:
                ev.set()
            if self._pending_job is not None and self._pending_job[0]["run_id"] == run_id:
                stale_run = self._pending_job[0]
                self._pending_job = None
                stale_run["cancel_requested"] = True
                self._mark_cancelled_before_start_locked(stale_run)
            elif run["status"] == "running":
                run["cancel_requested"] = True
            return {"run_id": run_id, "status": run["status"], "cancel_requested": True}

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._state_lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(f"Run {run_id!r} not found")
            snap: dict[str, Any] = {}
            for k, v in run.items():
                if k.startswith("_"):
                    continue
                if isinstance(v, dict):
                    snap[k] = dict(v)
                elif isinstance(v, list):
                    snap[k] = list(v)
                else:
                    snap[k] = v
            return snap

    def get_capture_image_bytes(self, capture_id: str) -> bytes:
        with self._state_lock:
            cap = self._captures.get(capture_id)
        if cap is None or not Path(cap["saved_path"]).is_file():
            raise KeyError(f"Capture image {capture_id!r} not found")
        return Path(cap["saved_path"]).read_bytes()

    def _mark_cancelled_before_start_locked(self, run: dict[str, Any]) -> None:
        now = time.perf_counter()
        req_mono = float(run.get("_request_received_mono", now))
        req_total = round((now - req_mono) * 1000.0, 2)
        run["status"] = "cancelled"
        run["stage"] = "cancelled_before_start"
        run["cache_status"] = "cancelled_before_start"
        run["timings_ms"]["server_pipeline_wall_ms"] = 0.0
        run["timings_ms"]["server_request_total_ms"] = req_total
        run["timings_ms"]["total_wall_ms"] = req_total

    def _launch_run(
        self,
        cap_entry: dict[str, Any],
        settings: dict[str, Any],
        *,
        is_reprocess: bool,
        request_received_mono: float,
        request_read_ms: float,
        request_ingest_ms: float,
    ) -> dict[str, Any]:
        run_id = f"run_{secrets.token_hex(6)}"
        cancel_ev = threading.Event()
        launch_mono = time.perf_counter()
        with self._queue_cond:
            self._run_seq += 1
            run_seq = self._run_seq
            run_state: dict[str, Any] = {
                "run_id": run_id,
                "run_seq": run_seq,
                "capture_id": cap_entry["capture_id"],
                "config_id": settings["config_id"],
                "backend_id": self.backend_id,
                "device": str(self.device),
                "device_name": self.device_name,
                "precision": "fp32",
                "bundle_id": self.bundle["bundle_id"],
                "settings": settings,
                "is_reprocess": is_reprocess,
                "status": "running",
                "stage": "queued",
                "cancel_requested": False,
                "capture": {
                    "capture_id": cap_entry["capture_id"],
                    "filename": cap_entry["filename"],
                    "capture_source": cap_entry["capture_source"],
                    "stream_width": cap_entry["stream_width"],
                    "stream_height": cap_entry["stream_height"],
                    "stream_rotation": cap_entry["stream_rotation"],
                    "exif_orientation": cap_entry["exif_orientation"],
                    "saved_width": cap_entry["saved_width"],
                    "saved_height": cap_entry["saved_height"],
                    "stream_saved_parity": cap_entry["stream_saved_parity"],
                    "stream_saved_parity_mode": cap_entry["stream_saved_parity_mode"],
                    "image_sha256": cap_entry["image_sha256"],
                },
                "counts": {
                    "raw_detections": 0,
                    "postprocessed_lines": 0,
                    "completed_crops": 0,
                    "failed_crops": 0,
                    "total_windows": 0,
                    "crops_truncated": False,
                },
                "cache_status": "pending",
                "reused_cached_logits": False,
                "reused_cached_detections": False,
                "cold_or_warm": "warm" if self._loaded else "cold",
                "timings_ms": {
                    "request_read_ms": request_read_ms,
                    "decode_normalize_ms": 0.0 if is_reprocess else cap_entry["decode_normalize_ms"],
                    "persist_ms": 0.0 if is_reprocess else cap_entry["persist_ms"],
                    "request_ingest_ms": request_ingest_ms,
                    "queue_wait_ms": 0.0,
                    "cold_load_ms": 0.0,
                    "detect_ms": 0.0,
                    "postprocess_ms": 0.0,
                    "recognize_ms": 0.0,
                    "decode_ms": 0.0,
                    "lookup_ms": 0.0,
                    "first_result_ms": None,
                    "server_pipeline_wall_ms": None,
                    "server_request_total_ms": None,
                    "total_wall_ms": None,
                    "summed_crop_work_ms": 0.0,
                },
                "crops": [],
                "error": None,
                "_cancel_event": cancel_ev,
                "_request_received_mono": request_received_mono,
                "_launch_mono": launch_mono,
                "_run_seq": run_seq,
            }

            # Cancel any stale queued job immediately BEFORE it ever reaches model work
            if self._pending_job is not None:
                stale_run, _, _, stale_ev = self._pending_job
                self._pending_job = None
                stale_ev.set()
                stale_run["cancel_requested"] = True
                self._mark_cancelled_before_start_locked(stale_run)

            # Signal any currently running job to abort cleanly at its next checkpoint
            if self._active_cancel is not None:
                self._active_cancel.set()

            self._active_cancel = cancel_ev
            self._active_run_id = run_id
            self._runs[run_id] = run_state
            self._pending_job = (run_state, cap_entry, settings, cancel_ev)
            self._queue_cond.notify()

        return self.get_run(run_id)

    def _worker_loop(self) -> None:
        while True:
            with self._queue_cond:
                while self._pending_job is None and not self._shutdown:
                    self._queue_cond.wait()
                if self._shutdown and self._pending_job is None:
                    return
                job = self._pending_job
                self._pending_job = None
                if job is None:
                    continue
                run_state, cap_entry, settings, cancel_ev = job
                if cancel_ev.is_set():
                    self._mark_cancelled_before_start_locked(run_state)
                    continue
            self._execute_run(run_state, cap_entry, settings, cancel_ev)

    def _finalize_run_timings(
        self,
        run: dict[str, Any],
        *,
        t_pipeline0: float,
        status: str,
        stage: str,
        error: Optional[str] = None,
    ) -> None:
        t_end = time.perf_counter()
        req_mono = float(run.get("_request_received_mono", t_pipeline0))
        pipeline_ms = round((t_end - t_pipeline0) * 1000.0, 2)
        req_total_ms = round((t_end - req_mono) * 1000.0, 2)
        with self._state_lock:
            run["timings_ms"]["server_pipeline_wall_ms"] = pipeline_ms
            run["timings_ms"]["server_request_total_ms"] = req_total_ms
            run["timings_ms"]["total_wall_ms"] = req_total_ms
            run["stage"] = stage
            run["status"] = status
            if error is not None:
                run["error"] = error

    def _execute_run(
        self,
        run: dict[str, Any],
        cap: dict[str, Any],
        settings: dict[str, Any],
        cancel_ev: threading.Event,
    ) -> None:
        with self._inference_lock:
            if cancel_ev.is_set():
                with self._state_lock:
                    self._mark_cancelled_before_start_locked(run)
                return

            t_pipeline0 = time.perf_counter()
            launch_mono = float(run.get("_launch_mono", t_pipeline0))
            req_mono = float(run.get("_request_received_mono", t_pipeline0))
            run_seq = int(run.get("_run_seq", 0))
            conf = float(settings["confidence"])
            alpha = float(settings["alpha_lm"])
            mode = str(settings["decode_mode"])

            with self._state_lock:
                run["timings_ms"]["queue_wait_ms"] = round((t_pipeline0 - launch_mono) * 1000.0, 2)
                run["stage"] = "starting"
                self._inflight_model_runs += 1
                if self._inflight_model_runs > self._max_concurrent_model_runs:
                    self._max_concurrent_model_runs = self._inflight_model_runs

            try:
                cold_ms = self.ensure_loaded()
                with self._state_lock:
                    run["timings_ms"]["cold_load_ms"] = cold_ms
                if cancel_ev.is_set():
                    self._finalize_run_timings(run, t_pipeline0=t_pipeline0, status="cancelled", stage="cancelled")
                    return

                assert self._detector is not None and self._model is not None and self._lm is not None
                assert self._lex_pipe is not None and self._repo is not None and self._lex_index is not None

                # Read immutable settings-keyed cache snapshots under _state_lock
                with self._state_lock:
                    ocr_snap = cap["ocr_cache_by_conf"].get(conf)
                    raw_det_entry = None
                    if ocr_snap is None:
                        # Reuse raw detections from any cached threshold <= conf (pick lowest threshold)
                        eligible = [
                            (c_k, entry)
                            for c_k, entry in cap["raw_detections_by_conf"].items()
                            if c_k <= conf + 1e-6
                        ]
                        if eligible:
                            eligible.sort(key=lambda kv: kv[0])
                            raw_det_entry = eligible[0][1]

                can_reuse_logits = ocr_snap is not None
                can_reuse_raw_dets = (not can_reuse_logits) and (raw_det_entry is not None)

                raw_count_for_cache = 0
                crops_truncated_for_cache = False

                if can_reuse_logits:
                    assert ocr_snap is not None
                    crops_meta = ocr_snap["crops_meta"]
                    cached_logits = ocr_snap["logits"]
                    raw_count_for_cache = int(ocr_snap["raw_count"])
                    crops_truncated_for_cache = bool(ocr_snap["crops_truncated"])
                    with self._state_lock:
                        run["cache_status"] = "reused_cached_logits"
                        run["reused_cached_logits"] = True
                        run["reused_cached_detections"] = True
                        run["counts"]["raw_detections"] = raw_count_for_cache
                        run["counts"]["postprocessed_lines"] = len(crops_meta)
                        run["counts"]["crops_truncated"] = crops_truncated_for_cache
                else:
                    if can_reuse_raw_dets:
                        assert raw_det_entry is not None
                        with self._state_lock:
                            run["cache_status"] = "reused_raw_detections_fresh_ocr"
                            run["reused_cached_logits"] = False
                            run["reused_cached_detections"] = True
                        raw_dets = [d for d in raw_det_entry["detections"] if float(d["confidence"]) >= conf]
                    else:
                        with self._state_lock:
                            run["cache_status"] = "fresh_full_inference"
                            run["reused_cached_logits"] = False
                            run["reused_cached_detections"] = False
                            run["stage"] = "detecting"
                        t_d0 = time.perf_counter()
                        raw_dets, _ = self._detector.detect(cap["rgb"], confidence=conf)
                        _sync_cuda(self.device)
                        det_ms = round((time.perf_counter() - t_d0) * 1000.0, 2)
                        with self._state_lock:
                            run["timings_ms"]["detect_ms"] = det_ms
                            if not cancel_ev.is_set():
                                prev_raw = cap["raw_detections_by_conf"].get(conf)
                                if prev_raw is None or int(prev_raw["source_run_seq"]) < run_seq:
                                    cap["raw_detections_by_conf"][conf] = {
                                        "detections": tuple(raw_dets),
                                        "source_run_seq": run_seq,
                                        "source_run_id": run["run_id"],
                                    }

                    if cancel_ev.is_set():
                        self._finalize_run_timings(run, t_pipeline0=t_pipeline0, status="cancelled", stage="cancelled")
                        return

                    with self._state_lock:
                        run["stage"] = "postprocessing"
                    ext = extract_menu_crops(cap["rgb"], raw_dets, max_crops=MAX_CROPS_PER_MENU)
                    raw_count_for_cache = int(ext["raw_count"])
                    crops_truncated_for_cache = bool(ext["crops_truncated"])
                    crops_meta = tuple(ext["crops"])
                    cached_logits = tuple([None] * len(crops_meta))
                    with self._state_lock:
                        run["timings_ms"]["postprocess_ms"] = ext["postprocess_ms"]
                        run["counts"]["raw_detections"] = raw_count_for_cache
                        run["counts"]["postprocessed_lines"] = ext["returned_count"]
                        run["counts"]["crops_truncated"] = crops_truncated_for_cache

                # Recognize (if needed), decode, and translate ALL crops with per-crop failure isolation
                with self._state_lock:
                    run["stage"] = "recognizing_and_translating"
                dec_cfg = DecodeConfig(
                    mode="greedy" if mode == "greedy" else "beam4g_dn",
                    alpha_lm=0.0 if mode == "greedy" else alpha,
                    beam_width=5,
                    topk_t=6,
                    digit_neutral=True,
                    routing="none" if mode == "greedy" else "r1_short_greedy",
                    short_greedy_max_chars=2,
                    allow_mismatched_schema=False,
                )
                lookup_cfg = LookupConfig()

                rec_ms_tot = dec_ms_tot = look_ms_tot = crop_work_tot = 0.0
                total_windows = 0
                new_cached_logits: list[Any] = list(cached_logits)

                for idx, cm in enumerate(crops_meta):
                    if cancel_ev.is_set():
                        self._finalize_run_timings(run, t_pipeline0=t_pipeline0, status="cancelled", stage="cancelled")
                        return

                    t_c0 = time.perf_counter()
                    try:
                        ori = int(cm["orientation_flag"])
                        entry_logits = new_cached_logits[idx]
                        if entry_logits is None:
                            t_r0 = time.perf_counter()
                            strip = cm["ocr_strip_preview"]
                            if strip.width > 512:
                                ranges = window_ranges(strip.width, max_width=512, overlap=100)
                                canon_paths = []
                                with torch.inference_mode():
                                    for left, right in ranges:
                                        fitted, meta = bucket_image(strip.crop((left, 0, right, 64)), max_width=512)
                                        tensor, _, _ = preprocess(fitted, IMAGENET_MEAN, IMAGENET_STD, "batch", max_width=meta["canvas"])
                                        vl = int(meta["input_length"])
                                        logits = self._model(
                                            tensor[None].to(self.device),
                                            torch.tensor([ori], device=self.device),
                                            torch.tensor([vl], device=self.device),
                                        ).detach().cpu()
                                        folded = fold_parenthesis_logits(logits.float(), self._char_to_id, drop_aliases=False)
                                        canon_paths.append(folded[:vl, 0].argmax(-1).tolist())
                                _sync_cuda(self.device)
                                merged = "".join(self._id_to_char.get(t, "") for t in owned_tokens(canon_paths, ranges))
                                entry_logits = {
                                    "multi_window": True,
                                    "window_count": len(ranges),
                                    "merged_greedy": merged,
                                    "logp": None,
                                }
                            else:
                                fitted, meta = bucket_image(cm["rectified_strip"], max_width=512)
                                tensor, _, _ = preprocess(fitted, IMAGENET_MEAN, IMAGENET_STD, "batch", max_width=meta["canvas"])
                                vl = int(meta["input_length"])
                                with torch.inference_mode():
                                    logits = self._model(
                                        tensor[None].to(self.device),
                                        torch.tensor([ori], device=self.device),
                                        torch.tensor([vl], device=self.device),
                                    ).detach().cpu()
                                _sync_cuda(self.device)
                                folded = fold_parenthesis_logits(logits.float(), self._char_to_id, drop_aliases=False)[:vl, 0, :]
                                entry_logits = {
                                    "multi_window": False,
                                    "window_count": 1,
                                    "merged_greedy": None,
                                    "logp": F.log_softmax(folded, dim=-1),
                                }
                            rec_ms_tot += (time.perf_counter() - t_r0) * 1000.0
                            new_cached_logits[idx] = entry_logits

                        total_windows += int(entry_logits["window_count"])

                        # Decode
                        t_dec_c0 = time.perf_counter()
                        if entry_logits["multi_window"]:
                            greedy_c = canon_text(entry_logits["merged_greedy"])
                            decoded_c = greedy_c
                            route_used = "overlap512_merged_canonical_greedy_no_joint_lm"
                        else:
                            dec_out = decode_with_config(entry_logits["logp"], self._id_to_char, dec_cfg, lm=self._lm)
                            greedy_c = canon_text(dec_out["greedy"])
                            decoded_c = canon_text(dec_out["text"])
                            route_used = dec_out["route_used"]
                        dec_ms_tot += (time.perf_counter() - t_dec_c0) * 1000.0

                        num_disagree = bool(NUM_SPAN_RE.findall(greedy_c) != NUM_SPAN_RE.findall(decoded_c))

                        # Lexicon + read-only DB lookup
                        t_lk0 = time.perf_counter()
                        lex_res = self._lex_pipe.lookup_text(decoded_c, config=lookup_cfg, raw_ocr=decoded_c, decoder_mode=route_used)
                        parts = probable_particles(decoded_c, self._lex_index, lookup_cfg)
                        tr_payload = format_crop_translation(decoded_c, lex_res, self._repo, parts)
                        look_ms_tot += (time.perf_counter() - t_lk0) * 1000.0

                        crop_item = {
                            "crop_id": cm["crop_id"],
                            "crop_index": cm["crop_index"],
                            "status": "completed",
                            "confidence": cm["confidence"],
                            "orientation": cm["orientation_label"],
                            "polygon": cm["crop_polygon"],
                            "core_polygon": cm["core_polygon"],
                            "window_count": int(entry_logits["window_count"]),
                            "multi_window_fallback": bool(entry_logits["multi_window"]),
                            "crop_preview_png_b64": _pil_to_png_b64(cm["upright_crop"]),
                            "greedy_ocr": greedy_c,
                            "decoded_ocr": decoded_c,
                            "decoder_route": route_used,
                            "numeric_disagreement": num_disagree,
                            "translation": tr_payload,
                            "error": None,
                        }
                        crop_failed = False
                    except Exception as exc:
                        crop_item = {
                            "crop_id": cm.get("crop_id", f"crop_{idx+1}"),
                            "crop_index": cm.get("crop_index", idx + 1),
                            "status": "failed",
                            "confidence": cm.get("confidence", 0.0),
                            "orientation": cm.get("orientation_label", "horizontal"),
                            "polygon": cm.get("crop_polygon", []),
                            "core_polygon": cm.get("core_polygon", []),
                            "window_count": 0,
                            "multi_window_fallback": False,
                            "crop_preview_png_b64": None,
                            "greedy_ocr": "",
                            "decoded_ocr": "",
                            "decoder_route": "failed",
                            "numeric_disagreement": False,
                            "translation": None,
                            "error": str(exc),
                        }
                        crop_failed = True

                    crop_work_tot += (time.perf_counter() - t_c0) * 1000.0
                    with self._state_lock:
                        if crop_failed:
                            run["counts"]["failed_crops"] += 1
                        else:
                            run["counts"]["completed_crops"] += 1
                        if run["timings_ms"]["first_result_ms"] is None:
                            run["timings_ms"]["first_result_ms"] = round((time.perf_counter() - req_mono) * 1000.0, 2)
                        run["crops"].append(crop_item)
                        run["counts"]["total_windows"] = total_windows

                if cancel_ev.is_set():
                    self._finalize_run_timings(run, t_pipeline0=t_pipeline0, status="cancelled", stage="cancelled")
                    return

                # Commit immutable settings-keyed cache snapshot ONLY if not cancelled and not superseded
                if not can_reuse_logits:
                    immutable_snap = {
                        "confidence": conf,
                        "raw_count": raw_count_for_cache,
                        "crops_truncated": crops_truncated_for_cache,
                        "crops_meta": tuple(crops_meta),
                        "logits": tuple(new_cached_logits),
                        "source_run_seq": run_seq,
                        "source_run_id": run["run_id"],
                    }
                    with self._state_lock:
                        if not cancel_ev.is_set():
                            prev_snap = cap["ocr_cache_by_conf"].get(conf)
                            if prev_snap is None or int(prev_snap["source_run_seq"]) < run_seq:
                                cap["ocr_cache_by_conf"][conf] = immutable_snap

                # Verify 147 BN buffers remained bit-identical!
                assert verify_bn_buffers_identical(self._model, self._bn_ref), "147 BN buffers mutated during inference!"

                with self._state_lock:
                    run["timings_ms"]["recognize_ms"] = round(rec_ms_tot, 2)
                    run["timings_ms"]["decode_ms"] = round(dec_ms_tot, 2)
                    run["timings_ms"]["lookup_ms"] = round(look_ms_tot, 2)
                    run["timings_ms"]["summed_crop_work_ms"] = round(crop_work_tot, 2)
                self._finalize_run_timings(run, t_pipeline0=t_pipeline0, status="completed", stage="done")
            except Exception as exc:
                self._finalize_run_timings(run, t_pipeline0=t_pipeline0, status="failed", stage="error", error=str(exc))
            finally:
                with self._state_lock:
                    self._inflight_model_runs -= 1


def create_stage4_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8790,
    engine: Stage4Engine | None = None,
    session_token: str | None = None,
    web_dir: Path | None = None,
    extra_allowed_origins: tuple[str, ...] = (),
) -> tuple[ThreadingHTTPServer, str]:
    """Create loopback-only HTTP server with session-token auth and strict CORS."""
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise ValueError(f"Refusing non-loopback bind host={host!r}; Stage 4 desktop backend is loopback-only.")
    bind_host = "127.0.0.1" if host == "localhost" else host
    token = session_token or secrets.token_hex(24)
    eng = engine or Stage4Engine()
    web_root = Path(web_dir).resolve() if (web_dir and Path(web_dir).is_dir()) else None

    allowed_origins = {
        f"http://127.0.0.1:{port}",
        f"http://localhost:{port}",
        *extra_allowed_origins,
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return  # quiet structured logs

        def _cors_headers(self) -> None:
            origin = self.headers.get("Origin")
            if origin and origin in allowed_origins:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Stage4-Token, X-Capture-Id, X-Stream-Width, X-Stream-Height, X-Stream-Rotation, X-Capture-Source, X-Filename, X-Confidence, X-Alpha-Lm, X-Decode-Mode")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self._cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _require_auth(self) -> bool:
            got = self.headers.get("X-Stage4-Token", "")
            if not secrets.compare_digest(got, token):
                self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "Unauthorized: missing or invalid X-Stage4-Token"})
                return False
            return True

        def do_OPTIONS(self) -> None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self._cors_headers()
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/api/health":
                self._send_json(HTTPStatus.OK, eng.health_summary())
                return

            if path == "/api/bootstrap":
                # Loopback same-origin bootstrap so the served Flutter Web app receives its session token cleanly
                host_hdr = (self.headers.get("Host") or "").split(":")[0]
                if host_hdr not in ("127.0.0.1", "localhost"):
                    self._send_json(HTTPStatus.FORBIDDEN, {"error": "Bootstrap allowed on loopback only"})
                    return
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "session_token": token,
                        "backend_id": eng.backend_id,
                        "device": str(eng.device),
                        "device_name": eng.device_name,
                        "bundle_id": eng.bundle["bundle_id"],
                        "ui_mode": "flutter_web_desktop_demo",
                    },
                )
                return

            if path == "/api/bundle":
                if not self._require_auth():
                    return
                self._send_json(HTTPStatus.OK, eng.bundle)
                return

            m_run = re.fullmatch(r"/api/runs/([A-Za-z0-9_.-]+)", path)
            if m_run:
                if not self._require_auth():
                    return
                try:
                    self._send_json(HTTPStatus.OK, eng.get_run(m_run.group(1)))
                except KeyError as exc:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                return

            m_img = re.fullmatch(r"/api/captures/([A-Za-z0-9_.-]+)/image", path)
            if m_img:
                if not self._require_auth():
                    return
                try:
                    raw = eng.get_capture_image_bytes(m_img.group(1))
                    self.send_response(HTTPStatus.OK)
                    self._cors_headers()
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(raw)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(raw)
                except KeyError as exc:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                return

            # Serve built Flutter web assets strictly inside web_root (path-traversal protected)
            if web_root is not None and not path.startswith("/api/"):
                rel = "index.html" if path in ("", "/") else path.lstrip("/")
                cand = (web_root / rel).resolve()
                if not cand.is_relative_to(web_root) or not cand.is_file():
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                    return
                ctype, _ = mimetypes.guess_type(str(cand))
                raw = cand.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", ctype or "application/octet-stream")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found"})

        def do_POST(self) -> None:
            req_mono = time.perf_counter()
            parsed = urlparse(self.path)
            path = parsed.path
            if not self._require_auth():
                return

            content_len = int(self.headers.get("Content-Length", "0"))
            if content_len > MAX_MENU_UPLOAD_BYTES + 4096:
                self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Upload exceeds 16 MB limit"})
                return
            t_read0 = time.perf_counter()
            body = self.rfile.read(content_len) if content_len > 0 else b""
            request_read_ms = round((time.perf_counter() - t_read0) * 1000.0, 2)

            if path == "/api/capture":
                try:
                    sw = self.headers.get("X-Stream-Width")
                    sh = self.headers.get("X-Stream-Height")
                    srot = self.headers.get("X-Stream-Rotation")
                    run = eng.start_capture_job(
                        image_bytes=body,
                        filename=self.headers.get("X-Filename", "capture.png"),
                        capture_id=self.headers.get("X-Capture-Id") or None,
                        stream_width=int(sw) if sw else None,
                        stream_height=int(sh) if sh else None,
                        stream_rotation=int(srot) if srot else 0,
                        capture_source=self.headers.get("X-Capture-Source", "import"),
                        confidence=float(self.headers.get("X-Confidence", "0.35")),
                        alpha_lm=float(self.headers.get("X-Alpha-Lm", "0.30")),
                        decode_mode=self.headers.get("X-Decode-Mode", "charlm_r1"),
                        request_received_mono=req_mono,
                        request_read_ms=request_read_ms,
                    )
                    self._send_json(HTTPStatus.ACCEPTED, run)
                except (ValueError, OverflowError) as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            if path == "/api/reprocess":
                try:
                    obj = json.loads(body.decode("utf-8"))
                    run = eng.start_reprocess_job(
                        capture_id=str(obj["capture_id"]),
                        confidence=float(obj.get("confidence", 0.35)),
                        alpha_lm=float(obj.get("alpha_lm", 0.30)),
                        decode_mode=str(obj.get("decode_mode", "charlm_r1")),
                        request_received_mono=req_mono,
                        request_read_ms=request_read_ms,
                    )
                    self._send_json(HTTPStatus.ACCEPTED, run)
                except KeyError as exc:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                except (ValueError, json.JSONDecodeError) as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            m_can = re.fullmatch(r"/api/runs/([A-Za-z0-9_.-]+)/cancel", path)
            if m_can:
                try:
                    self._send_json(HTTPStatus.OK, eng.cancel_run(m_can.group(1)))
                except KeyError as exc:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found"})

    srv = ThreadingHTTPServer((bind_host, int(port)), Handler)
    return srv, token


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 4 Desktop Reference Python Backend (loopback-only)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--web-dir", type=Path, default=None)
    ap.add_argument("--state-file", type=Path, default=None)
    ap.add_argument("--preload", action="store_true", help="Preload models on startup before serving")
    args = ap.parse_args(argv)

    eng = Stage4Engine()
    if args.preload:
        cold_ms = eng.ensure_loaded()
        print(f"[INFO] Preloaded Stage 4 models on {eng.device_name} in {cold_ms}ms", flush=True)

    srv, token = create_stage4_server(host=args.host, port=args.port, engine=eng, web_dir=args.web_dir)
    if args.state_file:
        args.state_file.parent.mkdir(parents=True, exist_ok=True)
        args.state_file.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "url": f"http://127.0.0.1:{args.port}/",
                    "health_url": f"http://127.0.0.1:{args.port}/api/health",
                    "backend_id": eng.backend_id,
                    "device": str(eng.device),
                    "device_name": eng.device_name,
                    "bundle_id": eng.bundle["bundle_id"],
                    "web_dir": str(args.web_dir) if args.web_dir else None,
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"[LIVE] Stage 4 Desktop Reference Backend listening on http://127.0.0.1:{args.port}/ (PID={os.getpid()})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        eng.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
