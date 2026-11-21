"""Unit and HTTP integration tests for full-menu photo upload, crop extraction, and async jobs."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import torch
from PIL import Image
import scripts.recognition.serve_lexicon_inspector as inspector

from scripts.recognition.serve_lexicon_inspector import (
    HTML_PAGE,
    InspectorAppState,
    make_handler,
)
from src.recognition.lexicon_lookup import (
    LexiconEntry,
    LexiconIndex,
    LexiconPipeline,
    LookupConfig,
    MenuJobManager,
    decode_uploaded_image_bytes,
    extract_menu_crops,
    run_full_menu_pipeline,
)


def _mini_index() -> LexiconIndex:
    entries = (
        LexiconEntry(0, "定食", "定食", "curated_core", ("test:core",)),
        LexiconEntry(1, "豚ロース", "豚ロース", "jmdict_food_cleaned", ("test:exp",)),
    )
    return LexiconIndex(entries=entries, metadata={"schema_version": 1, "index_format": "lexicon_trie_v1"})


def test_explicit_inspector_candidate_requires_hash_and_loads_raw_not_ema(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 1)
    model.vocabulary = {"円": 1}
    monkeypatch.setattr(inspector, "ROOT", tmp_path)
    monkeypatch.setattr(inspector, "load_current_recognizer",
                        lambda device: (model, {"active_architecture": "h4_k41", "active_state_key": "ema"}))
    p = tmp_path / "epoch_05.pt"
    raw = {k: torch.full_like(v, 3) for k, v in model.state_dict().items()}
    ema = {k: torch.full_like(v, 4) for k, v in model.state_dict().items()}
    torch.save({"raw": raw, "ema": ema, "epoch": 5}, p)
    with pytest.raises(ValueError, match="SHA-256"):
        inspector.load_inspection_model("cpu", p, "wrong")
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    loaded, meta = inspector.load_inspection_model("cpu", p, digest, "raw", "dishlex50 epoch5")
    assert all(torch.equal(v, raw[k]) for k, v in loaded.state_dict().items())
    assert not loaded.training
    assert meta["state_key"] == "raw" and meta["epoch"] == 5
    assert meta["version"] == "dishlex50 epoch5" and meta["registry_override"]


class _DummyRecognizer:
    def __init__(self, *, fail_on_call: int | None = None, empty: bool = False) -> None:
        self.vocabulary = {"定": 1, "食": 2, "(": 3, "（": 4, ")": 5, "）": 6}
        self.calls: list[tuple[int, int, int]] = []  # (H, W, orientation)
        self._fail_on = fail_on_call
        self._empty = empty
        self._count = 0

    def __call__(self, x: torch.Tensor, ori: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        self._count += 1
        if self._fail_on is not None and self._count == self._fail_on:
            raise RuntimeError("injected_crop_inference_failure")
        h, w = int(x.shape[2]), int(x.shape[3])
        self.calls.append((h, w, int(ori[0].item())))
        T = int(lengths[0].item())
        C = len(self.vocabulary) + 1
        out = torch.full((T, 1, C), -8.0)
        out[:, 0, 0] = 6.0  # blank
        if not self._empty and T >= 4:
            out[0, 0, 1] = 8.0  # 定
            out[2, 0, 2] = 8.0  # 食
        return out


class _MockDetector:
    def __init__(self, dets: list[dict], *, delay_sec: float = 0.0) -> None:
        self._dets = dets
        self._delay = delay_sec
        self.call_count = 0

    def detect(self, rgb_image: Image.Image, *, confidence: float = 0.35):
        self.call_count += 1
        if self._delay > 0:
            time.sleep(self._delay)
        return list(self._dets), {
            "weights_name": "mock_v5.pt",
            "weights_sha256": "0" * 64,
            "confidence_threshold": confidence,
            "detection_ms": 1.2,
        }


def _img_bytes(w: int, h: int, fmt: str = "PNG", exif_orientation: int | None = None) -> bytes:
    im = Image.new("RGB", (w, h), color=(245, 245, 245))
    buf = io.BytesIO()
    if fmt == "JPEG" and exif_orientation is not None:
        exif = Image.Exif()
        exif[274] = exif_orientation
        im.save(buf, format="JPEG", exif=exif.tobytes())
    else:
        im.save(buf, format=fmt)
    return buf.getvalue()


def test_overlapping_fragments_preserve_nested_merge_provenance():
    dets = [
        {"id": "a", "label": "dish_horizontal", "confidence": 0.9,
         "points": [[10, 10], [70, 10], [70, 30], [10, 30]]},
        {"id": "b", "label": "dish_horizontal", "confidence": 0.8,
         "points": [[50, 10], [110, 10], [110, 30], [50, 30]]},
    ]
    extracted = extract_menu_crops(Image.new("RGB", (150, 50)), dets)
    assert len(extracted["crops"]) == 1
    crop = extracted["crops"][0]
    assert set(crop["source_ids"]) == {"a", "b"}
    merged = [d for d in crop["merge_decisions"] if d["action"] == "merge_line_fragments"]
    assert len(merged) == 1
    assert merged[0]["sources"] == [["a"], ["b"]]
    assert crop["rectified_strip"].width > 90


def test_exif_transpose_formats_and_heic_or_pixel_rejection():
    # JPEG with EXIF orientation=6 (rotate 90 CW -> 120x80 becomes 80x120)
    jpg_b = _img_bytes(120, 80, fmt="JPEG", exif_orientation=6)
    im, meta = decode_uploaded_image_bytes(jpg_b)
    assert meta["format"] == "JPEG"
    assert meta["exif_transposed"] is True
    assert (im.width, im.height) == (80, 120)

    # PNG and WebP both supported
    for fmt in ("PNG", "WEBP"):
        im2, m2 = decode_uploaded_image_bytes(_img_bytes(100, 60, fmt=fmt))
        assert m2["format"] == fmt
        assert (im2.width, im2.height) == (100, 60)

    # HEIC magic bytes rejected with actionable message
    fake_heic = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00"
    with pytest.raises(ValueError, match="HEIC"):
        decode_uploaded_image_bytes(fake_heic)

    # Oversized bytes & oversized pixel bounds rejected cleanly
    with pytest.raises(OverflowError):
        decode_uploaded_image_bytes(jpg_b, max_bytes=32)
    with pytest.raises(ValueError, match="pixel"):
        decode_uploaded_image_bytes(_img_bytes(200, 200, fmt="PNG"), max_pixels=10_000)


def test_vertical_single_rotation_no_double_rotate_and_oversized_overlap_route():
    img_b = _img_bytes(1200, 400, fmt="PNG")
    dets = [
        # Horizontal normal crop (160x40)
        {"id": "r0", "label": "dish_horizontal", "confidence": 0.92, "points": [[20, 20], [180, 20], [180, 60], [20, 60]]},
        # Vertical crop (40x180 -> height >> width)
        {"id": "r1", "label": "dish_vertical", "confidence": 0.88, "points": [[1100, 20], [1140, 20], [1140, 200], [1100, 200]]},
        # Oversized wide horizontal crop (900x40 -> H64 native width ~1440 > 512 -> overlap512_h64!)
        {"id": "r2", "label": "dish_horizontal", "confidence": 0.90, "points": [[20, 300], [920, 300], [920, 340], [20, 340]]},
    ]
    model = _DummyRecognizer()
    pipe = LexiconPipeline(_mini_index(), model=model, device="cpu")
    det = _MockDetector(dets)

    out = run_full_menu_pipeline(
        image_bytes=img_b,
        filename="menu_test.png",
        detector=det,
        pipeline=pipe,
        lookup_config=LookupConfig(),
        decode_mode="greedy",
    )
    ov = out["overview"]
    assert ov["counts"] == {"total": 3, "ok": 3, "empty": 0, "error": 0}
    assert ov["auto_corrected"] is False

    # Verify vertical crop: upright preview is taller than wide, OCR strip is wider than tall (rotated CCW ONCE!)
    vert_detail = next(d for d in out["crop_details"].values() if d["geometry"]["orientation_flag"] == 1)
    u_w, u_h = vert_detail["geometry"]["upright_size"]
    s_w, s_h = vert_detail["geometry"]["ocr_strip_size"]
    assert u_h > u_w, f"Expected upright vertical preview taller than wide, got {u_w}x{u_h}"
    assert s_w > s_h and s_h == 64, f"Expected rectified H64 strip wider than tall, got {s_w}x{s_h}"
    # Model received orientation=1 AND width > height (proving NO double rotation!)
    vert_model_calls = [c for c in model.calls if c[2] == 1]
    assert len(vert_model_calls) == 1
    assert vert_model_calls[0][0] == 64 and vert_model_calls[0][1] >= 128

    # Verify oversized crop routed via overlap512_h64 with multi-window warning
    wide_detail = next(
        d for d in out["crop_details"].values() if d["result"]["provenance"]["crop"]["long_line_route"] == "overlap512_h64"
    )
    assert wide_detail["result"]["provenance"]["crop"]["window_count"] >= 3
    assert "visual_rescoring_unavailable_for_multi_window_overlap" in wide_detail["result"]["warnings"]


def test_empty_detections_no_text_crop_and_per_crop_exception_isolation():
    img_b = _img_bytes(400, 300, fmt="PNG")

    # 1. Empty detections
    pipe_ok = LexiconPipeline(_mini_index(), model=_DummyRecognizer(), device="cpu")
    out_zero = run_full_menu_pipeline(
        image_bytes=img_b,
        filename="blank.png",
        detector=_MockDetector([]),
        pipeline=pipe_ok,
        lookup_config=LookupConfig(),
    )
    assert out_zero["overview"]["counts"] == {"total": 0, "ok": 0, "empty": 0, "error": 0}

    # 2. No-text crop -> empty canonical_ocr, never forces dictionary match
    dets_one = [{"id": "r0", "label": "dish_horizontal", "confidence": 0.6, "points": [[20, 20], [160, 20], [160, 60], [20, 60]]}]
    pipe_empty = LexiconPipeline(_mini_index(), model=_DummyRecognizer(empty=True), device="cpu")
    out_empty = run_full_menu_pipeline(
        image_bytes=img_b,
        filename="notext.png",
        detector=_MockDetector(dets_one),
        pipeline=pipe_empty,
        lookup_config=LookupConfig(),
    )
    assert out_empty["overview"]["counts"] == {"total": 1, "ok": 0, "empty": 1, "error": 0}
    c0 = out_empty["crop_details"]["line_000"]
    assert c0["status"] == "empty" and c0["result"]["canonical_ocr"] == "" and c0["result"]["suggested_line"] == ""

    # 3. Per-crop exception isolation: crop 2 throws, crop 1 still succeeds
    dets_two = [
        {"id": "r0", "label": "dish_horizontal", "confidence": 0.9, "points": [[20, 20], [160, 20], [160, 60], [20, 60]]},
        {"id": "r1", "label": "dish_horizontal", "confidence": 0.9, "points": [[20, 120], [160, 120], [160, 160], [20, 160]]},
    ]
    pipe_fail = LexiconPipeline(_mini_index(), model=_DummyRecognizer(fail_on_call=2), device="cpu")
    out_iso = run_full_menu_pipeline(
        image_bytes=img_b,
        filename="partial.png",
        detector=_MockDetector(dets_two),
        pipeline=pipe_fail,
        lookup_config=LookupConfig(),
    )
    assert out_iso["overview"]["counts"] == {"total": 2, "ok": 1, "empty": 0, "error": 1}
    assert out_iso["crop_details"]["line_000"]["status"] == "ok"
    assert out_iso["crop_details"]["line_001"]["status"] == "error"
    assert "injected_crop_inference_failure" in out_iso["crop_details"]["line_001"]["error"]


def test_menu_job_manager_http_lifecycle_rerun_caching_and_xss_safety():
    assert ".innerHTML" not in HTML_PAGE

    img_b = _img_bytes(400, 300, fmt="PNG")
    b64_url = "data:image/png;base64," + base64.b64encode(img_b).decode("ascii")
    dets = [{"id": "r0", "label": "dish_horizontal", "confidence": 0.91, "points": [[20, 20], [180, 20], [180, 60], [20, 60]]}]
    det = _MockDetector(dets)
    model = _DummyRecognizer()
    pipe = LexiconPipeline(_mini_index(), model=model, device="cpu")
    mgr = MenuJobManager(detector=det, pipeline=pipe, max_resident_jobs=2, max_queue_size=4)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(InspectorAppState(pipe, {}, menu_jobs=mgr)))
    port = server.server_address[1]
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{port}"
        # 1. Health reports has_yolo=True
        with urllib.request.urlopen(f"{base}/api/health") as r:
            h = json.loads(r.read().decode("utf-8"))
            assert h["has_yolo"] is True and h["auto_corrected_default"] is False

        # 2. Submit full-menu job -> HTTP 202
        req = urllib.request.Request(
            f"{base}/api/menu/jobs",
            data=json.dumps({"image_base64": b64_url, "filename": "m1.png", "decoder_mode": "greedy"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as r:
            assert r.status == 202
            st = json.loads(r.read().decode("utf-8"))
            jid = st["job_id"]

        # 3. Poll status until completed (fast mocked run)
        for _ in range(40):
            with urllib.request.urlopen(f"{base}/api/menu/jobs/{jid}/status") as r:
                st = json.loads(r.read().decode("utf-8"))
            if st["status"] in ("completed", "failed"):
                break
            time.sleep(0.05)
        assert st["status"] == "completed"
        calls_after_run = det.call_count

        # 4. Fetch overview and crop detail from cache (must NOT re-run detector!)
        with urllib.request.urlopen(f"{base}/api/menu/jobs/{jid}/result") as r:
            ov = json.loads(r.read().decode("utf-8"))["overview"]
        assert ov["counts"]["total"] == 1
        cid = ov["crops"][0]["crop_id"]

        with urllib.request.urlopen(f"{base}/api/menu/jobs/{jid}/crops/{cid}") as r:
            cd = json.loads(r.read().decode("utf-8"))["crop_detail"]
        assert cd["crop_id"] == cid and cd["result"]["canonical_ocr"] == "定食"
        assert det.call_count == calls_after_run  # Strictly cached!

        # 5. Rerun with different config -> distinct job_id and config_signature
        req_re = urllib.request.Request(
            f"{base}/api/menu/jobs/{jid}/rerun",
            data=json.dumps({"decoder_mode": "plain_beam", "exact_only": True, "confidence": 0.45}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_re) as r:
            assert r.status == 202
            st2 = json.loads(r.read().decode("utf-8"))
        assert st2["job_id"] != jid
        assert st2["config_signature"] != st["config_signature"]

        # 6. Unknown job_id / crop_id -> HTTP 404
        with pytest.raises(urllib.error.HTTPError) as e404:
            urllib.request.urlopen(f"{base}/api/menu/jobs/deadbeef1234/status")
        assert e404.value.code == 404
    finally:
        mgr.close()
        server.shutdown()
        server.server_close()


def test_extract_menu_crops_handles_merge_line_fragments_nested_sources():
    """Regression: merge_line_fragments emits nested list `sources=[['r0'], ['r1']]`."""
    im = Image.new("RGB", (600, 200), color=(255, 255, 255))
    # Two overlapping collinear horizontal boxes that trigger `merge_line_fragments` (iom >= 0.20)
    raw_dets = [
        {"id": "raw_000", "label": "dish_horizontal", "confidence": 0.90, "points": [[40, 60], [180, 60], [180, 100], [40, 100]]},
        {"id": "raw_001", "label": "dish_horizontal", "confidence": 0.85, "points": [[130, 60], [270, 60], [270, 100], [130, 100]]},
    ]
    extracted = extract_menu_crops(im, raw_dets)
    assert extracted["raw_count"] == 2
    assert extracted["processed_count"] == 1
    assert any(d["action"] == "merge_line_fragments" for d in extracted["decisions"])
    c0 = extracted["crops"][0]
    assert set(c0["source_ids"]) == {"raw_000", "raw_001"}
    assert any(d["action"] == "merge_line_fragments" for d in c0["merge_decisions"])
