"""Focused tests for Stage 4 desktop reference backend: bundle manifest, security,
loopback/token/CORS guards, cancellation, cached-logits reprocessing, and real full-menu E2E.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

from PIL import Image
import pytest

from src.stage4.bundle_manifest import verify_pinned_stage4_bundle
from src.stage4.desktop_backend import (
    Stage4Engine,
    create_stage4_server,
    validate_capture_resolution_parity,
)

ROOT = Path(__file__).resolve().parents[2]
GENAI_MENU_101 = ROOT / "data/processed/gemini_banana_mobile_menus_50_crops"  # crops dir; find actual full menu photo below


def _make_png_bytes(w: int = 320, h: int = 240) -> bytes:
    im = Image.new("RGB", (w, h), (245, 240, 230))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _http_req(url: str, *, method: str = "GET", headers: dict[str, str] | None = None, data: bytes | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def test_pinned_bundle_manifest_verifies_all_6_assets():
    bundle = verify_pinned_stage4_bundle(ROOT, strict=True)
    assert bundle["all_verified"] is True
    assert len(bundle["assets"]) == 6
    assert all(a["sha256_ok"] for a in bundle["assets"].values())


def test_non_loopback_bind_refused():
    with pytest.raises(ValueError, match="non-loopback"):
        create_stage4_server(host="0.0.0.0", port=0)


def test_http_security_auth_bounds_and_real_full_menu_e2e_with_reprocess_and_cancel(tmp_path: Path):
    eng = Stage4Engine(root=ROOT, cache_dir=tmp_path / "cap_cache", max_cached_pages=2)
    srv, token = create_stage4_server(host="127.0.0.1", port=0, engine=eng)
    port = srv.server_address[1]
    base = f"http://127.0.0.1:{port}"
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        # 1. Public health works without token and leaks no secrets
        code, health = _http_req(f"{base}/api/health")
        assert code == 200 and health["status"] == "ok" and "session_token" not in health

        # 2. Protected endpoints reject missing/wrong token (401)
        code, _ = _http_req(f"{base}/api/bundle")
        assert code == 401
        code, _ = _http_req(f"{base}/api/bundle", headers={"X-Stage4-Token": "wrong"})
        assert code == 401

        auth = {"X-Stage4-Token": token}
        code, bnd = _http_req(f"{base}/api/bundle", headers=auth)
        assert code == 200 and bnd["all_verified"] is True

        # 3. Invalid settings rejected cleanly (400)
        code, err = _http_req(
            f"{base}/api/capture",
            method="POST",
            headers={**auth, "X-Confidence": "0.99", "X-Filename": "bad.png"},
            data=_make_png_bytes(),
        )
        assert code == 400 and "confidence" in err["error"]

        # 4. Real full-menu E2E using an actual generated full menu image if present (else synthetic menu)
        full_menus = sorted((ROOT / "data/raw/gemini_banana_mobile_menus_50").glob("menu-*.png")) + sorted(
            (ROOT / "data/raw/gemini_banana_mobile_menus_50").glob("menu-*.jpg")
        )
        if full_menus:
            menu_bytes = full_menus[0].read_bytes()
            menu_name = full_menus[0].name
            with Image.open(full_menus[0]) as im:
                sw, sh = im.size
        else:
            menu_bytes = _make_png_bytes(640, 480)
            menu_name = "blank_test.png"
            sw, sh = 640, 480

        code, run1 = _http_req(
            f"{base}/api/capture",
            method="POST",
            headers={
                **auth,
                "X-Filename": menu_name,
                "X-Stream-Width": str(sw),
                "X-Stream-Height": str(sh),
                "X-Capture-Source": "import",
                "X-Confidence": "0.35",
                "X-Alpha-Lm": "0.30",
                "X-Decode-Mode": "charlm_r1",
            },
            data=menu_bytes,
        )
        assert code == 202
        run1_id = run1["run_id"]
        cap_id = run1["capture_id"]

        # Poll until completed (bounded timeout)
        deadline = time.time() + 120
        while time.time() < deadline:
            code, cur = _http_req(f"{base}/api/runs/{run1_id}", headers=auth)
            assert code == 200
            if cur["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(0.25)
        assert cur["status"] == "completed", f"Run failed: {cur.get('error')}"
        assert cur["cache_status"] == "fresh_full_inference"
        assert cur["reused_cached_logits"] is False
        assert cur["reused_cached_detections"] is False
        assert cur["capture"]["stream_saved_parity"] is True
        assert cur["capture"]["stream_saved_parity_mode"] == "exact"
        tm1 = cur["timings_ms"]
        assert tm1["server_request_total_ms"] is not None and tm1["server_request_total_ms"] > 0
        assert tm1["server_pipeline_wall_ms"] is not None and tm1["server_pipeline_wall_ms"] > 0
        assert tm1["total_wall_ms"] == tm1["server_request_total_ms"]
        assert tm1["server_request_total_ms"] >= tm1["decode_normalize_ms"] + tm1["persist_ms"]
        if full_menus:
            assert cur["counts"]["completed_crops"] >= 1
            assert any(c["decoded_ocr"] for c in cur["crops"])

        # 5. Reprocess with alpha=0.15 (same conf) MUST reuse cached logits!
        reproc_body = json.dumps({"capture_id": cap_id, "confidence": 0.35, "alpha_lm": 0.15, "decode_mode": "charlm_r1"}).encode("utf-8")
        code, run2 = _http_req(f"{base}/api/reprocess", method="POST", headers=auth, data=reproc_body)
        assert code == 202
        deadline = time.time() + 60
        while time.time() < deadline:
            code, cur2 = _http_req(f"{base}/api/runs/{run2['run_id']}", headers=auth)
            if cur2["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(0.1)
        assert cur2["status"] == "completed"
        assert cur2["cache_status"] == "reused_cached_logits"
        assert cur2["reused_cached_logits"] is True
        assert cur2["reused_cached_detections"] is True
        assert cur2["timings_ms"]["detect_ms"] == 0.0
        assert cur2["timings_ms"]["recognize_ms"] == 0.0

        # 6. Misreported stream dimensions (unjustified W/H swap) rejected with 400
        code, swap_err = _http_req(
            f"{base}/api/capture",
            method="POST",
            headers={
                **auth,
                "X-Filename": "swapped.png",
                "X-Stream-Width": "240",
                "X-Stream-Height": "320",
                "X-Capture-Source": "webcam_stream_buffer",
            },
            data=_make_png_bytes(320, 240),
        )
        assert code == 400 and "Misreported stream dimensions" in swap_err["error"]

        # 7. Cancel endpoint responds cleanly
        code, can = _http_req(f"{base}/api/runs/{run2['run_id']}/cancel", method="POST", headers=auth)
        assert code == 200 and can["cancel_requested"] is True
    finally:
        srv.shutdown()
        srv.server_close()
        eng.close()


def test_capture_resolution_parity_ordered_and_rotation_semantics():
    # Exact match passes
    ok_exact = validate_capture_resolution_parity(
        stream_width=1280, stream_height=720, saved_width=1280, saved_height=720
    )
    assert ok_exact["stream_saved_parity"] is True
    assert ok_exact["stream_saved_parity_mode"] == "exact"

    # Swapped W/H passes ONLY with justified 90/270 rotation or EXIF axis swap (5..8)
    ok_rot90 = validate_capture_resolution_parity(
        stream_width=1280, stream_height=720, saved_width=720, saved_height=1280, stream_rotation=90
    )
    assert ok_rot90["stream_saved_parity"] is True
    assert ok_rot90["stream_saved_parity_mode"] == "rotated_90_270_swap"

    ok_exif6 = validate_capture_resolution_parity(
        stream_width=1280, stream_height=720, saved_width=720, saved_height=1280, exif_orientation=6
    )
    assert ok_exif6["stream_saved_parity"] is True
    assert ok_exif6["stream_saved_parity_mode"] == "rotated_90_270_swap"

    # Swapped W/H WITHOUT justified rotation MUST be rejected (no set-comparison loophole)
    with pytest.raises(ValueError, match="Misreported stream dimensions"):
        validate_capture_resolution_parity(
            stream_width=1280, stream_height=720, saved_width=720, saved_height=1280, stream_rotation=0
        )

    # Rotation=90 claiming unswapped non-square dimensions MUST be rejected
    with pytest.raises(ValueError, match="Misreported stream dimensions"):
        validate_capture_resolution_parity(
            stream_width=1280, stream_height=720, saved_width=1280, saved_height=720, stream_rotation=90
        )

    # Mismatched dimensions MUST be rejected
    with pytest.raises(ValueError, match="Misreported stream dimensions"):
        validate_capture_resolution_parity(
            stream_width=640, stream_height=360, saved_width=1280, saved_height=720
        )


def test_server_request_total_accounts_for_delayed_decode_and_persist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import src.stage4.desktop_backend as db_mod

    orig_decode = db_mod.decode_uploaded_image_bytes

    def delayed_decode(*args, **kwargs):
        time.sleep(0.06)
        return orig_decode(*args, **kwargs)

    monkeypatch.setattr(db_mod, "decode_uploaded_image_bytes", delayed_decode)
    monkeypatch.setattr(db_mod, "verify_bn_buffers_identical", lambda *a, **k: True)

    eng = Stage4Engine(root=ROOT, cache_dir=tmp_path / "delay_cache", max_cached_pages=2)
    # Use lightweight stubs so we measure decode+persist accounting cleanly and fast
    class _FastDetector:
        def detect(self, rgb, confidence=0.35):
            return [], 0.1

    eng._loaded = True
    eng._detector = _FastDetector()
    eng._model = object()
    eng._lm = object()
    eng._lex_pipe = object()
    eng._repo = object()
    eng._lex_index = object()
    try:
        run = eng.start_capture_job(
            image_bytes=_make_png_bytes(320, 240),
            filename="delayed.png",
            stream_width=320,
            stream_height=240,
            capture_source="webcam_stream_buffer",
        )
        deadline = time.time() + 10
        while time.time() < deadline:
            cur = eng.get_run(run["run_id"])
            if cur["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(0.02)
        assert cur["status"] == "completed"
        tm = cur["timings_ms"]
        assert tm["decode_normalize_ms"] >= 50.0
        assert tm["request_ingest_ms"] >= tm["decode_normalize_ms"] + tm["persist_ms"] - 1.0
        assert tm["server_request_total_ms"] >= tm["request_ingest_ms"] + tm["server_pipeline_wall_ms"] - 2.0
        assert tm["server_request_total_ms"] > tm["server_pipeline_wall_ms"] + 40.0
        assert tm["total_wall_ms"] == tm["server_request_total_ms"]
    finally:
        eng.close()


def test_rapid_reprocess_retake_cancel_serializes_inference_and_prevents_stale_cache_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import src.stage4.desktop_backend as db_mod

    monkeypatch.setattr(db_mod, "verify_bn_buffers_identical", lambda *a, **k: True)

    eng = Stage4Engine(root=ROOT, cache_dir=tmp_path / "serial_cache", max_cached_pages=4)
    conc_lock = threading.Lock()
    state = {"active": 0, "max_active": 0, "detect_calls": []}

    class _SlowTrackingDetector:
        def detect(self, rgb, confidence=0.35):
            with conc_lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
                state["detect_calls"].append(round(float(confidence), 2))
            try:
                time.sleep(0.12)
                return [], 120.0
            finally:
                with conc_lock:
                    state["active"] -= 1

    eng._loaded = True
    eng._detector = _SlowTrackingDetector()
    eng._model = object()
    eng._lm = object()
    eng._lex_pipe = object()
    eng._repo = object()
    eng._lex_index = object()
    try:
        # 1. Start initial capture (conf=0.35)
        r1 = eng.start_capture_job(
            image_bytes=_make_png_bytes(320, 240),
            filename="rapid.png",
            stream_width=320,
            stream_height=240,
            confidence=0.35,
        )
        cap_id = r1["capture_id"]
        # Give r1 a moment to enter detect()
        time.sleep(0.03)

        # 2. Fire rapid reprocess storm: r2 (conf=0.40), r3 (conf=0.45), r4 (conf=0.50)
        r2 = eng.start_reprocess_job(capture_id=cap_id, confidence=0.40, alpha_lm=0.20, decode_mode="charlm_r1")
        r3 = eng.start_reprocess_job(capture_id=cap_id, confidence=0.45, alpha_lm=0.10, decode_mode="greedy")
        r4 = eng.start_reprocess_job(capture_id=cap_id, confidence=0.50, alpha_lm=0.30, decode_mode="charlm_r1")

        # Wait for all 4 runs to reach terminal state
        deadline = time.time() + 10
        while time.time() < deadline:
            statuses = [eng.get_run(r["run_id"])["status"] for r in (r1, r2, r3, r4)]
            if all(s in ("completed", "failed", "cancelled") for s in statuses):
                break
            time.sleep(0.02)

        f1 = eng.get_run(r1["run_id"])
        f2 = eng.get_run(r2["run_id"])
        f3 = eng.get_run(r3["run_id"])
        f4 = eng.get_run(r4["run_id"])

        # Strict serialization invariant: at most 1 in-flight model execution ever!
        assert state["max_active"] <= 1
        assert eng._max_concurrent_model_runs <= 1

        # Stale queued runs r2 and r3 MUST be cancelled BEFORE doing any model work!
        assert f2["status"] == "cancelled" and f2["stage"] == "cancelled_before_start"
        assert f3["status"] == "cancelled" and f3["stage"] == "cancelled_before_start"
        assert f2["timings_ms"]["server_pipeline_wall_ms"] == 0.0
        assert f3["timings_ms"]["server_pipeline_wall_ms"] == 0.0

        # Superseded r1 was cancelled and MUST NOT have committed its conf=0.35 cache!
        assert f1["status"] == "cancelled"
        # Latest run r4 MUST complete cleanly and own the conf=0.50 cache!
        assert f4["status"] == "completed"

        with eng._state_lock:
            cap_obj = eng._captures[cap_id]
            assert 0.35 not in cap_obj["ocr_cache_by_conf"]
            assert 0.40 not in cap_obj["ocr_cache_by_conf"]
            assert 0.45 not in cap_obj["ocr_cache_by_conf"]
            assert 0.50 in cap_obj["ocr_cache_by_conf"]
            assert cap_obj["ocr_cache_by_conf"][0.50]["source_run_id"] == r4["run_id"]
    finally:
        eng.close()
