"""Unit tests for modular YOLO V5 native-resolution Tabelog quality filtering & durable resume."""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.data import tabelog_collection as crawler
from src.data import tabelog_quality as tq


def _jpeg_bytes(size=(640, 480), color="navy", exif_orientation: int | None = None) -> bytes:
    im = Image.new("RGB", size, color)
    buf = io.BytesIO()
    if exif_orientation is not None:
        exif = im.getexif()
        exif[274] = exif_orientation
        im.save(buf, format="JPEG", exif=exif.tobytes())
    else:
        im.save(buf, format="JPEG")
    return buf.getvalue()


def _rect_quad(x0: float, y0: float, w: float, h: float) -> list[list[float]]:
    return [[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]]


def _rotated_quad(cx: float, cy: float, w: float, h: float, deg: float) -> list[list[float]]:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    u = np.array([c, s])
    v = np.array([-s, c])
    pts = [
        np.array([cx, cy]) - 0.5 * w * u - 0.5 * h * v,
        np.array([cx, cy]) + 0.5 * w * u - 0.5 * h * v,
        np.array([cx, cy]) + 0.5 * w * u + 0.5 * h * v,
        np.array([cx, cy]) - 0.5 * w * u + 0.5 * h * v,
    ]
    return [p.tolist() for p in pts]


class _Resp:
    def __init__(self, content: bytes, content_type: str):
        self.content = content
        self.headers = {"Content-Type": content_type}
        self.text = content.decode("utf-8") if content_type == "text/html" else ""


def test_exact_resolution_boundaries_16_24_32():
    assert tq.classify_crop_resolution(_rect_quad(0, 0, 100, 15.99))["category"] == "too_small"
    assert tq.classify_crop_resolution(_rect_quad(0, 0, 100, 16.0))["category"] == "borderline"
    assert tq.classify_crop_resolution(_rect_quad(0, 0, 100, 23.99))["category"] == "borderline"

    c24 = tq.classify_crop_resolution(_rect_quad(0, 0, 24.0, 24.0))
    assert c24["category"] == "accepted" and c24["is_closeup_32px"] is False

    c31 = tq.classify_crop_resolution(_rect_quad(0, 0, 120.0, 31.99))
    assert c31["category"] == "accepted" and c31["is_closeup_32px"] is False

    c32 = tq.classify_crop_resolution(_rect_quad(0, 0, 120.0, 32.0))
    assert c32["category"] == "accepted" and c32["is_closeup_32px"] is True


def test_core_measured_before_resize_and_not_inflated_by_padding():
    # Core short side = 22px (borderline), padded short side = 28px (>=24px): MUST remain borderline!
    core = _rect_quad(100, 100, 140, 22)
    padded = _rect_quad(95, 97, 150, 28)
    cls = tq.classify_crop_resolution(core, padded)
    assert cls["category"] == "borderline"
    assert cls["core_short_px"] == 22.0
    assert cls["padded_short_px"] == 28.0

    # Also verify through evaluate_detections_for_photo with line_crops.process padding enabled:
    # a 22px core gets padded (~25.5px) by line_crops, yet MUST still be classified as borderline.
    img = Image.new("RGB", (640, 480), "white")
    dets = [{"id": "d0", "label": "dish_horizontal", "confidence": 0.9, "points": _rect_quad(50, 50, 180, 22)}]
    ev = tq.evaluate_detections_for_photo(img, dets)
    assert ev["photo_accepted"] is False
    assert ev["counts"]["accepted"] == 0
    assert ev["counts"]["borderline"] == 1
    assert ev["crop_records"][0]["core_short_px"] == 22.0
    assert ev["crop_records"][0]["padded_short_px"] > 24.0
    # Native rectified strip height stays native (~25-26px), never inflated to H64.
    assert ev["crop_records"][0]["native_rectified_height"] < 32


def test_oriented_vertical_and_bad_geometry():
    img = Image.new("RGB", (640, 480), "white")
    dets = [
        # Rotated vertical 34x160 closeup -> accepted + closeup_32px
        {"id": "v1", "label": "dish_vertical", "confidence": 0.85, "points": _rotated_quad(200, 200, 160, 34, 95)},
        # Degenerate zero-area line
        {"id": "bad1", "label": "dish_horizontal", "confidence": 0.9, "points": [[10, 10], [100, 10], [100, 10], [10, 10]]},
        # NaN coordinate
        {"id": "bad2", "label": "dish_horizontal", "confidence": 0.9, "points": [[10, 10], [float("nan"), 10], [50, 40], [10, 40]]},
    ]
    ev = tq.evaluate_detections_for_photo(img, dets)
    assert ev["photo_accepted"] is True
    assert ev["counts"]["accepted"] == 1
    assert ev["counts"]["closeup_32px"] == 1
    assert ev["counts"]["invalid_geometry"] == 2


def test_exif_transpose_applied_before_inference():
    # Raw 640x480 with EXIF orientation=6 (rotate 90 CW -> 480x640)
    blob = _jpeg_bytes(size=(640, 480), exif_orientation=6)
    seen_sizes = []

    def mock_det(rgb: Image.Image):
        seen_sizes.append(rgb.size)
        return [{"id": "d0", "label": "dish_horizontal", "confidence": 0.9, "points": _rect_quad(20, 20, 200, 36)}]

    ev = tq.YoloMenuQualityEvaluator(detector_fn=mock_det).evaluate_bytes(blob)
    assert seen_sizes == [(480, 640)]
    assert ev["image_meta"]["raw_width"] == 640 and ev["image_meta"]["raw_height"] == 480
    assert ev["image_meta"]["width"] == 480 and ev["image_meta"]["height"] == 640
    assert ev["image_meta"]["exif_orientation"] == 6 and ev["image_meta"]["exif_transposed"] is True


def test_quality_rejection_alternate_photo_and_durable_resume(tmp_path, monkeypatch):
    # Venue 13000001: all 2 photos fail quality -> rejected, never increments success
    # Venue 13000002: photo 0 fails quality, photo 1 passes (closeup 36px) -> accepted on photo 1, photo 2 never requested
    # Venue 13000003 (on page 2): photo 0 passes (3 accepted 26px lines) -> accepted on resume!
    photo_requests = []
    listing_requests = []

    class Client:
        def __init__(self, delay):
            pass
        def close(self):
            pass
        def get(self, url, referer=None):
            if url.endswith("/robots.txt"):
                return _Resp(b"User-agent: *\nDisallow: /private/\n", "text/html")
            if "/rstLst/" in url:
                listing_requests.append(url)
                if "PG=2" not in url:
                    html = (
                        '<a class="cpy-rst-name" href="https://tabelog.com/en/tokyo/A1301/A130103/13000001/">Rest 1</a>'
                        '<a class="cpy-rst-name" href="https://tabelog.com/en/tokyo/A1301/A130103/13000002/">Rest 2</a>'
                        '<a rel="next" href="https://tabelog.com/en/tokyo/rstLst/izakaya/?PG=2">Next</a>'
                    )
                else:
                    html = '<a class="cpy-rst-name" href="https://tabelog.com/en/tokyo/A1301/A130103/13000003/">Rest 3</a>'
                return _Resp(html.encode(), "text/html")
            if "/dtlmenu/photo/" in url:
                rid = url.split("/")[7]
                n_ph = 2 if rid == "13000001" else 3
                html = "".join(
                    f'<a data-id="{rid}_{i}" href="https://tblg.k-img.com/restaurant/images/Rvw/1/{rid}_{i}.jpg">'
                    f'<img class="rstdtl-photo-list__img" alt="Menu {i}"></a>'
                    for i in range(n_ph)
                )
                return _Resp(html.encode(), "text/html")
            photo_requests.append(url)
            if "13000001" in url or url.endswith("13000002_0.jpg"):
                return _Resp(_jpeg_bytes(color=(10, 20, 30 if url.endswith("_0.jpg") else 40)), "image/jpeg")
            if "13000002_1.jpg" in url:
                return _Resp(_jpeg_bytes(color=(200, 10, 10)), "image/jpeg")
            return _Resp(_jpeg_bytes(color=(10, 200, 10)), "image/jpeg")

    def mock_det(rgb: Image.Image):
        r, g, b = rgb.getpixel((5, 5))
        if r > 150:  # 13000002_1 -> 1 closeup 36px + 1 borderline 20px + 1 too_small 12px
            return [
                {"id": "a", "label": "dish_horizontal", "confidence": 0.9, "points": _rect_quad(20, 20, 200, 36)},
                {"id": "b", "label": "dish_horizontal", "confidence": 0.8, "points": _rect_quad(20, 100, 140, 20)},
                {"id": "c", "label": "dish_horizontal", "confidence": 0.7, "points": _rect_quad(20, 160, 80, 12)},
            ]
        if g > 150:  # 13000003_0 -> 3 accepted 26px lines
            return [
                {"id": f"l{i}", "label": "dish_horizontal", "confidence": 0.85, "points": _rect_quad(20, 20 + i * 60, 180, 26)}
                for i in range(3)
            ]
        # Rejected candidates: only 12px too-small line
        return [{"id": "s", "label": "dish_horizontal", "confidence": 0.8, "points": _rect_quad(20, 20, 80, 12)}]

    monkeypatch.setattr(crawler, "PublicClient", Client)
    ev = tq.YoloMenuQualityEvaluator(detector_fn=mock_det)
    status_file = tmp_path / "status.json"

    # Run 1: target=1 -> rejects 13000001 (2 photos), accepts 13000002 on alternate photo 1 (photo 2 skipped)
    assert crawler.run_crawler(tmp_path, target=1, delay=1, photo_kind="menu", quality_evaluator=ev, status_path=status_file) == 0
    mf = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert mf["downloaded_images"] == mf["unique_restaurants"] == 1
    assert mf["images"][0]["restaurant_id"] == "13000002"
    assert mf["images"][0]["photo_candidate_index"] == 2
    assert not (tmp_path / "menu_13000001.jpg").exists()
    assert (tmp_path / "menu_13000002.jpg").is_file()
    assert not any(u.endswith("13000002_2.jpg") for u in photo_requests)
    # Accepted & borderline crops saved; too_small NOT saved as PNG, but recorded in JSONL!
    crop_rows = [json.loads(l) for l in (tmp_path / "crops_manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(crop_rows) == 3
    assert len(list((tmp_path / "crops/accepted").glob("*.png"))) == 1
    assert len(list((tmp_path / "crops/borderline").glob("*.png"))) == 1
    assert (tmp_path / "review/index.html").is_file()
    assert "menu_13000002.jpg" in (tmp_path / "review/index.html").read_text(encoding="utf-8")

    # Run 2: resume with target=2 -> MUST start from PG=2, NOT re-fetch PG=1 or retry rejected 13000001!
    reqs_before = list(photo_requests)
    assert crawler.run_crawler(tmp_path, target=2, delay=1, photo_kind="menu", quality_evaluator=ev, status_path=status_file) == 0
    mf2 = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert mf2["downloaded_images"] == mf2["unique_restaurants"] == 2
    assert {r["restaurant_id"] for r in mf2["images"]} == {"13000002", "13000003"}
    new_photo_reqs = photo_requests[len(reqs_before):]
    assert all("13000001" not in u and "13000002" not in u for u in new_photo_reqs)
    assert len(listing_requests) == 2 and "PG=2" in listing_requests[1]


def test_access_blocked_stops_immediately_and_reports_honestly(tmp_path, monkeypatch):
    class BlockedClient:
        def __init__(self, delay):
            self.calls = 0
        def close(self):
            pass
        def get(self, url, referer=None):
            self.calls += 1
            if url.endswith("/robots.txt"):
                return _Resp(b"User-agent: *\nDisallow: /private/\n", "text/html")
            raise crawler.AccessBlocked("Access blocked or rate limited (HTTP 429)")

    monkeypatch.setattr(crawler, "PublicClient", BlockedClient)
    ev = tq.YoloMenuQualityEvaluator(detector_fn=lambda im: [])
    status_file = tmp_path / "status.json"
    rc = crawler.run_crawler(tmp_path, target=5, delay=1, photo_kind="menu", quality_evaluator=ev, status_path=status_file)
    assert rc == 1
    mf = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    st = json.loads(status_file.read_text(encoding="utf-8"))
    assert mf["status"] == "blocked" and st["status"] == "blocked"
    assert "429" in mf["blocked_error"]


def test_parser_picks_largest_srcset_when_no_anchor_photo_href():
    parser = crawler.TabelogHTML("https://tabelog.com/en/tokyo/A1301/A130101/13000001/dtlmenu/photo/", "menu")
    parser.feed(
        '<img class="rstdtl-photo-list__img" '
        'srcset="https://tblg.k-img.com/restaurant/images/Rvw/1/320x320_a.jpg 320w, '
        'https://tblg.k-img.com/restaurant/images/Rvw/1/640x640_a.jpg 640w" '
        'src="https://tblg.k-img.com/restaurant/images/Rvw/1/150x150_a.jpg" alt="Menu">'
    )
    assert len(parser.photos) == 1
    assert "640x640_a.jpg" in parser.photos[0]["image_url"]
