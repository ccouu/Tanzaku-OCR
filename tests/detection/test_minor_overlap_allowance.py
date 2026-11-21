"""Unit tests for conservative minor core-overlap allowance (<10% area & perpendicular intrusion)."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.line_crops import (
    MINOR_OVERLAP_POLICY_VERSION,
    box,
    classify_boxes_core_overlaps,
    classify_core_overlap,
    process,
)


def _rect_pts(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _rot_pts(pts: list[list[float]], deg: float, cx: float = 500.0, cy: float = 500.0) -> list[list[float]]:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    out = []
    for x, y in pts:
        dx, dy = x - cx, y - cy
        out.append([cx + dx * c - dy * s, cy + dx * s + dy * c])
    return out


def test_no_overlap_and_padding_only_remain_clear():
    # Two horizontal lines separated by 4 px (padding would touch, cores do not)
    dets = [
        {"id": "0", "label": "dish_horizontal", "confidence": 0.9, "points": _rect_pts(100, 100, 400, 150)},
        {"id": "1", "label": "dish_horizontal", "confidence": 0.9, "points": _rect_pts(100, 154, 400, 204)},
    ]
    res = process(dets, 1000, 1000)
    assert len(res["conflicts"]) == 0
    for c in res["crops"]:
        assert c["overlap_severity"] == "clear"
        assert c["minor_overlap_allowed"] is False
        assert c["review_required"] is False
        assert c["legacy_review_required"] is False
        assert c["minor_overlap_policy_version"] == MINOR_OVERLAP_POLICY_VERSION


def test_strict_boundary_9_9pct_allowed_10_0pct_rejected():
    # Two identical horizontal boxes of height=100, width=300
    # 9.9 px vertical overlap -> perp_frac=0.099, area_frac=0.099 -> minor_allowed
    a = box(_rect_pts(100, 100, 400, 200), "dish_horizontal", 0.9, ["0"])
    b_ok = box(_rect_pts(100, 190.1, 400, 290.1), "dish_horizontal", 0.9, ["1"])
    cls_ok = classify_core_overlap(a, b_ok)
    assert cls_ok["severity"] == "minor_allowed"
    assert cls_ok["minor_allowed"] is True
    assert cls_ok["review_required"] is False
    assert cls_ok["area_frac_a"] < 0.10 and cls_ok["perp_frac_a"] < 0.10

    # Exactly 10.0 px vertical overlap -> perp_frac=0.10, area_frac=0.10 -> strict '<' rejects!
    b_rej = box(_rect_pts(100, 190.0, 400, 290.0), "dish_horizontal", 0.9, ["2"])
    cls_rej = classify_core_overlap(a, b_rej)
    assert cls_rej["severity"] == "major_review"
    assert cls_rej["minor_allowed"] is False
    assert cls_rej["review_required"] is True


def test_asymmetric_small_neighbor_rejected():
    # Large box (height=100, width=400) overlaps small box (height=30, width=100) by 5 px
    # Large box: perp_frac=5/100=0.05 (<0.10), area_frac=(5*100)/40000=0.0125 (<0.10)
    # Small box: perp_frac=5/30=0.1667 (>=0.10), area_frac=(5*100)/3000=0.1667 (>=0.10) -> REJECTED
    big = box(_rect_pts(100, 100, 500, 200), "dish_horizontal", 0.9, ["0"])
    small = box(_rect_pts(200, 195, 300, 225), "dish_horizontal", 0.9, ["1"])
    cls = classify_core_overlap(big, small)
    assert cls["area_frac_a"] < 0.10 and cls["perp_frac_a"] < 0.10
    assert cls["area_frac_b"] >= 0.10 and cls["perp_frac_b"] >= 0.10
    assert cls["minor_allowed"] is False
    assert cls["review_required"] is True


def test_vertical_reading_axes_minor_overlap():
    # Two vertical columns of width=100 (thickness along perpendicular x-axis), height=300
    # 8 px horizontal overlap -> perp_frac=0.08, area_frac=0.08 -> minor_allowed
    v1 = box(_rect_pts(100, 100, 200, 400), "dish_vertical", 0.9, ["0"])
    v2 = box(_rect_pts(192, 100, 292, 400), "dish_vertical", 0.9, ["1"])
    cls = classify_core_overlap(v1, v2)
    assert cls["severity"] == "minor_allowed"
    assert cls["minor_allowed"] is True
    assert abs(cls["perp_frac_a"] - 0.08) < 1e-6


def test_rotated_parallel_allowed_vs_crossing_rejected():
    # Two parallel boxes rotated 25 degrees, overlapping 6 px along their shared perpendicular axis
    p1 = _rot_pts(_rect_pts(300, 400, 700, 500), 25.0)
    p2 = _rot_pts(_rect_pts(300, 494, 700, 594), 25.0)
    a = box(p1, "dish_horizontal", 0.9, ["0"])
    b = box(p2, "dish_horizontal", 0.9, ["1"])
    cls_par = classify_core_overlap(a, b)
    assert cls_par["angle_deg"] < 1e-3
    assert cls_par["minor_allowed"] is True

    # Crossing horizontal + vertical box (90 deg angle > 8 deg) even with tiny area overlap -> rejected!
    h = box(_rect_pts(100, 100, 500, 200), "dish_horizontal", 0.9, ["0"])
    v = box(_rect_pts(492, 192, 592, 592), "dish_vertical", 0.9, ["1"])
    cls_cross = classify_core_overlap(h, v)
    assert cls_cross["angle_deg"] > 80.0
    assert cls_cross["minor_allowed"] is False
    assert cls_cross["review_required"] is True


def test_multiple_conflicts_one_major_vetoes_all():
    # Box B has minor overlap (6%) with A, and major overlap (20%) with C -> B must be major_review!
    a = box(_rect_pts(100, 100, 400, 200), "dish_horizontal", 0.9, ["0"])
    b = box(_rect_pts(100, 194, 400, 294), "dish_horizontal", 0.9, ["1"])
    c = box(_rect_pts(100, 274, 400, 374), "dish_horizontal", 0.9, ["2"])
    conflicts, per_crop = classify_boxes_core_overlaps([a, b, c])
    assert len(conflicts) == 2
    # A only has the minor conflict with B -> minor_allowed
    assert per_crop[0]["overlap_severity"] == "minor_allowed"
    assert per_crop[0]["review_required"] is False
    # B has minor(A) + major(C) -> vetoed -> major_review
    assert per_crop[1]["overlap_severity"] == "major_review"
    assert per_crop[1]["review_required"] is True
    assert per_crop[1]["minor_overlap_allowed"] is False
    # C has major(B) -> major_review
    assert per_crop[2]["overlap_severity"] == "major_review"
    assert per_crop[2]["review_required"] is True


def test_degenerate_nonfinite_geometry_fails_safe_to_major_review():
    a = box(_rect_pts(100, 100, 400, 200), "dish_horizontal", 0.9, ["0"])
    bad = dict(a)
    bad["core"] = np.array([[float("nan"), 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cls = classify_core_overlap(a, bad)
    assert cls["minor_allowed"] is False
    assert cls["review_required"] is True
    assert cls["severity"] == "major_review"


def test_process_preserves_exact_crop_polygons():
    # Verify adding minor overlap metadata never alters core or crop_polygon coordinates
    dets = [
        {"id": "0", "label": "dish_horizontal", "confidence": 0.9, "points": _rect_pts(100, 100, 400, 200)},
        {"id": "1", "label": "dish_horizontal", "confidence": 0.9, "points": _rect_pts(100, 194, 400, 294)},
    ]
    res = process(dets, 1000, 1000)
    assert len(res["crops"]) == 2
    assert res["crops"][0]["padding_scale"] == 0.0
    assert res["crops"][1]["padding_scale"] == 0.0
    assert res["crops"][0]["core"] == res["crops"][0]["crop_polygon"]
    assert res["crops"][0]["minor_overlap_allowed"] is True
    assert res["crops"][0]["review_required"] is False
    assert res["crops"][0]["legacy_review_required"] is True
