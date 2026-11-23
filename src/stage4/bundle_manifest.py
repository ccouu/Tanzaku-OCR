"""Pinned Stage 4 experimental asset bundle manifest with strict SHA-256 verification."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

PINNED_STAGE4_ASSETS: dict[str, dict[str, str]] = {
    "ocr_checkpoint_raw": {
        "rel_path": "outputs/recognition/experiments/l2_selected2105_short_recovery_1000/downloads/short_recovery/step_1000.pt",
        "expected_sha256": "737cd2f82ebba5c85d95a3eb466040b5511de900db6e4ace3a7ee42de8b6c7c8",
        "state_key": "raw",
    },
    "vocab_2105": {
        "rel_path": "outputs/recognition/experiments/l2_selected2105_short_recovery_1000/payload/target_vocab_active.json",
        "expected_sha256": "9e73532f1e6a9435a169ddab937cf736d66e1e57ce52eced13169172bee1ad9e",
    },
    "detector_v5_obb": {
        "rel_path": "outputs/detection/training/banana130_finetune_v5/yolo11n_menu_obb_banana130_best.pt",
        "expected_sha256": "0cab3e1205c964621fa7f9d7e9647262895a9009260a174e3bb99174d8d5bf99",
    },
    "char_lm_v2": {
        "rel_path": "outputs/recognition/validation/char_lm_followup_20261119/candidates/char_lm_4g_hp1010_v2.json.gz",
        "expected_sha256": "58721de9d92d9daf165dfc64d3f979b1ff9da83d253a6d2e11a2a7c8f99fbefb",
    },
    "lexicon_index_v3": {
        "rel_path": "data/processed/lexicon/lexicon_index_v3_22112026.json.gz",
        "expected_sha256": "192059a006582c3034a638dea4d0b27bb925e37750c73bf0f58d8614802d0fc5",
    },
    "culinary_menu_db": {
        "rel_path": "data/processed/lexicon/culinary_menu.db",
        "expected_sha256": "ce8553abe75bce77bdb860da7d3773aabed975b8d74a5c1620ccca268bd39e59",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_pinned_stage4_bundle(root: Path = ROOT, *, strict: bool = True) -> dict[str, Any]:
    """Verify all 6 pinned Stage 4 assets exist and match their exact SHA-256 digests."""
    verified: dict[str, Any] = {}
    mismatches: list[str] = []
    for key, spec in PINNED_STAGE4_ASSETS.items():
        p = (root / spec["rel_path"]).resolve()
        if not p.is_file():
            mismatches.append(f"{key}: missing file {p}")
            verified[key] = {"path": str(p), "rel_path": spec["rel_path"], "exists": False, "sha256_ok": False}
            continue
        actual = sha256_file(p)
        ok = actual == spec["expected_sha256"]
        if not ok:
            mismatches.append(f"{key}: SHA mismatch ({actual} != {spec['expected_sha256']})")
        verified[key] = {
            "path": str(p),
            "rel_path": spec["rel_path"],
            "exists": True,
            "bytes": p.stat().st_size,
            "expected_sha256": spec["expected_sha256"],
            "actual_sha256": actual,
            "sha256_ok": ok,
        }
    if strict and mismatches:
        raise RuntimeError("Pinned Stage 4 asset bundle verification failed:\n" + "\n".join(mismatches))
    return {
        "bundle_id": "stage4_step1000_raw_2105_v5_lmv2_lexv3_20261123",
        "experimental_not_production_promoted": True,
        "all_verified": len(mismatches) == 0,
        "mismatches": mismatches,
        "assets": verified,
        "interfaces": {
            "detector": {"model": "YOLO11n-OBB V5", "imgsz": 640, "default_conf": 0.35, "iou": 0.70},
            "recognizer": {
                "arch": "h4_k41",
                "vocab_size": 2105,
                "output_classes": 2106,
                "precision": "fp32",
                "height": 64,
                "buckets": [128, 256, 384, 512],
                "overlap512_px": 100,
                "vertical_rotation": "once_only_ccw_90",
                "bn_buffers_frozen": 147,
            },
            "char_lm": {
                "schema_version": 2,
                "context_transform": "dig_unk_v2",
                "lm_vocab_size": 1779,
                "expanded_chars_unicode_fallback": 326,
                "default_alpha": 0.30,
                "beam_width": 5,
                "topk_t": 6,
                "r1_short_greedy_max_chars": 2,
            },
            "lexicon": {
                "index_entries": 12471,
                "db_base_rows": 47399,
                "curated_overrides": 35,
                "curated_phrase_plans": 230,
                "runtime_mode": "read_only",
            },
        },
    }
