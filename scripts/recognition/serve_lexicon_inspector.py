"""Dedicated localhost inspection server for the Python lexicon-trie & full-menu pipeline.

Security & concurrency guarantees:
- Binds strictly to 127.0.0.1 (no public network exposure).
- Stateless per-request LookupConfig and DecodeConfig; ImmutableCharNGramLM +
  immutable LexiconIndex + read-only SQLite repository (thread-safe).
- Bounded single-worker async MenuJobManager for full-menu photo uploads
  (JPEG/PNG/WebP with EXIF transpose, pixel/byte limits, per-crop isolation).
- Strictly allowlisted sample IDs for real evaluation crops (no arbitrary paths).
- Safe DOM rendering only (ZERO `.innerHTML` dynamic interpolation).
- Review annotations stay client-side with JSON export/import; never modifies
  ground-truth labels or the SQLite database.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, UnidentifiedImageError  # noqa: E402

from src.recognition.char_lm_beam import ImmutableCharNGramLM, IncompatibleLMSchemaError  # noqa: E402
from src.recognition.current import load_current_recognizer  # noqa: E402
from src.recognition.checkpointing import sha256_file  # noqa: E402
from src.recognition.lexicon_lookup import (  # noqa: E402
    DEFAULT_CONFUSABLE_CATALOG,
    DEFAULT_V5_YOLO_WEIGHTS,
    MAX_MENU_UPLOAD_BYTES,
    EditPolicyConfig,
    LexiconIndex,
    LexiconPipeline,
    LookupConfig,
    MenuJobManager,
    ReadOnlyTranslationRepo,
    V5YoloDetector,
)

DEFAULT_WORK_ROOT = ROOT / "outputs/recognition/implementation/lexicon_trie_python_20261119"
DEFAULT_INDEX_PATH = DEFAULT_WORK_ROOT / "artifacts/lexicon_index_v1.json.gz"
DEFAULT_DB_PATH = ROOT / "data/processed/lexicon/culinary_menu.db"
DEFAULT_LM_PATH = ROOT / "checkpoints/l2_v06/char_lm_4g_hp1010.json.gz"
VAL_MANIFEST = ROOT / "outputs/recognition/experiments/h4_k41_l2_v06_p5pct_pairs_ep04/manifests/yolo_val_crops.jsonl"
VAL_IMAGES_DIR = ROOT / "outputs/recognition/experiments/h4_k41_l2_v06_p5pct_pairs_ep04/images"

MAX_CROP_BODY_BYTES = 8 * 1024 * 1024  # 8 MB for single crop upload
MAX_MENU_BODY_BYTES = 24 * 1024 * 1024  # 24 MB base64 JSON envelope (~16 MB binary photo)

_RE_MENU_STATUS = re.compile(r"^/api/menu/jobs/([0-9a-f]{8,32})/status$")
_RE_MENU_RESULT = re.compile(r"^/api/menu/jobs/([0-9a-f]{8,32})/result$")
_RE_MENU_CANCEL = re.compile(r"^/api/menu/jobs/([0-9a-f]{8,32})/cancel$")
_RE_MENU_RERUN = re.compile(r"^/api/menu/jobs/([0-9a-f]{8,32})/rerun$")
_RE_MENU_CROP = re.compile(r"^/api/menu/jobs/([0-9a-f]{8,32})/crops/([A-Za-z0-9_-]{1,32})$")

CANNED_TEXT_SAMPLES: tuple[dict[str, Any], ...] = (
    {
        "id": "canned_tokujo_rosu",
        "kind": "text",
        "label": "Cross-script lookalike: 特上口ース (口 U+53E3 → ロ U+30ED)",
        "text": "特上口ース",
        "note": "Real val10 OCR error: 特上 exact + 口ース→ロース (visual_confusable_suggestion, contingent DB translation).",
    },
    {
        "id": "canned_pork_loin_price",
        "kind": "text",
        "label": "Cross-script lookalike + price: 豚ロ一ス定食 1,200円",
        "text": "豚ロ一ス定食 1,200円",
        "note": "Expected: suggest 豚ロース (一 U+4E00 → ー U+30FC, visual_confusable_suggestion) + exact 定食, keep 1,200円 protected.",
    },
    {
        "id": "canned_kobi_senbei",
        "kind": "text",
        "label": "Cross-script lookalike: 工ビ煎餅 (工 U+5DE5 → エ U+30A8)",
        "text": "工ビ煎餅",
        "note": "Note: standalone エビ is not in current lexicon index; エビ煎餅 is in index.",
    },
    {
        "id": "canned_nirareba",
        "kind": "text",
        "label": "Cross-script lookalike: 二ラレバ (二 U+4E8C → ニ U+30CB)",
        "text": "二ラレバ",
        "note": "Note: standalone ニラ is not in current lexicon index; ニラレバ is in index.",
    },
    {
        "id": "canned_negative_kanji_words",
        "kind": "text",
        "label": "Negative kanji/numeral cases: 辛口 入口 一口 力うどん 一杯 二人前",
        "text": "辛口 入口 一口 力うどん 一杯 二人前",
        "note": "Valid kanji words 辛口/一杯 stay exact; 入口/一口/力うどん/二人前 are not blanket-rewritten to katakana.",
    },
    {
        "id": "canned_isolated_single_chars",
        "kind": "text",
        "label": "Negative isolated 1-char lookalikes: 口 ロ 力 カ",
        "text": "口 ロ 力 カ",
        "note": "1-char policy is exact-only; isolated lookalikes are never frequency-converted.",
    },
    {
        "id": "canned_ordinary_ocr_confusion",
        "kind": "text",
        "label": "Ordinary OCR confusion (NOT visual-only): エビビラフ",
        "text": "エビビラフ",
        "note": "Dakuten/handakuten ビ→ピ is ordinary_ocr_confusion (ordinary_fuzzy_suggestion, never visual_confusable_suggestion).",
    },
    {
        "id": "canned_short_fish",
        "kind": "text",
        "label": "Short ambiguous fish names (exact preserved)",
        "text": "あじ いか たこ さば 580円",
        "note": "Short 2-char fish names must stay exact and not be rewritten by frequency.",
    },
    {
        "id": "canned_fullwidth_norm",
        "kind": "text",
        "label": "Literal vs canonical alias (full-width digits & parens)",
        "text": "生ビール（中） ２杯 １,１００円",
        "note": "Canonicalizes only full-width digits/parens; protects 2杯 and 1,100円; 中 is not auto-numeric.",
    },
    {
        "id": "canned_unknown_survives",
        "kind": "text",
        "label": "Unknown house special survives without forced match",
        "text": "店主特製ゾルバ風スパイス鍋 1,480円(税込)",
        "note": "Unknown OOV house-special words must survive as unknown segments.",
    },
)


def load_allowlisted_real_samples(max_samples: int = 8) -> dict[str, dict[str, Any]]:
    """Load a small allowlist of existing val10 crop samples (including real OCR failures)."""
    samples: dict[str, dict[str, Any]] = {}
    for item in CANNED_TEXT_SAMPLES:
        samples[item["id"]] = dict(item)

    if not VAL_MANIFEST.is_file() or not VAL_IMAGES_DIR.is_dir():
        return samples

    target_truths = {"特上ロース", "シマチョウ", "エビピラフ", "水餃子", "1880円", "特上カルビ"}
    picked = 0
    try:
        for line in VAL_MANIFEST.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            truth = str(row.get("text") or row.get("original_text") or "")
            img_rel = str(row.get("image") or row.get("ctc_crop") or "")
            if not img_rel:
                continue
            img_path = (VAL_IMAGES_DIR / img_rel).resolve()
            if not img_path.is_relative_to(VAL_IMAGES_DIR.resolve()) or not img_path.is_file():
                continue
            if truth in target_truths or picked < 2:
                sid = f"val10_{picked:02d}_{Path(img_rel).stem[:24]}"
                samples[sid] = {
                    "id": sid,
                    "kind": "crop",
                    "label": f"Real val10 crop ({truth})",
                    "truth_reference": truth,
                    "orientation": int(row.get("orientation", 0)),
                    "image_path": str(img_path),
                    "note": "Existing val10 evaluation crop (read-only; ground truth shown for inspection only).",
                }
                picked += 1
                if picked >= max_samples:
                    break
    except Exception:
        pass
    return samples


def parse_lookup_config(payload: dict[str, Any]) -> LookupConfig:
    """Build an immutable per-request LookupConfig from untrusted JSON payload."""
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")

    exact_only = bool(payload.get("exact_only", False))
    max_e6 = int(payload.get("max_edits_6plus", 2))
    if max_e6 not in (0, 1, 2):
        raise ValueError("max_edits_6plus must be 0, 1, or 2")
    max_e2 = int(payload.get("len2_max_edits", 1))
    if max_e2 not in (0, 1):
        raise ValueError("len2_max_edits must be 0 or 1")
    max_e35 = int(payload.get("len3_5_max_edits", 1))
    if max_e35 not in (0, 1):
        raise ValueError("len3_5_max_edits must be 0 or 1")

    if exact_only:
        policy = EditPolicyConfig(len1_max_edits=0, len2_max_edits=0, len3_5_max_edits=0, len6plus_max_edits=0)
    else:
        policy = EditPolicyConfig(
            len1_max_edits=0,
            len2_max_edits=max_e2,
            len3_5_max_edits=max_e35,
            len6plus_max_edits=max_e6,
            conservative_boundary_enforcement=True,
        )

    return LookupConfig(
        edit_policy=policy,
        use_confusion_weights=bool(payload.get("use_confusion_weights", True)),
        include_context_phrases=bool(payload.get("include_context_phrases", True)),
        include_expanded_culinary=bool(payload.get("include_expanded_culinary", True)),
        enable_visual_rescoring=bool(payload.get("enable_visual_rescoring", True)),
        protect_particles=bool(payload.get("protect_particles", True)),
        auto_correct=False,
    )


def _decode_base64_field(b64_raw: Any, max_bytes: int) -> bytes:
    if not isinstance(b64_raw, str) or not b64_raw:
        raise ValueError("Field 'image_base64' must be a non-empty string")
    if "," in b64_raw and b64_raw.startswith("data:"):
        b64_data = b64_raw.split(",", 1)[1]
    else:
        b64_data = b64_raw
    raw = base64.b64decode(b64_data, validate=True)
    if len(raw) > max_bytes:
        raise OverflowError(f"Decoded image exceeds {max_bytes // (1024 * 1024)} MB limit")
    return raw


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Menu OCR &amp; Lexicon-Trie Inspection Tool</title>
<style>
  :root {
    --bg: #0f172a; --panel: #1e293b; --border: #334155; --text: #e2e8f0;
    --muted: #94a3b8; --accent: #38bdf8; --ok: #22c55e; --warn: #f59e0b;
    --prot: #a855f7; --unk: #64748b; --err: #ef4444;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }
  header { padding: 12px 20px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; background: #0b1220; }
  header h1 { margin: 0; font-size: 18px; }
  .tabs { display: flex; gap: 8px; padding: 10px 20px; background: #111827; border-bottom: 1px solid var(--border); }
  .tab-btn { padding: 8px 16px; border-radius: 8px; border: 1px solid var(--border); background: transparent; color: var(--text); font-weight: 600; cursor: pointer; }
  .tab-btn.active { background: #0369a1; border-color: var(--accent); color: #fff; }
  .badge { display: inline-block; padding: 3px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; background: #1e3a5f; color: var(--accent); }
  .badge.warn { background: #451a03; color: #fbbf24; }
  .badge.ok { background: #052e16; color: #4ade80; }
  .badge.err { background: #450a0a; color: #fca5a5; }
  main { max-width: 1440px; margin: 0 auto; padding: 16px; display: grid; grid-template-columns: 390px 1fr; gap: 16px; }
  @media (max-width: 1024px) { main { grid-template-columns: 1fr; } }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px; margin-bottom: 14px; }
  .card h2 { margin: 0 0 10px 0; font-size: 15px; color: #f8fafc; }
  label { display: block; font-size: 12px; color: var(--muted); margin: 8px 0 4px; }
  input[type="text"], input[type="number"], textarea, select { width: 100%; padding: 8px 10px; border-radius: 6px; border: 1px solid var(--border); background: #0f172a; color: var(--text); font-size: 14px; }
  textarea { min-height: 64px; resize: vertical; }
  button { padding: 8px 14px; border-radius: 6px; border: 1px solid var(--accent); background: #0369a1; color: #fff; font-weight: 600; cursor: pointer; font-size: 13px; }
  button:hover { background: #0284c7; }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  button.secondary { background: transparent; border-color: var(--border); color: var(--text); }
  button.danger { background: #7f1d1d; border-color: var(--err); }
  .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .check { display: flex; align-items: center; gap: 6px; font-size: 13px; color: var(--text); margin: 4px 0; }
  .check input { width: auto; }
  .seg-strip { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; }
  .seg { padding: 6px 10px; border-radius: 8px; border: 1px solid var(--border); background: #0f172a; }
  .seg.exact { border-color: var(--ok); }
  .seg.fuzzy_suggestion { border-color: var(--warn); }
  .seg.protected { border-color: var(--prot); }
  .seg.unknown { border-color: var(--unk); border-style: dashed; }
  .seg .top { font-size: 15px; font-weight: 600; }
  .seg .sub { font-size: 11px; color: var(--muted); margin-top: 2px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
  th { color: var(--muted); font-weight: 600; }
  .op-sub { background: #78350f; color: #fde68a; padding: 1px 4px; border-radius: 4px; }
  .op-ins { background: #064e3b; color: #a7f3d0; padding: 1px 4px; border-radius: 4px; }
  .op-del { background: #7f1d1d; color: #fecaca; padding: 1px 4px; border-radius: 4px; text-decoration: line-through; }
  .mono { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; }
  .preview-img { max-width: 100%; max-height: 140px; border: 1px solid var(--border); border-radius: 6px; background: #000; display: none; margin-bottom: 8px; }
  .error-box { background: #450a0a; border: 1px solid var(--err); color: #fecaca; padding: 10px; border-radius: 6px; display: none; margin-bottom: 10px; }
  .hidden { display: none !important; }
  /* Full Menu Interactive Canvas & Crop List */
  .menu-stage-bar { height: 8px; background: #0f172a; border-radius: 999px; overflow: hidden; margin: 8px 0; border: 1px solid var(--border); }
  .menu-stage-fill { height: 100%; width: 0%; background: var(--accent); transition: width 0.2s ease; }
  .menu-canvas-wrap { overflow: auto; max-height: 620px; border: 1px solid var(--border); border-radius: 8px; background: #020617; padding: 8px; }
  .menu-svg-box { position: relative; display: inline-block; transform-origin: top left; }
  .menu-svg-box img { display: block; max-width: 100%; height: auto; user-select: none; }
  .menu-svg-box svg { position: absolute; left: 0; top: 0; width: 100%; height: 100%; }
  .poly-crop { fill: rgba(52, 211, 153, 0.14); stroke: #34d399; stroke-width: 2.5; cursor: pointer; vector-effect: non-scaling-stroke; }
  .poly-crop.vert { fill: rgba(56, 189, 248, 0.14); stroke: #38bdf8; }
  .poly-crop.err { fill: rgba(239, 68, 68, 0.18); stroke: #ef4444; }
  .poly-crop:hover { fill: rgba(251, 191, 36, 0.28); stroke: #fbbf24; stroke-width: 3.5; }
  .poly-crop.selected { fill: rgba(251, 191, 36, 0.32); stroke: #f59e0b; stroke-width: 4; }
  .poly-raw { fill: none; stroke: #94a3b8; stroke-width: 1.5; stroke-dasharray: 5 4; pointer-events: none; vector-effect: non-scaling-stroke; }
  .crop-list { max-height: 320px; overflow-y: auto; border: 1px solid var(--border); border-radius: 6px; }
  .crop-row { display: grid; grid-template-columns: 36px 76px 1fr; gap: 8px; align-items: center; padding: 6px 8px; border-bottom: 1px solid var(--border); cursor: pointer; }
  .crop-row:hover { background: #0f172a; }
  .crop-row.selected { background: #1e3a5f; border-left: 3px solid #f59e0b; }
  .crop-thumb { max-width: 72px; max-height: 36px; border: 1px solid var(--border); border-radius: 4px; background: #000; object-fit: contain; }
</style>
</head>
<body>
<header>
  <div>
    <h1>Menu OCR &amp; Lexicon-Trie Inspection Tool</h1>
    <span id="activeModelLabel" class="mono" style="color:var(--muted)">Local V5 YOLO11n-OBB → recognizer → Char-LM R1 → Lexicon Trie → Read-Only DB</span>
  </div>
  <div class="row">
    <span class="badge warn">Auto-Correction OFF</span>
    <span class="badge" id="statusBadge">Loading...</span>
  </div>
</header>
<div class="tabs">
  <button class="tab-btn active" id="tabFullMenu">Full Menu Photo Mode</button>
  <button class="tab-btn" id="tabSingleCrop">Single Crop / Text Mode</button>
</div>
<main>
  <section>
    <!-- FULL MENU CONTROLS -->
    <div class="card" id="panelFullMenu">
      <h2>1. Full Menu Photo Upload</h2>
      <label>Menu Photo (JPEG, PNG, WebP · EXIF auto-transposed · max 16 MB)</label>
      <input type="file" id="menuFileInput" accept="image/jpeg,image/png,image/webp" />
      <label>V5 YOLO Confidence Threshold (default 0.35)</label>
      <input type="number" id="menuConfInput" value="0.35" min="0.05" max="0.95" step="0.05" />
      <div class="row" style="margin-top:10px;">
        <button id="runMenuBtn">Run Full Menu Pipeline</button>
        <button class="secondary" id="rerunMenuBtn" disabled>Re-run with Config</button>
        <button class="danger" id="cancelMenuBtn" disabled>Cancel</button>
      </div>
      <div class="menu-stage-bar"><div class="menu-stage-fill" id="menuProgressFill"></div></div>
      <div id="menuJobStatus" class="mono" style="color:var(--muted)">Idle. Upload a full menu photo to run V5 YOLO → L2_v06 → Lexicon Trie.</div>
      <div id="menuProvenance" class="mono" style="color:var(--muted);margin-top:6px;"></div>

      <h2 style="margin-top:14px;">Detected Crops (Click Polygon or Row)</h2>
      <div id="cropList" class="crop-list"></div>
    </div>

    <!-- SINGLE CROP / TEXT CONTROLS -->
    <div class="card hidden" id="panelSingleCrop">
      <h2>1. Single Crop / Text Input</h2>
      <label>Canned &amp; Real val10 Samples</label>
      <div class="row">
        <select id="sampleSelect"></select>
        <button id="runSampleBtn">Run Sample</button>
      </div>
      <div id="sampleNote" style="font-size:12px;color:var(--muted);margin-top:6px;"></div>

      <label>Pure Text Query</label>
      <input type="text" id="textInput" value="特上口ース" />
      <div class="row" style="margin-top:8px;">
        <button id="runTextBtn">Lookup Text</button>
      </div>

      <label>Local Single Crop Upload (max 8 MB)</label>
      <input type="file" id="fileInput" accept="image/jpeg,image/png,image/webp" />
      <div class="row" style="margin-top:8px;">
        <select id="orientationSelect" style="width:auto;">
          <option value="0">Horizontal (0)</option>
          <option value="1">Vertical (1)</option>
        </select>
        <button id="runUploadBtn">Infer &amp; Lookup Crop</button>
      </div>
    </div>

    <!-- SHARED PER-REQUEST CONFIG -->
    <div class="card">
      <h2>2. Per-Request Config</h2>
      <label>Decoder Mode</label>
      <select id="decoderSelect">
        <option value="greedy">Canonical Greedy (default)</option>
        <option value="beam4g_dn_r1">Reviewed Char-LM R1 (HP1010, &lt;=2 greedy)</option>
        <option value="plain_beam">Plain CTC Beam (alpha=0)</option>
      </select>
      <label class="check"><input type="checkbox" id="chkExpanded" checked /> Include cleaned JMdict expansion (10,539)</label>
      <label class="check"><input type="checkbox" id="chkContext" checked /> Include context phrases (1,327)</label>
      <label class="check"><input type="checkbox" id="chkConfusion" checked /> Use visual-confusion weights (0.45 lookalike / 0.65 ordinary)</label>
      <label class="check"><input type="checkbox" id="chkExactOnly" /> Exact match only (0 edits)</label>
      <label>Max edits for &gt;=6 chars</label>
      <select id="maxE6Select">
        <option value="2" selected>2 edits (suggestion-only default)</option>
        <option value="1">1 edit</option>
        <option value="0">0 edits</option>
      </select>
    </div>

    <!-- SHARED REVIEW NOTES -->
    <div class="card">
      <h2>3. Review Notes (Local Browser Only)</h2>
      <label>Verdict for Current Crop / Query</label>
      <select id="reviewVerdict">
        <option value="accept">Accept</option>
        <option value="reject">Reject</option>
        <option value="needs_edit">Needs Edit</option>
      </select>
      <label>Comment</label>
      <textarea id="reviewComment" placeholder="Notes on detection, OCR, segmentation, suggestion, or translation..."></textarea>
      <div class="row" style="margin-top:8px;">
        <button id="saveReviewBtn">Save Review</button>
        <button class="secondary" id="exportReviewsBtn">Export JSON</button>
        <label class="secondary" style="margin:0;cursor:pointer;">
          <input type="file" id="importReviewsInput" accept="application/json" style="display:none;" />
          <span style="padding:7px 12px;border:1px solid var(--border);border-radius:6px;display:inline-block;">Import JSON</span>
        </label>
      </div>
      <div id="reviewCount" class="mono" style="color:var(--muted);margin-top:8px;">0 reviews saved in browser</div>
    </div>
  </section>

  <section>
    <div id="errorBox" class="error-box"></div>

    <!-- FULL MENU INTERACTIVE PHOTO OVERLAY -->
    <div class="card" id="menuOverlayCard">
      <div class="row" style="justify-content:space-between;margin-bottom:8px;">
        <h2 style="margin:0;">Full Menu Photo &amp; Selectable Crop Overlays</h2>
        <div class="row">
          <label class="check" style="margin:0;"><input type="checkbox" id="chkShowRawBoxes" /> Show raw detector boxes</label>
          <button class="secondary" id="zoomFitBtn">Fit</button>
          <button class="secondary" id="zoom100Btn">100%</button>
          <button class="secondary" id="zoom150Btn">150%</button>
          <button class="secondary" id="zoom200Btn">200%</button>
        </div>
      </div>
      <div id="menuTimings" class="mono" style="color:var(--muted);margin-bottom:8px;">No full-menu photo processed yet.</div>
      <div class="menu-canvas-wrap" id="menuCanvasWrap">
        <div class="menu-svg-box" id="menuSvgBox">
          <img id="menuPhotoImg" alt="Uploaded menu photo" style="display:none;" />
          <svg id="menuOverlaySvg" xmlns="http://www.w3.org/2000/svg"></svg>
        </div>
      </div>
    </div>

    <!-- SHARED CROP / QUERY DETAIL PANEL -->
    <div class="card">
      <h2>Selected Crop / OCR &amp; Decoder Summary</h2>
      <div class="row" style="align-items:flex-start;">
        <div>
          <div class="mono" style="color:var(--muted);margin-bottom:4px;">Upright Rectified Crop</div>
          <img id="cropPreview" class="preview-img" alt="crop preview" />
        </div>
        <div>
          <div class="mono" style="color:var(--muted);margin-bottom:4px;">Native H=64 OCR Strip (No 512 Letterbox)</div>
          <img id="ocrStripPreview" class="preview-img" alt="ocr strip preview" />
        </div>
      </div>
      <table>
        <tbody>
          <tr><th style="width:190px">Crop Geometry &amp; Merge</th><td id="outGeom" class="mono">-</td></tr>
          <tr><th>Raw OCR</th><td id="outRaw" class="mono">-</td></tr>
          <tr><th>Canonical OCR</th><td id="outCanon" class="mono">-</td></tr>
          <tr><th>LM R1 Alternative</th><td id="outLmAlt" class="mono">-</td></tr>
          <tr><th>Selected for Lookup</th><td id="outSelected" class="mono">-</td></tr>
          <tr><th>Suggested Line (Text DP)</th><td id="outSuggested" class="mono">-</td></tr>
          <tr><th>Visual Reranked Line (CTC+Text)</th><td id="outVisualReranked" class="mono">-</td></tr>
          <tr><th>Reference GT (val10 only)</th><td id="outTruth" class="mono">-</td></tr>
          <tr><th>Protected Spans</th><td id="outProtected" class="mono">-</td></tr>
          <tr><th>Warnings / Diagnostics</th><td id="outDiag" class="mono">-</td></tr>
        </tbody>
      </table>
    </div>

    <div class="card">
      <h2>Nonoverlapping Segmentation</h2>
      <div id="segStrip" class="seg-strip"></div>
    </div>

    <div class="card">
      <h2>Exact Matches &amp; Translations (Literal Canonical Exact Only)</h2>
      <table id="exactTable">
        <thead><tr><th>Span</th><th>Term</th><th>Tier</th><th>Status</th><th>English / Reading</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>

    <div class="card">
      <h2>Suggested Matches &amp; Translations (Contingent on Accepting Suggestion)</h2>
      <div class="mono" style="color:var(--muted);margin-bottom:8px;">
        Note: Visual-confusable suggestions are bounded OCR hypotheses, NOT Unicode normalization or proven character equivalence.
        Translations shown below are read-only DB lookups contingent on accepting the suggestion (never inserted into Exact Matches).
        Whole-line CTC status reflects full-line sequence log-probability, NOT a per-character posterior.
      </div>
      <table id="fuzzyTable">
        <thead>
          <tr>
            <th>Original → Proposed</th>
            <th>Changed Chars (Code Points)</th>
            <th>Match Type &amp; Sources</th>
            <th>Edits (Hard / Heuristic)</th>
            <th>Whole-Line CTC Evidence</th>
            <th>Contingent DB Translation</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>

    <div class="card">
      <h2>Visual CTC + Text Whole-Line Ranking (Original Folded Logits)</h2>
      <table id="rescoreTable">
        <thead><tr><th>Rank</th><th>Candidate Line</th><th>Scorable</th><th>Text Score</th><th>CTC Delta</th><th>Combined</th><th>Reason / OOV</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </section>
</main>
<script>
let samplesList = [];
let lastResult = null;
let lastCropMeta = null;
let reviews = [];
let activeMenuJobId = null;
let activeMenuOverview = null;
let selectedCropId = null;
let pollTimer = null;
let currentZoom = 'fit';

function mkEl(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined && text !== null) el.textContent = String(text);
  return el;
}

function mkSvg(tag, attrs) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, String(v));
  return el;
}

function getConfigPayload() {
  return {
    decoder_mode: document.getElementById('decoderSelect').value,
    include_expanded_culinary: document.getElementById('chkExpanded').checked,
    include_context_phrases: document.getElementById('chkContext').checked,
    use_confusion_weights: document.getElementById('chkConfusion').checked,
    exact_only: document.getElementById('chkExactOnly').checked,
    max_edits_6plus: parseInt(document.getElementById('maxE6Select').value, 10),
  };
}

function showError(msg) {
  const el = document.getElementById('errorBox');
  if (!msg) { el.style.display = 'none'; el.textContent = ''; return; }
  el.style.display = 'block';
  el.textContent = msg;
}

function switchMode(mode) {
  const isMenu = mode === 'menu';
  document.getElementById('tabFullMenu').classList.toggle('active', isMenu);
  document.getElementById('tabSingleCrop').classList.toggle('active', !isMenu);
  document.getElementById('panelFullMenu').classList.toggle('hidden', !isMenu);
  document.getElementById('menuOverlayCard').classList.toggle('hidden', !isMenu);
  document.getElementById('panelSingleCrop').classList.toggle('hidden', isMenu);
}

document.getElementById('tabFullMenu').addEventListener('click', () => switchMode('menu'));
document.getElementById('tabSingleCrop').addEventListener('click', () => switchMode('single'));

function renderEditOpsDom(ops) {
  const wrap = document.createElement('span');
  for (const o of (ops || [])) {
    if (o.op === 'match') {
      wrap.appendChild(mkEl('span', '', o.cand_char));
    } else if (o.op === 'substitute') {
      const label = `${o.query_char}(${o.query_codepoint || ''})→${o.cand_char}(${o.cand_codepoint || ''})`;
      const sp = mkEl('span', 'op-sub', label);
      sp.title = `cost=${o.cost} cat=${o.confusion_category || 'none'}`;
      wrap.appendChild(sp);
    } else if (o.op === 'insert') {
      wrap.appendChild(mkEl('span', 'op-ins', `+${o.cand_char}(${o.cand_codepoint || ''})`));
    } else if (o.op === 'delete') {
      wrap.appendChild(mkEl('span', 'op-del', `-${o.query_char}(${o.query_codepoint || ''})`));
    }
    wrap.appendChild(document.createTextNode(' '));
  }
  return wrap;
}

function wholeLineCtcSummary(r) {
  const vrs = r.visual_rescores || [];
  if (!vrs.length) return 'unavailable (text-only; no logits)';
  if (vrs.every(v => !v.scorable)) return `unavailable (${vrs[0].reason || 'unscorable'})`;
  const top = vrs[0];
  if (top.is_unchanged_ocr) {
    return `whole-line CTC kept base OCR at rank #1 (combined=${top.combined_score})`;
  }
  return `whole-line CTC ranked "${top.candidate_line}" #1 (delta=${top.delta_log_prob}, combined=${top.combined_score})`;
}

function renderResult(data) {
  showError('');
  lastResult = data;
  lastCropMeta = data.geometry || null;

  const imgEl = document.getElementById('cropPreview');
  if (data.crop_data_url) { imgEl.src = data.crop_data_url; imgEl.style.display = 'block'; }
  else { imgEl.style.display = 'none'; }

  const stripImg = document.getElementById('ocrStripPreview');
  if (data.ocr_strip_data_url) { stripImg.src = data.ocr_strip_data_url; stripImg.style.display = 'block'; }
  else { stripImg.style.display = 'none'; }

  if (data.geometry) {
    const g = data.geometry;
    document.getElementById('outGeom').textContent =
      `${g.crop_id} (#${g.crop_index}) | label=${g.label} conf=${g.confidence} | ori=${g.orientation_label} (flag=${g.orientation_flag}) | sources=${JSON.stringify(g.source_ids)} | pad_scale=${g.padding_scale} | merge_decisions=${(g.merge_decisions||[]).length} | strip=${g.ocr_strip_size[0]}x${g.ocr_strip_size[1]}`;
  } else {
    document.getElementById('outGeom').textContent = '(single-crop / text mode)';
  }

  const r = data.result;
  if (!r) {
    document.getElementById('outRaw').textContent = data.error ? `[ERROR] ${data.error}` : '(no result)';
    document.getElementById('outCanon').textContent = '-';
    document.getElementById('outLmAlt').textContent = '-';
    document.getElementById('outSelected').textContent = '-';
    document.getElementById('outSuggested').textContent = '-';
    document.getElementById('outVisualReranked').textContent = '-';
    document.getElementById('outProtected').textContent = '-';
    document.getElementById('outDiag').textContent = data.error || '-';
    document.getElementById('segStrip').replaceChildren();
    document.querySelector('#exactTable tbody').replaceChildren();
    document.querySelector('#fuzzyTable tbody').replaceChildren();
    document.querySelector('#rescoreTable tbody').replaceChildren();
    return;
  }

  document.getElementById('outRaw').textContent = r.raw_ocr || '(empty)';
  document.getElementById('outCanon').textContent = r.canonical_ocr || '(empty)';
  document.getElementById('outLmAlt').textContent = r.lm_alternative ?? '(not run / unavailable)';
  document.getElementById('outSelected').textContent = `${r.selected_text_for_lookup || '(empty)'}  [${r.decoder_mode}]`;
  document.getElementById('outSuggested').textContent = `${r.suggested_line || '(empty)'}  (auto_corrected=${r.auto_corrected})`;
  document.getElementById('outVisualReranked').textContent = r.visual_reranked_line ? `${r.visual_reranked_line}  (auto_corrected=${r.auto_corrected})` : '(unavailable / text-only)';
  document.getElementById('outTruth').textContent = data.truth_reference ?? '(none)';
  document.getElementById('outProtected').textContent = r.protected_spans.length
    ? r.protected_spans.map(p => `[${p.start}:${p.end}] ${p.text} (${p.kind})`).join(', ')
    : '(none)';
  document.getElementById('outDiag').textContent =
    `elapsed=${r.diagnostics.elapsed_ms}ms | visits=${r.diagnostics.trie_node_visits} | truncated=${r.diagnostics.truncated} | warnings=${JSON.stringify(r.warnings)}`;

  const strip = document.getElementById('segStrip');
  strip.replaceChildren();
  for (const seg of r.segments) {
    const div = mkEl('div', `seg ${seg.kind}`);
    const c = seg.selected_candidate;
    const label = c ? `${seg.text} → ${c.term}` : seg.text;
    const sub = c
      ? `${c.match_type || seg.kind} · [${seg.start}:${seg.end}] · ${c.source_tier}`
      : `${seg.kind} · [${seg.start}:${seg.end}]`;
    div.appendChild(mkEl('div', 'top', label));
    div.appendChild(mkEl('div', 'sub', sub));
    strip.appendChild(div);
  }

  const exactTb = document.querySelector('#exactTable tbody');
  exactTb.replaceChildren();
  for (const ex of r.exact_translations) {
    const tr = document.createElement('tr');
    const en = ex.translation && ex.translation.status === 'found'
      ? `${ex.translation.name_en} (${ex.translation.reading_hiragana})`
      : `[${ex.translation_status}]`;
    tr.appendChild(mkEl('td', 'mono', `[${ex.start}:${ex.end}]`));
    tr.appendChild(mkEl('td', '', ex.term));
    tr.appendChild(mkEl('td', 'mono', ex.source_tier));
    tr.appendChild(mkEl('td', 'mono', ex.translation_status));
    tr.appendChild(mkEl('td', '', en));
    exactTb.appendChild(tr);
  }

  const fuzzyTb = document.querySelector('#fuzzyTable tbody');
  fuzzyTb.replaceChildren();
  const ctcNote = wholeLineCtcSummary(r);
  for (const fz of r.fuzzy_suggestions) {
    const tr = document.createElement('tr');
    const tdSpan = mkEl('td', 'mono');
    tdSpan.appendChild(mkEl('span', '', `[${fz.start}:${fz.end}] ${fz.query_span} → `));
    tdSpan.appendChild(mkEl('b', '', fz.term));
    tr.appendChild(tdSpan);

    const tdOps = mkEl('td', 'mono');
    tdOps.appendChild(renderEditOpsDom(fz.edit_ops));
    tr.appendChild(tdOps);

    const confSrcs = [...new Set((fz.edit_ops || []).flatMap(o => o.confusion_sources || []))];
    const tdMeta = mkEl('td', 'mono');
    const mtBadge = mkEl('span', fz.match_type === 'visual_confusable_suggestion' ? 'badge' : 'badge warn', fz.match_type || 'ordinary_fuzzy_suggestion');
    tdMeta.appendChild(mtBadge);
    if ((fz.suppression_reasons || []).length) {
      tdMeta.appendChild(document.createElement('br'));
      tdMeta.appendChild(mkEl('span', 'badge warn', 'Suppressed: probable particle/context protection'));
      tdMeta.appendChild(mkEl('div', '', fz.suppression_reasons.join('; ')));
    }
    tdMeta.appendChild(document.createElement('br'));
    tdMeta.appendChild(mkEl('span', '', `dict=${fz.source_tier}`));
    if (confSrcs.length) {
      tdMeta.appendChild(document.createElement('br'));
      tdMeta.appendChild(mkEl('span', '', `conf=${confSrcs.join(',')}`));
    }
    tr.appendChild(tdMeta);

    const comps = Object.entries(fz.score_components || {}).map(([k,v]) => `${k}=${v}`).join(', ');
    const tdEdits = mkEl('td', 'mono', `${fz.hard_edits} / ${fz.weighted_edit_cost} (total=${fz.total_score})`);
    tdEdits.title = comps;
    tr.appendChild(tdEdits);

    tr.appendChild(mkEl('td', 'mono', (fz.suppression_reasons || []).length
      ? 'Inspection only: excluded from segmentation and visual ranking' : ctcNote));

    const tdTrans = mkEl('td', '');
    if (fz.translation && fz.translation.status === 'found') {
      tdTrans.appendChild(mkEl('div', '', `${fz.translation.name_en} (${fz.translation.reading_hiragana})`));
      tdTrans.appendChild(mkEl('div', 'mono', `contingent on accepting "${fz.term}" · db_id=${fz.db_id}`));
    } else {
      tdTrans.appendChild(mkEl('span', 'mono', `[${fz.translation_status || 'missing'}]`));
    }
    tr.appendChild(tdTrans);
    fuzzyTb.appendChild(tr);
  }

  const resTb = document.querySelector('#rescoreTable tbody');
  resTb.replaceChildren();
  for (const vr of r.visual_rescores || []) {
    const tr = document.createElement('tr');
    const tag = vr.is_unchanged_ocr ? ' [base OCR]' : '';
    tr.appendChild(mkEl('td', 'mono', vr.rank ?? '-'));
    tr.appendChild(mkEl('td', 'mono', `${vr.candidate_line}${tag}`));
    tr.appendChild(mkEl('td', 'mono', String(vr.scorable)));
    tr.appendChild(mkEl('td', 'mono', vr.text_score ?? '-'));
    tr.appendChild(mkEl('td', 'mono', vr.delta_log_prob ?? '-'));
    const tdComb = mkEl('td', 'mono');
    tdComb.appendChild(mkEl('b', '', vr.combined_score ?? '-'));
    tr.appendChild(tdComb);
    tr.appendChild(mkEl('td', 'mono', `${vr.reason} ${(vr.unsupported_chars || []).join(',')}`.trim()));
    resTb.appendChild(tr);
  }
}

async function postJson(url, body) {
  const resp = await fetch(url, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
  return data;
}

function applyZoom() {
  const box = document.getElementById('menuSvgBox');
  const img = document.getElementById('menuPhotoImg');
  if (!activeMenuOverview) return;
  if (currentZoom === 'fit') {
    img.style.width = '100%';
    box.style.width = '100%';
  } else {
    const scale = parseFloat(currentZoom);
    const px = Math.round(activeMenuOverview.image.width * scale);
    img.style.width = `${px}px`;
    box.style.width = `${px}px`;
  }
}

document.getElementById('zoomFitBtn').addEventListener('click', () => { currentZoom = 'fit'; applyZoom(); });
document.getElementById('zoom100Btn').addEventListener('click', () => { currentZoom = '1.0'; applyZoom(); });
document.getElementById('zoom150Btn').addEventListener('click', () => { currentZoom = '1.5'; applyZoom(); });
document.getElementById('zoom200Btn').addEventListener('click', () => { currentZoom = '2.0'; applyZoom(); });

function renderMenuOverlay() {
  if (!activeMenuOverview) return;
  const ov = activeMenuOverview;
  const img = document.getElementById('menuPhotoImg');
  img.src = ov.source_photo_data_url;
  img.style.display = 'block';
  const svg = document.getElementById('menuOverlaySvg');
  svg.setAttribute('viewBox', `0 0 ${ov.image.width} ${ov.image.height}`);
  svg.replaceChildren();

  const showRaw = document.getElementById('chkShowRawBoxes').checked;
  if (showRaw) {
    for (const rd of (ov.raw_detections || [])) {
      const pts = rd.points.map(p => `${p[0]},${p[1]}`).join(' ');
      svg.appendChild(mkSvg('polygon', { points: pts, class: 'poly-raw' }));
    }
  }

  for (const c of (ov.crops || [])) {
    const pts = c.crop_polygon.map(p => `${p[0]},${p[1]}`).join(' ');
    const cls = [
      'poly-crop',
      c.orientation_flag === 1 ? 'vert' : '',
      c.status === 'error' ? 'err' : '',
      c.crop_id === selectedCropId ? 'selected' : '',
    ].filter(Boolean).join(' ');
    const poly = mkSvg('polygon', { points: pts, class: cls, 'data-crop-id': c.crop_id });
    poly.addEventListener('click', () => selectMenuCrop(c.crop_id));
    svg.appendChild(poly);

    const p0 = c.crop_polygon[0] || [0, 0];
    const lblBg = mkSvg('rect', {
      x: Math.max(0, p0[0] - 2),
      y: Math.max(0, p0[1] - 18),
      width: 28,
      height: 18,
      rx: 3,
      fill: c.crop_id === selectedCropId ? '#f59e0b' : '#0f172a',
      'fill-opacity': '0.85',
    });
    const lbl = mkSvg('text', {
      x: Math.max(2, p0[0] + 2),
      y: Math.max(13, p0[1] - 5),
      fill: '#ffffff',
      'font-size': 12,
      'font-weight': 'bold',
      'pointer-events': 'none',
    });
    lbl.textContent = `#${c.crop_index}`;
    svg.appendChild(lblBg);
    svg.appendChild(lbl);
  }
  applyZoom();
}

document.getElementById('chkShowRawBoxes').addEventListener('change', renderMenuOverlay);

function renderCropList() {
  const list = document.getElementById('cropList');
  list.replaceChildren();
  if (!activeMenuOverview) return;
  for (const c of (activeMenuOverview.crops || [])) {
    const row = mkEl('div', `crop-row${c.crop_id === selectedCropId ? ' selected' : ''}`);
    row.dataset.cropId = c.crop_id;
    row.addEventListener('click', () => selectMenuCrop(c.crop_id));

    row.appendChild(mkEl('div', 'mono', `#${c.crop_index}`));
    const thumb = document.createElement('img');
    thumb.className = 'crop-thumb';
    thumb.src = c.upright_crop_data_url;
    thumb.alt = c.crop_id;
    row.appendChild(thumb);

    const info = mkEl('div', '');
    const txt = c.status === 'error' ? `[ERR] ${c.error}` : (c.canonical_ocr || '(empty)');
    info.appendChild(mkEl('div', '', txt));
    info.appendChild(mkEl('div', 'mono', `${c.orientation_label} · conf=${c.confidence} · ${c.status}${c.long_line_route === 'overlap512_h64' ? ' · overlap512' : ''}`));
    row.appendChild(info);
    list.appendChild(row);
  }
}

async function selectMenuCrop(cropId) {
  if (!activeMenuJobId || !activeMenuOverview) return;
  selectedCropId = cropId;
  renderMenuOverlay();
  renderCropList();
  try {
    const resp = await fetch(`/api/menu/jobs/${activeMenuJobId}/crops/${encodeURIComponent(cropId)}`);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
    const det = data.crop_detail;
    renderResult({
      result: det.result,
      error: det.error,
      geometry: det.geometry,
      crop_data_url: det.upright_crop_data_url,
      ocr_strip_data_url: det.ocr_strip_data_url,
      menu_provenance: det.menu_provenance,
    });
  } catch (e) { showError(e.message); }
}

function stopPolling() {
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
}

async function pollMenuJob(jobId) {
  if (jobId !== activeMenuJobId) return;
  try {
    const st = await fetch(`/api/menu/jobs/${jobId}/status`).then(r => r.json());
    if (jobId !== activeMenuJobId) return;
    const pct = st.status === 'completed' ? 100
      : st.stage === 'detecting' ? 20
      : st.stage === 'postprocessing' ? 35
      : st.stage === 'recognizing' && st.crops_total > 0 ? Math.round(35 + 65 * (st.crops_done / st.crops_total))
      : 10;
    document.getElementById('menuProgressFill').style.width = `${pct}%`;
    document.getElementById('menuJobStatus').textContent =
      `job=${jobId} · status=${st.status} · stage=${st.stage} (${st.crops_done}/${st.crops_total}) · mode=${st.decode_mode}`;

    if (st.status === 'completed') {
      stopPolling();
      document.getElementById('cancelMenuBtn').disabled = true;
      document.getElementById('rerunMenuBtn').disabled = false;
      const res = await fetch(`/api/menu/jobs/${jobId}/result`).then(r => r.json());
      if (jobId !== activeMenuJobId) return;
      activeMenuOverview = res.overview;
      const tm = activeMenuOverview.timings_ms;
      const cnt = activeMenuOverview.counts;
      const det = activeMenuOverview.detector;
      document.getElementById('menuTimings').textContent =
        `crops=${cnt.total} (ok=${cnt.ok}, empty=${cnt.empty}, err=${cnt.error}) | raw_dets=${activeMenuOverview.postprocess.raw_count} | decode=${tm.decode_ms}ms detect=${tm.detection_ms}ms post=${tm.postprocess_ms}ms ocr+lookup=${tm.recognition_lookup_ms}ms total=${tm.total_ms}ms`;
      document.getElementById('menuProvenance').textContent =
        `V5=${det.weights_name} (conf=${det.confidence_threshold}, sha=${(det.weights_sha256||'').slice(0,12)}) · photo=${activeMenuOverview.image.width}x${activeMenuOverview.image.height} · cfg=${activeMenuOverview.config_signature}`;
      selectedCropId = (activeMenuOverview.crops && activeMenuOverview.crops.length) ? activeMenuOverview.crops[0].crop_id : null;
      renderMenuOverlay();
      renderCropList();
      if (selectedCropId) selectMenuCrop(selectedCropId);
      return;
    }
    if (st.status === 'failed' || st.status === 'cancelled') {
      stopPolling();
      document.getElementById('cancelMenuBtn').disabled = true;
      if (st.error) showError(`Menu job ${st.status}: ${st.error}`);
      return;
    }
    pollTimer = setTimeout(() => pollMenuJob(jobId), 350);
  } catch (e) {
    stopPolling();
    showError(e.message);
  }
}

document.getElementById('runMenuBtn').addEventListener('click', () => {
  const f = document.getElementById('menuFileInput').files[0];
  if (!f) { showError('Choose a menu photo (JPEG/PNG/WebP) first.'); return; }
  showError('');
  const reader = new FileReader();
  reader.onload = async () => {
    try {
      stopPolling();
      const payload = {
        ...getConfigPayload(),
        image_base64: reader.result,
        filename: f.name,
        confidence: parseFloat(document.getElementById('menuConfInput').value || '0.35'),
      };
      const st = await postJson('/api/menu/jobs', payload);
      activeMenuJobId = st.job_id;
      activeMenuOverview = null;
      selectedCropId = null;
      document.getElementById('cancelMenuBtn').disabled = false;
      document.getElementById('rerunMenuBtn').disabled = true;
      document.getElementById('menuProgressFill').style.width = '10%';
      pollMenuJob(st.job_id);
    } catch (e) { showError(e.message); }
  };
  reader.readAsDataURL(f);
});

document.getElementById('rerunMenuBtn').addEventListener('click', async () => {
  if (!activeMenuJobId) return;
  try {
    stopPolling();
    const payload = {
      ...getConfigPayload(),
      confidence: parseFloat(document.getElementById('menuConfInput').value || '0.35'),
    };
    const st = await postJson(`/api/menu/jobs/${activeMenuJobId}/rerun`, payload);
    activeMenuJobId = st.job_id;
    document.getElementById('cancelMenuBtn').disabled = false;
    document.getElementById('rerunMenuBtn').disabled = true;
    pollMenuJob(st.job_id);
  } catch (e) { showError(e.message); }
});

document.getElementById('cancelMenuBtn').addEventListener('click', async () => {
  if (!activeMenuJobId) return;
  try {
    await postJson(`/api/menu/jobs/${activeMenuJobId}/cancel`, {});
    stopPolling();
    document.getElementById('cancelMenuBtn').disabled = true;
    document.getElementById('menuJobStatus').textContent = `job=${activeMenuJobId} cancelled.`;
  } catch (e) { showError(e.message); }
});

async function init() {
  const h = await fetch('/api/health').then(r => r.json());
  document.getElementById('activeModelLabel').textContent =
    `V5 YOLO → ${h.model?.version || 'recognizer'} (${h.model?.state_key || 'unavailable'}) → Char-LM R1 → Lexicon Trie → Read-Only DB`;
  document.getElementById('statusBadge').textContent =
    `Ready · ${h.index_entries} terms · model=${h.has_model} · yolo=${h.has_yolo}`;
  const s = await fetch('/api/samples').then(r => r.json());
  samplesList = s.samples || [];
  const sel = document.getElementById('sampleSelect');
  sel.replaceChildren();
  for (const item of samplesList) {
    const opt = document.createElement('option');
    opt.value = item.id;
    opt.textContent = `[${item.kind}] ${item.label}`;
    sel.appendChild(opt);
  }
  sel.addEventListener('change', () => {
    const cur = samplesList.find(x => x.id === sel.value);
    document.getElementById('sampleNote').textContent = cur ? cur.note : '';
  });
  if (samplesList.length) {
    document.getElementById('sampleNote').textContent = samplesList[0].note;
  }
}

document.getElementById('runTextBtn').addEventListener('click', async () => {
  try {
    const payload = { ...getConfigPayload(), text: document.getElementById('textInput').value };
    renderResult(await postJson('/api/lookup_text', payload));
  } catch (e) { showError(e.message); }
});

document.getElementById('runSampleBtn').addEventListener('click', async () => {
  try {
    const payload = { ...getConfigPayload(), sample_id: document.getElementById('sampleSelect').value };
    renderResult(await postJson('/api/lookup_sample', payload));
  } catch (e) { showError(e.message); }
});

document.getElementById('runUploadBtn').addEventListener('click', async () => {
  const f = document.getElementById('fileInput').files[0];
  if (!f) { showError('Choose an image file first.'); return; }
  const reader = new FileReader();
  reader.onload = async () => {
    try {
      const payload = {
        ...getConfigPayload(),
        image_base64: reader.result,
        filename: f.name,
        orientation: parseInt(document.getElementById('orientationSelect').value, 10),
      };
      renderResult(await postJson('/api/lookup_upload', payload));
    } catch (e) { showError(e.message); }
  };
  reader.readAsDataURL(f);
});

document.getElementById('saveReviewBtn').addEventListener('click', () => {
  if (!lastResult || !lastResult.result) return;
  const r = lastResult.result;
  const mp = lastResult.menu_provenance || null;
  reviews.push({
    timestamp_utc: new Date().toISOString(),
    verdict: document.getElementById('reviewVerdict').value,
    comment: document.getElementById('reviewComment').value,
    menu_id: mp ? mp.menu_id : null,
    crop_id: lastCropMeta ? lastCropMeta.crop_id : null,
    config_signature: mp ? mp.config_signature : null,
    detector_weights_sha256: (mp && mp.detector) ? mp.detector.weights_sha256 : null,
    raw_ocr: r.raw_ocr,
    canonical_ocr: r.canonical_ocr,
    suggested_line: r.suggested_line,
    visual_reranked_line: r.visual_reranked_line ?? null,
    decoder_mode: r.decoder_mode,
    match_types: (r.fuzzy_suggestions || []).map(f => f.match_type || 'ordinary_fuzzy_suggestion'),
    suggested_translations_count: (r.fuzzy_suggestions || []).filter(f => f.translation && f.translation.status === 'found').length,
  });
  document.getElementById('reviewComment').value = '';
  document.getElementById('reviewCount').textContent = `${reviews.length} reviews saved in browser`;
});

document.getElementById('exportReviewsBtn').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(reviews, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'lexicon_inspector_reviews.json';
  a.click();
});

document.getElementById('importReviewsInput').addEventListener('change', (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  f.text().then(txt => {
    const arr = JSON.parse(txt);
    if (!Array.isArray(arr)) throw new Error('Expected JSON array');
    reviews = arr.map(item => ({
      timestamp_utc: item.timestamp_utc || new Date().toISOString(),
      verdict: item.verdict || 'needs_edit',
      comment: item.comment || '',
      menu_id: item.menu_id ?? null,
      crop_id: item.crop_id ?? null,
      config_signature: item.config_signature ?? null,
      detector_weights_sha256: item.detector_weights_sha256 ?? null,
      raw_ocr: item.raw_ocr || '',
      canonical_ocr: item.canonical_ocr || '',
      suggested_line: item.suggested_line || '',
      visual_reranked_line: item.visual_reranked_line ?? null,
      decoder_mode: item.decoder_mode || 'text_only',
      match_types: Array.isArray(item.match_types) ? item.match_types : [],
      suggested_translations_count: Number.isFinite(item.suggested_translations_count) ? item.suggested_translations_count : 0,
    }));
    document.getElementById('reviewCount').textContent = `${reviews.length} reviews imported`;
  }).catch(e => showError(e.message));
});

init().catch(e => showError(e.message));
</script>
</body>
</html>
"""


class InspectorAppState:
    """Immutable shared server state holding pipeline, allowlisted samples, and optional MenuJobManager."""

    def __init__(
        self,
        pipeline: LexiconPipeline,
        samples: dict[str, dict[str, Any]],
        menu_jobs: Optional[MenuJobManager] = None,
    ) -> None:
        self.pipeline = pipeline
        self.samples = samples
        self.menu_jobs = menu_jobs


def make_handler(state: InspectorAppState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stdout.write(f"[inspector] {self.address_string()} - {fmt % args}\n")
            sys.stdout.flush()

        def _send_json(self, code: int, obj: dict[str, Any]) -> None:
            raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _read_json_body(self, max_bytes: int = MAX_CROP_BODY_BYTES) -> dict[str, Any]:
            cl = self.headers.get("Content-Length")
            if cl is None:
                raise ValueError("Missing Content-Length header")
            length = int(cl)
            if length < 0 or length > max_bytes:
                raise OverflowError(f"Request body exceeds {max_bytes} bytes")
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object")
            return data

        def do_GET(self) -> None:
            try:
                if self.path in ("/", "/index.html"):
                    body = HTML_PAGE.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path == "/api/health":
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "status": "ok",
                            "index_entries": len(state.pipeline.index.entries),
                            "index_schema_version": state.pipeline.index.metadata.get("schema_version"),
                            "has_model": state.pipeline.has_model,
                            "model": state.pipeline.model_metadata,
                            "has_lm": state.pipeline.has_lm,
                            "has_yolo": state.menu_jobs is not None,
                            "auto_corrected_default": False,
                        },
                    )
                    return

                if self.path == "/api/confusables":
                    self._send_json(HTTPStatus.OK, DEFAULT_CONFUSABLE_CATALOG.to_dict())
                    return

                if self.path == "/api/samples":
                    public_samples = [
                        {
                            "id": s["id"],
                            "kind": s["kind"],
                            "label": s["label"],
                            "note": s.get("note", ""),
                            "truth_reference": s.get("truth_reference"),
                        }
                        for s in state.samples.values()
                    ]
                    self._send_json(HTTPStatus.OK, {"samples": public_samples})
                    return

                m_st = _RE_MENU_STATUS.match(self.path)
                if m_st:
                    if state.menu_jobs is None:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Full-menu detector not enabled"})
                        return
                    self._send_json(HTTPStatus.OK, state.menu_jobs.get_status(m_st.group(1)))
                    return

                m_res = _RE_MENU_RESULT.match(self.path)
                if m_res:
                    if state.menu_jobs is None:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Full-menu detector not enabled"})
                        return
                    self._send_json(HTTPStatus.OK, state.menu_jobs.get_result(m_res.group(1)))
                    return

                m_crop = _RE_MENU_CROP.match(self.path)
                if m_crop:
                    if state.menu_jobs is None:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Full-menu detector not enabled"})
                        return
                    self._send_json(HTTPStatus.OK, state.menu_jobs.get_crop_detail(m_crop.group(1), m_crop.group(2)))
                    return

                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {self.path}"})
            except KeyError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:
                traceback.print_exc()
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self) -> None:
            try:
                is_menu_submit = self.path == "/api/menu/jobs"
                body = self._read_json_body(max_bytes=MAX_MENU_BODY_BYTES if is_menu_submit else MAX_CROP_BODY_BYTES)

                # Full-menu async job endpoints
                if is_menu_submit:
                    if state.menu_jobs is None:
                        self._send_json(
                            HTTPStatus.BAD_REQUEST,
                            {"error": "Full-menu YOLO detector is disabled on this server instance"},
                        )
                        return
                    cfg = parse_lookup_config(body)
                    decode_mode = str(body.get("decoder_mode", "greedy"))
                    conf = float(body.get("confidence", 0.35))
                    if not (0.05 <= conf <= 0.95):
                        raise ValueError("confidence must be between 0.05 and 0.95")
                    img_bytes = _decode_base64_field(body.get("image_base64"), MAX_MENU_UPLOAD_BYTES)
                    fname = Path(str(body.get("filename", "menu.jpg"))).name
                    st = state.menu_jobs.submit(
                        image_bytes=img_bytes,
                        filename=fname,
                        decode_mode=decode_mode,
                        lookup_config=cfg,
                        confidence=conf,
                    )
                    self._send_json(HTTPStatus.ACCEPTED, st)
                    return

                m_rerun = _RE_MENU_RERUN.match(self.path)
                if m_rerun:
                    if state.menu_jobs is None:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Full-menu detector not enabled"})
                        return
                    cfg = parse_lookup_config(body)
                    decode_mode = str(body.get("decoder_mode", "greedy"))
                    conf = float(body.get("confidence", 0.35))
                    st = state.menu_jobs.rerun_from_existing(
                        m_rerun.group(1),
                        decode_mode=decode_mode,
                        lookup_config=cfg,
                        confidence=conf,
                    )
                    self._send_json(HTTPStatus.ACCEPTED, st)
                    return

                m_cancel = _RE_MENU_CANCEL.match(self.path)
                if m_cancel:
                    if state.menu_jobs is None:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Full-menu detector not enabled"})
                        return
                    self._send_json(HTTPStatus.OK, state.menu_jobs.cancel(m_cancel.group(1)))
                    return

                # Existing single-crop & text endpoints
                cfg = parse_lookup_config(body)
                decode_mode = str(body.get("decoder_mode", "greedy"))

                if self.path == "/api/lookup_text":
                    text = body.get("text")
                    if not isinstance(text, str):
                        raise ValueError("Field 'text' must be a string")
                    res = state.pipeline.lookup_text(text, config=cfg)
                    self._send_json(HTTPStatus.OK, {"result": res.to_dict()})
                    return

                if self.path == "/api/lookup_sample":
                    sid = str(body.get("sample_id", ""))
                    sample = state.samples.get(sid)
                    if sample is None:
                        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown sample_id: {sid!r}"})
                        return
                    if sample["kind"] == "text":
                        res = state.pipeline.lookup_text(sample["text"], config=cfg)
                        self._send_json(
                            HTTPStatus.OK,
                            {"result": res.to_dict(), "truth_reference": sample.get("truth_reference")},
                        )
                        return
                    img_path = Path(sample["image_path"])
                    raw_bytes = img_path.read_bytes()
                    img = Image.open(io.BytesIO(raw_bytes))
                    res = state.pipeline.lookup_image(
                        img,
                        orientation=int(sample.get("orientation", 0)),
                        config=cfg,
                        decode_mode=decode_mode,
                        crop_provenance={"sample_id": sid, "file": img_path.name},
                    )
                    b64 = base64.b64encode(raw_bytes).decode("ascii")
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "result": res.to_dict(),
                            "truth_reference": sample.get("truth_reference"),
                            "crop_data_url": f"data:image/png;base64,{b64}",
                        },
                    )
                    return

                if self.path == "/api/lookup_upload":
                    if not state.pipeline.has_model:
                        self._send_json(
                            HTTPStatus.BAD_REQUEST,
                            {"error": "Model inference is disabled on this server instance"},
                        )
                        return
                    b64_raw = body.get("image_base64")
                    img_bytes = _decode_base64_field(b64_raw, MAX_CROP_BODY_BYTES)
                    img = Image.open(io.BytesIO(img_bytes))
                    img.load()
                    ori = int(body.get("orientation", 0))
                    if ori not in (0, 1):
                        raise ValueError("orientation must be 0 or 1")
                    fname = Path(str(body.get("filename", "upload.png"))).name
                    res = state.pipeline.lookup_image(
                        img,
                        orientation=ori,
                        config=cfg,
                        decode_mode=decode_mode,
                        crop_provenance={"uploaded_filename": fname},
                    )
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "result": res.to_dict(),
                            "crop_data_url": b64_raw if isinstance(b64_raw, str) and b64_raw.startswith("data:") else None,
                        },
                    )
                    return

                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {self.path}"})

            except OverflowError as exc:
                self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": str(exc)})
            except KeyError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            except (ValueError, json.JSONDecodeError, IncompatibleLMSchemaError, UnidentifiedImageError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"{type(exc).__name__}: {exc}"})
            except Exception as exc:
                traceback.print_exc()
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def load_inspection_model(device, checkpoint=None, checkpoint_sha256=None, checkpoint_state="raw", model_label=None):
    """Load a hash-pinned same-architecture candidate without changing promotion."""
    if checkpoint is not None:
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_relative_to(ROOT) or not checkpoint.is_file():
            raise ValueError("Inspection checkpoint must be an existing workspace file")
        digest = sha256_file(checkpoint)
        if not checkpoint_sha256 or digest != checkpoint_sha256:
            raise ValueError("Inspection checkpoint SHA-256 mismatch or missing expected hash")
    model, saved = load_current_recognizer(device=device)
    meta = {
        "version": "L2_v06",
        "architecture": saved.get("active_architecture"),
        "state_key": saved.get("active_state_key"),
        "device": device,
        "registry_override": False,
    }
    if checkpoint is not None:
        import torch
        candidate = torch.load(checkpoint, map_location="cpu", weights_only=False)
        aliases = {"raw": "model", "model": "raw", "ema": "ema_model", "ema_model": "ema"}
        key = checkpoint_state if checkpoint_state in candidate else aliases.get(checkpoint_state, checkpoint_state)
        if key not in candidate:
            raise ValueError(f"Inspection state {checkpoint_state!r} missing")
        if "vocab" in candidate and candidate["vocab"]["char_to_id"] != model.vocabulary:
            raise ValueError("Inspection vocabulary differs from promoted vocabulary")
        model.load_state_dict(candidate[key], strict=True)
        model.eval()
        meta.update(version=model_label or checkpoint.parent.name, state_key=key,
                    checkpoint=str(checkpoint.relative_to(ROOT)), sha256=digest,
                    epoch=candidate.get("epoch"), registry_override=True)
    return model, meta


def create_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8772,
    index_path: Path = DEFAULT_INDEX_PATH,
    db_path: Path = DEFAULT_DB_PATH,
    lm_path: Optional[Path] = DEFAULT_LM_PATH,
    yolo_weights: Optional[Path] = DEFAULT_V5_YOLO_WEIGHTS,
    load_model: bool = True,
    load_yolo: bool = True,
    device: str = "cpu",
    checkpoint: Optional[Path] = None,
    checkpoint_sha256: Optional[str] = None,
    checkpoint_state: str = "raw",
    model_label: Optional[str] = None,
) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError(f"Refusing non-localhost bind address: {host!r}")

    index = LexiconIndex.load_gz(index_path)
    repo = ReadOnlyTranslationRepo(db_path)
    lm = ImmutableCharNGramLM.load_gz(lm_path, allow_legacy_v1=False) if (lm_path and lm_path.is_file()) else None

    model = None
    model_meta: dict[str, Any] = {}
    if load_model:
        model, model_meta = load_inspection_model(
            device, checkpoint, checkpoint_sha256, checkpoint_state, model_label)

    pipeline = LexiconPipeline(index, repo, immutable_lm=lm, model=model, model_meta=model_meta, device=device)
    menu_jobs: Optional[MenuJobManager] = None
    if load_model and load_yolo and yolo_weights is not None and Path(yolo_weights).is_file():
        detector = V5YoloDetector(yolo_weights, device=device)
        menu_jobs = MenuJobManager(detector=detector, pipeline=pipeline)

    samples = load_allowlisted_real_samples()
    state = InspectorAppState(pipeline=pipeline, samples=samples, menu_jobs=menu_jobs)
    server = ThreadingHTTPServer((host, int(port)), make_handler(state))
    server._menu_jobs = menu_jobs  # type: ignore[attr-defined]
    return server


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Localhost full-menu & lexicon-trie inspection server.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8772)
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    p.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    p.add_argument("--lm", type=Path, default=DEFAULT_LM_PATH)
    p.add_argument("--yolo-weights", type=Path, default=DEFAULT_V5_YOLO_WEIGHTS)
    p.add_argument("--no-model", action="store_true", help="Serve text-only without loading PyTorch model")
    p.add_argument("--no-yolo", action="store_true", help="Disable full-menu YOLO detector")
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint", type=Path, help="Explicit same-architecture inspection candidate; does not promote")
    p.add_argument("--checkpoint-sha256", help="Required expected SHA256 for --checkpoint")
    p.add_argument("--checkpoint-state", choices=("raw", "ema", "model", "ema_model"), default="raw")
    p.add_argument("--model-label", help="Display label for the explicit candidate")
    args = p.parse_args(argv)

    server = create_server(
        host=args.host,
        port=args.port,
        index_path=args.index,
        db_path=args.db,
        lm_path=args.lm,
        yolo_weights=args.yolo_weights,
        load_model=not args.no_model,
        load_yolo=not args.no_yolo,
        device=args.device,
        checkpoint=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        checkpoint_state=args.checkpoint_state,
        model_label=args.model_label,
    )
    print(f"READY http://{args.host}:{server.server_address[1]}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        mj = getattr(server, "_menu_jobs", None)
        if mj is not None:
            mj.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
