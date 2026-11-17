"""Remote synthetic menu-line renderer for V65 (160k corpus with glyph-height stratification, safe margins, and V58 procedural backgrounds)."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.recognition.data import minimum_ctc_steps, output_length
from .menu_background_v58 import BG_FAMILIES, choose_ink_and_audit, contrast_ratio_map, generate_background

BUCKETS = (128, 256, 384, 512)
WEAK_TRAIN_BRUSH_FONTS = {
    "KouzanBrushFont.ttf",
    "YujiSyuku-Regular.ttf",
    "YuseiMagic-Regular.ttf",
    "AoyagiSouseki.ttf",
    "HachiMaruPop-Regular.ttf",
    "ZenKurenaido-Regular.ttf",
}
_G_FONT_PATHS: list[str] = []
_G_FONT_CMAPS: dict[str, set[int]] = {}
_G_OUT_DIR: Path = Path(".")


def load_font_cmap(font_path: str, vocab_chars: list[str]) -> set[int]:
    cmap_keys: set[int] = set()
    try:
        from fontTools.ttLib import TTFont

        tt = TTFont(font_path, fontNumber=0)
        cmap = tt.getBestCmap() or {}
        tt.close()
        cmap_keys = set(cmap.keys())
    except Exception:
        cmap_keys = set()

    font = ImageFont.truetype(font_path, size=36)
    verified: set[int] = set()
    for ch in vocab_chars:
        if ch == " ":
            verified.add(ord(ch))
            continue
        if cmap_keys and ord(ch) not in cmap_keys:
            continue
        bb = font.getbbox(ch)
        if not bb:
            continue
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        if w <= 0 or h <= 0:
            continue
        im = Image.new("L", (w + 8, h + 8), 0)
        ImageDraw.Draw(im).text((4 - bb[0], 4 - bb[1]), ch, font=font, fill=255)
        if np.asarray(im).max() > 12:
            verified.add(ord(ch))
    return verified


def check_font_covers(font_path: str, text: str, cmaps: dict[str, set[int]]) -> bool:
    cmap = cmaps.get(font_path)
    if not cmap:
        return False
    for ch in text:
        if ch == " ":
            continue
        if ord(ch) not in cmap:
            return False
    return True


def _init_worker(font_paths: list[str], font_cmaps: dict[str, set[int]], out_dir: str):
    global _G_FONT_PATHS, _G_FONT_CMAPS, _G_OUT_DIR
    _G_FONT_PATHS = font_paths
    _G_FONT_CMAPS = font_cmaps
    _G_OUT_DIR = Path(out_dir)


def _render_mask(text: str, font_path: str, font_size: int) -> np.ndarray:
    font = ImageFont.truetype(font_path, size=font_size)
    est_w = max(64, int(len(text) * font_size * 1.45) + 40)
    est_h = max(48, int(font_size * 2.2) + 28)
    im = Image.new("L", (est_w, est_h), 0)
    ImageDraw.Draw(im).text((16, 12), text, font=font, fill=255)
    arr = np.asarray(im)
    ys, xs = np.where(arr > 8)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError(f"Blank glyph render for {text!r} with font {font_path}")
    pad = 1
    y0, y1 = max(0, int(ys.min()) - pad), min(arr.shape[0], int(ys.max()) + pad + 1)
    x0, x1 = max(0, int(xs.min()) - pad), min(arr.shape[1], int(xs.max()) + pad + 1)
    return arr[y0:y1, x0:x1]


def _render_one(rec: dict) -> dict:
    idx = int(rec["id"])
    rng = random.Random(idx * 10007 + 20261125)
    np_rng = np.random.default_rng(idx * 10007 + 20261125)
    render_text = rec["render_text"]
    canon_text = rec["canonical_text"]
    appearance = rec["appearance"]
    bg_family = rec["bg_family"]
    bg_tier = rec["bg_tier"]

    eligible_fonts = [fp for fp in _G_FONT_PATHS if check_font_covers(fp, render_text, _G_FONT_CMAPS)]
    excluded_fonts = len(_G_FONT_PATHS) - len(eligible_fonts)
    assert eligible_fonts, f"No eligible training font covers {render_text!r}"
    weights = [2.0 if Path(fp).name in WEAK_TRAIN_BRUSH_FONTS else 1.0 for fp in eligible_fonts]
    font_path = rng.choices(eligible_fonts, weights=weights, k=1)[0]

    min_steps = minimum_ctc_steps(canon_text)
    font_size = rng.randint(28, 52)
    mask = _render_mask(render_text, font_path, font_size)

    if appearance == "mild" and rng.random() < 0.35:
        scale_f = rng.uniform(0.90, 1.10)
        nw = max(4, round(mask.shape[1] * scale_f))
        nh = max(4, round(mask.shape[0] * scale_f))
        mask = np.asarray(Image.fromarray(mask).resize((nw, nh), Image.Resampling.BICUBIC))

    if appearance in ("dilate", "erode"):
        pre_max = int(mask.max())
        filt = ImageFilter.MaxFilter(3) if appearance == "dilate" else ImageFilter.MinFilter(3)
        cand = np.asarray(Image.fromarray(mask).filter(filt))
        if pre_max > 0 and int(cand.max()) > 20 and int((cand > 16).sum()) >= max(8, int(0.35 * (mask > 16).sum())):
            mask = cand

    gh_bin = rec.get("target_gh_bin", "replay")
    if gh_bin == "small":
        target_gh = rng.randint(28, 37)
    elif gh_bin == "medium":
        target_gh = rng.randint(38, 47)
    elif gh_bin == "large":
        target_gh = rng.randint(48, 56)
    else:
        target_gh = rng.randint(32, 54)

    gh, gw = mask.shape
    scale = target_gh / max(1, gh)
    scaled_w = max(4, round(gw * scale))
    if scaled_w > 488:
        scaled_w = 488
        target_gh = max(16, round(gh * (488.0 / max(1, gw))))

    mask_s = np.asarray(Image.fromarray(mask).resize((scaled_w, target_gh), Image.Resampling.BICUBIC))
    top = max(4, min(64 - target_gh - 4, (64 - target_gh) // 2)) if (64 - target_gh) >= 8 else max(0, (64 - target_gh) // 2)
    alpha_64 = np.zeros((64, scaled_w), dtype=np.uint8)
    alpha_64[top : top + target_gh, :] = mask_s

    ys_a, xs_a = np.where(alpha_64 > 16)
    actual_gh = int(ys_a.max() - ys_a.min() + 1) if len(ys_a) else target_gh
    actual_gw = int(xs_a.max() - xs_a.min() + 1) if len(xs_a) else scaled_w
    actual_bin = "small" if actual_gh <= 37 else ("medium" if actual_gh <= 47 else "large")

    lm_band = rec.get("target_lm_band", "replay")
    if lm_band == "wide_16_24":
        left_margin = rng.randint(16, 24)
        right_margin = rng.randint(4, 24)
    elif lm_band == "normal_4_15":
        left_margin = rng.randint(4, 15)
        right_margin = rng.randint(4, 24)
    else:
        left_margin = rng.randint(6, 18)
        right_margin = rng.randint(6, 18)

    valid_width = left_margin + scaled_w + right_margin
    while output_length(valid_width) < min_steps and valid_width < 512:
        right_margin += 4
        valid_width = left_margin + scaled_w + right_margin
    valid_width = min(512, max(16, valid_width))
    bucket = next((b for b in BUCKETS if valid_width <= b), 512)

    is_thin = (font_size < 30) or (actual_gh < 38)
    min_median_cr = 4.5 if (is_thin or rng.random() < 0.92) else 3.3

    amp_scale = 1.0
    for _attempt in range(3):
        strip_bg, polarity = generate_background(bg_family, bg_tier, 64, valid_width, rng, np_rng, amp_scale=amp_scale)
        bg_roi = strip_bg[:, left_margin : left_margin + scaled_w]
        ink, pre_med_cr, pre_p05_cr, pre_below3 = choose_ink_and_audit(bg_roi, alpha_64, polarity, min_median_cr, rng)
        if pre_p05_cr >= 3.0 and pre_below3 <= 0.05 and pre_med_cr >= min_median_cr:
            break
        amp_scale *= 0.45

    strip = strip_bg.copy()
    a_f = alpha_64.astype(np.float32)[..., None] / 255.0
    fg_arr = np.array(ink, dtype=np.float32)[None, None, :]
    roi = strip[:, left_margin : left_margin + scaled_w].astype(np.float32)
    strip[:, left_margin : left_margin + scaled_w] = np.clip(roi * (1.0 - a_f) + fg_arr * a_f, 0, 255).astype(np.uint8)

    if appearance == "mild":
        if rng.random() < 0.4:
            strip = np.asarray(Image.fromarray(strip).filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.35, 0.75))))
        if rng.random() < 0.65:
            strip = np.clip(strip.astype(np.float32) * rng.uniform(0.88, 1.12) + rng.uniform(-10, 10), 0, 255).astype(np.uint8)
        if rng.random() < 0.5:
            g_noise = np_rng.normal(0.0, rng.uniform(1.5, 4.5), size=strip.shape)
            strip = np.clip(strip.astype(np.float32) + g_noise, 0, 255).astype(np.uint8)

    core_mask = alpha_64 >= 180
    if int(core_mask.sum()) < 8:
        core_mask = alpha_64 >= 96
    if int(core_mask.sum()) < 4:
        core_mask = alpha_64 > 16
    post_fg = strip[:, left_margin : left_margin + scaled_w][core_mask]
    post_bg = strip_bg[:, left_margin : left_margin + scaled_w][core_mask]
    post_crs = contrast_ratio_map(post_fg, post_bg)
    post_med_cr = float(np.median(post_crs))
    post_p05_cr = float(np.percentile(post_crs, 5))

    if valid_width < bucket:
        strip = np.pad(strip, ((0, 0), (0, bucket - valid_width), (0, 0)), mode="edge")

    sub_dir = _G_OUT_DIR / "train" / f"{idx // 1000:03d}"
    sub_dir.mkdir(parents=True, exist_ok=True)
    rel_path = f"train/{idx // 1000:03d}/{idx:06d}.png"
    Image.fromarray(strip).save(_G_OUT_DIR / rel_path, compress_level=1)

    return {
        "id": idx,
        "stream": rec["stream"],
        "sub_slot": rec["sub_slot"],
        "render_text": render_text,
        "canonical_text": canon_text,
        "image_rel": rel_path,
        "font": Path(font_path).name,
        "excluded_fonts_for_text": excluded_fonts,
        "appearance": appearance,
        "bg_family": bg_family,
        "bg_tier": bg_tier,
        "polarity": polarity,
        "native_font_size": font_size,
        "target_gh_bin": gh_bin,
        "actual_gh_bin": actual_bin,
        "actual_gh": actual_gh,
        "actual_gw": actual_gw,
        "left_margin": left_margin,
        "right_margin": right_margin,
        "pre_med_cr": round(pre_med_cr, 3),
        "pre_p05_cr": round(pre_p05_cr, 3),
        "post_med_cr": round(post_med_cr, 3),
        "post_p05_cr": round(post_p05_cr, 3),
        "valid_width": int(valid_width),
        "valid_T": int(output_length(valid_width)),
        "bucket": int(bucket),
        "orientation_flag": 0,
    }


def _save_contact_sheet(results: list[dict], images_dir: Path, out_path: Path) -> None:
    rng = random.Random(20261125)
    picks = []
    for fam in BG_FAMILIES:
        cand = [r for r in results if r["bg_family"] == fam]
        picks.extend(rng.sample(cand, min(4, len(cand))))
    cols, rows_n = 4, 6
    cell_w, cell_h = 520, 72
    sheet = Image.new("RGB", (cols * cell_w, rows_n * cell_h), (20, 20, 20))
    for i, r in enumerate(picks[: cols * rows_n]):
        im = Image.open(images_dir / r["image_rel"]).convert("RGB")
        cx, cy = (i % cols) * cell_w, (i // cols) * cell_h
        sheet.paste(im, (cx + 4, cy + 4))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def render_v65_corpus(base_dir: Path) -> dict:
    t0 = time.time()
    payload_dir = base_dir / "payload"
    manifests_dir = base_dir / "manifests"
    images_dir = base_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    vocab = json.loads((payload_dir / "vocab.json").read_text(encoding="utf-8"))
    vocab_chars = vocab["characters"]
    font_paths = sorted(str(p) for p in (payload_dir / "fonts" / "train").iterdir() if p.is_file() and p.suffix.lower() in (".ttf", ".otf", ".ttc"))
    cmap_chars = list(dict.fromkeys(list(vocab_chars) + list("０１２３４５６７８９")))
    font_cmaps = {fp: load_font_cmap(fp, cmap_chars) for fp in font_paths}

    corpus_rows = [json.loads(l) for l in (manifests_dir / "v65_train_corpus.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(corpus_rows) == 160000

    n_workers = min(8, os.cpu_count() or 4)
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(font_paths, font_cmaps, str(images_dir))) as ex:
        for res in ex.map(_render_one, corpus_rows, chunksize=250):
            results.append(res)
            if len(results) % 20000 == 0:
                print(f"[v65_render] {len(results)}/{len(corpus_rows)}", flush=True)

    rendered_path = images_dir / "v65_train_rendered.jsonl"
    rendered_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n", encoding="utf-8")

    short_res = [r for r in results if r["stream"] in ("short_1char", "short_2char")]
    audit = {
        "count": len(results),
        "appearance_counts": dict(Counter(r["appearance"] for r in results)),
        "bucket_counts": dict(Counter(str(r["bucket"]) for r in results)),
        "bg_family_counts": dict(Counter(r["bg_family"] for r in results)),
        "bg_tier_counts": dict(Counter(r["bg_tier"] for r in results)),
        "polarity_counts": dict(Counter(r["polarity"] for r in results)),
        "short_actual_gh_bins": dict(Counter(r["actual_gh_bin"] for r in short_res)),
        "short_left_margin_ge16": sum(1 for r in short_res if r["left_margin"] >= 16),
        "short_total": len(short_res),
        "font_counts": dict(Counter(r["font"] for r in results)),
        "post_med_cr_ge_45_rate": round(sum(1 for r in results if r["post_med_cr"] >= 4.5) / len(results), 4),
        "post_p05_cr_ge_30_rate": round(sum(1 for r in results if r["post_p05_cr"] >= 3.0) / len(results), 4),
        "manifest_sha256": hashlib.sha256(rendered_path.read_bytes()).hexdigest(),
        "render_seconds": round(time.time() - t0, 2),
    }
    (manifests_dir / "v65_render_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    _save_contact_sheet(results, images_dir, manifests_dir / "v65_qa_contact_sheet.png")
    return audit


if __name__ == "__main__":
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/content/v65_run")
    print(json.dumps(render_v65_corpus(base), indent=2, ensure_ascii=False))
