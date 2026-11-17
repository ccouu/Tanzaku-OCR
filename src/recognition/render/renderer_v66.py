"""Remote synthetic menu-line renderer for V66 (160k corpus: 30% genuine native vertical flag=1 / 70% horizontal flag=0)."""
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
from .menu_background_v58 import choose_ink_and_audit, contrast_ratio_map, generate_background
from .renderer_v65 import BUCKETS, WEAK_TRAIN_BRUSH_FONTS, _render_mask, check_font_covers, load_font_cmap

VERTICAL_MAP = {"ー": "｜", "「": "﹁", "」": "﹂", "（": "︵", "）": "︶", "(": "︵", ")": "︶"}
_G_FONT_PATHS: list[str] = []
_G_FONT_CMAPS: dict[str, set[int]] = {}
_G_OUT_DIR: Path = Path(".")


def _init_worker(font_paths: list[str], font_cmaps: dict[str, set[int]], out_dir: str):
    global _G_FONT_PATHS, _G_FONT_CMAPS, _G_OUT_DIR
    _G_FONT_PATHS = font_paths
    _G_FONT_CMAPS = font_cmaps
    _G_OUT_DIR = Path(out_dir)


def _render_vertical_glyph_tile(ch: str, font_path: str, step: int, cmap: set[int]) -> Image.Image:
    tile = Image.new("L", (step, step), 0)
    draw = ImageDraw.Draw(tile)
    if ch == "ー":
        # Vertical prolonged sound mark; becomes horizontal 'ー' after ROTATE_90 CCW
        draw.line((step // 2, max(1, step // 8), step // 2, min(step - 2, step * 7 // 8)), fill=255, width=max(2, step // 12))
        return tile
    mapped = VERTICAL_MAP.get(ch, ch)
    if mapped != ch and ord(mapped) not in cmap:
        return _render_vertical_glyph_tile(ch, font_path, step, cmap).transpose(Image.Transpose.ROTATE_270) if False else (
            _render_upright_single(ch, font_path, step).transpose(Image.Transpose.ROTATE_270)
        )
    return _render_upright_single(mapped, font_path, step)


def _render_upright_single(ch: str, font_path: str, step: int) -> Image.Image:
    tile = Image.new("L", (step, step), 0)
    draw = ImageDraw.Draw(tile)
    font = ImageFont.truetype(font_path, size=max(8, int(round(step * 0.84))))
    bb = draw.textbbox((0, 0), ch, font=font)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    draw.text(((step - w) / 2.0 - bb[0], (step - h) / 2.0 - bb[1]), ch, font=font, fill=255)
    return tile


def _render_vertical_mask(text: str, font_path: str, step: int, cmap: set[int]) -> np.ndarray:
    n = len(text)
    strip = Image.new("L", (step + 8, n * step + 8), 0)
    for i, ch in enumerate(text):
        tile = _render_vertical_glyph_tile(ch, font_path, step, cmap)
        strip.paste(tile, (4, 4 + i * step))
    arr = np.asarray(strip)
    ys, xs = np.where(arr > 8)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError(f"Blank vertical glyph render for {text!r} with font {font_path}")
    return arr[max(0, int(ys.min()) - 1) : min(arr.shape[0], int(ys.max()) + 2), max(0, int(xs.min()) - 1) : min(arr.shape[1], int(xs.max()) + 2)]


def _apply_appearance_and_composite(
    alpha_64: np.ndarray,
    scaled_w: int,
    left_margin: int,
    valid_width: int,
    appearance: str,
    bg_family: str,
    bg_tier: str,
    is_thin: bool,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> tuple[np.ndarray, str, float, float, float, float]:
    min_median_cr = 4.5 if (is_thin or rng.random() < 0.92) else 3.3
    amp_scale = 1.0
    for _ in range(3):
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
    return strip, polarity, pre_med_cr, pre_p05_cr, float(np.median(post_crs)), float(np.percentile(post_crs, 5))


def _render_one(rec: dict) -> dict:
    idx = int(rec["id"])
    rng = random.Random(20261128 + idx)
    np_rng = np.random.default_rng(20261128 + idx)

    render_text = rec["render_text"]
    canon_text = rec["canonical_text"]
    appearance = rec["appearance"]
    bg_family = rec["bg_family"]
    bg_tier = rec["bg_tier"]
    orientation_flag = int(rec["orientation_flag"])
    min_steps = minimum_ctc_steps(canon_text)

    eligible_fonts = [fp for fp in _G_FONT_PATHS if check_font_covers(fp, render_text, _G_FONT_CMAPS)]
    if not eligible_fonts:
        raise RuntimeError(f"No font covers {render_text!r}")
    font_path = rng.choice(eligible_fonts)

    gh_bin = rec.get("target_gh_bin", "replay")
    if gh_bin == "small":
        target_gh = rng.randint(28, 37)
    elif gh_bin == "medium":
        target_gh = rng.randint(38, 47)
    elif gh_bin == "large":
        target_gh = rng.randint(48, 56)
    else:
        target_gh = rng.randint(34, 52)

    if orientation_flag == 1:
        # Native vertical: upright glyphs top-to-bottom, then ROTATE_90 CCW
        n = len(render_text)
        step = min(46, max(28, int(round(460.0 / max(1, n)))))
        if gh_bin == "small":
            step = min(step, rng.randint(28, 37))
        elif gh_bin == "medium":
            step = min(step, rng.randint(38, 46))
        mask_arr = _render_vertical_mask(render_text, font_path, step, _G_FONT_CMAPS[font_path])
        if appearance == "dilate":
            mask_arr = np.asarray(Image.fromarray(mask_arr).filter(ImageFilter.MaxFilter(3)))
        elif appearance == "erode" and min(mask_arr.shape) >= 28:
            eroded = np.asarray(Image.fromarray(mask_arr).filter(ImageFilter.MinFilter(3)))
            if int((eroded > 128).sum()) >= 16:
                mask_arr = eroded
        # Rotate upright vertical mask 90 deg CCW -> height is column width, width is column height
        rot_mask = np.asarray(Image.fromarray(mask_arr).transpose(Image.Transpose.ROTATE_90))
        rh, rw = rot_mask.shape
        target_h = min(56, max(28, target_gh))
        scale = min(target_h / float(max(1, rh)), 488.0 / float(max(1, rw)))
        scaled_w = max(12, int(round(rw * scale)))
        scaled_h = max(20, min(58, int(round(rh * scale))))
        alpha_im = Image.fromarray(rot_mask).resize((scaled_w, scaled_h), Image.Resampling.BICUBIC)
        y_off = max(2, (64 - scaled_h) // 2)
        alpha_64 = np.zeros((64, scaled_w), dtype=np.uint8)
        alpha_64[y_off : y_off + scaled_h, :] = np.asarray(alpha_im)
        if int((alpha_64 > 64).sum()) < 8 and int(alpha_64.max()) > 0:
            alpha_64 = np.clip(alpha_64.astype(np.float32) * (220.0 / float(alpha_64.max())), 0, 255).astype(np.uint8)
        font_size = step
    else:
        # Horizontal control (identical to V65)
        font_size = rng.randint(28, 48)
        mask_arr = _render_mask(render_text, font_path, font_size)
        if appearance == "dilate":
            mask_arr = np.asarray(Image.fromarray(mask_arr).filter(ImageFilter.MaxFilter(3)))
        elif appearance == "erode" and font_size >= 26:
            eroded = np.asarray(Image.fromarray(mask_arr).filter(ImageFilter.MinFilter(3)))
            if int((eroded > 128).sum()) >= 16:
                mask_arr = eroded
        mh, mw = mask_arr.shape
        scale = min(target_gh / float(max(1, mh)), 488.0 / float(max(1, mw)))
        scaled_w = max(8, int(round(mw * scale)))
        scaled_h = max(16, min(58, int(round(mh * scale))))
        alpha_im = Image.fromarray(mask_arr).resize((scaled_w, scaled_h), Image.Resampling.BICUBIC)
        y_off = max(2, min(64 - scaled_h - 2, (64 - scaled_h) // 2 + rng.randint(-3, 3)))
        alpha_64 = np.zeros((64, scaled_w), dtype=np.uint8)
        alpha_64[y_off : y_off + scaled_h, :] = np.asarray(alpha_im)
        if int((alpha_64 > 64).sum()) < 8 and int(alpha_64.max()) > 0:
            alpha_64 = np.clip(alpha_64.astype(np.float32) * (220.0 / float(alpha_64.max())), 0, 255).astype(np.uint8)

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

    if left_margin + scaled_w + right_margin > 512:
        left_margin = max(4, min(left_margin, 512 - scaled_w - 4))
        right_margin = max(4, min(right_margin, 512 - scaled_w - left_margin))

    valid_width = left_margin + scaled_w + right_margin
    while output_length(valid_width) < min_steps and valid_width < 512:
        right_margin += 4
        valid_width = left_margin + scaled_w + right_margin
    valid_width = min(512, max(16, valid_width))
    bucket = next((b for b in BUCKETS if valid_width <= b), 512)

    is_thin = (font_size < 30) or (actual_gh < 38)
    strip, polarity, pre_med_cr, pre_p05_cr, post_med_cr, post_p05_cr = _apply_appearance_and_composite(
        alpha_64, scaled_w, left_margin, valid_width, appearance, bg_family, bg_tier, is_thin, rng, np_rng
    )

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
        "orientation_flag": orientation_flag,
    }


def render_v66_dataset(base_dir: Path, workers: int | None = None) -> dict:
    t0 = time.time()
    payload_dir = base_dir / "payload"
    manifests_dir = base_dir / "manifests"
    images_dir = base_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    vocab = json.loads((payload_dir / "vocab.json").read_text(encoding="utf-8"))
    vocab_chars = vocab["characters"] + list("０１２３４５６７８９")
    font_paths = sorted(str(p) for p in (payload_dir / "fonts" / "train").iterdir() if p.is_file() and p.suffix.lower() in (".ttf", ".otf", ".ttc"))
    font_cmaps = {fp: load_font_cmap(fp, vocab_chars) for fp in font_paths}

    corpus_rows = [json.loads(l) for l in (manifests_dir / "v66_train_corpus.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    workers = workers or min(12, max(2, os.cpu_count() or 4))

    rendered: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(font_paths, font_cmaps, str(images_dir))) as ex:
        for i, res in enumerate(ex.map(_render_one, corpus_rows, chunksize=250), 1):
            rendered.append(res)
            if i % 40000 == 0:
                print(f"[render_v66] rendered {i}/{len(corpus_rows)} in {time.time() - t0:.1f}s", flush=True)

    rendered_path = images_dir / "v66_train_rendered.jsonl"
    rendered_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rendered) + "\n", encoding="utf-8")

    # Compact QA contact sheet: 12 vertical (flag=1) + 4 horizontal (flag=0)
    vert_samples = [r for r in rendered if r["orientation_flag"] == 1][:12]
    horiz_samples = [r for r in rendered if r["orientation_flag"] == 0][:4]
    qa_samples = vert_samples + horiz_samples
    sheet = Image.new("RGB", (600, len(qa_samples) * 76), (255, 255, 255))
    for k, r in enumerate(qa_samples):
        with Image.open(images_dir / r["image_rel"]) as im:
            sheet.paste(im.resize((min(512, im.width), 64)), (12, k * 76 + 6))
    sheet.save(manifests_dir / "v66_qa_contact_sheet.png")

    orient_counts = Counter(r["orientation_flag"] for r in rendered)
    vert_vws = [r["valid_width"] for r in rendered if r["orientation_flag"] == 1]
    audit = {
        "count": len(rendered),
        "elapsed_seconds": round(time.time() - t0, 2),
        "orientation_counts": {str(k): v for k, v in sorted(orient_counts.items())},
        "vertical_max_valid_width": max(vert_vws),
        "vertical_min_valid_width": min(vert_vws),
        "appearance_counts": dict(Counter(r["appearance"] for r in rendered)),
        "bucket_counts": dict(Counter(str(r["bucket"]) for r in rendered)),
        "bg_family_counts": dict(Counter(r["bg_family"] for r in rendered)),
        "bg_tier_counts": dict(Counter(r["bg_tier"] for r in rendered)),
        "polarity_counts": dict(Counter(r["polarity"] for r in rendered)),
        "post_med_cr_ge_45_rate": round(sum(1 for r in rendered if r["post_med_cr"] >= 4.5) / len(rendered), 4),
        "post_p05_cr_ge_30_rate": round(sum(1 for r in rendered if r["post_p05_cr"] >= 3.0) / len(rendered), 4),
        "manifest_sha256": hashlib.sha256(rendered_path.read_bytes()).hexdigest(),
    }
    (manifests_dir / "v66_render_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit


if __name__ == "__main__":
    b_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/content/v66_run")
    print(json.dumps(render_v66_dataset(b_dir), indent=2, ensure_ascii=False))
