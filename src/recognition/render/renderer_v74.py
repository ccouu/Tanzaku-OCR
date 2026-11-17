"""V74 synthetic replay ProcessPool renderer + combined 4N=6,244 manifest builder."""
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
from PIL import Image, ImageFilter

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.recognition.data import minimum_ctc_steps, output_length
from src.recognition.label_normalization import normalize_digits, normalize_parentheses
from .renderer_v65 import BUCKETS, WEAK_TRAIN_BRUSH_FONTS, _render_mask, check_font_covers, load_font_cmap
from .renderer_v66 import _apply_appearance_and_composite, _render_vertical_mask

PRIORITY_4 = ("海胆", "軟骨", "雲丹", "馬刺")
_G_FONT_PATHS: list[str] = []
_G_FONT_CMAPS: dict[str, set[int]] = {}
_G_OUT_DIR: Path = Path(".")


def _init_worker(font_paths: list[str], font_cmaps: dict[str, set[int]], out_dir: str):
    global _G_FONT_PATHS, _G_FONT_CMAPS, _G_OUT_DIR
    _G_FONT_PATHS = font_paths
    _G_FONT_CMAPS = font_cmaps
    _G_OUT_DIR = Path(out_dir)


def _sample_target_gh(bin_name: str, rng: random.Random) -> int:
    if bin_name == "small":
        return rng.randint(28, 37)
    if bin_name == "medium":
        return rng.randint(38, 47)
    if bin_name == "large":
        return rng.randint(48, 56)
    return rng.randint(36, 54)


def _render_one_v74(task: tuple) -> dict:
    idx, rec = task
    seed_val = int(rec["render_seed"])
    rng = random.Random(seed_val)
    np_rng = np.random.default_rng(seed_val)

    ori_flag = int(rec["orientation_flag"])
    orig_text = rec["original_text"]
    canon_text = rec["canonical_text"]
    appearance = rec["appearance"]
    bg_family = rec["bg_family"]
    bg_tier = rec["bg_tier"]
    target_gh = _sample_target_gh(rec.get("target_gh_bin", "replay"), rng)
    min_steps = minimum_ctc_steps(canon_text)

    eligible_fonts = [f for f in _G_FONT_PATHS if check_font_covers(f, orig_text, _G_FONT_CMAPS)]
    if not eligible_fonts:
        raise RuntimeError(f"No font covers text: {orig_text!r}")

    font_override = rec.get("font_override")
    if font_override:
        matches = [f for f in eligible_fonts if Path(f).name == font_override]
        if matches:
            font_path = matches[0]
        else:
            font_path = rng.choice(eligible_fonts)
    else:
        if rec["slot"] in ("short_1char", "short_2char", "price"):
            non_weak = [f for f in eligible_fonts if Path(f).name not in WEAK_TRAIN_BRUSH_FONTS]
            if non_weak and rng.random() < 0.80:
                eligible_fonts = non_weak
        font_path = rng.choice(eligible_fonts)

    if ori_flag == 1:
        step = min(46, max(28, int(round(460.0 / max(1, len(orig_text))))))
        mask_arr = _render_vertical_mask(orig_text, font_path, step, _G_FONT_CMAPS[font_path])
        if appearance == "dilate":
            mask_arr = np.asarray(Image.fromarray(mask_arr).filter(ImageFilter.MaxFilter(3)))
        elif appearance == "erode" and min(mask_arr.shape) >= 28:
            eroded = np.asarray(Image.fromarray(mask_arr).filter(ImageFilter.MinFilter(3)))
            if int((eroded > 128).sum()) >= 16:
                mask_arr = eroded
        rot_mask = np.asarray(Image.fromarray(mask_arr).transpose(Image.Transpose.ROTATE_90))
        rh, rw = rot_mask.shape
        target_h = min(56, max(28, target_gh))
        scale = min(target_h / float(max(1, rh)), 488.0 / float(max(1, rw)))
        scaled_w = max(4, int(round(rw * scale)))
        scaled_h = max(4, min(58, int(round(rh * scale))))
        alpha_im = Image.fromarray(rot_mask).resize((scaled_w, scaled_h), Image.Resampling.BICUBIC)
        y_off = max(2, min(64 - scaled_h - 2, (64 - scaled_h) // 2 + rng.randint(-2, 2)))
        alpha_64 = np.zeros((64, scaled_w), dtype=np.uint8)
        alpha_64[y_off : y_off + scaled_h, :] = np.asarray(alpha_im)
        if int((alpha_64 > 64).sum()) < 8 and int(alpha_64.max()) > 0:
            alpha_64 = np.clip(alpha_64.astype(np.float32) * (220.0 / float(alpha_64.max())), 0, 255).astype(np.uint8)
        font_size = step
    else:
        font_size = rng.randint(28, 48)
        mask_arr = _render_mask(orig_text, font_path, font_size)
        if appearance == "dilate":
            mask_arr = np.asarray(Image.fromarray(mask_arr).filter(ImageFilter.MaxFilter(3)))
        elif appearance == "erode" and font_size >= 26:
            eroded = np.asarray(Image.fromarray(mask_arr).filter(ImageFilter.MinFilter(3)))
            if int((eroded > 128).sum()) >= 16:
                mask_arr = eroded
        mh, mw = mask_arr.shape
        scale = min(target_gh / float(max(1, mh)), 488.0 / float(max(1, mw)))
        scaled_w = max(4, int(round(mw * scale)))
        scaled_h = max(4, min(58, int(round(mh * scale))))
        alpha_im = Image.fromarray(mask_arr).resize((scaled_w, scaled_h), Image.Resampling.BICUBIC)
        y_off = max(2, min(64 - scaled_h - 2, (64 - scaled_h) // 2 + rng.randint(-3, 3)))
        alpha_64 = np.zeros((64, scaled_w), dtype=np.uint8)
        alpha_64[y_off : y_off + scaled_h, :] = np.asarray(alpha_im)
        if int((alpha_64 > 64).sum()) < 8 and int(alpha_64.max()) > 0:
            alpha_64 = np.clip(alpha_64.astype(np.float32) * (220.0 / float(alpha_64.max())), 0, 255).astype(np.uint8)

    ys_a, xs_a = np.where(alpha_64 > 16)
    actual_gh = int(ys_a.max() - ys_a.min() + 1) if len(ys_a) else target_gh
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
    strip, polarity, _, _, _, _ = _apply_appearance_and_composite(
        alpha_64, scaled_w, left_margin, valid_width, appearance, bg_family, bg_tier, is_thin, rng, np_rng
    )
    if valid_width < bucket:
        strip = np.pad(strip, ((0, 0), (0, bucket - valid_width), (0, 0)), mode="edge")

    sub_dir = _G_OUT_DIR / "replay" / f"{idx // 1000:03d}"
    sub_dir.mkdir(parents=True, exist_ok=True)
    rel_path = f"replay/{idx // 1000:03d}/{idx:06d}.png"
    Image.fromarray(strip).save(_G_OUT_DIR / rel_path, compress_level=1)

    return {
        "id": f"rep_{idx:06d}",
        "source_kind": "synthetic_replay",
        "slot": rec["slot"],
        "original_text": orig_text,
        "canonical_text": canon_text,
        "image_rel": rel_path,
        "font": Path(font_path).name,
        "appearance": appearance,
        "bg_family": bg_family,
        "bg_tier": bg_tier,
        "actual_gh_bin": actual_bin,
        "valid_width": int(valid_width),
        "valid_T": int(output_length(valid_width)),
        "bucket": int(bucket),
        "orientation_flag": ori_flag,
    }


def render_v74_replay_and_combine(
    base_dir: Path,
    workers: int | None = None,
    manifest_name: str = "v74_replay_corpus.jsonl",
    expected_replay_count: int | None = 4683,
    combine_with_yolo: bool = True,
) -> dict:
    t0 = time.time()
    payload_dir = base_dir / "payload"
    manifests_dir = base_dir / "manifests"
    images_dir = base_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    train_fonts = sorted(
        str(p) for p in (payload_dir / "fonts" / "train").iterdir()
        if p.is_file() and p.suffix.lower() in (".ttf", ".otf", ".ttc")
    )
    assert len(train_fonts) == 15
    vocab = json.loads((payload_dir / "vocab.json").read_text(encoding="utf-8"))
    vocab_chars = vocab["characters"] + list("０１２３４５６７８９")
    cmaps = {fp: load_font_cmap(fp, vocab_chars) for fp in train_fonts}

    recs = [json.loads(l) for l in (manifests_dir / manifest_name).read_text(encoding="utf-8").splitlines() if l.strip()]
    if expected_replay_count is not None:
        assert len(recs) == expected_replay_count, f"Expected {expected_replay_count}, got {len(recs)}"

    workers = workers or min(8, max(2, os.cpu_count() or 4))
    tasks = [(int(r["id"]), r) for r in recs]
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(train_fonts, cmaps, str(images_dir))) as ex:
        for res in ex.map(_render_one_v74, tasks, chunksize=100):
            results.append(res)

    replay_rendered_path = images_dir / "v74_replay_rendered.jsonl"
    replay_rendered_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n", encoding="utf-8", newline="\n")

    # QA sheet (priority 4 + sample slots)
    qa = []
    for pt in PRIORITY_4:
        for ori in (0, 1):
            m = next((r for r in results if r["canonical_text"] == pt and int(r["orientation_flag"]) == ori), None)
            if m:
                qa.append(m)
    if qa:
        sheet = Image.new("RGB", (512, 64 * len(qa)), (24, 24, 24))
        for i, r in enumerate(qa):
            sheet.paste(Image.open(images_dir / r["image_rel"]).convert("RGB"), (0, i * 64))
        sheet.save(manifests_dir / "v74_qa_contact_sheet.png")

    combined_count = len(results)
    if combine_with_yolo:
        yolo_train = [json.loads(l) for l in (manifests_dir / "yolo_train_crops.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        combined = []
        for r in yolo_train:
            img_p = images_dir / r["ctc_crop"]
            assert img_p.exists(), f"Missing uploaded YOLO train crop: {img_p}"
            combined.append({
                "id": r["crop_id"],
                "source_kind": "yolo_train",
                "slot": "yolo_real",
                "original_text": r["original_text"],
                "canonical_text": r["canonical_text"],
                "image_rel": r["ctc_crop"],
                "valid_width": int(r["valid_width"]),
                "valid_T": int(r["valid_T"]),
                "bucket": int(r["bucket"]),
                "orientation_flag": int(r["orientation_flag"]),
            })
        combined.extend(results)
        combined_count = len(combined)

        # Post-render 521/521 + priority-4 verification
        lex = json.loads((payload_dir / "menu_lexicon.json").read_text(encoding="utf-8"))
        set_521 = {normalize_digits(normalize_parentheses(t)) for t in lex["terms_521"]}
        replay_canons = {r["canonical_text"] for r in results}
        assert set_521 <= replay_canons, f"Post-render missing 521 terms: {sorted(set_521 - replay_canons)}"
        for pt in PRIORITY_4:
            sub = [r for r in results if r["canonical_text"] == pt]
            assert len(sub) >= 4
            assert sum(1 for r in sub if r["orientation_flag"] == 0) >= 2
            assert sum(1 for r in sub if r["orientation_flag"] == 1) >= 2
            assert len({r["font"] for r in sub}) >= 2

        comb_path = images_dir / "v74_epoch_combined.jsonl"
        comb_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in combined) + "\n", encoding="utf-8", newline="\n")

    audit = {
        "replay_count": len(results),
        "combined_count": combined_count,
        "elapsed_seconds": round(time.time() - t0, 2),
        "replay_orientation_counts": dict(Counter(str(r["orientation_flag"]) for r in results)),
        "replay_appearance_counts": dict(Counter(r["appearance"] for r in results)),
        "replay_manifest_sha256": hashlib.sha256(replay_rendered_path.read_bytes()).hexdigest(),
    }
    (manifests_dir / "v74_render_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit
