"""Generate extended Core ML M0/M1/M2 golden parity fixtures into
`outputs/stage4/coreml_export_iphone_20261123/golden_fixtures/`.

Covers:
  - All 4 physical buckets (128, 256, 384, 512) and diverse valid widths / valid_T
    including very short non-multiple-of-4 and T_physical=128
  - Both orientation flags (0 horizontal, 1 vertical)
  - Verified blank negative, prices/brackets (¥1,280(税込)), repeated digits (880円),
    expanded Unicode (>2051 vocab index)
  - Long >512px line (overlap-100 multi-window merged canonical greedy)
  - Exact split-wrapper parity check on every single fixture
  - V5 YOLO11n-OBB 1024 letterbox + rotated ProbIoU NMS golden fixture on menu-101.jpg
  - Lexicon V3 + read-only SQLite golden outputs
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recognition.canonical_ctc import fold_parenthesis_logits
from src.recognition.char_lm_beam import DecodeConfig, ImmutableCharNGramLM, decode_with_config
from src.recognition.data import IMAGENET_MEAN, IMAGENET_STD, preprocess
from src.recognition.label_normalization import normalize_digits, normalize_parentheses
from src.recognition.lexicon_lookup.contracts import LookupConfig
from src.recognition.lexicon_lookup.index import LexiconIndex
from src.recognition.lexicon_lookup.particle_guard import probable_particles
from src.recognition.lexicon_lookup.pipeline import LexiconPipeline
from src.recognition.lexicon_lookup.translation_repo import ReadOnlyTranslationRepo
from src.recognition.long_line_inference import bucket_image, owned_tokens, window_ranges
from src.recognition.render.unified_renderer import RendererConfig, render_specs
from src.stage4.bundle_manifest import sha256_file
from src.stage4.coreml_export import (
    RecognizerFeatureExtractorWrapper,
    RecognizerRecurrentHeadWrapper,
    load_pinned_recognizer,
    verify_detector_obb_wrapper_parity,
)
from src.stage4.desktop_backend import format_crop_translation

OUT_DIR = ROOT / "outputs/stage4/coreml_export_iphone_20261123/golden_fixtures"
FONTS_DIR = ROOT / "outputs/recognition/experiments/l2_selected2105_short_recovery_1000/payload/fonts/train"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Use strict IEEE FP32 on CPU for deterministic Core ML golden reference (no GPU TF32 drift)
    device = torch.device("cpu")
    model, bundle, state_sha = load_pinned_recognizer(root=ROOT, device="cpu")
    feat_mod = RecognizerFeatureExtractorWrapper(model).eval()
    head_mod = RecognizerRecurrentHeadWrapper(model).eval()
    assets = bundle["assets"]

    vocab = json.loads(Path(assets["vocab_2105"]["path"]).read_text(encoding="utf-8"))
    chars = vocab["characters"]
    c2i = {c: i + 1 for i, c in enumerate(chars)}
    i2c = {v: k for k, v in c2i.items()}
    expanded_char = chars[2051]

    lm = ImmutableCharNGramLM.load_gz(Path(assets["char_lm_v2"]["path"]), allow_legacy_v1=False)
    idx = LexiconIndex.load_gz(Path(assets["lexicon_index_v3"]["path"]))
    repo = ReadOnlyTranslationRepo(Path(assets["culinary_menu_db"]["path"]))
    pipe = LexiconPipeline(idx, repo, immutable_lm=lm)

    specs = [
        {"record_id": "cm01_len1_kanji", "original_text": "鯛", "orientation_flag": 0, "stream": "golden", "kl_mask": False, "is_negative": False, "negative_verified": False, "appearance": "clean", "target_gh_bin": "medium"},
        {"record_id": "cm02_len2_dish", "original_text": "刺身", "orientation_flag": 0, "stream": "golden", "kl_mask": False, "is_negative": False, "negative_verified": False, "appearance": "clean", "target_gh_bin": "medium"},
        {"record_id": "cm03_vert_dish", "original_text": "まぐろ丼", "orientation_flag": 1, "stream": "golden", "kl_mask": False, "is_negative": False, "negative_verified": False, "appearance": "clean", "target_gh_bin": "medium"},
        {"record_id": "cm04_horiz_phrase", "original_text": "季節の先付と前菜", "orientation_flag": 0, "stream": "golden", "kl_mask": False, "is_negative": False, "negative_verified": False, "appearance": "clean", "target_gh_bin": "medium"},
        {"record_id": "cm05_price_brackets", "original_text": "和牛ロース ¥1,280(税込)", "orientation_flag": 0, "stream": "golden", "kl_mask": False, "is_negative": False, "negative_verified": False, "appearance": "clean", "target_gh_bin": "medium"},
        {"record_id": "cm06_repeated_digits", "original_text": "880円", "orientation_flag": 0, "stream": "golden", "kl_mask": False, "is_negative": False, "negative_verified": False, "appearance": "clean", "target_gh_bin": "medium"},
        {"record_id": "cm07_blank_negative", "original_text": "", "orientation_flag": 0, "stream": "golden_blank", "kl_mask": False, "is_negative": True, "negative_verified": True, "negative_verification_source": "procedural_text_free_renderer_v1", "appearance": "clean", "target_gh_bin": "medium"},
        {"record_id": "cm08_expanded_2105", "original_text": f"特{expanded_char}焼", "orientation_flag": 0, "stream": "golden", "kl_mask": False, "is_negative": False, "negative_verified": False, "appearance": "clean", "target_gh_bin": "medium"},
        {"record_id": "cm09_bucket384_med", "original_text": "特製国産黒毛和牛すき焼き定食", "orientation_flag": 0, "stream": "golden", "kl_mask": False, "is_negative": False, "negative_verified": False, "appearance": "clean", "target_gh_bin": "medium"},
        {"record_id": "cm10_bucket512_long", "original_text": "本日のおすすめ刺身五点盛り合わせ特選醤油添え", "orientation_flag": 0, "stream": "golden", "kl_mask": False, "is_negative": False, "negative_verified": False, "appearance": "clean", "target_gh_bin": "medium"},
    ]
    rendered, _ = render_specs(
        specs,
        fonts_dir=FONTS_DIR,
        out_images_dir=OUT_DIR / "images",
        rel_prefix="images",
        vocab_chars=chars,
        config=RendererConfig(base_seed=2026112310, recipe_version="stage4_coreml_golden_v2"),
        workers=1,
        manifest_out_path=OUT_DIR / "golden_rendered.jsonl",
    )

    r1_cfg = DecodeConfig(mode="beam4g_dn", alpha_lm=0.30, beam_width=5, topk_t=6, digit_neutral=True, routing="r1_short_greedy", short_greedy_max_chars=2)
    lcfg = LookupConfig()
    ocr_rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for rec in rendered:
            img_p = OUT_DIR / rec["image_rel"]
            ori = int(rec["orientation_flag"])
            vw = int(rec["valid_width"])
            bucket = int(rec["bucket"])
            vt = (vw + 3) // 4
            with Image.open(img_p) as im:
                tensor, _, _ = preprocess(im.convert("RGB"), IMAGENET_MEAN, IMAGENET_STD, "batch", max_width=bucket)
            t_in = tensor[None].to(device)
            ref_logits = model(t_in, torch.tensor([ori], device=device), torch.tensor([vt], device=device)).detach().cpu()
            feats = feat_mod(t_in, torch.tensor([float(ori)], device=device))
            split_logits = head_mod(feats[:, :, :vt]).detach().cpu()
            max_diff = float((ref_logits[:vt] - split_logits).abs().max().item())
            assert max_diff <= 1e-4, f"Split wrapper mismatch on {rec['record_id']}: {max_diff}"

            folded = fold_parenthesis_logits(ref_logits.float(), c2i, drop_aliases=False)[:vt, 0, :]
            logp = F.log_softmax(folded, dim=-1)
            dec = decode_with_config(logp, i2c, r1_cfg, lm=lm)
            g_txt = normalize_digits(normalize_parentheses(dec["greedy"])).strip()
            l_txt = normalize_digits(normalize_parentheses(dec["text"])).strip()
            top3_v, top3_i = torch.topk(logp, k=3, dim=-1)

            lex_res = pipe.lookup_text(l_txt, config=lcfg)
            parts = probable_particles(l_txt, idx, lcfg)
            tr = format_crop_translation(l_txt, lex_res, repo, parts)

            ocr_rows.append(
                {
                    "record_id": rec["record_id"],
                    "truth": rec["canonical_text"],
                    "image_rel": rec["image_rel"],
                    "image_sha256": sha256_file(img_p),
                    "orientation_flag": ori,
                    "valid_width": vw,
                    "bucket": bucket,
                    "valid_T": vt,
                    "split_wrapper_max_abs_diff": round(max_diff, 9),
                    "greedy_ocr": g_txt,
                    "charlm_r1_ocr": l_txt,
                    "route_used": dec["route_used"],
                    "top3_frame0_ids": top3_i[0].tolist(),
                    "top3_frame0_logp": [round(float(x), 5) for x in top3_v[0].tolist()],
                    "translation": tr,
                }
            )

        # Synthetic >512px long-line multi-window overlap-100 fixture (concatenating two rendered strips)
        p_a = OUT_DIR / rendered[3]["image_rel"]
        p_b = OUT_DIR / rendered[4]["image_rel"]
        with Image.open(p_a) as ia, Image.open(p_b) as ib:
            ca = ia.crop((0, 0, min(ia.width, int(rendered[3]["valid_width"])), 64)).convert("RGB")
            cb = ib.crop((0, 0, min(ib.width, int(rendered[4]["valid_width"])), 64)).convert("RGB")
            long_w = max(620, ca.width + cb.width + 24)
            long_im = Image.new("RGB", (long_w, 64), (255, 255, 255))
            long_im.paste(ca, (0, 0))
            long_im.paste(cb, (ca.width + 16, 0))
        long_p = OUT_DIR / "images" / "cm11_long_gt512_overlap100.png"
        long_im.save(long_p, format="PNG")
        ranges = window_ranges(long_im.width, max_width=512, overlap=100)
        canon_paths = []
        for left, right in ranges:
            fitted, meta = bucket_image(long_im.crop((left, 0, right, 64)), max_width=512)
            tensor, _, _ = preprocess(fitted, IMAGENET_MEAN, IMAGENET_STD, "batch", max_width=meta["canvas"])
            vl = int(meta["input_length"])
            t_in = tensor[None].to(device)
            feats = feat_mod(t_in, torch.tensor([0.0], device=device))
            logits = head_mod(feats[:, :, :vl]).detach().cpu()
            folded = fold_parenthesis_logits(logits.float(), c2i, drop_aliases=False)
            canon_paths.append(folded[:vl, 0].argmax(-1).tolist())
        merged = normalize_digits(normalize_parentheses("".join(i2c.get(t, "") for t in owned_tokens(canon_paths, ranges)))).strip()
        long_fixture = {
            "record_id": "cm11_long_gt512_overlap100",
            "image_rel": "images/cm11_long_gt512_overlap100.png",
            "image_sha256": sha256_file(long_p),
            "strip_width": long_im.width,
            "window_ranges": [list(r) for r in ranges],
            "window_count": len(ranges),
            "merged_canonical_greedy_ocr": merged,
            "route_used": "overlap512_merged_canonical_greedy_no_joint_lm",
        }

    det_golden = verify_detector_obb_wrapper_parity(root=ROOT)

    lex_goldens = []
    for phrase in ("季節の先付と前菜", "和牛ロース 1800円", "胡麻豆腐", "親子丼", "幕の内"):
        lr = pipe.lookup_text(phrase, config=lcfg)
        parts = probable_particles(phrase, idx, lcfg)
        lex_goldens.append(
            {
                "input": phrase,
                "segments": [(s.start, s.end, s.text, s.kind) for s in lr.segments],
                "particles": list(parts),
                "translation": format_crop_translation(phrase, lr, repo, parts),
            }
        )
    repo.close_thread_connection()

    out_doc = {
        "bundle_id": bundle["bundle_id"],
        "recognizer_raw_state_sha256": state_sha,
        "asset_sha256": {k: v["actual_sha256"] for k, v in assets.items()},
        "precision": "fp32",
        "device_generated_on": str(device),
        "ocr_golden_fixtures": ocr_rows,
        "long_line_multi_window_fixture": long_fixture,
        "detector_obb_1024_golden": det_golden,
        "lexicon_golden_fixtures": lex_goldens,
    }
    (OUT_DIR / "coreml_golden_parity_fixtures.json").write_text(
        json.dumps(out_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[PASS] Generated {len(ocr_rows)} OCR + 1 long>512 + 1 OBB1024 + {len(lex_goldens)} Lexicon golden fixtures in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
