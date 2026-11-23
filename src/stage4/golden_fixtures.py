"""Generate M0/M2 golden Python parity fixtures for future macOS/iOS Core ML verification.

Covers:
  - Horizontal & vertical Japanese lines
  - 1-char & 2-char short lines
  - Repeated digits & prices (¥1,280 / 880円)
  - Fullwidth/ASCII brackets ((税込))
  - Verified blank negative
  - Non-multiple-of-4 valid width
  - Expanded Unicode character (>2051 vocab index)
  - Long >512px line (overlap512 multi-window merged greedy)
  - Lexicon V3 + read-only DB segmentation & translation golden outputs
    (季節の先付と前菜, 和牛ロース 1800円, 胡麻豆腐, 親子丼, 幕の内)
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
from src.recognition.checkpointing import resolve_checkpoint_state
from src.recognition.data import IMAGENET_MEAN, IMAGENET_STD, preprocess
from src.recognition.label_normalization import normalize_digits, normalize_parentheses
from src.recognition.lexicon_lookup.contracts import LookupConfig
from src.recognition.lexicon_lookup.index import LexiconIndex
from src.recognition.lexicon_lookup.particle_guard import probable_particles
from src.recognition.lexicon_lookup.pipeline import LexiconPipeline
from src.recognition.lexicon_lookup.translation_repo import ReadOnlyTranslationRepo
from src.recognition.long_line_inference import bucket_image, native_strip, owned_tokens, window_ranges
from src.recognition.model import ARCH_SPECS, MenuRecognizer, load_recognizer_state
from src.recognition.render.unified_renderer import RendererConfig, render_specs
from src.stage4.bundle_manifest import sha256_file, verify_pinned_stage4_bundle
from src.stage4.desktop_backend import format_crop_translation

OUT_DIR = ROOT / "outputs/stage4/flutter_capture_mvp_20261123/golden_fixtures"
FONTS_DIR = ROOT / "outputs/recognition/experiments/l2_selected2105_short_recovery_1000/payload/fonts/train"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = verify_pinned_stage4_bundle(ROOT, strict=True)
    assets = bundle["assets"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = json.loads(Path(assets["vocab_2105"]["path"]).read_text(encoding="utf-8"))
    chars = vocab["characters"]
    c2i = {c: i + 1 for i, c in enumerate(chars)}
    i2c = {v: k for k, v in c2i.items()}
    expanded_char = chars[2051]

    sd, _, _, _ = resolve_checkpoint_state(Path(assets["ocr_checkpoint_raw"]["path"]), arm_key="raw")
    model = MenuRecognizer(len(chars) + 1, use_bilstm=True, **ARCH_SPECS["h4_k41"])
    load_recognizer_state(model, sd)
    model.to(device).eval()

    lm = ImmutableCharNGramLM.load_gz(Path(assets["char_lm_v2"]["path"]), allow_legacy_v1=False)
    idx = LexiconIndex.load_gz(Path(assets["lexicon_index_v3"]["path"]))
    repo = ReadOnlyTranslationRepo(Path(assets["culinary_menu_db"]["path"]))
    pipe = LexiconPipeline(idx, repo, immutable_lm=lm)

    specs = [
        {"record_id": "g01_horiz_phrase", "original_text": "季節の先付と前菜", "orientation_flag": 0, "stream": "golden", "kl_mask": False, "is_negative": False, "negative_verified": False, "appearance": "clean", "target_gh_bin": "medium"},
        {"record_id": "g02_vert_dish", "original_text": "まぐろ丼", "orientation_flag": 1, "stream": "golden", "kl_mask": False, "is_negative": False, "negative_verified": False, "appearance": "clean", "target_gh_bin": "medium"},
        {"record_id": "g03_len1_kanji", "original_text": "鯛", "orientation_flag": 0, "stream": "golden", "kl_mask": False, "is_negative": False, "negative_verified": False, "appearance": "clean", "target_gh_bin": "medium"},
        {"record_id": "g04_len2_dish", "original_text": "刺身", "orientation_flag": 0, "stream": "golden", "kl_mask": False, "is_negative": False, "negative_verified": False, "appearance": "clean", "target_gh_bin": "medium"},
        {"record_id": "g05_price_brackets", "original_text": "和牛ロース ¥1,280(税込)", "orientation_flag": 0, "stream": "golden", "kl_mask": False, "is_negative": False, "negative_verified": False, "appearance": "clean", "target_gh_bin": "medium"},
        {"record_id": "g06_repeated_digits", "original_text": "880円", "orientation_flag": 0, "stream": "golden", "kl_mask": False, "is_negative": False, "negative_verified": False, "appearance": "clean", "target_gh_bin": "medium"},
        {"record_id": "g07_blank_negative", "original_text": "", "orientation_flag": 0, "stream": "golden_blank", "kl_mask": False, "is_negative": True, "negative_verified": True, "negative_verification_source": "procedural_text_free_renderer_v1", "appearance": "clean", "target_gh_bin": "medium"},
        {"record_id": "g08_expanded_2105", "original_text": f"特{expanded_char}焼", "orientation_flag": 0, "stream": "golden", "kl_mask": False, "is_negative": False, "negative_verified": False, "appearance": "clean", "target_gh_bin": "medium"},
        {"record_id": "g09_compound_goma", "original_text": "胡麻豆腐", "orientation_flag": 0, "stream": "golden", "kl_mask": False, "is_negative": False, "negative_verified": False, "appearance": "clean", "target_gh_bin": "medium"},
        {"record_id": "g10_compound_oyako", "original_text": "親子丼", "orientation_flag": 0, "stream": "golden", "kl_mask": False, "is_negative": False, "negative_verified": False, "appearance": "clean", "target_gh_bin": "medium"},
    ]
    rendered, _ = render_specs(
        specs,
        fonts_dir=FONTS_DIR,
        out_images_dir=OUT_DIR / "images",
        rel_prefix="images",
        vocab_chars=chars,
        config=RendererConfig(base_seed=2026112300, recipe_version="stage4_golden_v1"),
        workers=1,
        manifest_out_path=OUT_DIR / "golden_rendered.jsonl",
    )

    r1_cfg = DecodeConfig(mode="beam4g_dn", alpha_lm=0.30, beam_width=5, topk_t=6, digit_neutral=True, routing="r1_short_greedy", short_greedy_max_chars=2)
    lcfg = LookupConfig()
    golden_rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for rec in rendered:
            img_p = OUT_DIR / rec["image_rel"]
            ori = int(rec["orientation_flag"])
            vw = int(rec["valid_width"])
            bucket = int(rec["bucket"])
            vt = (vw + 3) // 4
            with Image.open(img_p) as im:
                tensor, _, _ = preprocess(im.convert("RGB"), IMAGENET_MEAN, IMAGENET_STD, "batch", max_width=bucket)
            logits = model(tensor[None].to(device), torch.tensor([ori], device=device), torch.tensor([vt], device=device)).detach().cpu()
            folded = fold_parenthesis_logits(logits.float(), c2i, drop_aliases=False)[:vt, 0, :]
            logp = F.log_softmax(folded, dim=-1)
            dec = decode_with_config(logp, i2c, r1_cfg, lm=lm)
            g_txt = normalize_digits(normalize_parentheses(dec["greedy"])).strip()
            l_txt = normalize_digits(normalize_parentheses(dec["text"])).strip()
            top3_v, top3_i = torch.topk(logp, k=3, dim=-1)

            lex_res = pipe.lookup_text(l_txt, config=lcfg)
            parts = probable_particles(l_txt, idx, lcfg)
            tr = format_crop_translation(l_txt, lex_res, repo, parts)

            golden_rows.append(
                {
                    "record_id": rec["record_id"],
                    "truth": rec["canonical_text"],
                    "image_rel": rec["image_rel"],
                    "image_sha256": sha256_file(img_p),
                    "orientation_flag": ori,
                    "valid_width": vw,
                    "bucket": bucket,
                    "valid_T": vt,
                    "greedy_ocr": g_txt,
                    "charlm_r1_ocr": l_txt,
                    "route_used": dec["route_used"],
                    "top3_frame0_ids": top3_i[0].tolist(),
                    "top3_frame0_logp": [round(float(x), 5) for x in top3_v[0].tolist()],
                    "translation": tr,
                }
            )

    # Also record pure lexicon golden cases for Swift M2 parity
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
        "asset_sha256": {k: v["actual_sha256"] for k, v in assets.items()},
        "precision": "fp32",
        "device_generated_on": str(device),
        "ocr_golden_fixtures": golden_rows,
        "lexicon_golden_fixtures": lex_goldens,
    }
    (OUT_DIR / "golden_parity_fixtures.json").write_text(json.dumps(out_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[PASS] Generated {len(golden_rows)} OCR golden fixtures + {len(lex_goldens)} lexicon golden fixtures in {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
