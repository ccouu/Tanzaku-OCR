"""Thin CLI adapter for the Python lexicon-trie lookup pipeline.

No duplicate matching logic: delegates directly to `src/recognition/lexicon_lookup/`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recognition.lexicon_lookup import (  # noqa: E402
    EditPolicyConfig,
    LexiconIndex,
    LexiconPipeline,
    LookupConfig,
    ReadOnlyTranslationRepo,
    build_lexicon_index,
)

DEFAULT_WORK_ROOT = ROOT / "outputs/recognition/implementation/lexicon_trie_python_20261119"
DEFAULT_INDEX_PATH = DEFAULT_WORK_ROOT / "artifacts/lexicon_index_v1.json.gz"
DEFAULT_MENU_LEXICON = ROOT / "outputs/recognition/experiments/h4_k41_l2_v06_p5pct_pairs_ep04/payload/menu_lexicon.json"
DEFAULT_JMDICT_CLEANED = ROOT / "data/processed/lexicon_cleaned_argon/jmdict_food_cleaned.json"
DEFAULT_DB_PATH = ROOT / "data/processed/lexicon/culinary_menu.db"
DEFAULT_LM_PATH = ROOT / "checkpoints/l2_v06/char_lm_4g_hp1010.json.gz"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Python lexicon-trie lookup CLI (suggestion-only).")
    p.add_argument("--build-index", action="store_true", help="Build and save versioned lexicon index.")
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH, help="Path to lexicon_index_v1.json.gz")
    p.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Read-only culinary_menu.db path")
    p.add_argument("--lm", type=Path, default=DEFAULT_LM_PATH, help="Schema-v2 HP1010 Char-LM path")
    p.add_argument("--text", type=str, default=None, help="Run pure text lookup on this string")
    p.add_argument("--image", type=Path, default=None, help="Run recognizer + lookup on a local crop image")
    p.add_argument("--orientation", type=int, default=0, choices=(0, 1), help="0=horizontal, 1=vertical")
    p.add_argument("--decoder", type=str, default="greedy", choices=("greedy", "beam4g_dn_r1", "plain_beam"))
    p.add_argument("--no-expanded", action="store_true", help="Disable jmdict_food_cleaned expansion tier")
    p.add_argument("--no-context", action="store_true", help="Disable context_phrase tier")
    p.add_argument("--exact-only", action="store_true", help="Disable all fuzzy edits (0-edit exact only)")
    p.add_argument("--max-edits-6plus", type=int, default=2, choices=(0, 1, 2), help="Max edits for >=6 chars")
    p.add_argument("--no-confusion-weights", action="store_true", help="Use uniform substitution cost 1.0")
    p.add_argument("--device", type=str, default="cpu", help="PyTorch device for --image mode")
    p.add_argument("--json", action="store_true", dest="as_json", help="Print full JSON result")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.build_index:
        idx = build_lexicon_index(
            menu_lexicon_path=DEFAULT_MENU_LEXICON,
            jmdict_cleaned_path=DEFAULT_JMDICT_CLEANED,
            culinary_db_path=args.db,
            include_context_phrases=True,
            include_expanded_culinary=True,
        )
        out = idx.save_gz(args.index)
        print(json.dumps({"saved": str(out), "metadata": dict(idx.metadata)}, ensure_ascii=False, indent=2))
        if args.text is None and args.image is None:
            return 0

    if not args.index.is_file():
        print(f"Index not found at {args.index}; run with --build-index first.", file=sys.stderr)
        return 2

    index = LexiconIndex.load_gz(args.index)
    repo = ReadOnlyTranslationRepo(args.db)

    policy = (
        EditPolicyConfig(len1_max_edits=0, len2_max_edits=0, len3_5_max_edits=0, len6plus_max_edits=0)
        if args.exact_only
        else EditPolicyConfig(len6plus_max_edits=args.max_edits_6plus)
    )
    cfg = LookupConfig(
        edit_policy=policy,
        use_confusion_weights=not args.no_confusion_weights,
        include_context_phrases=not args.no_context,
        include_expanded_culinary=not args.no_expanded,
    )

    if args.image is not None:
        from PIL import Image
        from src.recognition.char_lm_beam import ImmutableCharNGramLM
        from src.recognition.current import load_current_recognizer

        model, saved = load_current_recognizer(device=args.device)
        lm = ImmutableCharNGramLM.load_gz(args.lm, allow_legacy_v1=False) if args.lm.is_file() else None
        model_meta = {
            "version": "L2_v06",
            "architecture": saved.get("active_architecture"),
            "state_key": saved.get("active_state_key"),
        }
        pipe = LexiconPipeline(index, repo, immutable_lm=lm, model=model, model_meta=model_meta, device=args.device)
        img = Image.open(args.image)
        res = pipe.lookup_image(
            img,
            orientation=args.orientation,
            config=cfg,
            decode_mode=args.decoder,
            crop_provenance={"source_file": args.image.name},
        )
    elif args.text is not None:
        pipe = LexiconPipeline(index, repo)
        res = pipe.lookup_text(args.text, config=cfg)
    else:
        print("Provide --text, --image, or --build-index.", file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"Raw OCR       : {res.raw_ocr}")
        print(f"Canonical OCR : {res.canonical_ocr}")
        if res.lm_alternative is not None:
            print(f"LM R1 Alt     : {res.lm_alternative}")
        print(f"Suggested Line: {res.suggested_line} (auto_corrected={res.auto_corrected})")
        print(f"Protected     : {[(p.text, p.kind, p.start, p.end) for p in res.protected_spans]}")
        print("Segments:")
        for seg in res.segments:
            cand = seg.selected_candidate
            if cand is not None:
                en = cand.translation.name_en if (cand.translation and cand.translation.status == "found") else f"[{cand.translation_status}]"
                print(
                    f"  [{seg.start}:{seg.end}] {seg.kind:16s} {seg.text!r} -> {cand.term!r} "
                    f"(edits={cand.hard_edits}, w={cand.weighted_edit_cost}, tier={cand.source_tier}) :: {en}"
                )
            else:
                print(f"  [{seg.start}:{seg.end}] {seg.kind:16s} {seg.text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
