"""Reusable post-parse CTC label character inventory & append-only vocabulary candidate builder.

Follows `docs/data/yolo_crop_canvas_pipeline.md` (Steps 4.0 & 4.2):
- Reads complete validated CTC transcriptions from `ctc_labels.txt` (NOT LLM-extracted terms).
- Explicitly detects real TAB vs literal `\\t` delimiter format and rejects malformed lines.
- Cross-checks every exported crop/text against `labeled_crops.jsonl`, `canvas_manifest.json`,
  and `vlm_responses.txt` bijection.
- Applies ONLY the repository's narrow canonicalization policy (`normalize_digits` +
  `normalize_parentheses`), preserving raw transcriptions and never silently deleting characters.
- Extends a verified append-only parent vocabulary via `src.recognition.vocabulary.extend_vocabulary`,
  preserving `blank_id=0` and every existing `(char, id)` mapping.
- Scans the actual training font binaries (`.ttf`/`.otf`/`.ttc` only) with `fontTools` to report
  glyph coverage and flag any zero-font-support or whitespace/non-printable review issues.
- Emits an auditable JSON report; NEVER overwrites frozen base/parent vocabs, checkpoints, LM,
  trie, or DB, and NEVER activates the candidate vocabulary.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recognition.checkpointing import sha256_file
from src.recognition.label_normalization import normalize_digits, normalize_parentheses
from src.recognition.vocabulary import extend_vocabulary


def canonicalize_narrow(raw_text: str) -> str:
    """Narrow repository policy: full-width decimal digits -> ASCII and full-width parens -> ASCII.

    Never applies broad NFKC, confusable rewrites, or silent character deletion.
    """
    return normalize_parentheses(normalize_digits(raw_text))


def parse_ctc_labels_file(path: Path) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    """Parse `ctc_labels.txt` with explicit delimiter detection (real TAB vs literal `\\t`)."""
    raw_bytes = path.read_bytes()
    has_crlf = b"\r\n" in raw_bytes
    text = raw_bytes.decode("utf-8")
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"Empty CTC labels file: {path}")

    real_tab_lines = sum(1 for l in lines if "\t" in l)
    lit_tab_lines = sum(1 for l in lines if "\\t" in l and "\t" not in l)

    if real_tab_lines == len(lines):
        delimiter_mode = "real_tab"
        delim = "\t"
    elif lit_tab_lines == len(lines):
        delimiter_mode = "literal_backslash_t"
        delim = "\\t"
    else:
        raise ValueError(
            f"Mixed or missing delimiter in {path}: total={len(lines)}, "
            f"real_tab={real_tab_lines}, literal_backslash_t={lit_tab_lines}"
        )

    parsed: list[tuple[str, str]] = []
    for idx, line in enumerate(lines, start=1):
        parts = line.split(delim)
        if len(parts) != 2:
            raise ValueError(f"Malformed line {idx} in {path}: expected 2 fields, got {len(parts)} ({line!r})")
        img_rel, label_text = parts
        if not img_rel or not img_rel.endswith((".png", ".jpg", ".jpeg", ".webp")):
            raise ValueError(f"Invalid image path at line {idx} in {path}: {img_rel!r}")
        if not label_text:
            raise ValueError(f"Empty transcription at line {idx} in {path} for {img_rel!r}")
        if "\r" in label_text or "\n" in label_text or "\t" in label_text:
            raise ValueError(f"Control separator leaked into transcription at line {idx}: {label_text!r}")
        parsed.append((img_rel, label_text))

    meta = {
        "line_count": len(lines),
        "has_crlf": has_crlf,
        "delimiter_mode": delimiter_mode,
        "real_tab_lines": real_tab_lines,
        "literal_backslash_t_lines": lit_tab_lines,
    }
    return parsed, meta


def verify_bijection_and_cross_check(
    yc_dir: Path,
    ctc_rows: list[tuple[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify vlm_responses <-> canvas_manifest bijection and 1:1 match with ctc_labels.txt."""
    labeled_path = yc_dir / "labeled_crops.jsonl"
    canvas_path = yc_dir / "canvas_manifest.json"
    resp_path = yc_dir / "vlm_responses.txt"

    labeled = [json.loads(l) for l in labeled_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    canvas_m = json.loads(canvas_path.read_text(encoding="utf-8"))

    # 1. Response <-> canvas manifest bijection
    resp_lines = resp_path.read_text(encoding="utf-8").splitlines()
    cur_canvas = None
    resp_slots: list[tuple[str, int]] = []
    for line in resp_lines:
        m = re.match(r"^===\s*(canvas_\d+)\.png\s*===$", line.strip())
        if m:
            cur_canvas = m.group(1)
            continue
        m2 = re.match(r"^\[(\d+)\]\s*\|", line.strip())
        if m2 and cur_canvas:
            resp_slots.append((cur_canvas, int(m2.group(1))))

    expected_slots = [
        (c["canvas_id"], int(s["slot_index"]))
        for c in canvas_m["canvases"]
        for s in c["slots"]
    ]
    if not (
        len(resp_slots) == len(expected_slots) == len(set(resp_slots)) == len(labeled)
        and set(resp_slots) == set(expected_slots)
    ):
        raise ValueError(
            f"Bijection check failed: resp={len(resp_slots)}, uniq_resp={len(set(resp_slots))}, "
            f"expected={len(expected_slots)}, labeled={len(labeled)}"
        )

    # 2. Cross-check ctc_labels.txt against OK rows in labeled_crops.jsonl
    ok_by_ctc = {}
    status_counts: Counter[str] = Counter()
    for r in labeled:
        status_counts[str(r.get("status", ""))] += 1
        if r.get("status") == "OK":
            key = str(r["ctc_crop"])
            if key in ok_by_ctc:
                raise ValueError(f"Duplicate ctc_crop key in labeled_crops.jsonl: {key}")
            ok_by_ctc[key] = r

    if len(ctc_rows) != len(ok_by_ctc):
        raise ValueError(f"ctc_labels.txt count ({len(ctc_rows)}) != OK labeled_crops count ({len(ok_by_ctc)})")

    joined_ok: list[dict[str, Any]] = []
    for img_rel, ctc_text in ctc_rows:
        if img_rel not in ok_by_ctc:
            raise ValueError(f"ctc_labels.txt crop not found among OK labeled_crops: {img_rel}")
        rec = ok_by_ctc[img_rel]
        if str(rec.get("text", "")) != ctc_text:
            raise ValueError(
                f"Text mismatch for {img_rel}: ctc_labels={ctc_text!r} vs labeled_crops={rec.get('text')!r}"
            )
        joined_ok.append(rec)

    summary = {
        "bijection_verified": True,
        "total_manifest_slots": len(expected_slots),
        "status_counts": dict(status_counts),
        "exported_ok_cross_check_exact_match": True,
        "exported_ok_count": len(joined_ok),
    }
    return joined_ok, summary


def scan_font_binaries(fonts_dir: Path) -> tuple[list[dict[str, Any]], dict[str, set[int]]]:
    """Scan actual `.ttf`/`.otf`/`.ttc` binaries (explicitly handling `.ttc` collections)."""
    from fontTools.ttLib import TTCollection, TTFont

    font_files = sorted(
        p for p in fonts_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".ttf", ".otf", ".ttc")
    )
    font_meta: list[dict[str, Any]] = []
    font_cmaps: dict[str, set[int]] = {}
    for fp in font_files:
        cps: set[int] = set()
        if fp.suffix.lower() == ".ttc":
            ttc = TTCollection(str(fp))
            for f in ttc.fonts:
                cmap = f.getBestCmap() or {}
                cps.update(cmap.keys())
            face_count = len(ttc.fonts)
        else:
            tt = TTFont(str(fp))
            cmap = tt.getBestCmap() or {}
            cps.update(cmap.keys())
            tt.close()
            face_count = 1
        font_cmaps[fp.name] = cps
        font_meta.append({
            "filename": fp.name,
            "sha256": sha256_file(fp),
            "bytes": fp.stat().st_size,
            "faces": face_count,
            "glyph_codepoint_count": len(cps),
        })
    return font_meta, font_cmaps


def build_vocab_candidate(
    *,
    yc_dir: Path,
    base_vocab_path: Path,
    parent_vocab_path: Path,
    out_vocab_path: Path,
    out_dir: Path,
    fonts_dir: Path,
    split_audit_path: Path | None = None,
    batch_id: str = "tabelog_selected_menu_manual/16112026",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_vocab_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load & verify base (1779) and append-only parent (1830) vocabularies
    base_vocab = json.loads(base_vocab_path.read_text(encoding="utf-8"))
    parent_vocab = json.loads(parent_vocab_path.read_text(encoding="utf-8"))
    base_c2i: dict[str, int] = base_vocab["char_to_id"]
    parent_c2i: dict[str, int] = parent_vocab["char_to_id"]

    assert base_vocab.get("blank_id", 0) == 0, "Base vocab blank_id must be 0"
    assert parent_vocab.get("blank_id", 0) == 0, "Parent vocab blank_id must be 0"
    assert set(base_c2i.values()) == set(range(1, len(base_c2i) + 1)), "Base vocab IDs not contiguous"
    assert set(parent_c2i.values()) == set(range(1, len(parent_c2i) + 1)), "Parent vocab IDs not contiguous"
    for ch, cid in base_c2i.items():
        if parent_c2i.get(ch) != cid:
            raise ValueError(f"Parent vocab drifted from base vocab for {ch!r}: base={cid}, parent={parent_c2i.get(ch)}")

    # 2. Parse ctc_labels.txt and cross-check labeled_crops / canvas_manifest / vlm_responses
    ctc_path = yc_dir / "ctc_labels.txt"
    ctc_rows, delimiter_meta = parse_ctc_labels_file(ctc_path)
    joined_ok, cross_check = verify_bijection_and_cross_check(yc_dir, ctc_rows)

    # 3. Optional split mapping (retain existing train/val/test membership)
    menu_to_split: dict[str, str] = {}
    quarantine_oov_crop_ids: set[str] = set()
    if split_audit_path is not None and split_audit_path.is_file():
        sa = json.loads(split_audit_path.read_text(encoding="utf-8"))
        for m in sa.get("train_menus", ()):
            menu_to_split[m] = "train"
        for m in sa.get("val_menus", ()):
            menu_to_split[m] = "val"
        for m in sa.get("test_menus", ()):
            menu_to_split[m] = "test"
        q_path = split_audit_path.parent / "new_tabelog_train_quarantine.jsonl"
        if q_path.is_file():
            for l in q_path.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    qr = json.loads(l)
                    if qr.get("quarantine_reason") == "oov_1779":
                        quarantine_oov_crop_ids.add(str(qr.get("raw_crop_id") or qr["crop_id"]))

    # 4. Inventory every character in complete canonical CTC labels
    raw_char_freq: Counter[str] = Counter()
    canon_char_freq: Counter[str] = Counter()
    canon_crop_freq: Counter[str] = Counter()
    canon_menus: dict[str, set[str]] = defaultdict(set)
    canon_splits: dict[str, set[str]] = defaultdict(set)
    canon_examples: dict[str, list[dict[str, str]]] = defaultdict(list)

    norm_changes: Counter[str] = Counter()
    norm_changed_crops = 0
    whitespace_audit: Counter[str] = Counter()
    nonprintable_audit: list[dict[str, Any]] = []

    for rec in joined_ok:
        raw_t = str(rec["text"])
        canon_t = canonicalize_narrow(raw_t)
        menu = str(rec["source_menu"])
        cid = str(rec["crop_id"])
        sp = menu_to_split.get(menu, "unknown")

        if raw_t != canon_t:
            norm_changed_crops += 1
            for rc, cc in zip(raw_t, canon_t):
                if rc != cc:
                    norm_changes[f"{rc} (U+{ord(rc):04X}) -> {cc} (U+{ord(cc):04X})"] += 1

        for ch in raw_t:
            raw_char_freq[ch] += 1

        seen_in_crop: set[str] = set()
        for ch in canon_t:
            canon_char_freq[ch] += 1
            if ch.isspace():
                whitespace_audit[f"{ch!r} (U+{ord(ch):04X} {unicodedata.name(ch, 'UNKNOWN')})"] += 1
            if not ch.isprintable() and not ch.isspace():
                nonprintable_audit.append({"crop_id": cid, "char": repr(ch), "codepoint": f"U+{ord(ch):04X}"})
            if ch not in seen_in_crop:
                seen_in_crop.add(ch)
                canon_crop_freq[ch] += 1
                canon_menus[ch].add(menu)
                canon_splits[ch].add(sp)
                if len(canon_examples[ch]) < 4:
                    canon_examples[ch].append({
                        "crop_id": cid,
                        "menu": menu,
                        "split": sp,
                        "raw_text": raw_t,
                        "canonical_text": canon_t,
                    })

    # 5. Append-only vocabulary extension on top of parent_vocab (1830 -> new)
    ext_vocab, newly_added = extend_vocabulary(parent_vocab, canon_char_freq.keys())
    ext_c2i: dict[str, int] = ext_vocab["char_to_id"]

    # Strict invariants
    assert ext_vocab["blank_id"] == 0
    assert set(ext_c2i.values()) == set(range(1, len(ext_c2i) + 1))
    assert len(ext_vocab["characters"]) == len(ext_c2i)
    assert all(len(c) == 1 for c in ext_c2i.keys()), "Every token must be exactly one Unicode codepoint"
    for ch, cid in base_c2i.items():
        assert ext_c2i[ch] == cid, f"Base ID drift for {ch!r}"
    for ch, cid in parent_c2i.items():
        assert ext_c2i[ch] == cid, f"Parent ID drift for {ch!r}"
    assert newly_added == sorted(newly_added, key=ord), "Newly added tokens must be sorted by ord"

    # Enrich metadata cleanly without changing char_to_id / characters / blank_id
    ext_vocab["extension"] = {
        "policy": "Append only; old IDs (base 1..1779 and parent 1780..1830) and blank0 preserved",
        "source_batch": batch_id,
        "provenance_scope": "label_informed_all_splits",
        "activation_status": "candidate_only_not_recognition_ready",
        "trained_base_vocab": str(base_vocab_path.relative_to(ROOT).as_posix()),
        "trained_base_vocab_sha256": sha256_file(base_vocab_path),
        "trained_base_size": len(base_c2i),
        "parent_candidate_vocab": str(parent_vocab_path.relative_to(ROOT).as_posix()),
        "parent_candidate_vocab_sha256": sha256_file(parent_vocab_path),
        "parent_candidate_size": len(parent_c2i),
        "normalization_policy": "narrow_fullwidth_digits_and_parens_only",
        "added_characters": newly_added,
        "added_count": len(newly_added),
        "total_size": len(ext_c2i),
    }
    out_vocab_path.write_text(json.dumps(ext_vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 6. Scan actual 15 training font binaries for glyph support
    font_meta, font_cmaps = scan_font_binaries(fonts_dir)
    n_fonts = len(font_meta)
    zero_font_support: list[dict[str, Any]] = []
    partial_font_support: list[dict[str, Any]] = []

    added_details: list[dict[str, Any]] = []
    heldout_only_tokens: list[str] = []
    for ch in newly_added:
        cp = ord(ch)
        supported = [f["filename"] for f in font_meta if cp in font_cmaps[f["filename"]]]
        missing_f = [f["filename"] for f in font_meta if cp not in font_cmaps[f["filename"]]]
        sp_set = sorted(canon_splits[ch])
        is_ho_only = bool(sp_set) and all(s in ("val", "test") for s in sp_set)
        if is_ho_only:
            heldout_only_tokens.append(ch)
        entry = {
            "id": ext_c2i[ch],
            "char": ch,
            "codepoint": f"U+{cp:04X}",
            "unicode_name": unicodedata.name(ch, "UNKNOWN"),
            "unicode_category": unicodedata.category(ch),
            "total_char_freq": canon_char_freq[ch],
            "crop_freq": canon_crop_freq[ch],
            "menu_freq": len(canon_menus[ch]),
            "splits_seen": sp_set,
            "heldout_only": is_ho_only,
            "font_support_count": len(supported),
            "total_train_fonts": n_fonts,
            "missing_in_fonts": missing_f,
            "examples": canon_examples[ch],
        }
        added_details.append(entry)
        if len(supported) == 0:
            zero_font_support.append(entry)
        elif len(supported) < n_fonts:
            partial_font_support.append(entry)

    # 7. Compute OOV before/after overall, by split, and on primary 280 TRAIN OOV quarantine
    base_set = set(base_c2i.keys())
    parent_set = set(parent_c2i.keys())
    new_set = set(ext_c2i.keys())

    def _oov_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        oov_base = oov_parent = oov_new = 0
        c_base: set[str] = set()
        c_parent: set[str] = set()
        c_new: set[str] = set()
        for r in rows:
            ct = canonicalize_narrow(str(r["text"]))
            mb = [c for c in ct if c not in base_set]
            mp = [c for c in ct if c not in parent_set]
            mn = [c for c in ct if c not in new_set]
            if mb:
                oov_base += 1; c_base.update(mb)
            if mp:
                oov_parent += 1; c_parent.update(mp)
            if mn:
                oov_new += 1; c_new.update(mn)
        return {
            "total_crops": len(rows),
            "oov_base_1779": {"crops": oov_base, "unique_chars": len(c_base)},
            "oov_parent_1830": {"crops": oov_parent, "unique_chars": len(c_parent)},
            "oov_candidate_new": {"crops": oov_new, "unique_chars": len(c_new)},
        }

    by_split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    q280_rows: list[dict[str, Any]] = []
    for r in joined_ok:
        sp = menu_to_split.get(str(r["source_menu"]), "unknown")
        by_split_rows[sp].append(r)
        if str(r["crop_id"]) in quarantine_oov_crop_ids:
            q280_rows.append(r)

    oov_report = {
        "overall_all_ok": _oov_stats(joined_ok),
        "by_split": {sp: _oov_stats(rows) for sp, rows in sorted(by_split_rows.items())},
        "primary_train_oov_quarantine_280": _oov_stats(q280_rows) if q280_rows else None,
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id,
        "provenance_scope": "label_informed_all_splits",
        "activation_status": "candidate_only_not_recognition_ready",
        "source_hashes": {
            "ctc_labels_txt": {"path": str(ctc_path.relative_to(ROOT).as_posix()), "sha256": sha256_file(ctc_path)},
            "labeled_crops_jsonl": {"path": str((yc_dir / "labeled_crops.jsonl").relative_to(ROOT).as_posix()), "sha256": sha256_file(yc_dir / "labeled_crops.jsonl")},
            "canvas_manifest_json": {"path": str((yc_dir / "canvas_manifest.json").relative_to(ROOT).as_posix()), "sha256": sha256_file(yc_dir / "canvas_manifest.json")},
            "vlm_responses_txt": {"path": str((yc_dir / "vlm_responses.txt").relative_to(ROOT).as_posix()), "sha256": sha256_file(yc_dir / "vlm_responses.txt")},
            "trained_base_vocab": {"path": str(base_vocab_path.relative_to(ROOT).as_posix()), "sha256": sha256_file(base_vocab_path), "size": len(base_c2i)},
            "parent_candidate_vocab": {"path": str(parent_vocab_path.relative_to(ROOT).as_posix()), "sha256": sha256_file(parent_vocab_path), "size": len(parent_c2i)},
            "new_candidate_vocab": {"path": str(out_vocab_path.relative_to(ROOT).as_posix()), "sha256": sha256_file(out_vocab_path), "size": len(ext_c2i)},
        },
        "delimiter_audit": delimiter_meta,
        "cross_check": cross_check,
        "normalization_audit": {
            "policy": "narrow_fullwidth_digits_and_parens_only (no NFKC, no confusable rewrite, no silent deletion)",
            "crops_modified_by_canonicalization": norm_changed_crops,
            "character_substitutions": dict(norm_changes),
            "whitespace_policy": "Original ASCII space (U+0020, already in base vocab ID 1723) retained verbatim; non-ASCII whitespace or non-printable chars flagged, never silently deleted.",
            "whitespace_counts": dict(whitespace_audit),
            "nonprintable_issues": nonprintable_audit,
        },
        "vocab_size_progression": {
            "trained_base": len(base_c2i),
            "parent_candidate": len(parent_c2i),
            "new_candidate": len(ext_c2i),
            "appended_over_parent": len(newly_added),
            "appended_over_base": len(ext_c2i) - len(base_c2i),
            "heldout_only_new_tokens_count": len(heldout_only_tokens),
            "heldout_only_new_tokens": heldout_only_tokens,
        },
        "oov_before_after": oov_report,
        "font_coverage_summary": {
            "fonts_dir": str(fonts_dir.relative_to(ROOT).as_posix()),
            "actual_font_binaries_count": n_fonts,
            "all_fonts_supported_count": len(newly_added) - len(partial_font_support) - len(zero_font_support),
            "partial_font_support_count": len(partial_font_support),
            "zero_font_support_count": len(zero_font_support),
            "zero_font_support_tokens": [{"id": e["id"], "char": e["char"], "codepoint": e["codepoint"], "unicode_name": e["unicode_name"]} for e in zero_font_support],
            "partial_font_support_tokens": [{"id": e["id"], "char": e["char"], "codepoint": e["codepoint"], "support": f"{e['font_support_count']}/{n_fonts}", "missing_in_fonts": e["missing_in_fonts"]} for e in partial_font_support],
            "font_binaries": font_meta,
        },
        "newly_added_tokens": added_details,
        "activation_and_training_implications": {
            "existing_classifier_outputs": len(base_c2i) + 1,
            "required_classifier_outputs_for_candidate": len(ext_c2i) + 1,
            "is_recognition_ready": False,
            "leakage_disclosure": (
                "Candidate vocabulary is label-informed across all splits (train/val/test) per explicit user request "
                "for full batch CTC-label coverage. Retain current train/val/test split membership, never use held-out "
                "transcriptions to generate training examples or LM counts, and do not report later eval on this batch "
                "as an untouched blind test."
            ),
            "required_next_steps_before_activation": [
                "Authorize ExtendedCTCHead classifier expansion (1780 -> 1902 output rows) with below-blank initialization.",
                "Train expanded rows with synthetic/real coverage for all added tokens (addressing any partial/zero font support).",
                "Rebuild compatible forward/backward Char-LM against the same 1901-char vocabulary if LM decoding is used.",
                "Pass selection & retention evaluation before any registry or inspector switch.",
            ],
        },
    }

    audit_path = out_dir / "vocab_candidate_audit.json"
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Build append-only CTC vocabulary candidate from complete ctc_labels.txt")
    ap.add_argument("--yolo-crops-dir", required=True)
    ap.add_argument("--base-vocab", default="checkpoints/l2_v06/vocab.json")
    ap.add_argument("--parent-vocab", required=True)
    ap.add_argument("--out-vocab", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fonts-dir", default="outputs/recognition/experiments/h4_k41_l2_v06_p5pct_pairs_ep04/payload/fonts/train")
    ap.add_argument("--split-audit", default=None)
    ap.add_argument("--batch-id", default="tabelog_selected_menu_manual/16112026")
    args = ap.parse_args()

    rep = build_vocab_candidate(
        yc_dir=Path(args.yolo_crops_dir).resolve(),
        base_vocab_path=(ROOT / args.base_vocab).resolve() if not Path(args.base_vocab).is_absolute() else Path(args.base_vocab),
        parent_vocab_path=(ROOT / args.parent_vocab).resolve() if not Path(args.parent_vocab).is_absolute() else Path(args.parent_vocab),
        out_vocab_path=(ROOT / args.out_vocab).resolve() if not Path(args.out_vocab).is_absolute() else Path(args.out_vocab),
        out_dir=(ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir),
        fonts_dir=(ROOT / args.fonts_dir).resolve() if not Path(args.fonts_dir).is_absolute() else Path(args.fonts_dir),
        split_audit_path=((ROOT / args.split_audit).resolve() if not Path(args.split_audit).is_absolute() else Path(args.split_audit)) if args.split_audit else None,
        batch_id=args.batch_id,
    )
    print(json.dumps({
        "vocab_size_progression": rep["vocab_size_progression"],
        "delimiter_audit": rep["delimiter_audit"],
        "cross_check": rep["cross_check"],
        "oov_before_after": rep["oov_before_after"],
        "font_coverage_counts": {
            "total_fonts": rep["font_coverage_summary"]["actual_font_binaries_count"],
            "all_supported": rep["font_coverage_summary"]["all_fonts_supported_count"],
            "partial": rep["font_coverage_summary"]["partial_font_support_count"],
            "zero": rep["font_coverage_summary"]["zero_font_support_count"],
        },
        "new_vocab_sha256": rep["source_hashes"]["new_candidate_vocab"]["sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
