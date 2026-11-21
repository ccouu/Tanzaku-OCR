"""Regression tests for post-parse CTC label vocabulary candidate construction."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_ctc_label_vocab_candidate import (
    build_vocab_candidate,
    canonicalize_narrow,
    parse_ctc_labels_file,
)
from src.recognition.checkpointing import sha256_file
from src.recognition.vocabulary import extend_vocabulary


BASE_VOCAB = ROOT / "checkpoints/l2_v06/vocab.json"
PARENT_VOCAB = ROOT / "data/processed/recognition/vocab_v41_16112026.json"
NEW_VOCAB = ROOT / "data/processed/recognition/vocab_ctc_labels_16112026_v2.json"
YC_DIR = ROOT / "data/raw/tabelog_selected_menu_manual/16112026/yolo_crops"
FONTS_DIR = ROOT / "outputs/recognition/experiments/h4_k41_l2_v06_p5pct_pairs_ep04/payload/fonts/train"
SPLIT_AUDIT = ROOT / "outputs/recognition/experiments/l2_dishlex_tabelog30_yolo20_synth50_1000/manifests/new_tabelog_split_audit.json"


def test_parse_ctc_labels_delimiter_modes_and_rejection(tmp_path: Path):
    # Real TAB (raw CRLF bytes)
    f_tab = tmp_path / "tab.txt"
    f_tab.write_bytes("crops_ctc/a.png\t味噌ラーメン 850円\r\n".encode("utf-8"))
    rows, meta = parse_ctc_labels_file(f_tab)
    assert meta["delimiter_mode"] == "real_tab"
    assert rows == [("crops_ctc/a.png", "味噌ラーメン 850円")]

    # Literal \t
    f_lit = tmp_path / "lit.txt"
    f_lit.write_text("crops_ctc/a.png\\t塩ラーメン\n", encoding="utf-8")
    rows2, meta2 = parse_ctc_labels_file(f_lit)
    assert meta2["delimiter_mode"] == "literal_backslash_t"
    assert rows2 == [("crops_ctc/a.png", "塩ラーメン")]

    # Malformed line with extra TAB rejected
    f_bad = tmp_path / "bad.txt"
    f_bad.write_text("crops_ctc/a.png\tfoo\tbar\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed line"):
        parse_ctc_labels_file(f_bad)

    # Empty transcription rejected
    f_empty = tmp_path / "empty.txt"
    f_empty.write_text("crops_ctc/a.png\t\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Empty transcription"):
        parse_ctc_labels_file(f_empty)


def test_narrow_normalization_no_unexpected_lexical_rewrites():
    # Narrow fullwidth digits & parens ONLY
    assert canonicalize_narrow("（１２００円）") == "(1200円)"
    # Visual lookalikes, fullwidth symbols, circled digits, and spaces MUST remain untouched
    sample = "① ＜テスト＞ ！ 口ロ 〇0 〜 … A,B"
    assert canonicalize_narrow(sample) == sample


def test_append_only_chain_and_frozen_source_invariants():
    # Frozen source assets unchanged
    assert sha256_file(BASE_VOCAB) == "40d532789de881c88edf12f6ce857120159a85f568325036b5ea89378e6e7bc9"
    assert sha256_file(PARENT_VOCAB) == "fe2ff30704e7ac1a5671fe5953568e66aa64a3a946c97c0c981fffc72bcaaebc"
    reg = json.loads((ROOT / "src/recognition/current_model.json").read_text(encoding="utf-8"))
    assert reg["characters"] == 1779
    assert reg["vocab_sha256"] == sha256_file(BASE_VOCAB)

    v_base = json.loads(BASE_VOCAB.read_text(encoding="utf-8"))
    v_parent = json.loads(PARENT_VOCAB.read_text(encoding="utf-8"))
    v_new = json.loads(NEW_VOCAB.read_text(encoding="utf-8"))

    assert v_new["blank_id"] == 0
    c2i = v_new["char_to_id"]
    assert len(c2i) == len(v_new["characters"]) == 1901
    assert set(c2i.values()) == set(range(1, 1902))
    assert all(len(ch) == 1 for ch in c2i.keys())

    # Every base ID 1..1779 and parent ID 1780..1830 preserved identically
    for ch, cid in v_base["char_to_id"].items():
        assert c2i[ch] == cid
    for ch, cid in v_parent["char_to_id"].items():
        assert c2i[ch] == cid

    # Newly appended IDs 1831..1901 sorted deterministically by ord
    added = v_new["extension"]["added_characters"]
    assert len(added) == 71
    assert added == sorted(added, key=ord)
    for idx, ch in enumerate(added, start=1831):
        assert c2i[ch] == idx
    assert v_new["extension"]["provenance_scope"] == "label_informed_all_splits"
    assert v_new["extension"]["activation_status"] == "candidate_only_not_recognition_ready"


def test_deterministic_and_idempotent_rebuild(tmp_path: Path):
    out_vocab_1 = tmp_path / "vocab_rebuild.json"
    out_dir_1 = tmp_path / "audit_1"
    rep1 = build_vocab_candidate(
        yc_dir=YC_DIR,
        base_vocab_path=BASE_VOCAB,
        parent_vocab_path=PARENT_VOCAB,
        out_vocab_path=out_vocab_1,
        out_dir=out_dir_1,
        fonts_dir=FONTS_DIR,
        split_audit_path=SPLIT_AUDIT,
    )
    # Byte-identical deterministic vocab output vs committed artifact
    assert out_vocab_1.read_bytes() == NEW_VOCAB.read_bytes()
    assert rep1["oov_before_after"]["overall_all_ok"]["oov_candidate_new"]["crops"] == 0

    # Idempotent: extending again using out_vocab_1 as parent adds 0 tokens
    v_rebuilt = json.loads(out_vocab_1.read_text(encoding="utf-8"))
    ctc_rows, _ = parse_ctc_labels_file(YC_DIR / "ctc_labels.txt")
    all_chars = {ch for _, txt in ctc_rows for ch in canonicalize_narrow(txt)}
    _, added_again = extend_vocabulary(v_rebuilt, all_chars)
    assert added_again == []
