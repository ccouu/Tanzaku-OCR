"""Comprehensive unit & integration tests for src/recognition/lexicon_lookup/."""
from __future__ import annotations

import itertools
import json
import math
import sqlite3
import tempfile
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from scripts.recognition.serve_lexicon_inspector import (
    HTML_PAGE,
    InspectorAppState,
    load_allowlisted_real_samples,
    make_handler,
)
from src.recognition.char_lm_beam import (
    DecodeConfig,
    ImmutableCharNGramLM,
    IncompatibleLMSchemaError,
    decode_with_config,
)
from src.recognition.lexicon_lookup import (
    CATEGORY_CROSS_SCRIPT,
    CATEGORY_ORDINARY_OCR,
    DEFAULT_CONFUSABLE_CATALOG,
    MATCH_TYPE_EXACT,
    MATCH_TYPE_ORDINARY_FUZZY,
    MATCH_TYPE_VISUAL_CONFUSABLE,
    EditPolicyConfig,
    LexiconEntry,
    LexiconIndex,
    LexiconPipeline,
    LookupConfig,
    ReadOnlyTranslationRepo,
    align_edit_ops,
    build_lexicon_index,
    canonicalize_text,
    classify_match_type,
    ctc_forward_log_prob_from_log_probs,
    extract_protected_spans,
    normalize_with_span_map,
    prepare_canonical_log_probs,
    score_candidate_line_ctc,
    search_span_exact_and_fuzzy,
)
from http.server import ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data/processed/lexicon/culinary_menu.db"
INDEX_PATH = ROOT / "outputs/recognition/implementation/lexicon_trie_python_20261119/artifacts/lexicon_index_v1.json.gz"
LM_PATH = ROOT / "checkpoints/l2_v06/char_lm_4g_hp1010.json.gz"


def _brute_levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        nxt = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            nxt[j] = min(dp[j] + 1, nxt[j - 1] + 1, dp[j - 1] + cost)
        dp = nxt
    return dp[n]


def _make_mini_index(terms: list[tuple[str, str]]) -> LexiconIndex:
    entries = tuple(
        LexiconEntry(
            entry_id=i,
            canonical_term=t,
            display_term=t,
            source_tier=tier,
            sources=(f"test:{tier}",),
        )
        for i, (t, tier) in enumerate(terms)
    )
    return LexiconIndex(entries=entries, metadata={"schema_version": 1, "index_format": "lexicon_trie_v1"})


def test_trie_matches_brute_force_oracle_including_ins_del_repeats():
    vocab = [
        ("豚ロース", "jmdict_food_cleaned"),
        ("定食", "curated_core"),
        ("ラーメン", "curated_core"),
        ("ラーーメン", "context_phrase"),
        ("水餃子", "curated_core"),
        ("生ビール", "context_phrase"),
        ("特上ロース", "jmdict_food_cleaned"),
        ("焼き鳥", "curated_core"),
        ("焼鳥", "curated_core"),
        ("あじ", "curated_core"),
        ("いか", "curated_core"),
        ("たこ", "curated_core"),
        ("さば", "curated_core"),
        ("串", "curated_core"),
    ]
    idx = _make_mini_index(vocab)
    policy = EditPolicyConfig(len1_max_edits=0, len2_max_edits=1, len3_5_max_edits=1, len6plus_max_edits=2)
    queries = [
        "豚ロ一ス",     # visual confusion sub
        "豚ロス",       # deletion
        "豚ローース",   # insertion / repeat
        "水餃子子",     # trailing repeat
        "特上口ース",   # visual confusion sub
        "ラーメン",     # exact + 1-edit repeat neighbor
        "焼鳥",         # exact + 1-edit insertion neighbor 焼き鳥
        "あじ",         # 2-char
        "串",           # 1-char exact only
        "串串",         # 2-char cannot match 1-char under conservative boundary
    ]
    for q in queries:
        max_e = policy.allowed_edits_for_length(len(q))
        hits, visits, truncated = search_span_exact_and_fuzzy(idx, q, max_edits=max_e, policy=policy)
        assert not truncated and visits > 0
        got = {e.canonical_term: h for e, h, _, _ in hits}
        expected = {}
        for term, _ in vocab:
            d = _brute_levenshtein(q, term)
            if d <= policy.max_allowed_edits(len(q), len(term)):
                expected[term] = d
        assert got == expected, f"Mismatch for query={q!r}: got={got}, expected={expected}"

    # Verify visual confusion lowers weighted cost while preserving hard_edits == 1
    hard, weighted, ops = align_edit_ops("豚ロ一ス", "豚ロース", use_confusion_weights=True, confusion_sub_cost=0.45)
    assert hard == 1
    assert weighted == pytest.approx(0.45)
    assert any(op.op == "substitute" and op.confusion_pair and op.query_char == "一" and op.cand_char == "ー" for op in ops)


def test_normalization_and_span_offsets():
    raw = "生ビール（中） １,２００円 豚ロ一ス"
    smap = normalize_with_span_map(raw)
    assert smap.canonical_text == "生ビール(中) 1,200円 豚ロ一ス"
    # Visually similar 一 and ロ must NOT be globally replaced
    assert "豚ロ一ス" in smap.canonical_text
    # Check exact code-point mapping back to original
    s = smap.canonical_text.index("1,200円")
    e = s + len("1,200円")
    os_, oe_ = smap.to_original_span(s, e)
    assert raw[os_:oe_] == "１,２００円"
    assert len(smap.changes) == 6  # （, ）, １, ２, ０, ０


def test_edit_policy_boundaries():
    policy = EditPolicyConfig(conservative_boundary_enforcement=True)
    # 1-char exact only; cannot bridge 1 <-> 2
    assert policy.max_allowed_edits(1, 1) == 0
    assert policy.max_allowed_edits(1, 2) == 0
    assert policy.max_allowed_edits(2, 1) == 0
    # 2-char <=1, 3-5 <=1, >=6 <=2; boundary 5 <-> 6 stays <=1
    assert policy.max_allowed_edits(2, 2) == 1
    assert policy.max_allowed_edits(5, 5) == 1
    assert policy.max_allowed_edits(5, 6) == 1
    assert policy.max_allowed_edits(6, 5) == 1
    assert policy.max_allowed_edits(6, 6) == 2


def test_protected_spans_prices_counts_and_size_not_numeric():
    smap = normalize_with_span_map("生ビール（中） 特上ロース 大 ２杯 10%OFF 1,200円(税込) ¥500~¥800")
    spans = extract_protected_spans(smap)
    texts = [(sp.text, sp.kind) for sp in spans]
    assert ("2杯", "count") in texts
    assert ("10%", "percent") in texts
    assert ("1,200円(税込)", "price") in texts
    assert ("¥500~¥800", "price") in texts
    # Size/rank characters 中, 特上, 大 must NOT be treated as protected numeric spans
    protected_joined = " ".join(sp.text for sp in spans)
    for sz in ("中", "特上", "大"):
        assert sz not in protected_joined


def test_pork_loin_example_and_short_fish_and_unknown_fallback():
    if not INDEX_PATH.is_file():
        pytest.skip("Built lexicon index not present")
    idx = LexiconIndex.load_gz(INDEX_PATH)
    repo = ReadOnlyTranslationRepo(DB_PATH)
    pipe = LexiconPipeline(idx, repo)

    # 1. WORK_PLAN example: 豚ロ一ス定食 1,200円 -> suggest 豚ロース + 定食, price intact, auto_corrected=False
    res = pipe.lookup_text("豚ロ一ス定食 1,200円")
    assert res.auto_corrected is False
    assert res.canonical_ocr == "豚ロ一ス定食 1,200円"
    assert res.suggested_line == "豚ロース定食 1,200円"
    assert any(sp.text == "1,200円" and sp.kind == "price" for sp in res.protected_spans)
    seg_kinds = [(s.text, s.kind, s.selected_candidate.term if s.selected_candidate else None) for s in res.segments]
    assert ("豚ロ一ス", "fuzzy_suggestion", "豚ロース") in seg_kinds
    assert ("定食", "exact", "定食") in seg_kinds
    assert ("1,200円", "protected", None) in seg_kinds

    # 2. Ambiguous short fish names must remain exact and not be rewritten by frequency
    for fish in ("あじ", "いか", "たこ", "さば"):
        fres = pipe.lookup_text(fish)
        assert fres.suggested_line == fish
        assert len(fres.segments) == 1
        assert fres.segments[0].kind == "exact"
        assert fres.segments[0].selected_candidate.term == fish

    # 3. Unknown OOV text must survive without forced dictionary coverage
    ures = pipe.lookup_text("ゾルバ風オリジナル 850円")
    assert any(s.kind == "unknown" and "ゾルバ" in s.text for s in ures.segments)
    assert any(s.kind == "protected" and s.text == "850円" for s in ures.segments)


def test_dictionary_duplicates_aliases_and_readonly_db():
    with tempfile.TemporaryDirectory() as td:
        menu_json = Path(td) / "menu.json"
        menu_json.write_text(
            json.dumps(
                {
                    "approved_terms": [
                        {"term": "定食", "category": "dish"},
                        {"term": "生ビール（中）", "category": "drink"},
                    ],
                    "terms_521": ["定食"],
                    "context_phrases": ["定食", "生ビール(中)"],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        idx = build_lexicon_index(menu_lexicon_path=menu_json, culinary_db_path=DB_PATH)
        e_tei = idx.get_by_term("定食")
        assert e_tei is not None
        assert e_tei.source_tier == "curated_core"
        assert "menu_lexicon:approved_terms" in e_tei.sources
        assert "menu_lexicon:terms_521" in e_tei.sources
        assert "menu_lexicon:context_phrases" in e_tei.sources
        e_beer = idx.get_by_term("生ビール(中)")
        assert e_beer is not None
        assert "生ビール（中）" in e_beer.aliases

    # Verify DB repository is strictly read-only
    repo = ReadOnlyTranslationRepo(DB_PATH)
    found = repo.get_by_term("定食")
    assert found.status == "found" and found.name_en
    missing = repo.get_by_term("存在しない料理名XYZ")
    assert missing.status == "missing" and missing.name_en == ""
    with pytest.raises(sqlite3.OperationalError):
        repo._get_conn().execute("CREATE TABLE should_fail (x INT)")


def _brute_force_ctc_log_prob(log_probs: list[list[float]], target: list[int], blank_id: int = 0) -> float:
    T = len(log_probs)
    C = len(log_probs[0])
    total = -math.inf
    for path in itertools.product(range(C), repeat=T):
        collapsed = []
        prev = None
        for tok in path:
            if tok != prev and tok != blank_id:
                collapsed.append(tok)
            prev = tok
        if collapsed == target:
            lp = sum(log_probs[t][path[t]] for t in range(T))
            total = lp if total == -math.inf else (max(total, lp) + math.log1p(math.exp(-abs(total - lp))))
    return total


def test_ctc_forward_scorer_matches_brute_force_oracle_and_handles_oov():
    torch.manual_seed(7)
    # T=5, C=6 (blank=0, A=1, B=2, '('=3, '（'=4, ')'=5, '）'=6 -> use C=7)
    mapping = {"A": 1, "B": 2, "(": 3, "（": 4, ")": 5, "）": 6}
    logits = torch.randn(5, 7)
    canon_lp = prepare_canonical_log_probs(logits, valid_length=5, mapping=mapping)

    # 1. Exact parity on normal, repeated-symbol ("AA"), and empty/all-blank ("")
    for text, toks in [("", []), ("A", [1]), ("AB", [1, 2]), ("AA", [1, 1]), ("ABA", [1, 2, 1]), ("(A)", [3, 1, 5])]:
        got = ctc_forward_log_prob_from_log_probs(canon_lp, toks, blank_id=0)
        ref = _brute_force_ctc_log_prob(canon_lp, toks, blank_id=0)
        assert got == pytest.approx(ref, abs=1e-6), f"CTC oracle mismatch for {text!r}"

    # 2. Full-width parenthesis canonicalizes to ASCII and pools probability mass
    r_ascii = score_candidate_line_ctc("(A)", canonical_log_probs=canon_lp, mapping=mapping)
    r_wide = score_candidate_line_ctc("（A）", canonical_log_probs=canon_lp, mapping=mapping)
    assert r_ascii.scorable and r_wide.scorable
    assert r_ascii.ctc_log_prob == pytest.approx(r_wide.ctc_log_prob, abs=1e-6)

    # 3. OOV character is explicitly marked unscorable, never assigned fake prob
    r_oov = score_candidate_line_ctc("AZ", canonical_log_probs=canon_lp, mapping=mapping)
    assert r_oov.scorable is False
    assert r_oov.ctc_log_prob is None
    assert "Z" in r_oov.unsupported_chars

    # 4. Length-infeasible repeated sequence ("AAAA" requires 7 steps > T=5)
    r_infeas = score_candidate_line_ctc("AAAA", canonical_log_probs=canon_lp, mapping=mapping)
    assert r_infeas.scorable is False
    assert r_infeas.ctc_log_prob is None


def test_immutable_lm_and_concurrent_http_server():
    if not INDEX_PATH.is_file() or not LM_PATH.is_file():
        pytest.skip("Index or schema-v2 LM not present")

    lm = ImmutableCharNGramLM.load_gz(LM_PATH, allow_legacy_v1=False)
    assert lm.schema_version == 2
    assert lm.digit_neutral is True
    with pytest.raises(AttributeError):
        lm.digit_neutral = False  # type: ignore[misc]
    with pytest.raises(IncompatibleLMSchemaError):
        lm.validate_mode(digit_neutral=False)

    idx = LexiconIndex.load_gz(INDEX_PATH)
    repo = ReadOnlyTranslationRepo(DB_PATH)
    pipe = LexiconPipeline(idx, repo, immutable_lm=lm)
    samples = load_allowlisted_real_samples(max_samples=2)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(InspectorAppState(pipe, samples)))
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{port}"
        # Health & GET /
        with urllib.request.urlopen(f"{base}/api/health") as r:
            assert r.status == 200
            h = json.loads(r.read().decode("utf-8"))
            assert h["auto_corrected_default"] is False

        # Concurrent POST /api/lookup_text with varying per-request configs
        def _req(i: int) -> dict:
            payload = {
                "text": "豚ロ一ス定食 1,200円" if i % 2 == 0 else "あじ 580円",
                "exact_only": (i % 3 == 0),
                "include_expanded_culinary": (i % 4 != 0),
            }
            req = urllib.request.Request(
                f"{base}/api/lookup_text",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 200
                return json.loads(resp.read().decode("utf-8"))

        with ThreadPoolExecutor(max_workers=6) as ex:
            results = list(ex.map(_req, range(18)))
        assert len(results) == 18
        assert all(r["result"]["auto_corrected"] is False for r in results)

        # Malformed JSON -> HTTP 400
        bad_req = urllib.request.Request(
            f"{base}/api/lookup_text",
            data=b"{bad json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(bad_req)
        assert exc_info.value.code == 400

        # Unknown / traversal sample_id -> HTTP 404
        trav_req = urllib.request.Request(
            f"{base}/api/lookup_sample",
            data=json.dumps({"sample_id": "../../etc/passwd"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info2:
            urllib.request.urlopen(trav_req)
        assert exc_info2.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_lookup_from_logits_greedy_r1_and_plain_beam():
    if not INDEX_PATH.is_file() or not LM_PATH.is_file():
        pytest.skip("Index or schema-v2 LM not present")
    idx = LexiconIndex.load_gz(INDEX_PATH)
    repo = ReadOnlyTranslationRepo(DB_PATH)
    lm = ImmutableCharNGramLM.load_gz(LM_PATH, allow_legacy_v1=False)
    pipe = LexiconPipeline(idx, repo, immutable_lm=lm)

    # Synthetic logits peaking on "豚ロ一ス" (T=8, blank-separated)
    vocab_chars = ["豚", "ロ", "一", "ス", "ー", "(", "（", ")", "）"]
    mapping = {ch: i + 1 for i, ch in enumerate(vocab_chars)}
    T, C = 8, len(mapping) + 1
    logits = torch.full((T, 1, C), -8.0)
    seq = [mapping["豚"], 0, mapping["ロ"], 0, mapping["一"], 0, mapping["ス"], 0]
    for t, tok in enumerate(seq):
        logits[t, 0, tok] = 6.0

    for mode in ("greedy", "plain_beam", "beam4g_dn_r1"):
        res = pipe.lookup_from_logits(logits, T, mapping, decode_mode=mode)
        assert res.auto_corrected is False
        assert res.canonical_ocr == "豚ロ一ス"
        assert res.decoder_mode == mode
        assert res.lm_alternative is not None
        assert res.suggested_line == "豚ロース"
        assert any(vr.candidate_line == "豚ロース" and vr.scorable for vr in res.visual_rescores)
        assert any(vr.is_unchanged_ocr and vr.scorable and vr.rank is not None for vr in res.visual_rescores)
        assert res.visual_reranked_line in ("豚ロ一ス", "豚ロース")


def test_overlong_input_preserves_suffix_and_protected_digits():
    idx = _make_mini_index([("豚ロース", "jmdict_food_cleaned"), ("定食", "curated_core")])
    pipe = LexiconPipeline(idx)
    text = "豚ロ一ス定食 特製ソース付き １,２００円(税込)"
    cfg = LookupConfig(max_query_length=10)
    res = pipe.lookup_text(text, config=cfg)

    assert res.diagnostics["unprocessed_suffix_preserved"] is True
    assert any("unprocessed_suffix_preserved" in w for w in res.warnings)
    # Zero characters dropped: suffix and its protected price survive intact
    assert "特製ソース付き" in res.suggested_line
    assert "1,200円(税込)" in res.suggested_line
    assert any(sp.text == "1,200円(税込)" and sp.raw_text == "１,２００円(税込)" for sp in res.protected_spans)
    # Every segment offset maps cleanly into canonical & original strings with full coverage
    assert res.segments[0].start == 0
    assert res.segments[-1].end == len(res.canonical_ocr)
    for prev, cur in zip(res.segments, res.segments[1:]):
        assert prev.end == cur.start
        assert res.raw_ocr[cur.orig_start : cur.orig_end] == cur.raw_text


def test_candidate_budget_enforcement_and_invalid_bounds():
    with pytest.raises(ValueError):
        LookupConfig(max_candidates_per_span=0)
    with pytest.raises(ValueError):
        LookupConfig(max_total_candidates=0)
    with pytest.raises(ValueError):
        LookupConfig(max_query_length=0)

    idx = _make_mini_index(
        [
            ("豚ロース", "jmdict_food_cleaned"),
            ("定食", "curated_core"),
            ("豚", "curated_core"),
            ("定", "curated_core"),
        ]
    )
    pipe = LexiconPipeline(idx)
    # Total budget of 1 means DP can only consume 1 candidate total across the whole line
    cfg = LookupConfig(max_total_candidates=1, max_candidates_per_span=1)
    res = pipe.lookup_text("豚ロース定食", config=cfg)
    assert res.diagnostics["candidates_after_cap"] == 1
    assert res.diagnostics["candidates_before_cap"] > 1
    assert res.diagnostics["candidates_truncated"] is True
    matched_segs = [s for s in res.segments if s.kind in ("exact", "fuzzy_suggestion")]
    assert len(matched_segs) == 1


def test_visual_ctc_ranking_uses_logits_and_includes_unchanged_ocr():
    idx = _make_mini_index([("サラミ", "jmdict_food_cleaned")])
    pipe = LexiconPipeline(idx)
    mapping = {"ハ": 1, "サ": 2, "ラ": 3, "ミ": 4, "(": 5, "（": 6, ")": 7, "）": 8}
    T, C = 6, len(mapping) + 1

    # Case A: Model is VERY confident in "ハラミ" (strong peak on ハ, near-zero on サ)
    # Text-only DP would suggest "サラミ" (1 edit), but CTC ranking must keep unchanged "ハラミ" at rank 1!
    logits_strong = torch.full((T, 1, C), -15.0)
    for t, tok in enumerate([mapping["ハ"], 0, mapping["ラ"], 0, mapping["ミ"], 0]):
        logits_strong[t, 0, tok] = 10.0
    res_a = pipe.lookup_from_logits(logits_strong, T, mapping, config=LookupConfig(visual_ctc_weight=0.25))
    assert res_a.suggested_line == "サラミ"
    assert res_a.visual_reranked_line == "ハラミ"
    assert res_a.visual_rescores[0].candidate_line == "ハラミ"
    assert res_a.visual_rescores[0].is_unchanged_ocr is True
    assert res_a.visual_rescores[0].rank == 1

    # Case B: Ambiguous logits between ハ (5.0) and サ (4.8) -> CTC delta is tiny, text score wins -> "サラミ" rank 1
    logits_ambig = logits_strong.clone()
    logits_ambig[0, 0, mapping["ハ"]] = 5.0
    logits_ambig[0, 0, mapping["サ"]] = 4.8
    res_b = pipe.lookup_from_logits(logits_ambig, T, mapping, config=LookupConfig(visual_ctc_weight=0.25))
    assert res_b.visual_reranked_line == "サラミ"
    assert res_b.visual_rescores[0].candidate_line == "サラミ"
    assert res_b.visual_rescores[0].rank == 1


def test_lookup_image_geometry_oversized_overlap_and_orientation_validation():
    from PIL import Image

    class _DummyModel:
        def __init__(self) -> None:
            self.vocabulary = {"豚": 1, "ロ": 2, "ー": 3, "ス": 4, "(": 5, "（": 6, ")": 7, "）": 8}
            self.calls: list[tuple[int, int]] = []

        def __call__(self, x: torch.Tensor, ori: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
            h, w = int(x.shape[2]), int(x.shape[3])
            self.calls.append((h, w))
            T = int(lengths[0].item())
            C = len(self.vocabulary) + 1
            out = torch.full((T, 1, C), -6.0)
            out[:, 0, 0] = 5.0
            if T >= 4:
                out[0, 0, 1] = 8.0
                out[2, 0, 2] = 8.0
            return out

    idx = _make_mini_index([("豚ロース", "jmdict_food_cleaned")])
    dummy = _DummyModel()
    pipe = LexiconPipeline(idx, model=dummy, device="cpu")

    # Strictly reject invalid orientations in public API
    small_img = Image.new("RGB", (240, 64), color=(255, 255, 255))
    with pytest.raises(ValueError):
        pipe.lookup_image(small_img, orientation=2)
    with pytest.raises(ValueError):
        pipe.lookup_image(small_img, orientation=-1)
    with pytest.raises(ValueError):
        pipe.lookup_image(small_img, orientation=True)  # type: ignore[arg-type]

    # <=512 strip uses single-window route with zero height shrink
    res_small = pipe.lookup_image(small_img, orientation=0)
    assert res_small.provenance["crop"]["long_line_route"] == "single_window_le512"
    assert res_small.provenance["crop"]["scaled_height"] == 64

    # Oversized 1200x64 strip MUST use overlap512_h64 route without height shrinking
    dummy.calls.clear()
    wide_img = Image.new("RGB", (1200, 64), color=(255, 255, 255))
    res_wide = pipe.lookup_image(wide_img, orientation=0)
    crop_prov = res_wide.provenance["crop"]
    assert crop_prov["long_line_route"] == "overlap512_h64"
    assert crop_prov["window_count"] >= 3
    assert all(w["scaled_height"] == 64 for w in crop_prov["windows"])
    assert all(h == 64 and w <= 512 for h, w in dummy.calls)
    assert res_wide.visual_reranked_line is None
    assert len(res_wide.visual_rescores) == 1
    assert res_wide.visual_rescores[0].scorable is False
    assert res_wide.visual_rescores[0].reason == "multi_window_overlap_logits_not_jointly_scorable"


def test_normalization_aware_protected_span_invariance():
    idx = _make_mini_index([("豚ロース", "jmdict_food_cleaned"), ("定食", "curated_core")])
    pipe = LexiconPipeline(idx)
    raw = "豚ロ一ス定食 １,２００円（税込）"
    res = pipe.lookup_text(raw)
    assert res.raw_ocr == raw
    assert res.canonical_ocr == "豚ロ一ス定食 1,200円(税込)"
    assert res.suggested_line == "豚ロース定食 1,200円(税込)"
    assert len(res.protected_spans) == 1
    sp = res.protected_spans[0]
    assert sp.text == "1,200円(税込)"
    assert sp.raw_text == "１,２００円（税込）"
    # Protected span is preserved identically in suggested_line
    assert sp.text in res.suggested_line


def test_confusable_catalog_symmetric_non_transitive_immutable_and_no_zero_cost():
    cat = DEFAULT_CONFUSABLE_CATALOG
    # Immutability
    with pytest.raises(AttributeError):
        cat.extra = 1  # type: ignore[attr-defined]
    # Zero-cost edits strictly forbidden
    with pytest.raises(ValueError):
        LookupConfig(confusion_sub_cost=0.0)
    with pytest.raises(ValueError):
        LookupConfig(ordinary_confusion_sub_cost=0.0)

    # Symmetric lookup for cross-script lookalikes
    p1 = cat.lookup("口", "ロ")
    p2 = cat.lookup("ロ", "口")
    assert p1 is not None and p1 == p2
    assert p1.category == CATEGORY_CROSS_SCRIPT
    assert "sharp_pct1_ss_guide" in p1.sources

    # Ordinary OCR confusions stay in ordinary category, never cross-script
    p_shi = cat.lookup("シ", "ツ")
    p_bi = cat.lookup("ビ", "ピ")
    assert p_shi is not None and p_shi.category == CATEGORY_ORDINARY_OCR
    assert p_bi is not None and p_bi.category == CATEGORY_ORDINARY_OCR

    # NO transitive closure: ー/一 and ー/- exist, so 一/- MUST be None
    assert cat.lookup("ー", "一") is not None
    assert cat.lookup("ー", "-") is not None
    assert cat.lookup("一", "-") is None

    # Provenance serialization
    payload = cat.to_dict()
    assert payload["catalog_version"] == 1
    assert payload["transitive_closure"] is False
    assert payload["zero_cost_edits_allowed"] is False
    assert payload["counts"][CATEGORY_CROSS_SCRIPT] == 11
    assert "icann72_jp_lgr_p31" in payload["bibliography"]


def test_match_type_visual_only_vs_ordinary_vs_mixed_and_exact_separation():
    if not INDEX_PATH.is_file():
        pytest.skip("Built lexicon index not present")
    idx = LexiconIndex.load_gz(INDEX_PATH)
    repo = ReadOnlyTranslationRepo(DB_PATH)
    pipe = LexiconPipeline(idx, repo)

    # 1. 特上口ース: 特上 is exact, 口ース->ロース is visual_confusable_suggestion with contingent DB translation
    res = pipe.lookup_text("特上口ース")
    assert res.raw_ocr == "特上口ース"
    assert res.canonical_ocr == "特上口ース"
    assert res.auto_corrected is False
    # Exact translations table MUST contain ONLY literal exact matches (特上), NEVER ロース
    assert [e["term"] for e in res.exact_translations] == ["特上"]
    rosu_cands = [c for c in res.fuzzy_suggestions if c.term == "ロース"]
    assert len(rosu_cands) == 1
    rc = rosu_cands[0]
    assert rc.match_type == MATCH_TYPE_VISUAL_CONFUSABLE
    assert rc.hard_edits == 1
    assert rc.translation is not None and rc.translation.status == "found"
    assert "loin" in rc.translation.name_en.lower()
    sub_ops = [o for o in rc.edit_ops if o.op == "substitute"]
    assert len(sub_ops) == 1
    assert sub_ops[0].query_codepoint == "U+53E3" and sub_ops[0].cand_codepoint == "U+30ED"
    assert sub_ops[0].confusion_category == CATEGORY_CROSS_SCRIPT

    # 2. Ordinary OCR confusion (ビ->ピ) and mixed edits (lookalike + insert/delete/ordinary) NEVER get visual-only label
    h_ord, _, ops_ord = align_edit_ops("エビビラフ", "エビピラフ")
    assert h_ord == 1 and classify_match_type(h_ord, ops_ord) == MATCH_TYPE_ORDINARY_FUZZY

    h_mix, _, ops_mix = align_edit_ops("口ースー", "ロース")  # cross-script sub + deletion
    assert h_mix == 2 and classify_match_type(h_mix, ops_mix) == MATCH_TYPE_ORDINARY_FUZZY

    h_mix2, _, ops_mix2 = align_edit_ops("口ビラフ", "ロピラフ")  # cross-script + ordinary
    assert h_mix2 == 2 and classify_match_type(h_mix2, ops_mix2) == MATCH_TYPE_ORDINARY_FUZZY


def test_negative_kanji_words_isolated_single_chars_and_ctc_rejects_visual_lookalike():
    if not INDEX_PATH.is_file():
        pytest.skip("Built lexicon index not present")
    idx = LexiconIndex.load_gz(INDEX_PATH)
    repo = ReadOnlyTranslationRepo(DB_PATH)
    pipe = LexiconPipeline(idx, repo)

    # Known kanji words stay exact; unknown kanji words survive without blanket katakana rewrite
    for exact_word in ("辛口", "一杯"):
        r = pipe.lookup_text(exact_word)
        assert r.suggested_line == exact_word
        assert r.segments[0].kind == "exact"
        assert r.segments[0].selected_candidate.match_type == MATCH_TYPE_EXACT

    # Isolated 1-char lookalikes remain exact-only (never converted across script)
    for ch in ("口", "ロ", "力", "カ"):
        r = pipe.lookup_text(ch)
        assert r.suggested_line == ch
        assert not any(c.hard_edits > 0 for c in r.fuzzy_suggestions)

    # Whole-line CTC gate still rejects a visual_confusable_suggestion when logits strongly favor original char
    mini = _make_mini_index([("ロース", "jmdict_food_cleaned")])
    mpipe = LexiconPipeline(mini)
    mapping = {"口": 1, "ロ": 2, "ー": 3, "ス": 4, "(": 5, "（": 6, ")": 7, "）": 8}
    T, C = 6, len(mapping) + 1
    logits = torch.full((T, 1, C), -15.0)
    for t, tok in enumerate([mapping["口"], 0, mapping["ー"], 0, mapping["ス"], 0]):
        logits[t, 0, tok] = 10.0
    vres = mpipe.lookup_from_logits(logits, T, mapping, config=LookupConfig(visual_ctc_weight=0.25))
    assert vres.suggested_line == "ロース"
    assert vres.fuzzy_suggestions[0].match_type == MATCH_TYPE_VISUAL_CONFUSABLE
    # Strong CTC evidence for 口 keeps unchanged OCR at rank #1!
    assert vres.visual_reranked_line == "口ース"
    assert vres.visual_rescores[0].is_unchanged_ocr is True


def test_inspector_xss_safe_rendering_and_confusables_endpoint():
    # Ensure touched HTML_PAGE does not use innerHTML for dynamic table/segment rendering
    assert ".innerHTML" not in HTML_PAGE

    idx = _make_mini_index([("豚ロース", "jmdict_food_cleaned")])
    pipe = LexiconPipeline(idx)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(InspectorAppState(pipe, {})))
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(f"{base}/api/confusables") as r:
            assert r.status == 200
            cat = json.loads(r.read().decode("utf-8"))
            assert cat["catalog_version"] == 1
            assert cat["counts"][CATEGORY_CROSS_SCRIPT] == 11

        xss_text = '<img src=x onerror="alert(1)"> 豚ロ一ス 1,200円'
        req = urllib.request.Request(
            f"{base}/api/lookup_text",
            data=json.dumps({"text": xss_text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            d = json.loads(resp.read().decode("utf-8"))
            assert d["result"]["raw_ocr"] == xss_text
            assert any(sp["text"] == "1,200円" for sp in d["result"]["protected_spans"])
    finally:
        server.shutdown()
        server.server_close()
