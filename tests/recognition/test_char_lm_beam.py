import json
import math
from pathlib import Path
import torch

from src.recognition.char_lm_beam import CharNGramLM, clean_hotpepper_title, ctc_char_lm_beam_decode


def test_clean_hotpepper_title():
    assert clean_hotpepper_title("【おすすめ】地鶏炭火焼♪", "food") == "地鶏炭火焼"
    assert clean_hotpepper_title("宴会コース 5,000円（税込）", "course") == "宴会コース"
    assert clean_hotpepper_title("ボトルキープが可能", "food") is None


def test_digit_neutral_and_gz_roundtrip(tmp_path: Path):
    char_to_id = {"ニ": 1, "二": 2, "ラ": 3, "人": 4, "前": 5, "0": 6, "8": 7, "1": 8}
    lm_dn = CharNGramLM._from_weighted_strings([("ニラ", 20), ("二人前", 20), ("100", 50)], char_to_id, digit_neutral=True)
    assert lm_dn.step_logp("<BOS>", "1", "0", "0") == math.log(0.1)
    assert lm_dn.step_logp("<BOS>", "1", "0", "8") == math.log(0.1)

    gz_dn = tmp_path / "lm_dn.json.gz"
    lm_dn.save_gz(gz_dn)
    loaded_dn = CharNGramLM.load_gz(gz_dn)
    assert loaded_dn.digit_neutral is True
    assert math.isclose(loaded_dn.step_logp("<BOS>", "<BOS>", "ニ", "ラ"), lm_dn.step_logp("<BOS>", "<BOS>", "ニ", "ラ"))

    # Verify digit_neutral=False is preserved across save_gz/load_gz
    lm_raw = CharNGramLM._from_weighted_strings([("100", 50)], char_to_id, digit_neutral=False)
    gz_raw = tmp_path / "lm_raw.json.gz"
    lm_raw.save_gz(gz_raw)
    loaded_raw = CharNGramLM.load_gz(gz_raw)
    assert loaded_raw.digit_neutral is False
    assert loaded_raw.step_logp("<BOS>", "1", "0", "0") > loaded_raw.step_logp("<BOS>", "1", "0", "8")


def test_oov_breaks_context_without_false_adjacency():
    char_to_id = {"鯖": 1, "味": 2, "噌": 3}
    lm = CharNGramLM._from_weighted_strings([("鯖、味噌", 10)], char_to_id)
    # '、' is OOV: must NOT glue '鯖' -> '味' or treat '味' as <BOS>
    assert "味" not in lm.bi.get("鯖", {})
    assert "味" not in lm.bi.get("<BOS>", {})
    # Valid contiguous bigram '味' -> '噌' IS preserved
    assert lm.bi["味"]["噌"] == 10


def test_beam_decode_resolves_ni_vs_two():
    char_to_id = {"ニ": 1, "二": 2, "ラ": 3}
    id_to_char = {v: k for k, v in char_to_id.items()}
    lm = CharNGramLM._from_weighted_strings([("ニラ", 50)], char_to_id)

    logits = torch.tensor(
        [
            [-5.0, 2.95, 3.00, -5.0],
            [-5.0, -5.0, -5.0, 6.0],
        ],
        dtype=torch.float32,
    )
    logp = torch.log_softmax(logits, dim=-1)
    assert ctc_char_lm_beam_decode(logp, id_to_char, lm, alpha_lm=0.0) == "二ラ"
    assert ctc_char_lm_beam_decode(logp, id_to_char, lm, alpha_lm=0.30, beam_width=4, topk_t=3) == "ニラ"


def test_ctc_collapse_and_repeated_chars():
    char_to_id = {"A": 1, "B": 2}
    id_to_char = {1: "A", 2: "B"}
    lm = CharNGramLM._from_weighted_strings([("AAB", 20)], char_to_id)

    # Sequence: A, A, <blank>, A, B -> CTC collapse must yield "AAB" (not "AAAB" or "AB")
    logits = torch.tensor(
        [
            [-5.0, 6.0, -5.0],  # A
            [-5.0, 6.0, -5.0],  # A (collapsed with previous A)
            [6.0, -5.0, -5.0],  # blank (separates next A)
            [-5.0, 6.0, -5.0],  # A (emits second A)
            [-5.0, -5.0, 6.0],  # B
        ],
        dtype=torch.float32,
    )
    logp = torch.log_softmax(logits, dim=-1)
    assert ctc_char_lm_beam_decode(logp, id_to_char, lm, alpha_lm=0.0) == "AAB"
    assert ctc_char_lm_beam_decode(logp, id_to_char, lm, alpha_lm=0.30, beam_width=5, topk_t=2) == "AAB"


def test_digit_neutral_preserves_price_under_strong_lm_bias():
    char_to_id = {"1": 1, "2": 2, "0": 3, "8": 4, "円": 5}
    id_to_char = {v: k for k, v in char_to_id.items()}
    # Heavy corpus bias toward "120円"
    lm_dn = CharNGramLM._from_weighted_strings([("120円", 500)], char_to_id, digit_neutral=True)
    lm_biased = CharNGramLM._from_weighted_strings([("120円", 500)], char_to_id, digit_neutral=False)

    # Visual logits favor "128円", with '8' only slightly ahead of '0'
    logits = torch.tensor(
        [
            [-5.0, 6.0, -5.0, -5.0, -5.0, -5.0],  # 1
            [-5.0, -5.0, 6.0, -5.0, -5.0, -5.0],  # 2
            [-5.0, -5.0, -5.0, 2.70, 3.00, -5.0],  # 8 slightly ahead of 0
            [-5.0, -5.0, -5.0, -5.0, -5.0, 6.0],  # 円
        ],
        dtype=torch.float32,
    )
    logp = torch.log_softmax(logits, dim=-1)
    assert ctc_char_lm_beam_decode(logp, id_to_char, lm_dn, alpha_lm=0.30, beam_width=5, topk_t=5) == "128円"
    assert ctc_char_lm_beam_decode(logp, id_to_char, lm_biased, alpha_lm=0.30, beam_width=5, topk_t=5) == "120円"


def test_build_from_cleaned_files(tmp_path: Path):
    lex_path = tmp_path / "menu_lexicon.json"
    lex_path.write_text(
        json.dumps(
            {
                "terms_521": ["ラーメン"],
                "approved_terms": ["餃子"],
                "context_phrases": ["1人前"],
                "db_terms": [
                    {"term": "特製味噌ラーメン", "category": "curated_grammar"},
                    {"term": "100歩譲って", "category": "jmdict_food"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    cleaned_dir = tmp_path / "cleaned"
    cleaned_dir.mkdir()
    (cleaned_dir / "jmdict_food_cleaned.json").write_text(
        json.dumps({"kept": ["冷やし中華"], "dropped": ["100歩譲って"]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (cleaned_dir / "hotpepper_310_cleaned.json").write_text(
        json.dumps({"kept_per_restaurant": ["レモンサワー"]}, ensure_ascii=False),
        encoding="utf-8",
    )
    char_to_id = {c: i + 1 for i, c in enumerate("ラーメン餃子1人前特製味噌冷やし中華レモンサワ")}
    lm = CharNGramLM.build_from_cleaned_files(lex_path, cleaned_dir, char_to_id)
    assert "レ" in lm.uni
    assert "歩" not in lm.uni
