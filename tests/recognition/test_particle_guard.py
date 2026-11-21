import torch

from src.recognition.lexicon_lookup import LexiconIndex, LexiconEntry, LexiconPipeline, LookupConfig
from src.recognition.lexicon_lookup.particle_guard import probable_particles


def make_pipeline():
    words = ["焼きふ", "焼き目", "おでん", "うに", "辛口", "一杯", "二人前", "ロース"]
    entries = tuple(LexiconEntry(i, w, w, "curated_core", ("test",)) for i, w in enumerate(words))
    return LexiconPipeline(LexiconIndex(entries, {"schema_version": 1}))


def test_correct_particle_phrase_preserved_and_alternative_inspectable():
    pipe = make_pipeline()
    text = "荒火焼きで食べる"
    assert pipe.lookup_text(text, config=LookupConfig(protect_particles=False)).suggested_line != text
    out = pipe.lookup_text(text)
    assert out.suggested_line == text and out.raw_ocr == text
    assert out.diagnostics["probable_particles"] == [
        {"start": 4, "end": 5, "text": "で", "clause_start": 0, "clause_end": 8,
         "reason": "menu_action_context"}]


def test_real_index_does_not_replace_another_word_in_the_protected_clause():
    from pathlib import Path
    path = Path("outputs/recognition/implementation/lexicon_trie_python_20261119/artifacts/lexicon_index_v1.json.gz")
    if not path.exists():
        import pytest
        pytest.skip("Local derived index unavailable")
    pipe = LexiconPipeline(LexiconIndex.load_gz(path))
    out = pipe.lookup_text("荒火焼きで食べる")
    assert out.suggested_line == "荒火焼きで食べる"
    alt = next(c for c in out.fuzzy_suggestions if c.query_span == "焼きで" and c.term == "焼きふ")
    assert alt.suppression_reasons
    assert all(not s.selected_candidate or not s.selected_candidate.suppression_reasons for s in out.segments)


def test_word_internal_de_and_ni_are_not_global_particle_replacements():
    pipe = make_pipeline()
    for text in ["おでん", "おでんを食べる", "うに食べる"]:
        particles = probable_particles(text, pipe.index, LookupConfig())
        assert all(p["text"] not in ("で", "に") for p in particles)
        assert pipe.lookup_text(text).suggested_line == text
    assert probable_particles("塩で", pipe.index, LookupConfig())[0]["text"] == "で"
    assert probable_particles("ゆで", pipe.index, LookupConfig()) == ()


def test_particles_preserve_numeric_spans_and_keep_legitimate_visual_corrections():
    pipe = make_pipeline()
    for text in ["辛口", "一杯", "二人前"]:
        assert pipe.lookup_text(text).suggested_line == text
    out = pipe.lookup_text("口ースで食べる 1,200円")
    assert out.suggested_line == "ロースで食べる 1,200円"
    assert out.protected_spans[0].text == "1,200円"


def test_guarded_candidate_never_enters_visual_ranking():
    pipe = make_pipeline()
    text = "焼きで食べる"
    mapping = {ch: i + 1 for i, ch in enumerate(dict.fromkeys(text + "ふ目()（）"))}
    logits = torch.full((14, 1, len(mapping) + 1), -12.)
    logits[:, :, 0] = 10.
    for i, ch in enumerate(text):
        logits[i * 2, 0, mapping[ch]] = 12.
    result = pipe.lookup_from_logits(logits, 14, mapping)
    assert result.suggested_line == text
    assert all(not ("ふ" in r.candidate_line) for r in result.visual_rescores)
