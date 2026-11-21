"""Immutable, versioned, provenance-bearing Japanese OCR confusable catalog.

Strictly separates two categories of substitution hypotheses:
1. `cross_script_lookalike`: Small research-supported cross-script lookalike set
   (Sharp handwriting guide, ICANN72 RZ-LGR Japanese-script p31 [search-extracted],
   Casio EX-word manual p196). Font-dependent visual similarity is NOT Unicode
   normalization, NOT universal glyph identity, and NOT transitive equivalence.
2. `ordinary_ocr_confusion`: Ordinary menu-OCR confusions (katakana shape pairs
   like シ/ツ and ソ/ン, small/large kana, dakuten/handakuten, distinct kanji
   like 大/太). Retained as ordinary fuzzy hypotheses, NEVER labeled visual-only.

All substitution costs are engineering heuristics (strictly > 0; no zero-cost
edits, no removal of true hard Levenshtein edit counts, no transitive closure).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from types import MappingProxyType
from typing import Any, Mapping, Optional


CONFUSABLE_CATALOG_VERSION = 1
CONFUSABLE_CATALOG_FORMAT = "jp_menu_confusables_v1"

CATEGORY_CROSS_SCRIPT = "cross_script_lookalike"
CATEGORY_ORDINARY_OCR = "ordinary_ocr_confusion"

MATCH_TYPE_EXACT = "exact"
MATCH_TYPE_VISUAL_CONFUSABLE = "visual_confusable_suggestion"
MATCH_TYPE_ORDINARY_FUZZY = "ordinary_fuzzy_suggestion"

SOURCE_BIBLIOGRAPHY: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "sharp_pct1_ss_guide": MappingProxyType(
            {
                "title": "Sharp handwriting-recognition guide (pct1_ss_guide.pdf)",
                "url": "https://jp.sharp/support/mit/doc/pct1_ss_guide.pdf",
                "provenance": "primary_pdf_verified_for_kana_kanji_similar_forms",
            }
        ),
        "icann72_jp_lgr_p31": MappingProxyType(
            {
                "title": "ICANN72 RZ-LGR Japanese-script visual-confusables presentation (p.31)",
                "url": "https://archive.icann.org/meetings/icann72/files/content%3Dt_attachment%2Cf_%22ICANN72_RZ-LGR-14oct21-v2.pdf%22/ICANN72_RZ-LGR-14oct21-v2.pdf",
                "provenance": "search_engine_extracted_table_direct_fetch_cache_miss",
            }
        ),
        "casio_exword_p196": MappingProxyType(
            {
                "title": "Casio EX-word manual (EuGFSF_WB.pdf, p.196)",
                "url": "https://www.casio.com/content/dam/casio/global/support/manuals/exword/pdf/003-ja/eugf/EuGFSF_WB.pdf",
                "provenance": "extracted_text_corroborating_contextual_kanji_kana_selection",
            }
        ),
        "unicode_uts39_non_goal": MappingProxyType(
            {
                "title": "Unicode Technical Standard #39 (Unicode Security Mechanisms)",
                "url": "https://www.unicode.org/reports/tr39/",
                "provenance": "non_goal_reference_security_only_not_ocr_normalization",
            }
        ),
        "repo_ocr_heuristic": MappingProxyType(
            {
                "title": "Repository menu OCR error-analysis heuristic",
                "url": "",
                "provenance": "engineering_heuristic_uncalibrated_not_external_standard",
            }
        ),
    }
)


def codepoint_hex(ch: str) -> str:
    if not ch:
        return ""
    return f"U+{ord(ch):04X}"


@dataclass(frozen=True)
class ConfusablePair:
    """One symmetric, non-transitive substitution hypothesis."""

    char_a: str
    char_b: str
    codepoint_a: str
    codepoint_b: str
    pair_id: str
    category: str  # cross_script_lookalike | ordinary_ocr_confusion
    default_heuristic_cost: float
    sources: tuple[str, ...]
    numeral_risk: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pair(
    a: str,
    b: str,
    *,
    category: str,
    cost: float,
    sources: tuple[str, ...],
    numeral_risk: bool = False,
    note: str = "",
) -> ConfusablePair:
    if len(a) != 1 or len(b) != 1 or a == b:
        raise ValueError(f"Invalid confusable pair: ({a!r}, {b!r})")
    if cost <= 0.0 or cost >= 1.0:
        raise ValueError(f"Heuristic cost must be strictly in (0, 1), got {cost}")
    ca, cb = codepoint_hex(a), codepoint_hex(b)
    pid = f"{ca}/{cb}"
    return ConfusablePair(
        char_a=a,
        char_b=b,
        codepoint_a=ca,
        codepoint_b=cb,
        pair_id=pid,
        category=category,
        default_heuristic_cost=float(cost),
        sources=tuple(sources),
        numeral_risk=bool(numeral_risk),
        note=note,
    )


# 1. Research-supported small cross-script lookalike set (no transitive closure).
_CROSS_SCRIPT_LOOKALIKES: tuple[ConfusablePair, ...] = (
    _pair("エ", "工", category=CATEGORY_CROSS_SCRIPT, cost=0.45, sources=("sharp_pct1_ss_guide", "icann72_jp_lgr_p31", "casio_exword_p196")),
    _pair("カ", "力", category=CATEGORY_CROSS_SCRIPT, cost=0.45, sources=("sharp_pct1_ss_guide", "icann72_jp_lgr_p31", "casio_exword_p196")),
    _pair("タ", "夕", category=CATEGORY_CROSS_SCRIPT, cost=0.45, sources=("sharp_pct1_ss_guide", "icann72_jp_lgr_p31", "casio_exword_p196")),
    _pair("ト", "卜", category=CATEGORY_CROSS_SCRIPT, cost=0.45, sources=("sharp_pct1_ss_guide", "icann72_jp_lgr_p31")),
    _pair("ニ", "二", category=CATEGORY_CROSS_SCRIPT, cost=0.45, sources=("sharp_pct1_ss_guide", "icann72_jp_lgr_p31"), numeral_risk=True, note="Kanji numeral two; preserve valid kanji/count words."),
    _pair("ハ", "八", category=CATEGORY_CROSS_SCRIPT, cost=0.45, sources=("sharp_pct1_ss_guide", "icann72_jp_lgr_p31"), numeral_risk=True, note="Kanji numeral eight; preserve valid kanji/count words."),
    _pair("ロ", "口", category=CATEGORY_CROSS_SCRIPT, cost=0.45, sources=("sharp_pct1_ss_guide", "icann72_jp_lgr_p31", "casio_exword_p196")),
    _pair("へ", "ヘ", category=CATEGORY_CROSS_SCRIPT, cost=0.45, sources=("sharp_pct1_ss_guide", "icann72_jp_lgr_p31")),
    _pair("べ", "ベ", category=CATEGORY_CROSS_SCRIPT, cost=0.45, sources=("icann72_jp_lgr_p31",)),
    _pair("ぺ", "ペ", category=CATEGORY_CROSS_SCRIPT, cost=0.45, sources=("icann72_jp_lgr_p31",)),
    _pair(
        "ー",
        "一",
        category=CATEGORY_CROSS_SCRIPT,
        cost=0.45,
        sources=("icann72_jp_lgr_p31",),
        numeral_risk=True,
        note="Prolonged sound mark vs kanji numeral one; requires multi-character lexicon context.",
    ),
)

# 2. Ordinary OCR-confusion hypotheses (NOT visual-equivalent aliases).
_ORDINARY_OCR_RAW: tuple[tuple[str, str], ...] = (
    ("ー", "-"),
    ("ー", "―"),
    ("シ", "ツ"),
    ("ソ", "ン"),
    ("リ", "り"),
    ("ヶ", "ケ"),
    ("ッ", "ツ"),
    ("ャ", "ヤ"),
    ("ュ", "ユ"),
    ("ョ", "ヨ"),
    ("ァ", "ア"),
    ("ィ", "イ"),
    ("ゥ", "ウ"),
    ("ェ", "エ"),
    ("ォ", "オ"),
    ("バ", "パ"),
    ("ビ", "ピ"),
    ("ブ", "プ"),
    ("ベ", "ペ"),
    ("ボ", "ポ"),
    ("ダ", "タ"),
    ("デ", "テ"),
    ("ド", "ト"),
    ("ガ", "カ"),
    ("ギ", "キ"),
    ("グ", "ク"),
    ("ゲ", "ケ"),
    ("ゴ", "コ"),
    ("ザ", "サ"),
    ("ジ", "シ"),
    ("ズ", "ス"),
    ("ゼ", "セ"),
    ("ゾ", "ソ"),
    ("ば", "ぱ"),
    ("び", "ぴ"),
    ("ぶ", "ぷ"),
    ("べ", "ぺ"),
    ("ぼ", "ぽ"),
    ("だ", "た"),
    ("で", "て"),
    ("ど", "と"),
    ("が", "か"),
    ("ぎ", "き"),
    ("ぐ", "く"),
    ("げ", "け"),
    ("ご", "こ"),
    ("ざ", "さ"),
    ("じ", "し"),
    ("ず", "す"),
    ("ぜ", "せ"),
    ("ぞ", "そ"),
    ("大", "太"),
    ("土", "士"),
    ("日", "目"),
    ("白", "百"),
    ("未", "末"),
    ("人", "入"),
    ("牛", "午"),
    ("鳥", "烏"),
)

_ORDINARY_OCR_CONFUSIONS: tuple[ConfusablePair, ...] = tuple(
    _pair(a, b, category=CATEGORY_ORDINARY_OCR, cost=0.65, sources=("repo_ocr_heuristic",))
    for a, b in _ORDINARY_OCR_RAW
)


class ConfusableCatalog:
    """Immutable symmetric lookup table with explicit category & source provenance."""

    __slots__ = ("_pairs", "_lookup", "_pair_set")

    def __init__(self, pairs: tuple[ConfusablePair, ...]) -> None:
        table: dict[tuple[str, str], ConfusablePair] = {}
        for p in pairs:
            for key in ((p.char_a, p.char_b), (p.char_b, p.char_a)):
                if key in table:
                    raise ValueError(f"Duplicate confusable pair in catalog: {key!r}")
                table[key] = p
        object.__setattr__(self, "_pairs", tuple(pairs))
        object.__setattr__(self, "_lookup", MappingProxyType(table))
        object.__setattr__(self, "_pair_set", frozenset(table.keys()))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("ConfusableCatalog is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("ConfusableCatalog is immutable")

    @property
    def pairs(self) -> tuple[ConfusablePair, ...]:
        return self._pairs

    @property
    def pair_set(self) -> frozenset[tuple[str, str]]:
        return self._pair_set

    def lookup(self, a: str, b: str) -> Optional[ConfusablePair]:
        """Direct symmetric lookup only; NO transitive closure."""
        if a == b:
            return None
        return self._lookup.get((a, b))

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_version": CONFUSABLE_CATALOG_VERSION,
            "catalog_format": CONFUSABLE_CATALOG_FORMAT,
            "transitive_closure": False,
            "zero_cost_edits_allowed": False,
            "heuristic_disclaimer": (
                "Substitution costs are uncalibrated engineering heuristics for bounded "
                "candidate ranking only; no external source establishes menu OCR rates."
            ),
            "uts39_note": (
                "Unicode UTS39 is for identifier security detection, not semantic "
                "normalization or OCR accuracy; this catalog does not import UTS39 "
                "skeletons or claim UTS39 compliance."
            ),
            "counts": {
                CATEGORY_CROSS_SCRIPT: sum(1 for p in self._pairs if p.category == CATEGORY_CROSS_SCRIPT),
                CATEGORY_ORDINARY_OCR: sum(1 for p in self._pairs if p.category == CATEGORY_ORDINARY_OCR),
                "total_pairs": len(self._pairs),
            },
            "bibliography": {k: dict(v) for k, v in SOURCE_BIBLIOGRAPHY.items()},
            "pairs": [p.to_dict() for p in self._pairs],
        }


DEFAULT_CONFUSABLE_CATALOG = ConfusableCatalog(_CROSS_SCRIPT_LOOKALIKES + _ORDINARY_OCR_CONFUSIONS)


def classify_match_type(hard_edits: int, edit_ops: tuple[Any, ...]) -> str:
    """Classify a candidate match strictly from its true edit operations.

    - `exact`: ONLY when hard_edits == 0 (literal canonical equality).
    - `visual_confusable_suggestion`: ONLY when hard_edits > 0 and EVERY non-match
      operation is a substitution whose category is `cross_script_lookalike`.
      Mixed edits, ordinary OCR confusions, or any insert/delete are NEVER
      labeled `visual_confusable_suggestion`.
    - `ordinary_fuzzy_suggestion`: all other non-exact matches.
    """
    if hard_edits == 0:
        return MATCH_TYPE_EXACT
    non_matches = [op for op in edit_ops if op.op != "match"]
    if not non_matches:
        return MATCH_TYPE_EXACT
    if all(op.op == "substitute" and op.confusion_category == CATEGORY_CROSS_SCRIPT for op in non_matches):
        return MATCH_TYPE_VISUAL_CONFUSABLE
    return MATCH_TYPE_ORDINARY_FUZZY
