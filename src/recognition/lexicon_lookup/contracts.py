"""Immutable data contracts and per-request configuration for lexicon lookup.

All span offsets (`start`, `end`, `orig_start`, `orig_end`) use Python Unicode
code-point indices (`str` slice coordinates, half-open `[start, end)`).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass(frozen=True)
class EditPolicyConfig:
    """Conservative edit-distance bounds by span length."""

    len1_max_edits: int = 0
    len2_max_edits: int = 1
    len3_5_max_edits: int = 1
    len6plus_max_edits: int = 2
    conservative_boundary_enforcement: bool = True

    def allowed_edits_for_length(self, length: int) -> int:
        if length <= 1:
            return max(0, int(self.len1_max_edits))
        if length == 2:
            return max(0, int(self.len2_max_edits))
        if length <= 5:
            return max(0, int(self.len3_5_max_edits))
        return max(0, int(self.len6plus_max_edits))

    def max_allowed_edits(self, query_len: int, candidate_len: int) -> int:
        q_lim = self.allowed_edits_for_length(query_len)
        c_lim = self.allowed_edits_for_length(candidate_len)
        if self.conservative_boundary_enforcement:
            return min(q_lim, c_lim)
        return q_lim


@dataclass(frozen=True)
class LookupConfig:
    """Per-request immutable configuration for lexicon trie search and ranking."""

    edit_policy: EditPolicyConfig = field(default_factory=EditPolicyConfig)
    use_confusion_weights: bool = True
    confusion_sub_cost: float = 0.45
    ordinary_confusion_sub_cost: float = 0.65
    max_candidates_per_span: int = 8
    max_total_candidates: int = 160
    max_query_length: int = 128
    max_span_length: int = 24
    max_trie_node_visits: int = 60000
    include_context_phrases: bool = True
    include_expanded_culinary: bool = True
    enable_visual_rescoring: bool = True
    max_visual_rescore_lines: int = 6
    auto_correct: bool = False  # Suggestion-only by default; never silently rewrite.
    protect_particles: bool = True
    # Explicit DP segmentation & ranking weights (unknown baseline = 0.0 per char)
    char_coverage_reward: float = 1.00
    unknown_char_penalty: float = 0.00
    segment_boundary_penalty: float = 0.45
    short_term_1char_penalty: float = 0.40
    fuzzy_2char_penalty: float = 0.80
    edit_cost_weight: float = 2.40
    curated_prior_bonus: float = 0.12
    context_prior_bonus: float = 0.06
    expanded_prior_bonus: float = 0.0
    visual_ctc_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.max_candidates_per_span < 1:
            raise ValueError(f"max_candidates_per_span must be >= 1, got {self.max_candidates_per_span}")
        if self.max_total_candidates < 1:
            raise ValueError(f"max_total_candidates must be >= 1, got {self.max_total_candidates}")
        if self.max_query_length < 1:
            raise ValueError(f"max_query_length must be >= 1, got {self.max_query_length}")
        if self.max_span_length < 1:
            raise ValueError(f"max_span_length must be >= 1, got {self.max_span_length}")
        if self.max_trie_node_visits < 1:
            raise ValueError(f"max_trie_node_visits must be >= 1, got {self.max_trie_node_visits}")
        if self.max_visual_rescore_lines < 1:
            raise ValueError(f"max_visual_rescore_lines must be >= 1, got {self.max_visual_rescore_lines}")
        if not (0.0 < self.confusion_sub_cost <= 1.0):
            raise ValueError(f"confusion_sub_cost must be in (0, 1], got {self.confusion_sub_cost}")
        if not (0.0 < self.ordinary_confusion_sub_cost <= 1.0):
            raise ValueError(f"ordinary_confusion_sub_cost must be in (0, 1], got {self.ordinary_confusion_sub_cost}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpanMapping:
    """Maps canonical Unicode code-point offsets back to original input offsets."""

    original_text: str
    canonical_text: str
    # For each canonical index i in [0, len(canonical_text)),
    # original code points are [orig_starts[i], orig_ends[i]).
    orig_starts: tuple[int, ...]
    orig_ends: tuple[int, ...]
    changes: tuple[dict[str, Any], ...] = ()

    def to_original_span(self, canon_start: int, canon_end: int) -> tuple[int, int]:
        n = len(self.canonical_text)
        if canon_start < 0 or canon_end < canon_start or canon_end > n:
            raise ValueError(f"Invalid canonical span [{canon_start}, {canon_end}) for length {n}")
        if canon_start == canon_end:
            if canon_start == n:
                end_pos = len(self.original_text)
                return end_pos, end_pos
            pos = self.orig_starts[canon_start]
            return pos, pos
        return self.orig_starts[canon_start], self.orig_ends[canon_end - 1]


@dataclass(frozen=True)
class ProtectedSpan:
    """Contiguous structured span protected from fuzzy editing or splitting."""

    start: int
    end: int
    orig_start: int
    orig_end: int
    text: str
    raw_text: str
    kind: str  # price | currency_amount | percent | count | numeric_sequence
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EditOp:
    """Single alignment operation between query span and dictionary candidate."""

    op: str  # match | substitute | insert | delete
    query_pos: Optional[int]
    cand_pos: Optional[int]
    query_char: str
    cand_char: str
    cost: float
    confusion_pair: bool = False
    query_codepoint: str = ""
    cand_codepoint: str = ""
    confusion_category: str = ""  # "" | cross_script_lookalike | ordinary_ocr_confusion
    confusion_pair_id: str = ""
    confusion_sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LexiconEntry:
    """Immutable provenance-bearing lexicon entry."""

    entry_id: int
    canonical_term: str
    display_term: str
    source_tier: str  # curated_core | context_phrase | jmdict_food_cleaned
    sources: tuple[str, ...]
    category: str = ""
    standalone_tier: str = ""
    occurrence_tier: str = ""
    db_id: Optional[int] = None
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TranslationRecord:
    """Read-only translation record from culinary_menu.db (or explicit missing)."""

    status: str  # found | missing | unlinked
    db_id: Optional[int] = None
    canonical_jp: str = ""
    reading_hiragana: str = ""
    romaji: str = ""
    name_en: str = ""
    description_en: str = ""
    category: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateMatch:
    """One exact or fuzzy dictionary match for a canonical input span."""

    start: int
    end: int
    orig_start: int
    orig_end: int
    query_span: str
    raw_query_span: str
    term: str
    display_term: str
    entry_id: int
    source_tier: str
    sources: tuple[str, ...]
    db_id: Optional[int]
    hard_edits: int
    weighted_edit_cost: float
    edit_ops: tuple[EditOp, ...]
    is_exact: bool
    score_components: dict[str, float]
    total_score: float
    match_type: str = "ordinary_fuzzy_suggestion"  # exact | visual_confusable_suggestion | ordinary_fuzzy_suggestion
    translation_status: str = "missing"
    translation: Optional[TranslationRecord] = None
    suppression_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass(frozen=True)
class SegmentResult:
    """One nonoverlapping segment covering part of the canonical input line."""

    kind: str  # exact | fuzzy_suggestion | protected | unknown | whitespace
    start: int
    end: int
    orig_start: int
    orig_end: int
    text: str
    raw_text: str
    selected_candidate: Optional[CandidateMatch] = None
    alternatives: tuple[CandidateMatch, ...] = ()
    protected_span: Optional[ProtectedSpan] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualRescoreResult:
    """Whole-line CTC forward log-probability score & combined text+CTC rank."""

    candidate_line: str
    scorable: bool
    ctc_log_prob: Optional[float]
    greedy_ctc_log_prob: Optional[float]
    delta_log_prob: Optional[float]
    unsupported_chars: tuple[str, ...] = ()
    reason: str = ""
    text_score: Optional[float] = None
    visual_ctc_weight: Optional[float] = None
    combined_score: Optional[float] = None
    is_unchanged_ocr: bool = False
    rank: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LookupResult:
    """Complete structured output for review, CLI, evaluation, and inspection UI."""

    raw_ocr: str
    canonical_ocr: str
    selected_text_for_lookup: str
    decoder_mode: str
    decoder_config: dict[str, Any]
    lm_alternative: Optional[str]
    protected_spans: tuple[ProtectedSpan, ...]
    segments: tuple[SegmentResult, ...]
    fuzzy_suggestions: tuple[CandidateMatch, ...]
    exact_translations: tuple[dict[str, Any], ...]
    visual_rescores: tuple[VisualRescoreResult, ...]
    suggested_line: str
    warnings: tuple[str, ...]
    diagnostics: dict[str, Any]
    provenance: dict[str, Any]
    visual_reranked_line: Optional[str] = None
    auto_corrected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
