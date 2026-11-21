"""Modular Python lexicon-trie lookup pipeline for Japanese menu OCR inspection."""
from .confusables import (
    CATEGORY_CROSS_SCRIPT,
    CATEGORY_ORDINARY_OCR,
    CONFUSABLE_CATALOG_FORMAT,
    CONFUSABLE_CATALOG_VERSION,
    DEFAULT_CONFUSABLE_CATALOG,
    MATCH_TYPE_EXACT,
    MATCH_TYPE_ORDINARY_FUZZY,
    MATCH_TYPE_VISUAL_CONFUSABLE,
    SOURCE_BIBLIOGRAPHY,
    ConfusableCatalog,
    ConfusablePair,
    classify_match_type,
    codepoint_hex,
)
from .contracts import (
    CandidateMatch,
    EditOp,
    EditPolicyConfig,
    LexiconEntry,
    LookupConfig,
    LookupResult,
    ProtectedSpan,
    SegmentResult,
    SpanMapping,
    TranslationRecord,
    VisualRescoreResult,
)
from .ctc_scorer import (
    ctc_forward_log_prob_from_log_probs,
    prepare_canonical_log_probs,
    score_candidate_line_ctc,
)
from .index import (
    INDEX_FORMAT,
    INDEX_SCHEMA_VERSION,
    LexiconIndex,
    build_lexicon_index,
    sha256_file,
)
from .menu_jobs import MenuJobManager
from .menu_pipeline import (
    DEFAULT_V5_YOLO_WEIGHTS,
    MAX_CROPS_PER_MENU,
    MAX_MENU_PIXELS,
    MAX_MENU_UPLOAD_BYTES,
    V5YoloDetector,
    decode_uploaded_image_bytes,
    extract_menu_crops,
    make_config_signature,
    run_full_menu_pipeline,
)
from .normalization import canonicalize_text, normalize_with_span_map
from .pipeline import DEFAULT_R1_DECODE_PARAMS, LexiconPipeline, rank_whole_lines_with_ctc
from .protected_spans import extract_protected_spans, overlaps_protected
from .segmentation import score_candidate_hit, segment_line_dp
from .translation_repo import ReadOnlyTranslationRepo
from .trie_search import (
    CONFUSION_PAIR_SET,
    RawTrieHit,
    align_edit_ops,
    search_line_substrings,
    search_span_exact_and_fuzzy,
)

__all__ = [
    "CATEGORY_CROSS_SCRIPT",
    "CATEGORY_ORDINARY_OCR",
    "CONFUSABLE_CATALOG_FORMAT",
    "CONFUSABLE_CATALOG_VERSION",
    "CONFUSION_PAIR_SET",
    "DEFAULT_CONFUSABLE_CATALOG",
    "DEFAULT_R1_DECODE_PARAMS",
    "INDEX_FORMAT",
    "INDEX_SCHEMA_VERSION",
    "MATCH_TYPE_EXACT",
    "MATCH_TYPE_ORDINARY_FUZZY",
    "MATCH_TYPE_VISUAL_CONFUSABLE",
    "SOURCE_BIBLIOGRAPHY",
    "CandidateMatch",
    "ConfusableCatalog",
    "ConfusablePair",
    "EditOp",
    "EditPolicyConfig",
    "LexiconEntry",
    "LexiconIndex",
    "LexiconPipeline",
    "LookupConfig",
    "LookupResult",
    "ProtectedSpan",
    "RawTrieHit",
    "ReadOnlyTranslationRepo",
    "SegmentResult",
    "SpanMapping",
    "TranslationRecord",
    "VisualRescoreResult",
    "align_edit_ops",
    "build_lexicon_index",
    "canonicalize_text",
    "ctc_forward_log_prob_from_log_probs",
    "extract_protected_spans",
    "normalize_with_span_map",
    "overlaps_protected",
    "prepare_canonical_log_probs",
    "score_candidate_hit",
    "score_candidate_line_ctc",
    "search_line_substrings",
    "search_span_exact_and_fuzzy",
    "segment_line_dp",
    "sha256_file",
]
