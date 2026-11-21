"""Narrow canonical normalization with exact Unicode code-point span mappings.

Only agreed canonical aliases are normalized:
- Full-width decimal digits (０-９ -> 0-9) via `label_normalization.normalize_digits`
- Full-width parentheses (（ -> (, ） -> )) via `label_normalization.normalize_parentheses`

Visually similar kanji/kana (e.g. ー vs 一, ロ vs 口, カ vs 力) are NEVER globally
replaced here; they remain distinct characters and may only receive optional
weighted substitution costs during bounded fuzzy trie search.
"""
from __future__ import annotations

from ..label_normalization import normalize_digits, normalize_parentheses
from .contracts import SpanMapping


def canonicalize_text(text: str) -> str:
    """Apply narrow agreed canonicalization only."""
    return normalize_parentheses(normalize_digits(text))


def normalize_with_span_map(text: str) -> SpanMapping:
    """Normalize text while retaining exact per-character original code-point spans.

    Supports length-changing character mappings cleanly by recording the exact
    `[orig_start, orig_end)` interval for every emitted canonical code point.
    """
    canon_chars: list[str] = []
    orig_starts: list[int] = []
    orig_ends: list[int] = []
    changes: list[dict[str, object]] = []

    for orig_idx, ch in enumerate(text):
        mapped = canonicalize_text(ch)
        if mapped != ch:
            changes.append(
                {
                    "orig_start": orig_idx,
                    "orig_end": orig_idx + 1,
                    "canon_start": len(canon_chars),
                    "canon_end": len(canon_chars) + len(mapped),
                    "from": ch,
                    "to": mapped,
                }
            )
        for out_ch in mapped:
            canon_chars.append(out_ch)
            orig_starts.append(orig_idx)
            orig_ends.append(orig_idx + 1)

    return SpanMapping(
        original_text=text,
        canonical_text="".join(canon_chars),
        orig_starts=tuple(orig_starts),
        orig_ends=tuple(orig_ends),
        changes=tuple(changes),
    )
