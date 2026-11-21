"""Contiguous structured protected-span parser for prices, counts, percents, and numbers.

Protected spans are parsed on canonical text BEFORE fuzzy matching. Candidates
and segmentation cannot cross or modify protected spans. Size/rank characters
(e.g. 大, 中, 小, S, M, L, 特上) are NOT automatically treated as numeric spans
without an adjacent numeric/structured context.
"""
from __future__ import annotations

import re
from .contracts import ProtectedSpan, SpanMapping


# Contiguous numeric core: 1,200 | 500 | 8.5 | 1,200.50
_NUM = r"\d+(?:,\d{3})*(?:\.\d+)?"
_RANGE_SEP = r"(?:[~〜～\-－–—])"
_NUM_OR_RANGE = rf"{_NUM}(?:{_RANGE_SEP}{_NUM})?"

# Ordered from most specific structured expressions to general numeric sequences.
_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    # Price with currency prefix (optional range + optional tax/suffix marker)
    (
        "price",
        "currency_prefix_amount",
        re.compile(
            rf"[¥￥$€]\s*{_NUM}(?:\s*{_RANGE_SEP}\s*(?:[¥￥$€]\s*)?{_NUM})?"
            rf"(?:\s*(?:円|YEN|yen))?(?:\s*[~〜～＋\+])?"
            rf"(?:\s*\((?:税込|税抜|税別|各|込|別)\))?"
        ),
    ),
    # Price with Japanese yen suffix (e.g. 1,200円, 800~1,200円, 500円(税込))
    (
        "price",
        "yen_suffix_amount",
        re.compile(
            rf"{_NUM_OR_RANGE}\s*(?:円|YEN|yen)(?:\s*[~〜～＋\+])?"
            rf"(?:\s*\((?:税込|税抜|税別|各|込|別)\))?"
        ),
    ),
    # Percentage expressions (e.g. 10%, 8.5％, 100%)
    (
        "percent",
        "percentage_expression",
        re.compile(rf"{_NUM_OR_RANGE}\s*[%％]"),
    ),
    # Count / quantity / volume / weight / serving expressions
    (
        "count",
        "quantity_or_unit_expression",
        re.compile(
            rf"{_NUM_OR_RANGE}\s*(?:"
            r"人前|名様|名|品|点|種|個|本|枚|切(?:れ)?|貫|巻|皿|杯|合|升|丁|串|尾|匹|玉|束|缶|瓶|袋|箱|席|卓|時間|分|秒|"
            r"ml|mL|ML|cc|CC|kg|KG|mg|MG|g|G|L|ℓ"
            r")"
        ),
    ),
    # Time / date expressions (e.g. 11:30, 11:30~14:00, 2026/11/19)
    (
        "numeric_sequence",
        "time_or_date_expression",
        re.compile(
            rf"\d{{1,2}}:\d{{2}}(?:{_RANGE_SEP}\d{{1,2}}:\d{{2}})?|"
            r"\d{2,4}/\d{1,2}(?:/\d{1,2})?"
        ),
    ),
    # Standalone contiguous numeric sequences / ranges (e.g. 1,200 or 500~800)
    (
        "numeric_sequence",
        "contiguous_numeric_sequence",
        re.compile(rf"{_NUM_OR_RANGE}(?:[~〜～])?"),
    ),
)


def extract_protected_spans(span_map: SpanMapping) -> tuple[ProtectedSpan, ...]:
    """Extract nonoverlapping contiguous protected spans in canonical order."""
    text = span_map.canonical_text
    if not text:
        return ()

    occupied = [False] * len(text)
    found: list[ProtectedSpan] = []

    for kind, reason, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            s, e = match.span()
            if s >= e:
                continue
            if any(occupied[i] for i in range(s, e)):
                continue
            for i in range(s, e):
                occupied[i] = True
            orig_s, orig_e = span_map.to_original_span(s, e)
            found.append(
                ProtectedSpan(
                    start=s,
                    end=e,
                    orig_start=orig_s,
                    orig_end=orig_e,
                    text=text[s:e],
                    raw_text=span_map.original_text[orig_s:orig_e],
                    kind=kind,
                    reason=reason,
                )
            )

    found.sort(key=lambda sp: (sp.start, sp.end))
    return tuple(found)


def overlaps_protected(start: int, end: int, protected_spans: tuple[ProtectedSpan, ...]) -> bool:
    """Return True if [start, end) overlaps any protected span."""
    for sp in protected_spans:
        if sp.end <= start:
            continue
        if sp.start >= end:
            break
        return True
    return False
