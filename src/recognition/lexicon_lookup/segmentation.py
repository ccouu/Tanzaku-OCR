"""Candidate ranking and DP nonoverlapping whole-line segmentation.

Features:
- Explicit score component breakdown (no hidden magic weights, no double-counted LM).
- Protected spans are locked as atomic segments and never crossed or modified.
- Unknown-span fallback: no forced full dictionary coverage; unknown text survives.
- Boundary and 1-char penalties prevent greedy 1-char oversegmentation.
- Ambiguous 2-char exact terms are never overridden by fuzzy frequency.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from .confusables import classify_match_type
from .contracts import (
    CandidateMatch,
    LookupConfig,
    ProtectedSpan,
    SegmentResult,
    SpanMapping,
    TranslationRecord,
)
from .translation_repo import ReadOnlyTranslationRepo
from .trie_search import RawTrieHit


_SEPARATORS = frozenset(" \t\u3000/・|｜,、。:：;；")


def _tier_prior(source_tier: str, config: LookupConfig) -> float:
    if source_tier == "curated_core":
        return float(config.curated_prior_bonus)
    if source_tier == "context_phrase":
        return float(config.context_prior_bonus)
    return float(config.expanded_prior_bonus)


def score_candidate_hit(
    hit: RawTrieHit,
    span_map: SpanMapping,
    config: LookupConfig,
    translation_repo: Optional[ReadOnlyTranslationRepo] = None,
) -> CandidateMatch:
    """Compute explicit score components and wrap a RawTrieHit as a CandidateMatch."""
    span_len = hit.end - hit.start
    is_exact = hit.hard_edits == 0
    coverage = config.char_coverage_reward * span_len
    edit_pen = -config.edit_cost_weight * hit.weighted_edit_cost
    boundary_pen = -config.segment_boundary_penalty
    short_pen = -config.short_term_1char_penalty if span_len == 1 else 0.0
    fuzzy_2char_pen = -config.fuzzy_2char_penalty if (span_len == 2 and not is_exact) else 0.0
    prior = _tier_prior(hit.entry.source_tier, config)

    components = {
        "coverage_reward": round(coverage, 6),
        "edit_penalty": round(edit_pen, 6),
        "boundary_penalty": round(boundary_pen, 6),
        "short_1char_penalty": round(short_pen, 6),
        "fuzzy_2char_penalty": round(fuzzy_2char_pen, 6),
        "source_tier_prior": round(prior, 6),
    }
    total = round(sum(components.values()), 6)

    orig_s, orig_e = span_map.to_original_span(hit.start, hit.end)
    q_span = span_map.canonical_text[hit.start:hit.end]
    raw_q_span = span_map.original_text[orig_s:orig_e]

    trans: Optional[TranslationRecord] = None
    trans_status = "unlinked"
    if translation_repo is not None:
        if hit.entry.db_id is not None:
            trans = translation_repo.get_by_id(hit.entry.db_id)
            trans_status = trans.status
        else:
            trans = translation_repo.get_by_term(hit.entry.canonical_term)
            trans_status = trans.status

    return CandidateMatch(
        start=hit.start,
        end=hit.end,
        orig_start=orig_s,
        orig_end=orig_e,
        query_span=q_span,
        raw_query_span=raw_q_span,
        term=hit.entry.canonical_term,
        display_term=hit.entry.display_term,
        entry_id=hit.entry.entry_id,
        source_tier=hit.entry.source_tier,
        sources=hit.entry.sources,
        db_id=trans.db_id if (trans is not None and trans.db_id is not None) else hit.entry.db_id,
        hard_edits=hit.hard_edits,
        weighted_edit_cost=hit.weighted_edit_cost,
        edit_ops=hit.edit_ops,
        is_exact=is_exact,
        score_components=components,
        total_score=total,
        match_type=classify_match_type(hit.hard_edits, hit.edit_ops),
        translation_status=trans_status,
        translation=trans,
    )


def _cand_sort_key(c: CandidateMatch) -> tuple[object, ...]:
    # Exact always beats fuzzy; higher score beats lower; lower hard_edits; deterministic term tiebreak
    return (not c.is_exact, -c.total_score, c.hard_edits, c.weighted_edit_cost, c.term)


def segment_line_dp(
    span_map: SpanMapping,
    protected_spans: tuple[ProtectedSpan, ...],
    candidates: list[CandidateMatch],
    config: LookupConfig,
    *,
    active_end: Optional[int] = None,
) -> tuple[tuple[SegmentResult, ...], list[CandidateMatch], dict[str, object]]:
    """DP nonoverlapping segmentation covering [0, active_end) + preserved suffix.

    Applies both `max_candidates_per_span` and `max_total_candidates` BEFORE
    constructing DP transitions so the DP strictly respects the configured budget.

    Returns:
        (segments, bounded_ranked_candidates, budget_diagnostics)
    """
    text = span_map.canonical_text
    n = len(text)
    if n == 0:
        return (), [], {
            "candidates_before_cap": 0,
            "candidates_after_cap": 0,
            "candidates_truncated": False,
        }

    dp_limit = n if active_end is None else max(0, min(n, int(active_end)))

    # 1. Bound per-(start, end) span first
    by_span_raw: dict[tuple[int, int], list[CandidateMatch]] = defaultdict(list)
    for cand in candidates:
        if cand.end <= dp_limit:
            by_span_raw[(cand.start, cand.end)].append(cand)

    per_span_kept: list[CandidateMatch] = []
    before_cap = 0
    for group in by_span_raw.values():
        before_cap += len(group)
        group.sort(key=_cand_sort_key)
        per_span_kept.extend(group[: config.max_candidates_per_span])

    # 2. Apply global max_total_candidates cap BEFORE building DP transitions
    per_span_kept.sort(key=_cand_sort_key)
    bounded_all = per_span_kept[: config.max_total_candidates]
    candidates_truncated = len(bounded_all) < before_cap

    by_span: dict[tuple[int, int], list[CandidateMatch]] = defaultdict(list)
    for cand in bounded_all:
        by_span[(cand.start, cand.end)].append(cand)

    pruned_by_start: dict[int, list[tuple[int, CandidateMatch, tuple[CandidateMatch, ...]]]] = defaultdict(list)
    for (s, e), group in sorted(by_span.items()):
        group.sort(key=_cand_sort_key)
        if not group:
            continue
        best = group[0]
        alts = tuple(group[1:])
        pruned_by_start[s].append((e, best, alts))

    prot_by_start: dict[int, ProtectedSpan] = {sp.start: sp for sp in protected_spans}

    # DP arrays: dp[i] = best score covering [0, i) for i in [0, dp_limit]
    neg_inf = -1e18
    dp = [neg_inf] * (dp_limit + 1)
    back: list[Optional[tuple[int, str, object]]] = [None] * (dp_limit + 1)
    dp[0] = 0.0

    for i in range(dp_limit):
        if dp[i] <= neg_inf / 2:
            continue

        # 1. Protected span (locked atomic transition; cannot be skipped or split)
        if i in prot_by_start and prot_by_start[i].end <= dp_limit:
            sp = prot_by_start[i]
            nxt = dp[i] + 0.0
            if nxt > dp[sp.end]:
                dp[sp.end] = nxt
                back[sp.end] = (i, "protected", sp)
            continue

        # 2. Separator / whitespace step
        if text[i] in _SEPARATORS:
            nxt = dp[i] - 0.05
            if nxt > dp[i + 1]:
                dp[i + 1] = nxt
                back[i + 1] = (i, "whitespace", None)
        else:
            # 3. Unknown 1-char fallback (no forced dictionary coverage)
            nxt = dp[i] - config.unknown_char_penalty
            if nxt > dp[i + 1]:
                dp[i + 1] = nxt
                back[i + 1] = (i, "unknown", None)

        # 4. Dictionary candidate transitions starting at i (strictly within budget)
        for end, best, alts in pruned_by_start.get(i, ()):
            cand_kind = "exact" if best.is_exact else "fuzzy_suggestion"
            nxt = dp[i] + best.total_score
            if nxt > dp[end]:
                dp[end] = nxt
                back[end] = (i, cand_kind, (best, alts))

    # Backtrace [0, dp_limit)
    raw_segments: list[SegmentResult] = []
    pos = dp_limit
    while pos > 0:
        step = back[pos]
        assert step is not None, f"DP backtrace broken at position {pos}"
        prev, kind, payload = step
        orig_s, orig_e = span_map.to_original_span(prev, pos)
        seg_text = text[prev:pos]
        raw_text = span_map.original_text[orig_s:orig_e]
        if kind == "protected":
            sp = payload
            assert isinstance(sp, ProtectedSpan)
            raw_segments.append(
                SegmentResult(
                    kind="protected",
                    start=prev,
                    end=pos,
                    orig_start=orig_s,
                    orig_end=orig_e,
                    text=seg_text,
                    raw_text=raw_text,
                    protected_span=sp,
                )
            )
        elif kind in ("exact", "fuzzy_suggestion"):
            best, alts = payload  # type: ignore[misc]
            raw_segments.append(
                SegmentResult(
                    kind=kind,
                    start=prev,
                    end=pos,
                    orig_start=orig_s,
                    orig_end=orig_e,
                    text=seg_text,
                    raw_text=raw_text,
                    selected_candidate=best,
                    alternatives=alts,
                )
            )
        else:
            raw_segments.append(
                SegmentResult(
                    kind=kind,
                    start=prev,
                    end=pos,
                    orig_start=orig_s,
                    orig_end=orig_e,
                    text=seg_text,
                    raw_text=raw_text,
                )
            )
        pos = prev
    raw_segments.reverse()

    # Preserve any unprocessed suffix [dp_limit, n) without dropping text or protected spans
    if dp_limit < n:
        cur = dp_limit
        while cur < n:
            if cur in prot_by_start:
                sp = prot_by_start[cur]
                orig_s, orig_e = span_map.to_original_span(sp.start, sp.end)
                raw_segments.append(
                    SegmentResult(
                        kind="protected",
                        start=sp.start,
                        end=sp.end,
                        orig_start=orig_s,
                        orig_end=orig_e,
                        text=text[sp.start : sp.end],
                        raw_text=span_map.original_text[orig_s:orig_e],
                        protected_span=sp,
                    )
                )
                cur = sp.end
            else:
                orig_s, orig_e = span_map.to_original_span(cur, cur + 1)
                kind_s = "whitespace" if text[cur] in _SEPARATORS else "unknown"
                raw_segments.append(
                    SegmentResult(
                        kind=kind_s,
                        start=cur,
                        end=cur + 1,
                        orig_start=orig_s,
                        orig_end=orig_e,
                        text=text[cur : cur + 1],
                        raw_text=span_map.original_text[orig_s:orig_e],
                    )
                )
                cur += 1

    # Merge contiguous `unknown` and contiguous `whitespace` segments cleanly
    merged: list[SegmentResult] = []
    for seg in raw_segments:
        if (
            merged
            and seg.kind in ("unknown", "whitespace")
            and merged[-1].kind == seg.kind
            and merged[-1].end == seg.start
        ):
            prev_seg = merged[-1]
            new_s, new_e = prev_seg.start, seg.end
            orig_s, orig_e = span_map.to_original_span(new_s, new_e)
            merged[-1] = SegmentResult(
                kind=seg.kind,
                start=new_s,
                end=new_e,
                orig_start=orig_s,
                orig_end=orig_e,
                text=text[new_s:new_e],
                raw_text=span_map.original_text[orig_s:orig_e],
            )
        else:
            merged.append(seg)

    # Attach any span-level fuzzy alternatives to unknown spans that exactly match a candidate span
    final_segments: list[SegmentResult] = []
    for seg in merged:
        if seg.kind == "unknown" and (seg.start, seg.end) in by_span:
            alts = tuple(by_span[(seg.start, seg.end)][: config.max_candidates_per_span])
            final_segments.append(
                SegmentResult(
                    kind="unknown",
                    start=seg.start,
                    end=seg.end,
                    orig_start=seg.orig_start,
                    orig_end=seg.orig_end,
                    text=seg.text,
                    raw_text=seg.raw_text,
                    alternatives=alts,
                )
            )
        else:
            final_segments.append(seg)

    budget_diag: dict[str, object] = {
        "candidates_before_cap": before_cap,
        "candidates_after_cap": len(bounded_all),
        "candidates_truncated": candidates_truncated,
    }
    return tuple(final_segments), bounded_all, budget_diag
