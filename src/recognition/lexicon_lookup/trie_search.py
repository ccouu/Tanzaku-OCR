"""Levenshtein Trie traversal with DP rows, pruning, and exact EditOp backtrace.

Handles substitutions, insertions, deletions, and character repetitions.
Returns both unweighted `hard_edits` and optional `weighted_edit_cost` using
the versioned, provenance-bearing `ConfusableCatalog` (distinguishing
`cross_script_lookalike` from `ordinary_ocr_confusion`, with zero transitive
closure and strictly positive heuristic costs).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .confusables import (
    CATEGORY_CROSS_SCRIPT,
    ConfusableCatalog,
    ConfusablePair,
    DEFAULT_CONFUSABLE_CATALOG,
    codepoint_hex,
)
from .contracts import EditOp, EditPolicyConfig, LexiconEntry, LookupConfig
from .index import LexiconIndex, TrieNode


# Backward-compatible export of the symmetric confusable pair set.
CONFUSION_PAIR_SET: frozenset[tuple[str, str]] = DEFAULT_CONFUSABLE_CATALOG.pair_set


def substitution_cost(
    a: str,
    b: str,
    *,
    use_confusion_weights: bool = True,
    confusion_sub_cost: float = 0.45,
    ordinary_confusion_sub_cost: float = 0.65,
    catalog: ConfusableCatalog = DEFAULT_CONFUSABLE_CATALOG,
) -> tuple[float, Optional[ConfusablePair]]:
    if a == b:
        return 0.0, None
    pair = catalog.lookup(a, b)
    if pair is None:
        return 1.0, None
    if not use_confusion_weights:
        return 1.0, pair
    if pair.category == CATEGORY_CROSS_SCRIPT:
        return float(confusion_sub_cost), pair
    return float(ordinary_confusion_sub_cost), pair


def align_edit_ops(
    query: str,
    candidate: str,
    *,
    use_confusion_weights: bool = True,
    confusion_sub_cost: float = 0.45,
    ordinary_confusion_sub_cost: float = 0.65,
    catalog: ConfusableCatalog = DEFAULT_CONFUSABLE_CATALOG,
) -> tuple[int, float, tuple[EditOp, ...]]:
    """Compute exact Levenshtein hard edits, weighted cost, and EditOp sequence.

    Optimizes primarily on (hard_edits, weighted_cost) so hard_edits is ALWAYS
    guaranteed to be the true unweighted Levenshtein distance.
    """
    m, n = len(query), len(candidate)
    inf_h = m + n + 10
    inf_w = float(inf_h)
    dp_h = [[inf_h] * (n + 1) for _ in range(m + 1)]
    dp_w = [[inf_w] * (n + 1) for _ in range(m + 1)]
    back: list[list[Optional[tuple[str, Optional[ConfusablePair], float]]]] = [
        [None] * (n + 1) for _ in range(m + 1)
    ]

    dp_h[0][0] = 0
    dp_w[0][0] = 0.0
    for i in range(1, m + 1):
        dp_h[i][0] = i
        dp_w[i][0] = float(i)
        back[i][0] = ("delete", None, 1.0)
    for j in range(1, n + 1):
        dp_h[0][j] = j
        dp_w[0][j] = float(j)
        back[0][j] = ("insert", None, 1.0)

    for i in range(1, m + 1):
        qi = query[i - 1]
        for j in range(1, n + 1):
            cj = candidate[j - 1]
            sub_w, conf_pair = substitution_cost(
                qi,
                cj,
                use_confusion_weights=use_confusion_weights,
                confusion_sub_cost=confusion_sub_cost,
                ordinary_confusion_sub_cost=ordinary_confusion_sub_cost,
                catalog=catalog,
            )
            sub_h = 0 if qi == cj else 1
            best_h = dp_h[i - 1][j - 1] + sub_h
            best_w = dp_w[i - 1][j - 1] + sub_w
            best_op: tuple[str, Optional[ConfusablePair], float] = (
                "match" if sub_h == 0 else "substitute",
                conf_pair,
                sub_w,
            )

            del_h = dp_h[i - 1][j] + 1
            del_w = dp_w[i - 1][j] + 1.0
            if (del_h, del_w) < (best_h, best_w):
                best_h, best_w, best_op = del_h, del_w, ("delete", None, 1.0)

            ins_h = dp_h[i][j - 1] + 1
            ins_w = dp_w[i][j - 1] + 1.0
            if (ins_h, ins_w) < (best_h, best_w):
                best_h, best_w, best_op = ins_h, ins_w, ("insert", None, 1.0)

            dp_h[i][j] = best_h
            dp_w[i][j] = best_w
            back[i][j] = best_op

    ops: list[EditOp] = []
    i, j = m, n
    while i > 0 or j > 0:
        op_info = back[i][j]
        assert op_info is not None
        op, conf_pair, cost = op_info
        if op in ("match", "substitute"):
            qc, cc = query[i - 1], candidate[j - 1]
            ops.append(
                EditOp(
                    op=op,
                    query_pos=i - 1,
                    cand_pos=j - 1,
                    query_char=qc,
                    cand_char=cc,
                    cost=cost,
                    confusion_pair=conf_pair is not None,
                    query_codepoint=codepoint_hex(qc),
                    cand_codepoint=codepoint_hex(cc),
                    confusion_category=conf_pair.category if conf_pair is not None else "",
                    confusion_pair_id=conf_pair.pair_id if conf_pair is not None else "",
                    confusion_sources=conf_pair.sources if conf_pair is not None else (),
                )
            )
            i -= 1
            j -= 1
        elif op == "delete":
            qc = query[i - 1]
            ops.append(
                EditOp(
                    op="delete",
                    query_pos=i - 1,
                    cand_pos=None,
                    query_char=qc,
                    cand_char="",
                    cost=cost,
                    confusion_pair=False,
                    query_codepoint=codepoint_hex(qc),
                    cand_codepoint="",
                )
            )
            i -= 1
        else:
            cc = candidate[j - 1]
            ops.append(
                EditOp(
                    op="insert",
                    query_pos=None,
                    cand_pos=j - 1,
                    query_char="",
                    cand_char=cc,
                    cost=cost,
                    confusion_pair=False,
                    query_codepoint="",
                    cand_codepoint=codepoint_hex(cc),
                )
            )
            j -= 1
    ops.reverse()
    return dp_h[m][n], round(dp_w[m][n], 6), tuple(ops)


@dataclass(frozen=True)
class RawTrieHit:
    entry: LexiconEntry
    start: int
    end: int
    hard_edits: int
    weighted_edit_cost: float
    edit_ops: tuple[EditOp, ...]


def search_span_exact_and_fuzzy(
    index: LexiconIndex,
    query: str,
    *,
    max_edits: int,
    policy: EditPolicyConfig,
    include_context_phrases: bool = True,
    include_expanded_culinary: bool = True,
    use_confusion_weights: bool = True,
    confusion_sub_cost: float = 0.45,
    ordinary_confusion_sub_cost: float = 0.65,
    max_node_visits: int = 60000,
) -> tuple[list[tuple[LexiconEntry, int, float, tuple[EditOp, ...]]], int, bool]:
    """Search the trie for all entries within `max_edits` of whole string `query`."""
    qlen = len(query)
    if qlen == 0:
        return [], 0, False

    visits = 0
    truncated = False
    hits: list[tuple[LexiconEntry, int, float, tuple[EditOp, ...]]] = []
    init_row = list(range(qlen + 1))

    def dfs(node: TrieNode, depth: int, prev_row: list[int]) -> None:
        nonlocal visits, truncated
        if truncated:
            return
        if node.entry_ids:
            hard = prev_row[qlen]
            if hard <= max_edits:
                allowed = policy.max_allowed_edits(qlen, depth)
                if hard <= allowed:
                    for eid in node.entry_ids:
                        entry = index.get_entry(eid)
                        if not index.tier_allowed(
                            entry.source_tier,
                            include_context_phrases=include_context_phrases,
                            include_expanded_culinary=include_expanded_culinary,
                        ):
                            continue
                        h_exact, w_cost, ops = align_edit_ops(
                            query,
                            entry.canonical_term,
                            use_confusion_weights=use_confusion_weights,
                            confusion_sub_cost=confusion_sub_cost,
                            ordinary_confusion_sub_cost=ordinary_confusion_sub_cost,
                        )
                        if h_exact <= allowed:
                            hits.append((entry, h_exact, w_cost, ops))

        if depth + 1 + node.min_remaining_len - 1 > qlen + max_edits:
            return
        if depth + node.max_remaining_len < qlen - max_edits:
            return

        for ch, child in node.children.items():
            visits += 1
            if visits > max_node_visits:
                truncated = True
                return
            curr_row = [depth + 1]
            row_min = depth + 1
            for j in range(1, qlen + 1):
                ins = curr_row[j - 1] + 1
                dele = prev_row[j] + 1
                sub = prev_row[j - 1] + (0 if query[j - 1] == ch else 1)
                val = ins if ins < dele else dele
                if sub < val:
                    val = sub
                curr_row.append(val)
                if val < row_min:
                    row_min = val
            if row_min <= max_edits:
                dfs(child, depth + 1, curr_row)

    dfs(index.root, 0, init_row)
    hits.sort(key=lambda x: (x[1], x[2], x[0].source_tier != "curated_core", x[0].canonical_term))
    return hits, visits, truncated


def search_line_substrings(
    index: LexiconIndex,
    text: str,
    *,
    blocked_mask: tuple[bool, ...],
    config: LookupConfig,
) -> tuple[list[RawTrieHit], dict[str, object]]:
    """Run bounded prefix-DP Levenshtein trie search from every valid start position."""
    n = len(text)
    policy = config.edit_policy
    global_max_edits = max(
        policy.len1_max_edits,
        policy.len2_max_edits,
        policy.len3_5_max_edits,
        policy.len6plus_max_edits,
    )
    total_visits = 0
    truncated = False
    all_hits: list[RawTrieHit] = []
    seen: set[tuple[int, int, int]] = set()

    for start in range(n):
        if blocked_mask[start]:
            continue
        if total_visits >= config.max_trie_node_visits:
            truncated = True
            break
        run_end = start
        max_end = min(n, start + config.max_span_length)
        while run_end < max_end and not blocked_mask[run_end]:
            run_end += 1
        window = text[start:run_end]
        wlen = len(window)
        if wlen == 0:
            continue

        max_e = policy.allowed_edits_for_length(wlen)
        if max_e > global_max_edits:
            max_e = global_max_edits
        init_row = list(range(wlen + 1))

        def dfs(node: TrieNode, depth: int, prev_row: list[int]) -> None:
            nonlocal total_visits, truncated
            if truncated:
                return
            if node.entry_ids:
                c_allowed = policy.allowed_edits_for_length(depth)
                j_lo = max(1, depth - c_allowed)
                j_hi = min(wlen, depth + c_allowed)
                for j in range(j_lo, j_hi + 1):
                    hard = prev_row[j]
                    if hard > c_allowed:
                        continue
                    allowed = policy.max_allowed_edits(j, depth)
                    if hard > allowed:
                        continue
                    end = start + j
                    q_span = window[:j]
                    for eid in node.entry_ids:
                        key = (start, end, eid)
                        if key in seen:
                            continue
                        entry = index.get_entry(eid)
                        if not index.tier_allowed(
                            entry.source_tier,
                            include_context_phrases=config.include_context_phrases,
                            include_expanded_culinary=config.include_expanded_culinary,
                        ):
                            continue
                        h_exact, w_cost, ops = align_edit_ops(
                            q_span,
                            entry.canonical_term,
                            use_confusion_weights=config.use_confusion_weights,
                            confusion_sub_cost=config.confusion_sub_cost,
                            ordinary_confusion_sub_cost=config.ordinary_confusion_sub_cost,
                        )
                        if h_exact <= allowed:
                            seen.add(key)
                            all_hits.append(
                                RawTrieHit(
                                    entry=entry,
                                    start=start,
                                    end=end,
                                    hard_edits=h_exact,
                                    weighted_edit_cost=w_cost,
                                    edit_ops=ops,
                                )
                            )

            if depth >= wlen + max_e:
                return

            for ch, child in node.children.items():
                total_visits += 1
                if total_visits > config.max_trie_node_visits:
                    truncated = True
                    return
                curr_row = [depth + 1]
                row_min = depth + 1
                for j in range(1, wlen + 1):
                    ins = curr_row[j - 1] + 1
                    dele = prev_row[j] + 1
                    sub = prev_row[j - 1] + (0 if window[j - 1] == ch else 1)
                    val = ins if ins < dele else dele
                    if sub < val:
                        val = sub
                    curr_row.append(val)
                    if val < row_min:
                        row_min = val
                if row_min <= max_e:
                    dfs(child, depth + 1, curr_row)

        dfs(index.root, 0, init_row)

    diag = {
        "trie_node_visits": total_visits,
        "truncated": truncated,
        "raw_hit_count": len(all_hits),
    }
    return all_hits, diag
