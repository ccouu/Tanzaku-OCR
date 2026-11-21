"""Proper CTC forward log-probability scorer with canonical parenthesis pooling.

Scores complete candidate lines against original model logits `(T, C)` over
valid sequence length `T`, using `fold_parenthesis_logits` so full-width and
ASCII parentheses share pooled probability mass identically to canonical CTC.

Out-of-vocabulary or length-infeasible candidates are NEVER assigned fake
probabilities; they are returned with `scorable=False` and explicit diagnostics.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional

import torch

from ..canonical_ctc import fold_parenthesis_logits
from ..data import minimum_ctc_steps
from .contracts import VisualRescoreResult
from .normalization import canonicalize_text


def _logsumexp2(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a >= b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def _logsumexp3(a: float, b: float, c: float) -> float:
    return _logsumexp2(_logsumexp2(a, b), c)


def ctc_forward_log_prob_from_log_probs(
    log_probs_2d: list[list[float]],
    token_ids: list[int],
    blank_id: int = 0,
) -> float:
    """Exact CTC forward algorithm in log space over `T x C` log-probabilities.

    Handles:
    - Empty target sequence (`token_ids == []` -> sum of blank log-probs across all T)
    - Repeated adjacent symbols (requires at least one blank between identical IDs)
    - Infeasible lengths (`T < minimum_ctc_steps`) -> returns `-inf`.
    """
    T = len(log_probs_2d)
    if T == 0:
        return 0.0 if len(token_ids) == 0 else -math.inf

    if len(token_ids) == 0:
        return sum(log_probs_2d[t][blank_id] for t in range(T))

    # Extended label sequence: [blank, y0, blank, y1, ..., blank], length S = 2*L + 1
    L = len(token_ids)
    S = 2 * L + 1
    ext = [blank_id] * S
    for idx, tok in enumerate(token_ids):
        ext[2 * idx + 1] = tok

    # Standard CTC forward DP
    neg_inf = -math.inf
    alpha = [neg_inf] * S
    alpha[0] = log_probs_2d[0][ext[0]]
    if S > 1:
        alpha[1] = log_probs_2d[0][ext[1]]

    for t in range(1, T):
        row = log_probs_2d[t]
        nxt = [neg_inf] * S
        for s in range(S):
            emit = row[ext[s]]
            if emit == neg_inf:
                continue
            val = alpha[s]
            if s > 0:
                val = _logsumexp2(val, alpha[s - 1])
            if s > 1 and ext[s] != blank_id and ext[s] != ext[s - 2]:
                val = _logsumexp2(val, alpha[s - 2])
            if val != neg_inf:
                nxt[s] = val + emit
        alpha = nxt

    return _logsumexp2(alpha[S - 1], alpha[S - 2])


def prepare_canonical_log_probs(
    logits: torch.Tensor,
    valid_length: int,
    mapping: Mapping[str, int],
) -> list[list[float]]:
    """Fold parenthesis logits and compute valid-T log-softmax in FP64/FP32."""
    if logits.ndim == 3:
        if logits.shape[1] != 1:
            raise ValueError(f"Expected single-item logits (T, 1, C) or (T, C), got {tuple(logits.shape)}")
        logits_2d = logits[:, 0, :]
    elif logits.ndim == 2:
        logits_2d = logits
    else:
        raise ValueError(f"Unsupported logits rank: {tuple(logits.shape)}")

    T = int(valid_length)
    if T < 1 or T > logits_2d.shape[0]:
        raise ValueError(f"Invalid valid_length={T} for logits T={logits_2d.shape[0]}")

    sliced = logits_2d[:T].detach().cpu().float()
    if "（" in mapping and "(" in mapping and "）" in mapping and ")" in mapping:
        folded = fold_parenthesis_logits(sliced, mapping, drop_aliases=False)
    else:
        folded = sliced
    log_probs = torch.log_softmax(folded, dim=-1).double()
    return log_probs.tolist()


def score_candidate_line_ctc(
    candidate_line: str,
    *,
    canonical_log_probs: list[list[float]],
    mapping: Mapping[str, int],
    greedy_log_prob: Optional[float] = None,
    blank_id: int = 0,
) -> VisualRescoreResult:
    """Score one complete canonical candidate line against precomputed canonical log-probs."""
    canon = canonicalize_text(candidate_line)
    unsupported = tuple(sorted({ch for ch in canon if ch not in mapping}))
    if unsupported:
        return VisualRescoreResult(
            candidate_line=canon,
            scorable=False,
            ctc_log_prob=None,
            greedy_ctc_log_prob=greedy_log_prob,
            delta_log_prob=None,
            unsupported_chars=unsupported,
            reason="out_of_vocabulary_characters",
        )

    T = len(canonical_log_probs)
    min_steps = minimum_ctc_steps(canon)
    if T < min_steps:
        return VisualRescoreResult(
            candidate_line=canon,
            scorable=False,
            ctc_log_prob=None,
            greedy_ctc_log_prob=greedy_log_prob,
            delta_log_prob=None,
            unsupported_chars=(),
            reason=f"ctc_length_infeasible(T={T}<min_steps={min_steps})",
        )

    token_ids = [int(mapping[ch]) for ch in canon]
    lp = ctc_forward_log_prob_from_log_probs(canonical_log_probs, token_ids, blank_id=blank_id)
    if not math.isfinite(lp):
        return VisualRescoreResult(
            candidate_line=canon,
            scorable=False,
            ctc_log_prob=None,
            greedy_ctc_log_prob=greedy_log_prob,
            delta_log_prob=None,
            unsupported_chars=(),
            reason="non_finite_ctc_log_prob",
        )

    delta = round(lp - greedy_log_prob, 6) if (greedy_log_prob is not None and math.isfinite(greedy_log_prob)) else None
    return VisualRescoreResult(
        candidate_line=canon,
        scorable=True,
        ctc_log_prob=round(lp, 6),
        greedy_ctc_log_prob=round(greedy_log_prob, 6) if greedy_log_prob is not None else None,
        delta_log_prob=delta,
        unsupported_chars=(),
        reason="ok",
    )
