"""Conservative particle-boundary hints, not a Japanese grammar checker.

Recognize a small set of menu action contexts without a tokenizer dependency.
Never globally protect characters: the で in おでん remains part of its word.
Rules and exact lexical exceptions are deliberately inspectable and bounded.
"""
from __future__ import annotations

import re
from dataclasses import replace

from .contracts import CandidateMatch, LookupConfig
from .index import LexiconIndex

_PARTICLES = frozenset("でのをにとはが")
_ACTIONS = re.compile(
    r"(?:お召し上が|召し上が|召上が|食べ|お楽しみ|楽しめ|味わ|"
    r"いただ|頂|どうぞ|ご提供|提供|ご注文|注文|選べ|選ん|お選び|"
    r"お試し|試せ|付け|つけ|飲む|飲ん|お飲み|お持ち帰り|持ち帰り|"
    r"焼く|焼い|揚げ|煮る|煮込|蒸す|蒸し|ご用意|用意)"
)
_PREPARATION_END = re.compile(r"(?:焼き|炭火|直火|塩|醤油|たれ|タレ|ソース|ポン酢|お好み)$")
_BOUNDARY = frozenset(" \t\u3000、。！？!?()（）")


def probable_particles(text: str, index: LexiconIndex, config: LookupConfig) -> tuple[dict, ...]:
    """Return probable one-character particles with canonical offsets/reasons.

    An exact multi-character lexical match containing a suspected particle
    vetoes the rule: e.g. ウニを食べる must not protect the ニ as a particle.
    It is a conservative fallback and intentionally misses ambiguous contexts.
    """
    found = []
    for pos, ch in enumerate(text):
        if ch not in _PARTICLES or pos == 0:
            continue
        right = text[pos + 1:].lstrip(" \t\u3000")
        action = bool(_ACTIONS.match(right))
        terminal_de = ch == "で" and (not right or right[0] in _BOUNDARY) and bool(
            _PREPARATION_END.search(text[:pos].rstrip()))
        if not (action or terminal_de):
            continue
        inside_word = False
        # Independent exact traversal: fuzzy search exhaustion must not decide
        # whether おでん/うに is considered a word. Respect enabled source tiers.
        for start in range(max(0, pos - config.max_span_length + 1), pos + 1):
            node = index.root
            for end in range(start, min(len(text), start + config.max_span_length)):
                node = node.children.get(text[end])
                if node is None:
                    break
                if end >= pos and end - start + 1 >= 2 and node.entry_ids:
                    if any(index.tier_allowed(index.get_entry(i).source_tier,
                           include_context_phrases=config.include_context_phrases,
                           include_expanded_culinary=config.include_expanded_culinary)
                           for i in node.entry_ids):
                        inside_word = True
                        break
            if inside_word:
                break
        if not inside_word:
            separators = "、。！？!?;；\n"
            clause_start = max((text.rfind(c, 0, pos) for c in separators), default=-1) + 1
            next_boundaries = [text.find(c, pos + 1) for c in separators]
            clause_end = min((p for p in next_boundaries if p >= 0), default=len(text))
            found.append({"start": pos, "end": pos + 1, "text": ch,
                          "clause_start": clause_start, "clause_end": clause_end,
                          "reason": "menu_action_context" if action else "trailing_preparation_de"})
    return tuple(found)


def guard_candidate(candidate: CandidateMatch, particles: tuple[dict, ...]) -> CandidateMatch:
    """Mark candidates editing a probable particle; preserve exact candidates."""
    if candidate.is_exact:
        return candidate
    reasons = []
    # In a recognized action clause, ordinary dictionary-neighbor edits must not
    # replace otherwise valid connective prose. Leave them inspection-only;
    # genuine lookalike edits elsewhere remain eligible for CTC ranking.
    if particles and candidate.match_type != "visual_confusable_suggestion":
        for particle in particles:
            # Clause bounds supplied by the detector; avoid blocking another
            # independent clause just because the same line has a particle.
            if candidate.start < particle.get("clause_end", particle["end"]) and candidate.end > particle.get("clause_start", particle["start"]):
                reasons.append("particle_context:ordinary_fuzzy_edit_requires_review")
                break
    for particle in particles:
        cursor = 0
        for op in candidate.edit_ops:
            relative_pos = cursor if op.query_pos is None else op.query_pos
            pos = candidate.start + relative_pos
            if op.op != "insert":
                cursor = relative_pos + 1
            touches = (particle["start"] <= pos < particle["end"] or
                       (op.op == "insert" and pos == particle["end"]))
            if op.op != "match" and touches:
                reasons.append(f"particle_boundary:{particle['text']}@{particle['start']}:{particle['reason']}")
                break
    return replace(candidate, suppression_reasons=tuple(reasons)) if reasons else candidate
