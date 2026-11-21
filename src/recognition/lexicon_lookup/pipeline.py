"""End-to-end thread-safe orchestration for the Python lexicon-trie pipeline.

Supports three entry points without duplicating any matching logic:
  1. `lookup_text(text, config=...)` — pure text lookup (no PyTorch required),
     preserving any unprocessed suffix beyond `max_query_length` without text loss.
  2. `lookup_from_logits(logits, valid_length, mapping, ...)` — decodes canonical
     greedy and optional reviewed Char-LM R1, runs lexicon lookup, and runs a
     bounded whole-line visual+text candidate ranking stage (including unchanged
     OCR) against original folded canonical logits.
  3. `lookup_image(pil_image, orientation=0, ...)` — runs promoted recognizer
     (`L2_v06` `epoch_02.pt` EMA, `h4_k41`) with exact H=64 native single-window
     (`<=512 px`) or explicit H=64/100px-overlap multi-window (`>512 px`) routing.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Mapping, Optional

from .confusables import CONFUSABLE_CATALOG_FORMAT, CONFUSABLE_CATALOG_VERSION
from .contracts import (
    CandidateMatch,
    LookupConfig,
    LookupResult,
    SegmentResult,
    VisualRescoreResult,
)
from .index import LexiconIndex
from .normalization import canonicalize_text, normalize_with_span_map
from .protected_spans import extract_protected_spans
from .particle_guard import probable_particles, guard_candidate
from .segmentation import score_candidate_hit, segment_line_dp
from .translation_repo import ReadOnlyTranslationRepo
from .trie_search import search_line_substrings


DEFAULT_R1_DECODE_PARAMS = {
    "mode": "beam4g_dn_r1",
    "alpha_lm": 0.30,
    "beam_width": 5,
    "topk_t": 6,
    "digit_neutral": True,
    "short_greedy_max_chars": 2,
}


def _reconstruct_suggested_line(segments: tuple[SegmentResult, ...]) -> str:
    parts: list[str] = []
    for seg in segments:
        if seg.kind in ("exact", "fuzzy_suggestion") and seg.selected_candidate is not None:
            parts.append(seg.selected_candidate.term)
        else:
            parts.append(seg.text)
    return "".join(parts)


def _segment_text_score(seg: SegmentResult, *, keep_original: bool = False) -> float:
    if seg.kind == "exact" and seg.selected_candidate is not None:
        return float(seg.selected_candidate.total_score)
    if seg.kind == "fuzzy_suggestion" and seg.selected_candidate is not None:
        return 0.0 if keep_original else float(seg.selected_candidate.total_score)
    if seg.kind == "whitespace":
        return -0.05 * (seg.end - seg.start)
    return 0.0


def _generate_ranked_candidate_lines(
    base_ocr: str,
    segments: tuple[SegmentResult, ...],
    max_lines: int,
) -> list[tuple[str, float, bool]]:
    """Build bounded list of `(candidate_line, text_score, is_unchanged_ocr)` for CTC ranking.

    Always includes unchanged `base_ocr` as an explicit competitor so visual
    ranking can keep unchanged OCR when CTC evidence rejects a fuzzy edit.
    """
    out: list[tuple[str, float, bool]] = []
    seen: set[str] = set()

    base_text_score = round(sum(_segment_text_score(s, keep_original=True) for s in segments), 6)
    out.append((base_ocr, base_text_score, True))
    seen.add(base_ocr)

    def _add(line: str, score: float) -> None:
        if line and line not in seen and len(out) < max_lines + 1:
            seen.add(line)
            out.append((line, round(score, 6), False))

    sugg_line = _reconstruct_suggested_line(segments)
    sugg_score = sum(_segment_text_score(s, keep_original=False) for s in segments)
    _add(sugg_line, sugg_score)

    # Single-segment alternative substitutions from top alternatives
    for idx, seg in enumerate(segments):
        if not seg.alternatives:
            continue
        for alt in seg.alternatives[:2]:
            rebuilt_parts: list[str] = []
            alt_score = 0.0
            for j, other in enumerate(segments):
                if j == idx:
                    rebuilt_parts.append(alt.term)
                    alt_score += float(alt.total_score)
                elif other.kind in ("exact", "fuzzy_suggestion") and other.selected_candidate is not None:
                    rebuilt_parts.append(other.selected_candidate.term)
                    alt_score += float(other.selected_candidate.total_score)
                else:
                    rebuilt_parts.append(other.text)
                    alt_score += _segment_text_score(other, keep_original=True)
            _add("".join(rebuilt_parts), alt_score)
            if len(out) >= max_lines + 1:
                return out

    return out


def rank_whole_lines_with_ctc(
    *,
    base_ocr: str,
    segments: tuple[SegmentResult, ...],
    canonical_log_probs: list[list[float]],
    mapping: Mapping[str, int],
    config: LookupConfig,
) -> tuple[tuple[VisualRescoreResult, ...], str]:
    """Bounded whole-line ranking combining DP text_score + visual_ctc_weight * delta_log_prob."""
    from .ctc_scorer import score_candidate_line_ctc

    cand_entries = _generate_ranked_candidate_lines(base_ocr, segments, config.max_visual_rescore_lines)
    base_ctc = score_candidate_line_ctc(
        base_ocr,
        canonical_log_probs=canonical_log_probs,
        mapping=mapping,
    )
    base_lp = base_ctc.ctc_log_prob

    scored: list[VisualRescoreResult] = []
    for line, txt_score, is_base in cand_entries:
        vr = score_candidate_line_ctc(
            line,
            canonical_log_probs=canonical_log_probs,
            mapping=mapping,
            greedy_log_prob=base_lp,
        )
        combined: Optional[float] = None
        if vr.scorable and vr.delta_log_prob is not None:
            combined = round(txt_score + config.visual_ctc_weight * vr.delta_log_prob, 6)
        scored.append(
            VisualRescoreResult(
                candidate_line=vr.candidate_line,
                scorable=vr.scorable,
                ctc_log_prob=vr.ctc_log_prob,
                greedy_ctc_log_prob=vr.greedy_ctc_log_prob,
                delta_log_prob=vr.delta_log_prob,
                unsupported_chars=vr.unsupported_chars,
                reason=vr.reason,
                text_score=txt_score,
                visual_ctc_weight=config.visual_ctc_weight,
                combined_score=combined,
                is_unchanged_ocr=is_base,
                rank=None,
            )
        )

    # Scorable entries ranked by descending combined_score (tiebreak: unchanged OCR first);
    # unscorable entries always placed after scorable ones with rank=None.
    scorable_items = [x for x in scored if x.scorable and x.combined_score is not None]
    unscorable_items = [x for x in scored if not (x.scorable and x.combined_score is not None)]
    scorable_items.sort(key=lambda x: (-float(x.combined_score), not x.is_unchanged_ocr, x.candidate_line))

    ranked: list[VisualRescoreResult] = []
    for r_idx, item in enumerate(scorable_items, start=1):
        ranked.append(
            VisualRescoreResult(
                candidate_line=item.candidate_line,
                scorable=item.scorable,
                ctc_log_prob=item.ctc_log_prob,
                greedy_ctc_log_prob=item.greedy_ctc_log_prob,
                delta_log_prob=item.delta_log_prob,
                unsupported_chars=item.unsupported_chars,
                reason=item.reason,
                text_score=item.text_score,
                visual_ctc_weight=item.visual_ctc_weight,
                combined_score=item.combined_score,
                is_unchanged_ocr=item.is_unchanged_ocr,
                rank=r_idx,
            )
        )
    ranked.extend(unscorable_items)

    best_line = ranked[0].candidate_line if (ranked and ranked[0].scorable and ranked[0].rank == 1) else base_ocr
    return tuple(ranked), best_line


class LexiconPipeline:
    """Stateless, thread-safe lexicon-trie lookup pipeline with optional recognizer & LM."""

    def __init__(
        self,
        index: LexiconIndex,
        translation_repo: Optional[ReadOnlyTranslationRepo] = None,
        *,
        immutable_lm: Any = None,
        model: Any = None,
        model_meta: Optional[dict[str, Any]] = None,
        device: str = "cpu",
    ) -> None:
        self._index = index
        self._repo = translation_repo
        self._lm = immutable_lm
        self._model = model
        self._model_meta = dict(model_meta or {})
        self._device = str(device)
        self._model_lock = threading.Lock()

    @property
    def index(self) -> LexiconIndex:
        return self._index

    @property
    def translation_repo(self) -> Optional[ReadOnlyTranslationRepo]:
        return self._repo

    @property
    def has_model(self) -> bool:
        return self._model is not None

    @property
    def model_metadata(self) -> dict[str, Any]:
        return dict(self._model_meta)

    @property
    def has_lm(self) -> bool:
        return self._lm is not None

    def lookup_text(
        self,
        text: str,
        *,
        config: Optional[LookupConfig] = None,
        raw_ocr: Optional[str] = None,
        decoder_mode: str = "text_only",
        decoder_config: Optional[dict[str, Any]] = None,
        lm_alternative: Optional[str] = None,
        crop_provenance: Optional[dict[str, Any]] = None,
    ) -> LookupResult:
        """Pure text lookup with zero PyTorch dependency and zero silent suffix loss."""
        t0 = time.perf_counter()
        cfg = config if config is not None else LookupConfig()
        warnings: list[str] = []

        if cfg.auto_correct:
            warnings.append("auto_correct_requested_but_disabled_by_policy")

        raw_input = text if raw_ocr is None else raw_ocr
        span_map = normalize_with_span_map(text)
        canonical = span_map.canonical_text
        protected = extract_protected_spans(span_map)

        # Preserve full input: if canonical exceeds max_query_length, bound trie/DP to
        # prefix [0, active_end) without splitting any protected span, and keep the
        # entire suffix intact as protected + unknown segments with exact offsets.
        active_end = len(canonical)
        suffix_preserved = False
        if len(canonical) > cfg.max_query_length:
            active_end = cfg.max_query_length
            for sp in protected:
                if sp.start < active_end < sp.end:
                    active_end = sp.start
                    break
            suffix_preserved = True
            warnings.append(
                f"unprocessed_suffix_preserved(active_end={active_end},suffix_len={len(canonical) - active_end})"
            )

        # Build blocked mask over active prefix [0, active_end)
        prefix_canon = canonical[:active_end]
        blocked = [False] * active_end
        for sp in protected:
            if sp.start >= active_end:
                break
            for idx in range(sp.start, min(sp.end, active_end)):
                blocked[idx] = True
        for idx, ch in enumerate(prefix_canon):
            if ch in (" ", "\t", "\u3000"):
                blocked[idx] = True

        raw_hits, trie_diag = search_line_substrings(
            self._index,
            prefix_canon,
            blocked_mask=tuple(blocked),
            config=cfg,
        )
        if trie_diag.get("truncated"):
            warnings.append(f"trie_search_budget_reached({cfg.max_trie_node_visits})")

        scored_candidates: list[CandidateMatch] = [
            score_candidate_hit(h, span_map, cfg, self._repo) for h in raw_hits
        ]

        particles = probable_particles(prefix_canon, self._index, cfg) if cfg.protect_particles else ()
        scored_candidates = [guard_candidate(c, particles) for c in scored_candidates]
        suppressed = [c for c in scored_candidates if c.suppression_reasons]
        scored_candidates = [c for c in scored_candidates if not c.suppression_reasons]

        segments, bounded_cands, budget_diag = segment_line_dp(
            span_map,
            protected,
            scored_candidates,
            cfg,
            active_end=active_end,
        )
        if budget_diag.get("candidates_truncated"):
            warnings.append(
                f"candidate_budget_reached(kept={budget_diag['candidates_after_cap']},before={budget_diag['candidates_before_cap']})"
            )

        suggested_line = _reconstruct_suggested_line(segments)

        # Collect fuzzy suggestions separately from exact matches
        fuzzy_suggestions: list[CandidateMatch] = []
        exact_translations: list[dict[str, Any]] = []
        for seg in segments:
            if seg.kind == "exact" and seg.selected_candidate is not None:
                c = seg.selected_candidate
                exact_translations.append(
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "orig_start": seg.orig_start,
                        "orig_end": seg.orig_end,
                        "term": c.term,
                        "entry_id": c.entry_id,
                        "source_tier": c.source_tier,
                        "db_id": c.db_id,
                        "translation_status": c.translation_status,
                        "translation": c.translation.to_dict() if c.translation is not None else None,
                    }
                )
            elif seg.kind == "fuzzy_suggestion" and seg.selected_candidate is not None:
                fuzzy_suggestions.append(seg.selected_candidate)
            for alt in seg.alternatives:
                if not alt.is_exact and alt not in fuzzy_suggestions:
                    fuzzy_suggestions.append(alt)

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        # Inspection-only alternatives never enter DP or whole-line CTC ranking.
        fuzzy_suggestions.extend(sorted(suppressed, key=lambda c: (-c.total_score, c.term)))
        diagnostics = {
            **trie_diag,
            **budget_diag,
            "input_truncated": False,
            "unprocessed_suffix_preserved": suffix_preserved,
            "active_prefix_end": active_end,
            "bounded_candidate_count": len(bounded_cands),
            "elapsed_ms": elapsed_ms,
            "span_indexing_unit": "unicode_code_points",
            "normalization_changes": list(span_map.changes),
            "lookup_config": cfg.to_dict(),
            "probable_particles": list(particles),
            "particle_suppressed_candidates": len(suppressed),
            "particle_guard_method": "conservative_menu_context_rules_v1_not_grammar_validation",
        }
        provenance = {
            "index_schema_version": self._index.metadata.get("schema_version"),
            "index_format": self._index.metadata.get("index_format"),
            "index_sha256": self._index.metadata.get("index_sha256"),
            "index_total_entries": self._index.metadata.get("total_entries"),
            "confusable_catalog_version": CONFUSABLE_CATALOG_VERSION,
            "confusable_catalog_format": CONFUSABLE_CATALOG_FORMAT,
            "model": self._model_meta if self._model_meta else None,
            "crop": crop_provenance,
        }

        return LookupResult(
            raw_ocr=raw_input,
            canonical_ocr=canonicalize_text(raw_input),
            selected_text_for_lookup=canonical,
            decoder_mode=decoder_mode,
            decoder_config=dict(decoder_config or {}),
            lm_alternative=lm_alternative,
            protected_spans=protected,
            segments=segments,
            fuzzy_suggestions=tuple(fuzzy_suggestions[: cfg.max_total_candidates]),
            exact_translations=tuple(exact_translations),
            visual_rescores=(),
            suggested_line=suggested_line,
            visual_reranked_line=None,
            warnings=tuple(warnings),
            diagnostics=diagnostics,
            provenance=provenance,
            auto_corrected=False,
        )

    def lookup_from_logits(
        self,
        logits: Any,
        valid_length: int,
        mapping: Mapping[str, int],
        *,
        config: Optional[LookupConfig] = None,
        decode_mode: str = "greedy",
        crop_provenance: Optional[dict[str, Any]] = None,
    ) -> LookupResult:
        """Decode logits (canonical greedy + optional reviewed R1), then lookup & visual-CTC rank."""
        import torch
        from ..canonical_ctc import fold_parenthesis_logits
        from ..char_lm_beam import DecodeConfig, decode_with_config
        from ..data import decode as greedy_decode_raw
        from .ctc_scorer import prepare_canonical_log_probs

        cfg = config if config is not None else LookupConfig()
        if logits.ndim == 2:
            logits_3d = logits[:, None, :]
        elif logits.ndim == 3 and logits.shape[1] == 1:
            logits_3d = logits
        else:
            raise ValueError(f"Expected single-item logits (T, C) or (T, 1, C), got {tuple(logits.shape)}")

        T = int(valid_length)
        raw_ocr = greedy_decode_raw(logits_3d, mapping, [T])[0]
        folded = fold_parenthesis_logits(logits_3d, mapping, drop_aliases=False)
        canonical_greedy = greedy_decode_raw(folded, mapping, [T])[0]
        logp_tv = torch.log_softmax(folded[:T, 0, :].detach().cpu().float(), dim=-1)
        id_to_char = {int(v): k for k, v in mapping.items()}

        lm_alt: Optional[str] = None
        selected_text = canonical_greedy
        dec_cfg_dict: dict[str, Any] = {"mode": decode_mode}

        if decode_mode == "greedy":
            dec_cfg = DecodeConfig(mode="greedy")
            dec_cfg_dict = {
                "mode": dec_cfg.mode,
                "beam_width": dec_cfg.beam_width,
                "topk_t": dec_cfg.topk_t,
                "alpha_lm": dec_cfg.alpha_lm,
            }
            if self._lm is not None:
                r1_cfg = DecodeConfig(
                    mode="beam4g_dn_r1",
                    alpha_lm=DEFAULT_R1_DECODE_PARAMS["alpha_lm"],
                    beam_width=DEFAULT_R1_DECODE_PARAMS["beam_width"],
                    topk_t=DEFAULT_R1_DECODE_PARAMS["topk_t"],
                    digit_neutral=True,
                    short_greedy_max_chars=DEFAULT_R1_DECODE_PARAMS["short_greedy_max_chars"],
                )
                lm_alt = str(decode_with_config(logp_tv, id_to_char, r1_cfg, lm=self._lm)["text"])
        elif decode_mode == "beam4g_dn_r1":
            if self._lm is None:
                raise ValueError("decode_mode='beam4g_dn_r1' requires an ImmutableCharNGramLM")
            r1_cfg = DecodeConfig(
                mode="beam4g_dn_r1",
                alpha_lm=DEFAULT_R1_DECODE_PARAMS["alpha_lm"],
                beam_width=DEFAULT_R1_DECODE_PARAMS["beam_width"],
                topk_t=DEFAULT_R1_DECODE_PARAMS["topk_t"],
                digit_neutral=True,
                short_greedy_max_chars=DEFAULT_R1_DECODE_PARAMS["short_greedy_max_chars"],
            )
            dec_cfg_dict = {
                "mode": r1_cfg.mode,
                "alpha_lm": r1_cfg.alpha_lm,
                "beam_width": r1_cfg.beam_width,
                "topk_t": r1_cfg.topk_t,
                "digit_neutral": r1_cfg.digit_neutral,
                "short_greedy_max_chars": r1_cfg.short_greedy_max_chars,
            }
            lm_alt = str(decode_with_config(logp_tv, id_to_char, r1_cfg, lm=self._lm)["text"])
            selected_text = lm_alt
        elif decode_mode == "plain_beam":
            pb_cfg = DecodeConfig(mode="plain_beam", beam_width=5, topk_t=6, alpha_lm=0.0)
            dec_cfg_dict = {
                "mode": pb_cfg.mode,
                "beam_width": pb_cfg.beam_width,
                "topk_t": pb_cfg.topk_t,
                "alpha_lm": pb_cfg.alpha_lm,
            }
            selected_text = str(decode_with_config(logp_tv, id_to_char, pb_cfg, lm=None)["text"])
            if self._lm is not None:
                r1_cfg = DecodeConfig(
                    mode="beam4g_dn_r1",
                    alpha_lm=DEFAULT_R1_DECODE_PARAMS["alpha_lm"],
                    beam_width=DEFAULT_R1_DECODE_PARAMS["beam_width"],
                    topk_t=DEFAULT_R1_DECODE_PARAMS["topk_t"],
                    digit_neutral=True,
                    short_greedy_max_chars=DEFAULT_R1_DECODE_PARAMS["short_greedy_max_chars"],
                )
                lm_alt = str(decode_with_config(logp_tv, id_to_char, r1_cfg, lm=self._lm)["text"])
        else:
            raise ValueError(
                f"Unsupported decode_mode={decode_mode!r}; expected 'greedy', 'beam4g_dn_r1', or 'plain_beam'"
            )

        base_result = self.lookup_text(
            selected_text,
            config=cfg,
            raw_ocr=raw_ocr,
            decoder_mode=decode_mode,
            decoder_config=dec_cfg_dict,
            lm_alternative=lm_alt,
            crop_provenance=crop_provenance,
        )

        # Bounded whole-line visual+text candidate ranking stage (including unchanged OCR)
        visual_rescores: tuple[VisualRescoreResult, ...] = ()
        visual_reranked_line: Optional[str] = None
        if cfg.enable_visual_rescoring:
            canon_lp = prepare_canonical_log_probs(logits_3d, T, mapping)
            visual_rescores, visual_reranked_line = rank_whole_lines_with_ctc(
                base_ocr=base_result.selected_text_for_lookup,
                segments=base_result.segments,
                canonical_log_probs=canon_lp,
                mapping=mapping,
                config=cfg,
            )

        return LookupResult(
            raw_ocr=base_result.raw_ocr,
            canonical_ocr=canonical_greedy,
            selected_text_for_lookup=base_result.selected_text_for_lookup,
            decoder_mode=base_result.decoder_mode,
            decoder_config=base_result.decoder_config,
            lm_alternative=base_result.lm_alternative,
            protected_spans=base_result.protected_spans,
            segments=base_result.segments,
            fuzzy_suggestions=base_result.fuzzy_suggestions,
            exact_translations=base_result.exact_translations,
            visual_rescores=visual_rescores,
            suggested_line=base_result.suggested_line,
            visual_reranked_line=visual_reranked_line,
            warnings=base_result.warnings,
            diagnostics=base_result.diagnostics,
            provenance=base_result.provenance,
            auto_corrected=False,
        )

    def lookup_image(
        self,
        image: Any,
        *,
        orientation: int = 0,
        already_rectified_strip: bool = False,
        config: Optional[LookupConfig] = None,
        decode_mode: str = "greedy",
        crop_provenance: Optional[dict[str, Any]] = None,
    ) -> LookupResult:
        """Run promoted recognizer on a PIL Image crop without shrinking oversized strips.

        - Strictly validates `orientation in (0, 1)`.
        - If `already_rectified_strip=False` (upright user crop), rotates vertical
          (`orientation=1`) crops CCW 90° once. If `already_rectified_strip=True`
          (from `line_crops.rectify`), does NOT rotate again (prevents double rotation).
        - Strips with H=64 native width `<= 512` use single-window `bucket_image(..., 512)`
          (zero height letterboxing, exact parity with existing inference).
        - Strips with H=64 native width `> 512` use explicit `overlap512` (100px overlap)
          with source-coordinate `owned_tokens`, canonical parenthesis folding per window,
          parent-level lexicon lookup, and visual CTC rescoring explicitly marked unavailable.
        """
        if self._model is None:
            raise RuntimeError("No recognizer model attached to LexiconPipeline")
        if isinstance(orientation, bool) or orientation not in (0, 1):
            raise ValueError(f"orientation must be 0 (horizontal) or 1 (vertical), got {orientation!r}")

        import torch
        from ..canonical_ctc import fold_parenthesis_logits
        from ..data import IMAGENET_MEAN, IMAGENET_STD, preprocess
        from ..long_line_inference import bucket_image, native_strip, owned_tokens, window_ranges

        img = image.convert("RGB")
        if orientation == 1 and not already_rectified_strip:
            img = img.rotate(90, expand=True)

        strip = native_strip(img)
        if strip.width <= 512:
            fitted, meta = bucket_image(strip, max_width=512)
            tensor, _, _ = preprocess(fitted, IMAGENET_MEAN, IMAGENET_STD, padding="batch", max_width=meta["canvas"])
            valid_len = int(meta["input_length"])
            with self._model_lock:
                with torch.inference_mode():
                    x = tensor[None].to(self._device)
                    ori = torch.tensor([orientation], dtype=torch.long, device=self._device)
                    lengths = torch.tensor([valid_len], dtype=torch.long)
                    logits = self._model(x, ori, lengths).detach().cpu()

            prov = {
                **(crop_provenance or {}),
                "long_line_route": "single_window_le512",
                "native_strip_width": int(strip.width),
                "orientation": orientation,
                "valid_width": int(meta["valid_width"]),
                "canvas_width": int(meta["canvas"]),
                "valid_length": valid_len,
                "scaled_height": int(meta["scaled_height"]),
            }
            return self.lookup_from_logits(
                logits,
                valid_len,
                self._model.vocabulary,
                config=config,
                decode_mode=decode_mode,
                crop_provenance=prov,
            )

        # Explicit H64 / 100px-overlap long-line route for native strip width > 512
        ranges = window_ranges(strip.width, max_width=512, overlap=100)
        mapping = self._model.vocabulary
        id_to_char = {int(v): k for k, v in mapping.items()}
        raw_paths: list[list[int]] = []
        canon_paths: list[list[int]] = []
        windows_prov: list[dict[str, Any]] = []

        with self._model_lock:
            with torch.inference_mode():
                for w_idx, (left, right) in enumerate(ranges):
                    win_img = strip.crop((left, 0, right, 64))
                    fitted, meta = bucket_image(win_img, max_width=512)
                    tensor, _, _ = preprocess(
                        fitted, IMAGENET_MEAN, IMAGENET_STD, padding="batch", max_width=meta["canvas"]
                    )
                    valid_len = int(meta["input_length"])
                    x = tensor[None].to(self._device)
                    ori = torch.tensor([orientation], dtype=torch.long, device=self._device)
                    lengths = torch.tensor([valid_len], dtype=torch.long)
                    logits = self._model(x, ori, lengths).detach().cpu()
                    folded = fold_parenthesis_logits(logits, mapping, drop_aliases=False)
                    raw_p = logits[:valid_len, 0].argmax(-1).tolist()
                    can_p = folded[:valid_len, 0].argmax(-1).tolist()
                    raw_paths.append(raw_p)
                    canon_paths.append(can_p)
                    windows_prov.append(
                        {
                            "window_index": w_idx,
                            "left_px": int(left),
                            "right_px": int(right),
                            "valid_width": int(meta["valid_width"]),
                            "canvas_width": int(meta["canvas"]),
                            "valid_length": valid_len,
                            "scaled_height": int(meta["scaled_height"]),
                        }
                    )

        raw_merged = "".join(id_to_char[t] for t in owned_tokens(raw_paths, ranges))
        canon_merged = "".join(id_to_char[t] for t in owned_tokens(canon_paths, ranges))

        prov = {
            **(crop_provenance or {}),
            "long_line_route": "overlap512_h64",
            "native_strip_width": int(strip.width),
            "overlap_px": 100,
            "orientation": orientation,
            "window_count": len(ranges),
            "windows": windows_prov,
        }
        cfg = config if config is not None else LookupConfig()
        base_res = self.lookup_text(
            canon_merged,
            config=cfg,
            raw_ocr=raw_merged,
            decoder_mode=f"{decode_mode}_overlap512_greedy_merged",
            decoder_config={"mode": decode_mode, "long_line_route": "overlap512_h64", "window_count": len(ranges)},
            lm_alternative=None,
            crop_provenance=prov,
        )
        extra_warnings = list(base_res.warnings)
        extra_warnings.append("visual_rescoring_unavailable_for_multi_window_overlap")
        if decode_mode != "greedy":
            extra_warnings.append("overlap512_uses_merged_canonical_greedy_not_joint_lm_beam")
        unavailable_rescore = (
            VisualRescoreResult(
                candidate_line=base_res.suggested_line,
                scorable=False,
                ctc_log_prob=None,
                greedy_ctc_log_prob=None,
                delta_log_prob=None,
                unsupported_chars=(),
                reason="multi_window_overlap_logits_not_jointly_scorable",
                is_unchanged_ocr=(base_res.suggested_line == base_res.selected_text_for_lookup),
            ),
        )
        return LookupResult(
            raw_ocr=base_res.raw_ocr,
            canonical_ocr=canon_merged,
            selected_text_for_lookup=base_res.selected_text_for_lookup,
            decoder_mode=base_res.decoder_mode,
            decoder_config=base_res.decoder_config,
            lm_alternative=None,
            protected_spans=base_res.protected_spans,
            segments=base_res.segments,
            fuzzy_suggestions=base_res.fuzzy_suggestions,
            exact_translations=base_res.exact_translations,
            visual_rescores=unavailable_rescore,
            suggested_line=base_res.suggested_line,
            visual_reranked_line=None,
            warnings=tuple(extra_warnings),
            diagnostics=base_res.diagnostics,
            provenance=base_res.provenance,
            auto_corrected=False,
        )
