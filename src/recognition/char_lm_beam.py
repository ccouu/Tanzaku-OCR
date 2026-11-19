"""Lightweight 4-gram Character Language Model + CTC Prefix Beam Search decoder."""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import re
import unicodedata

import torch

PROMO_BRACKET_RE = re.compile(r"【[^】]*】")
TRAILING_PRICE_RE = re.compile(r"\s*[0-9,]+円(?:（税込）|\(税込\)|（税抜）|\(税抜\)|〜|~)*.*$")
DECOR_RE = re.compile(r"[♪★☆◆◇◎●○■□▲△▼▽！!？?]+")


def clean_hotpepper_title(raw_title: str, tab: str) -> str | None:
    s = unicodedata.normalize("NFKC", raw_title)
    s = PROMO_BRACKET_RE.sub("", s)
    s = DECOR_RE.sub(" ", s)
    if tab == "course":
        s = TRAILING_PRICE_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s or len(s) > 28 or s.endswith(("ます", "です", "可能", "どうぞ")):
        return None
    return s


def _extract_terms(items) -> list[str]:
    out = []
    for x in items:
        t = x.get("term") or x.get("text") or "" if isinstance(x, dict) else x
        if isinstance(t, str) and t:
            out.append(t)
    return out


class CharNGramLM:
    """Interpolated 4-gram character language model with fast dict lookup."""

    def __init__(
        self,
        uni: dict[str, int],
        bi: dict[str, dict[str, int]],
        tri: dict[str, dict[str, int]],
        quad: dict[str, dict[str, int]],
        vocab_chars: list[str],
        digit_neutral: bool = True,
        reverse: bool = False,
    ) -> None:
        self.uni = uni
        self.bi = bi
        self.tri = tri
        self.quad = quad
        self.digit_neutral = digit_neutral
        self.reverse = reverse
        self.uniform_digit_lp = math.log(0.1)
        self.vocab_size = max(1, len(vocab_chars))
        total_uni = sum(uni.values())
        self.p_uni = {ch: (uni.get(ch, 0) + 1.0) / (total_uni + self.vocab_size) for ch in vocab_chars}
        self.default_uni = 1.0 / (total_uni + self.vocab_size)
        self.bi_tot = {k: sum(v.values()) for k, v in bi.items()}
        self.tri_tot = {k: sum(v.values()) for k, v in tri.items()}
        self.quad_tot = {k: sum(v.values()) for k, v in quad.items()}

    def step_logp(self, p3: str, p2: str, p1: str, ch: str) -> float:
        if self.digit_neutral:
            if "0" <= ch <= "9":
                return self.uniform_digit_lp
            if "0" <= p1 <= "9":
                p1 = "<DIG>"
            if "0" <= p2 <= "9":
                p2 = "<DIG>"
            if "0" <= p3 <= "9":
                p3 = "<DIG>"
        V = self.vocab_size
        p_u = self.p_uni.get(ch, self.default_uni)
        bt = self.bi_tot.get(p1, 0)
        p_b = (self.bi[p1].get(ch, 0) + 0.5 * p_u * V) / (bt + 0.5 * V) if bt > 0 else p_u
        k2 = f"{p2}\t{p1}"
        tt = self.tri_tot.get(k2, 0)
        p_t = (self.tri[k2].get(ch, 0) + 0.5 * p_b * V) / (tt + 0.5 * V) if tt > 0 else p_b
        k3 = f"{p3}\t{p2}\t{p1}"
        qt = self.quad_tot.get(k3, 0)
        p_q = (self.quad[k3].get(ch, 0) + 0.5 * p_t * V) / (qt + 0.5 * V) if qt > 0 else p_t
        return math.log(max(1e-12, 0.45 * p_q + 0.30 * p_t + 0.15 * p_b + 0.10 * p_u))

    def score_seq(self, text: str) -> float:
        chars = list(reversed(text)) if self.reverse else list(text)
        p1 = p2 = p3 = "<BOS>"
        total = 0.0
        for c in chars:
            total += self.step_logp(p3, p2, p1, c)
            p3, p2, p1 = p2, p1, c
        return total

    def save_gz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocab_chars": list(self.p_uni.keys()),
            "reverse": self.reverse,
            "digit_neutral": self.digit_neutral,
            "uni": self.uni,
            "bi": self.bi,
            "tri": self.tri,
            "quad": self.quad,
        }
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        path.write_bytes(gzip.compress(raw, compresslevel=6))

    @classmethod
    def load_gz(cls, path: Path) -> "CharNGramLM":
        obj = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
        return cls(
            obj["uni"],
            obj["bi"],
            obj["tri"],
            obj["quad"],
            obj["vocab_chars"],
            digit_neutral=bool(obj.get("digit_neutral", True)),
            reverse=bool(obj.get("reverse", False)),
        )

    @classmethod
    def _from_weighted_strings(
        cls,
        weighted: list[tuple[str, int]],
        char_to_id: dict[str, int],
        reverse: bool = False,
        digit_neutral: bool = True,
    ) -> "CharNGramLM":
        uni: Counter[str] = Counter()
        bi: dict[str, Counter[str]] = defaultdict(Counter)
        tri: dict[str, Counter[str]] = defaultdict(Counter)
        quad: dict[str, Counter[str]] = defaultdict(Counter)

        for text, w in weighted:
            if not text:
                continue
            seq = list(reversed(text)) if reverse else list(text)
            p1 = p2 = p3 = "<BOS>"
            for c in seq:
                if c not in char_to_id:
                    # Break n-gram context on OOV chars so non-adjacent chars are never glued together
                    p1 = p2 = p3 = "<UNK>"
                    continue
                uni[c] += w
                bi[p1][c] += w
                tri[f"{p2}\t{p1}"][c] += w
                quad[f"{p3}\t{p2}\t{p1}"][c] += w
                ctx_c = "<DIG>" if (digit_neutral and "0" <= c <= "9") else c
                p3, p2, p1 = p2, p1, ctx_c

        return cls(
            dict(uni),
            {k: dict(v) for k, v in bi.items()},
            {k: dict(v) for k, v in tri.items()},
            {k: dict(v) for k, v in quad.items()},
            list(char_to_id.keys()),
            digit_neutral=digit_neutral,
            reverse=reverse,
        )

    @classmethod
    def build_from_lexicon_and_hotpepper(
        cls, lex_path: Path, hotpepper_jsonl: Path, char_to_id: dict[str, int]
    ) -> "CharNGramLM":
        lex = json.loads(lex_path.read_text(encoding="utf-8"))
        weighted: list[tuple[str, int]] = (
            [(t, 10) for t in _extract_terms(lex.get("terms_521", []))]
            + [(t, 5) for t in _extract_terms(lex.get("approved_terms", []))]
            + [(t, 3) for t in _extract_terms(lex.get("context_phrases", []))]
            + [(t, 1) for t in _extract_terms(lex.get("db_terms", []))]
        )
        if hotpepper_jsonl.exists():
            seen: set[tuple[str, str]] = set()
            for line in hotpepper_jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                ct = clean_hotpepper_title(r["title"], r["tab"])
                if ct and (r["restaurant_id"], ct) not in seen:
                    seen.add((r["restaurant_id"], ct))
                    weighted.append((ct, 2))
        return cls._from_weighted_strings(weighted, char_to_id)

    @classmethod
    def build_from_cleaned_files(
        cls, lex_path: Path, cleaned_dir: Path, char_to_id: dict[str, int], reverse: bool = False
    ) -> "CharNGramLM":
        """Build 4-gram Char-LM from menu_lexicon.json + LLM-cleaned jmdict_food & HotPepper titles."""
        lex = json.loads(lex_path.read_text(encoding="utf-8"))
        jf_clean = json.loads((cleaned_dir / "jmdict_food_cleaned.json").read_text(encoding="utf-8"))["kept"]
        hp_path = cleaned_dir / "hotpepper_1010_cleaned.json"
        if not hp_path.exists():
            hp_path = cleaned_dir / "hotpepper_710_cleaned.json"
        if not hp_path.exists():
            hp_path = cleaned_dir / "hotpepper_310_cleaned.json"
        hp_clean = json.loads(hp_path.read_text(encoding="utf-8"))["kept_per_restaurant"]
        curated_grammar = [x["term"] for x in lex.get("db_terms", []) if x.get("category") == "curated_grammar"]

        weighted: list[tuple[str, int]] = (
            [(t, 10) for t in _extract_terms(lex.get("terms_521", []))]
            + [(t, 5) for t in _extract_terms(lex.get("approved_terms", []))]
            + [(t, 3) for t in _extract_terms(lex.get("context_phrases", []))]
            + [(t, 1) for t in curated_grammar]
            + [(t, 1) for t in jf_clean]
            + [(t, 2) for t in hp_clean]
        )
        return cls._from_weighted_strings(weighted, char_to_id, reverse=reverse)


def _logsumexp2(a: float, b: float) -> float:
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    m = a if a > b else b
    return m + math.log1p(math.exp(-abs(a - b)))


def ctc_char_lm_beam_decode(
    logp_tv: torch.Tensor,
    id_to_char: dict[int, str],
    lm: CharNGramLM,
    alpha_lm: float = 0.30,
    beam_width: int = 5,
    topk_t: int = 6,
    bwd_lm: CharNGramLM | None = None,
) -> str:
    """Decode (T, V) log-probabilities using CTC prefix beam search + 4-gram Char-LM."""
    if alpha_lm <= 0.0:
        ids = logp_tv.argmax(dim=-1).tolist()
        out, prev = [], 0
        for idx in ids:
            if idx != 0 and idx != prev:
                ch = id_to_char.get(idx, "")
                if ch:
                    out.append(ch)
            prev = idx
        return "".join(out)

    NEG_INF = float("-inf")
    beams: dict[tuple[str, ...], tuple[float, float, float]] = {(): (0.0, NEG_INF, 0.0)}
    T = logp_tv.shape[0]
    lm_cache: dict[tuple[str, str, str, str], float] = {}

    p_blk_arr = logp_tv[:, 0].tolist()
    k_eff = min(topk_t, max(1, logp_tv.shape[1] - 1))
    vals_t, idxs_t = torch.topk(logp_tv[:, 1:], k=k_eff, dim=-1)
    vals_list = vals_t.tolist()
    idxs_list = idxs_t.tolist()

    for t in range(T):
        p_blk_t = p_blk_arr[t]
        cands = [(i + 1, v) for i, v in zip(idxs_list[t], vals_list[t])]
        next_beams: dict[tuple[str, ...], list[float]] = {}

        def _acc(pref, pb, pnb, lms):
            cur = next_beams.get(pref)
            if cur is None:
                next_beams[pref] = [pb, pnb, lms]
            else:
                cur[0] = _logsumexp2(cur[0], pb)
                cur[1] = _logsumexp2(cur[1], pnb)

        for pref, (pb, pnb, lms) in beams.items():
            ptot = _logsumexp2(pb, pnb)
            _acc(pref, ptot + p_blk_t, NEG_INF, lms)
            last_ch = pref[-1] if pref else None
            p3 = pref[-3] if len(pref) >= 3 else "<BOS>"
            p2 = pref[-2] if len(pref) >= 2 else "<BOS>"
            p1 = pref[-1] if len(pref) >= 1 else "<BOS>"

            for cid, lp_c in cands:
                ch = id_to_char.get(cid)
                if not ch:
                    continue
                if ch == last_ch:
                    _acc(pref, NEG_INF, pnb + lp_c, lms)
                    if pb > NEG_INF:
                        key = (p3, p2, p1, ch)
                        sl = lm_cache.get(key)
                        if sl is None:
                            sl = lm.step_logp(p3, p2, p1, ch)
                            lm_cache[key] = sl
                        _acc(pref + (ch,), NEG_INF, pb + lp_c, lms + sl)
                else:
                    key = (p3, p2, p1, ch)
                    sl = lm_cache.get(key)
                    if sl is None:
                        sl = lm.step_logp(p3, p2, p1, ch)
                        lm_cache[key] = sl
                    _acc(pref + (ch,), NEG_INF, ptot + lp_c, lms + sl)

        ranked = sorted(
            next_beams.items(),
            key=lambda kv: _logsumexp2(kv[1][0], kv[1][1]) + alpha_lm * kv[1][2],
            reverse=True,
        )[:beam_width]
        beams = {k: (v[0], v[1], v[2]) for k, v in ranked}

    if bwd_lm is None:
        best = max(beams.items(), key=lambda kv: _logsumexp2(kv[1][0], kv[1][1]) + alpha_lm * kv[1][2])[0]
        return "".join(best)

    best_text = ""
    best_score = NEG_INF
    for pref, (pb, pnb, fwd_s) in beams.items():
        text = "".join(pref)
        tot = _logsumexp2(pb, pnb) + alpha_lm * 0.5 * (fwd_s + bwd_lm.score_seq(text))
        if tot > best_score:
            best_score = tot
            best_text = text
    return best_text
