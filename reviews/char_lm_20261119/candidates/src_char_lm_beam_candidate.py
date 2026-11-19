"""Lightweight 4-gram Character Language Model + CTC Prefix Beam Search decoder."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping
import unicodedata

import torch

from src.recognition.label_normalization import normalize_parentheses

PROMO_BRACKET_RE = re.compile(r"【[^】]*】")
TRAILING_PRICE_RE = re.compile(r"\s*[0-9,]+円(?:（税込）|\(税込\)|（税抜）|\(税抜\)|〜|~)*.*$")
DECOR_RE = re.compile(r"[♪★☆◆◇◎●○■□▲△▼▽！!？?]+")

SCHEMA_VERSION = 2
CTX_DIG_UNK_V2 = "dig_unk_v2"
CTX_RAW_UNK_V2 = "raw_unk_v2"
CTX_LEGACY_V1 = "legacy_raw_v1"

NUM_SPAN_RE = re.compile(r"[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?")


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


@dataclass(frozen=True)
class DecodeConfig:
    """Immutable per-request decoding configuration (thread-safe across concurrent requests)."""

    mode: str = "beam4g_dn"  # "greedy" | "plain_beam" | "beam4g_dn" | "beam4g_raw" | "bidir4g_dn"
    alpha_lm: float = 0.30
    beam_width: int = 5
    topk_t: int = 6
    digit_neutral: bool = True
    routing: str = "none"  # "none" | "r1_short_greedy" | "r2_short_and_numeric_guard"
    short_greedy_max_chars: int = 2
    allow_mismatched_schema: bool = False


class IncompatibleLMSchemaError(ValueError):
    """Raised when a serialized LM schema or context transform is incompatible with requested decode mode."""


def _freeze_nested(d: dict[str, dict[str, int]]) -> Mapping[str, Mapping[str, int]]:
    return MappingProxyType({k: MappingProxyType(dict(v)) for k, v in d.items()})


def _freeze_value(v):
    if isinstance(v, Mapping):
        return MappingProxyType({k: _freeze_value(val) for k, val in v.items()})
    if isinstance(v, (list, tuple)):
        return tuple(_freeze_value(x) for x in v)
    return v


def _unfreeze_value(v):
    if isinstance(v, Mapping):
        return {k: _unfreeze_value(val) for k, val in v.items()}
    if isinstance(v, tuple):
        return [_unfreeze_value(x) for x in v]
    return v


class ImmutableCharNGramLM:
    """Thread-safe, fully immutable 4-gram Char-LM (frozen attributes + MappingProxyType tables) with schema provenance."""

    __slots__ = (
        "schema_version",
        "context_transform",
        "digit_neutral",
        "reverse",
        "vocab_chars",
        "vocab_sha256",
        "provenance",
        "supported_digit_neutral",
        "uni",
        "bi",
        "tri",
        "quad",
        "vocab_size",
        "p_uni",
        "default_uni",
        "uniform_digit_lp",
        "bi_tot",
        "tri_tot",
        "quad_tot",
    )

    def __init__(
        self,
        uni: dict[str, int],
        bi: dict[str, dict[str, int]],
        tri: dict[str, dict[str, int]],
        quad: dict[str, dict[str, int]],
        vocab_chars: list[str],
        *,
        context_transform: str = CTX_DIG_UNK_V2,
        reverse: bool = False,
        schema_version: int = SCHEMA_VERSION,
        vocab_sha256: str = "",
        provenance: dict | None = None,
    ) -> None:
        ctx_tf = str(context_transform)
        if ctx_tf == CTX_DIG_UNK_V2:
            supported_dn = (True,)
        elif ctx_tf in (CTX_RAW_UNK_V2, CTX_LEGACY_V1):
            supported_dn = (False,)
        else:
            raise IncompatibleLMSchemaError(f"Unknown context_transform: {ctx_tf}")

        vc_tuple = tuple(vocab_chars)
        v_sha = vocab_sha256 or hashlib.sha256("".join(vc_tuple).encode("utf-8")).hexdigest()
        uni_m: Mapping[str, int] = MappingProxyType(dict(uni))
        bi_m = _freeze_nested(bi)
        tri_m = _freeze_nested(tri)
        quad_m = _freeze_nested(quad)
        v_size = max(1, len(vc_tuple))
        total_uni = sum(uni_m.values())

        _set = object.__setattr__
        _set(self, "schema_version", int(schema_version))
        _set(self, "context_transform", ctx_tf)
        _set(self, "digit_neutral", ctx_tf == CTX_DIG_UNK_V2)
        _set(self, "reverse", bool(reverse))
        _set(self, "vocab_chars", vc_tuple)
        _set(self, "vocab_sha256", v_sha)
        _set(self, "provenance", _freeze_value(dict(provenance or {})))
        _set(self, "supported_digit_neutral", supported_dn)
        _set(self, "uni", uni_m)
        _set(self, "bi", bi_m)
        _set(self, "tri", tri_m)
        _set(self, "quad", quad_m)
        _set(self, "vocab_size", v_size)
        _set(
            self,
            "p_uni",
            MappingProxyType({ch: (uni_m.get(ch, 0) + 1.0) / (total_uni + v_size) for ch in vc_tuple}),
        )
        _set(self, "default_uni", 1.0 / (total_uni + v_size))
        _set(self, "uniform_digit_lp", math.log(0.1))
        _set(self, "bi_tot", MappingProxyType({k: sum(v.values()) for k, v in bi_m.items()}))
        _set(self, "tri_tot", MappingProxyType({k: sum(v.values()) for k, v in tri_m.items()}))
        _set(self, "quad_tot", MappingProxyType({k: sum(v.values()) for k, v in quad_m.items()}))

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError(f"ImmutableCharNGramLM is immutable; cannot set attribute {name!r}")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"ImmutableCharNGramLM is immutable; cannot delete attribute {name!r}")

    def validate_mode(self, digit_neutral: bool, allow_mismatched_schema: bool = False) -> None:
        if digit_neutral not in self.supported_digit_neutral and not allow_mismatched_schema:
            raise IncompatibleLMSchemaError(
                f"LM schema_version={self.schema_version} context_transform={self.context_transform!r} "
                f"supports digit_neutral in {self.supported_digit_neutral}, got digit_neutral={digit_neutral}. "
                f"Use a compatibly built LM or pass allow_mismatched_schema=True for explicit diagnostic testing."
            )

    def step_logp(self, p3: str, p2: str, p1: str, ch: str, *, digit_neutral: bool) -> float:
        if digit_neutral:
            if "0" <= ch <= "9":
                return self.uniform_digit_lp
            if self.context_transform == CTX_DIG_UNK_V2:
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

    def score_seq(self, text: str, *, digit_neutral: bool) -> float:
        chars = list(reversed(text)) if self.reverse else list(text)
        p1 = p2 = p3 = "<BOS>"
        total = 0.0
        for c in chars:
            total += self.step_logp(p3, p2, p1, c, digit_neutral=digit_neutral)
            p3, p2, p1 = p2, p1, c
        return total

    def save_gz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "context_transform": self.context_transform,
            "digit_neutral": self.context_transform == CTX_DIG_UNK_V2,
            "reverse": self.reverse,
            "vocab_sha256": self.vocab_sha256,
            "provenance": _unfreeze_value(self.provenance),
            "vocab_chars": list(self.vocab_chars),
            "uni": dict(self.uni),
            "bi": {k: dict(v) for k, v in self.bi.items()},
            "tri": {k: dict(v) for k, v in self.tri.items()},
            "quad": {k: dict(v) for k, v in self.quad.items()},
        }
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        path.write_bytes(gzip.compress(raw, compresslevel=6))

    @classmethod
    def load_gz(cls, path: Path, *, allow_legacy_v1: bool = False) -> "ImmutableCharNGramLM":
        obj = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
        sv = obj.get("schema_version")
        if sv is None:
            if not allow_legacy_v1:
                raise IncompatibleLMSchemaError(
                    f"{path.name} has no 'schema_version' (legacy format). "
                    f"Pass allow_legacy_v1=True to load explicitly under legacy semantics."
                )
            if "digit_neutral" in obj:
                ctx_tf = CTX_DIG_UNK_V2 if bool(obj["digit_neutral"]) else CTX_RAW_UNK_V2
                sv_eff = 1
            else:
                ctx_tf = CTX_LEGACY_V1
                sv_eff = 1
            return cls(
                obj["uni"],
                obj["bi"],
                obj["tri"],
                obj["quad"],
                obj["vocab_chars"],
                context_transform=ctx_tf,
                reverse=bool(obj.get("reverse", False)),
                schema_version=sv_eff,
                provenance={"loaded_as_legacy": True, "source_file": path.name},
            )
        return cls(
            obj["uni"],
            obj["bi"],
            obj["tri"],
            obj["quad"],
            obj["vocab_chars"],
            context_transform=str(obj["context_transform"]),
            reverse=bool(obj.get("reverse", False)),
            schema_version=int(sv),
            vocab_sha256=str(obj.get("vocab_sha256", "")),
            provenance=obj.get("provenance"),
        )

    @classmethod
    def from_weighted_strings(
        cls,
        weighted: list[tuple[str, int]],
        char_to_id: dict[str, int],
        *,
        digit_neutral: bool = True,
        reverse: bool = False,
        provenance: dict | None = None,
    ) -> "ImmutableCharNGramLM":
        ctx_tf = CTX_DIG_UNK_V2 if digit_neutral else CTX_RAW_UNK_V2
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
            context_transform=ctx_tf,
            reverse=reverse,
            schema_version=SCHEMA_VERSION,
            provenance=provenance,
        )

    @classmethod
    def build_from_lexicon_and_hotpepper(
        cls,
        lex_path: Path,
        hotpepper_jsonl: Path,
        char_to_id: dict[str, int],
        *,
        digit_neutral: bool = True,
        reverse: bool = False,
    ) -> "ImmutableCharNGramLM":
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
        return cls.from_weighted_strings(
            weighted,
            char_to_id,
            digit_neutral=digit_neutral,
            reverse=reverse,
            provenance={"builder": "build_from_lexicon_and_hotpepper"},
        )

    @classmethod
    def build_from_cleaned_files(
        cls,
        lex_path: Path,
        cleaned_dir: Path,
        char_to_id: dict[str, int],
        *,
        reverse: bool = False,
        digit_neutral: bool = True,
    ) -> "ImmutableCharNGramLM":
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
        return cls.from_weighted_strings(
            weighted,
            char_to_id,
            digit_neutral=digit_neutral,
            reverse=reverse,
            provenance={"builder": "build_from_cleaned_files", "hp_file": hp_path.name},
        )


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


def greedy_decode(logp_tv: torch.Tensor, id_to_char: Mapping[int, str]) -> str:
    ids = logp_tv.argmax(dim=-1).tolist()
    out, prev = [], 0
    for idx in ids:
        if idx != 0 and idx != prev:
            ch = id_to_char.get(idx, "")
            if ch:
                out.append(ch)
        prev = idx
    return "".join(out)


def ctc_beam_decode_stateless(
    logp_tv: torch.Tensor,
    id_to_char: Mapping[int, str],
    lm: ImmutableCharNGramLM | None = None,
    *,
    alpha_lm: float = 0.30,
    beam_width: int = 5,
    topk_t: int = 6,
    digit_neutral: bool = True,
    bwd_lm: ImmutableCharNGramLM | None = None,
    allow_mismatched_schema: bool = False,
) -> str:
    """TRUE CTC prefix beam search: alpha_lm==0.0 (or lm is None) performs genuine plain CTC beam, never silent argmax."""
    use_lm = lm is not None and alpha_lm > 0.0
    if use_lm:
        lm.validate_mode(digit_neutral, allow_mismatched_schema=allow_mismatched_schema)
        if bwd_lm is not None:
            bwd_lm.validate_mode(digit_neutral, allow_mismatched_schema=allow_mismatched_schema)

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
                        sl = 0.0
                        if use_lm:
                            key = (p3, p2, p1, ch)
                            sl = lm_cache.get(key)
                            if sl is None:
                                sl = lm.step_logp(p3, p2, p1, ch, digit_neutral=digit_neutral)
                                lm_cache[key] = sl
                        _acc(pref + (ch,), NEG_INF, pb + lp_c, lms + sl)
                else:
                    sl = 0.0
                    if use_lm:
                        key = (p3, p2, p1, ch)
                        sl = lm_cache.get(key)
                        if sl is None:
                            sl = lm.step_logp(p3, p2, p1, ch, digit_neutral=digit_neutral)
                            lm_cache[key] = sl
                    _acc(pref + (ch,), NEG_INF, ptot + lp_c, lms + sl)

        ranked = sorted(
            next_beams.items(),
            key=lambda kv: _logsumexp2(kv[1][0], kv[1][1]) + (alpha_lm * kv[1][2] if use_lm else 0.0),
            reverse=True,
        )[:beam_width]
        beams = {k: (v[0], v[1], v[2]) for k, v in ranked}

    if not use_lm or bwd_lm is None:
        best = max(
            beams.items(),
            key=lambda kv: _logsumexp2(kv[1][0], kv[1][1]) + (alpha_lm * kv[1][2] if use_lm else 0.0),
        )[0]
        return "".join(best)

    best_text = ""
    best_score = NEG_INF
    for pref, (pb, pnb, fwd_s) in beams.items():
        text = "".join(pref)
        tot = _logsumexp2(pb, pnb) + alpha_lm * 0.5 * (fwd_s + bwd_lm.score_seq(text, digit_neutral=digit_neutral))
        if tot > best_score:
            best_score = tot
            best_text = text
    return best_text


def decode_with_config(
    logp_tv: torch.Tensor,
    id_to_char: Mapping[int, str],
    config: DecodeConfig,
    lm: ImmutableCharNGramLM | None = None,
    bwd_lm: ImmutableCharNGramLM | None = None,
) -> dict:
    """Stateless, thread-safe decode entrypoint supporting explicit routing candidates."""
    greedy_text = greedy_decode(logp_tv, id_to_char)
    if config.mode == "greedy":
        return {"text": greedy_text, "greedy": greedy_text, "lm_alternative": None, "route_used": "greedy"}

    if config.mode == "plain_beam":
        pb = ctc_beam_decode_stateless(
            logp_tv, id_to_char, lm=None, alpha_lm=0.0, beam_width=config.beam_width, topk_t=config.topk_t
        )
        return {"text": pb, "greedy": greedy_text, "lm_alternative": None, "route_used": "plain_beam"}

    g_canon = normalize_parentheses(greedy_text).strip()

    # Candidate R1: inference-only short predicted canonical greedy fallback (<= short_greedy_max_chars)
    if config.routing in ("r1_short_greedy", "r2_short_and_numeric_guard"):
        if len(g_canon) <= config.short_greedy_max_chars:
            return {
                "text": greedy_text,
                "greedy": greedy_text,
                "lm_alternative": None,
                "route_used": "short_greedy_fallback",
            }

    use_bwd = bwd_lm if config.mode == "bidir4g_dn" else None
    dn = False if config.mode == "beam4g_raw" else config.digit_neutral
    lm_text = ctc_beam_decode_stateless(
        logp_tv,
        id_to_char,
        lm=lm,
        alpha_lm=config.alpha_lm,
        beam_width=config.beam_width,
        topk_t=config.topk_t,
        digit_neutral=dn,
        bwd_lm=use_bwd,
        allow_mismatched_schema=config.allow_mismatched_schema,
    )

    # Candidate R2: R1 + numeric-span disagreement guard (retains greedy and exposes LM alternative)
    if config.routing == "r2_short_and_numeric_guard":
        g_nums = NUM_SPAN_RE.findall(g_canon)
        l_nums = NUM_SPAN_RE.findall(normalize_parentheses(lm_text).strip())
        if g_nums and g_nums != l_nums:
            return {
                "text": greedy_text,
                "greedy": greedy_text,
                "lm_alternative": lm_text,
                "route_used": "numeric_disagreement_keep_greedy",
            }

    return {"text": lm_text, "greedy": greedy_text, "lm_alternative": None, "route_used": config.mode}


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
