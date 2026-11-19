"""Focused candidate tests: thread safety, schema compatibility, CTC brute-force (a=0 & a>0), and routing."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gzip
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile

import pytest
import torch

FOLLOWUP_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(FOLLOWUP_ROOT / "candidates")):
    if p not in sys.path:
        sys.path.insert(0, p)

from char_lm_beam_candidate import (
    CTX_DIG_UNK_V2,
    CTX_LEGACY_V1,
    CTX_RAW_UNK_V2,
    DecodeConfig,
    ImmutableCharNGramLM,
    IncompatibleLMSchemaError,
    ctc_beam_decode_stateless,
    decode_with_config,
    greedy_decode,
)


def _bruteforce_ctc_lm(logp_tv: torch.Tensor, id_to_char: dict[int, str], lm: ImmutableCharNGramLM | None, alpha: float, digit_neutral: bool = True) -> str:
    T, V = logp_tv.shape
    agg: dict[str, float] = {}
    for path in itertools.product(range(V), repeat=T):
        lp = sum(float(logp_tv[t, c].item()) for t, c in enumerate(path))
        out, prev = [], 0
        for c in path:
            if c != 0 and c != prev:
                ch = id_to_char.get(c, "")
                if ch:
                    out.append(ch)
            prev = c
        s = "".join(out)
        agg[s] = agg.get(s, 0.0) + math.exp(lp)

    best_s, best_sc = "", float("-inf")
    for s, p in agg.items():
        sc = math.log(max(1e-300, p))
        if lm is not None and alpha > 0.0:
            sc += alpha * lm.score_seq(s, digit_neutral=digit_neutral)
        if sc > best_sc:
            best_sc = sc
            best_s = s
    return best_s


def test_schema_roundtrip_all_modes_and_immutability():
    c2i = {"1": 1, "2": 2, "0": 3, "8": 4, "円": 5, "ニ": 6, "ラ": 7}
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        for dn in (True, False):
            for rev in (False, True):
                lm = ImmutableCharNGramLM.from_weighted_strings(
                    [("120円", 50), ("ニラ", 20)],
                    c2i,
                    digit_neutral=dn,
                    reverse=rev,
                    provenance={"nested": {"tag": "test"}},
                )
                p = tmp_path / f"lm_dn{dn}_rev{rev}.json.gz"
                lm.save_gz(p)
                loaded = ImmutableCharNGramLM.load_gz(p)
                assert loaded.schema_version == 2
                assert loaded.reverse is rev
                assert loaded.digit_neutral is dn
                assert loaded.context_transform == (CTX_DIG_UNK_V2 if dn else CTX_RAW_UNK_V2)
                assert loaded.supported_digit_neutral == (dn,)
                # 1. Public scalar attribute mutation blocked
                with pytest.raises(AttributeError):
                    loaded.digit_neutral = not dn  # type: ignore[misc]
                with pytest.raises(AttributeError):
                    loaded.schema_version = 1  # type: ignore[misc]
                with pytest.raises(AttributeError):
                    loaded.reverse = not rev  # type: ignore[misc]
                with pytest.raises(AttributeError):
                    del loaded.reverse  # type: ignore[misc]
                # 2. Table replacement blocked
                with pytest.raises(AttributeError):
                    loaded.uni = {}  # type: ignore[misc]
                with pytest.raises(AttributeError):
                    loaded.bi = {}  # type: ignore[misc]
                with pytest.raises(AttributeError):
                    loaded.vocab_chars = ()  # type: ignore[misc]
                # 3. Nested table & provenance dict writes blocked
                with pytest.raises(TypeError):
                    loaded.uni["1"] = 999  # type: ignore[index]
                first_bi_key = next(iter(loaded.bi.keys()))
                with pytest.raises(TypeError):
                    loaded.bi[first_bi_key]["1"] = 999  # type: ignore[index]
                with pytest.raises(TypeError):
                    loaded.provenance["nested"]["tag"] = "mutated"  # type: ignore[index]


def test_schema_rejects_mismatched_digit_mode_and_legacy_without_flag():
    c2i = {"1": 1, "2": 2, "0": 3, "8": 4, "円": 5}
    lm_dn = ImmutableCharNGramLM.from_weighted_strings([("120円", 100)], c2i, digit_neutral=True)
    # Toggling digit_neutral=False on a <DIG>-trained LM must raise IncompatibleLMSchemaError
    with pytest.raises(IncompatibleLMSchemaError):
        lm_dn.validate_mode(digit_neutral=False)
    # Confirm why: raw digit keys '0'..'9' do not exist in bi of a <DIG>-trained LM
    assert "<DIG>" in lm_dn.bi
    assert "0" not in lm_dn.bi and "1" not in lm_dn.bi

    # Legacy v1 file (no schema_version) must fail unless allow_legacy_v1=True
    with tempfile.TemporaryDirectory() as td:
        legacy_path = Path(td) / "legacy_v1.json.gz"
        legacy_obj = {"vocab_chars": list(c2i.keys()), "uni": {"1": 10}, "bi": {"1": {"2": 10}}, "tri": {}, "quad": {}}
        legacy_path.write_bytes(gzip.compress(json.dumps(legacy_obj).encode("utf-8")))
        with pytest.raises(IncompatibleLMSchemaError):
            ImmutableCharNGramLM.load_gz(legacy_path)
        loaded_legacy = ImmutableCharNGramLM.load_gz(legacy_path, allow_legacy_v1=True)
        assert loaded_legacy.context_transform == CTX_LEGACY_V1
        assert loaded_legacy.supported_digit_neutral == (False,)


def test_concurrent_stateless_requests_with_exceptions():
    c2i = {"1": 1, "2": 2, "0": 3, "8": 4, "円": 5, "ニ": 6, "二": 7, "ラ": 8}
    i2c = {v: k for k, v in c2i.items()}
    lm_dn = ImmutableCharNGramLM.from_weighted_strings([("120円", 300), ("ニラ", 50)], c2i, digit_neutral=True)
    lm_raw = ImmutableCharNGramLM.from_weighted_strings([("120円", 300), ("ニラ", 50)], c2i, digit_neutral=False)

    logits_price = torch.tensor(
        [
            [-5.0, 6.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0],
            [-5.0, -5.0, 6.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0],
            [-5.0, -5.0, -5.0, 2.7, 3.0, -5.0, -5.0, -5.0, -5.0],
            [-5.0, -5.0, -5.0, -5.0, -5.0, 6.0, -5.0, -5.0, -5.0],
        ],
        dtype=torch.float32,
    )
    lp_price = torch.log_softmax(logits_price, dim=-1)

    def _worker(idx: int):
        mod = idx % 4
        if mod == 0:
            cfg = DecodeConfig(mode="beam4g_dn", alpha_lm=0.30, digit_neutral=True)
            res = decode_with_config(lp_price, i2c, cfg, lm=lm_dn)
            assert res["text"] == "128円"
        elif mod == 1:
            cfg = DecodeConfig(mode="beam4g_raw", alpha_lm=0.30, digit_neutral=False)
            res = decode_with_config(lp_price, i2c, cfg, lm=lm_raw)
            assert res["text"] == "120円"
        elif mod == 2:
            # Forced schema mismatch exception must not corrupt shared LM for other threads
            bad_cfg = DecodeConfig(mode="beam4g_raw", alpha_lm=0.30, digit_neutral=False)
            with pytest.raises(IncompatibleLMSchemaError):
                decode_with_config(lp_price, i2c, bad_cfg, lm=lm_dn)
        else:
            cfg = DecodeConfig(mode="plain_beam", alpha_lm=0.0)
            res = decode_with_config(lp_price, i2c, cfg, lm=None)
            assert res["text"] == "128円"
        return True

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_worker, range(64)))
    assert all(results)
    assert lm_dn.supported_digit_neutral == (True,)


def test_bruteforce_ctc_alpha0_and_nonzero_lm_and_edge_cases():
    c2i = {"A": 1, "B": 2}
    i2c = {1: "A", 2: "B"}
    lm = ImmutableCharNGramLM.from_weighted_strings([("AB", 30), ("BA", 5)], c2i, digit_neutral=True)

    # 1. Brute-force parity for alpha=0.0 AND alpha=0.30 across 5 random tiny T=4, V=3 matrices (wide beam, top-all)
    for seed in range(5):
        g = torch.Generator().manual_seed(100 + seed)
        lp = torch.log_softmax(torch.randn(4, 3, generator=g), dim=-1)
        assert ctc_beam_decode_stateless(lp, i2c, lm=lm, alpha_lm=0.0, beam_width=50, topk_t=2) == _bruteforce_ctc_lm(lp, i2c, None, 0.0)
        assert ctc_beam_decode_stateless(lp, i2c, lm=lm, alpha_lm=0.30, beam_width=50, topk_t=2) == _bruteforce_ctc_lm(lp, i2c, lm, 0.30)

    # 2. All-blank, repeated with/without blank, unknown ID (cid=3 not in i2c)
    lp_blk = torch.log_softmax(torch.tensor([[6.0, -5.0, -5.0]] * 3), dim=-1)
    assert ctc_beam_decode_stateless(lp_blk, i2c, lm=None, alpha_lm=0.0, beam_width=5, topk_t=2) == ""

    lp_rep = torch.log_softmax(torch.tensor([[-5.0, 6.0, -5.0], [-5.0, 6.0, -5.0], [6.0, -5.0, -5.0], [-5.0, 6.0, -5.0]]), dim=-1)
    assert ctc_beam_decode_stateless(lp_rep, i2c, lm=None, alpha_lm=0.0, beam_width=5, topk_t=2) == "AA"

    lp_unk = torch.log_softmax(torch.tensor([[-5.0, -5.0, -5.0, 6.0], [-5.0, 6.0, -5.0, -5.0]]), dim=-1)
    assert ctc_beam_decode_stateless(lp_unk, i2c, lm=None, alpha_lm=0.0, beam_width=5, topk_t=3) == "A"

    # 3. True plain beam alpha=0 diverges from greedy on alignment-sum ambiguity
    lp_div = torch.log(torch.tensor([[0.40, 0.35, 0.25], [0.40, 0.35, 0.25]], dtype=torch.float32))
    assert greedy_decode(lp_div, i2c) == ""
    assert ctc_beam_decode_stateless(lp_div, i2c, lm=lm, alpha_lm=0.0, beam_width=5, topk_t=2) == "A"
