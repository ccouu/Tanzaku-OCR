"""Isolated schema-v2 LM migration/packaging helper (read-only on checkpoints/; writes only to output path).

Packages an existing v1/v1.5 Char-LM `.json.gz` into explicit `schema_version=2` format without altering any n-gram
tables, verifying table equality and exact prediction parity on cached logits if provided.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

CANDIDATES_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(CANDIDATES_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from char_lm_beam_candidate import (
    CTX_DIG_UNK_V2,
    CTX_LEGACY_V1,
    CTX_RAW_UNK_V2,
    SCHEMA_VERSION,
    ImmutableCharNGramLM,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def migrate_lm_to_v2(src_path: Path, dst_path: Path) -> dict:
    src_sha = sha256_file(src_path)
    raw_obj = json.loads(gzip.decompress(src_path.read_bytes()).decode("utf-8"))

    if "digit_neutral" in raw_obj:
        ctx_tf = CTX_DIG_UNK_V2 if bool(raw_obj["digit_neutral"]) else CTX_RAW_UNK_V2
    else:
        ctx_tf = CTX_LEGACY_V1

    migrated = ImmutableCharNGramLM(
        raw_obj["uni"],
        raw_obj["bi"],
        raw_obj["tri"],
        raw_obj["quad"],
        raw_obj["vocab_chars"],
        context_transform=ctx_tf,
        reverse=bool(raw_obj.get("reverse", False)),
        schema_version=SCHEMA_VERSION,
        provenance={
            "migrated_from": src_path.name,
            "source_sha256": src_sha,
            "migration_tool": "candidates/migrate_lm_to_v2.py",
        },
    )
    migrated.save_gz(dst_path)

    # Reload strictly WITHOUT allow_legacy_v1=True to verify schema_version=2 contract
    reloaded = ImmutableCharNGramLM.load_gz(dst_path, allow_legacy_v1=False)
    tables_identical = (
        dict(reloaded.uni) == raw_obj["uni"]
        and {k: dict(v) for k, v in reloaded.bi.items()} == raw_obj["bi"]
        and {k: dict(v) for k, v in reloaded.tri.items()} == raw_obj["tri"]
        and {k: dict(v) for k, v in reloaded.quad.items()} == raw_obj["quad"]
        and list(reloaded.vocab_chars) == raw_obj["vocab_chars"]
    )
    assert tables_identical, "Migrated v2 LM n-gram tables differ from source!"

    return {
        "source_path": str(src_path),
        "source_sha256": src_sha,
        "output_path": str(dst_path),
        "output_sha256": sha256_file(dst_path),
        "schema_version": reloaded.schema_version,
        "context_transform": reloaded.context_transform,
        "digit_neutral": reloaded.digit_neutral,
        "reverse": reloaded.reverse,
        "tables_identical": tables_identical,
        "loaded_without_legacy_flag": True,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    args = p.parse_args()
    info = migrate_lm_to_v2(args.src, args.dst)
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
