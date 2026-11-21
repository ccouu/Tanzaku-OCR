"""Versioned provenance-bearing lexicon index builder, serializer, and Trie.

Builds a tiered, deduplicated lexicon index:
  1. `curated_core`: approved_terms (646) + terms_521 (521 subset) from menu_lexicon.json
  2. `context_phrase`: context_phrases (1886) from menu_lexicon.json
  3. `jmdict_food_cleaned`: kept (10,539) from jmdict_food_cleaned.json (separately tagged)

Does NOT indiscriminately import all 45k raw DB terms or promotional HotPepper titles.
Translation IDs (`db_id`) are linked from read-only `culinary_menu.db`.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional

from .contracts import LexiconEntry
from .normalization import canonicalize_text
from .translation_repo import ReadOnlyTranslationRepo

INDEX_SCHEMA_VERSION = 1
INDEX_FORMAT = "lexicon_trie_v1"

TIER_PRIORITY = {
    "curated_core": 0,
    "context_phrase": 1,
    "jmdict_food_cleaned": 2,
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TrieNode:
    """Compact read-only trie node after construction."""

    __slots__ = ("children", "entry_ids", "min_remaining_len", "max_remaining_len")

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.entry_ids: list[int] = []
        self.min_remaining_len: int = 10**9
        self.max_remaining_len: int = 0

    def freeze(self) -> None:
        for child in self.children.values():
            child.freeze()
        if self.entry_ids:
            self.min_remaining_len = 0
        for child in self.children.values():
            if child.min_remaining_len + 1 < self.min_remaining_len:
                self.min_remaining_len = child.min_remaining_len + 1
            if child.max_remaining_len + 1 > self.max_remaining_len:
                self.max_remaining_len = child.max_remaining_len + 1
        self.children = MappingProxyType(self.children)  # type: ignore[assignment]
        self.entry_ids = tuple(self.entry_ids)  # type: ignore[assignment]


class LexiconIndex:
    """Immutable in-memory Trie + provenance-bearing lexicon table."""

    def __init__(
        self,
        entries: tuple[LexiconEntry, ...],
        metadata: Mapping[str, Any],
    ) -> None:
        self._entries: tuple[LexiconEntry, ...] = tuple(entries)
        self._by_term: Mapping[str, LexiconEntry] = MappingProxyType(
            {e.canonical_term: e for e in self._entries}
        )
        self._metadata: Mapping[str, Any] = MappingProxyType(dict(metadata))
        self._root = TrieNode()
        for entry in self._entries:
            node = self._root
            for ch in entry.canonical_term:
                nxt = node.children.get(ch)
                if nxt is None:
                    nxt = TrieNode()
                    node.children[ch] = nxt
                node = nxt
            node.entry_ids.append(entry.entry_id)
        self._root.freeze()

    @property
    def root(self) -> TrieNode:
        return self._root

    @property
    def entries(self) -> tuple[LexiconEntry, ...]:
        return self._entries

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    def get_entry(self, entry_id: int) -> LexiconEntry:
        return self._entries[entry_id]

    def get_by_term(self, canonical_term: str) -> Optional[LexiconEntry]:
        return self._by_term.get(canonical_term)

    def tier_allowed(
        self,
        tier: str,
        *,
        include_context_phrases: bool = True,
        include_expanded_culinary: bool = True,
    ) -> bool:
        if tier == "curated_core":
            return True
        if tier == "context_phrase":
            return include_context_phrases
        if tier == "jmdict_food_cleaned":
            return include_expanded_culinary
        return False

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "index_format": INDEX_FORMAT,
            "metadata": dict(self._metadata),
            "entries": [e.to_dict() for e in self._entries],
        }

    def save_gz(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_payload()
        with gzip.open(out, "wt", encoding="utf-8", compresslevel=6) as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        return out

    @classmethod
    def load_gz(cls, path: str | Path) -> "LexiconIndex":
        p = Path(path)
        with gzip.open(p, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("schema_version") != INDEX_SCHEMA_VERSION or payload.get("index_format") != INDEX_FORMAT:
            raise ValueError(
                f"Unsupported lexicon index schema: version={payload.get('schema_version')!r}, "
                f"format={payload.get('index_format')!r}"
            )
        entries = tuple(
            LexiconEntry(
                entry_id=int(r["entry_id"]),
                canonical_term=str(r["canonical_term"]),
                display_term=str(r["display_term"]),
                source_tier=str(r["source_tier"]),
                sources=tuple(r["sources"]),
                category=str(r.get("category", "")),
                standalone_tier=str(r.get("standalone_tier", "")),
                occurrence_tier=str(r.get("occurrence_tier", "")),
                db_id=int(r["db_id"]) if r.get("db_id") is not None else None,
                aliases=tuple(r.get("aliases", ())),
            )
            for r in payload["entries"]
        )
        meta = dict(payload.get("metadata", {}))
        meta["loaded_from"] = str(p.resolve())
        meta["index_sha256"] = sha256_file(p)
        return cls(entries=entries, metadata=meta)


@dataclass
class _MutableEntry:
    canonical_term: str
    display_term: str
    source_tier: str
    sources: set[str]
    category: str = ""
    standalone_tier: str = ""
    occurrence_tier: str = ""
    db_id: Optional[int] = None
    aliases: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.aliases is None:
            self.aliases = set()


def _is_pure_punctuation_or_space(text: str) -> bool:
    return not any(ch.isalnum() or ord(ch) >= 0x3000 for ch in text)


def build_lexicon_index(
    *,
    menu_lexicon_path: str | Path,
    jmdict_cleaned_path: Optional[str | Path] = None,
    culinary_db_path: Optional[str | Path] = None,
    include_context_phrases: bool = True,
    include_expanded_culinary: bool = True,
) -> LexiconIndex:
    """Build a provenance-bearing tiered LexiconIndex from frozen sources."""
    menu_path = Path(menu_lexicon_path).resolve()
    menu_data = json.loads(menu_path.read_text(encoding="utf-8"))

    source_hashes: dict[str, str] = {
        "menu_lexicon_json": sha256_file(menu_path),
    }

    db_map: dict[str, int] = {}
    if culinary_db_path is not None:
        db_p = Path(culinary_db_path).resolve()
        source_hashes["culinary_menu_db"] = sha256_file(db_p)
        repo = ReadOnlyTranslationRepo(db_p)
        try:
            db_map = repo.load_canonical_to_id_map()
        finally:
            repo.close_thread_connection()

    records: dict[str, _MutableEntry] = {}

    def _upsert(
        raw_term: str,
        *,
        tier: str,
        source_tag: str,
        category: str = "",
        standalone_tier: str = "",
        occurrence_tier: str = "",
    ) -> None:
        raw_clean = raw_term.strip()
        if not raw_clean or _is_pure_punctuation_or_space(raw_clean):
            return
        canon = canonicalize_text(raw_clean)
        if not canon or _is_pure_punctuation_or_space(canon):
            return
        db_id = db_map.get(canon) or db_map.get(raw_clean)
        existing = records.get(canon)
        if existing is None:
            rec = _MutableEntry(
                canonical_term=canon,
                display_term=raw_clean,
                source_tier=tier,
                sources={source_tag},
                category=category,
                standalone_tier=standalone_tier,
                occurrence_tier=occurrence_tier,
                db_id=db_id,
            )
            if raw_clean != canon:
                rec.aliases.add(raw_clean)
            records[canon] = rec
            return
        existing.sources.add(source_tag)
        if TIER_PRIORITY[tier] < TIER_PRIORITY[existing.source_tier]:
            existing.source_tier = tier
            existing.display_term = raw_clean
        if category and not existing.category:
            existing.category = category
        if standalone_tier and not existing.standalone_tier:
            existing.standalone_tier = standalone_tier
        if occurrence_tier and not existing.occurrence_tier:
            existing.occurrence_tier = occurrence_tier
        if db_id is not None and existing.db_id is None:
            existing.db_id = db_id
        if raw_clean != canon:
            existing.aliases.add(raw_clean)

    # 1. Curated core: approved_terms (646) + terms_521 (521)
    for row in menu_data.get("approved_terms", ()):
        _upsert(
            str(row["term"]),
            tier="curated_core",
            source_tag="menu_lexicon:approved_terms",
            category=str(row.get("category", "")),
            standalone_tier=str(row.get("standalone_tier", "")),
            occurrence_tier=str(row.get("occurrence_tier", "")),
        )
    for term in menu_data.get("terms_521", ()):
        _upsert(str(term), tier="curated_core", source_tag="menu_lexicon:terms_521")

    # 2. Context phrases (1886)
    if include_context_phrases:
        for phrase in menu_data.get("context_phrases", ()):
            _upsert(str(phrase), tier="context_phrase", source_tag="menu_lexicon:context_phrases")

    # 3. Optional cleaned JMdict culinary expansion (10,539 kept)
    if include_expanded_culinary and jmdict_cleaned_path is not None:
        jm_path = Path(jmdict_cleaned_path).resolve()
        source_hashes["jmdict_food_cleaned_json"] = sha256_file(jm_path)
        jm_data = json.loads(jm_path.read_text(encoding="utf-8"))
        for term in jm_data.get("kept", ()):
            _upsert(
                str(term),
                tier="jmdict_food_cleaned",
                source_tag="jmdict_food_cleaned:kept",
                category="jmdict_food",
            )

    sorted_keys = sorted(records.keys())
    entries: list[LexiconEntry] = []
    tier_counts: dict[str, int] = {k: 0 for k in TIER_PRIORITY}
    linked_db_count = 0

    for idx, key in enumerate(sorted_keys):
        rec = records[key]
        tier_counts[rec.source_tier] = tier_counts.get(rec.source_tier, 0) + 1
        if rec.db_id is not None:
            linked_db_count += 1
        entries.append(
            LexiconEntry(
                entry_id=idx,
                canonical_term=rec.canonical_term,
                display_term=rec.display_term,
                source_tier=rec.source_tier,
                sources=tuple(sorted(rec.sources)),
                category=rec.category,
                standalone_tier=rec.standalone_tier,
                occurrence_tier=rec.occurrence_tier,
                db_id=rec.db_id,
                aliases=tuple(sorted(rec.aliases)),
            )
        )

    metadata = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "index_format": INDEX_FORMAT,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": source_hashes,
        "build_config": {
            "include_context_phrases": bool(include_context_phrases),
            "include_expanded_culinary": bool(include_expanded_culinary),
            "imported_raw_db_45k": False,
        },
        "total_entries": len(entries),
        "tier_counts": tier_counts,
        "linked_db_entries": linked_db_count,
    }
    return LexiconIndex(entries=tuple(entries), metadata=metadata)
