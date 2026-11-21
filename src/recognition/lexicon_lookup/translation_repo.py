"""Strictly read-only SQLite repository for culinary_menu.db translations.

Never modifies the database, never invents translations, and never fabricates
allergen or dietary claims. Missing records are returned explicitly with
`status="missing"`.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Optional

from .contracts import TranslationRecord


class ReadOnlyTranslationRepo:
    """Thread-safe read-only adapter over `culinary_menu.db`."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path).resolve()
        if not self._db_path.is_file():
            raise FileNotFoundError(f"Translation SQLite DB not found: {self._db_path}")
        self._uri = f"file:{self._db_path.as_posix()}?mode=ro"
        self._local = threading.local()
        # Probe read-only connection immediately to fail fast on bad paths.
        conn = self._get_conn()
        conn.execute("SELECT 1 FROM menu_dictionary LIMIT 1").fetchone()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._uri, uri=True, check_same_thread=True)
            conn.execute("PRAGMA query_only = ON")
            self._local.conn = conn
        return conn

    def close_thread_connection(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def get_by_id(self, db_id: Optional[int]) -> TranslationRecord:
        if db_id is None:
            return TranslationRecord(status="unlinked")
        row = self._get_conn().execute(
            "SELECT id, canonical_jp, reading_hiragana, romaji, name_en, description_en, category, source "
            "FROM menu_dictionary WHERE id = ?",
            (int(db_id),),
        ).fetchone()
        if row is None:
            return TranslationRecord(status="missing", db_id=int(db_id))
        return TranslationRecord(
            status="found",
            db_id=int(row[0]),
            canonical_jp=str(row[1]),
            reading_hiragana=str(row[2]),
            romaji=str(row[3]),
            name_en=str(row[4]),
            description_en=str(row[5]),
            category=str(row[6]),
            source=str(row[7]),
        )

    def get_by_term(self, canonical_jp: str) -> TranslationRecord:
        if not canonical_jp:
            return TranslationRecord(status="missing")
        row = self._get_conn().execute(
            "SELECT id, canonical_jp, reading_hiragana, romaji, name_en, description_en, category, source "
            "FROM menu_dictionary WHERE canonical_jp = ?",
            (canonical_jp,),
        ).fetchone()
        if row is None:
            return TranslationRecord(status="missing", canonical_jp=canonical_jp)
        return TranslationRecord(
            status="found",
            db_id=int(row[0]),
            canonical_jp=str(row[1]),
            reading_hiragana=str(row[2]),
            romaji=str(row[3]),
            name_en=str(row[4]),
            description_en=str(row[5]),
            category=str(row[6]),
            source=str(row[7]),
        )

    def load_canonical_to_id_map(self) -> dict[str, int]:
        """Load `{canonical_jp: id}` mapping for offline index construction."""
        rows = self._get_conn().execute("SELECT canonical_jp, id FROM menu_dictionary").fetchall()
        return {str(term): int(db_id) for term, db_id in rows}
