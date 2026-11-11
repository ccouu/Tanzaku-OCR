"""Stage 1 filesystem and provenance helpers (no credentials or model code)."""
import hashlib
import json
from pathlib import Path

from .paths import ROOT, resolve_path


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def digest(path):
    path = resolve_path(path)
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(root, relative):
    root = Path(root).resolve()
    p = (root / relative).resolve()
    if not p.is_relative_to(root):
        raise ValueError('Path escapes dataset root')
    return p
