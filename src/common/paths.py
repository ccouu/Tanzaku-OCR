"""Canonical defaults and explicit compatibility with historical artifact paths."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_ROOT = ROOT / 'data/processed/synthetic_base_v1'
ANNOTATION_ROOT = ROOT / 'data/annotations/menu_obb'
DATASET_V1 = ROOT / 'data/datasets/detection/reviewed_v1'
DATASET_V2 = ROOT / 'data/datasets/detection/reviewed_v2'
TRAINING_OUTPUT = ROOT / 'outputs/detection/training/colab_v1'
INFERENCE_OUTPUT = ROOT / 'outputs/detection/inference/tabelog_100'
RUNTIME_ROOT = ROOT / '.cache/ultralytics'
_map_file = ROOT / 'workspace_paths.json'
RELOCATIONS = json.loads(_map_file.read_text(encoding='utf-8')) if _map_file.exists() else {}


def resolve_path(path):
    """Translate known workspace paths; leave unrelated/custom paths alone.

    Historical names remain aliases even if the obsolete directory is recreated.
    External paths never participate in the workspace mapping.
    """
    path = Path(path)
    absolute = path.resolve()
    try:
        relative = absolute.relative_to(ROOT).as_posix()
    except ValueError:
        return path
    for old in sorted(RELOCATIONS, key=len, reverse=True):
        if relative == old or relative.startswith(old + '/'):
            return ROOT / RELOCATIONS[old] / relative[len(old):].lstrip('/')
    return path


def dataset_root(root):
    root = Path(root).resolve()
    return SYNTHETIC_ROOT if root == ROOT / 'data' else resolve_path(root)


def historical_path(path):
    """Stable logical names for existing consolidation IDs after relocation."""
    absolute = Path(path).resolve()
    if not absolute.is_relative_to(ROOT):
        return Path(path).as_posix()
    relative = absolute.relative_to(ROOT).as_posix()
    for old, new in sorted(RELOCATIONS.items(), key=lambda pair: len(pair[1]), reverse=True):
        if relative == new or relative.startswith(new + '/'):
            return old + relative[len(new):]
    return relative


def asset(root, relative):
    """Resolve shared Stage 1 resources while preserving custom dataset layouts."""
    root = Path(root).resolve()
    if root in (ROOT / 'data', SYNTHETIC_ROOT):
        relative = Path(relative).as_posix()
        if relative == 'raw' or relative.startswith('raw/'):
            return ROOT / 'data' / relative
        if relative == 'fonts' or relative.startswith('fonts/'):
            return ROOT / 'data/resources' / relative
        if relative == 'lexicon' or relative.startswith('lexicon/'):
            return ROOT / 'data/processed' / relative
        return SYNTHETIC_ROOT / relative
    return root / relative


def source_kind(record):
    """Classify sources semantically; accept original manifests without new fields."""
    if record.get('source_kind') in ('tabelog_menu', 'tabelog_interior', 'generated'):
        return record['source_kind']
    source = Path(record.get('source_root') or record.get('source_path', '')).as_posix().rstrip('/')
    for name, kind in [('tabelog_menus', 'tabelog_menu'), ('tabelog_interiors', 'tabelog_interior')]:
        if name in Path(source).parts:
            return kind
    return 'generated'
