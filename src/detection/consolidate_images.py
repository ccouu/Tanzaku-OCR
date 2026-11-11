"""Copy every image into a flat collection, preserving originals and provenance."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import shutil

from PIL import Image
from src.common.paths import ROOT, resolve_path, source_kind, historical_path
from src.common import dump_json, digest
from .common import DEFAULT_ROOT, read_json

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
SOURCES = [ROOT/p for p in ['data/processed/augmentation/run_500', 'data/raw/tabelog_menus', 'data/raw/tabelog_interiors']]


def role(path):
    parts = path.parts
    if 'rejected' in parts or 'diagnostics' in parts:
        return 'excluded_reference'
    if any(x in path.stem for x in ['preview', 'contact_sheet', 'overview']):
        return 'review_reference'
    if 'ctc_lines' in parts:
        return 'line_crop'
    return 'page'


def sidecar(path):
    direct = path.with_suffix('.json')
    if direct.is_file():
        return direct
    parts = list(path.parts)
    if 'images' in parts:
        parts[parts.index('images')] = 'labels'
        alternate = Path(*parts).with_suffix('.json')
        if alternate.is_file():
            return alternate
    return None


def consolidate(output=DEFAULT_ROOT, sources=SOURCES):
    output = resolve_path(output)
    records = []
    for source in sources:
        source = resolve_path(source)
        if not source.is_dir():
            raise ValueError(f'Missing source: {source}')
        for path in sorted(source.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            rel = path.as_posix()
            logical = historical_path(path)
            ident = Path(historical_path(source)).name + '__' + hashlib.sha256(logical.encode()).hexdigest()[:16]
            name = ident + path.suffix.lower()
            target = output / 'images' / name
            sha = digest(path)
            if target.exists() and digest(target) != sha:
                raise ValueError(f'Refusing to overwrite changed image: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(path, target)
            if digest(target) != sha:
                raise ValueError(f'Copy verification failed: {path}')
            with Image.open(target) as image:
                width, height = image.size
                image.verify()
            info = sidecar(path)
            metadata = read_json(info) if info else None
            record = dict(id=ident, image=name, source_path=rel, source_root=source.as_posix(),
                          sha256=sha, width=width, height=height, role=role(path),
                          source_info_path=info.as_posix() if info else None,
                          source_info_sha256=digest(info) if info else None,
                          source_info=metadata)
            record['source_kind'] = source_kind(record)
            dump_json(output / 'provenance' / (ident + '.json'), record)
            records.append(record)
    manifest = dict(schema_version=1, copy_policy='byte-identical copies; original images and metadata retained',
                    records=records, counts=dict(Counter(r['role'] for r in records)))
    dump_json(output / 'manifest.json', manifest)
    print({'images': len(records), 'roles': manifest['counts']}, flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    consolidate(args.output)


if __name__ == '__main__':
    main()
