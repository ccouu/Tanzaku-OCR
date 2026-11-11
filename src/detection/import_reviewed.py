"""Explicitly approve corrected X-AnyLabeling JSON files into reviewed labels."""
import argparse
from pathlib import Path
from .common import DEFAULT_ROOT, read_json
from .review.obb_review_ui import ReviewStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--input', type=Path, required=True,
                        help='Directory containing only JSON files you have manually reviewed')
    parser.add_argument('--confirm-empty', action='store_true',
                        help='Explicitly approve empty annotations as negatives')
    args = parser.parse_args()
    store = ReviewStore(args.root)
    paths = sorted(args.input.glob('*.json'))
    if not paths:
        parser.error('No reviewed JSON files found')
    pending = []
    for path in paths:
        ident = path.stem
        current = store.load(ident)
        doc = read_json(path)
        record = current['record']
        if doc.get('imageWidth') != record['width'] or doc.get('imageHeight') != record['height']:
            parser.error(f'Image dimensions mismatch: {path}')
        if doc.get('image_sha256') != record['sha256'] or doc.get('source_id') != ident:
            parser.error(f'Provenance mismatch: {path}; retain custom JSON fields when editing')
        from .common import validate_shapes
        validate_shapes(doc['shapes'], record['width'], record['height'])
        if record['role'] != 'page':
            parser.error(f'Cannot approve a reference/crop as a page: {path}')
        if not doc['shapes'] and not args.confirm_empty:
            parser.error(f'Empty label requires --confirm-empty: {path}')
        pending.append((ident, current, doc))
    for ident, current, doc in pending:
        store.save(ident, dict(revision=current['revision'], shapes=doc['shapes'], confirm_empty=args.confirm_empty))
    print(f'Imported {len(pending)} explicitly reviewed annotations. Training has not started.')


if __name__ == '__main__':
    main()
