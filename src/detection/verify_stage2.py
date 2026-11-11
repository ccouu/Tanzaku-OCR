"""Audit image copies, original metadata, prediction geometry and reviewed labels."""
import argparse
from collections import Counter
from pathlib import Path
from src.common import digest, dump_json
from src.common.paths import resolve_path
from .common import DEFAULT_ROOT, read_json, validate_shapes, yolo_text


def verify(root=DEFAULT_ROOT, report_path=None):
    root = resolve_path(root)
    records = read_json(root / 'manifest.json')['records']
    errors = []
    counts = Counter()
    for record in records:
        try:
            ident = record['id']
            assert digest(root/'images'/record['image']) == record['sha256'], 'Consolidated hash mismatch'
            assert digest(record['source_path']) == record['sha256'], 'Original hash mismatch'
            assert read_json(root/'provenance'/(ident+'.json')) == record, 'Provenance snapshot mismatch'
            if record['source_info_path']:
                assert digest(record['source_info_path']) == record['source_info_sha256'], 'Original metadata changed'
                assert read_json(record['source_info_path']) == record['source_info'], 'Metadata snapshot mismatch'
            for folder in ['predictions', 'labels']:
                file = root/folder/(ident+'.json')
                if not file.exists():
                    if folder == 'predictions':
                        errors.append(dict(id=ident, error='Missing prediction'))
                    continue
                doc = read_json(file)
                assert doc['source_id'] == ident and doc['image_sha256'] == record['sha256'], 'Annotation identity mismatch'
                assert (file.parent/doc['imagePath']).resolve() == (root/'images'/record['image']).resolve(), 'Broken annotation image link'
                validate_shapes(doc['shapes'], record['width'], record['height'])
                assert file.with_suffix('.txt').read_text() == yolo_text(doc['shapes'],record['width'],record['height']), 'YOLO/JSON mismatch'
                assert doc['flags']['reviewed'] == (folder == 'labels'), 'Review state mismatch'
                if folder == 'labels':
                    assert record['role'] == 'page', 'Reference/crop promoted to training page'
                counts[folder] += 1
                counts[folder+'_boxes'] += len(doc['shapes'])
            counts['images'] += 1
            counts['role_'+record['role']] += 1
        except (AssertionError, OSError, ValueError, KeyError) as exc:
            errors.append(dict(id=record['id'], error=str(exc)))
    report = dict(status='passed' if not errors else 'failed', counts=dict(counts), errors=errors,
                  training_started=False, manual_review_pending=counts['labels'] < counts['role_page'])
    dump_json(report_path or root/'verification_report.json',report)
    print(report)
    if errors:
        raise ValueError('Stage 2 audit failed')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    verify(args.root, args.report)


if __name__ == '__main__':
    main()
