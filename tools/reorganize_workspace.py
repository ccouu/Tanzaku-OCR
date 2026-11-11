"""One-time, non-overwriting migration; run from the workspace root.

Records every migrated file's size and SHA-256 before moving bundles. Rerun
with --verify to check relocated bytes without changing historical artifacts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
MAP = {
    'data/fonts': 'data/resources/fonts',
    'data/lexicon': 'data/processed/lexicon',
    'data/augmentation': 'data/processed/augmentation/run_1100',
    'data/augmentation_500': 'data/processed/augmentation/run_500',
    'data/stage2': 'data/annotations/menu_obb',
    'data/stage2_training': 'data/datasets/detection/reviewed_v1',
    'data/stage2_training_v2': 'data/datasets/detection/reviewed_v2',
    'data/stage2_training.zip': 'data/datasets/detection/reviewed_v1/dataset.zip',
    'data/stage2_training_v2.zip': 'data/datasets/detection/reviewed_v2/dataset.zip',
    'data/stage2_runtime': '.cache/ultralytics',
    'data/test-runs': '.cache/test-runs',
    'data/.pytest-stage1-tmp': '.cache/pytest-stage1',
    'outputs/stage2_colab': 'outputs/detection/training/colab_v1',
    'outputs/stage2_colab_v2': 'outputs/detection/training/colab_v2',
    'outputs/stage2_local_test_100': 'outputs/detection/inference/tabelog_100',
}
for name in ['ctc_lines', 'yolo_obb', 'vocab.json', 'stride_config.json',
             'character_frequency.json', 'dataset_manifest.json',
             'verification_report.json', 'menu_dataset_10k.tar.gz', 'DATA_LICENSES.md']:
    MAP['data/' + name] = 'data/processed/synthetic_base_v1/' + name


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Migration target escapes workspace')
    return path


def verify(report):
    for entry in report['files']:
        path = checked(entry['destination'])
        if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
            raise ValueError('Changed file: ' + entry['destination'])
    print('Verified', len(report['files']), 'relocated files byte-for-byte')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    inventory = ROOT / 'outputs/reorganization/inventory.json'
    if args.verify:
        return verify(json.loads(inventory.read_text(encoding='utf-8')))
    if args.resume:
        report = json.loads(inventory.read_text(encoding='utf-8'))
        skipped = []
        for old, new in report['mappings'].items():
            source, target = checked(old), checked(new)
            if not source.exists():
                continue
            if old == 'data/.pytest-stage1-tmp':
                skipped.append(dict(source=old, reason='Windows denies access to legacy test scratch directory'))
                continue
            if target.exists():
                raise ValueError('Both migration source and destination exist: ' + old)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
        report['skipped'] = skipped
        inventory.write_text(json.dumps(report, indent=2), encoding='utf-8')
        mapping = {old:new for old,new in report['mappings'].items() if old not in {s['source'] for s in skipped}}
        (ROOT / 'workspace_paths.json').write_text(json.dumps(mapping, indent=2) + '\n', encoding='utf-8')
        return verify(report)
    if inventory.exists():
        raise ValueError('Inventory already exists; use --verify, do not repeat migration')
    files = []
    for old, new in MAP.items():
        source, target = checked(old), checked(new)
        if not source.exists():
            continue
        if target.exists():
            raise ValueError('Destination already exists: ' + new)
        for path in sorted(source.rglob('*')) if source.is_dir() else [source]:
            if path.is_file():
                suffix = path.relative_to(source) if source.is_dir() else Path('.')
                destination = target / suffix if source.is_dir() else target
                files.append(dict(source=path.relative_to(ROOT).as_posix(),
                                  destination=destination.relative_to(ROOT).as_posix(),
                                  bytes=path.stat().st_size, sha256=sha(path)))
    inventory.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, mappings=MAP, files=files)
    inventory.write_text(json.dumps(report, indent=2), encoding='utf-8')
    mapping = ROOT / 'workspace_paths.json'
    if mapping.exists():
        raise ValueError('Path map already exists')
    mapping.write_text(json.dumps(MAP, indent=2) + '\n', encoding='utf-8')
    for old, new in MAP.items():
        source, target = checked(old), checked(new)
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
    verify(report)


if __name__ == '__main__':
    main()
