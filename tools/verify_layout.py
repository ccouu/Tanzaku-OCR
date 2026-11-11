"""Offline migration checks: commands, document links, frozen split invariants."""
import collections
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import digest, dump_json
from src.common.paths import resolve_path, source_kind


def main():
    commands = [
        'scripts.download_fonts', 'scripts.merge_real_dataset', 'scripts.prelabel_real_menus',
        'scripts.data.augment_menus', 'scripts.data.verify_stage1_dataset',
        'scripts.data.verify_menu_augmentation', 'scripts.data.verify_tabelog_collection',
        'scripts.data.tabelog_scraper', 'scripts.stage2_yolo.consolidate_images',
        'scripts.stage2_yolo.predict_initial_boxes', 'scripts.stage2_yolo.obb_review_ui',
        'scripts.stage2_yolo.import_reviewed', 'scripts.stage2_yolo.prepare_training',
        'scripts.stage2_yolo.train_yolo_obb', 'scripts.stage2_yolo.colab_pipeline',
        'scripts.stage2_yolo.colab_transfer', 'scripts.stage2_yolo.monitor_colab',
        'scripts.stage2_yolo.test_local_photos', 'scripts.stage2_yolo.verify_stage2',
    ]
    for command in commands:
        result = subprocess.run([sys.executable, '-m', command, '--help'], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError(command + '\n' + result.stderr)
    # The uploaded Colab worker must still work without the repository on sys.path.
    result = subprocess.run([sys.executable, str(ROOT/'src/detection/train_yolo_obb.py'), '--help'],
                            cwd=ROOT/'.cache', capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    broken = []
    for path in [ROOT/'README.md', ROOT/'WORKFLOW.md', *(ROOT/'docs').rglob('*.md')]:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            target = target.split('#')[0]
            if not target or '://' in target or target.startswith('mailto:'):
                continue
            destination = (path.parent/unquote(target)).resolve()
            if not destination.exists():
                broken.append([path.relative_to(ROOT).as_posix(), target])
    assert not broken, broken
    from src.detection.common import DEFAULT_ROOT
    from src.detection.review.obb_review_ui import ReviewStore
    store = ReviewStore(DEFAULT_ROOT)
    first = next(iter(store.records))
    loaded = store.load(first)
    assert loaded['annotation']['image_sha256'] == loaded['record']['sha256']
    snapshot_checks = {}
    for version in ('v1', 'v2'):
        snapshot = ROOT / ('data/datasets/detection/reviewed_' + version)
        manifest = json.loads((snapshot/'training_manifest.json').read_text())
        records = manifest['records']
        groups = collections.defaultdict(set)
        for record in records:
            groups[record['split']].update(record['groups'])
            kind = source_kind(record)
            assert record.get('source_kind', kind) == kind
            assert record['real'] == (kind != 'generated')
        for left in groups:
            for right in groups:
                if left != right:
                    assert not groups[left] & groups[right]
        archive = snapshot/'dataset.zip'
        archive_info = json.loads((snapshot/'archive_info.json').read_text())
        assert digest(archive) == archive_info['sha256']
        assert resolve_path(archive_info['path']).resolve() == archive
        assert manifest['classes'] == ['dish_horizontal', 'dish_vertical']
        snapshot_checks[version] = dict(images=len(records), splits=dict(collections.Counter(r['split'] for r in records)),
                                       archive_sha256=digest(archive), groups_disjoint=True)
    assert snapshot_checks['v2']['splits'] == {'train':862, 'val':351}
    model = ROOT/'checkpoints/yolo11n_menu_obb_best.pt'
    evaluation = ROOT/'outputs/detection/training/colab_v2/evaluation_report.json'
    report = json.loads(evaluation.read_text())
    assert digest(model) == report['checkpoint_sha256']
    metadata = dict(checkpoint=model.relative_to(ROOT).as_posix(), sha256=digest(model),
                    architecture='YOLO11n-OBB', classes=['dish_horizontal', 'dish_vertical'],
                    dataset='data/datasets/detection/reviewed_v2',
                    dataset_archive_sha256=snapshot_checks['v2']['archive_sha256'],
                    training_manifest_sha256=digest(ROOT/'data/datasets/detection/reviewed_v2/training_manifest.json'),
                    training_run='outputs/detection/training/colab_v2',
                    evaluation_report=evaluation.relative_to(ROOT).as_posix(),
                    metrics=report['metrics'], accuracy_gates_met=False, independent_test_accuracy_available=False)
    dump_json(model.with_suffix('.metadata.json'), metadata)
    result = dict(status='passed', legacy_cli_help_checks=len(commands), standalone_worker_help=True,
                  documentation_links='passed', existing_review_annotation_load=True, frozen_snapshots=snapshot_checks,
                  selected_checkpoint_sha256=digest(model))
    dump_json(ROOT/'outputs/reorganization/layout_checks.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
