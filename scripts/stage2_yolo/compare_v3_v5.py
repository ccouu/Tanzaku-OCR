"""Re-evaluate preserved V3/V5 weights on the identical frozen V5 holdout."""
from datetime import datetime, timezone
from pathlib import Path

from src.common import digest, dump_json
from src.detection.common import read_json
from scripts.stage2_yolo.finetune_banana_latest import runtime, metrics, ROOT
from scripts.stage2_yolo.finetune_all130 import DATA, START, WORK as TRAIN


def main():
    out = ROOT / 'outputs/detection/evaluation' / ('v3_vs_v5_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    manifest = read_json(DATA / 'training_manifest.json')
    records = manifest['records']
    val = [r for r in records if r['split'] == 'val']
    assert len(val) == 13
    assert not {r['image_sha256'] for r in val} & {r['image_sha256'] for r in records if r['split'] == 'train'}
    for r in records:
        assert digest(DATA / 'images' / r['split'] / (r['id'] + '.jpg')) == r['image_sha256']
        assert digest(DATA / 'labels' / r['split'] / (r['id'] + '.txt')) == r['label_sha256']
    checkpoints = {'v3': START, 'v5': TRAIN / 'yolo11n_menu_obb_banana130_best.pt'}
    hashes = {k: digest(p) for k, p in checkpoints.items()}
    torch, YOLO = runtime()
    torch.set_num_threads(4)
    settings = dict(imgsz=640, batch=8, device=0, workers=0, half=False,
                    conf=0.001, iou=0.7, max_det=300, rect=True, split='val',
                    augment=False, seed=0, deterministic=True, plots=False,
                    verbose=False, save_json=False, save_txt=False)
    report = dict(validation_ids=[r['id'] for r in val], validation_boxes=sum(r['boxes'] for r in val),
                  manifest_sha256=digest(DATA / 'training_manifest.json'), settings=settings,
                  checkpoints={k: dict(path=str(p), sha256=hashes[k]) for k, p in checkpoints.items()},
                  results={}, note='Same V5 validation labels. Checkpoint-selection validation, not an independent test. P/R are evaluator best-F1 operating-point metrics, not a shared fixed confidence threshold.')
    for name, path in checkpoints.items():
        model = YOLO(str(path))
        report['results'][name] = {}
        for group in ('combined', 'first50', 'second50', 'new30'):
            yaml = DATA / ('dataset.yaml' if group == 'combined' else f'dataset_{group}.yaml')
            result = model.val(data=str(yaml), project=str(out / 'runs'), name=f'{name}_{group}', **settings)
            report['results'][name][group] = metrics(result)
            dump_json(out / 'report.json', report)
        assert digest(path) == hashes[name]
    report['state'] = 'completed'
    dump_json(out / 'report.json', report)
    lines = ['# V3 versus V5 on V5 validation', '',
             '13 held-out menus, latest frozen V5 labels; 166 annotated boxes. Both weights re-evaluated with identical FP32 settings.', '',
             '| Subset | V3 mAP50–95 | V5 mAP50–95 | Difference (pp) |',
             '|---|---:|---:|---:|']
    for group in ('combined', 'first50', 'second50', 'new30'):
        a = report['results']['v3'][group]['metrics/mAP50-95(B)'] * 100
        b = report['results']['v5'][group]['metrics/mAP50-95(B)'] * 100
        lines.append(f'| {group} | {a:.4f}% | {b:.4f}% | {b-a:+.4f} |')
    lines += ['', 'P/R in report.json use each model\'s evaluator-selected best-F1 confidence, not a common deployment threshold.',
              'This validation set was used for checkpoint selection. Small aggregate differences do not demonstrate statistical superiority.',
              'All source weights preserved; dataset image/label hashes verified.']
    (out / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('REPORT:', out / 'report.json', flush=True)
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
