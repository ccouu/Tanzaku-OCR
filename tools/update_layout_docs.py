"""Update documentation links and publish the current workspace map."""
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / 'docs/history').exists():
    raise SystemExit('Documentation already migrated; refusing to repeat the one-time update')
mapping = json.loads((ROOT / 'workspace_paths.json').read_text())
docs = {
    'doc/stage1_summary.md': 'docs/stage1_data/stage1_summary.md',
    'doc/menu_augmentation.md': 'docs/stage1_data/menu_augmentation.md',
    'doc/tabelog_scraper_review.md': 'docs/stage1_data/tabelog_scraper_review.md',
    'doc/stage2_summary.md': 'docs/stage2_detection/stage2_summary.md',
    'doc/stage2_review.md': 'docs/stage2_detection/stage2_review.md',
    'doc/stage2_training.md': 'docs/history/stage2_training_v1.md',
    'doc/stage2_training_v2.md': 'docs/stage2_detection/stage2_training_v2.md',
    'doc/japanese_menu_ocr_implementation_plan.md': 'docs/history/original_implementation_plan.md',
}
(ROOT / 'docs/history').mkdir(exist_ok=True)
shutil.move(str(ROOT/'docs/stage2_detection/stage2_training.md'), str(ROOT/'docs/history/stage2_training_v1.md'))
shutil.move(str(ROOT/'docs/overview.md'), str(ROOT/'docs/history/original_implementation_plan.md'))
mapping.update(docs)
mapping['notebooks/colab_stage2_yolo_obb.ipynb'] = 'notebooks/stage2_detection/colab_stage2_yolo_obb.ipynb'


def remap(value):
    for old in sorted(mapping, key=len, reverse=True):
        if value == old or value.startswith(old + '/'):
            return mapping[old] + value[len(old):]
    return value


for old, new in docs.items():
    path = ROOT / new
    text = path.read_text(encoding='utf-8')
    def link(match):
        import os
        target = match[1]
        if '://' in target or target.startswith('#'):
            return match[0]
        bare, sep, anchor = target.partition('#')
        original = (ROOT / Path(old).parent / bare).resolve()
        if not original.is_relative_to(ROOT):
            return match[0]
        destination = ROOT / remap(original.relative_to(ROOT).as_posix())
        relative = Path(os.path.relpath(destination, path.parent)).as_posix()
        return '](' + relative + (sep + anchor if sep else '') + ')'
    text = re.sub(r'\]\(([^)]+)\)', link, text)
    # Update code examples without rewriting the frozen reports they describe.
    if '/history/' not in new:
        path_pattern = re.compile(r'(?<![\w/])(' + '|'.join(re.escape(key) for key in sorted(mapping, key=len, reverse=True)) + r')(?=$|[/\s`\)\]>,.;])')
        text = path_pattern.sub(lambda match: mapping[match[0]], text)
    for before in sorted(mapping, key=len, reverse=True):
        text = text.replace('`' + before + '`', '`' + mapping[before] + '`')
        text = text.replace('--output ' + before, '--output ' + mapping[before])
        text = text.replace('--dataset ' + before, '--dataset ' + mapping[before])
        text = text.replace('--data-root ' + before, '--data-root ' + mapping[before])
        text = text.replace('--output-dir ' + before, '--output-dir ' + mapping[before])
    if '/history/' in new:
        text = '> Historical reference. Current commands and status: [project overview](../overview.md).\n\n' + text
    else:
        text = '> Paths updated for the reorganized workspace. Historical artifact contents and provenance paths are retained.\n\n' + text
    path.write_text(text, encoding='utf-8')

notebook = ROOT / mapping['notebooks/colab_stage2_yolo_obb.ipynb']
text = notebook.read_text(encoding='utf-8').replace('data/stage2_training_v2.zip', 'data/datasets/detection/reviewed_v2/dataset.zip')
notebook.write_text(text, encoding='utf-8')

(ROOT/'README.md').write_text('''# Japanese menu OCR and translation

A staged pipeline for handwritten Japanese menu detection, recognition, and mobile translation.

| Stage | Current state |
| --- | --- |
| 1 — Data preparation | Lexicon, synthetic datasets, photo collection and augmentation delivered; reviewed real text crops and hybrid assembly pending. |
| 2 — Text detection | Reviewed annotations and YOLO11n-OBB training delivered; accuracy gates unmet and fresh-photo inspection pending. |
| 3 — Recognition | Not started. |
| 4 — Mobile export | Not started. |

Use Python 3.12. Stage 1 dependencies are in `requirements-stage1.txt`; Stage 2 local proposal dependencies are in `requirements-stage2.txt`. Colab training pins its own Ultralytics version; see the training guide. Existing `python -m scripts...` commands remain supported.

```powershell
python -m pytest -q
python -m scripts.stage2_yolo.obb_review_ui
```

Open the review editor at http://localhost:7860. No training or remote session starts automatically.

| Location | Purpose |
| --- | --- |
| `src/common/`, `src/data/`, `src/detection/` | Shared utilities, data preparation and reusable detector workflows |
| `scripts/` | Command entrypoints and compatibility wrappers |
| `data/raw/`, `data/resources/` | Source assets, provenance, retained API ledger and fonts |
| `data/processed/` | Shared lexicon, synthetic base bundle and named augmentation runs |
| `data/annotations/menu_obb/` | Review images, proposals, reviewed labels, provenance and backups |
| `data/datasets/detection/` | Frozen reviewed v1/v2 training snapshots and archives |
| `outputs/detection/` | Training runs, recovery files, evaluation and inference outputs |
| `checkpoints/` | Selected detector weights and model provenance |
| `.cache/` | Local test scratch and runtime settings |

[Workflow](WORKFLOW.md) · [Overview and handoffs](docs/overview.md) · [Stage 1 guide](docs/stage1_data/README.md) · [Stage 2 guide](docs/stage2_detection/README.md) · [Migration notes](docs/reorganization.md)

Generated menu prompt text and detector proposals are unverified. Reviewed detector boxes do not establish OCR transcription ground truth. Source notices and licensing metadata remain with their assets.
''', encoding='utf-8')

(ROOT/'WORKFLOW.md').write_text('''# Project workflow

1. **Collect resources and sources:** retain originals, licenses, checksums, photo identities and the existing generation ledger under `data/raw/` and `data/resources/`.
2. **Prepare Stage 1 data:** build the shared lexicon and synthetic base; keep each augmentation run separate under `data/processed/`. Audit sample units and related-source splits before combining data.
3. **Review Stage 2 pages:** consolidate into `data/annotations/menu_obb/`, generate proposals and explicitly review/save labels. Predictions remain separate from reviewed labels. Reference crops never become detector pages.
4. **Freeze a detector dataset:** prepare a fresh directory under `data/datasets/detection/`. Preserve source-connected groups and reviewed-only admission. Keep the snapshot and its `dataset.zip` immutable.
5. **Train and evaluate:** choose a fresh output run under `outputs/detection/training/`; retain logs, recovery checkpoints and evaluation together. Select weights into `checkpoints/` with dataset/run provenance.
6. **Inspect fresh inference:** keep overlays and predictions under `outputs/detection/inference/`. The current 100-photo collection is unlabelled; it has no measured accuracy score.
7. **Stage 3 handoff:** reviewed detector geometry is available, but reviewed transcriptions and recognition training remain pending. Stage 4 follows validated recognition.

The current selected detector uses the reviewed v2 dataset with 862 training and 351 validation images; there is no independent test split. Validation accuracy gates remain unmet. Reorganization does not authorize promotion of proposals or a new training run.

See [Stage 2 training](docs/stage2_detection/stage2_training_v2.md) for explicit Colab commands and recovery. Agent routing is documented separately in [orchestration](docs/orchestration.md) and `AGENTS.md`.
''', encoding='utf-8')

(ROOT/'docs/overview.md').write_text('''# Pipeline overview

Stage 1 prepares resources and data. Stage 2 detects horizontal and vertical text boundaries. Stage 3 will recognize text; Stage 4 will export the validated pipeline for mobile use.

Stage 1 delivered 9,000 synthetic recognition lines and 1,000 three-class synthetic detector pages, alongside generated menus, real photographs and separate augmentation runs. Real recognition transcriptions and the hybrid dataset remain pending. [Stage 1 evidence](stage1_data/stage1_summary.md).

Stage 2 delivered proposals, explicit review tools, 1,213 reviewed pages, frozen datasets, and a trained YOLO11n-OBB checkpoint. The current training snapshot has 862 train / 351 validation images with source-connected splits and two classes. Validation mAP50 is 0.8427 and mAP50–95 is 0.6521; the planned gates remain unmet. Fresh-photo predictions await inspection. [Stage 2 evidence](stage2_detection/stage2_summary.md).

The three-class synthetic/proposal format includes `price_tag`; the current detector trains `dish_horizontal=0` and `dish_vertical=1`. Existing price conversions and historical annotations are retained. Detector review is independent of verified text transcription.

Use the [workflow](../WORKFLOW.md), [Stage 1 guide](stage1_data/README.md), and [Stage 2 guide](stage2_detection/README.md) for current entrypoints. The [original implementation plan](history/original_implementation_plan.md) is historical context.
''', encoding='utf-8')

(ROOT/'docs/stage1_data/README.md').write_text('''# Stage 1 data preparation

The synthetic base is `data/processed/synthetic_base_v1/`; shared resources are `data/resources/fonts/`, `data/raw/` and `data/processed/lexicon/`. Explicit custom `--data-root` directories keep the original self-contained layout.

Build commands below change derived data; they are not needed to inspect the existing datasets:

```powershell
python -m scripts.download_fonts --jmdict
python -m src.data.lexicon_builder
python -m src.data.synthetic_renderer --workers 1
```

Offline checks can write fresh reports without replacing the historical evidence:

```powershell
python -m scripts.data.verify_stage1_dataset --report outputs/reorganization/stage1_audit.json
python -m scripts.data.verify_menu_augmentation --data-root data/processed/augmentation/run_500 --report outputs/reorganization/augmentation_500_audit.json
```

The 1,100-pair and 500-pair augmentation runs remain separate. Do not combine correlated crops as independent examples. Retain the existing generation ledger and source licenses.

[Summary and limitations](stage1_summary.md) · [Augmentation](menu_augmentation.md) · [Photo collection](tabelog_scraper_review.md)
''', encoding='utf-8')

(ROOT/'docs/stage2_detection/README.md').write_text('''# Stage 2 detection

```powershell
python -m scripts.stage2_yolo.obb_review_ui
python -m scripts.stage2_yolo.verify_stage2 --report outputs/reorganization/stage2_audit.json
```

The default review root is `data/annotations/menu_obb/`. `predictions/` contains proposals; `labels/` contains explicitly reviewed annotations. `provenance/` and `label_backups/` retain original evidence. The UI HTML and server live together in `src/detection/review/`.

Frozen snapshots and their `dataset.zip` files live under `data/datasets/detection/reviewed_v1/` and `reviewed_v2/`. Do not rerun preparation over either snapshot. New preparations require a fresh output directory. The current v2 snapshot has no independent test split.

Training runs are under `outputs/detection/training/colab_v1/` and `colab_v2/`; fresh-photo inference is under `outputs/detection/inference/tabelog_100/`. The selected checkpoint and provenance sidecar are in `checkpoints/`.

Reusable modules are in `src/detection/`; old `scripts.stage2_yolo` imports and commands remain supported. `src/detection/train_yolo_obb.py` and `colab_transfer.py` remain standalone remote/adapter scripts.

[Summary](stage2_summary.md) · [Review guide](stage2_review.md) · [Current training](stage2_training_v2.md) · [Historical v1 training](../history/stage2_training_v1.md)
''', encoding='utf-8')

(ROOT/'docs/reorganization.md').write_text('''# Workspace reorganization

Code is grouped by capability, data by lifecycle, and guides by stage. `workspace_paths.json` is the explicit old-to-new artifact map. Python path resolution accepts these legacy paths at supported entrypoints. No symlinks are required.

Historical manifests, provenance snapshots, dataset archives, labels and image bytes are retained unchanged. Old paths inside them are historical evidence, resolved through the map. New consolidation records include `source_kind`; legacy records are classified through the same compatibility reader. Frozen snapshots keep their existing records and class IDs.

`outputs/reorganization/inventory.json` records each moved file's original path, destination, byte count and SHA-256. Run `python tools/reorganize_workspace.py --verify` to recheck migrated files. Audit CLIs support `--report` to preserve historical reports. New dataset archives are stored inside their snapshot as `dataset.zip`.

The inaccessible legacy `data/.pytest-stage1-tmp/` scratch directory remains in place because Windows denied access. All new test scratch is under `.cache/test-runs/`. `.stage2_packages/` is retained for the existing dependency bootstrap.

No datasets were rebuilt, no model was retrained, and no proposals were promoted to reviewed labels. No Git repository was initialized. Existing license notices and the generation ledger remain with the assets.

The one-time migration scripts refuse overwrites. `refactor_layout.py` and `update_layout_docs.py` describe the completed source migration and must not be rerun on the reorganized tree.
''', encoding='utf-8')

print('Updated current guides and historical links')
