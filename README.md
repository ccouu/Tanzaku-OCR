# Japanese menu OCR and translation

A staged pipeline for handwritten Japanese menu detection, recognition, and mobile translation.

| Stage | Current state |
| --- | --- |
| 1 — Data preparation | Lexicon, synthetic datasets, photo collection and augmentation delivered; reviewed real text crops and hybrid assembly pending. |
| 2 — Text detection | Reviewed annotations and YOLO11n-OBB training delivered; accuracy gates unmet and fresh-photo inspection pending. |
| 3 — Recognition | V27 is the active implementation:4.48% seen-font /5.27% held-out-font CER. GenAI-menu retention gate failed; reviewed real-menu validation pending. Older experiments are archived. |
| 4 — Mobile export | Not started. |

Use Python 3.12. Stage 1 dependencies are in `requirements-stage1.txt`; Stage 2 local proposal dependencies are in `requirements-stage2.txt`. Colab training pins its own Ultralytics version; see the training guide. Existing `python -m scripts...` commands remain supported.

```powershell
python -m pytest -q
python -m src.detection.review.obb_review_ui
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
| `src/recognition/`, `docs/stage3_recognition/` | Active V27 recognition implementation and guide |
| `archive/stage3_legacy_20261111/` | Superseded Stage3 experiments, code, docs and data; reversible move manifest |
| `checkpoints/` | Selected detector weights and model provenance |
| `.cache/` | Local test scratch and runtime settings |

[Workflow](WORKFLOW.md) · [Overview and handoffs](docs/overview.md) · [Stage 1 guide](docs/stage1_data/README.md) · [Stage 2 guide](docs/stage2_detection/README.md) · [Stage 3 guide](docs/stage3_recognition/README.md) · [Migration notes](docs/reorganization.md)

Generated menu prompt text and detector proposals are unverified. Reviewed detector boxes do not establish OCR transcription ground truth. Source notices and licensing metadata remain with their assets.

## Repository contents

This repository contains source, notebooks, selected experiment scripts and
written reports. Datasets, generated images, dictionary databases, curated
training-label files, checkpoints, logs, local environments and the historical
archive remain local and are excluded from Git. Running training or data-backed
evaluations requires restoring the separately managed assets at the documented
paths. Image galleries linked from reports also require those generated assets.

The reusable recognition implementation remains V27. Recent V29–V31 pilot
scripts are tracked under `outputs/recognition/`; generated output from those
scripts remains ignored. See the [V31 curated-term analysis](outputs/recognition/pilots/v31_curated_short_2000/analysis.md)
and [single-character decoder analysis](outputs/recognition/evaluation/v31_onechar_blank_decoders/analysis.md).
The decoder fallback is an evaluated candidate and has not been installed in
the production recognition path.
