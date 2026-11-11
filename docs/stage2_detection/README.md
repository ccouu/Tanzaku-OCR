# Stage 2 detection

```powershell
python -m src.detection.review.obb_review_ui
python -m src.detection.verify_stage2 --report outputs/reorganization/stage2_audit.json
```

The default review root is `data/annotations/menu_obb/`. `predictions/` contains proposals; `labels/` contains explicitly reviewed annotations. `provenance/` and `label_backups/` retain original evidence. The UI HTML and server live together in `src/detection/review/`.

Frozen snapshots and their `dataset.zip` files live under `data/datasets/detection/reviewed_v1/` and `reviewed_v2/`. Do not rerun preparation over either snapshot. New preparations require a fresh output directory. The current v2 snapshot has no independent test split.

Training runs are under `outputs/detection/training/colab_v1/` and `colab_v2/`; fresh-photo inference is under `outputs/detection/inference/tabelog_100/`. The selected checkpoint and provenance sidecar are in `checkpoints/`.

Reusable modules are in `src/detection/`; old `scripts.stage2_yolo` imports and commands remain supported. `src/detection/train_yolo_obb.py` and `colab_transfer.py` remain standalone remote/adapter scripts.

[Summary](stage2_summary.md) · [Review guide](stage2_review.md) · [Current training](stage2_training_v2.md) · [Historical v1 training](../history/stage2_training_v1.md)
