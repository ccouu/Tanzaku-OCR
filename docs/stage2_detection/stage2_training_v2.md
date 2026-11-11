> Paths updated for the reorganized workspace. Historical artifact contents and provenance paths are retained.

# Colab restart with source-specific training/validation ratios

The new snapshot is `data/datasets/detection/reviewed_v2/`. It includes only the 1,213
explicitly reviewed page images. Unmarked pages, detector-only proposals,
reference images, and the separate Stage 1 synthetic dataset are excluded.
The two orientation classes and the previous price-box conversions are retained.

| Source | Training | Validation | Total |
| --- | ---: | ---: | ---: |
| Tabelog photos, menus and interiors combined | 287 | 287 | 574 |
| Generated/augmented pages | 575 | 64 | 639 |
| Total | 862 | 351 | 1,213 |

The generated allocation rounds 10% of 639 to 64 validation images. There is
no separate test split in this run. Real validation images are also evaluated
separately from the mixed validation set. Validation is used for model selection,
so its metrics are not represented as independent test-set results.

Connected source-content, restaurant, reused-background, and duplicate-hash
groups remain disjoint. A binary integer allocation solves the combined
Tabelog 50/50 and generated 90/10 quotas without breaking any group. This leaves
146 menu photos and 141 interiors in training, and 228 menus and 59 interiors
in validation. The 50/50 ratio applies to all Tabelog photographs together;
individual Tabelog subcategories cannot both be halved exactly under these
connected-group constraints. Full per-source counts and IDs are recorded in
`training_manifest.json`.

Archive SHA-256:
`5153519ba9f1d15ae89e48467876541fd38350c390f1ffb4725060cd6fe80b4f`.

## Execution and recovery

The named Colab session is `japanese-menu-yolo-v2`, with a T4 GPU and
`ultralytics==8.4.206`. Training uses the same YOLO11n-OBB configuration:
100 epochs, 640-pixel images, batch 16, AdamW, initial learning rate 0.001,
cosine decay, rotation ±15°, perspective 0.0005, mosaic 0.5, and no mirroring.

Training now runs synchronously in the Jupyter kernel through `colab exec`
with a four-hour execution timeout. The local controller monitors through the
authenticated file API while the kernel remains busy doing actual training.
This replaces the prior detached training process.

At epoch 1, every five epochs, and the final epoch, the worker publishes complete,
immutable best/last checkpoint copies with hashes. The monitor downloads and
verifies these during training under `outputs/detection/training/colab_v2/recovery/`.
`latest_last.pt` retains optimizer state for recovery; `checkpoint_info.json`
records the saved epoch and source hashes. The monitor records explicit session
loss or repeated transport failure rather than retrying indefinitely. Remote
runtime lifetime remains controlled by Colab; periodic local copies limit lost
progress if it ends unexpectedly.

Live status is `outputs/detection/training/colab_v2/remote_status.json`. Final checkpoints,
evaluation reports, crop audit samples, logs, and the result archive are
downloaded to the same output directory. The final best checkpoint is also
published at `checkpoints/yolo11n_menu_obb_best.pt`. The runtime is released
after successful result retrieval.

The completed v2 snapshot and output directory are retained above. For a new
run, use fresh snapshot, output, and session names, for example:

```powershell
python -m src.detection.prepare_training --split-policy source_ratios --output data/datasets/detection/reviewed_v3
colab new -s japanese-menu-yolo-v3 --gpu T4
colab install -s japanese-menu-yolo-v3 ultralytics==8.4.206
python -m src.detection.colab_pipeline --dataset data/datasets/detection/reviewed_v3 --output outputs/detection/training/colab_v3 --session japanese-menu-yolo-v3
```

If that new worker is still running and only its monitor was interrupted,
reattach without restarting training:

```powershell
python -m src.detection.monitor_colab --session japanese-menu-yolo-v3 --output outputs/detection/training/colab_v3
```

Run only one monitor against a live worker. Do not run preparation over an
existing snapshot. Prior snapshots and failed-run status records are retained.
