> Historical reference; filesystem paths reflect the current layout. Current commands and status: [project overview](../overview.md).

# Reviewed-only YOLO11n-OBB training

This training snapshot follows the November 8 instruction: use only saved,
explicitly reviewed images, and replace the price class with horizontal or
vertical orientation. Unreviewed images and the separate Stage 1 synthetic
detector dataset are excluded.

## Dataset snapshot

`data/datasets/detection/reviewed_v1/` contains 1,213 reviewed images: 639 augmentation pages,
374 real menus, and 200 restaurant interiors. It excludes all other 4,503
consolidated records, including 372 unreviewed page images. Empty reviewed
images are retained only when their saved annotation explicitly confirms
them as negatives.

All 2,940 price boxes were reclassified using mean opposite-edge lengths:
width is the mean of edges 0–1 and 2–3; height is the mean of edges 1–2 and 3–0.
Height greater than width becomes `dish_vertical`; otherwise (including a tie)
it becomes `dish_horizontal`. This measures the marked rectangle's local axes
rather than the enclosing axis-aligned image bounds. Existing horizontal and
vertical assignments are retained.

Original versions of the reviewed JSON/text labels changed by conversion are backed up under
`data/annotations/menu_obb/label_backups/before_price_relabel_20261108T002534Z/`.
Updated labels remain under `data/annotations/menu_obb/labels/`, with conversion metadata.
Original detector proposals are unchanged.

The training classes are now:

| ID | Class |
| --- | --- |
| 0 | dish_horizontal |
| 1 | dish_vertical |

Source metadata is preserved per image. Connected components link source
menu IDs/content groups, restaurant IDs, reused backgrounds, and matching
pixel hashes. Entire components stay in one split. The deterministic split
has 355 independent components, the largest containing 212 images; source-key
intersection checks pass.

| Split | Images | Horizontal boxes | Vertical boxes |
| --- | ---: | ---: | ---: |
| Train | 971 | 9,788 | 4,092 |
| Validation | 121 | 2,145 | 890 |
| Test | 121 | 2,931 | 811 |

The test split includes 101 real photographs, evaluated separately as well as
within the full test split. Pixels are exported as full-resolution RGB JPEGs
(quality 95, no chroma subsampling); transparency is composited on white.
Original and exported pixel-file hashes are both recorded.

`training_manifest.json` records every selected ID, source path, approval hash,
split, source group, export hash, label hash, and individual price conversion.
`data/datasets/detection/reviewed_v1/dataset.zip` contains only that selected snapshot; its SHA-256 is
`0e73775c707f8fe0fa5209a2193b2d727df5eb4dd9efbef6fbbc80a64cd6b6c5`.

## Training and Colab execution

The installed official Google Colab CLI is invoked as `colab`. The named
session is `japanese-menu-yolo-stage2`, using a Tesla T4 and
`ultralytics==8.4.206`. The local controller is
`src/detection/colab_pipeline.py`; the standalone GPU worker is
`src/detection/train_yolo_obb.py`.

The CLI's original single-request upload failed in this environment. A
workspace-local adapter uses the installed CLI's authenticated session, standard
Colab proxy headers, and bounded 4 MiB Contents API chunks. The full archive
checksum is validated before extraction. TLS verification and normal CLI
authentication are retained. CLI errors are redacted before being surfaced.

Training uses `yolo11n-obb.pt`, 100 epochs, 640-pixel input, batch 16, AdamW,
initial learning rate 0.001, cosine decay, ±15° rotation, perspective 0.0005,
and mosaic 0.5. Horizontal/vertical flips are disabled to avoid mirroring text.
Training uses only the two orientation classes. The prior plan's three-class
price detection/association requirement is superseded by the user's instruction.

The worker verifies the reviewed flags and all uploaded image/label hashes,
trains on GPU, evaluates the best checkpoint on validation, full test, and real
test, and bundles diagnostics. The controller polls progress, retrieves results,
checks the best checkpoint hash, exports a Colab notebook log, and stops the
session after successful retrieval to release compute resources. Monitoring uses
the authenticated Contents API because execution-websocket polling was
intermittent; the independent GPU worker continues through monitor reconnects.

Live status is recorded in `outputs/detection/training/colab_v1/remote_status.json`. Completed
artifacts go to that directory, including `yolo11n_menu_obb_best.pt`,
`training_results.zip`, `evaluation_report.json`, `training.log`, and
`run_config.json`. Actual evaluation results, rather than training loss, determine
whether the real-holdout mAP50 ≥0.90 and mAP50–95 ≥0.75 targets pass.

## Reproduction

Preparation changes reviewed price labels after backing them up; always use a
fresh training output directory for a new snapshot:

```powershell
python -m src.detection.prepare_training --output data/datasets/detection/reviewed_v1_reproduction
colab new -s japanese-menu-yolo-v1-reproduction --gpu T4
colab install -s japanese-menu-yolo-v1-reproduction ultralytics==8.4.206
python -m src.detection.colab_pipeline --dataset data/datasets/detection/reviewed_v1_reproduction --output outputs/detection/training/colab_v1_reproduction --session japanese-menu-yolo-v1-reproduction
```

Do not launch a second controller against an already running training worker.
For an interrupted monitor, reattach without uploading or restarting training:

```powershell
python -m src.detection.monitor_colab --session japanese-menu-yolo-v1-reproduction --output outputs/detection/training/colab_v1_reproduction
```

Only one monitor should collect the results. The worker refuses to overwrite
an existing dataset/run. The monitor also publishes the verified checkpoint at
`checkpoints/yolo11n_menu_obb_best.pt` after download.
This pipeline does not start Stage 3 recognizer training.
