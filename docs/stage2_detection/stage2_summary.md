> Paths updated for the reorganized workspace. Historical artifact contents and provenance paths are retained.

# Stage 2 Summary — Japanese Menu Text Boundary Detection

Updated: 8 November 2026.

Stage 2 now includes consolidated images, automatic boundary proposals, manual
review tools, reviewed-only detector datasets, completed Colab training,
validation, and local inference on 100 fresh Tabelog photos. The best checkpoint
is saved locally. **The planned accuracy gates have not been met**, and manual
inspection of the fresh-photo predictions remains pending. Stage 3 has not started.

## Image consolidation and traceability

Copied all images from `data/processed/augmentation/run_500/`, `data/raw/tabelog_menus/`, and
`data/raw/tabelog_interiors/` into [data/annotations/menu_obb/images/](../../data/annotations/menu_obb/images).
Original files and metadata remain intact; copies were checksum-verified.

| Image role | Count |
| --- | ---: |
| Pages eligible for review | 1,585 |
| Existing CTC line crops, read-only references | 4,050 |
| Previews/contact sheets | 37 |
| Rejected/diagnostic references | 44 |
| Total | 5,716 |

The page collection contains 1,000 augmented menu/composite images, 385 active
Tabelog menu photos, and 200 interiors. The menu count differed from the earlier
Stage 1 summary's 400; no missing photos were invented or restored automatically.

[manifest.json](../../data/annotations/menu_obb/manifest.json) and
[provenance/](../../data/annotations/menu_obb/provenance) preserve stable IDs, original paths,
image/metadata hashes, dimensions, roles, and metadata snapshots. Source URLs,
restaurant IDs, source/background groups, augmentation transforms, and text
verification state remain traceable where supplied by Stage 1.

## Stage 2.1 — Initial box proposals

Implemented [predict_initial_boxes.py](../../src/detection/predict_initial_boxes.py)
using local PP-OCRv4 DBNet ONNX detection through RapidOCR 1.4.4. All 5,716
images were processed, producing **34,454 candidate boxes**, including 29,786
on pages, with zero inference failures. No OCR transcription or LLM calls were used.

Proposals were exported as YOLO-OBB text and X-AnyLabeling JSON under
[predictions/](../../data/annotations/menu_obb/predictions). Initial heuristic classes were
horizontal, vertical, and price. Proposals were unreviewed and did not become
training labels automatically. See [prediction_report.json](../../data/annotations/menu_obb/prediction_report.json).

## Stage 2.2 — Manual review and two-class labels

Implemented the [local canvas editor](../../src/detection/review/obb_review_ui.py)
at `http://localhost:7860`: draw, move, resize, rotate, change class, delete,
undo, and save-and-next. It validates geometry and image identity, detects stale
edits, requires confirmation for empty negatives, and separates reviewed labels
from predictions. X-AnyLabeling import is also implemented.

The user saved **1,213 reviewed page annotations**: 639 generated/augmented
pages, 374 real menus, and 200 interiors. All 372 unmarked pages and all
crop/reference records were excluded from training.

At the user's instruction, **2,940 reviewed price boxes** were reclassified:
2,114 horizontal and 826 vertical. Mean opposite-edge lengths define the marked
rectangle's local width and height; height greater than width becomes vertical,
otherwise horizontal. Existing horizontal/vertical assignments were retained.
Original changed labels were backed up under
`data/annotations/menu_obb/label_backups/before_price_relabel_20261108T002534Z/`.

| Final class ID | Label |
| --- | --- |
| 0 | dish_horizontal |
| 1 | dish_vertical |

Reviewed labels are in [labels/](../../data/annotations/menu_obb/labels). The original separate
price-class/price-association requirement was superseded. Detector labels do
not establish OCR transcription ground truth. See [stage2_review.md](stage2_review.md).

## Stage 2.3 — Dataset refinement and Colab training

The first reviewed-only snapshot used 971 train / 121 validation / 121 test
images. Its Colab session ended after the last recorded progress at epoch 52;
no checkpoint was retrieved. The cause was not confirmed.

The revised snapshot follows the requested source-specific ratios:

| Source | Training | Validation | Total |
| --- | ---: | ---: | ---: |
| Tabelog menus and interiors combined | 287 | 287 | 574 |
| Generated/augmented pages | 575 | 64 | 639 |
| Total | 862 | 351 | 1,213 |

Tabelog is exactly 1:1; generated images use the nearest whole-image 90/10
allocation. There is no separate test split in this run. The Tabelog breakdown
is 146 menus/141 interiors in training and 228 menus/59 interiors in validation;
the 1:1 quota applies to all Tabelog photos together.

Source menus, content groups, restaurants, reused backgrounds, and duplicate
hashes form 355 connected components, the largest with 212 images. An integer
allocation meets the quotas without breaking a component. Source-overlap checks
passed. No separate unreviewed Stage 1 synthetic images were added.

The snapshot is [data/datasets/detection/reviewed_v2/](../../data/datasets/detection/reviewed_v2).
Its archive SHA-256 is
`5153519ba9f1d15ae89e48467876541fd38350c390f1ffb4725060cd6fe80b4f`.

Training through Google Colab CLI (`colab`) on a Tesla T4 completed **100 epochs**
with `ultralytics==8.4.206` and these settings:

- YOLO11n-OBB; input 640; batch 16.
- AdamW, initial learning rate 0.001, cosine decay.
- Rotation ±15°, perspective 0.0005, mosaic 0.5.
- Horizontal/vertical mirroring disabled to preserve Japanese text.

The refined worker runs in the Jupyter kernel through `colab exec`. A local
chunked-transfer adapter and file-API monitor handle uploads and retrieval.
Complete best/last checkpoints are published at epoch 1, every five epochs,
and the final epoch, then downloaded and checksum-verified locally.
**Epoch 100 recovery checkpoints were retained.** Final results were retrieved,
and the Colab GPU was released.

Implementation: [prepare_training.py](../../src/detection/prepare_training.py),
[train_yolo_obb.py](../../src/detection/train_yolo_obb.py),
[colab_pipeline.py](../../src/detection/colab_pipeline.py),
[colab_transfer.py](../../src/detection/colab_transfer.py), and
[monitor_colab.py](../../src/detection/monitor_colab.py).
A [Colab notebook](../../notebooks/stage2_detection/colab_stage2_yolo_obb.ipynb) provides reproducible
code. See [stage2_training_v2.md](stage2_training_v2.md) for commands/recovery;
[stage2_training.md](../history/stage2_training_v1.md) describes the earlier snapshot.

## Stage 2.4 — Evaluation and outstanding gates

Results below evaluate the **best checkpoint**, not simply the final epoch:

| Validation set | Images | Precision | Recall | mAP50 | mAP50–95 | Matched orientation accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All validation images | 351 | 0.7883 | 0.8094 | 0.8427 | 0.6521 | 97.19% |
| Real Tabelog photos | 287 | 0.7794 | 0.8040 | 0.8340 | 0.6379 | 97.08% |

The planned targets—mAP50 ≥0.90, mAP50–95 ≥0.75, orientation ≥98%—were
**not met**. Orientation accuracy is conditional on matched detections; misses
are not counted as correctly classified examples. Validation was used for model
selection, so these are not independent test-set accuracy estimates.

The verified model is
[checkpoints/yolo11n_menu_obb_best.pt](../../checkpoints/yolo11n_menu_obb_best.pt),
SHA-256 `853659a53572315238798842f81ffcbfada5dd2a23629a0844cedf29258169bd`.

[outputs/detection/training/colab_v2/](../../outputs/detection/training/colab_v2) contains the result
archive, configuration, logs, recovery checkpoints, and
[evaluation report](../../outputs/detection/training/colab_v2/evaluation_report.json).
The archive includes confusion matrices, prediction examples, and crop audit
samples. Character clipping still requires human inspection.

## Fresh local test photos and inspection handoff

Extended the Tabelog scraper with seeded restaurant/photo shuffling and existing
manifest exclusions. Downloaded **100 menu photos from 100 distinct Osaka
restaurants**, excluding existing dataset restaurant identities and image hashes.
Seed `20261108` shuffles candidates within visited listing pages; this is not a
uniform random sample of the whole site.

Script checks passed for decoding, dimensions, unique hashes/restaurants,
source sidecars, and collection completeness. No visual inspection tools were
used; the photos remain unreviewed and unlabelled. Originals and source metadata
are in [data/raw/tabelog_test_100/](../../data/raw/tabelog_test_100).

[test_local_photos.py](../../src/detection/test_local_photos.py) ran the saved
checkpoint on CPU, input 640, confidence threshold 0.25:

| Result | Value |
| --- | ---: |
| Images processed | 100 |
| Predicted boxes | 4,980 |
| Horizontal / vertical boxes | 3,343 / 1,637 |
| Images with no detections | 2 |
| Development-set restaurant/image overlap | 0 / 0 |
| Mean / median / p95 prediction time | 44.42 / 32.44 / 51.41 ms |

Timing includes first-image warm-up and the local prediction call; it is not a
mobile or end-to-end OCR/translation measurement. **No accuracy score is
available without ground-truth annotations.** Predictions and run details are
in [outputs/detection/inference/tabelog_100/](../../outputs/detection/inference/tabelog_100).

All 100 copies with numbered oriented boxes are ready in
**[boxed_images/](../../outputs/detection/inference/tabelog_100/boxed_images)**.
Green is horizontal; blue is vertical. Box numbers follow JSON prediction order.
`index.json` links each copy to its original and predictions. Originals remain
unchanged. The user will perform inspection; its results are not yet recorded.

## Verification and remaining work

The recorded full regression run passed **66 tests** after the revised split
and recovery changes. A later scraper-only regression run passed **9 tests**,
including the large-exclusion-list case. These are separate recorded runs.
The consolidation/proposal integrity audit also passed.

Remaining work: inspect the fresh-photo overlays and crop samples, record
failure cases, and decide on further reviewed data or detector refinement to
address the unmet gates. Independent accuracy testing requires manual
annotations on the new photos. Their predictions must not be promoted to
verified training labels automatically. Stage 3 crop/transcription preparation
and recognizer training remain pending.
