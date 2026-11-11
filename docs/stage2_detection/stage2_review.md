> Paths updated for the reorganized workspace. Historical artifact contents and provenance paths are retained.

# Stage 2.1 and 2.2: boundary proposals and manual review

The consolidated collection is `data/annotations/menu_obb/images/`. Every image recursively
found under `data/processed/augmentation/run_500`, `data/raw/tabelog_menus`, and
`data/raw/tabelog_interiors` was copied byte-for-byte into this single flat
folder. The leading `/data` in the request is interpreted relative to the
project. Original images and JSON metadata remain intact so Stage 1 paths,
audits, and augmentation provenance continue to work.

## Inventory and traceability

There are 5,716 images, including 1,585 page images (1,000 augmented menus and
composites, 385 active Tabelog menus, 200 active interiors), 4,050 derived CTC
line crops, 37 previews/contact sheets, and 44 rejected/diagnostic references.
The Stage 1 summary recorded 400 active menus; the source directory at this
handoff contains 385. No replacement photos were downloaded.

The manifest is `data/annotations/menu_obb/manifest.json`; each entry has a stable ID derived
from its original path, its new filename, original path, image SHA-256,
dimensions, role, original metadata path/checksum, and a full metadata snapshot.
The same entry is independently saved in `data/annotations/menu_obb/provenance/<id>.json`.
These snapshots preserve restaurant/source URLs, collection/license state,
source-image hashes, split/group identities, transforms, and text verification
state when provided by Stage 1. Images without sidecars explicitly have null
metadata; no transcription or provenance is invented. Identical pixels from
different source paths remain separate records, traceable by checksum.

Only `page` records are editable as detector training pages. All images have
predictions, but crops and reference images remain read-only in the UI's
“All images” filter. This prevents contact sheets, quarantined images, and
correlated line crops from silently entering the page training corpus.

## Run or resume

From the repository root:

```powershell
python -m pip install -r requirements-stage2.txt
python -m src.detection.consolidate_images
python -m src.detection.predict_initial_boxes
python -m src.detection.review.obb_review_ui
```

On this machine the detector runtime is installed under `.stage2_packages/`
and loaded automatically. The standard install command above is for a fresh
Python environment. `--root` selects a different collection for prediction,
review, import, and audit. Consolidation uses `--output`. All commands must be
run from the project root. Consolidation retains reviewed labels; it rejects
an existing consolidated image whose bytes differ. Prediction skips completed
records, checks model/config/image identity, and resumes missing records.
Use a fresh output root when deliberately changing detector settings.

Stage 2.1 uses PP-OCRv4 DBNet ONNX weights distributed with RapidOCR 1.4.4,
running locally on CPU at a maximum side of 960 pixels. It loads detection
only; no recognition, LLM call, or runtime model download occurs. The exact
weights checksum and inference configuration are saved with every prediction
and in `prediction_report.json`. Optional `--weights` accepts a local PP-OCRv4
ONNX detector, `--side` changes resolution, and `--limit` supports smoke runs.
See the [RapidOCR model registry](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/default_models.yaml)
and [PaddleOCR model page](https://huggingface.co/PaddlePaddle/PP-OCRv4_mobile_det).

Transparent menu images are composited on white for inference, matching the
canvas background. Model quadrilaterals are preserved, rather than replacing
tilted boundaries with axis-aligned boxes. Each image gets a YOLO-OBB `.txt`
and X-AnyLabeling `.json` under `data/annotations/menu_obb/predictions/`, including successful
zero-detection results. Errors remain explicit and resumable in the report.

Class IDs are `0 dish_horizontal`, `1 dish_vertical`, and `2 price_tag`.
Height/width above 1.5 suggests vertical. A short region (both dimensions below
30% of image size) in the rightmost 18% or bottom 15% suggests price. The class
helper also supports currency text, but this detection-only run has no OCR
transcriptions and therefore uses geometry alone. These guesses, especially
prices, handwriting, and background signage, require manual correction.

## Manual review

Open **http://localhost:7860** after starting the server. It binds only to
127.0.0.1. The default queue shows unreviewed pages. Select another page using
the dropdown or Previous/Next; leaving unsaved changes requests confirmation.
Expand the metadata panel to trace the image back to its Stage 1 information.

- Click a box to select it; drag inside to move it.
- Drag a corner to resize; drag the round handle above it to rotate.
- Drag empty canvas to draw. Use **Draw new box** to draw over another box.
- **1/2/3** changes class (green horizontal, blue vertical, orange price).
- **Z/C** rotates −1°/+1°; **X/V** rotates −0.1°/+0.1°.
- **Delete/Backspace** removes a box; **Ctrl+Z** or Undo restores an edit.
- **Space/Enter** or **Save verified & next** approves and advances.

An empty annotation requires explicit confirmation as a negative. Opening an
image never approves it. The server validates class names, image bounds,
finite convex four-corner geometry, image hashes, and concurrent edits before
saving. Reviewed outputs are separate `.json`/`.txt` pairs in
`data/annotations/menu_obb/labels/`. Original predictions and source metadata are retained.
JSON is the authoritative completion record; an audit detects a mismatched
JSON/text pair if a process is interrupted during saving.

For X-AnyLabeling, open the prediction JSON files (their relative image links
resolve to the consolidated images). Keep the class names, four ordered
polygon corners, and custom identity/hash fields. Back up or copy predictions
with their sibling `images/` folder layout before editing. Place only corrected
JSON files into a separate import directory, then explicitly approve them:

```powershell
python -m src.detection.import_reviewed --input data/annotations/menu_obb/xany_reviewed
# Add --confirm-empty only when you have reviewed empty annotations as negatives.
```

The importer validates all selected files before beginning writes and rejects
reference/crop records. It preserves inference provenance and records approval
through the same review store. X-AnyLabeling itself is not installed or visually
tested in this environment.

## Validation and training handoff

```powershell
python -m pytest -q
python -m src.detection.verify_stage2
```

The audit checks original/copied hashes, metadata snapshots, image links,
prediction completeness, polygon geometry, YOLO/JSON agreement, and review
state. The report distinguishes successful integrity checks from pending manual
review. Browser automation was unavailable, and access to the unrelated open
Chrome session was rejected by automatic approval review; the canvas has
JavaScript syntax and HTTP API checks but no completed visual browser test.

Training and evaluation have since completed; see the [Stage 2 summary](stage2_summary.md)
for results and remaining accuracy gates. For another run, use only explicitly
reviewed page labels after completing the labeling pass. Before training, build
a joint grouped split: augmentation menu/composite pairs, synthetic content,
restaurant IDs, source menus, and reused backgrounds connect related records.
Existing per-run splits cannot be applied blindly to this combined folder.
Do not assume interiors are empty negatives, or generated prompt text is a
verified transcription. Stage 3 crop transcription review remains separate.
