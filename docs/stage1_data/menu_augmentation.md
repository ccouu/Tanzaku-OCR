> Paths updated for the reorganized workspace. Historical artifact contents and provenance paths are retained.

# Menu Geometry and Restaurant Compositing

## Purpose

The pipeline augments complete menu images, pastes the augmented menus onto
restaurant interiors, and preserves existing text information. It performs data
preparation only. **YOLO text-boundary label assignment remains deferred to
Stage 2**: this pipeline does not create new YOLO `.txt` files or assign new
detector classes.

Implementation: `src/data/menu_augmentation.py`.
CLI: `scripts/data/augment_menus.py`.
Audit: `scripts/data/verify_menu_augmentation.py`.

## Inputs

- `data/raw/genai_seeds/menu_images/`: generated menus and their JSON sidecars.
  The intended prompt text is retained, with `text_verified=false`. These files
  do not currently provide verified line polygons or transcriptions.
- `data/processed/synthetic_base_v1/yolo_obb/`: synthetic page images, original OBB files, and `pages.json`.
  The pipeline checks that existing OBB coordinates agree with the page-info
  polygons and preserves the exact rendered text for each line.
- `data/raw/tabelog_interiors/`: restaurant photos selected by its `manifest.json`.
  Contact sheets, rejected images, and diagnostic snapshots are not selected.
  An alternative background directory can contain ordinary image files without
  a manifest; filenames then identify background groups.

No API calls, model inference, new packages, or training are needed. The pipeline
uses the repository's existing NumPy and Pillow dependencies.

## Geometry and text preservation

Each source menu is treated as a 3D plane and projected through a pinhole camera.
Default angles are pitch magnitude 0–30 degrees, randomly forward/backward
(signed −30 to +30), roll −10 to +10 degrees, and yaw −10 to +10 degrees.
Rotation order is pitch about X, yaw about Y, then roll about Z. Positive pitch
brings the top edge toward the camera; positive yaw brings the right edge
toward it; positive roll is clockwise in image coordinates.

The full menu plane is fitted inside a 640×640 RGBA canvas with transparent
margins. Corner checks reject clipping, degenerate polygons, and invalid
projective geometry. The same homography transforms the existing text polygons.

The resulting augmented menu is scaled by a uniformly sampled factor of
**0.5–1.2** and pasted onto a fitted, randomly chosen restaurant background,
with subtle brightness variation and a soft shadow. The factor is relative to
the saved augmented menu. Scaling above 1.0 is applied without clamping; menu
geometry reserves enough space first, and placement uses the visible menu's
corners rather than its transparent canvas. Only transparent padding may leave
the scene frame. The paired
scene metadata contains the composed source-to-scene matrix and updated line
polygons. The original source images and labels remain unchanged.

Where existing source text and line boundaries are known, the pipeline also
extracts grayscale 384×64 CTC crops from the composite. Vertical lines are
rectified upright top-to-bottom, then rotated 90 degrees counterclockwise;
`orientation_flag=1` is retained. Horizontal lines and price strings use flag 0.
The grayscale PNG and orientation flag can be loaded through the existing
orientation tensor helper to form `(2,64,384)` input.

Generated menus without known line polygons still receive paired JSON files
containing their original intended text and verification status. No CTC
transcriptions or line crops are fabricated for those images.

## Output layout

```text
data/processed/augmentation/run_1100/
  config.json
  manifest.json
  verification_report.json
  preview.jpg
  ctc_preview.jpg
  menu/
    images/{train,validation,test}/<sample>.png
    labels/{train,validation,test}/<sample>.json
  composites/
    images/{train,validation,test}/<sample>.jpg
    labels/{train,validation,test}/<sample>.json
  ctc_lines/
    images/{train,validation,test}/<sample>__line00.png
    labels/{train,validation,test}/<sample>__line00.json
    labels.jsonl
```

JSON sidecars preserve source paths/checksums, split, exact known text,
unverified intended text where applicable, transform matrices, menu boundary,
and transformed source line polygons. Both menu and scene sidecars record
`pose.pitch_degrees`, `pose.pitch_magnitude_degrees`, `pose.pitch_direction`,
`pose.roll_degrees`, and `pose.yaw_degrees`. Scene sidecars also record the
actual `menu_scale` and selected background. Original class IDs are retained only as
source provenance (`source_class_id`), not as new YOLO labels.

Synthetic pages keep their original splits. Generated-menu handwriting variants
with the same intended content remain in one split. Restaurant/background
groups are partitioned independently and are used only within their assigned
split. Variants of one source never cross splits. At least three independent
generated-content/background groups are required for three-way splitting.

## Commands

```powershell
# Full source set, one augmentation per menu; also exports known CTC lines.
python -m scripts.data.augment_menus --output-dir data/processed/augmentation/run_1100 --variants 1 --workers 4

# A separate run with more variations and gentler geometry.
python -m scripts.data.augment_menus --output-dir data/processed/augmentation/run_1100-v2 --variants 3 --max-pitch 15 --max-roll 5 --max-yaw 5

# Exactly 500 pairs with the requested angle and scale ranges.
python -m scripts.data.augment_menus --output-dir data/processed/augmentation/run_500 --count 500 --max-pitch 30 --max-roll 10 --max-yaw 10 --min-scene-scale 0.5 --max-scene-scale 1.2

# Preview a small subset without CTC export.
python -m scripts.data.augment_menus --output-dir data/processed/augmentation/run_1100-preview --max-menus 5 --no-ctc

# Audit existing output and run the focused tests.
python -m scripts.data.verify_menu_augmentation --data-root data/processed/augmentation/run_1100
python -m pytest tests/test_menu_augmentation.py -q
```

Additional options select source directories, canvas size, random seed, scene
scale range, and worker count. Existing nonempty output directories are refused
to prevent accidental overwrites. Use a fresh output directory for another run.
The random seed and sample ID determine transforms independently of worker order.
Backgrounds are chosen uniformly at random within the source's split, with
replacement. A fixed seed makes choices reproducible. `--count` shuffles the
candidate pool across both menu source sets and returns exactly that many pairs;
additional variant IDs are created if the count exceeds the available sources.

## Verification and remaining review

The audit checks paired files, checksums, exact transcription preservation,
matrix composition, source/line corner placement, CTC shape and orientation,
generated-text verification flags, and disjoint menu/background groups. It
compares output annotations against the original source metadata.
For the 3D pipeline it also checks angle ranges, recreates the projected corners
from the recorded pose, and confirms that the actual scale matches its metadata.
The report includes measured angle/scale ranges and background diversity.

The final composites are candidates for later YOLO annotation. Restaurant
backgrounds can contain other menus or signage; inspect them when preparing
Stage 2 text-boundary labels. Source-image licensing and the existing need to
review generated Japanese text remain unchanged. The output is separate from
the original Stage 1 archive.

## Earlier default run

The run under `data/processed/augmentation/run_1100/` produced 1,100 augmented menus and 1,100
restaurant composites, split into 880 train / 110 validation / 110 test pairs.
It also exported 9,000 labeled CTC crops from the 1,000 synthetic pages' known
line regions. The 100 generated menus retained unverified intended-text metadata
and were not assigned CTC line labels. The full integrity audit passed. Use
`preview.jpg` to inspect menu/composite pairs and `ctc_preview.jpg` for normalized
line samples with their transcriptions and orientation flags.

The later `data/processed/augmentation/run_500/` run uses the explicit pitch/roll/yaw geometry,
200 available Tabelog interiors, random 0.5–1.2 menu scaling, and the exact-count
option. Its own `config.json`, `manifest.json`, and `verification_report.json`
record the parameters and results without overwriting the earlier run.

This 500-pair run passed the audit, with 400 training, 44 validation, and 56
test samples. It used 187 distinct restaurant backgrounds and included 140
actual enlargements above 1.0×. It retained 4,050 known text regions as CTC
crops and preserved unverified intended text for 50 generated-menu samples.
Measured pitch was −29.834° to +29.942°, roll −9.991° to +9.948°, yaw −9.987°
to +9.963°, and scale 0.5013–1.1998×. Each output has its paired JSON metadata.
