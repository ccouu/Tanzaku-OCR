# YOLO11n-OBB Line Cropping & Token-Efficient VLM Contact-Sheet Pipeline

- **Pipeline Script**: [`scripts/yolo_crop_canvas_pipeline.py`](file:///D:/profolio/japanese_translation/scripts/yolo_crop_canvas_pipeline.py)
- **Default Detector Checkpoint**: `outputs/detection/training/banana130_finetune_v5/yolo11n_menu_obb_banana130_best.pt` (V5).
- V5 was selected by combined latest-label validation mAP50–95; V3 has higher recall. Neither is proven superior in actual stroke-tightness because most reviewed boxes originated from V3 proposals. Use `--weights` to override; missing weights produce an error rather than silently selecting an older detector.
- The script prepares canvases/prompts and parses supplied responses. It does not send images to a VLM service automatically.

---

## 1. Overview & Token-Saving Architecture

Contact sheets combine numbered crops into fewer image attachments. Actual visual-token costs depend on the VLM, image sizes and layout.

[`scripts/yolo_crop_canvas_pipeline.py`](file:///D:/profolio/japanese_translation/scripts/yolo_crop_canvas_pipeline.py) automates the complete 3-step loop:

1. **`build` (YOLO11n-OBB Cropping + Contact-Sheet Packing)**
   - Runs V5 oriented bounding-box detection at confidence 0.25, NMS IoU 0.7, maximum 300 detections/page.
   - Uses `src/detection/line_crops.py`: rotated-IoU duplicate suppression (0.65), containment handling, and aligned fragment merging only when intersection/smaller-core area is at least 20%. Neighboring rows and merely adjacent words are not automatically merged.
   - Adds reading-axis head/tail margins of 12% of line height and perpendicular margins of 8%. Added padding shrinks near neighbors; detected cores are never eroded to manufacture separation. Remaining core overlaps are flagged in `line_processing.json`, crop/slot metadata, and generated prompts.
   - Canonicalizes each 4-corner OBB polygon `[TL, TR, BR, BL]` so angled or tilted mobile captures are perspective-rectified upright via homography (`PIL.Image.Transform.PERSPECTIVE`).
   - Saves **two synchronized versions** of every detected line:
     - `crops_display/<menu>__line<NN>.png`: Upright, native-aspect RGB crop used on the VLM contact-sheet canvas so complex kanji strokes remain sharp and unrotated.
     - `crops_ctc/<menu>__line<NN>.png`: RGB, aspect-preserving height-64 crop, edge-padded to its next 128/256/384/512 width bucket. Vertical lines rotate 90° CCW. Extreme long lines fit within 512×64 using vertical letterboxing; metadata records valid width, scaled height and CTC input length. Native-resolution display crops remain separate.
   - Packs crops into numbered contact sheets (`canvases/canvas_001.png`, `canvas_002.png`, ..., default **25 crops per canvas**):
     - **Horizontal crops** (`dish_horizontal`, `price_tag`) are stacked top-to-bottom in a single column with a high-contrast left gutter badge (`[01]`–`[25]`) and alternating row backgrounds.
     - **Vertical crops** (`dish_vertical`, `縦書き`) are packed side-by-side as upright vertical columns (`[01]`–`[25]`) with top badges so characters stay 100% upright (`0°` rotation) for maximum VLM accuracy.
   - Generates a companion prompt file (`canvases/canvas_001_prompt.txt`) for each canvas, automatically embedding any candidate strings from `menu-XXX.json` sidecars (`gt_lines` / `intended_text`) when available.

2. **`pack` (Re-Layout Canvases Without Re-Running YOLO)**
   - Re-packs an existing `crops_manifest.json` with a different `--items-per-canvas` or orientation policy (`--mixed-orientation`, `--no-hints`).

3. **`parse` (Join VLM Transcriptions Back to CTC Training Labels)**
   - Parses `[ID] | <text> | <OK|TRUNCATED|NOISE|UNCERTAIN>` (or JSONL), joins each slot against `canvas_manifest.json`, and exports only `OK` lines to CTC labels by default. All statuses and overlap-review flags remain in `labeled_crops.jsonl`.
   - Prompts explicitly state that YOLO proposals may be background or objects: use empty `NOISE` for no visible text; `UNCERTAIN` for visible but unreadable text; never invent menu terms, complete cut-off words, or treat sidecar hints as ground truth. Overlap flags are advisory and do not automatically exclude an otherwise `OK` response; inspect them before training.
     - `labeled_crops.jsonl` (full metadata + OBB coordinates + VLM transcription + status)
     - `ctc_labels.txt` (`crops_ctc/<crop>.png\t<transcription>`, ready for Stage-3 recognizer training or evaluation).

---

## 2. Quick-Start Commands

### Step 1: Run YOLO11n-OBB & Build Contact-Sheet Canvases (`build`)

Run on your menu photo folder using the compatible local environment and a new output directory:

```powershell
.\.venv-recognition\Scripts\python.exe scripts/yolo_crop_canvas_pipeline.py build `
  --input data/raw/gemini_banana_mobile_menus_50 `
  --output data/processed/gemini_banana_50_crops `
  --device 0 `
  --confidence 0.25 `
  --items-per-canvas 25
```

#### Optional Flags for `build`
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--weights` | V5 all130 checkpoint above | Explicit detector override, including V3 |
| `--confidence` | `0.25` | Minimum OBB detection confidence |
| `--imgsz` | `640` | YOLO inference resolution |
| `--device` | `cpu` | PyTorch device (`cpu` or `0` for CUDA GPU) |
| `--iou-dedup` | `0.65` | Rotated duplicate IoU threshold |
| `--merge-iom` | `0.20` | Minimum intersection/smaller-core area for aligned merging; range .20–1 |
| `--legacy-crops` | `False` | Reproduce AABB suppression and stretched grayscale exports |
| `--pad-x` | `0.04` | Legacy mode only: width padding ratio |
| `--pad-y` | `0.06` | Legacy mode only: height padding ratio |
| `--items-per-canvas` | `25` | Maximum number of crops packed onto each contact sheet |
| `--mixed-orientation` | `False` | Pack horizontal and vertical crops on the same row canvas instead of separate upright column canvases |
| `--no-hints` | `False` | Omit sidecar `gt_lines` candidate hints from `canvas_XXX_prompt.txt` |

---

### Step 2: Transcribe Contact Sheets Using Vision (`canvas_001.png` ...)

Attach `canvases/canvas_001.png` (or several canvases at once) in your prompt and use the generated instructions in `canvases/canvas_001_prompt.txt`:

```text
# canvas_001
[01] | 本日のおすすめ | OK
[02] | 枝豆 380円 | OK
[03] | 冷奴 420円 | OK
[04] | 唐揚 680円 | OK
[05] | 焼鳥盛合せ 980円 | TRUNCATED
[06] | | NOISE
[07] | | UNCERTAIN
```

> [!TIP]
> When saving responses for multiple canvases into a single file (`responses.txt`), include a `# canvas_001`, `# canvas_002`, ... header line before each block (or prefix each line with `canvas_001 [01] | ...`).

---

### Step 3: Parse VLM Output into Stage-3 CTC Labels (`parse`)

Once you save the VLM transcriptions to `responses.txt`:

```powershell
python scripts/yolo_crop_canvas_pipeline.py parse `
  --output data/processed/gemini_banana_50_crops `
  --responses data/processed/gemini_banana_50_crops/responses.txt `
  --keep-status OK
```

To include both `OK` and `TRUNCATED` crops in `ctc_labels.txt`:

```powershell
python scripts/yolo_crop_canvas_pipeline.py parse `
  --output data/processed/gemini_banana_50_crops `
  --responses data/processed/gemini_banana_50_crops/responses.txt `
  --keep-status OK TRUNCATED
```

---

### Optional: Re-Pack Canvases Without Re-Running YOLO (`pack`)

If you want to change the number of crops per canvas (e.g., from `25` to `20`) after YOLO cropping has already finished:

```powershell
python scripts/yolo_crop_canvas_pipeline.py pack `
  --output data/processed/gemini_banana_50_crops `
  --items-per-canvas 20
```

---

## 3. Output Directory Structure

```text
<output_dir>/
├── crops_manifest.json          # Full YOLO11n-OBB detection & crop provenance
├── canvas_manifest.json         # Reversible mapping: (canvas_id, [slot]) -> crop_id & OBB
├── crops_display/               # Upright RGB line crops rendered onto VLM contact sheets
│   ├── menu-001__line01.png
│   └── ...
├── line_processing.json         # Core geometry, merges, duplicate/padding decisions, overlap flags
├── crops_ctc/                   # RGB H64 bucketed crops; legacy flag retains old format
│   ├── menu-001__line01.png
│   └── ...
├── canvases/                    # Numbered contact-sheet images + ready-to-use prompt files
│   ├── canvas_001.png
│   ├── canvas_001_prompt.txt
│   └── ...
├── labeled_crops.jsonl          # Produced by `parse`: full crop records with VLM labels
├── ctc_labels.txt               # Produced by `parse`: <ctc_crop_relpath>\t<text>
└── parse_summary.json           # Produced by parse: counts including UNCERTAIN/UNLABELED
```
