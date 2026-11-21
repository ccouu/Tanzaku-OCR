# YOLO11n-OBB Line Cropping & Token-Efficient VLM Contact-Sheet Pipeline

Updated 2026-11-20. **Required ordering: VLM labeling → validated CTC label export → CTC character inventory/vocabulary candidate → separate LLM culinary-term extraction → optional LM/trie candidates.** **Implementation boundary:** `build`, `pack` and `parse` are implemented in the script. The post-labeling eligibility gates, reviewed delta updates and activation checks below are workflow requirements; they are **not all enforced automatically by that script**. Historical batch outputs are not proof of deployment or recognition support for new characters.

- **Pipeline Script**: [`scripts/yolo_crop_canvas_pipeline.py`](../../scripts/yolo_crop_canvas_pipeline.py)
- **Default Detector Checkpoint**: `outputs/detection/training/banana130_finetune_v5/yolo11n_menu_obb_banana130_best.pt` (V5).
- V5 was selected by combined latest-label validation mAP50–95; V3 has higher recall. Neither is proven superior in actual stroke-tightness because most reviewed boxes originated from V3 proposals. Use `--weights` to override; missing weights produce an error rather than silently selecting an older detector.
- The script prepares canvases/prompts and parses supplied responses. It does not send images to a VLM service automatically.

---

## Recognition geometry warning (V76 onward)

This document describes the **actual shared exporter**, including its legacy extreme-line height fit. It has not been changed by the V76 training adapter. For corrected recognizer inputs, start from native `crops_display`, apply one CCW90 rotation for nativevertical, proportionally resize toH64, then right-pad only. If width exceeds512, split into<=512 windows with100H64px overlap; do not shrink the whole line or pad its height to fit. Training windows require correct partial labels, while evaluation/inference uses label-independent boundaries and coordinate-based output merging. `predict_long(...,mode='overlap512')` is explicit; its default does not select splitting.

Keep the full-line VLM transcription and parent ID on the native crop. `ctc_labels.txt` is an export, not automatically a compliant training dataset: validate status/OOV/geometry/collection-qualified splits before using it. Do not train full labels on partial windows or count multiple windows as independent source menus. V74/V75 historical scores used precomputed exports; V76's post-completion audit separately measures nativeH64/split inference.

Old collection `data/processed/gemini_banana_mobile_menus_50_crops/` covers001–130. New collection `data/raw/gemini_banana_mobile_menus_train_50/yolo_crops/` contains a separate50menus. Identical basenames do not imply identical images. Labels are provisional VLM transcriptions; menu-generation sidecars are hints, never pixel-verified ground truth.

## 1. Overview & Token-Saving Architecture

Contact sheets combine numbered crops into fewer image attachments. Actual visual-token costs depend on the VLM, image sizes and layout.

[`scripts/yolo_crop_canvas_pipeline.py`](../../scripts/yolo_crop_canvas_pipeline.py) automates the complete 3-step loop:

1. **`build` (YOLO11n-OBB Cropping + Contact-Sheet Packing)**
   - Runs V5 oriented bounding-box detection at default confidence **0.35**, NMS IoU 0.7, maximum 300 detections/page. A lower threshold such as0.25 is an explicit collection experiment, not the default.
   - Uses `src/detection/line_crops.py`: rotated-IoU duplicate suppression (0.65), containment handling, and aligned fragment merging only when intersection/smaller-core area is at least 20%. Neighboring rows and merely adjacent words are not automatically merged.
   - Adds reading-axis head/tail margins of 12% of line height and perpendicular margins of 8%. Added padding shrinks near neighbors; detected cores are never eroded to manufacture separation. Remaining core overlaps (`iom > 0.02`) are classified under `minor_overlap_v1_strict_lt10pct` in `src/detection/line_crops.py`: pairs with **both** `intersection_area/core_area < 0.10`, **both** perpendicular intrusion fractions `< 0.10`, compatible reading axes (`<= 8°`), and finite non-degenerate geometry are marked `minor_allowed` (`minor_overlap_allowed=True`, `review_required=False`, `legacy_review_required=True`); any major/crossing/asymmetric conflict vetoes the crop to `major_review` (`review_required=True`). All conflicts and measurements remain preserved in `line_processing.json`, crop/slot metadata, and generated prompts.
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
     - `ctc_labels.txt` (`crops_ctc/<crop>.png\t<transcription>`, an intermediate export requiring the eligibility and geometry checks below before Stage-3 use).

---

## 2. Quick-Start Commands

### Step 1: Run YOLO11n-OBB & Build Contact-Sheet Canvases (`build`)

Run on your menu photo folder using the compatible local environment and a new output directory:

```powershell
.\.venv-recognition\Scripts\python.exe scripts/yolo_crop_canvas_pipeline.py build `
  --input data/raw/gemini_banana_mobile_menus_50 `
  --output data/processed/gemini_banana_50_crops `
  --device 0 `
  --confidence 0.35 `
  --items-per-canvas 25
```

#### Optional Flags for `build`
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--weights` | V5 all130 checkpoint above | Explicit detector override, including V3 |
| `--confidence` | `0.35` | Minimum OBB detection confidence; optionally lower explicitly for recall-focused collection |
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

**Camera photo input:** this batch script does not explicitly apply the inspector's EXIF-transpose step. Before building, create orientation-normalized copies so the detector image, Pillow crop image and coordinate frame agree. Record source/copy hashes and dimensions; preserve originals. Supported extensions are PNG/JPG/JPEG/WebP. HEIC is not accepted by this script: convert to an orientation-correct supported format first. Do not assume the separate localhost inspector's input validation also runs here.

For independent evaluation labeling, prefer `--no-hints` to reduce transcription anchoring. If hints were supplied, record that fact; it does not turn them into verified ground truth.

---

### Step 2: Transcribe Contact Sheets Using Vision (`canvas_001.png` ...)

Attach `canvases/canvas_001.png` (or several canvases at once) in your prompt and use the generated instructions in `canvases/canvas_001_prompt.txt`:

**Real-photo content/geometry gate (added 2026-11-21):** inspect source-page overview contact sheets **before** crop transcription. A rectified crop alone cannot reliably reveal that it came from a promotional banner or a severely distorted page. Record one source-quality decision per source image, bound to its image hash, plus crop-level exclusions for unrelated content on otherwise valid menus. Save the original VLM responses, model/version, prompts and all exclusion reasons. This is a required labeling workflow; the existing `build/pack/parse` commands do not themselves call a VLM or enforce source-page screening.

Use this source-screening instruction alongside the numbered full-photo overviews:

```text
For each source image, determine whether it contains a usable restaurant menu:
actual dishes, drinks, prices, portions or ordering information. Exclude unrelated
photos, decorative signs, shop names, promotional/advertising banners without a
usable menu, and visible non-menu text. Real menu boards, hanging dish tags,
vertical menus and banners that actually function as a menu are not automatically
non-menu merely because of their physical format.

Assess the EXIF-upright source image, not the rectified crop. For this conservative
collection, exclude pages rolled more than 45 degrees away from their intended
upright reading orientation; native vertical writing is not a 90-degree roll.
Also exclude severe perspective foreshortening, twisted/curved surfaces, or other
distortion that makes faithful line transcription unreliable, even below 45 degrees.
Do not claim a calibrated camera angle from pixels. Record estimated roll and its
confidence when assessable; mark uncertain angle/content cases for review instead
of inventing a measurement. Distinguish ordinary correctable slant from severe
distortion. When a photo contains both menu and unrelated text, retain only clearly
identified menu regions and mark other crops ineligible.

Return one structured decision per source ID: MENU, NON_MENU, SEVERE_DISTORTION
or REVIEW, with reasons, assessable roll estimate, and any mixed-content regions.
Never infer menu relevance from the detector confidence or filename.
```

Add this instruction to the generated crop-transcription prompt when using the gate:

```text
Use the source-page quality/context decisions as eligibility metadata. A readable
shop banner or advertisement is not automatically eligible menu training text.
Preserve visible text and the existing four transcription statuses, but separately
record crop eligibility and exclusion reasons (non_menu_content, severe_distortion,
excessive_roll or source_quality_review). Do not label readable rejected text as
NOISE; NOISE means no visible text. Use UNCERTAIN for unreadable/ambiguous strokes.
Do not infer original tilt from a rectified crop or reject native vertical writing.
Only source-approved, menu-relevant, fully readable OK crops enter training.
```

Keep source/crop quality decisions in a stable-ID sidecar (e.g. `source_quality.jsonl`, `crop_quality.jsonl`), separate from the parser's four-status pipe format. Never silently introduce new transcription statuses. Source rejections propagate to every child crop; mixed-page exclusions require crop-level decisions. Resolution thresholds, overlap eligibility, source quality and transcription status are independent gates. VLM approval remains provisional rather than independent human ground truth.

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

**Diagnostic export only:** the following can retain truncated examples for review. It does not make them valid CTC training samples; current training excludes TRUNCATED unless a correctly aligned visible-text label is independently established:

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

**Repacking invalidates slot-based responses:** the same `[01]` may now identify a different crop. Archive the previous canvas manifest/prompts/responses and hash the new manifest before repacking. Retranscribe, or explicitly remap validated responses through stable `crop_id`; never reuse old `(canvas_id, slot)` answers against a new layout. The current parser does not enforce manifest-version binding.

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
├── parse_summary.json           # Produced by `parse`: counts including UNCERTAIN/UNLABELED
├── focus_training_delta_<batch>.json  # Separate post-labeling workflow output, not emitted by build/parse
└── focus_training_delta_<batch>.log   # Separate post-labeling summary
```

---

## 4. End-to-End Post-Labeling Workflow: CTC Coverage First, Then Term Extraction

After parsing, apply Step4.0 before preparing any candidate training, vocabulary, LM or lexicon assets. `UNLABELED == 0` alone does not prove a correct response join. The following steps describe a versioned post-labeling workflow, not extra automatic stages of `yolo_crop_canvas_pipeline.py`.

> [!IMPORTANT]
> **Frozen-Asset Safety Rule:** Never overwrite promoted or test-pinned assets in place:
> - `checkpoints/l2_v06/vocab.json` (SHA-pinned in `src/recognition/current_model.json`)
> - `checkpoints/l2_v06/char_lm_4g_hp1010.json.gz` & `_bwd.json.gz` (SHA-pinned in `checkpoints/l2_v06/decoder_assets.json`)
> - `outputs/recognition/implementation/lexicon_trie_python_20261119/artifacts/lexicon_index_v1.json.gz` (referenced by unit tests)
> - `data/processed/lexicon/culinary_menu.db` (strictly read-only)
>
> Always write versioned outputs (`_<batch>` suffix) so existing checkpoints and tests remain unbroken until an explicit promotion.

### Step 4.0: Dataset Eligibility & Asset Compatibility Gate

**Required before adaptation; implement a separate validation report where the script lacks enforcement:**

1. **Bind labels to exact inputs:** record batch/collection ID, source menu and crop hashes, detector checkpoint/hash/config, crop policy, canvas-manifest hash, prompt hash, response hash and labeling model/version. Require exactly one valid response per expected `(canvas_id, slot)`. Reject conflicting duplicates, unknown canvas/slot IDs, malformed lines, unsupported statuses and empty `OK` text. Current parsing can overwrite duplicate keys and ignore orphan/unmatched responses; audit original response records, not just `parse_summary.json`.
2. **Split before extraction:** define collection-qualified menu-level train/validation/test groups and verify page/image-hash disjointness. All windows/views of a parent remain in its menu split. Exclude validation/test labels from training, term selection, LM count updates, lexicon expansion and synthetic generation. A later externally sourced dictionary addition that overlaps a test term must have independent provenance and its overlap disclosed. Once held-out labels informed adaptation, that set is development data, not blind confirmation.
3. **Validate eligible training records:** nonempty reviewed `OK` labels, resolved overlap flags, correct flag0/flag1 and one CCW rotation, vocabulary coverage/font support, correct native-H64 geometry/window labels, and sufficient CTC frames including repeated-character blank requirements. Keep NOISE, UNCERTAIN, UNLABELED and unresolved TRUNCATED records in the audit, outside default recognition training. Record raw labels separately from canonical labels. VLM `OK` is a transcription claim, not independent verification.
4. **CTC versus lexicon scope:** OCR vocabulary covers all visible transcription characters, including non-food words, Latin and punctuation. Culinary filtering belongs only to the later LLM/LM/trie workflow. A complete-batch character inventory is not authorization to train on held-out labels; record any full-batch candidate as label-informed.
5. **Pin the source recognition bundle:** actual checkpoint path/hash, state RAW/EMA, architecture, matching vocabulary/hash, blank/ID mapping and classifier output size. Read the source checkpoint's associated vocabulary; the production registry supplies it only if that is the chosen source. Do not choose `vocab_v40.json` or another historical file by filename recency.
6. **Keep candidate assets inactive:** vocabulary, classifier, forward/backward LM schema/transform/vocabulary hash, trie schema/source hashes and translation DB version must form a coherent bundle. A1830-character vocabulary requires1831 classifier outputs including blank0; the current1779-character model has1780 outputs. Mismatched bundles must fail explicitly. An untrained new token is not recognizable merely because its ID exists in JSON.

The gate report should list counts/exclusions, split membership, asset compatibility and unresolved blockers. A candidate may be saved for review with blockers, but must not be described as training-ready or activated.

### Step 4.1: Parallel VLM Canvas Transcription Best Practices
- Batch `5–6` canvases per parallel vision subagent (up to 5 subagents per batch) using `=== canvas_XXX.png ===` headers and `[ID] | <exact text> | <OK | TRUNCATED | NOISE | UNCERTAIN>`.
- Extract completed subagent outputs programmatically from their transcripts into `<output_dir>/vlm_responses.txt` in numerical order (`001..NNN`) to prevent truncation.
- Run `scripts/yolo_crop_canvas_pipeline.py parse`, inspect `parse_summary.json`, and perform the original-response/manifest bijection checks in Step4.0 before proceeding.

### Step 4.2: Full-Label CTC Character Inventory & Append-Only Dictionary Extension
- Preserve original text and apply the source bundle's agreed canonical policy to training labels/terms: full-width decimal digits (`０–９ -> 0–9`) and ASCII/full-width parentheses handling. Keep visual lookalikes distinct; do not apply a broad Unicode/NFKC or confusable-skeleton rewrite as an undocumented extra normalization.
- **Character source is the complete validated CTC transcription, not LLM-extracted terms.** Read the text field of `ctc_labels.txt`, cross-check each crop/text/status against `labeled_crops.jsonl` and the canvas manifest, and inventory every canonical label character. Do not scan image paths, TSV separators or line endings as text. Detect the actual delimiter format explicitly and reject malformed records; do not silently split transcription contents.
- Load the vocabulary pinned to the actual source checkpoint in Step4.0. If continuing an already published append-only candidate, verify it preserves all model IDs and retain its appended IDs as well; record both the trained base and candidate parent hashes. Call `extended_vocab, added_chars = extend_vocabulary(parent_vocab, all_label_chars)` from [`src/recognition/vocabulary.py`](../../src/recognition/vocabulary.py). Do not infer the base from a historical version name.
- Include legitimate Latin text, sentence kanji, price commas and other visible punctuation/symbols even when they are not culinary terms. Do not use term-extraction exclusions to filter OCR characters. Preserve raw/canonical labels and flag questionable VLM glyphs for review rather than deleting characters to achieve zero OOV. Distinguish every-character coverage from independent transcription verification.
- Publish counts/frequencies and representative crop references for raw versus canonical characters; report OOV against both the trained base and candidate vocabulary, split by train/validation/test. A full-batch audit candidate can cover all parsed `OK` labels when explicitly requested, but must be marked **label-informed across all splits**. Do not silently use its held-out-only characters as a training selection signal or call later tests untouched blind evaluations. The default adaptation inventory remains training-only; candidate activation requires its own compatibility/training approval.
- Verify strict append-only invariants:
  - `blank_id == 0`
  - Every existing `(char, id)` mapping in `1..N_old` is unchanged
  - Newly added tokens are appended at `N_old+1 .. N_new` in deterministic `ord` order
- Record glyph support across the **actual experiment font binaries** using `fontTools.ttLib.TTFont`; hash/count only `.ttf/.otf/.ttc` files, not licenses/metadata, and handle collection faces explicitly. The historical recipe used15 fonts; don't assume a resource-directory glob always reproduces that pool. Record unsupported term/font combinations and chosen fallbacks/exclusions.
- Save to `data/processed/recognition/vocab_v<NN>_<batch>.json`.
- **Classifier expansion and training are separate required work:** use the reviewed vocabulary/head helpers to preserve every old row and append new rows in exactly the same ID order. Select an explicit below-blank initialization, check zero-update old-prediction/new-emission behavior in the intended precisions, and train the new rows with measured character coverage and old-data retention checks. Initial parity is a test, not guaranteed by append-only JSON. Verify actual gradients/updates for every added row, new-character recall and old-character/price/blank behavior. Save a compatible fused checkpoint and vocabulary together. Do not switch the inspector or registry to an extended vocabulary before a compatible trained model is approved.

### Step 4.3: LLM Culinary Term Extraction
1. Use **training-split records only**. Deduplicate nonempty labels for extraction efficiency, retaining a mapping back to every collection/menu/crop, status and source hash. Keep non-OK material in a separate review queue; do not mix it into accepted term counts or language-model training.
2. Split into chunks (~300–400 unique labels per LLM worker) and extract clean, complete Japanese culinary/menu terms in strict `TERM<TAB>CATEGORY` format:
   - **Extract:** Full clean dish names AND standalone culinary sub-terms (e.g., `自家製パンチェッタのカルボナーラスパゲティ` → full dish + `パンチェッタ`, `カルボナーラスパゲティ`), drinks, sake/shochu/awamori/beer brands, seafood/meat cuts, produce, cooking methods, menu categories, and portion/tax modifiers.
   - **Exclude:** Prices/numbers/counts (`680円`, `2本`, `100g`), phone numbers, store hours, decorative bullets/arrows (`・`, `▽`, `※`, `☆`, `↑`, `→`, `……`), promotional/instructional sentences (`QRコードからご注文下さい`, `写真はイメージです`), pure English marketing slogans, and cut-off fragments from `TRUNCATED`/`UNCERTAIN` rows (only extract a term from a non-`OK` line if that specific term is 100% complete as written; never invent or complete cut-off characters).
   - **Allowed categories** (aligned with `menu_lexicon.json`): `dish`, `beverage_alcohol`, `beverage_nonalcoholic`, `menu_category`, `ingredient_seafood`, `ingredient_meat`, `ingredient_produce`, `ingredient_grain_noodle`, `ingredient_fruit_dessert`, `ingredient_egg_dairy`, `ingredient_other`, `cooking_method`, `seasoning_condiment`, `modifier_portion`, `modifier_flavor`, `modifier_topping`, `modifier_temperature`, `unit_counter`, `price_tax`, `meal_period`, `vessel_serving`.
3. Merge/deduplicate into **pending-review candidates**, retaining `term`, category, source crop/text span, split, raw/canonical spelling, extraction model/version, review status and reviewer/evidence. LLM extraction alone does not approve a term. Complete terms recovered from a non-OK line require explicit verification before acceptance. Do not silently repair OCR lookalikes (`口/ロ`, `二/ニ`, `〇/0`) in transcription labels.

### Step 4.4: 4-Gram Char-LM Delta Update (Forward & Backward)
- Load the current forward and backward `ImmutableCharNGramLM` artifacts (`checkpoints/l2_v06/char_lm_4g_hp1010.json.gz` and `_bwd.json.gz`) via [`src/recognition/char_lm_beam.py`](../../src/recognition/char_lm_beam.py) (`schema_version=2`, `context_transform="dig_unk_v2"`).
- For both `reverse=False` (forward) and `reverse=True` (backward):
  - Copy existing `uni`, `bi`, `tri`, and `quad` count tables.
  - Feed accepted **training-split** terms only, using the matching candidate vocabulary and the existing schema-v2 context builder (`<DIG>` and OOV boundary rules). Preserve backward reversal/BOS handling; don't hand-edit contexts as if a backward LM were a forward copy.
  - Weight `w=5` reproduces the existing approved-term builder policy, but is **an uncalibrated starting value for a new batch**, not a mandatory or validated optimum. Deduplicate within batch and document repeated terms across batches; do not count extraction duplicates or full-line/subterm overlap as independent real-menu frequency evidence.
  - Record every n-gram (`uni`, `bi`, `tri`, `quad`) whose prior count was `0` and updated count is `> 0` as a **newly added Char-LM dependency**.
  - Construct a new `ImmutableCharNGramLM` with `vocab_chars=ext_vocab["characters"]`, full provenance metadata, and save via `.save_gz(...)` to:
    - `data/processed/recognition/char_lm_4g_hp1010_<batch>.json.gz`
    - `data/processed/recognition/char_lm_4g_hp1010_<batch>_bwd.json.gz`
- **Idempotence:** persist base-LM hash, batch ID, accepted-term hash, split hash, weights/context-transform/version and applied-batch ledger. Reapplying the same input/config returns the same derived artifact/counts or is rejected; it must not silently add counts twice. Existing immutable LM serialization alone does not enforce this workflow.
- Validate candidate versus frozen LM on selection data before use: raw/LM CER, short strings, prices/currency, initially-correct damaged predictions and changed numeric spans. Keep test data report-only. Run compatibility tests with the actual recognition vocabulary; finite log-probabilities (normally non-positive) do not establish quality or new-row training. Preserve R1 short routing and flag any changed amount. Do not overwrite/publish a candidate automatically.

### Step 4.5: Lexicon & Trie Index Update
- Load the current `LexiconIndex` (`outputs/recognition/implementation/lexicon_trie_python_20261119/artifacts/lexicon_index_v1.json.gz`) and base `menu_lexicon.json`.
- For each extracted term, compute `canon = canonicalize_text(term)` via [`src/recognition/lexicon_lookup/normalization.py`](../../src/recognition/lexicon_lookup/normalization.py).
- Store newly extracted terms in a versioned **pending-review ledger** first. The current index builder has no pending-review tier: keep that ledger outside the runtime index. Only explicitly reviewed/accepted training-source terms may be appended to `approved_terms` (`curated_core`, with source/review provenance) in `data/processed/lexicon/menu_lexicon_<batch>.json`. Existing source membership alone does not confer approval on a newly generated term.
- Rebuild the provenance-bearing Trie via `build_lexicon_index(...)` in [`src/recognition/lexicon_lookup/index.py`](../../src/recognition/lexicon_lookup/index.py) using `menu_lexicon_<batch>.json`, `data/processed/lexicon_cleaned_argon/jmdict_food_cleaned.json`, and read-only `data/processed/lexicon/culinary_menu.db`.
- Save to `data/processed/lexicon/lexicon_index_v<N>_<batch>.json.gz`.
- **Translation coverage:** index membership does not guarantee an English translation. Report `found/missing/unlinked`, counts of newly matched but untranslated terms and source/review status. Keep unknown translations explicit. The existing DB is read-only; approved translation additions require a separate versioned DB migration/review process, not writes from lookup or invented English/allergen/dietary metadata.
- Preserve exact-versus-fuzzy classifications, numeric/particle-context protection, original OCR and unknown fallback. Test candidate trie damage as well as coverage; larger dictionaries can introduce wrong neighbors. Fuzzy auto-correction remains off until separately approved.

### Step 4.6: Focus-Training Delta Logs & Verification
Write two synchronized delta logs inside `<output_dir>/`:
1. **`focus_training_delta_<batch>.json`** (machine-readable):
   - `artifacts`: relative paths, SHA-256 digests and before/after sizes for every actual candidate artifact; list unchanged, not-built and incompatible artifacts explicitly rather than assuming exactly five outputs.
   - `provenance`: source checkpoint/state/vocabulary hashes, split hash, canvas/response hashes, accepted/pending term hashes, applied-batch ledger and normalization policy.
   - `eligibility`: training/excluded/review-needed counts, per-menu split membership, OOV/font/geometry/partial-label issues and activation blockers.
   - `summary`: counts of extracted terms, new CTC tokens, new lexicon terms, terms containing new CTC tokens, and forward/backward new Char-LM dependencies.
   - `new_ctc_tokens`: `[{char, id, codepoint, unicode_name, terms_using_token, font_support_count, total_train_fonts, missing_in_fonts}]`.
   - `new_lexicon_terms`: `[{term, canonical_term, category, has_new_ctc_token, new_ctc_tokens, db_id}]`.
   - `new_charlm_dependencies`: `{forward: {...}, backward: {...}}` listing all newly introduced unigrams, bigrams, trigrams, and quadgrams.
   - `all_extracted_terms`: full list with `in_lexicon_v1` and `new_ctc_tokens` flags.
2. **`focus_training_delta_<batch>.log`** (human-readable): concise summary of counts, artifact hashes, new token IDs, new lexicon terms, and newly added n-gram dependencies for quick review and focus-training setup.

#### Mandatory Verification Gate
Before finishing any batch:
1. Recheck response/manifest bijection, leakage, original-label preservation, append-only IDs, classifier/vocab size and bundle hashes. Reload candidate assets; verify correct schema/direction/context/vocabulary linkage, finite LM log-probabilities and lookup provenance. Test repeat application without double-counting. If no compatible trained expanded checkpoint exists, label the vocabulary/LM package **candidate-only, not recognition-ready**.
2. Verify hashes of frozen source assets unchanged. Run existing tests, plus explicit candidate-bundle/response/idempotence/geometry tests where implemented. Passing tests against old frozen assets does **not** establish new-candidate recognition retention:
   ```powershell
   .\.venv-recognition\Scripts\python.exe -m pytest tests/recognition/test_char_lm_beam.py tests/recognition/test_lexicon_lookup.py tests/recognition/test_particle_guard.py -q
   ```
3. Evaluate candidate checkpoints/LM/trie on fixed selection probes and report-only heldout sets with matched input geometry, precision and decoder. Report OCR versus suggested CER separately, numeric/price damage, unknowns and exact/suggested translation coverage. Do not relax gates or change the selected candidate using test labels. Record approval/activation separately; a completed extraction/delta batch is not a model promotion.

---

## 5. Reference Completed Run (`16112026` Manual Tabelog Batch)

Historical artifact inventory below records the earlier **term-derived** extension and its outputs, **not a fresh audit of all new Step4.0 gates**. In particular, the generated1830-character vocabulary and expanded LM/trie do not establish a trained1831-output classifier, reviewed approval of all extracted terms, or deployment in the current inspector. That 51-token extension is incomplete for full crop labels: the read-only audit found116 distinct OOV characters across1087 `OK` labels against the1779-character base; the1830 candidate still misses71 of those characters. Rebuild the character inventory from complete CTC labels in Step4.2, preserving all existing IDs. Review candidate assets under the new gates before activation. The localhost inspector uses a1779-character model/vocabulary bundle unless explicitly changed with compatibility checks.

| Stage | Metric / Result | Output Artifact |
| :--- | :--- | :--- |
| **Input & YOLO `build`** | 31 images (28 `.jpg`, 3 `.webp`) → 1,355 crops / 55 canvases | `data/raw/tabelog_selected_menu_manual/16112026/yolo_crops/` |
| **VLM Labeling & `parse`** | `1,355 / 1,355` slots (`0` missing): `1,087 OK`, `172 TRUNCATED`, `65 UNCERTAIN`, `31 NOISE` | `labeled_crops.jsonl`, `ctc_labels.txt`, `parse_summary.json` |
| **LLM Term Extraction** | 1,077 unique non-empty labels → **904** unique culinary/menu terms | Recorded in `focus_training_delta_16112026.json` |
| **CTC Dictionary** | `1,779 → 1,830` (**+51** tokens at IDs `1780..1830`; `49/51` in all 15 fonts; `渟`/`鱸` in `14/15`) | `data/processed/recognition/vocab_v41_16112026.json` |
| **4-Gram Char-LM (FWD / BWD)** | FWD: `+51` uni, `+201` bi, `+466` tri, `+683` quad; BWD: `+51` uni, `+206` bi, `+471` tri, `+673` quad | `data/processed/recognition/char_lm_4g_hp1010_16112026[_bwd].json.gz` |
| **Lexicon & Trie Index** | `11,740 → 12,358` entries (**+618** new `curated_core` terms) | `data/processed/lexicon/menu_lexicon_16112026.json`, `lexicon_index_v2_16112026.json.gz` |
| **Focus-Training Logs** | Full JSON + human-readable `.log` | `focus_training_delta_16112026.json`, `focus_training_delta_16112026.log` |
