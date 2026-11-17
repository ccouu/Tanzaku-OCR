> **Historical proposal; not current execution instructions.** Its V44-running/V38-promoted claims, sizes, “safe delete” lists and open questions are obsolete snapshots. Production is V43; current research V75/V76, pending V77, both YOLO collections, all source/vocab/split/checkpoint/geometry/coverage audits and human-review exports are protected dependencies. Do not delete rendered images or archived assets based only on a version number. Only the separately audited V46 cleanup is recorded as executed here; any new cleanup needs a fresh dependency inventory and its own authorization. See [current recognition status](stage3_recognition/current_status.md).

# Project Cleanup, Code Archiving, and Data Pruning Plan

> Historical proposal; not executed wholesale. On2026-11-13, after V46 completed,
> a narrower audited cleanup removed26,015disposablefiles/12.5094GiB. Active datasets,
> all non-cache checkpoints, rawmenus, annotations, fonts, source and environments
> were preserved; unproven duplicates were kept. See
> [actual cleanup execution](data_cleanup_execution_v46.md) and
> [V46 review](stage3_recognition/v46_focused_recurrent_training.md).

## Goal Description

A full audit of [docs/](.) and the `D:\profolio\japanese_translation` workspace (~42.8 GB across ~570,000 files) shows that most disk usage and folder clutter comes from:
1. **Unpruned caches & pytest temp runs** (`.cache/test-runs` at **10.84 GB** and `.cache/pip-recognition` at **1.91 GB**).
2. **Heavy binary checkpoints/images left inside the already-retired archive** (`archive/stage3_legacy_20261111/outputs/` at **7.00 GB**, even after dataset images were deleted on 2026-11-11).
3. **Superseded Stage 2 & Stage 3 smoke runs, failed layouts, and obsolete pilot outputs** (`outputs/recognition/` at **11.58 GB**, `outputs/detection/` at **1.05 GB**).
4. **Duplicate or superseded processed/snapshot datasets** (`data/processed/gemini_recognition/` at **197.54 MB** duplicating `gemini_banana_mobile_menus_50_crops/`, `data/datasets/detection/reviewed_v1/` at **331.43 MB**).
5. **One-off migration tools and superseded experiment scripts** (`tools/`, `scripts/recognition/` V38–V42, `scripts/stage2_yolo/` one-off prep/finetune scripts).

This plan defines a **tiered, reversible-first cleanup** that:
- Reclaims **~23–32 GB** of disk space safely.
- Archives unused/completed experiment scripts and one-time migration tools cleanly under `archive/`.
- **Strictly protects** the currently running **V44 pilot**, the promoted **V38 recognizer**, the active **V27/V41/V43** reference checkpoints, active **Stage 2 detectors (`colab_v2`, `banana50_finetune_v3`, `banana130_finetune_v5`)**, all raw/reviewed datasets, and all code/test invariants.

---

## User Review Required

> [!CAUTION]
> **Active Training Guard — V44 Pilot Is Currently Running**
> Per [`outputs/recognition/pilots/v44_latin_parentheses_long_2000/status.json`](../outputs/recognition/pilots/v44_latin_parentheses_long_2000/status.json), V44 is actively running (`state: "evaluating"`, `update: 1000`).
> **Do NOT modify or delete** `v44_latin_parentheses_long_2000/`, `v43_digits_prices_2000/final_raw.pt`, `data/processed/gemini_banana_mobile_menus_50_crops/`, `data/datasets/recognition/`, or `.venv-recognition/` while V44 is in progress.

> [!IMPORTANT]
> **Decisions Needed Before Execution**
> Please confirm your preference on the optional/medium-impact items in **Open Questions** below (especially whether to delete `.pt`/`.bin`/`.png` binaries inside `archive/stage3_legacy_20261111/outputs/` and whether to prune superseded V29–V42 training image caches/checkpoints in `outputs/recognition/`).

---

## Open Questions

1. **Archive Binary Pruning (`archive/stage3_legacy_20261111/outputs/` — 7.00 GB):**
   - Should we delete the **6.39 GB** of retired V1–V28 `.pt` (`4.33 GB`), `.bin` (`1.74 GB`), and `.png` (`309 MB`) files inside `archive/stage3_legacy_20261111/outputs/` (keeping `src/`, `docs/`, `manifest.json`, `v27_parity.json`, and JSON/MD/CSV reports), OR delete `archive/stage3_legacy_20261111/` completely, OR leave it untouched?
2. **Superseded Recognition Pilots (`outputs/recognition/` — 11.58 GB):**
   - **Tier A (Safe, ~1.48 GB):** Delete all `*_smoke*` and `v31_curated_short_failed_layout` directories.
   - **Tier B (Optional, ~5.5–7.5 GB):** For superseded full runs (`v29`, `v30`, `v31`, `v32`, `v33`, `v34`, `v35`, `v36`, `v37`, `v39`, `v40`, `v42`), should we delete their rendered `.png` / per-sample `.json` caches (`~4.4 GB`) and non-promoted `.pt` checkpoints while keeping `analysis.md`, `report.json`, `review.md`, `metrics.jsonl`, and `final_raw.pt`/`best.pt` for V27/V38/V41/V43?
3. **Legacy Compatibility Shims (`scripts/stage2_yolo/` & `scripts/prelabel_real_menus.py` / `merge_real_dataset.py`):**
   - `tests/` still imports `scripts.stage2_yolo.*`, `scripts.prelabel_real_menus`, and `scripts.merge_real_dataset`. Should we update `tests/detection/` and `tests/common/` to import `src.detection.*` directly so the `scripts/stage2_yolo/` shims can be archived, or keep those compatibility shims in place?

---

## Current Workspace Inventory (Audited)

| Directory | Files | Size | Status & Notes |
|---|---:|---:|---|
| `.cache/` | 18,338 | **12.76 GB** | `test-runs` (10.84 GB) + `pip-recognition` (1.91 GB) — pure scratch/wheel cache |
| `outputs/` | 217,271 | **12.65 GB** | `recognition/` (11.58 GB), `detection/` (1.05 GB), `reorganization/` (28.3 MB) |
| `archive/` | 46,100 | **7.22 GB** | `stage3_legacy_20261111/outputs/` holds 7.00 GB of retired V1–V28 `.pt`/`.bin`/`.png` |
| `data/` | 177,066 | **4.96 GB** | Mostly active raw/datasets/annotations; contains duplicate `gemini_recognition/` & `reviewed_v1/` |
| `.venv-recognition/` | 16,578 | **2.98 GB** | Active Python environment — **KEEP** |
| `.stage2_packages/` | 2,502 | **212.76 MB** | Stage 2 bootstrap packages (documented in `docs/reorganization.md`) — **KEEP** |
| `checkpoints/` | 4 | **15.60 MB** | Promoted `menu_recognizer_v38.pt` & `yolo11n_menu_obb_best.pt` — **KEEP** |
| `scripts/`, `src/`, `tests/`, `docs/`, `tools/` | 324 | **2.67 MB** | Source, tests, docs, active + historical scripts |

---

## Proposed Changes

### 1. Delete Caches & Temporary Scratch Data (~12.76 GB Reclaimed)

#### [DELETE] `.cache/test-runs/` (10.84 GB, 18,281 files)
- Leftover temporary directories from past pytest runs (`128 MB` and `2.81 MB` per UUID folder). Safe to delete completely.

#### [DELETE] `.cache/pip-recognition/` (1.91 GB, 21 files)
- Cached pip wheels already installed into `.venv-recognition/`. Safe to delete completely.

#### [DELETE] `.cache/ultralytics_latest_finetune/`, `.cache/matplotlib_latest_finetune/`, `.cache/ultralytics/` (~2.06 MB)
- Stale one-off detector training cache folders. Keep `.cache/huggingface-recognition/` (14.52 MB) and `.cache/recognition_bundle_v2/` (0.2 MB) if needed by offline recognition runs.

---

### 2. Prune Obsolete & Duplicate Data (`data/` — ~529 MB+ Reclaimed)

#### [DELETE] `data/processed/gemini_recognition/` (197.54 MB, 3,950 files)
- Exact duplicate of `data/processed/gemini_banana_mobile_menus_50_crops/` (`3,950` files, `197.54 MB`), with **zero** references anywhere in `src/`, `scripts/`, `tests/`, or `docs/`.

#### [DELETE] `data/datasets/detection/reviewed_v1/` (331.43 MB, 4,857 files)
- Frozen Stage 2 v1 snapshot whose Colab session failed at epoch 52 with no recovered checkpoint (`docs/stage2_detection/stage2_summary.md`). Fully superseded by `data/datasets/detection/reviewed_v2/`.

#### [KEEP / PROTECTED] Active Data
- `data/raw/` (`genai_seeds`, `gemini_banana_mobile_menus_50`, `tabelog_menus`, `tabelog_interiors`, `tabelog_test_100`)
- `data/resources/` (`fonts`, `fonts_v26`)
- `data/processed/gemini_banana_mobile_menus_50_crops/`, `synthetic_base_v1/`, `augmentation/`, `lexicon/`, `recognition/`
- `data/annotations/` (`menu_obb/`, `gemini_banana_*`)
- `data/datasets/recognition/` (`v27_fonts/`, `v27_validation/`)
- `data/datasets/detection/` (`reviewed_v2/`, `banana130_v5/`, and optionally `banana50_latest_v3/` + `banana51_100_v4/` unless user wants `v4` deleted as well — `v4` is 45.21 MB and was rejected vs `v3`).

---

### 3. Prune Unused Outputs (`outputs/` — ~1.53 GB Safe + ~5–7 GB Optional)

#### [DELETE] Detection Unused Outputs (~145 MB)
- `outputs/detection/colab_v1/` (empty failed run)
- `outputs/detection/preflight_snapshots/banana51_100_horizontal_only_holdout/` (45.14 MB — documented in `docs/stage2_detection/banana51_100_finetune.md` as an unused preflight snapshot)
- `outputs/detection/canvas_pipeline_v5_smoke/` (6.50 MB — smoke test output)
- *(Optional, 93.91 MB)* `outputs/detection/training/banana51_100_finetune_v4/` (rejected v4 candidate; keep `review.md`/`evaluation_report.json` or delete weights).

#### [DELETE] Recognition Smoke & Failed-Layout Outputs (~1.48 GB)
- `outputs/recognition/pilots/v29_application_mixture_smoke/` (73.44 MB)
- `outputs/recognition/pilots/v30_short_retention_smoke/`, `v30_short_retention_smoke_final/` (146.28 MB), `v30_short_retention_smoke_verified/` (1.20 MB)
- `outputs/recognition/pilots/v31_routing_smoke/` (202.51 MB), `v31_parent_preflight_smoke/` (202.51 MB), `v31_reconciled_smoke/` (202.50 MB), `v31_curated_short_failed_layout/` (58.97 MB)
- `outputs/recognition/pilots/v32_height_smoke/` (0.24 MB), `v32_height_smoke_v2/` (45.71 MB)
- `outputs/recognition/pilots/v33_tiered_smoke/` (202.47 MB)
- `outputs/recognition/pilots/v34_fullpass_smoke/` (104.44 MB)
- `outputs/recognition/pilots/v35_frozen_bn_smoke/` (71.64 MB), `v35_resume_smoke/` (72.63 MB)

#### [OPTIONAL PRUNE] Superseded Full Recognition Runs (`outputs/recognition/`)
For completed/superseded runs (`v29_application_mixture_2000`, `v30_short_retention_2000`, `v31_curated_short_2000`, `v32_height_collapse_2000`, `v33_curated50_tiered_2000`, `v34_broad_fullpasses6`, `v35_frozen_bn_*`, `v36_yolo_fullpasses2`, `v37_long512_2000`, `v39_v38_yolo_bn_swap`, `v40_new_characters_2000`, `v42_yolo35_jitter_long512_2000`), remove heavy `.pt` checkpoints and rendered `.png`/per-sample `.json` caches while preserving lightweight `.md`, `report.json`, `metrics.jsonl`, and `review.md` reports.
- **Strictly Protected:** `v44_latin_parentheses_long_2000/` (active), `v43_digits_prices_2000/`, `v41_new_rows_only_2000/`, `v38_yolo25_fullpasses4/`, `v27_font_full20/`, and `outputs/recognition/review/`.

---

### 4. Prune Retired Legacy Archive (`archive/stage3_legacy_20261111/` — ~6.39 GB Optional)

#### [DELETE] Binary Artifacts in `archive/stage3_legacy_20261111/outputs/`
- Remove `.pt` (`4.33 GB`), `.bin` (`1.74 GB`), and `.png` (`309 MB`) files inside `archive/stage3_legacy_20261111/outputs/` — matching the precedent set on 2026-11-11 when archived dataset images/ZIPs were removed while retaining `src/`, `docs/`, `manifest.json`, `v27_parity.json`, and JSON/MD/CSV reports.

---

### 5. Archive Unused Code & One-Time Scripts

Create a structured archive folder `archive/code_legacy_20261112/` (mirroring `archive/stage3_legacy_20261111/` conventions):

#### [ARCHIVE] One-Time Reorganization Tools (`tools/`)
- Move `tools/refactor_layout.py`, `tools/update_layout_docs.py`, `tools/reorganize_workspace.py`, `tools/verify_layout.py` → `archive/code_legacy_20261112/tools/`. (`docs/reorganization.md` explicitly states `refactor_layout.py` and `update_layout_docs.py` must not be rerun.)

#### [ARCHIVE] Superseded Stage 3 Pilot Scripts (`scripts/recognition/`)
- Move completed/superseded V38–V42 experiment scripts (`promote_v38.py`, `train_v38_yolo25.py`, `review_v38_yolo25.py`, `run_v38_yolo25.ps1`, `pilot_v39_bn_swap.py`, `prepare_v40.py`, `train_v40_new_characters.py`, `review_v40.py`, `run_v40.ps1`, `prepare_v41.py`, `train_v41_new_rows.py`, `review_v41.py`, `run_v41.ps1`, `prepare_v42.py`, `train_v42_yolo35.py`, `review_v42.py`, `run_v42.ps1`) → `archive/code_legacy_20261112/scripts/recognition/`.
- Keep active: `prepare_v43.py`/`train_v43_digits.py`/`run_v43.ps1` (reference for V44), `prepare_v44.py`/`train_v44_adaptation.py`/`run_v44.ps1`, and `build_failure_review.py`.

#### [ARCHIVE] One-Off Stage 2 Banana Prep/Finetune Scripts (`scripts/stage2_yolo/`)
- Move one-off historical batch scripts (`prepare_banana_review.py`, `prepare_banana_batch2_review.py`, `prepare_banana100_rereview.py`, `prepare_review_range.py`, `finetune_banana_latest.py`, `finetune_banana_batch2.py`, `run_banana_finetune.ps1`, `run_banana_batch2_finetune.ps1`, `compare_v3_v5.py`, `compare_crop_boundaries.py`, `build_crop_review.py`) → `archive/code_legacy_20261112/scripts/stage2_yolo/`.
- Keep `finetune_all130.py` (V5) and the `src.detection.*` compatibility shims (unless user approves updating `tests/` imports in Open Question #3).

#### [KEEP] Active Scripts
- `scripts/yolo_crop_canvas_pipeline.py`
- `scripts/download_fonts.py`
- `scripts/data/*` (Stage 1 augmentation, scraping, verification)
- `scripts/prelabel_real_menus.py`, `scripts/merge_real_dataset.py` (retained because `tests/data/test_synthetic_renderer.py` imports `export_annotation` and `validate_real` from them, unless tests are refactored).

---

## Verification Plan

### Automated Tests
After executing the approved cleanup (and **after** V44 completes if any shared folders were touched):
1. Run the full test suite to verify zero broken imports or missing fixtures:
   ```powershell
   .\.venv-recognition\Scripts\python.exe -m pytest tests -q
   ```
2. Verify active Stage 1, Stage 2, and Stage 3 CLI entrypoints:
   ```powershell
   .\.venv-recognition\Scripts\python.exe -m src.recognition.train --help
   .\.venv-recognition\Scripts\python.exe scripts/yolo_crop_canvas_pipeline.py --help
   .\.venv-recognition\Scripts\python.exe -c "from src.recognition.current import load_current_recognizer; m, ck = load_current_recognizer('cpu'); print('OK', len(ck['vocabulary']))"
   ```
3. Report exact disk space reclaimed (before vs. after) and record archived file paths in a concise `archive/code_legacy_20261112/README.md`.
