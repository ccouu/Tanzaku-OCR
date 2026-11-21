# Stage 3 current implementation and reviewed status

> **2026-11-21 checkpoint update:** [L2_menu1901_digit_recovery](preferred_checkpoint.md) is the descriptive name for preferred **Stage A epoch 3 RAW**, 1901 characters; not gate-eligible or promoted. Registry remains **L2_v06 epoch 2 EMA**; inspector override is **Dishlex50 epoch 5 RAW**, both 1779 characters. See [extended L2 training history](l2_v01_v02_dataset_training_summary.md). Original-lineage sections below are historical snapshots and do not override these roles.

> **2026-11-19 explicit registry promotion:** current recognizer is **L2_v06 epoch2 EMA (H4)**, verified vocabulary and weights. V43 is preserved for rollback. See [promotion details](l2_v06_promotion.md) and [Char-LM validation](char_lm_validation.md). Decoder/R1 patches remain unapplied review candidates; the original-lineage snapshot below does not override this update.

> **Latest Implementation Plan (2026-11-17):** See [docs/implementation_plan.md](../implementation_plan.md) for the current `L2_v06` (`checkpoints/l2_v06/epoch_02.pt` EMA) promoted checkpoint, HotPepper scraping, LLM morpheme/slot extraction, 4-gram Char-LM CTC beam search, and Morpheme Trie lookup plan.

Documentation snapshot: 2026-11-15, through reviewed V76 results and the authorized V77 pilot. A dispatch receipt is not proof that a trainer launched. Consult the experiment's latest audit/callback for runtime state; this page does not poll background jobs.

**2026-11-17 lineage addition:** former H4 V81 is now **L2 V01**, a separate random-start H4 glyph-first lineage with two completed phases. See [its dataset/training record](l2_v01_dataset_training_summary.md) and [the family map](model_lineage.md). The model-role table below covers the original H2 lineage snapshot; it does not rank later H4/L2 candidates. Registry V43 remains unchanged.

## Production and research are separate

| Role | Exact state | Location |
|---|---|---|
| User-promoted recognizer | **L2_v06 epoch2 EMA**, H4, 1,779 characters + blank 0 | `checkpoints/l2_v06/epoch_02.pt`; registry `src/recognition/current_model.json` |
| Rollback | V43 RAW; older V38 retained locally | `checkpoints/menu_recognizer_v43.pt`; previous registry `checkpoints/l2_v06/previous_registry_v43.json` |
| Preferred reviewed YOLO-adaptation reference | **V75 epoch 4 raw**, retention-eligible, **not promoted** | `outputs/recognition/experiments/v75_v74_new_clean_yolo_fullpasses5_colab/downloads/epoch_04.pt` |
| Alternative spacing candidate | V76 Attempt 2, epoch 1 raw, eligible under its declared guards, **not a proven improvement over V75** | `outputs/recognition/experiments/v76_v75_dynamic_spacing_fullpasses5_colab/downloads/epoch_01.pt` |
| Next authorized experiment | V77: two independent matched 500-update arms from V75 raw, standard versus standard/tight synthetic mixture | `outputs/recognition/experiments/v77_v75_tight_gap_matched500_colab/WORK_PLAN.md`; results not reviewed in this snapshot |

V75 SHA-256: `275d530bc4df0ea63ed812e3bf6b4afeda4f8a695267b3d1513673bf3eedd94b`.
V76 selected SHA-256: `215c60668bab8750394d71e9c71305fab8162321a4830685092edbc1c6e0de38`.

Use `src.recognition.current.load_current_recognizer()` for the promoted V43 only. Its default model uses mean height pooling. Research Conv2d checkpoints require `height_pool='conv2d'`, `use_bilstm=True`, the correct state key, class count and matching vocabulary. Historical state keys include `model`, `raw`, `ema`, and `ema_model`; inspect rather than infer from a filename. Not every experiment checkpoint embeds its vocabulary: preserve the run's referenced vocabulary/config payload as well as weights.

## Actual architecture

The implemented recognizer is **MobileNetV4 Conv Small**, not a ViT or a LoRA-only recognizer. RGB H64 inputs produce `B×960×2×T` encoder features, with effective horizontal stride 4.

- Promoted V43 and V53–V59: `mean(dim=2)` followed by `Conv1d(960,256,1)`.
- V60 onward: learned `Conv2d(960,256,kernel_size=(2,1),stride=(2,1))`, then squeeze height. This preserves temporal resolution and adds 245,760 parameters.
- Orientation conditioning follows projection; it does not switch the early CNN filters.
- Two residual depthwise-separable temporal Conv1d blocks, then residual packed BiLSTM, 128 hidden units per direction, then CTC classifier.
- Packed recurrence, CTC loss and decoding each receive `ceil(valid_width/4)`. Packing does not automatically trim the decoder. Canonical parenthesis decoding pools ASCII/full-width probability mass before CTC collapse.

`src/recognition/model.py` supports both height modes and retains mean pooling as its compatibility default. Review verified exact CPU FP32 logits against preserved V73 implementation at widths 128/256/512. Mobile export, accelerator compatibility and device latency are still unvalidated; desktop GPU timing is not a mobile benchmark.

## Vertical data and line geometry

Horizontal input uses flag 0. Genuine native vertical text consists of upright glyphs stacked top-to-bottom, with appropriate vertical punctuation, then rotated **90° counter-clockwise** into a reading strip, flag 1. Rotating a horizontal word is not equivalent. Preserve the flag through renderer, manifest, dataset, collate and model.

V60–V65 had zero native-vertical training examples. V66/V67 adaptation from the horizontal-specialized V65 model did not solve vertical recognition without trade-offs. V68 restarted randomly with 98,000 horizontal and 98,000 vertical records and all parameters/BN statistics updating. Its selected epoch 13 EMA reached 6.12% horizontal and 7.46% vertical CER on the fixed paired synthetic holdout. V69 onward kept both orientations during adaptation.

**Latest required training geometry (V76 Attempt 2 onward):** normalize the native rectified strip proportionally to height 64. Do not shrink the whole line to meet a width limit or introduce width-driven vertical letterboxing. Choose the next 128/256/384/512 bucket and pad right with replicated edge pixels. Source-image margins are distinct from artificial height padding.

If proportional H64 width exceeds 512, split into windows of width at most 512 with 100 H64-normalized pixels of overlap. A 374×34 strip is approximately 704×64, not 512×47 plus height padding. Training windows require correct partial labels and complete glyph coverage; never copy the entire label onto each window. Real training splits used CTC forced alignment; their validity audit is not a substitute for human review. Test/inference boundaries must be **label-independent**, with coordinate-based overlap ownership/merging.

**Implementation caveat:** this policy is implemented in the isolated V76 training/review adapters, not universally in the shared export API. `src/detection/line_crops.py:recognition_input()` and `scripts/yolo_crop_canvas_pipeline.py` still produce width-fitting/height-letterboxed `crops_ctc` for extreme lines. Generic `src/recognition/data.py:preprocess()` can width-compress. `predict_long()` splits only with explicit `mode='overlap512'`; its default is `wide768`. Rebuild from native `crops_display`, not already downscaled CTC exports, for the corrected path. See [crop pipeline guide](../data/yolo_crop_canvas_pipeline.md).

## Recent adaptation results: keep cohorts and geometry fixed

These are canonical greedy FP32 CERs from reused development cohorts with provisional VLM labels, not fresh blind production scores.

| Checkpoint | Old test CER, 88 crops | New test CER, 75 OK crops | Fixed 2,400-image H/V CER | Interpretation |
|---|---:|---:|---:|---|
| V74 epoch 2 raw | 35.30% | 40.18% before new-data training | 6.69% | First two-pass YOLO adaptation; baseline for the new collection |
| V75 epoch 4 raw | 34.43% | 34.82% | 7.12% | Preferred reviewed adaptation reference; old column uses legacy CTC exports |
| V75B epoch 5 raw | 34.96% | 42.86% | 7.11% | BN statistics updated; no retention-eligible checkpoint |
| V76 Attempt 2 epoch 1 raw | 34.09% | 33.93% | 7.28% | Selected after five passes; legacy-export evaluation, modest uncertain gains |

**Matched corrected native-H64/split inference:** V75 and V76 both scored **34.09%** on old test, exact 33/88 versus 32/88. New test scored 34.82% versus 33.93%, exact 28/75 versus 29/75. Only two old-test lines were oversized. V76 is therefore not a demonstrated overall upgrade. Its fixed minimum-4px spacing probe worsened from 23.13% to 24.38% CER; broader/random spacing did better. This motivates V77's matched short pilot, not automatic longer training.

The 1,920-image paired short probe and 2,400-image all-length native H/V probe are different benchmarks. “Balanced H/V” in an earlier handover may refer to either; use explicit cohort name, image/character counts, orientation, decoding and precision. One-character and two-character breakdowns must not be inferred from their aggregate.

## Data, splits and vocabulary coverage

- Old collection: `data/processed/gemini_banana_mobile_menus_50_crops/`, despite the name, covers menus 001–130. V74 used 1,561 training crops/117 menus, 68 validation crops/6 menus, and 88 test crops/7 menus.
- New collection: `data/raw/gemini_banana_mobile_menus_train_50/yolo_crops/`: 740 proposals, 702 OK labels, 615 in-vocabulary OK labels, 492 eligible training crops/40 menus; 5 validation and 5 test menus. New evaluation retains all 66/75 OK val/test references and separately reports in-vocabulary subsets.
- Namespace menu and crop IDs by **collection**, not basename: new `menu-001` is not old `menu-001`. Keep source menus and image variants in a single split; audit image hashes and menu provenance.
- V75: 2,053 real parents + 6,159 synthetic records = 8,212/epoch, five passes, 41,060 presentations. V76 corrected geometry: 2,086 real windows from those parents + 6,258 synthetic = 8,344/epoch, five passes, 41,720 presentations. Parent counts are not window counts.
- V72/V73 actually covered **475/521** required standalone labels. The earlier count of 12 was an audit bug, but the 46-label gap was real. Four labels were entirely absent: 海胆、軟骨、雲丹、馬刺. V74 onward asserts all 521 exact full-line canonical labels and actual sampler encounters. Substring occurrences do not meet standalone coverage requirements. Each priority term receives at least two horizontal and two native-vertical views per pass, with font/appearance variation.
- Current 1,779-character vocabulary lacks **8** characters from the new OK references: comma (77 canonical occurrences), 熊/欧/宵/ぁ/贅 (2 each), 『/』 (1 each). Total 89 occurrences across 87 crops; comma is the only OOV in 76 crops. The initial inventory's comma count 78 was corrected by canonical-label recount.
- Removing numeric thousands-separator commas was discussed, **not implemented or authorized**. Vocabulary IDs remain unchanged. Keep original references; OOV evaluation is not in-vocabulary accuracy and OOV labels must not train as blank tokens.
- NOISE/TRUNCATED/UNCERTAIN status alone is not a human verification. A faint/unreadable text crop is not automatically background. Confirmed text-free proposals can become detector hard negatives; do not train them as nonempty CTC labels.

## Training policy and refactor gate

Random V60/V68 training updated all parameters and BN statistics. Current adaptation policy trains CNN/head/BN scale and bias but freezes all **147 BN buffers across 49 layers**, for raw and EMA states. V75B's updating-BN experiment failed retention. Use explicit module policies; “frozen BN” does not mean frozen encoder weights.

V74–V77 adaptation uses encoder LR `2e-7`, other modules `1e-6`, fresh AdamW, matrix-weight decay 0.01 and zero normalization/bias decay, gradient clip 1, batch 32, BF16 forward/FP32 CTC, EMA 0.998 and the declared actual-update warmup/cosine horizon. Historical V72/V73 code instead used weight decay `1e-4` on all groups and clip 5; record actual-versus-intended policy rather than claiming exact reproduction.

The reusable Strategy-pattern refactor is **not cleared for long training**. The 56-test review still found a failing trainer import (`IMAGENET_MEAN`/`STD`), incomplete resume, stale-job ownership, late checksum publication/incomplete retrieval, frozen-EMA drift and missing matched-plan enforcement. Model compatibility passed; trainer/runner readiness did not. No reviewed fix acknowledgement exists in this snapshot. See [review findings](../../outputs/recognition/review/refactor_v68_v73_review/REVIEW.md) and [training skill](../../.agents/skills/recognition-training-pipeline/SKILL.md). Reuse existing validated experiment functionality through narrow configuration/adapters under an approved plan; do not duplicate trainers or launch the blocked core blindly.

## Review and source records

Selected V75 old/new test review: `outputs/recognition/review/v75_old_new_test/index.html`, 163 crops. Its `server.json` records a temporary localhost URL; verify availability rather than treating its port as permanent. Exported corrections are review evidence, not automatically applied training labels. Keep these test results tagged as already inspected development data.

Run roots and their `HANDOVER.md`, `evaluation/*report.json`, source SHA/state keys, corpus/render/encounter audits and downloaded epoch files are authoritative. V76's `post_completion_audit/` contains matched native geometry and spacing results. V77 has no reviewed result here. No documentation edit promotes a checkpoint, authorizes training, or changes a test split.
