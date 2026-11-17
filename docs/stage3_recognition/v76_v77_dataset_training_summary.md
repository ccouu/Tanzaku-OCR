# V76–V77: crop geometry and synthetic character spacing

This follows the consolidated [V60–V75 dataset and training history](v60_v75_dataset_training_summary.md). The former V66–V77 overview duplicated that history and was replaced by this focused report. Both experiments start from **V75 selected epoch 4 RAW**, SHA-256 `275d530bc4df0ea63ed812e3bf6b4afeda4f8a695267b3d1513673bf3eedd94b`. Neither was promoted.

## Shared architecture and training policy

- V60 architecture: learned Conv2d height collapse, width stride 4, two temporal convolution blocks, one residual packed BiLSTM with 128 hidden units per direction, and 1,779 characters plus blank 0.
- All learnable parameters and BN affine train. All **147 BN buffers remain fixed** to V75 in RAW and EMA.
- AdamW: encoder `2e-7`, others `1e-6`; weight decay 0.01 on matrix/convolution weights, zero on normalization/bias; gradient clipping 1. Batch 32, BF16 forward / FP32 CTC, EMA 0.998.
- Training uses 25% real YOLO windows and 75% synthetic replay, with native horizontal and vertical examples. Synthetic replay retains readable procedural menu backgrounds, glyph/font variation and mild augmentation. All 521 required exact terms are covered.

## V76: dynamic spacing with corrected long-line geometry

**Purpose:** vary character spacing while preserving glyph shape, and correct real-crop preprocessing so overwide lines do not acquire artificial height padding.

### Dataset and geometry

The original 2,053 real training parents contain 33 overwide old-collection crops. After aspect-preserving normalization to height 64:

| Parent group | Result |
|---|---|
| 2,020 parents fitting width 512 | 2,020 windows |
| 33 overwide parents | 66 windows with 100-pixel overlap |
| Total | **2,086 real windows**, no height letterboxing |

Partial training labels for the split real windows use CTC Viterbi forced alignment. This reported full character coverage for all 33 parents; it does not independently prove every split boundary is visually perfect. Evaluation uses label-independent split geometry and overlap merging, never ground-truth alignment.

Each epoch contains **2,086 real + 6,258 synthetic = 8,344 presentations**. Five full epochs completed: **41,720 presentations / 1,315 updates**.

Synthetic multi-character ink gaps are sampled independently from integer `Uniform(4, Gmax)` at final H64, where `Gmax` fits the complete line within its bucket. One-character inputs have no inter-character gap. Measured gaps ranged 4–130 pixels, mean 8.92. Glyphs are not clipped or horizontally squashed to fit.

**Attempt distinction:** Attempt 1 reused historical height-letterboxed CTC exports and was safely stopped. The completed Attempt 2 restarted from V75 with a fresh optimizer and corrected native-H64 windows. Failed-attempt logs were preserved; completed results do not describe Attempt 1.

### Results

Selected **epoch 1 RAW**, eligible; SHA prefix `215c60668bab`. Five epochs were completed, but the selected state is not the final checkpoint.

| Native-H64 evaluation, FP32 CER | V75 source | V76 selected |
|---|---:|---:|
| Pooled validation, 134 crops | 40.97% | 40.58% |
| Old test, 88 crops | 34.09% | 34.09% |
| New test, 75 crops | 34.82% | 33.93% |

These are modest changes. On the 192-image spacing diagnostic, uniformly crowded 4-pixel gaps remained difficult: CER **23.13% → 24.38%**, concentrated in long lines and vertical price duplication. Standard/random/max-fit modes were much easier. This motivated V77 rather than another broad V76 extension.

## V77: matched standard versus 4–8-pixel gap mixture

**Purpose:** isolate a limited spacing mixture from ordinary continuation training. Both arms independently start from V75; V76 is not their source.

| Setting | Control | Treatment |
|---|---|---|
| Updates / presentations | 500 / 16,000 | 500 / 16,000 |
| Real YOLO windows | 4,000, covering all 2,086 windows | Identical |
| Synthetic standard views | 12,000 | 6,000 |
| Synthetic 4–8-pixel gap views | 0 | 6,000, multi-character only |

Base labels, fonts, glyph sizes, non-spacing appearance and batch plan are matched. The shared standard half was pixel-identical in all 6,000 pairs. Tight-view orientation is balanced: 3,000 horizontal / 3,000 vertical. All 521 terms were included; RAW/EMA BN audits passed.

**Important interpretation:** 4–8 pixels was not universally tighter than standard font spacing. The audit measured 24,867 widened gaps versus 4,315 narrowed gaps; many natural gaps were below 4 pixels. This is a controlled absolute-gap mixture, not a pure crowding treatment.

### Results and checkpoint distinction

Selected **treatment step 500 EMA**, eligible, not promoted. Its checkpoint file is `downloads/treatment/step_0500.pt`; explicitly select EMA weights when loading it.

| Native-H64 evaluation, FP32 CER | V75 source | V77 selected EMA |
|---|---:|---:|
| Pooled validation | 40.97% | 40.18% |
| Old test | 34.09% | 34.61% |
| New test | 34.82% | 34.23% |
| Fixed 2,400-image synthetic H/V holdout | 7.12% | 7.41% |

On the new held-out-font tight-spacing probe, step-500 **RAW** treatment versus RAW control improved by **0.83 percentage points**, but its 95% interval crossed zero: **[−2.18, +0.42 pp]**, bootstrap fraction favoring treatment 86.45%. Selected **EMA** treatment and control effectively tied. Do not attribute the RAW effect size to the selected EMA.

For menus 101–130, treatment step 500 RAW had the lowest reported CER among these V77 candidates, **16.72%**, but failed broader retention. Selected treatment EMA scored **16.96%**, versus V75 **17.20%** under the same native-H64 evaluation. This collection includes 367 training crops and only 37 held-out crops; it is not a blind generalization benchmark.

The later 885-crop / 60-menu evaluation (`evaluation/v75_v77_train60_yolo/summary.json`) found V75 **23.57%** versus V77 **23.73%** CER. The menu-clustered interval for V77 minus V75 was **[−0.18, +0.50 pp]**: no clear gain. That summary does not record the V77 RAW/EMA state, so resolve it from evaluator metadata before treating it as a comparison to selected EMA.

## Interpretation and sources

V75 remains a useful adaptation baseline. V76 and V77 are eligible alternatives with small tradeoffs, not demonstrated generalization upgrades. The corrected native-H64 splitting path is an isolated experiment implementation; the shared crop exporter is not automatically corrected. See [current status](current_status.md).

Authoritative artifacts under `outputs/recognition/experiments/`:

- `v76_v75_dynamic_spacing_fullpasses5_colab/HANDOVER.md`, `evaluation/v76_report.json` and `post_completion_audit/`.
- `v77_v75_tight_gap_matched500_colab/HANDOVER.md`, `evaluation/v77_report.json`, `manifests/` and both arms' `downloads/*/train_audit.json`.

Menu labels remain provisional VLM transcriptions. Development probe reuse, precision and preprocessing differences must be disclosed. These experiments did not measure mobile inference latency or establish production readiness.
