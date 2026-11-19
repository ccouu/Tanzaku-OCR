# Stage 3 — recognition implementation, status and experiment records

Start with [current implementation/status](current_status.md), [compact handoff](HANDOFF.md), and [experiment summary](stage3_summary.md). Historical results below describe their own datasets and policies, not the currently deployed model.

**Promoted (2026-11-19): L2_v06 epoch2 EMA**, H4 architecture, at `checkpoints/l2_v06/epoch_02.pt`, resolved by `src/recognition/current_model.json`. V43 remains rollback. See the [promotion record](l2_v06_promotion.md). Original V75/V76/V77 notes below describe an earlier research lineage, not the current model ranking.

**New independent lineage (2026-11-17): [L2 V01](l2_v01_dataset_training_summary.md), formerly H4 V81.** Both renamed phase folders and all ten epoch hashes are verified. This random-start glyph-first model is separate from the original V75/V76/V77 and H4 V75 families. The preceding research-reference paragraph is the earlier original-lineage snapshot; use the [lineage guide](model_lineage.md) for later branches. No new promotion.

The promoted V43 uses mean height pooling. V60 onward uses learned Conv2d height collapse with the same MobileNetV4/temporal/residual BiLSTM128 architecture and horizontal stride 4. Native vertical data and corrected H64 splitting are covered in the status guide. The shared crop exporter still letterboxes extreme lines; do not assume the V76 correction is universal.

The Strategy refactor's model passed parity, but its trainer/runner remain under a [request-changes review](../../outputs/recognition/review/refactor_v68_v73_review/REVIEW.md). Passing helper tests is not a Colab-launch clearance.

## Current guides and dataset reports

- [L2_v06 registry promotion](l2_v06_promotion.md) and [Char-LM validation / unapplied candidate fixes](char_lm_validation.md).
- **[Model family and checkpoint lineage guide](model_lineage.md)** — start here to distinguish original versions, H4 branches, random starts and sibling experiments.
- [L2 V01 datasets, two-phase training and checkpoint results](l2_v01_dataset_training_summary.md) — current name for former H4 V81.
- [V53–V59 datasets](v53_v59_dataset_summary.md)
- [V60–V75 datasets, source weights and BN settings](v60_v75_dataset_training_summary.md)
- [V76–V77 crop geometry and spacing experiments](v76_v77_dataset_training_summary.md)
- [AGY/Colab operating guide](../agy_colab_training_workflow.md)
- [YOLO/VLM crop export and preprocessing caveats](../data/yolo_crop_canvas_pipeline.md)

## Historical experiment records

| Versions | Records |
|---|---|
| V27 | [20 full passes](font_full20_v27.md) |
| V33–V34 | [Curated pilot](v33_curated50_plan.md), [full passes](v34_fullpasses_plan.md) |
| V35 | [Frozen BN](v35_frozen_bn_plan.md), [short-text inspection](v35_short_inspection.md), [averaging](v35_checkpoint_averages.md) |
| V36 | [Plan](v36_yolo_fullpasses_plan.md), [result](v36_result.md) |
| V37 | [Long512 plan](v37_long512_plan.md), [result](v37_result.md) |
| V38–V42 | [V38](v38_yolo25_training.md), [V40](v40_new_characters_plan.md), [V41](v41_new_rows_only_plan.md), [V42](v42_yolo35_crop_jitter_plan.md) |
| V43 | [Digit/price run](v43_digit_price_pilot.md), [explicit promotion](v43_promotion.md) |
| V44–V46 | [V44](v44_latin_parentheses_long_pilot.md), [V45](v45_latin_head_only_pilot.md), [V46](v46_focused_recurrent_training.md) |
| V47–V50 | [V47/V48](v47_v48_isolated_tests.md), [V49](v49_selective_head_pilot.md), [V50](v50_normalized_bilstm_matched.md) |
| V51–V52 | [Resolution](v51_resolution_diagnostic.md), [beam/lexicon](v52_beam_lexicon_diagnostic.md) |

Old reports may reference retired assets, old validation caps or source fallbacks. Preserve their recorded numbers; use current status and each new run's immutable configuration for continuation. Mobile export and independently human-verified production evaluation remain unfinished.
