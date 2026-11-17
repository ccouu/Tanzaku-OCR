> Historical snapshot preserved by the documentation audit. “Current/latest/running” statements below apply only to the earlier snapshot. Use [current recognition status](../stage3_recognition/current_status.md) now.

# Stage 3 — V43 recognizer, V27 architecture

For direct AGY delegation and Colab execution without Sol6, read the
[AGY Colab workflow guide](../agy_colab_training_workflow.md).

Dataset, purpose and weight-setup reports:
[V53–V59](../stage3_recognition/v53_v59_dataset_summary.md) and
[V60–V75](../stage3_recognition/v60_v75_dataset_training_summary.md).

**Current promoted recognizer: V43 final raw, update2000**, explicitly promoted
2026-11-13 for YOLO-menu recognition. Weights:
`checkpoints/menu_recognizer_v43.pt`. Registry:
`src/recognition/current_model.json`. Load with
`from src.recognition.current import load_current_recognizer` and
`model, checkpoint = load_current_recognizer('cpu')` (or `'cuda:0'`).
The loader verifies the checkpoint checksum and returns an eval-mode model.

**V52 beam/lexicon diagnostic completed:** V43 normalized YOLO CER41.50%→40.77%
withbeam25+lightlexicon;V48 39.73%→39.00%withbeam10. V43 opening brackets remain
0/13. Beam25 oracle still29.82%/28.68%CER, so decoding alone is insufficient.
No production changes. See [V52 results](../stage3_recognition/v52_beam_lexicon_diagnostic.md).

**V51 resolution diagnostic completed:** H64/stride2 and H96/stride4 severely
regress on YOLO and synthetic probes with frozen weights. V43 YOLO CER rises
41.50% → 86.13% / 100%; native-image and FP32 sanity checks agree. Keep H64/stride4.
No training or promotion. See [V51 results](../stage3_recognition/v51_resolution_diagnostic.md).

**V50 completed:** both matched normalized-parenthesis arms ran 2,000 updates.
BiLSTM adaptation improves clean bracket CER 59.88% → 47.46%, but new-Latin
recall falls 33.69% → 23.25% and 1-char CER worsens 33% → 37%. Its YOLO gain
is inconclusive; no BiLSTM checkpoint passed selection. Control best is raw1,500.
V43 remains promoted; no training is running. See [V50 results](../stage3_recognition/v50_normalized_bilstm_matched.md).

**LatestcompletedV49:**2,000updates,only52Latin+14digit/bracketrowschanged;
BiLSTM/allsharedfeatures/remainingrows/blank/BNfixed. MultifontLatinrecall44.68%,
drivenbyexisting19letters;new33recall33.75%stillweak. Digitrecall86.72%vs87.72%
source,YOLOCER42.02%vs41.61%,realbrackethitsunchanged3/26. Selectedbest=raw1500,
source-relativeguardspassbut11absolutegatesfail. V49 and V50 are complete.
See [V49results](../stage3_recognition/v49_selective_head_pilot.md).

**Previous completed tests:V47/V48.** V47fits208cleanLatinimagesat1.05%trainCER,
butgeneralizationisweak(42.5%samefontnewlayout/67.31%heldfontCER)andretention
collapses. V48finalrawYOLOCER39.94%vsV43's41.61%,withdigitretentionfailure.
V48selectedbestisEMA1000,notfinal:41.19%YOLO,source-relativeguardspassbut
absolutegatesfail. KeepV43promoted;notrainingrunning.
See [V47/V48 results](../stage3_recognition/v47_v48_isolated_tests.md).

**Latest completed V46:**4,000updates/128,000presentations in44minutes, exact
30%Latin/10%parentheses/15%digits/5%basicJapanese/20%YOLO/20%syntheticreplay.
CNN/temporal/BN stayed frozen. Parenthesissymbolrecall23.05%→44.68%, but Latin
heldoutrecall23.88%remainsweak;digits/basicJapaneseregress. YOLOCER41.61%→41.29%
raw,confidenceintervalcrosseszero;exact50→48/156. No selectedbest or promotion.
Cleanupreclaimed12.51GiB;activedata/modelsverifiedpreserved. No trainingrunning.
See [V46 results](../stage3_recognition/v46_focused_recurrent_training.md).

V38 is preserved at `checkpoints/menu_recognizer_v38.pt` for rollback. Matched
156-crop YOLO CER improves51.82%→41.61%, exact24→50; historical and synthetic
short-text accuracy regress. V43 fails11existing retention gates in both precisions.
This specific promotion has an explicit user-authorized override; retention
thresholds are unchanged. See [promotion decision](../stage3_recognition/v43_promotion.md).

**Previous experiment: V44**, 2,000 updates from **V43 final raw**.
Execution and append-only Latin coverage were verified, but the recipe was not
successful overall: heldout-font Latin CER remains99.58%, parentheses-symbol recall
23.05%→23.74%, and YOLO CER41.61%→41.50%is essentially unchanged. Extra-long768
CER improves66.33%→48.17%, while historical CER worsens13.88%→17.53%.
No trained checkpoint passes retention; `best.pt` is prediction-preserving expanded
V43 at update0. V44 was not promoted. See [V44 results](../stage3_recognition/v44_latin_parentheses_long_pilot.md).

**Previous completed pilot: V45 Stage 1A**, 1,000 updates from
V43 final raw, training only the 33 appended Latin classifier rows, with exact
40% Latin / 40% synthetic / 20% YOLO replay. All old weights, BN affine and 147
buffers remained fixed. Final Latin training CER74.34%, heldout macro recall23.92%,
YOLO CER42.34%; feasibility targets missed and no source-relative retention-qualified
checkpoint. No Stage 1B or further training is running.
See [V45 results](../stage3_recognition/v45_latin_head_only_pilot.md).

**Previous completed experiment: V43**, 2,000 updates / 64,000 presentations.
YOLO holdout CER improved **45.46% → 41.61%** and heldout-font digit macro recall
**69.28% → 87.72%**, but historical and short-text retention regressed. No trained
candidate passed all gates. `best.pt` is the unchanged V42 fallback, not trained
V43; the V44 source is explicitly `final_raw.pt`.
See [V43 audited results](../stage3_recognition/v43_digit_price_pilot.md).

Vocabulary sizes are checkpoint-specific: rollback V38 has 1,722 characters;
promoted V43 has 1,779; V44/V45 candidates have 1,812. Each also has blank ID 0.
V44 covers all ASCII A–Z/a–z; added rows are appended without reordering old IDs.
Full-width decimal digits are normalized to ASCII in V43/V44 training labels.
Use the vocabulary embedded in the checkpoint; this is separate from the fuzzy
menu dictionary at `data/processed/lexicon/culinary_menu.db`.

V41 completed: corrected blank-relative initialization passed exact parity;
only57newrows trained, with50%oldreplay and frozen-teacher consistency. All old
weights/BNaffine/buffers remained identical. SelectedEMA1000passesbothprecision
gates; finalrawnewcharheldfontrecall85.67% but noYOLOholdoutCERgain. See
[V41 plan/results](../stage3_recognition/v41_new_rows_only_plan.md); V41 was not promoted.

V40 expanded-vocabulary training completed from V38:57newcharacters,
2000updates,actual>=532encounters pernewcharacter. The versioned candidate
dictionary is `data/processed/recognition/vocab_v40.json`; do not pair that
1779-character vocabulary with the old1722-character V38 classifier. See
[V40 plan/results](../stage3_recognition/v40_new_characters_plan.md). V40 was not promoted: synthetic
retention regressed materially.

Promotion was explicitly authorized after observing V38's YOLO holdout
improvement51.53%→41.19%CER. Retention policy `menu_adaptation_v2` increases only
the pooled two-character cap13.5%→14.0%(54→56edits/400); the other16gates and
both-precision checks remain unchanged. This is a retrospective acceptance
decision, not a claim of passing the original gates. See
[V38 run and promotion](../stage3_recognition/v38_yolo25_training.md). All source checkpoints and the
original V38 `best.pt` (V36 fallback) remain unchanged; use the promoted path
above, not that historical `best.pt`.

The reusable implementation is V27. V29–V31 experiment scripts and selected
reports live under `outputs/recognition/`;their architecture is unchanged.
See the [current experiment summary](../stage3_recognition/stage3_summary.md) and
[compact handoff](../stage3_recognition/HANDOFF.md) for recent checkpoints and decoding results.
The historical [V33 reviewed-label plan](../stage3_recognition/v33_curated50_plan.md) records the
AGY source audit and gated long training proposal.
The [V34 full-pass run](../stage3_recognition/v34_fullpasses_plan.md) is the subsequently authorized
broad-label continuation with minimum521-term coverage.
Earlier versions and the V28
experiment are preserved under
[archive/stage3_legacy_20261111](../../archive/stage3_legacy_20261111/README.md).

## V27 reference model

Use [epoch_20.pt](../../outputs/recognition/training/v27_font_full20/epoch_20.pt)
for the trained V27 font-recognition candidate. `last.pt` has the same model
tensors plus recovery state. **best.pt retains the V26 initialization** because
no later epoch passed all agreed retention gates. No V27 checkpoint was promoted
at that time; V38 above is the current promoted recognizer.

Seen-font CER4.48%, held-out-font CER5.27%; provisional GenAI-menu crop CER96.38%.
The menu probe has only38 provisional labels and is not an independent
real-handwriting or end-to-end benchmark.

## Active layout

| Location | Purpose |
|---|---|
| [src/recognition](../../src/recognition/) | V27 model, datasets, training, metrics and checkpoint helpers |
| [v27_fonts](../../data/datasets/recognition/v27_fonts/) | Original64,000-record training corpus plus seen/held-out font validation; pixels and labels unchanged |
| [v27_validation](../../data/datasets/recognition/v27_validation/) | Standalone clean/historical retention data and frozen transfer probes |
| [v27_font_full20](../../outputs/recognition/training/v27_font_full20/) | Completed20-epoch checkpoints, logs and metrics |
| [tests/recognition](../../tests/recognition/) | Current V27 tests |

The15 font assets and their licenses remain in `data/resources/fonts/` and
`data/resources/fonts_v26/`; these shared assets are also used by Stage1.
Their existing names do not imply a dependency on archived experiment code.

## Commands

From the repository root, using the existing recognition environment:

```powershell
.\.venv-recognition\Scripts\python.exe -m src.recognition.train --help
.\.venv-recognition\Scripts\python.exe -m pytest tests/recognition -q
```

`python -m src.recognition.font_training` remains supported. `train.py` is now
a small entry point for that V27 trainer. To reproduce the V27 recipe, use a
fresh output directory and the preserved `initial.pt`:

```powershell
.\.venv-recognition\Scripts\python.exe -m src.recognition.train --epochs 20 --output outputs/recognition/training/v27_new_run
```

This command trains; it was **not executed during cleanup**. Default weights
are `v27_font_full20/initial.pt`, the exact starting checkpoint used for V27.
For continuation, explicitly pass `--checkpoint .../epoch_20.pt` and use a
fresh output directory. The trainer retains atomic epoch-boundary recovery
for runs created with the current code. The completed original run is preserved
as an immutable historical record; its embedded code/config hashes predate
this reorganization and must not be rewritten or bypassed to resume it.

## Code map

- `model.py`: MobileNetV4 encoder, temporal convolutions, packed residual BiLSTM, CTC head.
- `data.py`: normalization, collation, valid lengths and greedy CTC decode.
- `datasets.py`: font masks and guarded retention views,128/256/384 routing.
- `font_training.py`: full-pass sampling,5% morphology,50% margin variation,
  evaluation, retention gates and recovery.
- `optimization.py`: AdamW groups and separate encoder/head learning rates.
- `metrics.py`: edit alignments, CER summaries and additive blank penalty.
- `utils.py`: checksums, seeding, device checks, FP32 CTC loss and guarded margins.
- `checkpointing.py`: CPU snapshots and atomic saves.
- `vocabulary.py`: append-only vocabulary expansion and separately optimized new rows.
- `label_normalization.py`: full-width decimal digit → ASCII label normalization.
- `long_line_inference.py`: 768px routing and coordinate-based overlapping CTC decode.
- `review/`: crop-failure review UI; decisions are saved separately from labels.

Experiment launchers and preparation code live in `scripts/recognition/`.
The latest crop review uses **V42 final raw**, not V43/V44:
`http://127.0.0.1:7867/`, with its frozen queue under
`outputs/recognition/review/v42_training_failures/`. A server URL describes an
existing local launch, not a guarantee that the process is still serving.

See the [Stage3 summary](../stage3_recognition/stage3_summary.md),
[original completed run report](../stage3_recognition/font_full20_v27.md), and
[cleanup manifest](../../archive/stage3_legacy_20261111/manifest.json).
