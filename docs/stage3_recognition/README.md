# Stage 3 — V38 recognizer, V27 architecture

**Current promoted recognizer: V38 final raw, update686.** Weights:
`checkpoints/menu_recognizer_v38.pt`. Registry:
`src/recognition/current_model.json`. Load with
`from src.recognition.current import load_current_recognizer` and
`model, checkpoint = load_current_recognizer('cpu')` (or `'cuda:0'`).
The loader verifies the checkpoint checksum and returns an eval-mode model.

Latest V42 pilot completed:35%geometricallyjitteredYOLO/30%genuine512longlines.
Menuholdout51.82→45.46%CER andheldfontlong5126.64→1.88%,but historical/short
retention worsened; no trained checkpoint passed all revised gates. V38 stays
promoted. See [V42 plan/results](v42_yolo35_crop_jitter_plan.md).

V41 completed: corrected blank-relative initialization passed exact parity;
only57newrows trained, with50%oldreplay and frozen-teacher consistency. All old
weights/BNaffine/buffers remained identical. SelectedEMA1000passesbothprecision
gates; finalrawnewcharheldfontrecall85.67% but noYOLOholdoutCERgain. See
[V41 plan/results](v41_new_rows_only_plan.md); promoted model remains V38.

V40 expanded-vocabulary training completed from V38:57newcharacters,
2000updates,actual>=532encounters pernewcharacter. The versioned candidate
dictionary is `data/processed/recognition/vocab_v40.json`; do not pair that
1779-character vocabulary with the old1722-character V38 classifier. See
[V40 plan/results](v40_new_characters_plan.md). V40 was not promoted: synthetic
retention regressed materially. V38 remains the active model.

Promotion was explicitly authorized after observing V38's YOLO holdout
improvement51.53%→41.19%CER. Retention policy `menu_adaptation_v2` increases only
the pooled two-character cap13.5%→14.0%(54→56edits/400); the other16gates and
both-precision checks remain unchanged. This is a retrospective acceptance
decision, not a claim of passing the original gates. See
[V38 run and promotion](v38_yolo25_training.md). All source checkpoints and the
original V38 `best.pt` (V36 fallback) remain unchanged; use the promoted path
above, not that historical `best.pt`.

The reusable implementation is V27. V29–V31 experiment scripts and selected
reports live under `outputs/recognition/`;their architecture is unchanged.
See the [current experiment summary](stage3_summary.md) and
[compact handoff](HANDOFF.md) for recent checkpoints and decoding results.
The [V33 reviewed-label plan](v33_curated50_plan.md) covers the current pilot,
AGY source audit and gated long training proposal.
The [V34 full-pass run](v34_fullpasses_plan.md) is the subsequently authorized
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

See the [Stage3 summary](stage3_summary.md),
[original completed run report](font_full20_v27.md), and
[cleanup manifest](../../archive/stage3_legacy_20261111/manifest.json).
