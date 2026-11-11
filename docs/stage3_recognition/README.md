# Stage 3 — active V27 recognizer

The reusable implementation is V27. V29–V31 experiment scripts and selected
reports live under `outputs/recognition/`;their architecture is unchanged.
See the [current experiment summary](stage3_summary.md) and
[compact handoff](HANDOFF.md) for recent checkpoints and decoding results.
Earlier versions and the V28
experiment are preserved under
[archive/stage3_legacy_20261111](../../archive/stage3_legacy_20261111/README.md).

## V27 reference model

Use [epoch_20.pt](../../outputs/recognition/training/v27_font_full20/epoch_20.pt)
for the trained V27 font-recognition candidate. `last.pt` has the same model
tensors plus recovery state. **best.pt retains the V26 initialization** because
no later epoch passed all agreed retention gates. No checkpoint was promoted
as a production recognizer.

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
