# Stage3 compact handoff

Workspace:`D:\profolio\japanese_translation`;PowerShell;Python environment
`.venv-recognition\Scripts\python.exe`;local RTX3070. No active experiment or
training is pending. V31 and its inference diagnostics completed. No production
checkpoint or decoder was promoted.

## Code and model

- Active reusable code:`src/recognition/`,V27 architecture:MobileNetV4 ConvSmall,
  horizontal stride4,960→256 projection,orientation conditioning,two residual
  depthwise temporal blocks with BN,packed residual BiLSTM128/direction,CTC
  1,723 classes (blank0 +1,722 characters).
- H64,next fitting128/256/384 bucket,right-edge replication;packed recurrence
  excludes artificial padding via valid lengths. CNN/temporal BN is not strictly
  padding-invariant. Standard greedy CTC at beta0 remains active.
- Pilots/evaluators:`outputs/recognition/*.py`. Long training is not running.

## Checkpoints: do not confuse best.pt with the trained candidate

| Purpose | Local path |
|---|---|
| V27 trained reference |`outputs/recognition/training/v27_font_full20/epoch_20.pt`|
| V27 best.pt |V26 initialization;no trained epoch passed every gate|
| Recent adaptation reference |`outputs/recognition/pilots/v30_short_retention_2000/control/best.pt` (update2000)|
| Latest trained candidate |`outputs/recognition/pilots/v31_curated_short_2000/curated_70/step_2000.pt`|
| V31 curated best.pt |Update0,V30 source weights;curated arm failed retention|
| V31 control best.pt |`outputs/recognition/pilots/v31_curated_short_2000/control/best.pt` (update500)|

V31 curated improves menu-probe clean1/clean2/noisy1/noisy2 CER to
29/34.5/33.5/39.5%,versus matched control32.5/36.75/39.5/41.75%. It uses the
50%original/25%clean-render/25%noisy-render mixture,with70%curated labels only
within generated short examples. Overall length quota stays unchanged.
Both final arms made61/200 original one-character errors;cap60,baseline58.
V30's earlier20/20 short oversampling did not beat its control.

## Latest decoder finding

Frozen V31 curated step2000,3,434 one-character images,35 decoder modes:

| CER | Standard | Empty-only fallback |
|---|---:|---:|
| Menu clean1 |29%|24%|
| Menu noisy1 |33.5%|29%|
| Original200 |30.5%|28%|

Fallback changes only an entirely empty greedy output:choose one nonblank token
at the valid frame with highest nonblank softmax probability. Keep other outputs
and temporal CTC blanks unchanged. No dictionary or known-length oracle used.
It is evaluated only,not installed. Need matched V30 decoding,multi-character
and detector false-positive tests before adoption. Yen remains weak when noisy.

## Dictionary and data

- Main fuzzy dictionary:`data/processed/lexicon/culinary_menu.db`,47,216 entries;
  `src/data/lexicon_builder.py:CulinaryLookup`. Not applied in reported raw CER.
- Main database has menu-sense selection issues (e.g.ピザ,中生,大福);not repaired.
- Short review:`outputs/recognition/curation/v31_short_terms/`;Gemini Argon via
  Herdr plus parent review. Approved496 terms (114one/382two),485 agreements plus
  11 sourced essentials.3,474 uncertain;1,407 vocabulary-gap entries retained.
  Read `VALIDATION.md`;raw Gemini summary has corrected arithmetic/count notes.
- Active datasets:`data/datasets/recognition/v27_fonts/` and `v27_validation/`.
  Fonts/shared licenses remain in `data/resources/fonts/` and `fonts_v26/`.
- Archive:`archive/stage3_legacy_20261111/`,V1–V28 history/source/reports. Archived
  images and ZIPs were deleted (7.08GiB);labels/metadata remain. Active data intact.

## Git and workflow

- Remote:`https://github.com/ccouu/Tanzaku-OCR.git`,branch `main`.
- Source and selected reports only. Never commit datasets,curated training lists,
  weights,databases,generated images,logs,credentials or environments.
- Root `/data/` is ignored;`src/data/` and `scripts/data/` must remain tracked.
- Last functional check:15 recognition tests passed. Documentation updates do not
  change training weights or decoding. User prefers autonomous execution and
  minimal polling/token use for background jobs.
- Preserve raw CER versus post-lexicon accuracy. Existing probes are development
  diagnostics;real-handwriting accuracy and mobile latency remain unestablished.

For complete current metrics and limitations,read [stage3_summary.md](stage3_summary.md).
