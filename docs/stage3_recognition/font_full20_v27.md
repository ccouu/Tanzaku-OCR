# V27 — 20 full passes of the 15-font training corpus

This is the original completed-run report. The active code and dataset paths
were subsequently reorganized without changing model weights or input pixels;
see the [current V27 guide](README.md). The original smoke run and earlier
experiments now reside in the [archive](../../archive/stage3_legacy_20261111/README.md).
The completed run's embedded configurations and source.zip remain unchanged.

## Completed result and interpretation

All20 epochs completed:40,020 updates,1,280,000 presentations,64,000 training records covered exactly once per epoch. The run took about92 minutes excluding initial setup/validation. Source V26 checkpoint unchanged; BN counters advance by40,020. Epoch20 is preserved in `last.pt`.

Matched beta0 results:

| Metric | Starting V26 clean | V27 epoch20 |
|---|---:|---:|
|Seen-font validation CER|7.30%|4.48%|
|Heldout-font validation CER|7.73%|5.27%|
|Original-clean retention subset CER|5.90%|4.70%|
|Historical synthetic crops, bucketed CER|12.13%|11.27%|
|Original-clean one-character subset CER (100 images)|27.00%|18.00%|
|Original-clean two-character subset CER (100 images)|21.00%|12.00%|
|Provisional GenAI YOLO subset CER (38 crops)|93.21%|96.38%|

Original-family macro CER improves6.75%→5.00%, and new-family macro CER7.78%→4.03%. Heldout-font exact lines rise270→304/400. These are validation improvements; the heldout families were used in selection and do not constitute an untouched test set. The short-text figures are the fixed retention subset, not the earlier full3,444-image one-character benchmark.

The declared composite font score improves7.42%→4.77%. All later candidate checkpoints fail at least the provisional YOLO retention gate. At epoch20 the other three gates pass; YOLO CER exceeds its94.21% cap. Hence `best.pt` correctly retains epoch0 (starting V26), while `last.pt` contains the genuinely trained epoch20 model. No criterion has been changed after observing these results.

The YOLO rejection corresponds to206→213 errors across221 reference characters: seven more errors, with deletions133→158, substitutions71→55 and insertions2→0. Both checkpoints have zero exact labelled menu lines. The criterion rejection should not be mistaken for absence of font/short-text learning; it exposes the unresolved target-domain failure on a small provisional sample.

Strong optical distortion remains difficult. The50-image monitoring half changes63.70%→64.44% CER; the50-image half reserved from this run changes71.72%→68.97%. Combined, the100 existing strong-appearance images change67.86%→66.79% (six fewer edits). This is a modest improvement, not transfer readiness. The images were previously used in diagnostics and are not a new independent test set.

Validation is unstable at several epochs, with the composite score spiking at4,6,12,16–19 despite the eventual improvement. Homogeneous-width batches with normal BN are one plausible source of sensitivity, but the current run does not isolate BN from weight/optimization effects. More epochs alone are not established as the remedy.

Assessment: successful consolidation of clean-font and short-text recognition, unsuccessful GenAI-menu transfer. Preserve epoch20 as the next **font-recognition candidate** and keep V26 as the baseline. Before another long run, test the complete short-text benchmark and a small target-domain adaptation arm with clean-font replay. If checkpoint selection is revised later, explicitly separate font recognition from provisional target-domain acceptance rather than silently relabelling epoch20 as having passed the original gates.

Starts from the V26 clean checkpoint at update2,000. Every one of the64,000 labelled generated training records appears exactly once in each of20 epochs:1,280,000 presentations, no oversampling, no dropped partial batches. The600 seen-font and400 heldout-font validation records remain excluded from training. Auxiliary augmented masks are alternate views of training records, not extra training records.

Training uses the existing15 fonts,32,000 original-family and32,000 new-family records per epoch, and the existing5/45/35/15% length distribution. Model inputs use only nearest128/256/384 buckets; all one-character inputs use128. Partial batches stay in their own bucket instead of being promoted. The separate fixed384 jitter stream is removed. Stored384-wide alpha masks are cropped before composing the actual bucket-sized model input.

Per epoch,95% of views use clean masks and5% use the tested mild morphology:2.5% dilation,2.5% erosion,3×3 cross kernel,25% operated-mask blend. The chosen records change deterministically each epoch. Every record still appears only once. Glyph bounds and valid lengths follow the selected mask; only background guards can shrink to fit, never glyphs.

Half the records retain canonical margins; half vary left/right margin allocation inside the same assigned bucket. Left placement is sampled from4px to the available space after retaining the right guard (or the smaller available guard on an almost-full canvas). Bucket assignment is not enlarged by margin jitter. Background fills are constant native border color. Valid sequence lengths include left margin, glyphs and right guard, excluding remaining artificial right fill. The packed BiLSTM remains enabled; no internal CNN masking is introduced.

Fresh AdamW, full unfreeze and normal training BN. Encoder LR starts5e-6 and head/BiLSTM LR3e-5. The first two epochs retain those rates, then cosine decay reaches10% of the initial rates in epoch20; no epoch uses zero LR. Batch size is up to32.

## Evaluation and checkpoint selection

Every epoch evaluates beta0 and beta0.2, with beta0 primary. Metrics retain all per-length CERs, including separate one- and two-character results.

The primary score equally averages original-font, added-font and heldout-font macro CER. A candidate must also satisfy each retention cap: no more than **1.0 percentage point absolute** regression relative to the starting V26 clean checkpoint on:

- Original seven-font macro CER.
- A fixed969-image original-clean subset processed through nearest buckets.
- The1,000 historical synthetic guarded crops processed through nearest buckets.
- The38 provisionally labelled GenAI YOLO crops with audited border-color bucket padding.

The last set is a small, non-random, provisional crop diagnostic, not whole-page or independent real-menu CER. Its gate is conservative; the starting checkpoint remains eligible if later epochs fail retention. All20 epochs are still executed and `last.pt` preserves the endpoint independently of `best.pt`.

The strong-appearance probe is split by transcription group:25 texts/50 images for epoch monitoring, the other25 texts/50 images evaluated only after training on baseline/best/last. Neither subset is used for checkpoint selection or training. These images were evaluated in earlier experiments, so the final half is reserved from this run rather than claimed as an untouched external test set.

## Artifacts and recovery

Main directory: `outputs/recognition/training/v27_font_full20/`.

- `status.json`: current epoch, step and coverage progress.
- `metrics.jsonl`: all epoch summaries together, including per-length metrics, retention gates and learning rates.
- `epoch_XX_sampling.json`: exact full-pass plan, cohort/width/morphology counts and hash.
- `epoch_XX.json`: detailed predictions and metrics.
- `best.pt`, `last.pt`: model, optimizer, RNG and configuration; last embeds its historical best for consistent recovery.
- `best_validation.json`, `completion.json`, `reserved_strong_final.json`.
- Recovery checkpoints every five epochs and a source-code archive.

Launcher: `outputs/recognition/run_v27.ps1`. Logs: `outputs/recognition/v27_console.log`, `v27_stderr.log`. The launcher permits at most two retries; an interrupted epoch is replayed from the last completed boundary with its optimizer/RNG restored. Configuration, code and dataset hashes must match. Sampling plans and actual consumed IDs independently check every record exactly once.

A bounded two-epoch GPU smoke test verifies input shapes, losses, full coverage, retention selection, checkpoint recovery across an epoch boundary and final audits. Smoke output is separate: `outputs/recognition/training/v27_full20_smoke/`. The smoke uses640 stratified records, not the main64,000-record corpus.

Training progresses through local files and the background supervisor. No continuous assistant polling is required.
