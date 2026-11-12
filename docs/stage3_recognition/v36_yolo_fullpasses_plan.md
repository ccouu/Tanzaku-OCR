# V36: two full epochs with every eligible YOLO crop

## Authorization and handoff

User authorized skipping all pilots and proceeding directly to full-epoch training
after AGY finishes preparing approximately 300 labeled YOLO crops. User will be
away for six hours; no repeated polling is requested. AGY was sent one Herdr
message on 2026-11-12 requesting its labels/manifest paths, source-menu grouping,
verification status and exclusions in a completion message to `w2:p1`.

Current state: **completed two full epochs; no training running**.
Selected checkpoint:rawupdate7000best.pt,all17gates in bothFP16/BF16.
Finalraw/EMA8002failBF16twocharbyoneedit. See[v36_result.md](v36_result.md)
and output `review.md`/`completion_verification.json` for independent completion
audit and recognition analysis. No third epoch or model promotion.
On AGY's completion notification, continue this authorization without asking for
permission again. Inspect the delivered schema and adapt the training reader;
do not assume that guessed fields are authoritative. Send one Herdr completion
or failure notification after the run rather than monitoring in a loop.

## Dataset

Delivered 443 verified eligible crops:405 from45 training menus and38 from five
held-out menus (010/020/030/040/050). Excluded235 of678detections remain excluded.
Actual frozen selection:63799original +31898application clean +31898application
noisy +405YOLO =128000records/pass,4001updates/pass,8002updates total.
All521required exact labels retained. Each YOLO training crop appears once/pass
(0.3164%ofimages), not10%. Source-menu partitions are kept as delivered.

The supplied CTC exports had been stretched to384x64; training instead uses the
native RGB display crops with the established CCW rotation for vertical lines.
Aspect ratio is preserved.32of443eligible crops exceed width384atH64; these fit
within384x64with proportional scaling and vertical edge letterboxing. No glyph
clipping or anisotropic squeeze. Native/normalized crop hashes and exact-source
overlap checks are in sampling.json. AGY visual verification is recorded as AGY's
review, not an independent Codex re-transcription. Representative normalized
horizontal/vertical/wide crops were visually inspected before launch.

Preflight checked all443normalized inputs, same-bucket collation, both complete
epoch plans, required-term coverage and LR endpoints/monotonic decay. The first
launch stopped before GPU updates because serialized source-menu hash ordering
changed between preflight and startup. Sorting menu IDs fixed the metadata
instability; the run was relaunched. No optimizer/training progress was lost.

Output: `outputs/recognition/training/v36_yolo_fullpasses2/`.
Launcher: `outputs/recognition/run_v36.ps1`; stdout/stderr:
`outputs/recognition/v36.stdout.log`, `v36.stderr.log`.
Status: output `status.json` and `outputs/recognition/v36_pipeline_status.json`.
Completion/failure is delivered once through Herdr to `w2:p1`; no polling loop.

- Target 128,000 records per epoch; freeze the selected corpus before training.
- Use every verified, in-vocabulary, training-eligible delivered crop exactly once
  each epoch. Do not repeat 300 crops to force a nominal 10% share.
- Respect existing source-menu splits. If no split exists, reserve approximately
  20% by source menu for evaluation, subject to the actual source diversity.
  All crops and augmented variants from a source menu stay in one partition.
  If independent menus are insufficient, report that limitation explicitly.
- Preserve the existing 38 provisional YOLO evaluation crops as held out. Audit
  crop/image hashes and source-menu provenance against the existing transfer
  benchmark; if overlap exists, exclude the entire matching source menu from
  training. Unverified labels and unreadable/truncated-label mismatches are not
  silently admitted. Report all exclusions and counts.
- Fill the remaining slots from the immutable V34/V35 corpus, retaining its
  approximate 50% original / 25% application clean / 25% application noisy ratio
  among **non-YOLO** records. Use deterministic selection without replacement;
  preserve clean/noisy base-render pairs and all 521 required exact labels.
  This becomes a new frozen selected corpus: a full pass means every selected
  record, not every record in the larger source pool.
- Preserve required-term representatives explicitly before dropping any filler
  records. Audit all 521 exact whole-line labels at corpus construction and
  actual per-epoch encounters, not merely substring coverage.
- Proportionally resize crops to height 64; use next fitting 128/256/384 bucket,
  right-edge replication and exact valid CTC lengths. Do not invent margins,
  rotate without honoring recorded orientation, clip glyphs or silently squeeze
  extreme-width crops. Report extreme widths and choose an explicit handling
  policy based on delivered data before launch.
- Group training batches by physical bucket, at most 32 images; keep remainder
  batches and visit every selected record exactly once per epoch. Compute actual
  updates per epoch from these plans. Do not hardcode 4001 if new bucket counts
  change that total.

## Model and training

- Start from `outputs/recognition/evaluation/v35_checkpoint_averages/average3.pt`
  (uniform steps 2000/3500/4001). Preserve its original file and hash.
- Existing V27 mean-height MobileNetV4/temporal-BN/packed-BiLSTM architecture.
- Keep 147 V30 BN buffers bit-identical throughout raw and EMA models; all BN
  gamma/beta and other learned weights remain trainable. No BN recalibration.
- Fresh AdamW with existing norm/bias no-decay grouping; encoder peak LR 5e-7,
  remaining parameters 3e-6; gradient clipping 5.
- 200-update linear warmup, cosine to 0.10 of peak at the **actual final update**.
  Horizon equals the sum of both complete epoch plans, not the old 24,006-step
  horizon. Two full epochs only; do not extend automatically.
- Keep BF16 training/FP32 CTC and packed recurrence; maintain EMA of learned
  parameters with decay 0.998, initialized from the starting averaged weights.
  EMA buffers stay fixed to V30; do not run update_bn.
- No new augmentation, curated-term sampler, tight-crop anchor slice or model
  modification. Delivered crop normalization is the new adaptation axis.

## Evaluation and reporting

- Evaluate initial raw candidate to reproduce saved average3 predictions before
  interpreting training results; use unchanged V30 FP16/BF16 reference caps.
- Evaluate raw and EMA every 1000 updates and at completed epoch boundaries;
  avoid a redundant evaluation within 100 updates of a boundary.
- FP16 is the main inference comparison. Check retention-eligible promising
  candidates in BF16 as well; require all 17 existing gates in both precisions
  for the preferred robust candidate. Keep source, selected, raw-final and
  EMA-final distinct. Initialize retained best from average3, not from a model
  that fails gates, and never overwrite an earlier experiment's checkpoint.
- Existing recognition composite controls ranking among eligible candidates.
  Report the original clean/noisy menu tradeoff explicitly, not just pass/fail.
- Evaluate independent delivered-menu holdout separately. Do not tune on the
  existing 38 YOLO test crops or use them for training. Report their results as
  an existing diagnostic rather than claiming independent confirmation.
- Track clean training CER, one-/two-char CER and D/I/S, held-out YOLO CER/exact
  match, all existing 15 probe sets/17 gates, and strong appearance.
- Save `run_config.json`, data hashes and split provenance, exact epoch plans,
  `sampling.json`, `metrics.jsonl`, checkpoint/EMA/RNG/optimizer state, BN buffer
  audits and `report.json`. Final report must prove complete selected-corpus
  passes, eligible YOLO encounter counts and 521 exact-term coverage per pass.
- Launch hidden local process with logs and one Herdr completion message. No
  constant polling while user is away. No automatic production promotion.
