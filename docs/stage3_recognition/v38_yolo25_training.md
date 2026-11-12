# V38: four complete YOLO-crop passes at 25%

Authorized adaptation run from V36's retention-selected `best.pt`. V37's
trained weights were not promoted and are not used.

Source: `data/processed/gemini_banana_mobile_menus_50_crops` (historical folder
name; actual images cover menus 001–130). Of 1,894 proposed crops, 1,717 have
VLM status OK; 246 contain out-of-vocabulary characters and are excluded without
rewriting labels. The resulting 1,471 eligible crops split into 1,336 training
and 135 validation crops. NOISE, TRUNCATED and UNCERTAIN crops are excluded.

Holdout menus: 010,020,030,040,050,060,070,080,086,100,109,120,130. Training and
holdout menus and exact crop pixel hashes do not cross partitions. Holdout is
used for checkpoint selection, not an independent final test. Crop labels were
produced by a visual LLM; this run does not establish independent human
verification. Existing overlap flags remain available and OK labels are trusted
provisionally.

Each of four passes contains every eligible training crop once plus 4,008
synthetic images: 2,004 original, 1,002 application-clean and 1,002
application-noisy. Total: 21,376 presentations / 5,344 YOLO presentations
(exactly 25%) / 686 updates. These are full YOLO passes, not full passes of the
128,000-image synthetic source corpus. Each selected record appears once within
a pass; synthetic records are reselected between passes, protecting all 521
required exact terms once per pass. Every partial width/domain batch is kept.

Delivered RGB crops stay unchanged at H64 and physical widths 128/256/384/512.
Valid CTC lengths come from valid image width; backward recurrence stays packed.
No extra crop distortion, margin jitter or glyph scaling is introduced.

Architecture unchanged. Encoder LR 5e-7, head LR 3e-6, 200 warmup updates,
cosine decay over the actual 686-update horizon to factor .1. All 147 V30 BN
buffers fixed; convolutional/recurrent weights and BN affine parameters train.
EMA .998. Logical batch up to32, training microbatch4 with sample-weighted
gradient accumulation to bound activation memory. Frozen BN statistics make
microbatch accumulation compatible with the logical mean CTC objective.

Validate initial predictions against saved V36 predictions in FP16 and BF16.
Evaluate raw and EMA after each pass. Selection requires all17 existing gates
in both precisions, and improves `0.7 * current YOLO holdout CER + 0.3 * existing
application/short-text composite`. Initial V36 is the fallback. Report YOLO
train/holdout separately, plus existing synthetic, short, historical and
appearance probes. No automatic production promotion.

Entry: `scripts/recognition/train_v38_yolo25.py`; launcher:
`scripts/recognition/run_v38_yolo25.ps1`. Output:
`outputs/recognition/training/v38_yolo25_fullpasses4`. `sampling.json` and four
saved epoch plans audit the counts and label provenance. `metrics.jsonl` records
all scheduled epoch evaluations. `best.pt` is selected; `last.pt`/`final_raw.pt`
and `final_ema.pt` preserve final weights. Existing source checkpoints stay intact.

The launcher sends one completion/failure message through Herdr; it does not
poll the training process repeatedly.

## Completed result

Finished in approximately19minutes:686updates, four audited passes, exact25%YOLO.
All521 required terms per pass and147fixed BN buffers verified. All98BNaffine
tensors updated. Initial V36 predictions reproduce in both precisions.

FP16 current YOLO train CER43.02%→33.62%; holdout51.53%→41.19%, exact lines
14→24/135. Holdout improves on12/13menus; paired menu-bootstrap95%interval for
the CER change is[-14.33,-6.81]pp. Final EMA holdout45.56%. Final raw has fewer
deletions199→156/substitutions138→102, but more insertions17→25.

No checkpoint passed all17gates in both precisions. Final raw passes FP16 but
BF16 two-character has56/400edits against54cap. `best.pt` is unchanged V36
weights at update0; final raw/EMA are research candidates, not promoted models.
Horizontal holdout improvement is much larger than vertical;512bucket has
only6heldout lines and no aggregate gain. Strong-appearance CER worsens.

Independent audit and breakdowns: output `review.md`,
`completion_verification.json`, `paired_analysis.json`. No new training or
inference was run in the review. Suggested next test is a cheap V36/V38 weight
blend against unchanged gates before further optimization.

## User-authorized promotion

The user subsequently prioritized YOLO-crop gains for the intended application
and explicitly authorized promotion with looser retention. V38 final raw at
update686 is now promoted as `checkpoints/menu_recognizer_v38.pt`. Registry:
`src/recognition/current_model.json`; checksum-verifying loader:
`src/recognition/current.py`. The compact artifact contains model/vocabulary/
normalization/config and promotion provenance, not optimizer recovery state.

Policy `menu_adaptation_v2` raises only pooled_original_2char cap by0.005
absolute CER:13.5%→14.0%,54→56edits/400. All other16caps stay unchanged; both
FP16 and BF16 must pass. Final raw passes and has the lowest existing selection
score among saved candidates eligible under the revised policy. No new
optimization or GPU inference was needed. Promotion was validated by strict
state loading, tensor equality to final_raw.pt, checksum, frozen BN buffers and
a CPU forward pass. This policy was adopted after observing V38 results.

The original run's `report.json`, metrics, best.pt and source archive are not
rewritten. They accurately reflect the original policy and V36 fallback.
`promotion.json` and `retention_policy_v2_candidates.json` record the later
decision separately. V36 remains available for rollback. Future invocations of
the V38 trainer apply the versioned policy and record it in run_config; existing
completed runs cannot silently resume with altered configuration.
