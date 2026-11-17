> Historical experiment record. Status, sources, vocabulary, geometry and gates below belong to this version; they are not current deployment instructions. See [current implementation/status](current_status.md) for V75/V76, V77 authorization and production V43. Referenced cached assets may have been retired; do not regenerate or overwrite them merely to follow this record.

# V51: forward-only resolution diagnostic

## Completed result

Completed both checkpoints and all three arms in95.90seconds, evaluating5,697
probe records per checkpoint per arm plus a separate156-record native-YOLO
comparison. No training occurred. All model parameters and BN buffers remained
bit-identical. All reported CERs below use pooled canonical parentheses.

| Checkpoint / cohort | H64 stride4 | H64 stride2 | H96 stride4 |
|---|---:|---:|---:|
| V43 YOLO156 | 41.50% | 86.13% | 100.00% |
| V43 original synthetic969 | 6.28% | 84.32% | 99.98% |
| V43 clean short400 | 31.33% | 103.00% | 100.17% |
| V43 held-out brackets120 | 63.92% | 76.65% | 100.00% |
| V43 held-out digits500 | 13.73% | 53.45% | 100.00% |
| V50 control selected YOLO156 | 41.71% | 88.63% | 100.00% |
| V50 control selected held-out Latin416 | 62.27% | 94.15% | 100.00% |

Stride2 increases aligned YOLO opening-bracket hits from0/13to2/13(V43)and3/13
(V50control), but whole-line CER deteriorates drastically. This cannot be called
a recognition improvement; alignment of a bracket in otherwise incorrect text
does not establish correct localization. V43 clean1char CER48.50%→122.00%,
2char22.75%→93.50%. CER can exceed100% because insertions count as edits.
V43 original synthetic CER delta is+78.03pp, paired line-bootstrap CI
[+76.59,+79.59]; YOLO delta+44.63pp,13-menu CI[+36.32,+53.08].

H96 collapses almost entirely to blank for both checkpoints. Resizing from native
YOLO crops, rather than enlarging H64 inputs, does not rescue it. NativeH64 tensors
match all156stored CTC tensors exactly in both checkpoint evaluations. The small
FP32 sanity file verifies finite logits and the same blank collapse on a YOLO,
synthetic bracket and digit example; mean blank probabilities are approximately
99.90%,99.88%,99.69%. Restoring baseline geometry restores FP32 logits exactly.
This is not explained by an FP16 numerical failure or an incorrect H64 native
reconstruction. Input lengths were recomputed for each geometry, not reused.

Interpretation: these weights are strongly calibrated to their trained spatial
and temporal geometry. Increased resolution changes encoder features, frozen-BN
operating distributions, temporal receptive fields and recurrent sequence scale.
The experiment does not isolate which module dominates the failure and does not
prove that a separately trained higher-resolution model would be worse.

Recommendation: keep production H64/stride4 and V43. Prioritize the reviewed
qualifier-size/spacing data intervention first. Any stride2/H96 architecture
pilot needs matched training and retention evaluation; it is not a safe direct
inference toggle. Full records and paired intervals are in `report.json`;
FP32 restoration checks are in `fp32_sanity.json`. No promotion or training
was launched after this diagnostic.

## Protocol

Compare the same frozen weights and BN statistics in three modes:

1. Baseline: H64, effective horizontal stride4.
2. Stride2: H64, change `encoder.blocks.0.0.conv.stride` from `(2,2)` to `(2,1)`;
   recompute valid lengths as `ceil(valid_width/2)`.
3. H96: original stride4, scale both physical width and valid width by1.5,
   preserve aspect ratio and recompute `ceil(valid_width/4)`.

Checkpoints: promoted V43 and V50 frozen-BiLSTM control selected raw1500. The
latter supplies complete Latin coverage; V43 does not contain33Latin letters.
Reference text remains unchanged for scoring. OOV target tensor placeholders are
unused: no loss, training, classifier expansion or vocabulary migration occurs.

Probe cohorts: original and historical crops, clean/noisy short Japanese,
clean/noisy application images, strong appearance, known/held-out fonts for
clean parentheses, digits and Latin, plus the156-crop menu-disjoint YOLO holdout.
All arms evaluate identical records with literal and pooled-parenthesis decoding.
Per-line D/I/S, opening/closing/digit recall and1/2char CER are retained.

The matched H96 synthetic test enlarges existing H64 images. It tests sampling
and distribution sensitivity, and cannot recover information lost before H64.
A separately named `yolo_native` cohort reconstructs the same crop at H64/H96
directly from native display images, using production aspect fit and CCW rotation
for vertical crops. Its H64 tensors are checked against stored CTC inputs.
No changes to detector boxes, labels or split. The reviewed holdout is development
data, not a fresh blind test.

FP16 CUDA forward timings are desktop diagnostics, not mobile benchmarks. Both
geometry changes are outside the trained distribution: a negative zero-update
result does not prove the architecture cannot benefit after matched training.
All state tensors are checked bit-identical after every arm. No promotion.

Runner: `scripts/recognition/evaluate_v51_resolution.py`.
Artifacts: `outputs/recognition/evaluation/v51_resolution_zero_update/`.
Logs: `outputs/recognition/v51.stdout.log` and `v51.stderr.log`.
