# V35 paired short-text inspection and inference precision

No training. V30andV35fullpass4001predictions exactly reproduced on all1969
original validation images;400short cases extracted with original full-batch
composition. Twenty-five lines regress,16improve.

## Main residual patterns

- BF16horizontal1-char improves32.98→26.60%,vertical1-char worsens25.47→33.96%.
  Twelve of14one-char regressions and all5new blanks are vertical. New blank cases:
  恨,題,叩,柚,猿. Both renderer code and exact crops confirm rotated glyphs.
- Generated application images are all horizontal;V34overall vertical exposure
  is24.22%. V30prior mixture is24.30%,so no new ratio bug is established.
- Four newly deleted2-char prefixes:鰻炭→炭,前主→主,菊蒸→蒸,鰹塩→塩.
  Missing-token probability peaks stay inside validT and lose to blank. Extra
  one-frame emissions include百五→百丑五,員→員頁,せ蒼→せ蒼替.
- Full-width decoding changes none of400predictions in either model. Keep valid
  lengths. Incorrect orientation flags recover selected cases but worsen aggregate
  strata;do not change inference flags. Several fonts show mixed gains/losses.
- Canonical crop margins do not separate improved/regressed cases. An exploratory
  central blank-gap proxy differs between old2-char crops and new renders;
  investigate spacing separately,not as a proven cause.

## Important qualification to the BF16 gate failure

Exact blank/target BF16ties prompted inference-only precision checks. FullFP32and
CUDAFP16improve borderline candidate cases;FP32classifier-only does not solve it.
Pooled1/2-char errors:BF16V30=58/50,V35=61/56;FP16V30=59/50,V35=58/53.

The complete15-set FP16suite subsequently passes **all** source-relative retention
gates using the matching V30FP16baseline and unchanged+1pp rules. TrainedV35FP16:
seen5.03%,held5.45%,original4.82%,historical10.56%,resized clean/noisy9.82/12.12%,
confirmation clean/noisy25.60/29.80%. Heavy appearance64.29% versus63.93%source;
YOLO88.24%,still0/38exact.

This is a post-hoc development diagnostic,not phone inference or independent
confirmation. Original BF16reports/gates remain intact;best.pt stays V30. Candidate
is `outputs/recognition/pilots/v35_frozen_bn_fullpass4001/last.pt`. No promotion.
Recommended next step:mobile export/device precision and prediction parity before
more training. Vertical application-style evaluation and balanced orientation
sampling can follow if the weakness persists;spacing is a separate intervention.

## Artifacts

- [Full paired analysis and frame evidence](../../outputs/recognition/evaluation/v35_paired_short_inspection/analysis.md)
- [Complete FP16 evaluation](../../outputs/recognition/evaluation/v35_paired_short_inspection/fp16_full_analysis.md)
- [Annotated examples](../../outputs/recognition/evaluation/v35_paired_short_inspection/regressions_01.png)
- Raw traces:`cases.jsonl`;counterfactuals and precision:`precision_cases.jsonl`;
  original reports/checkpoint hashes:`summary.json`. No source checkpoint changes.
