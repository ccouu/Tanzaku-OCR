# V29 frozen decoding comparison

Same nine frozen evaluation sets and V29 update2000. Identical model forward passes for all22 modes. No training, lexicon or beam search. Baseline predictions exactly reproduced for every line.

| Decoder | App clean | App texture | App noisy | Seen | Held-out | Original969 | Historical | YOLO | Strong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| penalty_0.0 | 15.70% | 20.18% | 20.91% | 4.70% | 6.00% | 4.89% | 10.17% | 90.50% | 65.00% |
| penalty_0.1 | 15.64% | 20.12% | 20.97% | 4.79% | 6.09% | 4.95% | 10.27% | 91.40% | 64.64% |
| penalty_0.2 | 15.70% | 20.30% | 21.03% | 4.76% | 6.05% | 4.97% | 10.40% | 91.86% | 64.64% |
| penalty_0.3 | 15.58% | 20.30% | 21.03% | 4.73% | 6.14% | 5.07% | 10.37% | 92.76% | 64.82% |
| penalty_0.4 | 15.64% | 20.24% | 20.97% | 4.70% | 6.18% | 5.16% | 10.51% | 92.76% | 64.82% |
| penalty_0.5 | 15.64% | 20.24% | 20.91% | 4.76% | 6.18% | 5.13% | 10.52% | 93.21% | 65.00% |
| penalty_0.6 | 15.64% | 20.30% | 20.73% | 4.91% | 6.27% | 5.22% | 10.64% | 93.21% | 65.54% |
| penalty_0.7 | 15.64% | 20.42% | 20.91% | 4.91% | 6.27% | 5.26% | 10.71% | 94.12% | 66.07% |
| penalty_0.8 | 15.82% | 20.42% | 20.91% | 4.88% | 6.45% | 5.30% | 10.81% | 94.12% | 66.43% |
| penalty_0.9 | 15.88% | 20.42% | 21.03% | 4.88% | 6.45% | 5.32% | 10.95% | 95.48% | 66.43% |
| penalty_1.0 | 15.94% | 20.55% | 21.27% | 4.94% | 6.41% | 5.30% | 11.07% | 95.93% | 66.79% |
| penalty_1.1 | 16.06% | 20.67% | 21.64% | 5.00% | 6.59% | 5.39% | 11.21% | 96.38% | 67.50% |
| penalty_1.2 | 16.06% | 20.97% | 21.64% | 5.06% | 6.64% | 5.41% | 11.40% | 97.74% | 67.32% |
| penalty_1.3 | 16.24% | 21.09% | 21.64% | 5.18% | 6.77% | 5.55% | 11.48% | 98.19% | 67.32% |
| penalty_1.4 | 16.36% | 21.58% | 22.06% | 5.33% | 6.77% | 5.66% | 11.61% | 99.10% | 68.21% |
| penalty_1.5 | 16.42% | 21.58% | 22.12% | 5.58% | 6.86% | 5.78% | 11.83% | 99.55% | 68.39% |
| penalty_1.6 | 16.30% | 21.64% | 22.42% | 5.64% | 7.18% | 5.88% | 12.15% | 100.00% | 68.39% |
| penalty_1.7 | 16.42% | 22.00% | 22.67% | 5.76% | 7.41% | 6.05% | 12.32% | 100.45% | 68.93% |
| penalty_1.8 | 16.67% | 22.18% | 22.73% | 5.79% | 7.59% | 6.15% | 12.60% | 100.90% | 69.46% |
| penalty_1.9 | 16.85% | 22.61% | 23.09% | 5.97% | 7.64% | 6.30% | 12.72% | 101.36% | 70.54% |
| penalty_2.0 | 16.97% | 23.03% | 23.64% | 6.03% | 7.68% | 6.38% | 12.96% | 102.26% | 70.71% |
| threshold_0.5 | 15.88% | 20.67% | 21.33% | 4.82% | 6.23% | 5.05% | 10.43% | 92.76% | 65.89% |

## Best penalty on each development set

| Dataset | Lowest CER | Penalties attaining minimum |
|---|---:|---|
| application_clean_control | 15.58% | penalty_0.3 |
| application_texture_color | 20.12% | penalty_0.1 |
| application_texture_color_noise | 20.73% | penalty_0.6 |
| seen | 4.70% | penalty_0.0, penalty_0.4 |
| heldout | 6.00% | penalty_0.0 |
| original_bucket | 4.89% | penalty_0.0 |
| historical_bucket | 10.17% | penalty_0.0 |
| yolo_provisional | 90.50% | penalty_0.0 |
| strong_appearance | 64.64% | penalty_0.1, penalty_0.2 |

Arm1 subtracts beta from the blank logit. Arm2 uses unmodified softmax over all1723 classes: choose blank iff P(blank)>0.5, otherwise choose the highest-probability nonblank token. Both apply ordinary CTC repeat-collapse and blank removal, using unchanged valid lengths.

Per-dataset best penalties are descriptive development-set optima, not independently validated recommendations. The same application-generator family was used in training; YOLO labels are provisional. Frame scores, per-line predictions and length/font/contrast/texture metrics are saved for inspection.

## Interpretation

Keep standard greedy decoding (penalty0) as the default. The threshold0.5 arm
worsens CER on every one of the nine evaluation sets. It permits a nonblank
token whenever blank probability is at most0.5, even if blank remains the most
probable individual class; this can force uncertain characters into the output.

The best noisy-application penalty0.6 reduces345→342 edits on1,650 characters,
only0.18 percentage points, while worsening held-out-font CER6.00%→6.27%,
original retention4.89%→5.22%, and historical10.17%→10.64%. Clean application
best penalty0.3 saves only two edits (259→257). These small, dataset-specific
optima do not support a universal positive blank penalty.

On noisy application images, penalty0→2 reduces deletions165→76 but increases
insertions50→150 and substitutions130→164. Total CER worsens20.91%→23.64%.
Threshold0.5 yields148 deletions,63 insertions and141 substitutions, versus
165/50/130 for baseline; total edits rise345→352.

Short text does not improve consistently. Noisy one-character CER is46.67%
at baseline,50.00% with threshold0.5 and63.33% with penalty2. Held-out
one-character CER improves37.50%→30.00% with penalty2, but total held-out CER
worsens6.00%→7.68%. Thus an aggregate deletion count cannot justify forcing
more emissions, and this decoding change does not resolve the short-text issue.

All baseline transcriptions matched the prior V29 report exactly; penalty0.2
aggregate metrics also matched on all nine sets. The checkpoint and model state
were unchanged. No model updates or production decoder changes were made.
