# V31 single-character blank decoding

Frozen curated step2000 checkpoint;3434 one-character images. No training or lexicon. Original full-batch baseline predictions reproduced on existing V31 sets.

| Decoder | Menu clean | Menu noisy |16terms clean |16terms noisy |Original200 |Broad clean |Broad noisy |Font clean |Font noisy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| penalty_0.0 | 29.00% | 33.50% | 15.33% | 26.49% | 30.50% | 45.50% | 57.50% | 49.20% | 53.80% |
| penalty_0.1 | 29.50% | 33.50% | 15.62% | 26.19% | 30.00% | 44.50% | 57.00% | 49.00% | 54.80% |
| penalty_0.2 | 30.00% | 33.50% | 14.88% | 26.34% | 30.00% | 44.00% | 57.00% | 49.20% | 55.00% |
| penalty_0.3 | 30.00% | 33.50% | 14.73% | 26.49% | 30.00% | 44.50% | 57.00% | 49.40% | 55.20% |
| penalty_0.4 | 29.00% | 33.50% | 14.73% | 26.34% | 30.00% | 44.00% | 57.00% | 49.60% | 55.60% |
| penalty_0.5 | 29.00% | 34.00% | 14.73% | 26.34% | 30.00% | 44.00% | 56.50% | 50.20% | 55.40% |
| penalty_0.6 | 28.50% | 34.50% | 14.73% | 25.89% | 29.50% | 44.00% | 56.50% | 49.00% | 55.80% |
| penalty_0.7 | 27.50% | 35.50% | 14.88% | 25.89% | 29.50% | 44.00% | 56.50% | 49.60% | 56.40% |
| penalty_0.8 | 27.50% | 34.50% | 14.73% | 25.74% | 30.00% | 44.50% | 56.50% | 49.40% | 56.60% |
| penalty_0.9 | 27.00% | 35.00% | 14.58% | 26.49% | 30.00% | 44.50% | 56.50% | 49.40% | 57.00% |
| penalty_1.0 | 27.50% | 35.00% | 14.43% | 26.04% | 29.50% | 44.50% | 56.50% | 49.60% | 57.00% |
| penalty_1.1 | 28.50% | 34.50% | 14.43% | 26.19% | 30.00% | 45.50% | 56.00% | 49.80% | 56.80% |
| penalty_1.2 | 29.50% | 34.00% | 14.58% | 26.34% | 31.00% | 46.50% | 56.50% | 50.00% | 56.80% |
| penalty_1.3 | 30.00% | 34.00% | 14.58% | 26.64% | 31.00% | 46.50% | 57.00% | 50.00% | 57.40% |
| penalty_1.4 | 30.00% | 33.50% | 14.73% | 26.64% | 30.00% | 46.50% | 57.50% | 50.60% | 58.20% |
| penalty_1.5 | 30.00% | 34.00% | 15.18% | 26.79% | 30.00% | 47.00% | 58.50% | 50.60% | 58.40% |
| penalty_1.6 | 30.00% | 34.50% | 15.33% | 26.93% | 30.00% | 47.00% | 59.00% | 51.00% | 58.60% |
| penalty_1.7 | 31.50% | 35.50% | 15.48% | 27.23% | 30.00% | 47.00% | 59.50% | 51.00% | 60.60% |
| penalty_1.8 | 31.50% | 36.00% | 15.77% | 27.23% | 30.00% | 47.00% | 60.00% | 51.20% | 61.00% |
| penalty_1.9 | 32.00% | 37.00% | 16.22% | 27.38% | 30.50% | 48.00% | 61.00% | 51.60% | 61.60% |
| penalty_2.0 | 32.50% | 37.00% | 16.22% | 27.68% | 30.50% | 48.00% | 61.50% | 51.80% | 62.40% |
| threshold_0 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| threshold_0.1 | 30.50% | 38.50% | 18.15% | 29.46% | 35.00% | 46.00% | 59.00% | 51.60% | 57.40% |
| threshold_0.2 | 30.00% | 35.00% | 16.82% | 27.23% | 32.50% | 46.00% | 57.50% | 49.40% | 56.60% |
| threshold_0.3 | 27.50% | 34.50% | 15.62% | 26.64% | 30.00% | 47.00% | 57.50% | 48.40% | 54.20% |
| threshold_0.4 | 28.50% | 33.00% | 14.73% | 26.79% | 29.50% | 45.50% | 56.50% | 48.60% | 54.20% |
| threshold_0.5 | 30.00% | 35.00% | 14.58% | 26.19% | 29.50% | 44.50% | 57.50% | 49.80% | 54.40% |
| threshold_0.6 | 28.50% | 34.50% | 14.58% | 25.89% | 29.50% | 44.50% | 56.50% | 50.20% | 57.00% |
| threshold_0.7 | 28.50% | 34.00% | 14.88% | 26.64% | 30.00% | 45.00% | 58.00% | 50.00% | 56.20% |
| threshold_0.8 | 32.00% | 35.00% | 15.03% | 27.83% | 31.00% | 47.00% | 62.00% | 52.00% | 59.80% |
| threshold_0.9 | 34.50% | 40.50% | 16.67% | 31.40% | 34.00% | 48.50% | 67.00% | 55.20% | 64.20% |
| threshold_0.95 | 37.50% | 49.50% | 20.24% | 36.90% | 36.00% | 52.50% | 71.00% | 60.60% | 72.00% |
| threshold_0.99 | 55.50% | 62.00% | 33.18% | 51.04% | 45.50% | 76.50% | 91.00% | 74.60% | 91.80% |
| threshold_1 | 803.00% | 833.50% | 755.65% | 791.52% | 689.00% | 888.00% | 899.50% | 874.40% | 881.60% |
| nonempty_fallback | 24.00% | 29.00% | 10.71% | 21.58% | 28.00% | 41.00% | 54.00% | 43.80% | 48.80% |

## Best observed settings per set

| Set | Best penalty | CER | Best threshold | CER | Empty-only fallback CER |
|---|---|---:|---|---:|---:|
| menu_clean | penalty_0.9 | 27.00% | threshold_0.3 | 27.50% | 24.00% |
| menu_noisy | penalty_0.0 | 33.50% | threshold_0.4 | 33.00% | 29.00% |
| meaningful_clean | penalty_1.0 | 14.43% | threshold_0.5 | 14.58% | 10.71% |
| meaningful_noisy | penalty_0.8 | 25.74% | threshold_0.6 | 25.89% | 21.58% |
| pooled_original_1char | penalty_0.6 | 29.50% | threshold_0.4 | 29.50% | 28.00% |
| fresh_short_clean | penalty_0.2 | 44.00% | threshold_0.5 | 44.50% | 41.00% |
| fresh_short_noisy | penalty_1.1 | 56.00% | threshold_0.4 | 56.50% | 54.00% |
| font_audit_clean | penalty_0.1 | 49.00% | threshold_0.3 | 48.40% | 43.80% |
| font_audit_noisy | penalty_0.0 | 53.80% | threshold_0.3 | 54.20% | 48.80% |

Nonempty text detection and CTC temporal blank are different. Thresholds apply per frame,not per crop. Threshold1 removes every blank frame and can create many insertions. Empty-only fallback preserves all existing nonempty predictions and all temporal decoding;it changes only an entirely empty result.

For a true one-character crop,forcing one character only when the output was empty cannot worsen edit distance:correct fallback saves an error,incorrect fallback trades one deletion for one substitution. This guarantee does not extend to blank detector false positives. Per-dataset optimal settings are development diagnostics,not independently validated universal decoder choices.
