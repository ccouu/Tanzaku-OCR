# V30 short-text retention pilot

Both arms continue V29 update2000 for2,000 updates, encoder5e-7/head3e-6,50/25/25 image streams, unchanged source images. Standard greedy CTC at beta0. Fresh probe:200 examples per length, clean/noisy paired views, half known and half held-out fonts.

| Arm | Update | Fresh clean1 | Fresh clean2 | Fresh noisy1 | Fresh noisy2 | App clean | App noisy | Seen | Held-out | Original | Retention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| control | 0 | 45.00% | 21.00% | 60.00% | 27.25% | 15.70% | 20.91% | 4.70% | 6.00% | 4.89% | pass |
| control | 500 | 45.00% | 19.75% | 60.50% | 26.75% | 14.91% | 20.00% | 4.67% | 5.68% | 4.55% | pass |
| control | 1000 | 48.00% | 19.50% | 61.50% | 27.00% | 14.24% | 18.00% | 5.30% | 6.00% | 5.39% | pooled_original_1char |
| control | 1500 | 45.50% | 18.50% | 60.00% | 25.00% | 14.55% | 17.58% | 4.91% | 6.05% | 5.16% | pooled_original_1char |
| control | 2000 | 44.50% | 17.50% | 58.50% | 25.50% | 13.33% | 16.61% | 4.97% | 5.86% | 4.95% | pass |

control: best eligible update2000. Step0 remains eligible.

| short_20_20 | 0 | 45.00% | 21.00% | 60.00% | 27.25% | 15.70% | 20.91% | 4.70% | 6.00% | 4.89% | pass |
| short_20_20 | 500 | 48.50% | 22.75% | 60.00% | 29.50% | 18.06% | 22.24% | 5.09% | 6.23% | 5.57% | historical_bucket, application_clean_control, application_texture_color_noise |
| short_20_20 | 1000 | 47.50% | 20.50% | 59.00% | 26.25% | 16.36% | 20.06% | 4.97% | 5.91% | 5.55% | historical_bucket, pooled_original_2char |
| short_20_20 | 1500 | 47.00% | 20.00% | 57.00% | 25.75% | 16.61% | 19.27% | 5.09% | 6.05% | 5.26% | pass |
| short_20_20 | 2000 | 46.50% | 19.75% | 56.50% | 25.25% | 14.61% | 18.30% | 4.97% | 6.41% | 5.49% | pass |

short_20_20: best eligible update2000. Step0 remains eligible.


Selection and constraints were fixed before training. Fresh probes use new renderings/full strings relative to V29 generated training, but trained characters and procedural textures; this is not real-menu accuracy. Historical training may already contain some probe strings. Current replay excludes all fresh-probe strings in both arms.

Control reuses V29 batches except reserved-probe original labels are replaced within the same length/cohort/bucket. Treatment uses20% one-character and20% two-character batches, with60%3–10, sampling with replacement. Homogeneous length grouping, bucket mix, marginal font/orientation frequencies and unique-image coverage can differ; sampling.json audits these limits. This compares length-focused replay policies, not a perfectly isolated per-example weighting effect.
