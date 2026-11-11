# V31 reviewed short-term pilot

Matched2,000-update arms from V30 control. Original replay,batch order,lengths,font settings and physical buckets are identical. Fresh short renders in both arms; treatment changes70%of short generated labels to approved menu terms. No higher overall short-text quota.

| Arm | Update | Menu clean1 | Menu clean2 | Menu noisy1 | Menu noisy2 | App clean | App noisy | Seen | Held-out | Retention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| control | 0 | 32.50% | 38.50% | 39.50% | 42.50% | 13.33% | 16.61% | 4.97% | 5.86% | pass |
| control | 500 | 33.50% | 38.75% | 38.50% | 43.75% | 13.33% | 16.06% | 5.03% | 5.68% | pass |
| control | 1000 | 34.00% | 38.50% | 42.50% | 42.50% | 12.12% | 15.94% | 5.64% | 6.23% | pooled_original_1char, pooled_original_2char, fresh_short_clean2, fresh_short_noisy1 |
| control | 1500 | 32.50% | 37.75% | 39.00% | 41.75% | 12.30% | 15.52% | 5.18% | 5.64% | pooled_original_1char |
| control | 2000 | 32.50% | 36.75% | 39.50% | 41.75% | 12.00% | 14.67% | 4.94% | 5.73% | pooled_original_1char |
| curated_70 | 0 | 32.50% | 38.50% | 39.50% | 42.50% | 13.33% | 16.61% | 4.97% | 5.86% | pass |
| curated_70 | 500 | 30.50% | 38.25% | 38.00% | 41.00% | 12.97% | 16.24% | 5.03% | 5.73% | pooled_original_1char, fresh_short_clean2 |
| curated_70 | 1000 | 33.00% | 38.00% | 37.50% | 40.50% | 11.88% | 15.82% | 5.55% | 6.18% | pooled_original_1char, pooled_original_2char, fresh_short_clean1, fresh_short_noisy1 |
| curated_70 | 1500 | 29.50% | 36.00% | 38.50% | 41.00% | 12.97% | 15.82% | 5.18% | 5.82% | historical_bucket, pooled_original_1char, pooled_original_2char, fresh_short_clean1 |
| curated_70 | 2000 | 29.00% | 34.50% | 33.50% | 39.50% | 11.94% | 14.91% | 5.15% | 5.86% | pooled_original_1char |

See sampling.json for exact rounded70/30 fractions and per-term counts. Term split refers only to the fresh generated short stream,not historical exposure. New menu-probe images are frozen before training. No lexicon corrections are applied. best.pt may remain the starting checkpoint if no eligible improvement occurs.
