# Single-character font audit:1,000 images

V30 control update2000; raw greedy CTC,blank penalty0. Primary font comparison uses the same23 characters per setting,each with clean/noisy views.19 fonts plus2 regular-weight Noto settings.17 extra pairs complete1000 and are excluded from the ranking.

| Font setting | Clean CER | Clean exact /23 | Noisy CER | Noisy exact /23 | Clean D/I/S | Noisy D/I/S |
|---|---:|---:|---:|---:|---|---|
| YuseiMagic-Regular | 69.57% | 8 | 86.96% | 5 | 3/1/12 | 3/4/13 |
| SawarabiGothic-Regular | 60.87% | 9 | 69.57% | 7 | 3/3/8 | 3/4/9 |
| HachiMaruPop-Regular | 56.52% | 10 | 65.22% | 9 | 4/3/6 | 4/3/8 |
| YuGothR | 56.52% | 10 | 56.52% | 10 | 1/4/8 | 2/2/9 |
| NotoSansJP[wght]_weight400 | 56.52% | 11 | 47.83% | 12 | 1/1/11 | 2/1/8 |
| KouzanBrushFont | 56.52% | 11 | 43.48% | 13 | 4/4/5 | 4/1/5 |
| AoyagiSouseki | 52.17% | 11 | 78.26% | 6 | 2/1/9 | 3/2/13 |
| meiryo | 52.17% | 11 | 52.17% | 11 | 3/1/8 | 1/2/9 |
| NotoSerifJP[wght]_default200 | 52.17% | 11 | 47.83% | 12 | 2/2/8 | 2/3/6 |
| yumin | 52.17% | 12 | 47.83% | 13 | 2/2/8 | 3/1/7 |
| ZenKurenaido-Regular | 47.83% | 12 | 78.26% | 7 | 3/1/7 | 5/4/9 |
| ShipporiMincho-Bold | 47.83% | 12 | 65.22% | 8 | 3/1/7 | 3/1/11 |
| NotoSerifJP[wght]_weight400 | 47.83% | 12 | 56.52% | 13 | 2/1/8 | 1/3/9 |
| Yomogi-Regular | 47.83% | 13 | 47.83% | 13 | 3/2/6 | 2/3/6 |
| NotoSansJP[wght]_default100 | 47.83% | 13 | 39.13% | 14 | 1/1/9 | 3/0/6 |
| KaiseiTokumin-Regular | 47.83% | 13 | 39.13% | 14 | 1/2/8 | 1/1/7 |
| KleeOne-SemiBold | 39.13% | 14 | 52.17% | 11 | 3/2/4 | 4/2/6 |
| UDDigiKyokashoN-R | 39.13% | 15 | 52.17% | 11 | 4/1/4 | 3/2/7 |
| YujiMai-Regular | 39.13% | 14 | 39.13% | 14 | 2/1/6 | 4/0/5 |
| YujiBoku-Regular | 34.78% | 15 | 34.78% | 15 | 3/1/4 | 5/0/3 |
| YujiSyuku-Regular | 26.09% | 17 | 34.78% | 15 | 1/1/4 | 4/1/3 |

Only23 shared characters per font: one edit changes CER4.35 points. Rankings are descriptive,not a reliable ordering of close fonts. All fonts explicitly support every character; no tofu/fallback substitutions were rendered. Native size and margin fractions are matched per character,but actual glyph proportions vary by font.

Noto default weights match the application renderer; weight400 matches the original V27 renderer. Paired weight results help distinguish font-family issues from stroke-weight mismatch.

[Image/prediction review](review.html) · [Detailed metrics](report.json) · [Manifest](manifest.json)
