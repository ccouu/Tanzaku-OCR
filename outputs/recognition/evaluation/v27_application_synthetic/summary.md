# V27 application-like synthetic pilot

300 paired examples, 15 fonts, 1–10 characters, 24–120px native font size. No training. V27 epoch20; raw greedy CTC, blank bias0.

| View | CER | Exact /300 | Deletions | Insertions | Substitutions |
|---|---:|---:|---:|---:|---:|
| clean_control | 28.85% | 87 | 337 | 23 | 116 |
| texture_color | 42.61% | 53 | 510 | 14 | 179 |
| texture_color_noise | 44.42% | 47 | 539 | 17 | 177 |
| noise_grayscale | 45.94% | 46 | 547 | 11 | 200 |

## Noisy RGB results by contrast

| Group | Lines | CER | Exact |
|---|---:|---:|---:|
| high | 100 | 35.29% | 24 |
| low | 100 | 60.04% | 7 |
| medium | 100 | 39.23% | 16 |

## Noisy RGB results by texture

| Group | Lines | CER | Exact |
|---|---:|---:|---:|
| chalkboard | 75 | 40.81% | 14 |
| cloth | 75 | 40.14% | 11 |
| paper | 75 | 45.90% | 12 |
| wood | 75 | 50.60% | 10 |

## Noisy RGB results by length

| Group | Lines | CER | Exact |
|---|---:|---:|---:|
| 1 | 30 | 63.33% | 11 |
| 2 | 30 | 46.67% | 11 |
| 3 | 30 | 52.22% | 4 |
| 4 | 30 | 49.17% | 4 |
| 5 | 30 | 47.33% | 4 |
| 6 | 30 | 48.89% | 2 |
| 7 | 30 | 38.10% | 4 |
| 8 | 30 | 42.92% | 0 |
| 9 | 30 | 41.85% | 4 |
| 10 | 30 | 41.67% | 3 |

## Noisy RGB results by canvas

| Group | Lines | CER | Exact |
|---|---:|---:|---:|
| 128 | 64 | 55.88% | 22 |
| 256 | 97 | 50.96% | 12 |
| 384 | 139 | 40.99% | 13 |

## Noisy RGB results by source_size

| Group | Lines | CER | Exact |
|---|---:|---:|---:|
| 24-47 | 72 | 49.88% | 8 |
| 48-79 | 103 | 43.27% | 18 |
| 80-120 | 125 | 42.15% | 21 |

Crops are resized proportionally to H64 and edge-extended to128/256/384. Lengths exclude artificial padding. Overflowing layouts are rejected, not squeezed. Mild native-scale Gaussian blur/noise precedes resizing.

These are known-font, reused-transcription procedural examples, not independent real-menu or YOLO accuracy. Contrast groups are an engineered stress test. Different groups contain different texts; paired views share exact geometry and labels. Native text sizes are randomly sampled, and long lines favor taller crops to fit384.

[Visual review](review.html) · [Full report](report.json) · [Generation manifest](manifest.json)

## Interpretation and verification

A separate evaluation of the original 600 seen-font validation crops reproduced
V27 epoch20 exactly: 148 edits /3,300 characters =4.4848% CER, 477 exact lines.
The checkpoint SHA256 and inference model state were unchanged.

The clean re-rendered control already reaches28.85% CER. This indicates a large
rendering/crop distribution gap: this pilot uses normal whole-line font rendering,
random native resolution and variable top/bottom/side crop margins, whereas V27
used normalized per-character masks. This experiment does not isolate which of
those changes causes that gap. Absolute source font size also changes effective
resolution after resizing; source-size groups contain different texts and fonts.

With geometry and labels held fixed, textured/color appearance adds13.76
percentage points of CER; native mild blur plus sensor noise adds another1.82
points. Thus appearance is a substantial additional weakness, but not the only
one. Deletions account for539/733 errors (73.5%) in noisy RGB. Grayscale gives
45.94% CER versus44.42% RGB and does not solve the problem on this set.

Recommended next experiment: a matched short continuation control versus a
mixture containing this application-like rendering/crop recipe, retaining clean
V27 validation constraints. Start with clean/high-contrast rendering geometry,
then introduce texture and moderate contrast; keep a separate held-out seed/text
set for selection. This pilot alone does not justify a backbone change. No
continuation training was run.
