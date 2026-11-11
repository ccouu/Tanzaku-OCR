# Stage 3 — V27 implementation summary

The active Stage3 tree contains V27 only. The complete V1–V28 history, source,
datasets, reports and unused outputs are retained in the
[dated archive](../../archive/stage3_legacy_20261111/README.md).
This cleanup reorganized code and resources; it did not train or alter weights.

## Architecture

H64 RGB → ImageNet-initialized MobileNetV4 Conv-Small, horizontal stride4 →
height mean-pooling →960-to-256 temporal projection → binary orientation
conditioning → two residual depthwise-separable1D temporal blocks → packed
residual BiLSTM,128 hidden units per direction →1,723-class CTC head.
There are1,722 recognized characters plus blank. Originally vertical crops
are rotated into strips; their orientation flag is retained.

The BiLSTM receives valid lengths and does not consume artificial right padding
in its backward direction. CNN and temporal blocks retain normal BatchNorm;
they are not strictly invariant to padding. No internal-mask prototype or
variable-character-size training branch is part of V27.

## Data and training

- 64,000 generated training records, each consumed exactly once per epoch.
- 15 fonts:7 original and8 new families,50/50 cohort mixture.
- Length categories1/2–4/5–7/8–10 at5/45/35/15% of the corpus.
- Nearest128/256/384 physical buckets, one-character crops at128.
- Half canonical margins, half variable left allocation within the bucket;
  retain a real right guard. No separate forced384px stream.
- 5% mild morphology, equally dilation and erosion, using stored25%-strength
  variants. Preserve glyph pixels and reduce excess background if needed.
- Valid CTC lengths ceil((left+ink span+right guard)/4), excluding outer fill.
- 20 full epochs,40,020 updates and1,280,000 presentations. Partial homogeneous
  batches are retained, so each epoch has2,001 updates.
- Fresh AdamW, no norm/bias weight decay; fully unfrozen model. Encoder5e-6,
  head3e-5 for the first two epochs, cosine decay to10% by epoch20. BF16 CNN,
  FP32 recurrence and CTC; gradient norm clip5.

Fonts and their licenses remain in the shared resources directories. Training
data is now `data/datasets/recognition/v27_fonts`; its original labels and
image hashes are unchanged. Validation dependencies are independently packaged
under `v27_validation`, with provenance in its manifest.

## Results and checkpoint choice

| CER, beta0 | Initial checkpoint | V27 epoch20 |
|---|---:|---:|
| Seen15-font validation | 7.30% | 4.48% |
| Held-out4-font validation | 7.73% | 5.27% |
| Original-clean969 retention | 5.90% | 4.70% |
| Historical1,000 crops | 12.13% | 11.27% |
| Provisional38 GenAI YOLO crops | 93.21% | 96.38% |

Selection uses equal original/new/held-out font macro CER, subject to four
retention caps: original-font, original-clean, historical-crop and provisional
YOLO CER each at most the starting value+1absolute percentage point.
Epoch20 fails the YOLO cap,94.21%. Every trained epoch failed at least one
cap, so `best.pt` is still the initial V26 model in V27's recovery schema.
Use **epoch_20.pt** when explicitly testing the trained V27 font candidate;
`last.pt` has the same model tensors. `initial.pt` is a byte-identical copy of
the original initialization, retained for reproducibility.

The model remains experimental. Strong appearance CER is around67%; the
provisional menu subset has zero exact matches. Reviewed real handwriting,
independent end-to-end CER, lexicon evaluation, mobile export/parity and device
latency remain outstanding. No production model has been promoted.

## Cleanup verification

The V27 support layer was extracted from the archived experiment modules.
Before moving files,290 old/new dataset samples were exactly equal, the
full-pass sampling plan was identical, and model logits on a32-image batch
were bit-identical. No training updates were performed. The original run's
checkpoints, source.zip, metrics and embedded provenance were preserved.

The [archive manifest](../../archive/stage3_legacy_20261111/manifest.json)
maps original paths to their new locations. Historical scripts retain their
original path assumptions and are reference/recovery material; they are not
the active training API. See [README](README.md) for current commands.
