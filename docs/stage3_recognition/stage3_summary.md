# Stage 3 — implementation and experiment summary

The reusable `src/recognition/` implementation remains V27. Subsequent V29–V31
continuations and inference diagnostics use that architecture through scripts
under `outputs/recognition/`. No production checkpoint or decoder was promoted.
V30 control is the retained reference for recent adaptation experiments; V31
curated update2000 is the latest trained candidate,subject to retention limits.

The complete V1–V28 history and retired source/reports are local in the
[dated archive](../../archive/stage3_legacy_20261111/README.md). Archived dataset
images and ZIP copies were subsequently deleted to reclaim7.08GiB;labels and
metadata were retained. Active V27 training/validation data were verified
unchanged. The cleanup itself did not train or alter model weights.

For a compact operational state,see [HANDOFF.md](HANDOFF.md).

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

## V27 data and training

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

## V27 results and checkpoint choice

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

## V28–V31 experiments

V28 tested variable per-character size/spacing with matched2,000-update control
and treatment continuations from V27 epoch20. Both drifted on validation;both
best checkpoints retained starting weights. The experiment is archived and is
not part of the active training implementation.

V29 introduced application-like rendering:normal whole-line text at24–120px
native font sizes,variable source margins,proportional H64 resizing,and right-edge
replication to the next fitting128/256/384 bucket. Four procedural background
textures,distinguishable low/medium/high-contrast font colors,and mild native
Gaussian blur/noise were used. Overflowing layouts were rejected rather than
squeezed or clipped. Training used50% original clean images,25% clean new renders
and25% textured/color renders with mild noise. Fresh generated labels excluded
monitored validation transcriptions. No additional morphology was introduced.

V29 and subsequent pilots used fresh AdamW,fully unfrozen weights,normal training
BatchNorm,constant backbone5e-7/head3e-6 learning rates,BF16 CNN,FP32 packed BiLSTM
and CTC,clip5,32-image homogeneous-width batches,and greedy decoding at beta0.
Each arm had2,000 updates/64,000 presentations;these were not full passes through
all original records. Each new continuation started from its named source weights
with a fresh optimizer.

| Candidate / source | Seen CER | Held-out CER | Application clean CER | Application noisy CER |
|---|---:|---:|---:|---:|
| V27 epoch20,application zero-update test |4.48%|5.27%|28.85%|44.42%|
| V29 update2000,from V27 epoch20 |4.70%|6.00%|15.70%|20.91%|
| V30 control update2000,from V29 |4.97%|5.86%|13.33%|16.61%|
| V30 short20/20 update2000,from V29 |4.97%|6.41%|14.61%|18.30%|
| V31 control update2000,from V30 control |4.94%|5.73%|12.00%|14.67%|
| V31 curated70 update2000,from V30 control |5.15%|5.86%|11.94%|14.91%|

The application columns use the same frozen300-example paired probe. These are
known-font/procedural development images,not independent real-menu accuracy.

**V30:** Tested whether allocating20% one-character,20% two-character and60%
longer samples within each image stream improved short-text recognition. The
control was stronger overall. Short replay used fewer unique generated pairs
(8,991 versus16,000) and changed length grouping/bucket mix;this does not prove
short-text emphasis is intrinsically harmful. Both arms selected update2000.
See [V30 analysis](../../outputs/recognition/pilots/v30_short_retention_2000/analysis.md).

**V31:** Tested curated label selection while matching original record IDs,batch
order,lengths,font settings,orientation and physical buckets. Both arms received
fresh short renders. Treatment replaced70%of generated short-label pairs with
reviewed menu terms;the30% random stream shared pixels with control. Overall
short-text quota was unchanged. Fifty-eight legacy256px short pairs were replayed
identically because ordinary replacement terms did not naturally fit that bucket.
There were1,858 curated pairs/2,654 short pairs:3,716 images,5.81%of all64,000
presentations. Median coverage was5 fresh pairs per one-character term and4 per
two-character term. This was not a50-occurrence-per-term curriculum.

| V31 frozen menu-probe CER | Starting V30 | Control2000 | Curated2000 |
|---|---:|---:|---:|
| Clean1 character |32.50%|32.50%|29.00%|
| Clean2 characters |38.50%|36.75%|34.50%|
| Noisy1 character |39.50%|39.50%|33.50%|
| Noisy2 characters |42.50%|41.75%|39.50%|
| Equal macro across four groups |38.25%|37.625%|34.125%|

Curation improves this intended-domain macro by3.50 absolute points versus the
matched control. A5,000-draw paired bootstrap clustered by100 term identities,
stratified by length,has95% interval[-5.00,-2.06]points for treatment minus control;
this is evaluation-sample uncertainty,not training-seed uncertainty. Gains also
appear on terms withheld from the new curated stream and on held-out fonts.
Earlier training/original replay can still contain those terms.

Both final V31 checkpoints fail the original one-character retention cap:
starting58/200 errors (29%),allowed60/200 (30%),final61/200 (30.5%). The limit was
not relaxed. V31 control selects update500;curated `best.pt` remains update0,
containing V30 source weights. The actual trained curated candidate is
`curated_70/step_2000.pt`. See [V31 analysis](../../outputs/recognition/pilots/v31_curated_short_2000/analysis.md).

## Font and terminology diagnostics

A1,000-image single-character audit covered19 fonts plus two Noto weight
variants. A matched23-character core showed Yusei Magic,Sawarabi Gothic and
Hachi Maru Pop weak on that sample,but rankings changed with the character set.
Cross-font errors included small kana (`ゅ→ゆ`),diacritics (`ぴ→ひ`),similar Kanji
(`瑠→溜`) and duplicate emissions (`華→華華`). Noto Sans default weight100 and
Noto Serif default200 differ from V27's original weight400;regular weight did
not consistently help across probes. No font was removed based on that audit.

A separate16-term test used genuine single-character menu terms (`円`,`茶`,`肉`,
etc.) and occurrence-matched Kanji controls. V30 CER was20.83% clean/31.10% noisy
on menu terms versus17.56%/27.38% on controls. This did not establish semantic
understanding. 円 was especially weak despite its clear meaning.

The local culinary database contains47,216 entries,including560 one-character
and4,509 two-character candidates. Independent parent/Gemini Argon review through
Herdr produced a conservative496-term training shortlist:114 one-character and
382 two-character terms,comprising485 high-confidence agreements plus11 separately
sourced essentials. Gemini left3,474 candidates uncertain. Its supplementary JSON
contains57 suggestions,not the62 stated in its raw summary;1,407 candidates are
outside the recognition vocabulary (27.76%,not37.23%). No vocabulary expansion
or main SQLite modification was made. See [curation validation](../../outputs/recognition/curation/v31_short_terms/VALIDATION.md).

Read-only checks confirmed unsuitable menu-default readings/glosses for entries
such asピザ,中生,大福,生 and杯. OCR curriculum uses surface text;dictionary
translation/sense selection still requires separate repair. The dictionary is
`data/processed/lexicon/culinary_menu.db`;lookup code is
`src/data/lexicon_builder.py:CulinaryLookup`. All reported recognition CER is raw
CTC,without this fuzzy lookup.

## Latest one-character decoding test

Using **V31 curated step2000**,evaluated3,434 frozen one-character images with
21 additive blank penalties0–2,13 probability thresholds0–1,and an empty-only
fallback:35 decoders. Existing V31 baseline transcriptions reproduced exactly.
No weights or production decoder changed.

| Decoder | Curated menu clean1 CER | Curated menu noisy1 CER | Original200 CER |
|---|---:|---:|---:|
| Standard greedy |29.00%|33.50%|30.50%|
| Blank penalty0.9 |27.00%|35.00%|30.00%|
| Blank threshold0.4 |28.50%|33.00%|29.50%|
| Blank threshold0.5 |30.00%|35.00%|29.50%|
| Empty-only fallback |24.00%|29.00%|28.00%|

The fallback preserves every nonempty greedy output. Only if the whole result
is empty,it returns one nonblank token from the valid frame having the highest
nonblank full-class softmax probability. It does not use ground-truth length,
dictionary filtering or artificial right-padding frames. It recovered10/19 clean
and9/17 noisy empty menu predictions correctly;other forced outputs remained
errors. On42 all-font円 examples per view,exact accuracy improved14→25 clean
and2→5 noisy. Yen remains difficult under appearance changes.

YOLO confidence that a crop contains text supports a string-level nonempty
assumption;CTC still needs temporal blanks for alignment. Suppressing blank at
every frame caused severe insertion errors. Empty-only fallback is the strongest
decoder candidate tested,but multi-character crops and real detector false
positives have not been evaluated. Its28% retention CER does not retroactively
pass the raw-training gate:compare V30 with the same decoder before promotion.
See [decoder analysis](../../outputs/recognition/evaluation/v31_onechar_blank_decoders/analysis.md).

## Current decisions and outstanding work

- Keep V30 control as the recent adaptation reference;preserve V31 curated
  step2000 as an experimental candidate. Respect the checkpoint distinctions above.
- Curated short terms are promising;no long training was authorized or launched
  after these pilots. Repeat with another seed/larger frozen one-character
  retention set before accepting the curriculum broadly.
- Keep standard greedy decoding in active code. Evaluate the empty-only fallback
  on matched V30/V31,multi-character crops and detector false positives before
  integrating it. It is not a replacement for temporal CTC blanks.
- No independent real-handwriting benchmark,mobile export/device latency or
  lexicon-assisted end-to-end accuracy has been established. Provisional YOLO
  recognition remains around90% CER with zero exact lines.
- GitHub source/reports are published at
  [ccouu/Tanzaku-OCR](https://github.com/ccouu/Tanzaku-OCR),branch `main`. Dataset
  images,labels/curated training lists,databases,weights,logs,environments and the
  archive remain local and ignored. Source under `src/data/` is versioned.

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
