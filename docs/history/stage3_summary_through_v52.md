> Historical snapshot preserved by the documentation audit. “Current/latest/running” statements below apply only to the earlier snapshot. Use [current recognition status](../stage3_recognition/current_status.md) now.

# Stage 3 — implementation and experiment summary

## Current state — 2026-11-13

**Promoted model: V43 final raw2000. Latest completed experiment: V50, not promoted.**
V50 completed two matched 2,000-update normalized-parenthesis arms. BiLSTM1e-5
versus frozen control improves clean bracket CER 59.88% → 47.46%, but new33
Latin recall falls 33.69% → 23.25% and 1-char CER rises 33% → 37%. YOLO CER
41.81% → 40.67% is inconclusive under menu bootstrap; openings stay 0/13.
No BiLSTM checkpoint passes guards. Control best is raw1,500; no promotion.
Independent raw/EMA weights, data/hash and 521-term audits passed. No training
is running. See [V50 results](../stage3_recognition/v50_normalized_bilstm_matched.md).
V49completed2,000updates/64,000presentations in25.70min,only66rowschanged;allshared
featuresincludingBiLSTM/BNfrozen. MultifontheldLatinrecall44.68%,largelyexisting19
letters(63.66%);new33still33.75%. Digits86.72%vs87.72%source,YOLO42.02%vs41.61%,
realbrackethits3/26unchanged. Best=raw1500,source-relativeguardspassbut11absolute
gatesfail. V49 and V50 are complete. See [V49results](../stage3_recognition/v49_selective_head_pilot.md).
V47tinyLatinfittingcompleted1,000updates:1.05%trainCER(202/208exact),but42.5%
samefontnewlayout/67.31%heldfontCERandcatastrophicretentionwithoutreplay/KL.
V48parenthesescompleted2,000updates:finalrawYOLOCER39.94%vsV43's41.61%,digitrecall
85.42%vs87.72%violatesguard. SelectedbestisEMA1000,41.19%YOLO/86.71%digitrecall,
source-relativeguardspassbutabsolute9FP16/10BF16gatesfail. Notrainingrunning.
See [V47/V48 results](../stage3_recognition/v47_v48_isolated_tests.md).
User explicitly promoted V43 on2026-11-13: matched156-crop YOLO CER51.82%→41.61%,
exact24→50, with historical/short-text regression accepted. V43fails11existing
retention gates both precisions; this specific override does not change thresholds.
See [promotion decision](../stage3_recognition/v43_promotion.md). V38 is preserved for rollback.
V45 head-only Latin completed1,000updates: only33appended rows trained, exact
40%Latin/40%synthetic/20%YOLO; oldweights/BNaffine/147buffers unchanged.
LatintrainCER74.34%,heldoutrecall23.92%,noqualifyingcheckpoint.
See [V45 results](../stage3_recognition/v45_latin_head_only_pilot.md).
V46completed4,000updates/128,000presentations in44.04minutes;exact30%Latin/10%paren/
15%digits/5%basicJapanese/20%YOLO/20%synthetic. FrozenCNN/temporal/BN verified.
Parenthesismacrorecall23.05%→44.68%;Latinheldout23.88%stillweak;digits/basicJapanese
regress. YOLOCER41.61%→41.29%raw,CIcrosseszero,exact50→48/156. No selectedbest;
keepV43. Cleanupremoved12.5094GiB;activeinputs/modelsverifiedpreserved. No trainingrunning.
See [V46 results](../stage3_recognition/v46_focused_recurrent_training.md).
See [compact handoff](../stage3_recognition/HANDOFF.md) for exact paths and operating instructions.

| Model | Role | Checkpoint / status |
|---|---|---|
| V38 final raw686 | Preserved rollback recognizer | `checkpoints/menu_recognizer_v38.pt`; 1,722 characters |
| V41 EMA1000 | Expanded-vocabulary, retention-passing development reference | `v41_new_rows_only_2000/best.pt`; 1,779 characters |
| V43 final raw2000 | User-promoted primary recognizer; explicit retention override | `checkpoints/menu_recognizer_v43.pt`; 1,779 characters |
| V44 | Completed Latin/parentheses/extra-long pilot; retention fails | `v44_latin_parentheses_long_2000/`; 1,812 characters; best.pt is expanded V43 at update0 |

V43 completed 2,000 updates / 64,000 presentations with 30% digit anchors,
25% YOLO, 20% long512, 10% new-character and 15% ordinary replay. All 6,000 digit
training images, all 1,561 YOLO bases, all 521 exact terms and 147 fixed BN buffers
were verified. ASCII digits 0–9 already existed at IDs 3–12; the weakness was
recognition. Full-width decimal digit labels now map to ASCII without changing
Japanese numerals, punctuation, existing IDs or source files.

| V43 FP16 result | V42 source | Final raw | Final EMA |
|---|---:|---:|---:|
| YOLO holdout CER, 156 crops |45.46%|41.61%|41.92%|
| Heldout-font digit macro recall |69.28%|87.72%|86.71%|
| YOLO holdout digit macro recall |57.81%|65.52%|65.00%|
| Seen-font CER |5.94%|7.00%|6.82%|
| Historical-crop CER |12.53%|13.88%|13.71%|
| Fresh short clean CER |28.83%|31.33%|31.00%|

Menu exact matches improve 43 → 50/156; paired menu-bootstrap CER change is
−3.86pp, 95% CI [−5.98, −1.92]. Deletions/insertions decrease, but synthetic
short-text retention worsens. Final raw fails 11 gates in both precisions;
`best.pt` is still the unchanged V42 fallback at V43 update0. It is not a trained,
gate-passing V43 model. Full results: [V43](../stage3_recognition/v43_digit_price_pilot.md).

V44 starts explicitly from V43 final raw. It appends 33 missing ASCII Latin
letters, giving all A–Z/a–z, with blank0 and all previous IDs preserved. Its
64,000-presentation plan is 30% YOLO / 15% Latin / 5% dedicated parentheses /
15% digits / 15% extra-long16/20/24-character lines / 5% existing long512 /
5% existing new-character replay / 10% ordinary replay. Parentheses actually
occur in 7.26% of planned samples. All 521 exact terms remain covered.
First200 updates train appended rows only; then lower-LR full fine-tuning retains
147 frozen BN buffers. Geometry/GPU checks, FP16/BF16 source parity, actual consumed counts and appended-row
updates were subsequently verified at completion.

For a 374×34 crop, H64 proportional width is704. Width512 reduces strip height
to approximately47, whereas768 preserves64. The 120-line synthetic geometry probe
gave heldout-font CER87.75% at512,66.17% at768,65.42% with overlapping512windows.
The 0.75pp overlap benefit did not meet the predeclared >1pp rule, so V44 uses768.
There were no eligible extra-long YOLO holdout crops for this comparison. Native
real-menu evidence and mobile performance are still required; wider input is not
proven to solve long lines. See [V44](../stage3_recognition/v44_latin_parentheses_long_pilot.md).

Completed milestones: V33–V36 established curated short-label coverage, fixed-BN
continuation, checkpoint averaging and initial real-crop adaptation; V37 extended
long512 recognition; V38 was promoted after stronger YOLO adaptation. V39's full
BN-stat swap failed. V40 expanded vocabulary but exposed an initialization flaw;
V41 corrected it with blank-relative initialization and new-row-only learning.
V42 improved menu/long recognition through crop geometry and replay, with retention
cost. V43 targeted digits; V44 tested remaining alphabet/punctuation/length gaps; only the extra-long synthetic
target improved substantially, with additional retention costs.

Do not compare V38's original **135-crop** holdout CER41.19% directly with the
expanded **156-crop** results without matching the evaluation subset. All reported
CER here is raw CTC, separate from fuzzy-dictionary or translation accuracy.

## V44 completion: narrow long-line gain, unsuccessful overall

Final raw versus V43: Latin heldout CER101.62%→99.58% (macro recall0.28%→0.67%);
parentheses-symbol recall23.05%→23.74%; YOLO CER41.61%→41.50%, paired95%CI
[−1.24,+1.33]pp. Extra-long768 CER66.33%→48.17%, overlap51248.00%; historical
CER13.88%→17.53%, heldout fonts6.95%→9.27%, old long512clean1.24%→2.30%.
Menu exact lines fall50→44/156, with deletions219→243. Final EMA does not resolve
these tradeoffs. All trained candidates fail retention; best.pt stays the expanded
V43 initialization, not a trained or eligible V44 checkpoint.

Latin failure also occurs on208training examples (99.86%CER). Diagnostic decoding
with only Latin+space+blank still produces97.84%CER, dominated by deletions. New
rows updated, but Latin fitting/alignment remains inadequate. Parentheses-line CER
improves while literal symbol recall hardly changes; full-width opening remains
0/47. Current YOLO holdout has no reference occurrence of the33appended letters,
so new Latin real-menu transfer cannot be inferred. No eligible real extra-long
holdout existed either. Both gaps need appropriate real evaluation examples.

Execution audit verified2,000updates/64,000presentations,33append-only IDs1780–1812,
all1561YOLO train bases,521exact terms, full-width-digit normalization and147fixedBN
buffers. Old/new optimizer counters1800/2000support the200-update new-row warmup,
which also had runtime old-weight assertions. See [V44 results](../stage3_recognition/v44_latin_parentheses_long_pilot.md).
At V44 review, V43 remained the research baseline and V38 was promoted. This
historical decision was superseded by the explicit V43 promotion above.

## Earlier completed adaptation experiments

The following promotion recommendations describe decisions at the time of each
experiment. The current primary model is V43, as recorded above.

V42completed2000updates:YOLOholdout51.82→45.46%raw,exact24→43/156,
heldfontlong512mean6.64→1.88%. Exact35%YOLO22400,30%true512long19200,
34%totalphysical512;2–5%positiveedgepadding/.95–1.05stretch,no added optical
YOLOnoise. SourceV41EMA1000;147BNbuffersfixed,98affinetensorsupdated.
Historical10.69→12.53%,shortretentionfails;best.ptretainsV41source,noV42
promotion. VerticalCERalso+1.12pp;menu gains primarilyhorizontal. KeepV38.
See [V42 results](../stage3_recognition/v42_yolo35_crop_jitter_plan.md).

V41completed2000updates: correctedinitialization exactparity,all oldweights/
BNaffine/147buffers unchanged,only57new classifier rows learned. Heldfontnewchar
recall85.67%/precision90.99%raw,vsV40recall53.51%; all57classeshaveaheldfont
success. Selectedbest=EMA1000,R52.05%,all17gatesbothprecisionspass. Finalraw/
EMA failBF16onechar62/200vs60(addedoutputs火→久火,株→抹). YOLOholdoutCER
unchanged51.82%,only1/36newcharoccurrencesrecognized. No promotion;keepV38.
See [V41 results](../stage3_recognition/v41_new_rows_only_plan.md).

V40 completed2000updates with57append-only new characters (1779total); all
received>=532actual encounters and all57classifier rows updated. Heldout-font
newcharrecall53.51%, but oldsynthetic retention regressed substantially and no
trained checkpoint passed revisedgates. Promoted model remains V38. Full156
YOLOholdout51.82→47.97%raw; confidence interval includes no improvement.
Initialization flaw:constant-12newrows could win over weak oldlogits, emitting
untrained spaces at update0. Best.pt is initialexpandedV38,nottrainedV40; do not
deploy it as unchangedV38. See [V40 results](../stage3_recognition/v40_new_characters_plan.md).

Latest diagnostic V39: full forward-only BN-stat recalibration of V38 using
1336YOLOtrainingcrops catastrophically regressed recognition (YOLOholdout
41.19→84.28%FP16;40.90→84.43%BF16). Zero weight/affine updates;147BNbuffers
changed. Candidate rejected, promoted V38 unchanged. See
`outputs/recognition/pilots/v39_v38_yolo_bn_swap/analysis.md`.

The reusable `src/recognition/` architecture remains V27. **V38 final raw at
update686 is now the user-promoted recognizer**, at
`checkpoints/menu_recognizer_v38.pt`; registry `src/recognition/current_model.json`
and loader `src/recognition/current.py`. Greedy CTC decoding remains beta0.

V38 used four full passes of1,336YOLOtrain crops at25%of21,376presentations,
with135heldout crops from13disjoint menus. Holdout CER51.53→41.19%; all521exact
synthetic terms/pass and147fixed BN buffers verified. Both precision gates pass
under user-authorized `menu_adaptation_v2`: only twochar cap changes13.5→14.0%,
the other16gates remain unchanged. Adoption is retrospective. V36 and original
V38best.pt remain preserved. See [V38 run and promotion](../stage3_recognition/v38_yolo25_training.md).

The complete V1–V28 history and retired source/reports are local in the
[dated archive](../../archive/stage3_legacy_20261111/README.md). Archived dataset
images and ZIP copies were subsequently deleted to reclaim7.08GiB;labels and
metadata were retained. Active V27 training/validation data were verified
unchanged. The cleanup itself did not train or alter model weights.

For a compact operational state,see [HANDOFF.md](../stage3_recognition/HANDOFF.md).

## Architecture

H64 RGB → ImageNet-initialized MobileNetV4 Conv-Small, horizontal stride4 →
height mean-pooling →960-to-256 temporal projection → binary orientation
conditioning → two residual depthwise-separable1D temporal blocks → packed
residual BiLSTM,128 hidden units per direction → vocabulary-sized CTC head.
The promoted V38 head has1,722 recognized characters plus blank; V40–V43 have
1,779 + blank, and the V44 candidate has1,812 + blank. Head size follows the
embedded checkpoint vocabulary; the encoder/temporal/BiLSTM topology is unchanged.
Originally vertical crops
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
latency remain outstanding. No model had been promoted at V27 completion;
V38 was subsequently promoted as described above.

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

## Historical V29–V31 font and terminology diagnostics

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

## Historical V31 one-character decoding test

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

## Historical V31 decisions (superseded by current state above)

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
the active training API. See [README](../stage3_recognition/README.md) for current commands.
