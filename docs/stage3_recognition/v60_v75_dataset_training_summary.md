> This report is scoped to the named versions. Later vertical coverage, YOLO adaptation, geometry corrections and current checkpoint roles: [current status](current_status.md).

# V60–V75: datasets, purpose and weight setup

All runs completed on Colab A100. None was automatically promoted. V60 introduced the learned height-collapse architecture: Conv2d(960,256,kernel=(2,1),stride=(2,1)), replacing V53's height mean pooling plus Conv1d projection. Temporal resolution and the residual BiLSTM with 128 hidden units per direction stayed unchanged. V61–V65 continued this architecture. Vocabulary: 1,779 characters plus blank 0.

Main lineage: **V60 random initialization → V61 → V62 → V63 → V65**. V64 was a separate matched pilot from V63; its selected checkpoint was not used to initialize V65.

## Dataset and purpose

| Version | Training dataset | Purpose and actual training |
|---|---|---|
| **V60** | Exact V53 128,000-record corpus: 20% parentheses, 20% digits/prices/counts, 20% meaningful 1–2-character labels, 15% long lines, 25% ordinary menu text. Appearance: 50% clean, 40% mild augmentation, 10% morphology. Reused V53's renderer, 15 training fonts and known/held-out evaluation manifests. | Isolate learned height collapse versus mean pooling, training from random weights under V53's matched recipe. **Ten full epochs / 10,020 updates / 1,280,000 presentations.** |
| **V61** | Exact V54 128,000-record corpus: 15% isolated digits, 15% meaningful single Japanese characters, 20% repeated digits/numbers/prices, 50% V53 replay. Corpus and render-manifest hashes match V54. | Test whether V60 adapts better than V53 under the same focused recipe. **2,000 updates / 64,000 presentations**, a partial corpus pass. |
| **V62** | Exact V55 160,000-record corpus: all 128,000 V53 records (80%) plus 32,000 compatible parenthesized phrases (20%). Appearance: 50% clean, 45% mild augmentation, 5% morphology. | Recover parentheses and long lines after V61's focused adaptation. **One full epoch / 5,002 updates / 160,000 presentations.** |
| **V63** | 160,000 images: 30% prices, equally split across 1–4 numeric digits; 5% standalone Japanese numerals 一–十; 65% broad replay. Price anchors pair numbers with either `¥` or `円`. Appearance: 50% clean, 45% mild, 5% morphology. | Recover digits/prices while retaining V62's broader recognition. **Five full epochs / 25,010 updates / 800,000 presentations.** |
| **V64 — matched arms** | One shared 128,000-record corpus: 20% valid one-character labels, 20% valid two-character terms, 60% replay equally divided among prices, parentheses, ordinary text and long lines. Identical hashed batch plan and rendered inputs for both arms. | Compare recurrent/head-only adaptation against full fine-tuning for meaningful short menu terms. **2,000 updates / 64,000 presentations per arm**; neither arm ran a full corpus pass. |
| **V65** | 160,000 images: 20% one-character labels, 20% two-character terms, 20% prices, 15% parentheses, 10% ordinary text, 15% long lines. Includes 8,000 Japanese numeral samples. Targets actual glyph-height variation and independent safe margins. Six procedural menu-background families: 25% plain, 50% light texture, 25% moderate texture, with readable ink contrast. Appearance: 50% clean, 45% mild, 5% morphology. | Address small-glyph, margin and textured-background weaknesses, rather than simply oversampling small buckets. **Five full epochs / 25,010 updates / 800,000 presentations.** |

All training images were rendered remotely using 15 training fonts. Held-out fonts remained excluded. Inputs use aspect-preserving height-64 RGB normalization, width buckets 128/256/384/512 and edge-replicated right padding. Valid sequence lengths exclude artificial right extension. No YOLO crops were used for training; menu-crop results are evaluation only.

## Starting weights and trainability

| Version | Starting checkpoint/state | Trainable versus frozen weights | Base learning rates |
|---|---|---|---|
| **V60** | Fully random initialization; no ImageNet or trained recognition weights. Same base initialization seed as V53, with isolated RNG for the new height projection. | All learnable parameters, including BN scale/bias, train; **BN running statistics also update** | All modules `3e-4` |
| **V61** | V60 best **raw**, epoch 10 | All learnable parameters, including BN scale/bias, train | Encoder `5e-7`; all other modules `3e-6` |
| **V62** | V61 step 2000 **raw** | All learnable parameters, including BN scale/bias, train | Encoder `2e-7`; all other modules `1e-6` |
| **V63** | V62 epoch 1 **raw** | All learnable parameters, including BN scale/bias, train | Encoder `2e-7`; all other modules `1e-6` |
| **V64 Arm A** | V63 epoch 5 **raw** | Train residual BiLSTM **and its output projection**, plus classifier. Freeze encoder, height projection, orientation conditioning, temporal blocks and their BN scale/bias. | Recurrent branch `1e-5`; classifier `1e-6` |
| **V64 Arm B** | Same V63 epoch 5 **raw**, independently initialized from source | All learnable parameters, including BN scale/bias, train | Encoder `2e-7`; height projection/conditioning/temporal blocks `1e-6`; recurrent branch `1e-5`; classifier `1e-6` |
| **V65** | V63 epoch 5 **raw**, not V64 selected weights | All learnable parameters, including BN scale/bias, train | Encoder `2e-7`; all other modules `1e-6` |

**V60 BN policy:** all 49 BatchNorm layers train and their 147 running buffers, including mean, variance and batch counters, update. Startup smoke verified updates in every module group and all 147 BN buffers. V60 adds 245,760 parameters versus V53, increasing the model from 2,627,924 to 2,873,684 parameters; 308 unchanged initial-state tensors matched V53 exactly.

**V61–V65 BN policy:** freeze all 147 running buffers across 49 BatchNorm layers, including mean, variance and batch counters, to the respective source checkpoint. Raw and EMA buffers remain unchanged. Frozen running statistics do **not** mean frozen CNN weights or frozen BN scale/bias; V64 Arm A is the explicit weight-freezing exception above.

**V60 optimization:** fresh AdamW, effective batch 128, base LR `3e-4` throughout the model, 5% warmup (501 updates) and cosine decay to a final factor of 0.05 over 10,020 updates; EMA decay 0.999. These are from-scratch settings, distinct from V61's much smaller adaptation learning rates.

**V61–V65 optimization:** fresh AdamW, batch 32 with microbatch accumulation as needed, 200-update warmup then cosine decay to 0.1 of base learning rate over the run's actual update horizon, and EMA decay 0.998. Shared across V60–V65: gradient clipping at 1, BF16 training with FP32 CTC, weight decay 0.01 on matrix/convolution weights and zero on normalization parameters and biases. No teacher KL was used.

## Outcome and checkpoint distinction

| Version | Main finding / checkpoint status |
|---|---|
| **V60** | Mixed architecture trade-off against matched V53 epoch 10 EMA: lower CER on textured application images (29.15% vs 34.48%) and YOLO holdout (55.16% vs 57.14%), but worse legacy held-out digits (36.34% vs 21.59%). **Best raw epoch 10** had remote held-out CER 6.84% and initialized V61; epoch 10 EMA scored 6.79% but was a separate state. No promotion. |
| **V61** | Stronger short-text adaptation than V54, but parentheses/long-line retention failed. No all-gate-passing checkpoint; final raw step 2000 started V62. |
| **V62** | Recovered parentheses and long lines, but digits and one-character recognition regressed. No eligible selected checkpoint; final raw epoch 1 started V63. |
| **V63** | Recovered prices/digits and improved parentheses; **epoch 5 raw** selected under V63 diagnostic guards. |
| **V64** | Full fine-tuning outperformed recurrent/head-only adaptation, but overall short-text gain over source was small. Selected **Arm B step 1500 EMA**; V65 nevertheless started from V63. |
| **V65** | **Epoch 1 EMA** passed all retention guards. **Final epoch 5 EMA** improved small glyphs, textured backgrounds and horizontal menu crops further, but failed parentheses retention. Selected and final states are different trade-offs. |

Important coverage limitation: **V60–V65 training contained zero native-vertical / orientation-flag-1 examples.** Later validation found approximately 91% CER on held-out synthetic vertical text for V65, versus approximately 4.4% on matched horizontal controls. Horizontal benchmark gains therefore do not establish vertical recognition capability.

## V66–V70: native vertical coverage and balanced restart

These runs kept V60's learned height projection, width stride 4, one residual packed BiLSTM (128 hidden units per direction), and the same 1,779-character vocabulary plus blank. Native vertical means upright glyphs stacked top-to-bottom, then rotated 90 degrees counterclockwise into a horizontal recognition strip with `orientation_flag=1`; horizontal inputs use flag 0. Changing the flag alone does not create vertical training data. Synthetic images were generated on Colab, with H64 inputs and width buckets 128/256/384/512. No YOLO crops entered training.

V66 and V67 started independently from **V65 epoch 5 EMA**, not from each other. The subsequent main lineage restarted from scratch: **V68 random initialization → V69 step 500 EMA → V70 step 1000 EMA**. Neither V69 nor V70 had an eligible selected checkpoint; these were explicitly reported continuation candidates.

| Version | Dataset, purpose and actual training | Starting weights and learning setup |
|---|---|---|
| **V66** | 160,000 V65-style records: 20% one-character, 20% two-character, 20% prices, 15% parentheses, 10% ordinary and 15% long text. Each stream uses 30% native vertical / 70% horizontal, totaling 48,000 vertical and 112,000 horizontal records. Includes 8,000 Japanese numeral records and all 591 curated terms. V65-style procedural backgrounds and readable ink contrast; 50% clean / 45% mild / 5% morphology. **Five full epochs / 25,010 updates / 800,000 presentations.** Introduce missing vertical coverage. | V65 epoch 5 **EMA**. All learnable weights and BN affine train; all 147 BN buffers frozen to source in RAW and EMA. Encoder `5e-7`, other modules `3e-6`; no teacher KL. |
| **V67 — matched arms** | Same V65-style category representation with 30% native vertical / 70% horizontal. Shared rendered corpus and batch plan across control and treatment; **2,000 updates / 64,000 presentations per arm**, not full passes. Test whether horizontal distillation reduces forgetting during vertical adaptation. | Both independently V65 epoch 5 **EMA**. All learnable weights and BN affine train; 147 BN buffers frozen. Encoder `5e-7`, others `3e-6`. Control uses CTC only; treatment additionally uses horizontal-only full-softmax teacher KL, weight 1, temperature 2, on valid frames. |
| **V68** | **196,000 records: 98,000 horizontal / 98,000 native vertical.** Actual V60 representation: 20% parentheses, 20% digits/prices/counts, 20% short labels, 15% long and 25% ordinary text; 50% clean / 40% mild / 10% morphology. **15 full epochs / 22,980 updates / 2,940,000 presentations.** Learn both orientations together from random weights. | Fully **random V60 architecture**, no trained source or teacher. All weights, BN affine and all 147 BN buffers update. AdamW `3e-4`, effective batch 128, 5% warmup, cosine floor 0.05, EMA 0.999. The completed run is `v68_v60_recipe_full15_colab`; earlier V65-source or V61-recipe V68 plans were stopped/superseded. |
| **V69** | V61 representation: 15% isolated digits, 15% meaningful single Japanese characters, 20% numbers/prices, 50% replay, now with **50/50 horizontal/native vertical**. **2,000 updates / actual 63,944 presentations**, a partial pass. Reapply short/digit adaptation to the balanced V68 model. | V68 selected **epoch 13 EMA**. All learnable weights and BN affine train; 147 BN buffers frozen. Encoder `5e-7`, other modules `3e-6`; no teacher KL. Continuation uses step 500 EMA, not final step 2000. |
| **V70** | 160,000 records: 80% V53-style broad replay plus 20% compatible parenthesized phrases, now with **50/50 horizontal/native vertical**. Appearance 50% clean / 45% mild / 5% morphology. **One full epoch / 5,002 updates / 160,000 presentations.** Reapply V62's broad/parentheses recovery after focused adaptation. | V69 **step 500 EMA** continuation. All learnable weights and BN affine train; 147 BN buffers frozen. Encoder `2e-7`, others `1e-6`; no teacher KL. Continuation uses step 1000 EMA, not final epoch state. |

V66/V67/V69/V70 use low-learning-rate adaptation with batch 32, 200-update warmup, cosine floor 0.1 and EMA 0.998. V68 uses the distinct random-start schedule above. Frozen BN buffers do not imply frozen CNN weights or frozen BN scale/bias.

| Version | Main outcome and checkpoint distinction |
|---|---|
| **V66** | Vertical synthetic CER improved from 91.01% to 69.79% RAW / 69.91% EMA; real vertical menus 101–130 improved from 89.24% to 60.76%. Horizontal synthetic CER regressed from 4.43% to approximately 9.6%. All ten RAW/EMA candidates failed horizontal retention; **selected=null**. |
| **V67** | **Treatment step 500 EMA** passed all nine guards in FP32 and BF16, but vertical synthetic CER remained 90.55%, versus source 91.01%. Horizontal retention was better than the matched CTC-only control. The small eligible gain did not establish useful vertical recognition. No automatic V68 fallback was triggered; V68 was subsequently authorized separately. |
| **V68** | Selected **epoch 13 EMA**, balanced full-length H/V CER 6.79% (H 6.12%, V 7.46%). Real menus 101–130 CER 28.08% overall (H 28.49%, V 22.15%). This was the effective balanced-orientation restart. |
| **V69** | No eligible selection. **Step 500 EMA** was retained as a continuation candidate: balanced H/V CER 7.22% (H 6.56%, V 7.87%), held-out digits 13.51%. This is not an all-guard-passing checkpoint or a promotion. |
| **V70** | No eligible selection. **Step 1000 EMA** was retained as a continuation candidate: balanced H/V CER 6.85% (H 6.07%, V 7.63%), held-out digits 15.35%, parentheses 21.41%. Later V71 continued this state; the final V70 epoch checkpoint is distinct. |

Balanced H/V figures above refer to the unchanged **2,400-image full-length synthetic holdout**, not later paired short-text probes. Native vertical writing and vertical translation jitter are different tests. All these results are research diagnostics, with no production promotion.

## V71–V75: short-text adaptation and YOLO domain transfer

Main lineage: **V70 step 1000 EMA → V71 epoch 5 EMA → V72 Arm B step 1500 RAW → V73 epoch 4 RAW → V74 epoch 2 RAW → V75 epoch 4 RAW**. V71 and V73 source states were continuation candidates, not all-guard-passing selections. Architecture and vocabulary remain unchanged from V60; these are weight/data adaptations, not the later descriptive stride-two or two-layer-BiLSTM architecture branches.

### Datasets and purpose

| Version | Dataset and representation | Purpose and actual training |
|---|---|---|
| **V71** | 160,000 records, 50/50 horizontal/native vertical. Replays V63: 30% prices equally divided among 1–4 digits, 5% Japanese numerals 一–十, 65% broad replay. Prices pair digits with `¥` or `円`. | Restore price/digit recognition after V70. **Five full epochs / 25,010 updates / 800,000 presentations.** |
| **V72 — matched arms** | Shared 128,000-record corpus, 50/50 horizontal/native vertical: 20% one-character, 20% two-character terms, 60% replay across prices, parentheses, ordinary and long lines. Identical rendered inputs and fixed batch plan across arms. | Repeat V64's recurrent/head-only versus full-model comparison on the balanced-orientation lineage. **2,000 updates / 64,000 presentations per arm**, not full passes. |
| **V73** | 160,000 records, 50/50 horizontal/native vertical: 20% one-character, 20% two-character, 20% prices, 15% parentheses, 10% ordinary, 15% long lines. V65-style glyph-scale variation, safe independent margins, six procedural menu-background families and readable font/background contrast; 50% clean / 45% mild / 5% morphology. | Improve small-glyph and margin robustness after V72. **Five full epochs / 25,010 updates / 800,000 presentations.** |
| **V74** | **1,561 old YOLO training crops + 4,683 augmented synthetic replay records = 6,244 records/epoch**, 25% real / 75% synthetic. Real training split: 117 menus. Menu-disjoint validation: 68 crops / 6 menus; test: 88 crops / 7 menus. Replay retains both orientations. | First YOLO-crop domain transfer on this lineage. **Two full epochs / 396 updates / 12,488 presentations.** Every real training crop encountered once per epoch. |
| **V75** | Add **492 eligible new training crops from 40 new menus** to the old 1,561: **2,053 real + 6,159 synthetic = 8,212 records/epoch**, still 25/75. New collection split: 40 train / 5 validation / 5 test menus. New validation has 66 OK crops; new test has 75. Old splits retained; pooled validation totals 134 crops / 11 menus. | Adapt to the new cleaner collection while retaining old crops. **Five full epochs / 1,290 updates / 41,060 presentations.** Baseline evaluation of the new data completed before training. |

**Coverage correction:** post-completion auditing found only **475/521 required standalone terms** in V72's corpus/batch plan and V73's corpus. Of the 46 missing standalone labels, 42 appeared only as substrings; four were absent entirely: **海胆, 軟骨, 雲丹, 馬刺**. The earlier `covered_521_terms=12` value was also an incorrect counter, not the true coverage. V74 and V75 restored **521/521 exact labels per epoch** and at least four views per priority term: two horizontal, two vertical, at least two fonts and clean/mild appearances. Do not describe the earlier runs as fully covered.

**YOLO data provenance:** the old collection is `data/processed/gemini_banana_mobile_menus_50_crops` (despite its name, menus 001–130); the new collection is `data/raw/gemini_banana_mobile_menus_train_50/yolo_crops`. The new collection contains 702 OK crops, of which 615 are in vocabulary; only eligible training records entered training. Preserve collection namespaces because menu numbers repeat. Labels are VLM-generated and not independently verified. OOV examples remain in full-OK evaluation; in-vocabulary metrics are separately reported.

**Geometry scope:** V74/V75 reused historical CTC crop exports, which can height-letterbox overwide crops. They did not yet implement V76's native-H64 splitting correction. Synthetic inputs retain H64, width buckets 128/256/384/512 and valid lengths excluding artificial right extension. Do not retrospectively claim all V74/V75 real inputs were native-H64 split windows.

### Starting weights and optimization

| Version | Source and trainable weights | Learning rates / BN policy |
|---|---|---|
| **V71** | V70 step 1000 **EMA** continuation; all learnable parameters and BN affine train. | Encoder `2e-7`, all others `1e-6`; 147 BN buffers frozen in RAW and EMA. |
| **V72 Arm A** | Independently V71 epoch 5 **EMA**; train residual BiLSTM and output projection plus classifier only. Freeze encoder, height projection, orientation conditioning, temporal blocks and BN affine. | Recurrent `1e-5`, classifier `1e-6`; 147 BN buffers frozen. |
| **V72 Arm B** | Independently the same V71 source; all learnable parameters and BN affine train. | Encoder `2e-7`, projection/conditioning/temporal `1e-6`, recurrent `1e-5`, classifier `1e-6`; 147 BN buffers frozen. |
| **V73** | V72 Arm B step 1500 **RAW**; all learnable parameters and BN affine train. | Encoder `2e-7`, all others `1e-6`; 147 BN buffers frozen. |
| **V74** | V73 epoch 4 **RAW** continuation; all learnable parameters and BN affine train. | Encoder `2e-7`, others `1e-6`; 147 BN buffers frozen and copied exactly in RAW/EMA. Fresh AdamW/schedule. |
| **V75** | V74 selected epoch 2 **RAW**; all learnable parameters and BN affine train. | Encoder `2e-7`, others `1e-6`; 147 BN buffers frozen and copied exactly in RAW/EMA. Fresh five-epoch AdamW/schedule, warmup 64 updates. |

All use batch 32, BF16 forward with FP32 CTC and EMA decay 0.998. **Actual V72/V73 legacy optimizer settings were weight decay `1e-4` and gradient clipping 5**, differing from the intended proper-decay recipe; preserve that historical distinction. V74/V75 explicitly use corrected weight decay 0.01 for matrix/convolution weights, zero for normalization/bias, and clipping 1. No teacher KL in V71–V75. Frozen running statistics are distinct from trainable BN affine and CNN weights.

### Results and checkpoint roles

| Version | Checkpoint and main outcome |
|---|---|
| **V71** | Epoch 5 **EMA continuation**, not eligible. Fixed 2,400-image H/V CER **6.44%** (H 5.40%, V 7.49%); held-out digits 6.83%. Real menus 101–130 overall CER 28.20%. |
| **V72** | **Arm B step 1500 RAW**, eligible under its original guards. Fixed 2,400-image H/V CER **6.06%** (H 5.02%, V 7.10%). Real menus 101–130 CER 27.24%. |
| **V73** | Epoch 4 **RAW continuation**, not eligible because original-bucket retention failed. Fixed 2,400-image H/V CER **6.41%** (H 5.56%, V 7.26%). Real menus 101–130 CER **21.07%** (H 21.43%, V 15.82%). The separate paired 1,920-image short probe scored 8.89%; that is not the full-length H/V metric. |
| **V74** | Selected **epoch 2 RAW**, eligible. Old validation CER **45.57%**, old test **35.30%**, versus source 48.96% / 37.57%. Fixed H/V holdout 6.69%. |
| **V75** | Selected **epoch 4 RAW**, eligible, ranked on pooled validation only: **40.97%**. Old test **34.43%** (32/88 exact); new test **34.82%** (28/75 exact), versus pretraining new-test 40.18%. Fixed H/V holdout 7.12%. Source SHA prefix `275d530bc4df`. This selected state differs from epoch 5 final RAW/EMA. |

**V75B — separate BN-unfrozen comparison:** independently initialized from V74 epoch 2 RAW with the same reported V75 data/budget, but allowed all 147 BN buffers to update. Epoch 5 RAW was an experimental continuation, not eligible: pooled validation 43.19%, old test 34.96%, new test 42.86%, with retention failures. It did not replace V75 or initialize V76. This supports retaining frozen BN statistics for this adaptation recipe, not a universal claim that BN updates are harmful.

Most menus 101–130 are now part of the training split: 367 of 404 eligible crops train, only 37 are held out. Full-collection scores therefore mix fitting and generalization. Report precision and preprocessing explicitly: the V75 old-test figure above is the historical export evaluation; later native-H64 splitting gives 34.09% on the same source checkpoint. No model promotion followed these experiments.

Sources: each experiment's `HANDOVER.md`, evaluation reports, training code and audits under `outputs/recognition/experiments/`, including V73's `post_completion_audit/`. Vertical validation is under `outputs/recognition/evaluation/v65_vertical_synthetic_holdout/`. For subsequent geometry and spacing experiments see [V76–V77](v76_v77_dataset_training_summary.md); for current checkpoint roles see [current status](current_status.md). These reused development benchmarks and provisional VLM menu labels are not a blind, human-verified production test.
