# L2 lineage: L2_v01–L2_v06 and the latest 1901-character checkpoints

Updated **2026-11-21**. The filename is retained for existing links. The original L2_v01–L2_v02 record below is preserved; later sections extend it through dish-lexicon adaptation, Tabelog training and vocabulary expansion.

**Start here: [L2_menu1901_digit_recovery and exact loading assets](preferred_checkpoint.md).** This is the stable name for preferred research candidate **`l2_1901_digits4_real40_1000`, Stage A epoch 3 RAW**, packaged in `checkpoints/l2_menu1901_digit_recovery/`. It is not retention-eligible or promoted. The registry remains **L2_v06 epoch 2 EMA**; the inspector's separate override remains **dishlex50 epoch 5 RAW**.

## Foundation record: L2_v01–L2_v02

Verified 2026-11-20 against the saved corpus manifests, training audits and evaluation reports, with a read-only cross-check by AGY. This report covers the foundation of the current L2 lineage, not the older sequential V60–V75 lineage. **L2 means lineage 2, not two BiLSTM layers.** L2_v01 was formerly named H4 V81; historical filenames inside its artifacts retain `v81`.

## Lineage and architecture

**Random weights → L2_v01 Phase1 epoch2 RAW → L2_v01 Phase2 epoch8 EMA → L2_v02.** Later L2_v03 continued from **L2_v02 epoch4 RAW**, not final epoch5 or an EMA state.

All three stages use MobileNetV4 Conv Small → learned Conv2d(960,256,kernel=(4,1)) height collapse → orientation conditioning → two residual temporal convolution blocks → **one packed residual BiLSTM, 128 hidden units per direction** → CTC classifier. Encoder height H4, delayed horizontal stride4, **3,365,204 parameters**, **1,779 characters plus blank0**. No architecture or vocabulary expansion occurred between these stages.

Phase1's saved random-initialization artifact was checked by AGY: RAW and EMA initially equal; BN means zero, variances one and counters zero. This is a fully random backbone/head start, with no ImageNet or earlier recognition checkpoint inherited. Phase2 and L2_v02 initialize from trained weights with a fresh optimizer (`resumed_from=null`); they are not full-state optimizer resumes.

## Dataset and purpose

| Stage | Fixed corpus per epoch | Purpose and actual training |
|---|---|---|
| **L2_v01 Phase1 — glyph foundation** | **160,000 synthetic images:** 40% single-character (64,000); 15% each random 2-, 3-, 4- and 5-character strings (24,000 each). **80,000 horizontal / 80,000 native vertical.** | Learn glyph identity and CTC alignment before menu-language structure. **2 full passes / 5,004 updates / 320,000 presentations.** Random strings are deliberate visual exercises, not meaningful phrases. |
| **L2_v01 Phase2 — menu integration** | **160,000 synthetic images:** 70% V71-menu representation (112,000), 10% single-character (16,000), 20% random 2–4-character strings (32,000: 10,668 / 10,666 / 10,666). **80,000 H / 80,000 V; 521/521 exact curated terms.** | Add menu sequences while retaining explicit glyph replay. **8 full passes / 20,016 updates / 1,280,000 presentations.** V71 identifies the data recipe, not the source weights. |
| **L2_v02 — digits and Japanese numerals** | **128,000 synthetic images:** 15% digits (19,200), 15% kanji-numeric text (19,200), 35% random strings (44,800), 35% dish vocabulary (44,800). Random lengths **1/2/3/4/6/8/12/16**, 5,600 records each. **64,000 H / 64,000 V; 521/521 exact curated terms.** Audit records 7,922 lines containing `〇` and 3,696 containing `零`. | Improve numbers, Japanese numerals and short text while retaining broad character/dish coverage. **5 full passes / 10,005 updates / 640,000 presentations.** |

Across the three stages: **15 phase-specific full passes, 35,025 updates and 2,240,000 presentations**. These are three different corpora, not 15 passes over the same images. **No YOLO crops were used for training.** Camera/YOLO results below are evaluation only.

### Rendering, orientation and preprocessing

- **15 font binaries** in each `payload/fonts/train/` directory; directory totals of 38 files include licenses/metadata and must not be reported as 38 fonts. Fonts span digital, Mincho, brush and handwriting styles. Individual font choice is deterministic from render seeds but is not stored per manifest row.
- Six procedural background families: warm off-white paper, pale wood, dark wood, chalkboard, mottled card and photographic gradient, with readable foreground contrast and glyph-scale/margin variation.
- **H64 RGB**, proportional rendering/preprocessing, width buckets **128/256/384/512**, replicated right extension and valid lengths `ceil(valid_width/4)` for recurrence/CTC/decoding. Final per-row bucket assignments are not present in the corpus manifest. Do not infer counts from text length alone.
- Flag0 is horizontal. Flag1 is genuinely native vertical text, with upright glyphs stacked top-to-bottom and rotated counterclockwise into the recognition strip. It is not the same horizontal image with a different flag.

| Assigned appearance category | L2_v01 Phase1 | L2_v01 Phase2 | L2_v02 |
|---|---:|---:|---:|
| Clean | 40,145 (25.09%) | 68,200 (42.63%) | 38,555 (30.12%) |
| Mild | 56,284 (35.18%) | 66,912 (41.82%) | 57,642 (45.03%) |
| Erode | 31,785 (19.87%) | 12,599 (7.87%) | 16,424 (12.83%) |
| Dilate | 31,786 (19.87%) | 12,289 (7.68%) | 15,379 (12.01%) |

These counts are assigned manifest categories, not a new pixel-level quality audit. The actual morphology proportions differ substantially from earlier 5%-morphology recipes.

## Starting weights and trainability

| Setting | L2_v01 Phase1 | L2_v01 Phase2 | L2_v02 |
|---|---|---|---|
| Source | Fully random initialization | Phase1 epoch2 **RAW** | Phase2 epoch8 **EMA** |
| Learnable parameters | All 163 tensors | All 163 tensors | All 163 tensors, including BN affine |
| BN running statistics | **Update** | **Update** | **Frozen: all 147 buffers in 49 modules**, bit-identical to source |
| Base LR | All modules **3e-4** | All modules **3e-4** | Encoder **2e-6**; projection/conditioning/temporal/BiLSTM/classifier **5e-6** |
| Warmup | 250 updates | 500 updates | 200 updates |
| Cosine minimum LR factor | 0.50 | 0.05 | 0.10 |
| Reported training time | 346.98 s | 1,478.32 s | 716.58 s |

All used Colab A100, AdamW with **weight decay0.01 excluding normalization/bias**, BF16 training and FP32 CTC. The reported update totals imply roughly 64 presentations/update; bucket tails account for additional partial batches. EMA and gradient-clipping scalar settings are not explicitly recorded in the inspected training audits, so this summary does not guess them.

**Important BN distinction:** Phase1/Phase2 audits report zero entries in the *frozen-BN audit set*, not zero BN layers. Their BN statistics update. Conversely, L2_v02 freezes running means/variances/counters while still learning CNN weights and BN scale/bias. `frozen_tensor_count=0` is consistent with that policy.

## Results and checkpoint roles

The Phase2 epoch8 EMA column is the exact L2_v02 initialization state. Metrics are CER percentages from the saved L2_v02 evaluation, with cohorts kept separate; lower is better.

| Evaluation cohort | L2_v01 P2 epoch8 EMA | L2_v02 epoch4 RAW — downstream source | L2_v02 epoch5 RAW — final | L2_v02 epoch5 EMA — final |
|---|---:|---:|---:|---:|
| Clean 1-character | 7.00% | 6.50% | 6.50% | 6.50% |
| Clean 2-character | 6.25% | 5.00% | 5.00% | 5.00% |
| Noisy 1-character | 14.50% | 11.00% | 11.50% | 11.50% |
| Noisy 2-character | 11.75% | 10.00% | 10.00% | 10.00% |
| Held-out digits | 15.64% | **11.01%** | 11.23% | 11.23% |
| Held-out parentheses | 8.98% | 8.53% | 8.53% | 8.53% |
| Resized held-out fonts | 4.89% | 5.13% | 5.15% | 5.15% |
| Clean long512 | 3.96% | 4.84% | 4.74% | 4.76% |
| Noisy long512 | 7.60% | 8.66% | 8.64% | 8.78% |
| Old YOLO validation — 68 crops | 50.52% | 49.74% | 49.48% | 49.48% |
| Old YOLO test — 88 crops | 38.09% | 38.78% | 38.78% | 38.96% |
| New60 collection — 885 crops | 27.50% | 26.75% | 26.88% | 27.02% |

L2_v02 improves the intended digit/short-text probes, with some loss on long lines and resized fonts. It does not establish a broad camera-domain improvement. **Epoch4 RAW is the verified downstream continuation choice, not evidence of an independently predeclared universal selection gate.** Epoch5 remains the final trained checkpoint; its RAW and EMA are distinct states even though some metrics tie.

For L2_v01 itself, Phase2 epoch8 RAW/EMA are useful synthetic/short-text candidates; epoch6 RAW performed better on the reported camera cohorts. Phase1 is a foundation stage rather than a production recognizer. See the [detailed L2_v01 record](l2_v01_dataset_training_summary.md) for its epoch comparisons. No paired confidence intervals or fixed 2,400-image H/V benchmark were included in these saved reports. Do not compare their YOLO numbers with later val10/test5 or Char-LM results as if the inputs/decoder were identical.

## Evidence and artifact locations

All experiment directories are under `outputs/recognition/experiments/`:

| Stage | Experiment directory | Key saved evidence |
|---|---|---|
| Phase1 | `h4_k41_l2_v01_p1_rand160k` | `payload/h4_k41_random_init.pt`; `manifests/v81_p1_corpus_audit.json`; `downloads/h4_k41_v81_p1/train_audit.json` |
| Phase2 | `h4_k41_l2_v01_p2_v71mix160k` | `manifests/v81_p2_corpus_audit.json`; `downloads/h4_k41_v81_p2/train_audit.json`; `evaluation/h4_k41_v81_p2_eval_report.json` |
| L2_v02 | `h4_k41_l2_v02_digits_kanjinum_128k` | `manifests/l2_v02_corpus_audit.json`; `downloads/h4_k41_l2_v02/train_audit.json`; `evaluation/h4_k41_l2_v02_eval_report.json` |

Each corpus is recorded in `manifests/v74_replay_corpus.jsonl`. Checkpoint SHA inventories are in the corresponding download directories. Verified parent file SHA prefixes: initial random `5fadcbde…`; Phase1 epoch2 `df76b1a7…`; Phase2 epoch8 `e158276f…`; L2_v02 epoch4 `ff9c019f…`; L2_v02 epoch5 `5de4e474…`. A file hash identifies the package, so always also specify **RAW versus EMA**.

This is historical documentation only. The current promoted recognizer is **L2_v06 epoch2 EMA**, recorded in `src/recognition/current_model.json`; writing this report does not change the registry or any training asset.

## L2_v03–L2_v06: real-menu adaptation and hard-pair replay

These stages keep the H4/k4×1 architecture, 1779 characters plus blank0, and all 163 parameter tensors trainable, including BN affine. **All 147 BN buffers remain frozen**, verified for RAW/EMA. AdamW uses weight decay 0.01 excluding normalization/bias, BF16 forward and FP32 CTC. Sources below are actual initialization states, not necessarily a run's final or universally best state.

| Stage | Actual initialization | Corpus per full pass and purpose | Completed budget | Encoder / other-module LR |
|---|---|---|---|---|
| L2_v03 | L2_v02 epoch 4 RAW (`ff9c019f…`) | 21,900: 8,760 YOLO presentations (2,920 unique parents ×3; 40%) + 13,140 synthetic (60%). Introduce real-menu appearance. All 521 terms covered; restaurant split 225 train / 10 val / 5 test. | 5 epochs, 3,435 updates, 109,500 presentations | `2e-6 / 5e-6` |
| L2_v04 | L2_v03 epoch 5 RAW (`d12ce539…`) | 58,400: 11,680 YOLO (20%) + 46,720 synthetic (80%). Synthetic hard phrases, single characters and sequences plus vocabulary/digit/numeral/random replay; 23,360 H / 23,360 V. | 5 epochs, 4,575 updates, 292,000 presentations | `1e-6 / 2.5e-6` |
| L2_v05 | L2_v04 epoch 4 RAW (`19bf06df…`) | Same 58,400-record hard-pair/YOLO representation, continued for a longer schedule. | 10 epochs, 9,150 updates, 584,000 presentations | `1e-6 / 2.5e-6` |
| L2_v06 | L2_v05 epoch 10 RAW (`ef42aa1f…`) | 58,400: 11,680 YOLO, 23,360 `p5pct_pair_line` synthetic slots, 23,360 other synthetic replay. The slot name is not a claim that only 5% of all images contain pairs. | 4 epochs, 3,660 updates, 233,600 presentations | `1e-6 / 2.5e-6` |

**L2_v06 epoch 2 EMA** (`d1592d6b…`) was explicitly promoted; epoch 4 is the training final, not the registry state. Baseline canonical-greedy FP32: old val10 **5.71%** (107/129 exact), reused test5 **7.84%** (50/62). Later Char-LM R1 evaluation and trie suggestions are separate inference changes, not additional neural training. See [promotion](l2_v06_promotion.md).

Evidence roots under `outputs/recognition/experiments/`: `h4_k41_l2_v03_yolo_all240`, `h4_k41_l2_v04_hardpair_yolo20`, `h4_k41_l2_v05_hardpair_yolo20_ep10`, `h4_k41_l2_v06_p5pct_pairs_ep04`. Their `downloads/<stage>/train_audit.json`, `manifests/` and `evaluation/` establish sources, budgets and results. These runs do not all have a top-level `HANDOVER.md`.

## Dishlex and first Tabelog adaptation: still 1779 characters

| Experiment | Source and purpose | Data and completed budget | Checkpoint role |
|---|---|---|---|
| `l2_v06_dishlex50_yolo20_full5_120k` | Promoted L2_v06 epoch 2 EMA; improve meaningful dishes and phrases | 120,000/pass: 24k old YOLO + 24k standalone terms + 24k phrases + 48k broad replay. Synthetic 96k is 50/50 H/V with approximately 50% clean / 45% mild / 5% morphology, 15 actual font binaries, 521/521 curated terms. 5 epochs, 9,385 updates, 600k presentations. | **Selected epoch 5 RAW**, `359fb52c…`; current inspector override, not registry promotion. |
| `l2_dishlex_tabelog30_yolo20_synth50_1000` | Dishlex epoch 5 RAW; first adaptation to the manually collected 16112026 real menus | Fixed 64k: 19,200 new-real / 12,800 old-YOLO / 32,000 synthetic; 1,000 updates. Initially 362 eligible new-real parents; OOV/overlap/geometry/status exclusions retained. | **Selected step 500 RAW**, `ce33b259…`; final step 1000 remains separate. This checkpoint did not become the source of the independent vocabulary-expansion branches below. |

Both train all parameters with BN running statistics frozen. Dishlex used encoder `5e-7`, other modules `1e-6`. First Tabelog selection improved new in-vocabulary val CER **40.97% → 40.11%**, old val10 **4.87% → 4.04%**, with old test5 unchanged at **7.52%**. Test results were report-only. Original full-label evaluations retain OOV references rather than silently dropping characters.

## Vocabulary expansion and overlap policy

The append-only asset chain is **1779 → 1830 → 1901 characters**, blank0 and every existing ID preserved. The 1830 asset added 51 culinary-term-derived characters offline. The 1901 candidate then added 71 more from the **complete OK CTC labels**, rather than restricting coverage to LLM-extracted culinary terms. Relative to trained 1779 models, **122 new classifier rows** are required, yielding **1902 outputs** and 3,396,558 parameters.

Matching vocabulary: `data/processed/recognition/vocab_ctc_labels_16112026_v2.json`, SHA-256 `8ef5a6004cef9c187c88f2ef8a9461e35be5632b1d97b87d2c433f99558df0bc`. All 1087 OK references are in-vocabulary. This does **not** cover every Latin letter: uppercase `Q V` and lowercase `j p q w x` remain absent. Vocabulary inventory used all splits, including eight heldout-only added tokens; disclose **label-informed character inventory**, not a fully blind evaluation.

Minor-overlap eligibility was implemented before these runs: after duplicate/fragment handling, both crops must have strictly **<10% core intersection area fraction and <10% perpendicular overlap fraction**, with parallel axes within 8 degrees. Major/crossing/invalid overlaps remain quarantined. New-real TRAIN eligibility rose **362 → 542** parents. This did not admit every unresolved crop: 263 major-overlap, 100 over-512 and 243 non-OK parents remained quarantined. Full-width digits/parentheses are canonicalized; original labels remain preserved.

## 1901-character branches: dataset, weight policy and checkpoint lineage

New rows use blank-row weight initialization with bias offset −8; initial prediction parity and zero new emissions were checked. Full-softmax teacher KL uses weight 1, temperature 2 and valid frames on the declared replay streams; it does not force new-character/digit targets to match an incompatible old teacher. BN buffers are frozen in all stages below. New-row-only training also freezes old classifier rows, blank, BN affine and all other parameters; full-model stages allow BN affine and all other parameters to learn.

| Branch / stage | Actual source | Dataset and completed budget | Trainability / LR | Saved outcome |
|---|---|---|---|---|
| Original expansion A | Dishlex epoch 5 RAW, expanded independently | 100k = 80k synth (40k targeted / 40k old replay) + 20k old YOLO; 1 epoch / 1,563 updates | Only 122 new rows+bias, `1e-3`; old-only replay KL | `stage_a_final.pt` RAW (`9b889f9c…`); new122 recall 83.2%, 121/122 recognized. |
| Original expansion B | Original A final RAW | 64k = 30% new-real (542 eligible parents) / 20% old YOLO / 50% synth; 1,000 updates | All weights; encoder `5e-7`, other modules `1e-6`; replay KL | No eligible selection. Diagnostic step 250 RAW; final step 1000 RAW/EMA (`38ee58c6…` file). Digits remained above guard. |
| Independent A.1 | Fresh independent expansion of Dishlex epoch 5 RAW; **not** original B | Same 100k A corpus, no new-real crops; 4 epochs / 6,252 updates / 400k presentations | Only new rows+bias, `1e-3`; old-only KL | Final epoch 4 RAW (`e36a97dd…`), new recall 88.9% RAW / 89.5% EMA, all 122 recognized. |
| Independent A.2 | A.1 final epoch 4 RAW | Same 100k corpus; 2 epochs / 3,126 updates / 200k presentations | Encoder `5e-7`, BiLSTM `1e-6`, projection/condition/temporal `1e-5`, classifier `1e-4`; original-source old-only KL | Epoch 2 EMA (`062f31ea…` file) reaches 92.08% new recall. **Final EMA and epoch 2 EMA are the same state**, not two candidates. No eligible selection. |
| Digit recovery A | Independent A.2 epoch 2 EMA | 100k/pass: 20k YOLO (13.5k old + 6.5k new), 24k digit synth, 56k other synth. Four epochs / **6,260** updates / 400k presentations. All old parents and all 542 eligible new parents covered; 521 terms and 122 new characters retained. | All weights; A.2 differential LRs above. Frozen source A.2 teacher on YOLO+other replay (304k encounters), no KL on 96k digit encounters. | **Preferred research: epoch 3 RAW**, `7fcf4b34…`. Final is epoch 4 RAW/EMA (`dcf67d29…` file). Neither is retention-eligible. |
| Digit recovery B | Digit A **epoch 4 RAW**, not preferred epoch 3 | 64k = 25.6k new-real / 12.8k old YOLO / 25.6k synth; 1,000 updates | All weights; encoder `5e-7`, other modules `1e-6`; frozen A-final teacher on 38.4k replay samples, no KL on 25.6k new-real | Diagnostic step 500 EMA (`c79e4fbe…` file); newest final step 1000 RAW/EMA (`9c548a13…` file). No eligible selection. |

Digit A's 24k synthetic samples contain 6k single digits, 6k two-digit/repeated sequences and 12k one-to-four-digit prices, including currency and thousands separators. Its 56k other-character stream includes 9,167 parentheses lines per pass and at least 120 encounters per new character. **Stage A already trains on new Tabelog crops**; its gains are not zero-shot synthetic-only transfer. Synthetic generation retains the menu-like backgrounds, font/appearance variety and both orientations; it is not all meaningful phrases—explicit glyph, numeric and broad random coverage is deliberate.

### Latest comparable results and recommendation

Canonical greedy FP32 from the latest identical-cohort evaluation; percentages below are CER except the explicitly marked recall. R1 LM was unavailable for the 1901 vocabulary, so no stale 1779 decoder was silently used.

| Checkpoint | New122 recall ↑ | New val, 99 crops ↓ | Old val10 ↓ | Digits ↓ | Parentheses ↓ | Noisy 1-char ↓ | Long512 clean / noisy ↓ | New test, 83 crops ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original Dishlex RAW, 1779 | 0% | 49.42% | 4.87% | 8.44% | 6.89% | 17.00% | 2.82 / 6.36% | 38.47% |
| A.2 epoch 2 EMA | 92.08% | 47.99% | 3.90% | 9.77% | 8.08% | 20.50% | 2.70 / 6.20% | 35.32% |
| **Digit A epoch 3 RAW — preferred research** | 90.44% | **43.58%** | 3.62% | **5.73%** | 9.58% | 20.50% | 2.20 / 5.96% | 32.84% |
| Digit A epoch 4 RAW — B initialization | 90.71% | 44.10% | 3.48% | 5.87% | 9.58% | 21.00% | 2.12 / 6.04% | 33.00% |
| Digit B step 500 EMA — diagnostic | 90.57% | 43.84% | 3.48% | 5.95% | 8.98% | 21.00% | 2.16 / 6.08% | 33.17% |
| Digit B step 1000 RAW — final | 90.03% | 44.62% | 3.48% | 5.80% | 8.53% | 21.00% | 2.44 / 5.98% | 34.00% |
| Digit B step 1000 EMA — final | 90.16% | 44.10% | 3.48% | 5.87% | 8.53% | 21.00% | 2.32 / 6.06% | 33.66% |

**Do not equate preferred with promoted or gate-selected.** Both latest stages have `selected=null`: parentheses cap is 7.89%, noisy 1-char cap 18.00%. Final B RAW also fails new-character recall (minimum approximately 90.08%) and new-validation improvement; final B EMA fails new-validation improvement. The handover's general “only two failures” sentence does not describe every final checkpoint. No thresholds were relaxed.

Digit recovery succeeded, but noisy single characters remain weak and parentheses worsened relative to A.2. Stage B's 33.17% test CER versus its actual 33.00% starting checkpoint is a small difference, not proof of overfitting without paired uncertainty analysis. It partially recovers parentheses, but is not a demonstrated overall upgrade over preferred A epoch 3. New test and old test5 are **report-only reused development data**, provisional VLM labels remain unverified, and no matched continuation control was run. The separately collected 200-restaurant quality-filtered batch has not been trained or evaluated in this record.

## Later evidence and exact discovery pointers

All following roots are under `outputs/recognition/experiments/`; use `HANDOVER.md`, audits, checkpoint inventories and evaluation JSON together:

| Experiment root | Key evidence |
|---|---|
| `l2_v06_dishlex50_yolo20_full5_120k` | `manifests/corpus_audit.json`, `evaluation/eval_report.json` |
| `l2_dishlex_tabelog30_yolo20_synth50_1000` | `manifests/new_tabelog_split_audit.json`, `evaluation/evaluation_summary.json` |
| `l2_vocab1901_minoroverlap_two_stage` | Stage A/B downloads and training audits; run handover |
| `l2_vocab1901_classifier4_full2_queued` | Stage A.1/A.2 downloads; run handover and evaluation |
| `l2_1901_digits4_real40_1000` | `stage_a/downloads/stage_a_train_audit.json`, `evaluation/evaluation_summary.json`, stage download checksums |

For loading the preferred candidate, use the [dedicated checkpoint pointer](preferred_checkpoint.md), which records the full verified file/vocabulary hashes and exact RAW state. Current promoted and inspector states are separate. This update changes documentation only; it does not activate a model or decoder.

## Subsequent automated Tabelog200 experiment

`l2_1901_tabelog200_auto1000` continued from the preferred RAW checkpoint, with no vocabulary expansion: 1000×64 presentations (40% new-real /20% old YOLO /40% synthetic), all weights learnable and 147 BN buffers frozen in RAW/EMA. VLM screening of the 200 retrieved restaurants yielded 132 MENU, 63 NON_MENU and 5 SEVERE_DISTORTION; these are provisional model decisions. Of 835 transcribed crop slots, 623 were OK. After quality/OOV/geometry eligibility, training used 288 crops from 76 restaurants; locked 160/20/20 source splits yielded 79 all-OK validation and 45 all-OK test crops. An inactive candidate records 53 training-only OOV characters; the trained head remains 1902 outputs.

Selected step 250 EMA passed source-relative guards but barely moved: new all-OK validation **67.49% → 67.28%**, in-vocabulary validation **54.58% unchanged**, all-OK test **53.41% unchanged**. Final step 1000 RAW improved digits **5.73% → 5.07%** and parentheses **9.58% → 5.99%**, but noisy 1-char regressed **20.50% → 23.00%**; final all-OK test was 52.59%. These changes do not establish a significant new-menu gain. The preferred **L2_menu1901_digit_recovery** and promoted registry remain unchanged. Run evidence and review page are under `outputs/recognition/experiments/l2_1901_tabelog200_auto1000/` (`HANDOVER.md`, `evaluation/review/index.html`).
