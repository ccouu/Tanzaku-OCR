> Historical snapshot preserved by the documentation audit. “Current/latest/running” statements below apply only to the earlier snapshot. Use [current recognition status](../stage3_recognition/current_status.md) now.

# Stage 3 compact handoff

Updated 2026-11-13 after explicit user-authorized **V43 final raw promotion**.
Active checkpoint `checkpoints/menu_recognizer_v43.pt`, 1,779 characters + blank0.
V38 remains available for rollback. V43 fails11retention gates both precisions;
this promotion accepts the menu-domain trade-off without changing any gate.
See [promotion decision](../stage3_recognition/v43_promotion.md).
**Latest V52 completed:** beam5/10/25 and optional dictionary reranking tested on
V43/V48finalraw,5,265records per decoder/model. Normalized YOLOCER V43
41.50%→40.77%(beam25lex0.3),V48 39.73%→39.00%(beam10). Opening hits0/13remain
forV43;V48 2→3/13. V43 beam25 oracle29.82%CER, so search alone cannot rescue
recognition. Decoder tests and baseline/source/digit-lock audits pass.
No production change or training. [Results](../stage3_recognition/v52_beam_lexicon_diagnostic.md).
**V50 completed and audited:** matched normalized control and BiLSTM1e-5,
2,000 updates / 64,000 samples each. Clean bracket CER 59.88% → 47.46%, but
new-Latin recall 33.69% → 23.25% and 1-char CER 33% → 37%. YOLO canonical
CER 41.81% → 40.67% has a menu-bootstrap CI crossing zero; openings remain 0/13.
Control best raw1,500 passes source-relative guards (11 absolute gates fail).
BiLSTM has no eligible best.pt; final raw fails 12 absolute gates, final EMA 11
and both fail short-text guards. 147 BN buffers/all other protected weights exact;
all 521 terms >=2, all 6,000 digit images and 1,561 YOLO train crops verified.
Root: `outputs/recognition/pilots/v50_normalized_matched_2000/` contains independent
`completion_verification.json` and `paired_analysis.json`. V43 stays promoted.
No further training is running or launched. See [V50 results](../stage3_recognition/v50_normalized_bilstm_matched.md).
**LatestcompletedV49:**selectivehead-only2,000updates/64,000presentationsin25.70min.
Only52Latinrows1e-3and14digit/bracketrows1e-5changed;BiLSTM/CNN/temporal/otherrows/
blank/BNaffine/147buffersfixed. MultifontLatinrecall44.68%,mostlyexisting19gain;
new33recall33.75%. Digits86.72%vs87.72%source,YOLOr42.02%vs41.61%,brackets3/26
unchanged. Best=raw1500(43.91%Latin,86.84%digit,41.92%YOLO),passesonlysource-relative
guards,notabsolute11failedgates. V49 and V50 are complete.
Output:`outputs/recognition/pilots/v49_selective_head_2000/`;launcher:
`scripts/recognition/run_v49.ps1`;logs:`outputs/recognition/v49.*.log`.
See [V49plan](../stage3_recognition/v49_selective_head_pilot.md). Launchunsandboxednormal-userforHerdr
notificationdelivery;noserverrestart/ACLchange,nopolling.
**Latest completed tests:**V47cleanLatinfitting1,000updatesandV48isolatedparentheses
2,000updates;independentV43sources. V47trainCER1.05%,samefontprobe42.5%,heldfont
67.31%;bestisraw1000fitting-only,neverdeploy. V48best=EMA1000(41.19%YOLO,
86.71%digitrecall;source-relativeguardspass,absolute9FP16/10BF16gatesfail).
V48finalrawYOLO39.94%butdigitrecall85.42%failsguard. KeepV43;notrainingrunning.
See [V47/V48](../stage3_recognition/v47_v48_isolated_tests.md).
Launcher:`scripts/recognition/run_v47_v48.ps1`;logs:`outputs/recognition/v47_v48.*.log`.
Outputroots:`outputs/recognition/pilots/v47_clean_latin_fit_1000/`and`v48_parentheses_2000/`.
**V46 completed and reviewed:**4,000updates/128,000presentations in44.04minutes,
not time-limited; exact30%Latin/10%parentheses/15%digits/5%basicJapanese/20%YOLO/
20%synthetic. CNN/projection/temporal/BNaffine/147buffers verified fixed;
BiLSTM/classifierchangedafter200new-row-onlyupdates. No selectedbest or promotion.
Latinheldoutrecall23.88%,parenthesissymbolrecall44.68%,YOLOCER41.29%rawvs41.61%
V43(CIcrosseszero),exact48vs50/156. Digits/basicJapanese regress. Cleanup deleted
26,015disposablefiles/12.5094GiB;preservedinputs/checkpointsverified. V47/V48arethe
newlyauthorizedexperimentsabove;V46itselfiscomplete.
Output:`outputs/recognition/training/v46_focused_recurrent_4000/`;
launcher:`scripts/recognition/run_v46.ps1`. Post-success cleanup audit/report:
`outputs/recognition/cleanup/v46_post_training/`. See [V46 plan](../stage3_recognition/v46_focused_recurrent_training.md).
**V45 Stage 1A completed 1,000 updates**. Only33newLatinrows changed; all old
weights, BN affine and147buffers remained fixed. Training CER74.34%, heldout recall
23.92%; no qualifying checkpoint or promotion. See [V45 results](../stage3_recognition/v45_latin_head_only_pilot.md). Outputs:
`outputs/recognition/pilots/v45_latin_head_only_1000/`; logs `outputs/recognition/v45.*.log`.
V45–V50 are complete; no further training has been launched.
V45 Herdr completion delivery failed due to server/client permissions; saved
results completed successfully.

## Checkpoints and vocabularies

| Role | Exact artifact | Interpretation |
|---|---|---|
| Promoted recognizer | `checkpoints/menu_recognizer_v43.pt` | V43 final raw, update 2,000; 1,779 characters + blank0 |
| Rollback recognizer | `checkpoints/menu_recognizer_v38.pt` | V38 final raw, update 686; 1,722 characters + blank0; unchanged |
| Promoted registry | `src/recognition/current_model.json` | Load through `src.recognition.current.load_current_recognizer`; checksum verified |
| Retention-passing expanded reference | `outputs/recognition/pilots/v41_new_rows_only_2000/best.pt` | V41 EMA, update 1,000; 1,779 characters; development candidate, not promoted |
| Preferred research source | `outputs/recognition/pilots/v43_digits_prices_2000/final_raw.pt` | V43 raw, update 2,000; 1,779 characters; better menu/digit recognition, fails retention |
| Latest completed run | `outputs/recognition/training/v46_focused_recurrent_4000/` | V46 from V43 final raw; 1,812 characters; no selectedbest; keepV43 |

Always load the vocabulary embedded in its checkpoint. V40 appended 57 characters
to 1,722; V44 appends 33 Latin letters to 1,779. Old IDs never move; blank remains 0.
V43/V44 normalize only full-width decimal digits to ASCII. ASCII 0–9 already exist
at IDs 3–12. The fuzzy dictionary is a separate database:
`data/processed/lexicon/culinary_menu.db`, accessed by `CulinaryLookup`.

**Do not assume best.pt means latest trained weights.** V38's historical training
best.pt is V36; V42's is V41; V43's is V42 at V43 update 0. V44 selection is gated
and may also retain its source. Final raw/EMA and selected best must be reported
separately. No post-V38 candidate has been automatically promoted.

## V43 completed and audited

Output: `outputs/recognition/pilots/v43_digits_prices_2000/`.
Verified 2,000 updates / 64,000 presentations: 30% digits, 25% YOLO, 20% long512,
10% new characters and 15% ordinary replay. All 6,000 digit renders, 1,561 YOLO train
bases, 521 exact terms (minimum once), 1,920 presentations per digit anchor and
147 fixed BN buffers were verified. FP16/BF16 initial prediction parity passed; source files preserved.

| FP16 metric | V42 source | V43 final raw | Final EMA |
|---|---:|---:|---:|
| YOLO holdout CER |45.46%|41.61%|41.92%|
| Heldout-font digit macro recall |69.28%|87.72%|86.71%|
| YOLO holdout digit macro recall |57.81%|65.52%|65.00%|
| Historical CER |12.53%|13.88%|13.71%|
| Fresh short clean CER |28.83%|31.33%|31.00%|

Holdout exact matches: 43 → 50/156; deletions 249 → 219, insertions 34 → 26,
substitutions 153 → 154. Menu-bootstrap CER change −3.86pp, 95% CI [−5.98, −1.92].
Raw fails 11 retention gates in both precisions; EMA fails 10 FP16 / 11 BF16. best.pt is unchanged V42 fallback, not passing trained V43.
Digits 4/7/9 remain weak on small menu samples; triple repeats and ¥ prices remain
weak on synthetic probes. Only 2.09% of V43 presentations contained parentheses.
The alphabet audit found 33 missing Latin letters. See [V43 report](../stage3_recognition/v43_digit_price_pilot.md).
Local audit:review.md,completion_verification.json,paired_analysis.json,sampling.json.

## V44 completed and audited

**Outcome:** extra-long768 heldout CER66.33%→48.17%raw (EMA50.33%);
overlap51248.00%raw. Latin remains99.58%CER/0.67%macro recall. Parentheses-symbol
recall23.05%→23.74%; full-width opening0/47. YOLO CER41.61%→41.50%,95%CI
[−1.24,+1.33]pp; exact50→44/156, deletions219→243. Historical13.88%→17.53%,
old long512clean1.24%→2.30%, YOLO digit recall65.52%→60.69%. All trained
candidates fail retention; final raw/EMA each fail12gates in both precisions.
best.pt is prediction-preserving expanded V43 at update0, not trained V44.

Audit passed: all64,000samples,1561YOLO bases,521exact terms,33appended row updates,
oldIDs/blank0,147fixedBNbuffers, full-width digit normalization. Optimizer counters
old1800/new2000 plus runtime assertions support the200-update protected warmup;
no separate update200snapshot was saved. Initial parity passed with zero new-token
emissions in both precisions. V43 source and promoted V38 hashes unchanged.

Latin diagnostic also fails on208training lines (99.86%CER). Restricting alphabet
to Latin+space+blank yields97.84%training/98.38%heldout CER, predominantly deletions.
Do not call this only a generalization or dictionary-coverage problem. Next: isolate
Latin fitting/alignment, then parentheses, rather than extend this mixture. No new
training launched. Artifacts:review.md,completion_verification.json,latin_diagnostic.json.

User selected **V43 final raw** and **30% YOLO** (V43 used 25%, so this is an increase).
Launch record:`outputs/recognition/v44_launcher.json`,2026-11-12 22:48 UTC.
Launcher:`scripts/recognition/run_v44.ps1`; no constant polling.

| Stream | Share | Presentations |
|---|---:|---:|
| YOLO crop jitter |30%|19,200|
| Latin targets |15%|9,600|
| Dedicated parentheses |5%|3,200|
| V43 digits |15%|9,600|
| New16/20/24-character lines |15%|9,600|
| Existing11–14-character long512 |5%|3,200|
| Existing57-character replay |5%|3,200|
| Original/application replay |10%|6,400|

Prepared plan covers all 1,561 YOLO train bases, 6,000 digit images, 2,080 Latin renders,
640 parentheses renders, 540 extra-long renders and 521 exact terms. All 52 ASCII
Latin letters have at least 207 planned token encounters. Parentheses occur in
4,647/64,000 samples (7.26%), counting other streams. All old IDs and blank 0 are
preserved; new rows copy the blank weights and its bias minus 8. Initial source-prediction parity and zero new emissions
were required before updates and passed, as verified in initial_parity.json.

First 200 updates train only 33 appended classifier rows. Then full fine-tuning:
encoder 2.5e-7 / head 1e-6 / new rows 1e-3; 147 BN buffers frozen, affine trainable after warmup.
Cosine horizon 2,000, warmup 200, minimum factor 0.1, EMA 0.998, microbatch 4 / batch 32.
Teacher KL weight 1 protects ordinary/digit/existing-long replay; not YOLO, Latin,
parentheses or the new extra-long labels. Raw/EMA evaluated every 500 updates in
FP16/BF16. Existing retention policy unchanged; no automatic promotion.

Geometry preflight selected **wide768**. Heldout-font synthetic CER:
fit512 87.75%, wide768 66.17%, overlap512 65.42%. The overlap advantage of 0.75pp
is below the predeclared >1pp threshold. There are 120 synthetic lines, 60 known /
60 heldout; no eligible
extra-long YOLO holdout example, so no paired real-menu geometry evidence.
A 374×34 crop scales proportionally to 704×64; fitting to 512 reduces height to about 47.
Overlap is 100 H64-normalized pixels, not 100 native pixels; decoder uses CTC run
centers/source-coordinate ownership, preserving legitimate repeats. Experimental
768training does not change promoted V38 inference buckets.

Finite GPU forward/backward preflight passed for Latin128 and long768, 1,813 outputs.
Four focused geometry tests passed. Inspect analysis.md,report.json,sampling.json,
initial_parity.json,geometry_preflight.json,metrics.jsonl and paired_analysis.json
for the completed evidence. Verified actual consumed counts, all 33 row updates, fixed BN buffers,
selected-versus-final checkpoints,Latin/parentheses/digit/long results and retention.
See [V44 plan/preflight](../stage3_recognition/v44_latin_parentheses_long_pilot.md).

## Data, model and evaluation constraints

- V27 architecture:MobileNetV4 ConvSmall, H64 RGB, horizontal stride 4, height mean,
  temporal 256 / two DW blocks, residual packed BiLSTM 128 per direction, CTC.
  Conv2d height collapse was not adopted after V32. CNN/temporal BN remains padding-sensitive.
- Freeze BN running statistics only; affine can train. V41 was the special case
  where all old weights/affine were frozen. Packed BiLSTM excludes artificial
  right padding. Valid CTC lengths are ceil(validwidth/4); loss isFP32.
- Promoted buckets128/256/384/512 use edge replication; V44 adds experimental768.
  Mobile export/parity/static-shape compilation/latency remain unmeasured.
- Crops:`data/processed/gemini_banana_mobile_menus_50_crops/` actually covers
  menus 001–130. Expanded labels:1,561 train / 156 holdout; 13 menu-disjoint validation
  pages010,020,030,040,050,060,070,080,086,100,109,120,130.
- Labels are VLM-generated, not independently human-verified. NOISE/TRUNCATED/
  UNCERTAIN images are excluded. Review decisions are separate from training labels.
- V38's original 135-crop holdout CER41.19% is not directly comparable with156-crop
  expanded-vocabulary results. Historical 38-crop yolo_provisional is another set.
- Report raw greedy CTC (beta 0) separately from lexicon/translation accuracy.
  Probes are repeatedly used development data, not independent handwriting tests.
- `menu_adaptation_v2` relaxes only pooled two-character cap 13.5% → 14.0% (54 → 56 edits/400);
  other 16 gates and both precisions remain required. V38 acceptance was retrospective.

## Review UI and source workflow

- Latest generated recognition queue uses **V42 final raw, update 2,000**, not V43/V44:
  `outputs/recognition/review/v42_training_failures/`,1,015 failures / 1,561 train crops.
  Existing local launch URL:`http://127.0.0.1:7867/`; availability not rechecked.
  V38 review/decisions remain separate at 7866. No holdout crop is in these queues.
- Core code:`src/recognition/`; recent experiments:`scripts/recognition/`.
  Normalization:`label_normalization.py`; long routes:`long_line_inference.py`;
  crop review:`review/`. Windows environment:`.venv-recognition/Scripts/python.exe`.
- Recent full test suite: 63 recognition/detection tests passed before V44; four
  new long-line geometry tests subsequently passed. Do not imply a new full-suite run.
- GitHub:`https://github.com/ccouu/Tanzaku-OCR.git`,main; last pushed62ce8cf,
  covering code/docs through V43. V44 and this documentation update are local
  until explicitly committed. Datasets,outputs,weights,databases,environments,
  credentials and archive stay ignored; never expose keys or .env contents.
- Preserve existing checkpoints/source images. Run background helpers hidden.
  Use one-shot Herdr completion/failure notifications, not periodic agent polling.

## History and remaining work

[V35 averaging](../stage3_recognition/v35_checkpoint_averages.md) produced gate-passing average3;
[V36](../stage3_recognition/v36_result.md) selected raw7000; [V37](../stage3_recognition/v37_result.md) improved long512 but
failed retention. [V38](../stage3_recognition/v38_yolo25_training.md) was promoted for YOLO-crop gains.
V39's full BN-stat swap failed. [V40](../stage3_recognition/v40_new_characters_plan.md) added 57 characters but
constant new-row initialization changed predictions. [V41](../stage3_recognition/v41_new_rows_only_plan.md)
fixed initialization and isolated new-row learning; [V42](../stage3_recognition/v42_yolo35_crop_jitter_plan.md)
improved YOLO/long recognition with retention costs. [V43](../stage3_recognition/v43_digit_price_pilot.md)
proved targeted digit gains; V44 improved extra-long synthetic recognition but did not solve Latin or parentheses.

Independent real-menu/handwriting evaluation, native 20+ character crop evidence,
vertical-crop robustness, reviewed label quality, dictionary sense selection,
mobile inference and translation evaluation remain outstanding. V1–V28 historical
source/reports are in `archive/stage3_legacy_20261111/`; archived pixels/ZIPs were
removed to reclaim 7.08GiB, while active V27 data and shared licensed fonts remain.
For the broader chronology, see [stage3_summary.md](../stage3_recognition/stage3_summary.md).
