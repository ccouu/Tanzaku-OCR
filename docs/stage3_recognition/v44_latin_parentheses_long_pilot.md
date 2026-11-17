> Historical experiment record. Status, sources, vocabulary, geometry and gates below belong to this version; they are not current deployment instructions. See [current implementation/status](current_status.md) for V75/V76, V77 authorization and production V43. Referenced cached assets may have been retired; do not regenerate or overwrite them merely to follow this record.

# V44: Latin coverage, parentheses and extra-long recognition

**Completed and independently audited. Not promoted.** No further training is
running. The execution met its coverage/invariance requirements, but Latin and
parentheses recognition remain weak and synthetic retention regressed.

User authorized 2,000 updates from **V43 final raw**, with **30% YOLO crops**.
This is an increase from V43's25%; the requested30% takes precedence.

Append all missing ASCII A–Z/a–z letters:33 rows,1779→1812 characters plus blank0.
Preserve every existing ID, initialize new logits by copying the blank row minus8,
and require zero source-prediction changes in FP16/BF16 before training.
Freeze original parameters for the first200 updates, training appended rows only.
Then train the full model with encoder2.5e-7/head1e-6/newrows1e-3, frozen147BN
buffers and trainable affine. Use200-step warmup/cosine2000to0.1/EMA0.998.
Frozen expanded-source teacher KL weight1 applies to ordinary/digit/existing11–14
character replay, excluding the new extra-long targets where the source is weak.

64,000 presentations in2,000 homogeneous logical batches32, microbatch4:

| Stream | Share | Presentations |
|---|---:|---:|
| YOLO crop jitter |30%|19,200|
| Latin targets, all52 letters |15%|9,600|
| Dedicated realistic parentheses |5%|3,200|
| V43 digit replay |15%|9,600|
| Extra-long16/20/24-character lines |15%|9,600|
| Existing11–14-character long512 |5%|3,200|
| Existing57-character replay |5%|3,200|
| Ordinary original/application replay |10%|6,400|

Visit all1561YOLO training crops, all6000digit renders and all521curated exact terms.
Retain156YOLO holdout crops/13disjoint menus unchanged. Normalize full-width decimal
digits to ASCII in all training streams; preserve original labels and all source files.
Parentheses cover both() and（） in realistic menu contexts, at least5% overall.
Latin training includes single/repeated letters, English menu words and mixed labels,
with fifteen training fonts and four heldout fonts. Confirm glyph coverage before
rendering; no tofu substitution. Probes use new seeds and no exact train/probe pixels.

Before updates, compare V43 on fit512 versus width768 versus balanced512windows
overlapping by100 **H64-normalized** pixels. This overlap is not100native pixels.
Use CTC argmax-run centers/source-coordinate ownership to stitch overlapping windows;
never remove legitimate repeated letters by string deduplication. Choose overlap only
if heldout-font synthetic CER beats768 by more than1pp, otherwise prefer one-pass768.
Report native extra-long YOLO holdout separately, without using it to choose geometry.
Synthetic long lines have widths513–768, including explicitly recorded background
adjustments; they are not a universal test of arbitrary real-menu aspect ratios.
If overlap wins, train exact synthetic glyph-box-based segments without cutting strokes.

Evaluate raw/EMA every500updates in both precisions: original retention, cleanYOLO,
digits, Latin recall/D-I-S, parentheses recall, long768 and overlap512 separately.
Selection score .4YOLOholdoutCER+.2LatinheldoutCER+.1parenthesesheldoutCER+
.2chosen-geometry-longheldoutCER+.1digitheldoutCER, with existing retention gates.
Source V43 already fails retention; unchanged update0best.pt is an ineligible fallback.
Keep final raw/EMA separately and do not promote automatically.

Output:`outputs/recognition/pilots/v44_latin_parentheses_long_2000/`.
Launcher:`scripts/recognition/run_v44.ps1`, one-shot completion/failure notification.
No periodic result polling.

## Preparation and launch record (historical)

Launched on 2026-11-12 at 22:48 UTC using `run_v44.ps1`; launcher record:
`outputs/recognition/v44_launcher.json`. This records a launch, not completed
training. The completion notification was subsequently reviewed as recorded below;
no periodic training-progress polling was used.

Prepared plan audit confirms all 64,000 presentations, all 1,561 YOLO bases,
all 6,000 digit renders, all 2,080 Latin training renders, all 640 parentheses
renders, and all 540 extra-long training renders. All 521 exact curated terms
occur at least once. Every ASCII Latin letter has at least 207 token encounters
in the plan. Parentheses appear in **4,647/64,000 = 7.26%** of presentations,
including the dedicated 3,200-image stream. Training/probe hashes are disjoint.

| Forward-only V43 geometry, FP16 CER | Known fonts | Heldout fonts |
|---|---:|---:|
| Fit line into 512px |87.75%|87.75%|
| Single-pass 768px |65.75%|66.17%|
| Overlapping 512px windows |65.58%|65.42%|

Selected **wide768**: overlap improves heldout-font CER by only 0.75pp, below
the predeclared >1pp threshold for accepting stitching complexity. This is a
120-line synthetic probe, with 60 known-font and 60 heldout-font examples.
No existing YOLO holdout crop met the extra-long geometry filter, so there is
**no paired real-menu geometry result**. The high residual CER is an explicit
training target, not evidence that width alone solves 20+ character recognition.

For the motivating 374×34 crop, proportional H64 width is 704: the 768 bucket
keeps height64, whereas width512 reduces strip height to approximately47px.
768 is an experimental route; promoted V38 still uses its recorded production
bucket configuration. Mobile export, static-shape compilation and latency at768
remain unmeasured.

GPU preflight passed finite forward/backward loss for Latin128 and long768
microbatches, with 1,813 CTC outputs (1,812 characters + blank0). Four focused
geometry/stitching tests passed, including legitimate repeats and overlap ownership.
The launcher requires full source prediction parity in FP16/BF16 and zero new-token
emissions before updates; both requirements passed in the completed run.

## Completed results

Verified2,000updates/64,000presentations, exact planned streams, all1,561YOLO bases,
6,000digit renders and521exact terms. Parentheses4,647/64,000(7.26%); minimum Latin
token exposure207. All33appended rows changed, all old IDs/blank0 preserved and147BN
buffers unchanged in initial/raw/EMA. Optimizer counters show2,000new-row steps and
1,800pretrained steps, consistent with runtime assertions protecting old weights
during the first200updates. No separate update200 checkpoint exists. SourceV43 and
promotedV38 hashes remain unchanged. Vocabulary coverage is not recognition accuracy.

| FP16 metric | V43 expanded initialization | V44 final raw | Final EMA |
|---|---:|---:|---:|
| YOLO holdout CER |41.61%|41.50%|41.61%|
| Latin heldout-font CER |101.62%|99.58%|99.86%|
| Latin heldout-font macro recall |0.28%|0.67%|0.60%|
| Parentheses-line heldout CER |58.90%|51.14%|51.83%|
| Parentheses-symbol heldout macro recall |23.05%|23.74%|23.97%|
| Extra-long768 heldout CER |66.33%|48.17%|50.33%|
| Same lines, overlap512 CER |65.42%|48.00%|50.17%|
| YOLO digit macro recall |65.52%|60.69%|60.69%|
| Seen-font CER |7.00%|8.85%|8.64%|
| Historical CER |13.88%|17.53%|17.20%|
| Existing long512 clean CER |1.24%|2.30%|2.12%|
| Existing long512 noisy CER |1.94%|3.60%|3.50%|

The raw menu CER change −0.10pp has13-menu bootstrap95%CI[−1.24,+1.33]; no clear
gain. Exact lines fall50→44/156. Deletions219→243, insertions26→24, substitutions
154→131. EMA1000has the lowest observed YOLO CER40.77%, but fails retention.
Final raw/EMA each fail12gates in both precisions; all trained candidates fail.
`best.pt` retains expanded V43 initialization at V44update0: prediction-preserving
in both precision suites, but not a byte-identical V43 checkpoint or passing V44.

Latin diagnostic:208sampled training lines have99.86%CER and0exact. Restricting the
diagnostic alphabet to Latin+space+blank still gives97.84%training /98.38%heldout
CER, almost entirely deletions. The model failed to fit Latin training examples;
this is not explained by unseen fonts or Japanese output competition alone.
See `latin_diagnostic.json`. Two small rendered inputs were inspected and readable;
this is not a comprehensive visual audit.

Parentheses-line improvement does not establish symbol learning. Full-width opening
parenthesis remains0/47correct on heldout fonts. Even after diagnostic normalization
of（）to(), opening recall falls28.75%→23.75%; closing53.75%→57.50%. On YOLO holdout,
literal parentheses matches stay3/26 and Latin stays0/3(A/B/C). None of the33new
letters occurs in that holdout's references, so new Latin real-menu transfer is
unmeasured. The extra-long comparison remains synthetic only.

Prefer V43 as the next Japanese/digit research source; keep V38 promoted. Do not
extend this mixture. First isolate Latin fitting/alignment with a concentrated
curriculum and measure training CER, then test parentheses in a separate matched
arm with explicit opening/closing metrics. No new training was launched.
Independent audit:`completion_verification.json`; analysis:`review.md` and
`latin_diagnostic.json` in the run directory. Original run artifacts remain intact.
