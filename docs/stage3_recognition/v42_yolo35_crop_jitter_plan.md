# V42:35%YOLO crop-jitter and increased true512line exposure

User requested a pilot oversampling YOLO to35%,random2–5%boundarypadding,
slight aspect stretch and morelong512samples. Budget remains2000updates with
effectivebatch32/microbatch4. Source is V41's gate-passing EMA1000best.pt,
not its finalraw and not the failed V39BNswap. Vocabulary stays1779,oldIDs and
blank0unchanged. Promoted V38 and sourceV41 stay untouched.

| Stream | Presentations | Fraction |
|---|---:|---:|
| Augmented native YOLO crops |22400|35%|
| Genuine11–14character512lines |19200|30%|
| Controlled one-new-character lines |9600|15%|
| Original synthetic replay |6400|10%|
| Application-clean replay |3200|5%|
| Application-noisy replay |3200|5%|

All1561eligible YOLOtrain crops included;156holdout crops from13menus remain
clean and disjoint. All521required synthetic terms protected. All57added
characters planned>=168total encounters; targeted scheduling168–169/class.
Thirty percent is real long content with natural validwidth385–512 and exactly
4800presentations each for11/12/13/14characters. These reuse V37's audited4800
training images,75%clean/25%mildlynoisy exposure; its800heldout images are
never trained. Including augmented wideYOLOviews, physical512share is34%.

YOLO augmentation begins from native upright RGBdisplay crops,rotating vertical
linesCCW into the established recognizer strip convention. Each side receives
an independent Uniform(.02,.05)margin,proportional to that side's original
width/height and rounded to native pixels(minimum1). Add margins with edge
replication; never trim sourcepixels or draw in neighboring menu text. Apply
Uniform(.95,1.05)horizontal stretch,then proportional H64normalization and
nearest128/256/384/512edgepadding. Extreme lines fit512withverticalletterbox;
no glyph clipping. ValidT includes the actual added margins but excludes
artificial bucketpadding. No extra photometric noise/blur onYOLO.

Twenty-two thousand four hundred deterministic views are generated on demand,
not saved as a newimage dataset. yolo_views.jsonl records exact seeds,ratios,
pixels,stretch and dimensions. Width-quota rounding duplicates16views to retain
700complete same-width batches;22384unique transforms. Allbasecrops retain at
least one view. Native crop size minimum is40×19 in reading-axis coordinates;
ratios are sampled continuously but pixel rounding slightly changes realized
percentages on tinycrops. augmentation_preview.png contains16examples.

This pilot fine-tunes the complete recognizer to enable old-menu/long-text
adaptation: encoder5e-7,existinghead3e-6,addedclassifierrows3e-5. AdamW excludes
norm/bias decay;200warmup,cosine2000to.1,EMA.998,gradientclip5. All147BNrunning
buffers frozen; BNaffine trainable. Initialization copies every source tensor
and must reproduce the source's savedFP16/BF16predictions before updates.

Frozen startingV41 supplies full-vocabulary teacherKL(T2,weight.5),validframes
only,on original/application/long synthetic replay. YOLO and new-character
targets are exempt so their supervised adaptation is not tied to wrong teacher
answers. This differs from V41's new-row-only run. Multiple recipe elements
change together,so this is a package test,not a single-factor ablation.

Evaluate every500updates,raw/EMA,bothprecisions: unchanged156YOLOholdout,
cleanYOLOtrain,matched256clean/augmentedtrainviews,long512/384pairedknown/
heldoutfonts,684new-characterprobes and existing17retention gates. Record
orientation exposure because the long stream is horizontal. Selection score
is .6YOLOholdoutCER+.2heldoutfontlong512meanCER+.2(1-newcharheldfontrecall),
subject to unchanged revised gates in both precisions. No automatic promotion.

Entry scripts/recognition/train_v42_yolo35.py,preparation prepare_v42.py,
launcher run_v42.ps1. Output outputs/recognition/pilots/v42_yolo35_jitter_long512_2000/.
One-shot completion/failure notification; do not poll results.

## Completed result and independent audit

Completed2000updates/64000presentations in approximately39minutes. Independent
audit verifies all planned/actualstream counts,22400YOLOviews/all1561bases,
19200genuinelylong512lines(4800eachlength11–14),21760physical512presentations
(34%),all57newchars>=168encounters andall521requiredterms. All transform
parameters reproduce from savedseeds. Holdoutpixelhashes/geometry unchanged;
all147BNbuffers fixed across raw/EMA/selected/step states. All98BNaffine tensors
updated in finalraw. SourceV41 and productionregistry unchanged.

Finalraw FP16YOLOholdout51.82→45.46%,exact24→43/156;13menu-bootstrap95%CI
forchange[-9.38,-2.83]pp. Nine menus improve,three worsen,one ties. Raw1500
haslowestobservedCER45.15%;finalEMA45.99%. Matched256clean/augtrainingprobe
gap shrinks2.00→.32pp. This does not isolate augmentation's causal contribution
from higherYOLOexposure/longdata/fullfine-tuning/teacherregularization.

Heldfontlong512mean6.64→1.88%;allfont512clean6.52→1.72%,noisy6.92→2.38%.
Same labels fitted384 remain18.12/20.90%CER. Newcharheldfontrecall52.05→65.20%,
precision92.15%;newcharYOLOmatches0→1/36 remain weak. Horizontalholdout
55.19→47.12%,vertical37.08→38.20%(twoextraedits). Trainingvertical11.26%;
longreplayallhorizontal. This is an orientation-balancing concern,not proof of
a single causal failure.

HistoricalCER10.69→12.53%,seen5.42→5.94%,fresh-short-clean25.17→28.83%.
Finalraw fails5FP16/6BF16revisedgates. Pooled2charFP16 58/400vscap56;
BF16onechar63/200vscap60,twochar60/400vscap56. No trainedcandidate clears
allgates;best.pt is unchanged startingV41EMA1000 atV42update0. Promoted V38
remains unchanged. Finalraw/EMA andraw1500remain researchcandidates.

Full review.md,completion_verification.json,review_comparison.json in output.
Before anotherlongrun,consider inexpensive source/candidateweight interpolation
to measure recovery of historical/shortretention while retainingmenu gains.
Do not infer independent production performance from reuseddevelopmentmenus
andVLM-generated labels. No new experiment launched during review.
