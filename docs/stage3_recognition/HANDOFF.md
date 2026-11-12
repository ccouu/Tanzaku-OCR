# Stage3 compact handoff

LATEST V42 COMPLETED/AUDITED:2000updates/64000presentations,35%YOLO22400views
(all1561trainbases),30%true11–14char51219200,34%physical51221760,15%newchar/
20%otherreplay. Allseeded2–5%margin/.95–1.05stretch params verified;22384unique
views+16quota duplicates. SourceV41EMA1000/frozen147BNbuffers/clean156holdout
preserved;98BNaffine tensors updated,teacher runtimechecks passed.
FinalrawYOLO51.82→45.46%CER,exact24→43/156,paired13menu95%CI[-9.38,-2.83]pp.
Raw1500lowestYOLO45.15%;finalEMA45.99%. Heldfontlong512mean6.64→1.88%raw;
matchedclean/augtrainprobe gap2.00→.32pp. NewcharheldfontR52.05→65.20%,but
YOLOnewcharTP0→1/36. Horizontal55.19→47.12%;vertical37.08→38.20%(2edits).
Trainingverticalonly11.26%. Historical10.69→12.53%;finalrawfails5FP16/6BF16
gates. best.pt=unchangedV41source atV42update0;V38stillpromoted,noautochange.
Independentreview/audit/review_comparison.json in V42output. Nextcheapcheck:
source/candidate weight interpolation for retention/menutradeoff;not launched.
No training running. See [V42 plan/results](v42_yolo35_crop_jitter_plan.md).

Historical V42 launch details (completed):

AUTHORIZED V42 PILOT:2000updates fromV41selectedEMA1000.35%nativeYOLOviews
(22400),random2–5%independentsideedgepadding +horizontalstretch.95–1.05;
30%true11–14charlong512(19200),15%controllednewchar,20%otheroldreplay.
Actualphysical512fraction34%,all1561YOLOtraincropsincluded,156heldoutclean.
All57newcharplanminimum168,all521termscovered. Fullfine-tuneencoder5e-7/
head3e-6/newrows3e-5;147BNbuffersfixed,affinetrainable. FrozenV41teacherKL
onoldsynthetic/longvalidframes only,T2/weight.5. NoYOLOteacher constraint.
Nativeviews on-demand:22384unique+16quota duplicates,no22400imagecopies.
Output `outputs/recognition/pilots/v42_yolo35_jitter_long512_2000/`.
V38stillpromoted;sourceV41preserved,noautopromotion. One-shotHerdrmessage;
do notpoll. See [V42 plan](v42_yolo35_crop_jitter_plan.md).

LATEST V41 COMPLETED:2000updates/64000presentations; strict0newemission/0
predictiondifferences atinitbothprecisions. Allpretrainedweights/oldclassifier/
BNaffine/147buffers bit-identical throughrun; optimizeronly2newrowtensors.
All57rowslearned;actualnewcharencounters280–1855,targeted280–281/class.
Best.pt=trainedEMA1000,all17revisedgates pass FP16/BF16,newcharheldfontR52.05%.
Finalraw2000newcharR85.67%,precision90.99%,F1 88.25%(V40R53.51%,F1 65.83%).
All57heldfontclasses have>=1success. Finalraw/EMA passFP16butfailBF16onechar
62/200vs60cap:火→久火,株→抹. New-output competition,not oldweightdrift.
YOLO156holdoutCERunchanged51.82%;newcharTPfinal1/36(selected0/36). No menu
transfer gain. KeepV38promoted;V41EMA1000expandeddevelopmentcandidateonly.
Audit/review `outputs/recognition/pilots/v41_new_rows_only_2000/review.md`,
completion_verification.json,review_comparison.json,short_text_differences.json.
No furthertrainingrunning. Nextfocusverifiedmenu-style additions;cheapnewrow
calibration/interpolationcouldtestsyntheticrecall/gatetradeoff,notlaunched.
See [V41 plan/results](v41_new_rows_only_plan.md).

Historical V41 launch details (completed):

AUTHORIZED V41 PILOT RUNNING:2000updates from promotedV38,train57newclassifier
rows ONLY for whole run. Oldencoder/temporal/BiLSTM/classifier/BNaffine and147
buffers muststay bit-identical. Initnewrows=blankweights,blankbias-8; strict
FP16/BF16zero-prediction-difference andzero-new-emission check before updates.
25%controlledtarget(2668existingrenders,exactly1newchar/line,<=5chars,128/256),
25%YOLO,50%oldreplay. Target scheduling280–281pernewchar; all1561YOLOtraincrops
and521requiredterms included. Same684fontprobes/156menuholdout asV40.
Fullsoftmax frozenoldteacher consistency on validframes/old-onlylabels,weight1.
NewrowAdamW1e-3,warmup200/cosine2000to.1,EMA.998,micro4/effective32.
Output `outputs/recognition/pilots/v41_new_rows_only_2000/`; useone-shotHerdr
completion/failure,no constantpolling. No autopromotion; keepV38 production.
See [V41 plan](v41_new_rows_only_plan.md). V40/V38source preserved.

LATEST V40 COMPLETED/NOT PROMOTED:2000updates/64000presentations; all57IDs
appended1723–1779,oldIDs/blank0unchanged. All57newrows updated,actualencounters
532–15388,147BNbuffersfixed; warmup200pretrained-weight-preservation assertion
passed and optimizer counters confirm2000new/1800oldsteps. Independent audit
`outputs/recognition/training/v40_new_characters_2000/completion_verification.json`.
Newcharheldoutfontmacrorecall53.51%raw/46.78%EMA; full156YOLOholdout original
V38 51.82%→47.97%raw/48.91%EMA(FP16),pairedCIcrosseszero. Old135subset
41.19→39.59%,new21subset78.68→69.12%. NovelYOLOcharrecallonly2/36; limited
12-character,punctuation-heavy reference sample. Synthetic retention markedly
worse(rawseen5.33→9.21%,historical10.67→18.05%). Rawfails11FP16/12BF16gates.
best.pt=untrained expandedV38(update0),nottrainedV40. Initialconstant-12newrow
bias wasNOTprediction-preserving:untrainedspace wins onweakframes; observed
parity changes are not merelyrounding. Keep promotedV38; no training running.
Recommended nextpilot:blank-row-minus-offset init,freezeoldfeatures/rows,more
replay/teacher consistency,one-newchar controlledcontexts. Not launched.
See V40 review.md/review_paired_analysis.json and[plan/results](v40_new_characters_plan.md).

Historical V40 launch details (completed):

AUTHORIZED RUNNING: V40 from promoted V38,2000updates/64000presentations.
Append57missing VLM characters to1722→1779(+blank0); preserveoldIDs. Vocabulary
`data/processed/recognition/vocab_v40.json` and run/vocab.json; old dictionaries
unchanged. 50%targetednewchar/25%YOLO/25%synthetic. Plan minimum532encounters
pernewchar; actualcounts and57nonzero learned rows audited at completion.
All1717OKYOLOlabels now in-vocab:1561train/156holdout,same13heldoutmenus.
First200updates newrows only(LR1e-3);then encoder5e-7/head3e-6,newrow1e-3,
frozen147BNbuffers,EMA.998. No V39stats. Raw/EMA evaluated500/1000/1500/2000;
bothprecisions,revisedgates; source/promotedV38 unchanged, no autopromotion.
Output `outputs/recognition/training/v40_new_characters_2000/`.
One-shot Herdr completion/failure; do not constantly poll. Entry/launcher under
scripts/recognition/. See [V40 plan](v40_new_characters_plan.md).

LATEST PILOT: V39 forward-only full BN swap on promoted V38 completed. All1336
YOLOtrain crops once/85FP32forwardbatches;49BNlayers/147buffers recalibrated;
allweights/affineunchanged,zerooptimizerupdates. YOLOholdoutCER41.19→84.28%FP16
and40.90→84.43%BF16;all17revisedgatesfail. Reject candidate. Promoted V38 and
registry unchanged. Output `outputs/recognition/pilots/v39_v38_yolo_bn_swap/`.
This was full pooled-stat replacement with upstream BN in batch mode,not
partial-layer or weak/mixed-stat adaptation. No further training running.

CURRENT: User explicitly promoted V38 final raw/update686 for the intended YOLO
crop use case. Active checkpoint `checkpoints/menu_recognizer_v38.pt`; authoritative
registry `src/recognition/current_model.json`; verified eval-mode loader in
`src/recognition/current.py`. Same V27 architecture, packed BiLSTM, beta0 decode.
147V30BNbuffersfixed. YOLO holdout51.53→41.19%,12/13menus improve. No training
running. V36/source checkpoints and historical V38best.pt remain unchanged.

User-authorized policy `menu_adaptation_v2`: only pooled_original_2char cap
increased by0.5pp,13.5→14.0%(54→56/400). Other16gates unchanged; both FP16/BF16
still required. Final raw passes this revised policy; it failed original BF16
twochar by2edits. Policy is retrospective, not an original-gate pass claim.
Promotion/evidence at `outputs/recognition/training/v38_yolo25_fullpasses4/promotion.json`;
all saved candidates assessed in `retention_policy_v2_candidates.json`.
See [V38 training and promotion](v38_yolo25_training.md). Do not load historical
V38best.pt expecting promoted weights: that file intentionally retains V36.

The chronology below is historical and superseded by the current state above.

LATEST Stage2v4COMPLETED/REJECTED:21epochs,besttrainedepoch1. Allreviewedlabels/
hash/splitauditspassed(45latesttrain,5new+5previousval,687train/126valboxes).
CombinedmAP50-9589.44→87.86%;new89.33→88.55%;previous89.75→87.30%.
V3sourcecheckpointunchanged. Keepv3bestdetector,notv4trainedbest. No training
running. Fullreview/auditsatoutputs/detection/training/banana51_100_finetune_v4/.
RecognitionpreferredstillV36raw7000;V37unselected. SeeStage2v4docfornextideas.

Historical Stage2v4launch: latest051-100labelscompleted,50pages760boxes. Userauthorized
fine-tuninglatestlabelspluspreviousvalidation. Startedlocalfromv3best:45new
trainmenus(687boxes),5newval060/070/080/086/100(73boxes),plus5previousval
010/020/030/040/050(53boxes). Combined10valselectsbest;baseline/finalmetrics
reportedcombined/latest/previousseparately. No previous trainingimagesmixedin.
100epochmax/patience20,samev3hyperparameters. Snapshotbanana51_100_v4;output
outputs/detection/training/banana51_100_finetune_v4/. OneHerdrcompletionmessage,
no polling. See docs/stage2_detection/banana51_100_finetune.md. Priorweightsand
recognizerV36/V37unchanged. NewStage3cropsmustrespectnewmenuholdoutsabove.

LATEST BOTH JOBS COMPLETED AND AUDITED: V37(2000updates,exact30%512/11-14chars)
hasnoeligibletrainedcheckpoint;best.pt=unchangedV36raw7000. LongheldfontCER
6.70→1.30%rawbutBF16onechar61/200>60cap;menu/strongretentiondrifts. KeepV36
default. UnchangedV36alreadybenefitsgreatlyfrom512routingonlongsyntheticlines.
All32wideYOLOprobesaretrainingmenus;nowideheldoutcrops. Verticalexposurewas
diluted24.22→16.91%. See[v37_result.md](v37_result.md),outputreview/auditJSONs.

YOLOlatest-onlyfine-tunecompleted56epochs,bestepoch36. Audited45train/5val,
50reviewedpages/668boxes/zeroolderimages. ValmAP50-9586.29→89.75%,recall96.05
→99.41%. Newcandidateoutputs/detection/training/banana50_finetune_v3/
yolo11n_menu_obb_banana50_best.pt;rootcheckpointunchanged,noautopromotion.
Fivevalmenususedforselection,noindependentrealtestorend-to-endCERmeasured.
No trainingrunning. Annotationservermayremainon7860. SeeStage2finetunedoc
andoutputs/detection/training/banana50_finetune_v3/review.md.

Historical Stage2launch: latest50human-reviewedYOLOpages frozen at
data/datasets/detection/banana50_latest_v3(45train/5sameholdout,668boxes,zeroold
images). Fine-tuningqueuedafterV37launcherexitsusingWait-Process(no polling),
100epochmax/patience20/640px/batch8/AdamW1e-4,existingYOLOweights. No concurrent
GPUtraining:uponV37completion,review savedmetrics onCPU while detectortrains.
Output/status:outputs/detection/training/banana50_finetune_v3/. Originaldetector
checkpoint remains unchanged. Initialqueuetimestampguardfixedandrequeued.
See docs/stage2_detection/banana50_latest_finetune.md.

Historical V37launch: userauthorized2000-updatepilotfromV36best(raw7000),30%512px
11-14charimages.600long512batches+1400existingV36batches,allbatch32. New labels
from curatedmenugrammar,75%clean/25%mildnoise;all15trainingfonts,fourheldout.
147BNbuffersfrozen,EMA.998,planned2000stepcosine. Paired512vs384longprobes and
YOLO512probes added. First preparation failed beforeGPUupdates on unfitfont
layout/generalJMdictphrase;fixed via curatedmenu-onlypool andbackground-only
guardextension,andrelaunched. Do not poll;oneHerdrcompletion/failuremessage.
See[v37_long512_plan.md](v37_long512_plan.md). V36best remains baseline/fallback;
no promotion andno sourceweights modified.

LATEST: V36 completed two full passes/8002updates/256000presentations. Independent
audit passed,including405YOLOtrain crops once/pass,38holdout/five disjointmenus,
521exact terms/pass,147fixedBNbuffers for raw/EMA and parity/hash checks.
Preferred development checkpoint now V36 best.pt =raw7000,all17gates in FP16
andBF16. Finalraw/EMA8002both fail BF16twochar by1edit;FP16both pass.
Application/menu improve;newYOLOheldout79.29→74.75%but exact remains3/38,
gain concentrated on2of5menus;legacy87.33%still0exact. No training running,
no production promotion. Stage2reviewserver remains independent on port7860.
See [V36 result](v36_result.md) and full output review.md/completion_verification.json.

Historical launch details: V36 run launched after AGY handoff: two full epochs from average3,
4001updates/pass,8002total. Selected128000records:63799original,31898appclean,
31898appnoisy,405YOLO;all521exact terms covered,eachtrainingYOLO once/pass.
38newYOLOcrops/five disjointmenus held out. Native RGB aspect-preserving crops
replace stretchedCTCexports;32wide crops letterboxed vertically after fit384.
First launch failed pre-GPU due unstable menu-key serialization;fixed and
relaunched. Check outputs/recognition/training/v36_yolo_fullpasses2/status.json
and outputs/recognition/v36.stderr.log for status. Hidden local launcher sends
one completion/failureHerdr message;do not poll. No further approval needed to
fix/resume this authorized run. No new pilot or production promotion.
See [V36 authorized plan](v36_yolo_fullpasses_plan.md) and
`outputs/recognition/v36_pending_handoff.json`.

Latest: two V35 checkpoint averages evaluated with zero training updates.
Average3 (2000/3500/4001) passes all 17 gates in both FP16 and BF16; average4
(1500/2000/3500/4001) passes FP16 but fails two BF16 short-text gates.
Average3 is the preferred averaged development candidate, with a real tradeoff:
better historical/one-char retention, worse menu probes than final V35 FP16.
No production promotion, checkpoint replacement, or new training.
See [averaging results](v35_checkpoint_averages.md).

Workspace:`D:\profolio\japanese_translation`;PowerShell;Python environment
`.venv-recognition\Scripts\python.exe`;local RTX3070. V34 completed2full epochs,
8002updates,256000presentations and early stopped with no eligible improvement.
V35 frozen-BN first-full-pass continuation completed and sent its Herdr message.
No training is running. Combined4001updates/128000records cover521exact terms;
147BNbuffers stay fixed and optimizer step ends4001. No checkpoint promoted.
Paired inspection completed:V35 passes all full-suite retention gates under
CUDAFP16with matching V30FP16baseline;original BF16failure/best.pt untouched.
See [short inspection/precision result](v35_short_inspection.md). Candidate remains
V35fullpass/last.pt;phone/export parity not established. No new training.
See [V35 plan/results](v35_frozen_bn_plan.md).
See [V34 run/result](v34_fullpasses_plan.md) and
[final analysis](../../outputs/recognition/training/v34_broad_fullpasses6/analysis.md).
V33 curated policy failed;V34 retains broad labels with only minimal missing-term
overrides to guarantee all521eligible exact labels. No curated50 sampling.
See [V33 plan and review](v33_curated50_plan.md). No production checkpoint or
decoder was promoted.

## Code and model

- Active reusable code:`src/recognition/`,V27 architecture:MobileNetV4 ConvSmall,
  horizontal stride4,960→256 projection,orientation conditioning,two residual
  depthwise temporal blocks with BN,packed residual BiLSTM128/direction,CTC
  1,723 classes (blank0 +1,722 characters).
- H64,next fitting128/256/384 bucket,right-edge replication;packed recurrence
  excludes artificial padding via valid lengths. CNN/temporal BN is not strictly
  padding-invariant. Standard greedy CTC at beta0 remains active.
- Pilots/evaluators:`outputs/recognition/*.py`. V32 learned height collapse gave
  inconsistent small gains;keep mean pooling for the next long run. V33 tests
  reviewed646-term pool (133one/513two),50%meaningful within generated short
  slots,coverage-first then standalone-tier weights2/1.5/1. The curated arm has no
  eligible trained checkpoint (best.pt=V30 update0);control best is update1500.
  Broader noisy1-char CER worsens53→59% versus matched control. Keep V30 and broad
  label policy as defaults;do not promote curated50. V34 uses those defaults with
  user-requested minimum521-term coverage;128000records per complete epoch.
  Audit verifies521/521exact labels per epoch,404minimal generated-pair overrides.
  V34 best.pt is tensor-identical V30 update0;last.pt=epoch_02.pt at update8002.
  Resized clean/noisy CER13.33/16.61→9.76/12.06%,but original clean4.95→7.24%,
  historical10→12.37%,seen4.97→7.67%. Clean regressions are deletion-dominated.
  All trained checks fail retention;do not promote or resume this recipe.
  The zero-update four-way BN-buffer swap completed:V34weights/V30BN recovers
  original clean7.24→5.43%,seen7.67→5.45%,held7.82→6.18%,while resized clean/noisy
  is11.33/14.00%. Historical and original pooled1/2-char gates still fail;heavy
  appearance worsens65.18%. No promotion. User authorized V35:matched
  2000-update V30 continuation with fixed V30 BN stats,affine gamma/beta trainable;
  exact V34 first2000batches and LR schedule,normal control already saved.
  V35 improves normal-BN seen5.52→4.94%,resized clean11.21→10.24%,noisy14.30→13.15%,
  historical10.29→9.92%. Final fails only original pooled2-char57/400=14.25%,
  cap54/400=13.5%;all earlier checkpoints also fail. best.pt=V30;last.pt=trained
  update2000. Verified147BNbuffers unchanged,98affine tensors trained.
  Actual63980presentations encounter313/521terms;full corpus retains521coverage.
  Full-pass result at4001:seen5.03%,held5.50%,original4.86%,historical10.51%,
  resized clean/noisy10.18/12.67%,confirmation25.43/29.60%. Final still fails
  pooled1-char61/200(cap60) and2-char56/400(cap54);best.pt=V30,last.pt=trained4001.
  Output:outputs/recognition/pilots/v35_frozen_bn_fullpass4001/. No second epoch.
  Inspection:12/14one-char regressions are vertical;all5new blanks vertical.
  No full-Tdecode changes across400cases. Precision shifts narrow gates:
  FP16pooled1/2errors V30=59/50,V35=58/53;all15-set FP16gates pass. No promotion;
  original BF16caps/results unchanged. Next suggestion:export/device parity,
  then paired vertical evaluation before more training. See v35_short_inspection.md.
  Full report:outputs/recognition/evaluation/v34_bn_swap/analysis.md. Source
  checkpoints unchanged;both original baseline predictions reproduce exactly.

## Checkpoints: do not confuse best.pt with the trained candidate

| Purpose | Local path |
|---|---|
| V27 trained reference |`outputs/recognition/training/v27_font_full20/epoch_20.pt`|
| V27 best.pt |V26 initialization;no trained epoch passed every gate|
| Recent adaptation reference |`outputs/recognition/pilots/v30_short_retention_2000/control/best.pt` (update2000)|
| Latest trained candidate |`outputs/recognition/pilots/v31_curated_short_2000/curated_70/step_2000.pt`|
| V31 curated best.pt |Update0,V30 source weights;curated arm failed retention|
| V31 control best.pt |`outputs/recognition/pilots/v31_curated_short_2000/control/best.pt` (update500)|

V31 curated improves menu-probe clean1/clean2/noisy1/noisy2 CER to
29/34.5/33.5/39.5%,versus matched control32.5/36.75/39.5/41.75%. It uses the
50%original/25%clean-render/25%noisy-render mixture,with70%curated labels only
within generated short examples. Overall length quota stays unchanged.
Both final arms made61/200 original one-character errors;cap60,baseline58.
V30's earlier20/20 short oversampling did not beat its control.

## Latest decoder finding

Frozen V31 curated step2000,3,434 one-character images,35 decoder modes:

| CER | Standard | Empty-only fallback |
|---|---:|---:|
| Menu clean1 |29%|24%|
| Menu noisy1 |33.5%|29%|
| Original200 |30.5%|28%|

Fallback changes only an entirely empty greedy output:choose one nonblank token
at the valid frame with highest nonblank softmax probability. Keep other outputs
and temporal CTC blanks unchanged. No dictionary or known-length oracle used.
It is evaluated only,not installed. Need matched V30 decoding,multi-character
and detector false-positive tests before adoption. Yen remains weak when noisy.

## Dictionary and data

- Main fuzzy dictionary:`data/processed/lexicon/culinary_menu.db`,47,216 entries;
  `src/data/lexicon_builder.py:CulinaryLookup`. Not applied in reported raw CER.
- Main database has menu-sense selection issues (e.g.ピザ,中生,大福);not repaired.
- Short review:`outputs/recognition/curation/v31_short_terms/`;Gemini Argon via
  Herdr plus parent review. Approved496 terms (114one/382two),485 agreements plus
  11 sourced essentials.3,474 uncertain;1,407 vocabulary-gap entries retained.
  Read `VALIDATION.md`;raw Gemini summary has corrected arithmetic/count notes.
- Active datasets:`data/datasets/recognition/v27_fonts/` and `v27_validation/`.
  Fonts/shared licenses remain in `data/resources/fonts/` and `fonts_v26/`.
- Archive:`archive/stage3_legacy_20261111/`,V1–V28 history/source/reports. Archived
  images and ZIPs were deleted (7.08GiB);labels/metadata remain. Active data intact.

## Git and workflow

- Remote:`https://github.com/ccouu/Tanzaku-OCR.git`,branch `main`.
- Source and selected reports only. Never commit datasets,curated training lists,
  weights,databases,generated images,logs,credentials or environments.
- Root `/data/` is ignored;`src/data/` and `scripts/data/` must remain tracked.
- Last functional check:15 recognition tests passed. Documentation updates do not
  change training weights or decoding. User prefers autonomous execution and
  minimal polling/token use for background jobs.
- Preserve raw CER versus post-lexicon accuracy. Existing probes are development
  diagnostics;real-handwriting accuracy and mobile latency remain unestablished.

For complete current metrics and limitations,read [stage3_summary.md](stage3_summary.md).
