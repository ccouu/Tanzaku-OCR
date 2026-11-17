> Historical experiment record. Status, sources, vocabulary, geometry and gates below belong to this version; they are not current deployment instructions. See [current implementation/status](current_status.md) for V75/V76, V77 authorization and production V43. Referenced cached assets may have been retired; do not regenerate or overwrite them merely to follow this record.

# V47 clean Latin fitting and V48 isolated parentheses

User authorized both experiments after the V46 review. Run sequentially and
independently from V43finalraw2000; V43 remains promoted. No automatic promotion,
additional architecture changes, or cleanup.

## Completed results

V47completed1,000updates in4.45minutes;V48completed2,000in27.74minutes.
No training running. Completionnotificationfailedfromrestrictedclienttoken;
laterdiagnosisconfirmedunsandboxednormal-userHerdrstatusworkswithoutrestart.

V47trainCER101.40%→1.05%(202/208exact),but samefontnewlayoutCER42.50%and
heldfont67.31%. Mostgeneralizationerrorsaredeletions. Single/doublerepeattraining
isperfect;remaining6editsare2repeaterrorsand4spaces. All52Latinrowscanfitthese
glyphswithfrozenCNN;generalizationandretentionarenotdemonstrated. Probechanges
size/text/margins together;notapurefont-onlyablation. NoKL/replaycausedYOLOCER
98.75%,all17gatesfail. `best.pt`raw1000isfitting-only,neveraproductioncandidate.

V48selected`best.pt`=EMA1000:YOLOCER41.19%,digitmacrorecall86.71%,9FP16/10BF16
failedgates,passesonlysource-relativeguards. FinalrawYOLO39.94%(52/156exact),
existingbracketprobeCER48.86%andstrictsymbolmacrorecall34.14%,butdigitmacrorecall
85.42%violatesthe2ppguardversus87.72%source. FinalEMA40.04%YOLO/85.45%digit.
FinalrawYOLOchange−1.67pp,95%menu-clusterCI[−2.95,−.46];BF16−1.56pp,
CI[−2.99,−.32]. SelectedEMA'sYOLOchangeCIcrosseszero. Reuseddevelopmentmenus,
VLMlabels,noprovenproductiongeneralization.

RealYOLObrackethits3/26→5/26rawfinal;openingASCII1/10,fullwidth0/3stillweak.
NewtargettrainingCER50%stillunderfits. KeepV43promoted;themoreconservative
V48EMA1000isanunpromoteddevelopmentcandidate. Sourcechecks,frozenCNN/temporal/
147buffers,nonLatinrowfreezing(V47),andbestcheckpointidentityindependentlyverified.
Evidence:eachrun's`review.md`,`completion_verification.json`;V48`paired_analysis.json`.

The sections below preserve the experiment configurations.

## V47: small clean Latin fitting

- Maximum1,000updates,32,000presentations,logical32/micro4.
-208cleanwhole-stringrenders:52letters×single/double/triple/shortwordcontexts.
  One fit font:NotoSansJP,48pxfontonH64canvas,128/256buckets. Background242/ink20,
  no noise, blur, random character spacing or per-character resizing. All glyphs
  checked in cmap and bounding boxes checked for clipping.
- Additional208same-fontprobes use new text/layout/44pxfont;208fontprobes use the
  four heldoutfontfamilies. Fonts and targetkind are coupled in this tiny probe;
  these are diagnostic controls, not an independent production benchmark.
- Append33Latinrows from V43 using blankweights/bias−8. **All52Latinrows** train,
  including19existingletters, via a selected-row classifier. All non-Latin rows
  remain bit-identical; export fuses into the normal classifier format.
- Train residualBiLSTMbranchLR1e-4 andLatinrowsLR1e-3;50updatewarmup,cosineactual
  1,000horizonto0.1;no weightdecay,noEMA,noKL,noJapanese/YOLOreplay. This deliberate
  unregularized fitting test can alter old predictions and is never auto-promoted.
- FreezeCNN/projection/orientation/temporal/allBNaffine/147buffers. Only recurrent
  module enters training mode for cuDNNbackward; single-layerLSTM has no dropout.
- Evaluate0,100,250,500,1,000; train/newlayout/newfontCER,letterrecall,new33/old19,
  single/repeats/words,D-I-S. FullFP16/BF16retention/digit/YOLO checks at0and1,000.
- `best.pt` means lowest trainingCER, **not** retention-qualified. Source-relative
  feature capacity is not disproven merely if this optimization setup fails.

Output:`outputs/recognition/pilots/v47_clean_latin_fit_1000/`.

## V48: separate parentheses arm

- Fresh V43finalraw,original1,779-character vocabulary; no Latin extension or V47
  weights. Maximum2,000updates/64,000presentations,logical32/micro4.
-30%brackettargets(19,200),45%syntheticreplay(28,800),25%YOLO(16,000). Synthetic
  replay:22.5%original,11.25%applicationclean,11.25%applicationnoisy oftotal.
-600cleantrainingrenders,240novel-renderprobes(120knownfont/120heldfont). Balance
  ASCII`()`andfull-width`（）`exactly9,600presentationseach, covering start, middle
  and after-space positions with realistic menu terms. OriginalV44parenthesis
  probes remain unchanged for matched comparison.
- Homogeneous native-widthbatches,replicatededgepadding; fonts32/34/36px fortrain
  and31/33/35px forprobe atH64. No glyph resizing/noise in dedicatedtargetrenders.
- Train only residualBiLSTMbranchLR1e-5/fullclassifierLR3e-6;100warmup,cosine2,000
  to0.1,EMA.998. CNN/projection/orientation/temporal/BNaffine/147buffers frozen.
- FrozenV43teacher,full-softmaxKLweight1,T2,validframesonly onallreplay,nottarget
  parentheses. SupervisedCTC throughout; all labels use full-widthdigit→ASCII
  normalization without normalizing punctuation width.
- All1,561YOLOtrainingcrops visited ineach1,000updateblock;156menu-disjointholdout
  unchanged. No claim of full corpus passes.
- Evaluate0,500,1,000,1,500,2,000 raw/EMA:fourbracketsindividually,newbalanced
  probes/existingV44probes/realYOLOsymboltransfer,digits,17retentiongatesFP16/BF16.
- Selectedbest minimizes newheldfonttargetCER,subjecttonoadditionalfailedgates
  versusV43 inbothprecisions,YOLOCER≤source+1pp,anddigitmacrorecall≥source−2pp.
  This is not an absolute retention pass. If nonequalifies,no`best.pt`iswritten.

Output:`outputs/recognition/pilots/v48_parentheses_2000/`.

Code:`scripts/recognition/prepare_v47_v48.py`,`train_v47_v48.py`,
`run_v47_v48.ps1`,`src/recognition/partial_classifier.py`.
Selected-rowgradient/exporttest passed;GPUforward/backward smoke checks passed
forLatin andallparenthesis/replaystreams. Two small sample crops visually checked.
Launcher sends one completion/failure notification and saves a localmessage if
Herdrdelivery fails. No continuous polling.
