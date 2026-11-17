> Historical experiment record. Status, sources, vocabulary, geometry and gates below belong to this version; they are not current deployment instructions. See [current implementation/status](current_status.md) for V75/V76, V77 authorization and production V43. Referenced cached assets may have been retired; do not regenerate or overwrite them merely to follow this record.

# V49: selective Latin/digit/parenthesis classifier pilot

User authorized the head-only plan after V47/V48 review. Source is promoted
V43finalraw2000, not V47/V48 weights. Maximum2,000updates/64,000presentations,
logicalbatch32,microbatch4;noautopromotion,nofurtherunfreezing,nocleanup.

## Completed and independently reviewed

Completed2,000updates/64,000presentations in25.70minutes;exactratiosandcoverage
verified.All521terms≥2times,all6,000digitimages/all1,561YOLOcrops,3,200ASCII/
3,200fullwidthbracketpresentations.21,635consumedimagehashes+156holdoutverified.
Only66rowschangedwithindependentLRgroups;BiLSTM/allsharedweights/remainingrows/
blank/BNaffine/147buffersbit-identicalinrawandEMA. Teacherpolicyandeligibility
verified(25,529eligible/71Latin-labelreplayskipped). Initialparitybothprecisions.

|FP16metric|V43|Selectedraw1500|Finalraw2000|FinalEMA2000|
|---|---:|---:|---:|---:|
|TinyLatintrainCER|101.40%|23.60%|22.55%|25.70%|
|MultifontheldLatinCER|101.62%|62.13%|62.98%|65.30%|
|MultifontheldLatinmacrorecall|0.28%|43.91%|44.68%|41.48%|
|Existingbracketsymbolmacrorecall|23.05%|25.71%|26.47%|26.69%|
|Digitheldfontmacrorecall|87.72%|86.84%|86.72%|87.26%|
|YOLOCER|41.61%|41.92%|42.02%|42.02%|
|YOLOexact|50/156|48/156|47/156|48/156|

`best.pt`=raw1500,sourcerelativeguardspassbothprecisions,still11absolutegates
failasV43. FinalrawYOLOchange+.42pp,95%menu-clusterCI[−.20,+1.05];nogain.
New33Latinrecall33.75%vsV45's37.24%;existing19recall63.66%drivesaggregate
improvementoverV45/V46. Recipesdiffer;notamatchedcausalLRablation.
RealYOLObrackethitsunchanged3/26andfullwidthopeningprobe0/47. Digitsapproximately
retainedbutnotimproved. Sharedfeatures/oldlogitweightsfixed,so residualold-task
errorscanarisefromcompetitionwithadaptedrows(newLatinfalseemissions2YOLOlines).

KeepV43promoted;donotblindlyextendthisrecipe. V49isausefulfrozen-featurecontrol
foramatchedlowLRBiLSTMarmwithsame66rowgroups/data/KL,ifsubsequentlyauthorized.
Nofurthertrainingrunning. Fullreview/audit:`outputs/recognition/pilots/v49_selective_head_2000/review.md`,
`completion_verification.json`,`paired_analysis.json`. Notificationdeliveredwith
thecorrectednormal-useroutside-sandboxlauncher;noHerdrrestart.

The sections below preserve the experiment configuration.

## Trainable rows and safeguards

- Append33missingLatinletters using blankweights/bias−8; old1,779IDs unchanged,
  blank0 unchanged;1,812characters/1,813outputs in this candidate only.
- Train all52Latinrows at1e-3, and10ASCII digit +4parenthesis rows`0–9()（）`at1e-5.
  Two disjoint parameter groups provide actual independent AdamW learning rates;
  gradient scaling alone would not provide this separation under Adam.
- Freeze allremainingclassifierrows,includingblank,andallsharedfeatures:
  BiLSTM,CNN,projection,orientationconditioning,temporalblocks,BNaffine/147buffers.
  Model stays eval throughout; no recurrent backward is needed.
- No weightdecay on the rowgroups;100updatewarmup,cosineactual2,000horizonto0.1;
  EMA.998. No automatic transition to BiLSTM training even if fitting stalls.
- FrozenexpandedV43teacher,fullsoftmaxKLweight1,T2,validframesonly onordinary/
  application/YOLOreplay whose labels contain noLatinletter. DedicatedLatin,
  digitandbracket targets useCTConly. Latin-labeledreplay hasCTCbutnoKL to avoid
  preserving the source's poorLatin predictions. Fullwidthdecimal labels normalize
  toASCII; punctuation width is preserved.
- Frozenweights andbuffers audited onraw/EMA; allnon-headparameters require_grad
  false withnogradients. Old/nonselectedrowsremainexactlyfrozen,butnewlogit
  competition can still change predictions. Retention is evaluated,not guaranteed.

## Data plan

| Stream | Presentations | Fraction |
|---|---:|---:|
|TinycleanLatin(V47's208renders)|9,600|15%|
|CleanmultifontLatin(V44's1,511renders)|9,600|15%|
|Digits/prices(V43's6,000renders)|12,800|20%|
|Balancedbrackets(V48's600renders)|6,400|10%|
|YOLOtrainingcrops|12,800|20%|
|Originalsynthetic|6,400|10%|
|Applicationclean/noisy|6,400|10%(5%each)|

Eachcompleted1,000updateblockvisitsallLatin/bracket/1,561YOLOtrainrecords andall
521requiredexactterms. All6,000digitimagesmandatoryacrossthe2,000updateplan.
Bracketpresentationssplit3,200ASCII/3,200fullwidth. Traininguseshomogeneouswidth
batches;noaddedopticalnoise,novocabularyreordering,and156menu-disjointholdout
cropsremainunchanged. Thesearepresentations,notfullcorpuspasses.

## Evaluation and selection

InitialFP16/BF16predictionparitywithV43required. Evaluate0,500,1,000,1,500,2,000,
rawandEMA:tinyfit/newlayout/newfonts,multifontLatintrain/known/heldout,new33versus
old19,bracketstylessingly,newbalancedandexistingV44probes,digit/repeat/price
metrics,YOLOtrain/holdout,old57characterprobe,and17retentiongatesbothprecisions.

BestminimizesweightedCER:25%multifontheldLatin+25%tinysetheldfontLatin+
20%existingbracketheldprobe+15%digitsheld+15%YOLOholdout. Requirebothprecisions:
noadditionalfailedgatesversusV43,YOLOCER≤source+1pp,digitmacrorecall≥source−2pp,
historicalCER≤source+1pp,andpooled1/2charCER≤source+2pp. Thisisnotanabsolute
retentionpass;V43alreadyfails11gates. Ifnonequalifies,no`best.pt`iswritten.

V43remainspromoted. Candidatecheckpointsusetheirownembedded1,812charvocabulary.

Output:`outputs/recognition/pilots/v49_selective_head_2000/`.
Code:`scripts/recognition/prepare_v49.py`,`train_v47_v48.py --arm selective`,
`run_v49.ps1`,`src/recognition/partial_classifier.py`.
Tests:two classifierfreeze/export/LR-group tests passed;seven-streamGPUforward/
backward smoke passed at`outputs/recognition/pilots/v49_selective_head_smoke/`.

Launch uses approved outside-sandbox execution so backgroundnotificationinherits
anunrestrictednormalusertoken. Herdrserver/panesandaccesssettingsunchanged.
One-shotcompletion/failurenotificationandlocalfallbackmessage;nopollingloop.
