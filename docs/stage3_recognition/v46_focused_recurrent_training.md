> Historical experiment record. Status, sources, vocabulary, geometry and gates below belong to this version; they are not current deployment instructions. See [current implementation/status](current_status.md) for V75/V76, V77 authorization and production V43. Referenced cached assets may have been retired; do not regenerate or overwrite them merely to follow this record.

# V46 focused recurrent adaptation — 2026-11-13

User authorized an unattended1–2hour run, increasing parentheses/digit coverage
alongside Latin, protecting basic Japanese, followed by cleanup of unused data.
V43 remains promoted throughout. No automatic promotion or further unfreezing.

## Completed results

All4,000updates/128,000presentations completed in44.04minutes, not time-limited.
Exact planned ratios and all521requiredterms(min4encounters),15trainingfonts,
1,561YOLOtraincrops,6,000digitimages and176basicanchors verified. Finalraw/EMA
preserveCNN/projection/temporal/BNaffine/147buffers; recurrent/oldheadoptimizer
steps3,800,newhead4,000 support the200updatewarmup. Independently verified32,520
consumedimagepathsand156heldoutimagehashesaftercleanup.

|FP16metric|V43|V46raw|V46EMA|
|---|---:|---:|---:|
|LatinheldoutCER|101.62%|69.25%|70.17%|
|Latinheldoutmacrorecall|0.28%|23.88%|21.98%|
|ParenthesisheldoutlineCER|58.90%|30.59%|31.28%|
|Parenthesissymbolmacrorecall|23.05%|44.68%|44.68%|
|Digitheldoutmacrorecall|87.72%|84.97%|84.15%|
|BasicJapaneseheldoutCER|42.05%|44.89%|45.45%|
|YOLOholdoutCER|41.61%|41.29%|41.19%|
|YOLOexact|50/156|48/156|47/156|

Neither final nor intermediate checkpoint meets source-relative selection;
`best.pt`doesnotexist. Both finals fail11absolutegatesbothprecisions. Rawdigitrecall
drops2.75pp and a newnoisy2chargate fails even though a noisy1chargate recovers.
YOLOrawchange−.31pp,95%menu-clusterCI[−1.51,+.93]; EMA−.42pp,[−1.72,+1.05].
V43remains promoted. Latin trainingCER66.21%stillunderfits, and heldoutrecall is
essentially tied with V45's23.92%despite extra exposure. No furthertrainingrunning.

ActualYOLOheldoutparenthesishits3/26→5/26,allclosingASCII; openingremainszero.
Latin0/3unchanged; no33newletters occur in these reference labels. Keep synthetic
improvements separate from real-menu claims. Next isolate small clean Latin
fitting and a separate parenthesis curriculum before another long continuation.

Cleanup completed:26,015disposablefiles,13,431,880,400bytes(12.5094GiB)removed.
Allnon-cachecheckpoints,unique/activedatasets,andV43/V38preserved;independent
deletionjournal/allowlist/model-load/post-cleanupimagehashverificationpassed.
Originalreport:`outputs/recognition/training/v46_focused_recurrent_4000/review.md`;
audit:`completion_verification.json`;cleanup:`outputs/recognition/cleanup/v46_post_training/`.

The following sections document the launch recipe and cleanup policy.

## Training

- Start from V43 `final_raw.pt`2000, not V45. Freshly append33Latinrows using blank weights/bias−8; old1,779IDs and blank0 unchanged.
- Maximum4,000updates/128,000presentations; logical32/micro4. This is a presentation budget, not full corpus passes.
- Planned exact mix per1,000updates:30%Latin,10%parentheses,15%digits/prices,5%basicJapanese,20%cleanYOLOcrops,10%originalsynthetic,5%applicationclean,5%applicationnoisy.
- Clean Latin targets1–6characters at128/256px; existing V44parenthesis and V43digit targets reused. Basic kana/simple kanji get clean single/repeated/two-character anchors from15trainingfonts, with4heldoutfonts reserved for probes. OOVanchors are recorded, not silently added.
- Each completed1,000updateblock visits all1,561YOLOtrainingcrops, allLatin/parenthesis/basictrainingrenders and all521requiredterms. All6,000digittrainingrenders are mandatory across the4,000updateplan. Partial/time-limited runs report actual coverage.
- Freeze encoder, projection, orientation conditioning, temporal blocks, allBNaffine and147buffers. Only33newclassifierrows train first200updates, then unfreeze the complete residualBiLSTM branch and existingclassifier.
- PeakLR:BiLSTM1e-5,oldclassifier3e-6,newrows1e-3. Warmup200,cosineactualplanned4,000horizonto0.1;AdamWdecayonlymulti-dimensionalweights. EMA.998.
- Backbone/temporalBNstay `.eval()`. The single-layerBiLSTM alone enters trainingmode for cuDNNbackward afterwarmup; it has no dropout. Frozen weights and buffers are audited.
- Frozen expandedV43teacher, full-softmaxKLweight1,T2,validframesonly on old-only synthetic/application/YOLOreplay. NoKL on dedicatedLatin/parenthesis/digit/basic targets or replaylabels containingnewLatinletters.
- Train for at most100minutes elapsed, including preparation/initial and intermediate evaluation. If capped, stop at an update boundary and evaluate final raw/EMA; intended overall window1–2hours, not a hardware-independent guarantee.
- Evaluate0,500,1,000,2,000,3,000,4,000andtime-limitendpoint. Track all17gatesFP16/BF16,Latinfit/fulltraining/heldfonts,new33versusold19,bracketsindividually,digits/repeats/prices,basicJapanese,YOLOholdout,oldlong512 andfalseLatinoutputs.
- Selected `best.pt` requires source-relative safeguards in both precisions: no additional absolute gate failures,YOLOCER≤source+1pp,historical≤+1pp,freshshortclean/noisy≤+2pp,digitheldoutmacrorecall≥source−2pp. Minimize weightedtargetCER40%Latin/20%paren/15%digit/10%basic/15%YOLO. This is not an absolute gate pass; no automatic promotion.

Output:`outputs/recognition/training/v46_focused_recurrent_4000/`.
Code:`scripts/recognition/prepare_v46.py`,`train_v46_focused.py`,`run_v46.ps1`.
Smoke:`outputs/recognition/pilots/v46_focused_smoke_verified/smoke.json`.
An initial smoke exposed cuDNN's eval-mode backward restriction; fixed by enabling
training mode only on the unfrozen recurrent branch. Original failed smoke retained.

## Cleanup after successful training/evaluation

`scripts/recognition/cleanup_after_v46.py --execute` runs only after completed
report/status,analysis andbothfinalcheckpoints exist. Audit-only mode cannotdelete.

Explicit candidates: oldpytestfixturecache,installed-packagewheelcache,unreferenced
allowlistedoldsmokerenderedimages,and`data/processed/gemini_recognition`onlyifits
entirerelative-fileinventoryandsizes/SHA256matchthepreservedcanonicalcropdataset.
Recentlymodifiedfiles,source-referenceddatasets,nonidenticalduplicates,andall
symlinks/reparsepoints are excluded or block cleanup. Every target resolves inside
theworkspaceandtheexplicitallowlist;filesarehashedagainbeforeunlinking.

Preserve all unique rawmenus,annotations,fonts,activecorpora,reviewassets,source,
environments andnon-cachecheckpoints. No code archiving or broad old-run pruning.
KeepV38rollback,V43active,andalltrainingreferences. Save deletion manifest/journal,
actualbytesreclaimed andpost-cleanupmodel-load/importverification under
`outputs/recognition/cleanup/v46_post_training/`.

Launcher performs training→evaluation→cleanup→one-shotHerdrnotification. A saved
localnotification/deliveryfailurefile remains available ifHerdrpermissionfails.
No continuous polling.
