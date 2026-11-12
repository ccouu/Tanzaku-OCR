# V37: 2000-update long-line512pilot

COMPLETED:exact2000updates/30%512presentationsaudited. No trainedcandidate
passesbothprecisionretentiongates;best.pt=unchangedV36start. See[v37_result.md](v37_result.md)
andoutputreview.md. No furthertrainingrunning.

User authorized one local2000-update run with30%512pximages containing11-14
characters. Start from V36best.pt(raw7000),not finalraw/EMA or earlierV35.
Architecture and learned parameter shapes unchanged;preprocessing and evaluation
support512inside this experiment without changing reusable production modules.

## Actual recipe

- 2000updates,batch32,64000presentations.600homogeneous512batches=19200images
  (exact30%by bothupdates andpresentations).Each of11/12/13/14chars gets4800
  presentations.1400full batches replay the existingV36selected-corpus recipe.
- Longlines use15existing trainingfonts,2400base renders/4800clean+noisyimages.
  Each cleanimage appears6times and noisyimage2times:14400clean/4800mild-noisy
  presentations (75/25),matching the prior aggregate clean/noisy balance.
- Label source:in-vocabulary11-14char `curated_grammar` menu labels in the
  existingculinary_menu.db. General JMdictmatches are excluded. These are
  compositional/syntheticmenu labels,not independently authenticated dishes.
- Split whole labels80/20before rendering;exclude existingevaluation texts
  from fresh longtraininglabels.400heldoutbase renders/800images,balanced by
  length,halfknownfonts/halffourheldoutfonts. Historical unseen-label guarantees
  are not claimed. Training longlines horizontal;vertical examples remain in
  the other70%V36stream,so orientation balance changes and must be reported.
- NaturalH64validwidth385-512,right-edge replication to512,no clipping or
  anisotropic squeeze. Try native margin layouts;when a font's visible height
  prevents fitting,extend background guards before proportionalH64resize.
  Record guard fallback counts and scaled glyph-height range.
- Paired heldout512versus384aspect-preserving fitted/letterboxedviews of the
  same renders. SeparateYOLO512train/holdoutprobes widen only32previously
  oversizednativecrops;othercrops retain V36pixels. NewYOLOholdout never trains.

## Optimization and selection

FreshAdamW,no norm/biasdecay;encoder5e-7/head3e-6;200warmupupdates;cosine to10%
at update2000. BF16training,FP32CTC/packedrecurrence,clip5.147V30BNbuffers fixed;
affine and otherweights train;EMA.998. Evaluate raw/EMAat500/1000/1500/2000.
Keep V36asfallback. Select improved heldout-fontlong512clean/noisymean CER only
when all17existingV30-relative gates pass in FP16andBF16. Save selected,rawfinal
andEMAfinal separately. No architecture/mobile promotion or extraupdates.

## Execution and checks

Code:`outputs/recognition/v37_long512.py`;launcher:`run_v37.ps1`.
Output:`outputs/recognition/pilots/v37_long512_2000/`.
Logs:`outputs/recognition/v37.stdout.log`,`v37.stderr.log`.
Status:output`status.json`;process metadata:`outputs/recognition/v37_process.json`.
One Herdr completion/failure notification to w2:p1,no polling loop.

Preflight exercised76font/lengthcombinations and512forwardoutput128x1x1723.
First preparation attempt stopped before training on an unfitbrushfontlayout;
its sampledJMdictphrase also exposed an overlybroadfooddictionarymatch.
Restricted labels tocuratedmenugrammar and implemented background-onlyguard
fallback,verified at nativefontsizes24/48/120. The failure is preserved in
output`preparation_failure.json`;restart rejects any output with checkpoints.
Relaunched revisedrun without modifying V36weights.

Important limitations:no matched no512training control,novellongrender-style
andorientationmix differ from V36. Paired384/512inference and originalprobe
retention separate some effects,but cannot isolate all causes of training gains.
