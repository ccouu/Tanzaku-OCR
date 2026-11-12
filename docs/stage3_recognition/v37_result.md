# V37 completed: excellent rendered long-lines, no eligible replacement

Verified2000updates/64000presentations:600512batches/19200images(exact30%),
4800presentations for each length11/12/13/14. 15trainingfonts/fourheldoutfonts,
disjointfreshlonglabels,allimage/plan/sourcehashesandinitialpredictionparity
verified.147BNbuffersfixed in raw/EMA/selected/final. No GPU used by this review.

**Keep V36best.pt(raw7000)asdefault. V37best.pt retains these exact weights at
V37update0.** None of eighttrainedraw/EMAevaluationspassedall17gatesinboth
precisions. Finalraw/EMApassFP16butfailBF16one-char61/200edits(cap60/200).

| FP16 probe | Starting V36 | Final raw2000 | Final EMA2000 |
|---|---:|---:|---:|
| Long512 clean,allfonts |6.50%|1.10%|1.14%|
| Long512 noisy,allfonts |6.92%|1.56%|1.72%|
| Paired fitted384 clean |30.12%|17.92%|18.58%|
| Paired fitted384 noisy |33.22%|21.68%|22.34%|
| Menu noisy |35.33%|37.67%|37.67%|
| Strong appearance |61.61%|63.57%|63.57%|
| New YOLO heldout,38crops |74.75%|72.73%|73.23%|
| Legacy YOLO provisional |87.33%|89.14%|90.50%|

Heldout-fontmean512clean/noisy improves6.70%→1.30%raw(1.40%EMA). Finalclean
known/heldout1.08%/1.12%,noisy1.64%/1.48%. 14charfitted384clean36.64%versus
5120.86%,noisy46.64%versus1.50%. Strong result on these renderedlonglines.

There is already a largezero-update512benefit withunchangedV36. Prefer testing
V36weightswith512routing as the nextcandidate before replacing recognizerweights.
Thewidthcomparison changes glyphresolutionandcanvas/Ttogether,notjustCTCcapacity.

All32wideYOLOcrops are fromtrainingmenus;zerooverwidthcropsinthe38cropheldout.
Wideroutingon32crops changesV36CER69.44%→57.57%;V37rawgets52.82%,butonly3/32
exact. No evidenceyet on independentlongYOLOlines. Overallheldoutexact falls3→2/38
despiteCERimprovement,andits paired95%CERinterval crosseszero. Domainshiftremains.

The30%newlongstreamishorizontal,droppingverticalexposure24.22%→16.91%and
dilutingoriginal/shortreplay. Thusshortretentiondriftcannotbeattributed solelyto
physicalwidth. No matched no512trainingcontrol. Ifanotherpilotisneeded,reduce
longfractionandpreserveexplicitshort/verticalcoverage. No newtrainingauthorized
orlaunched by this review. V36androotdetectorweights remain unchanged.

Full review:`outputs/recognition/pilots/v37_long512_2000/review.md`.
Independent audit:`completion_verification.json`;pairedbootstrap:`paired_analysis.json`;
orientation/short/wideYOLOmetrics:`stratified_analysis.json` in the same folder.
LatestYOLOfine-tuningalsofinished separately;newdetectorpredictionswereNOTused
inV37. End-to-endnewdetector+recognizerrecognitionhasnotbeenmeasured.
