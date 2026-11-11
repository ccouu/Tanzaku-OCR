# V29 pilot analysis

Completed all2,000 updates in13.56 minutes including rendering and validation.
Best eligible checkpoint is `best.pt`, update2,000; the source V27 epoch20 file
is unchanged. Each32-image batch contained16 original images,8 clean application
renders and8 textured/color renders with mild blur/noise. There were32,000 unique
original records and16,000 generated pairs (8,471 unique generated transcriptions).
This was a64,000-presentation pilot, not a full pass through all64,000 original
training records. Backbone/head learning rates were constant5e-7/3e-6.

## Main result

All numbers below are raw greedy CTC CER at blank bias0.

| Evaluation | V27 start | V29 update2000 | Change in percentage points |
|---|---:|---:|---:|
| Application clean |28.85%|15.70%|-13.15|
| Application textured/color, no added noise |42.61%|20.18%|-22.42|
| Application textured/color plus mild noise |44.42%|20.91%|-23.52|
| Original seen-font validation |4.48%|4.70%|+0.21|
| Held-out fonts |5.27%|6.00%|+0.73|
| Original retention969 |4.70%|4.89%|+0.19|
| Historical crops1000 |11.27%|10.17%|-1.09|
| Provisional YOLO38 |96.38%|90.50%|-5.88|
| Strong appearance100 |66.96%|65.00%|-1.96|

The application noisy set has52.9% fewer character errors,733→345. Exact lines
increase47→112/300. Clean exact lines increase87→144/300. Every one of the15
fonts improves on both application clean and noisy views. Low-contrast noisy CER
falls60.04%→27.03%; medium39.23%→18.04%; high35.29%→18.18%. All four procedural
texture groups improve.

Application clean/noisy CER improves at every500-update evaluation, ending at
15.70%/20.91%. Update1500 gives16.91%/22.55%, so there is no observed application
plateau by2,000 updates. This does not establish that a long continuation will
retain other distributions.

## Error mechanism and tradeoff

| Noisy application errors | Start | Update2000 |
|---|---:|---:|
| Deletions |539|165|
| Insertions |17|50|
| Substitutions |177|130|

Deletion reduction accounts for most of the improvement. However, insertions
increase on all clean retention sets. For held-out fonts, deletions59→38 are
outweighed by insertions18→54 and substitutions39→40. This pattern is consistent
with more aggressive character emission, but does not prove a particular CTC
calibration or BatchNorm mechanism. The current experiment cannot separate
weight adaptation from normalization-statistic adaptation.

Aggregate retention caps pass at every evaluated step. They monitor original
font macro, original969, historical1000 and provisional YOLO, each within
starting V27+1 absolute CER point. They do not separately protect held-out fonts
or short-length strata.

| Short-text CER | V27 start | V29 update2000 |
|---|---:|---:|
| Application clean,1 character (30 lines) |53.33%|43.33%|
| Application noisy,1 character (30 lines) |63.33%|46.67%|
| Original969,1 character (100 lines) |18.00%|22.00%|
| Seen fonts,1 character (60 lines) |28.33%|31.67%|
| Held-out fonts,1 character (40 lines) |32.50%|37.50%|
| Held-out fonts,2 characters (40 lines) |8.75%|17.50%|

Thus the application gain is real on this development set, but the longstanding
short-text problem is not solved. Meiryo held-out CER also rises6.36%→8.18%.
Secondary blank bias0.2 does not help the final application result: clean stays
15.70%, noisy becomes21.03% versus20.91% at bias0.

## Practical interpretation

This supports changing the training rendering/crop distribution before changing
the architecture. Substantial clean application improvement occurs with the
existing model, and textured/color improvement is larger still. There is no
separate matched original-only continuation arm, so this run cannot assign
causal gains individually to rendering, texture, noise, low learning rate or
BatchNorm changes.

The same procedural generator and known15 fonts were used for training and
application evaluation. Generated training labels exclude monitored validation
transcriptions, but the300-example application probe had already been examined
and is a development set. Natural handwriting, detector errors and real menu
backgrounds are not established by these results. YOLO still has0/38 exact
lines, and strong appearance remains6/100 exact despite lower CER.

Recommendation: keep V29 update2000 as the adaptation candidate and retain V27
as the clean reference. Before long training, evaluate a fresh application set
with unseen transcriptions/seed and held-out fonts. Add explicit1-/2-character
and held-out-font retention constraints to checkpoint selection. If those
regressions persist, test a targeted short-text replay adjustment under the
same budget rather than immediately increasing epochs or blank bias. No further
training was launched as part of this analysis.
