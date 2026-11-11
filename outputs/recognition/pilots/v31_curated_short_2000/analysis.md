# V31 curated short-term pilot analysis

Both matched arms completed2,000 updates in17.20 minutes total,including data
generation and evaluation. Starting V30 control checkpoint unchanged. Raw greedy
CTC at blank penalty0,no dictionary correction. Normal training BatchNorm,
constant backbone/head learning rates5e-7/3e-6.

## Main finding: curation helps relevant short terms,retention is not resolved

| Frozen menu-probe CER | Starting V30 | Control2000 | Curated70 arm2000 |
|---|---:|---:|---:|
| Clean1 character |32.50%|32.50%|29.00%|
| Clean2 characters |38.50%|36.75%|34.50%|
| Noisy1 character |39.50%|39.50%|33.50%|
| Noisy2 characters |42.50%|41.75%|39.50%|
| Equal macro across these four cells |38.25%|37.625%|34.125%|

Curation improves all four cells relative to the matched control. Macro CER is
3.50 absolute percentage points lower than control and4.125 lower than starting
V30. A paired bootstrap stratified by length and clustered by100 term identities
(50 per length),keeping clean/noisy/font views together,gives treatment-minus-
control macro CER -3.50 points,95% interval[-5.00,-2.06],5,000 draws,seed20261207.
This estimates evaluation-sample uncertainty,not variation over training seeds.

Character-weighted clean menu CER is36.50%→32.67%,and noisy41.50%→37.50% versus
starting V30. Clean exact lines increase215→230/400;noisy187→204/400. Compared
with control2000,curation saves16 clean and21 noisy edits. Most of this advantage
is fewer substitutions:clean77→64,noisy103→91;deletions97→93 and101→96.

## Does improvement extend beyond directly selected terms?

Terms withheld from the new curated-render stream also improve versus control:

| Withheld-stream term CER | Control2000 | Curated2000 |
|---|---:|---:|
| Clean1 character |32.89%|31.58%|
| Clean2 characters |43.00%|40.00%|
| Noisy1 character |44.74%|40.79%|
| Noisy2 characters |44.50%|42.50%|

These are not globally unseen labels:original replay and earlier training can
contain them. Benefits also appear with held-out fonts on this menu probe:
clean1/2 CER improves28%→24% and37%→34% versus control;noisy1 improves37%→33%,
while noisy2 ties39%. This supports useful curriculum adaptation but does not
establish semantic understanding or real-menu accuracy.

## Retention failure and checkpoint selection

Both final arms fail the predefined pooled original one-character gate:

- Starting V30:58 edits /200 characters =29.0% CER.
- Allowed cap:60/200 =30.0% (+1 absolute point).
- Both final arms:61/200 =30.5%.

The failure is one edit beyond the cap,three more than starting V30. It is small,
but should not be silently waived after inspecting results. Other final retention
checks pass. Curated one-character seen-font CER rises31.67%→36.67%;held-out
one-character improves37.5%→35%;original969 one-character rises24%→25%.

The selected checkpoints therefore are:

- `control/best.pt`:update500,the only evaluated trained control checkpoint
  passing every gate.
- `curated_70/best.pt`:update0,the unchanged V30 starting weights.
- `curated_70/step_2000.pt`:actual improved curated candidate,but ineligible.
  Do not confuse it with the treatment's best.pt.

Broader metrics at update2000:

| CER | Starting V30 | Control | Curated |
|---|---:|---:|---:|
| Application clean |13.33%|12.00%|11.94%|
| Application noisy |16.61%|14.67%|14.91%|
| Seen fonts |4.97%|4.94%|5.15%|
| Held-out fonts |5.86%|5.73%|5.86%|
| Historical1000 |10.00%|9.80%|10.12%|
| Provisional YOLO38 |90.95%|90.95%|90.05%|
| Strong appearance100 |63.93%|63.75%|64.46%|

There is no broad recognition win over control. On the older unconstrained
short probe,curated noisy1 CER is57.5% versus54.5% control,and clean2 is17.75%
versus16.5%. This is targeted specialization,not a universal short-text fix.
YOLO still has0/38 exact lines;real-menu deployment is not established.

## Exposure and remaining hard cases

Original replay,batch order,lengths,font settings,orientation and physical bucket
assignments match between arms. Of2,654 generated short pairs,1,858 (70.0075%)
have curated labels:484 one-character and1,374 two-character pairs. This affects
3,716/64,000 image presentations (5.81% of total),not70% of all training images.
The overall short-text quota is unchanged. Fifty-eight legacy wide short pairs
are replayed identically;the remaining replacements use eligible128px layouts.
Real crop dimensions/valid lengths can vary with glyph identity.

The curated stream samples94 unique one-character and303 two-character terms.
Median coverage is only5 and4 fresh pairs per term respectively (10 and8 image
presentations). It is not a50-occurrence-per-term curriculum.

`円` receives14 pairs (28 image presentations). On the new probe it remains
wrong/blank on all4 clean images;noisy exact accuracy changes1/4→2/4 versus
starting V30 (control0/4). Four images per term/view are too few for a reliable
individual-term conclusion,but there is clearly no robust円 recovery yet.
Similarly,鶏 remains difficult under appearance changes. Increasing relevant
labels does not automatically remove glyph-specific failures.

## Recommendation

Keep the curated70/30 idea as a promising training-data direction;it outperformed
its matched control on the intended menu probe. Keep V30 as the established
reference and preserve curated step2000 as an experimental candidate. Do not
promote treatment best.pt as a newly trained model—it contains starting weights.

Before long training,repeat the matched recipe with another seed and a larger
frozen one-character retention set. The immediate uncertainty is whether the
small retention regression is reproducible. Separately audit persistent failures
such as円 with more fonts/layouts and inspect CTC emission paths. If further
training is warranted,term-coverage floors/fresh variants deserve testing;simple
replay frequency or stronger blank penalties have not solved the issue.
No new training or model promotion was performed during this analysis.
