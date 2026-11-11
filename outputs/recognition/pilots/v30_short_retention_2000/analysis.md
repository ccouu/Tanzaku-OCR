# V30 result analysis

Both arms completed2,000 updates,4,000 total, in14.08 minutes including probe
generation and evaluation. Both selected their update2000 checkpoints under the
predeclared selection score and retention gates. The starting V29 file was
unchanged. All results below use standard greedy CTC, blank penalty0.

## Outcome: prefer the control continuation

| CER | Starting V29 | Control2000 | Short20/20 arm2000 |
|---|---:|---:|---:|
| Application clean |15.70%|13.33%|14.61%|
| Application noisy |20.91%|16.61%|18.30%|
| Seen fonts |4.70%|4.97%|4.97%|
| Held-out fonts |6.00%|5.86%|6.41%|
| Original969 |4.89%|4.95%|5.49%|
| Historical1000 |10.17%|10.00%|10.53%|
| Provisional YOLO38 |90.50%|90.95%|90.95%|
| Strong appearance100 |65.00%|63.93%|63.21%|

Continuing the existing V29 recipe improves application clean/noisy CER by2.36
and4.30 absolute percentage points. Control beats treatment by1.27 and1.70
points respectively. Treatment has a small strong-appearance advantage, but both
remain poor on that set and both produce0/38 exact YOLO lines. This does not
establish real-menu readiness.

## Fresh short-text probe

There are200 unique one-character strings and200 unique two-character strings,
each with paired clean/noisy images; half use known fonts and half held-out
fonts. Characters are in the trained vocabulary. Full strings are excluded from
current replay and V29 generated training, but can have appeared in original
historical training. This is rendered-image generalization, not unseen-character
recognition.

| CER | Starting V29 | Control2000 | Short20/20 arm2000 |
|---|---:|---:|---:|
| Clean1 character |45.00%|44.50%|46.50%|
| Clean2 characters |21.00%|17.50%|19.75%|
| Noisy1 character |60.00%|58.50%|56.50%|
| Noisy2 characters |27.25%|25.50%|25.25%|
| Equal macro of these four cells |38.31%|36.50%|37.00%|

The noisy one-character treatment advantage is just four fewer edits on200
characters. It costs four extra edits on the clean one-character view; clean
two-character performance is also worse. A paired, length-stratified bootstrap
over400 strings, keeping clean/noisy views paired, gives treatment-minus-control
macro CER +0.50 percentage points,95% interval[-1.19,+2.31] points,5,000 draws,
seed20261203. This measures evaluation-sample uncertainty, not training-seed
variation. There is no clear short-text superiority for the oversampling policy.

One-character substitution counts remain dominant: treatment clean has19
deletions,15 insertions and59 substitutions; noisy has23/18/72. Therefore this
fresh probe does not support blank suppression as the main remaining issue.

Original short-text retention is mixed too. Held-out one-character CER is37.5%
for control versus40.0% treatment; two-character is12.5% versus15.0%. On the
original969 subset, one-character CER favors treatment22% versus24%, while
two-character CER ties at11.5%. Passing pooled caps does not mean every short
stratum improves.

## Important experiment limits

Both arms used the same immutable original/rendered source pixels,50/25/25 image
mixture,learning rates,budget and architecture. Nevertheless, these replay
policies differ in more than the marginal short-text percentage:

- Control used31,884 unique original records and all16,000 generated pairs.
  Treatment used23,080 unique originals and8,991 unique generated pairs.
- Treatment replayed existing short images with replacement; it did not create
  fresh appearance variants. Its128px batch fraction increased18.55%→40%.
- Treatment grouped short lengths into homogeneous batches; control kept V29's
  mixed-length batches. Training BatchNorm therefore saw a different batch mix.
- Font/orientation frequencies and longer-text coverage can differ conditionally
  on length/bucket. Complete counts are in sampling.json.

These differences prevent concluding that short-text emphasis is inherently
harmful. The tested combination of oversampling, repeated pixels, coverage and
batch grouping did not beat the control.

## Recommendation

Keep `control/best.pt` (update2000) as the stronger application-adaptation
candidate, with V29 and V27 retained as references. Do not adopt the20/20 quota
for long training from this result. First inspect confidently wrong single-glyph
predictions by character/font/stroke size. If another pilot is warranted, use
fresh short-text render variants and preserve longer-image coverage, while
matching batch grouping between arms. No additional training or model promotion
was performed during analysis.
