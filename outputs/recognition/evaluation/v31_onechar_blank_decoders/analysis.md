# One-character decoding under a nonempty-crop assumption

Evaluated3,434 frozen one-character images using the actual V31 curated
update2000 weights (`curated_70/step_2000.pt`,not best.pt which contains V30).
No training or production decoder changes. Baselines on existing V31 validation
sets reproduce every prior transcription exactly. Model/checkpoint hashes are
unchanged. All decoders share the same logits and valid lengths.

## Result

Tested21 additive blank penalties0–2 in0.1 steps,13 full-softmax blank thresholds
0–1 including0.95/0.99,and one empty-output-only fallback:35 decoders total.

| Decoder | Curated menu clean1 CER | Curated menu noisy1 CER | Pooled original200 CER |
|---|---:|---:|---:|
| Standard greedy |29.00%|33.50%|30.50%|
| Blank penalty0.9 |27.00%|35.00%|30.00%|
| Blank threshold0.4 |28.50%|33.00%|29.50%|
| Blank threshold0.5 |30.00%|35.00%|29.50%|
| Blank threshold0.99 |55.50%|62.00%|45.50%|
| Empty-only fallback |24.00%|29.00%|28.00%|

Penalty0.9 is best for the clean menu set,but the noisy menu set prefers zero
penalty. Threshold0.4 gives a small joint improvement here;it does not improve
every evaluation distribution. Per-set optima are development diagnostics.

Empty-only fallback outperforms every tested penalty and threshold on each of
the nine primary evaluation groupings in summary.md. Other fallback results:

| One-character set | Standard CER | Empty-only fallback CER |
|---|---:|---:|
| Older16 meaningful menu terms,clean |15.33%|10.71%|
| Older16 meaningful menu terms,noisy |26.49%|21.58%|
| Broad short probe,clean |45.50%|41.00%|
| Broad short probe,noisy |57.50%|54.00%|
| All-font audit,clean |49.20%|43.80%|
| All-font audit,noisy |53.80%|48.80%|

## What the fallback does

1. Run standard greedy CTC with all its temporal blank frames intact.
2. Preserve any existing nonempty decoded string exactly,including multi-character
   output. No ground-truth length is supplied to the decoder.
3. Only for a completely empty result,choose one nonblank token from the valid
   frame where the best nonblank token has the highest full-class softmax
   probability. Artificial right padding is excluded.

This applies the assumption that a detector crop contains text at the string
level. It does not treat every temporal frame as containing a character.
YOLO text detection and the CTC alignment blank serve different purposes.

On clean menu1 images,19 empty outputs become10 correct and9 incorrect
single-character outputs. Total edits fall58→48;insertions stay13. On noisy
menu1 images,17 empty outputs become9 correct and8 incorrect outputs. Edits
fall67→58;insertions stay12. On original200,11 empty outputs produce5 correct
and6 incorrect recoveries;edits fall61→56. Thus deletion counts reaching zero do
not mean every empty output was correctly recognized:some deletions become
substitutions with no net edit-distance improvement.

For known nonempty one-character truth,returning one character only when the
baseline was empty cannot worsen edit distance:wrong recovery retains one
error,and correct recovery saves it. This is conditional on the nonempty ground
truth;actual blank or false-positive detector crops can instead acquire an
incorrect character. No such negative crops were evaluated.

## Why not remove temporal blanks?

Per-frame thresholding forces a nonblank token wherever P(blank) is not above
the chosen threshold. Raising the threshold therefore makes character emission
more aggressive. At0.99 it eliminates most empty strings but introduces many
extra characters. At1.0 it disallows blank everywhere:menu clean/noisy CER
explodes to803%/833.5%,with zero exact lines. CTC blank is needed for temporal
alignment even when a crop certainly contains text.

## Yen

On the42-clean/42-noisy all-font円 cases,empty-only fallback improves exact
accuracy14→25/42 clean and2→5/42 noisy. On the smaller new menu probe it improves
clean0→2/4,while noisy remains2/4. 円 still has substantial glyph-discrimination
problems,especially under appearance changes.

## Recommendation and scope

Prefer the empty-only fallback as the next application-decoder candidate for
detector-confirmed text crops,rather than a universal positive blank penalty.
Preserve or expose its confidence/fallback status to the caller. This evaluation
did not install a production decoding change.

The pooled original one-character result now falls below the raw-training cap,
but this does not retroactively pass the original V31 selection gate:the
decoder changed,and V30 must receive the same decoding before comparing model
promotion. Multi-character crops and real detector false positives also need
evaluation before a general pipeline change. Fonts/terms/images here are reused
development probes,not independent real-menu accuracy.
