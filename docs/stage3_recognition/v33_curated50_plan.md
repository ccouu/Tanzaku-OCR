# V33 reviewed short-label pilot and gated long-run plan

## Review outcome

AGY reviewed all496 original short terms and proposed220 supplements. Parent
validation checks identities, duplicates, supplied hashes, JSON fields, source IDs,
and the current1722-character OCR vocabulary. The separate final pool contains
646 unique terms:133 one-character and513 two-character. It uses470 original
keeps plus176 in-vocabulary additions. Review entries, exclusions, and39
out-of-vocabulary supplements remain outside the curated stream. The dictionary,
OCR vocabulary and original review artifacts are unchanged.

Read [parent review](../../outputs/recognition/curation/menu_term_frequency_review/REVIEW.md)
and [AGY evidence corrections](../../outputs/recognition/curation/menu_term_frequency_review/EVIDENCE_CORRECTIONS.md).
The original fish URL points to fermentation, the original Matsuya URL to ecommerce,
and one kaiseki dictionary entry cannot support every reading/counter claim.
Replacement sources and explicit expert-estimate markers correct the scope.
The exact original Matsuya URL in sources.json is e-shop.matsuyafoods.co.jp/collections/single;
AGY's correction text instead mentions the shop landing page. This does not change
the substantive ecommerce-versus-restaurant-menu finding.

High/Mid/Low are expert ordinal estimates, not measured probabilities. Source
support is representative/contextual rather than an attestation of every term.
The original review's claim that broad sampling guarantees uniform glyph exposure
is not accepted. Exposure must be counted. Exclusion from curated short labels
does not remove a character from recognition coverage.

## Implemented and completed: matched pilot

- Code: [tiered_short_pilot.py](../../outputs/recognition/tiered_short_pilot.py).
- Both arms start from V30 control/best.pt,update2000. Mean-height pooling,
  existing temporal BN and packed BiLSTM; no V32 learned height collapse.
- Each arm2000 updates,batch32:16 original,8 clean application renders,
  8 paired mildly distorted renders. Encoder5e-7/head3e-6,constant AdamW,
  no decay on norm/bias,gradient clipping5. Standard greedy beta0 evaluation.
- Original replay IDs,batch order,length quotas,font assignments,physical buckets,
  and longer generated images are matched. Short views are rerendered in both arms.
- Treatment replaces approximately50% of generated short pairs with reviewed
  meaningful terms; the rest use historical broad labels. Rounded fractions are
  audited separately for one and two characters. Rare inherently256px short
  layouts replay unchanged in both arms; curated selection comes from128px slots.
- Cover eligible unseen terms before repeats, then relative per-item standalone
  weights High2/Mid1.5/Low1. General-menu tiers remain metadata. This jointly
  tests the revised pool and weighting, not the50% ratio in isolation.
- Fresh menu probe uses50 terms per length,4 font draws each,clean/noisy pairs.
  About20% of eligible terms are withheld from current curated generation;
  historical/original training can contain them. Existing broader probes retained.
- Evaluate every500 updates with the prior selection and +1pp retention gates.
  best.pt may remain update0. Final and selected checkpoint results are distinct.

The10-update GPU smoke test completed both arms with finite training. Its tiny
evaluation subset is a technical check, not evidence of model improvement.

Outputs: `outputs/recognition/pilots/v33_curated50_tiered_2000/`.
Launcher: [run_v33.ps1](../../outputs/recognition/run_v33.ps1).
It sends one Herdr completion/failure message to Codex after the process exits;
no result-polling loop is used. Logs/status are under outputs/recognition/v33*.
The launcher runs the pilot and paired analysis only; it does not start a long run.

## Result: curated policy did not pass

Both2000-update arms completed. Curated50 has no eligible trained checkpoint;
its best.pt remains the V30 initialization. Broad control selected update1500.
At equal2000updates,menu clean/noisy CER improves30.17→29.33% /39.17→38.33%,
but broader noisy one-character CER worsens53→59% and clean one-character
CER worsens44→46%. Curated final fails original pooled one-character and fresh
clean one-character caps. All521eligible terms were sampled, but130receive
only one clean/noisy pair,including円 and茶. Do not infer causality from this
joint pool/ratio/weighting intervention. See the
[decision report](../../outputs/recognition/pilots/v33_curated50_tiered_2000/VERDICT.md).
No long run was launched;retain V30 and the broad-label policy as defaults.

## Long run remains gated

Adopt the short-label policy only if a selected retention-eligible checkpoint
improves menu-short evaluation against the matched control at the same update.
Review term exposure and uncertainty before drawing a conclusion from one seed.

The agreed long-run target is a maximum6 full epochs from V30,mean pooling,
same15 fonts and50/25/25 mixture. Prepare a frozen corpus of64000 original
images plus32000 clean/noisy render pairs:128000 image presentations per epoch,
each record visited once with no steps-per-epoch cap and no dropped remainders.
Group by nearest128/256/384 bucket with border replication. Keep native size
variation,aspect-preserving H64 resize,margin variation and mild appearance recipe.
No new morphology,stronger blur,fixed384 stream or short-length oversampling.

Encoder initial LR5e-7/head3e-6,fully unfrozen,cosine to20% of initial rates.
Evaluate every2000 updates and stop after3 successive evaluations without an
eligible improvement. Retain V30 as fallback. Larger frozen short confirmation
data should be prepared before this long run; it is not part of the current pilot.
No long-run dataset or long training has been launched yet.
