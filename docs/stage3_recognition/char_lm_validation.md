# Character-LM decoder validation and candidate fixes — 2026-11-19

Recognizer: **L2_v06 epoch2 EMA**, now explicitly user-promoted in the registry. The decoder candidates described here remain **unapplied**. Neural checkpoint promotion does not imply Char-LM/R1 promotion.

## Matched decoding evidence

Frozen model/vocab/LM assets, identical cached canonical FP32 logits and valid lengths. Alpha selected on val10 only from 0.10/0.20/0.30; test5 remains a reused development set, not blind confirmation.

| Decoder | Validation: 129 crops /10 menus | Test: 62 crops /5 menus |
|---|---|---|
| Greedy | CER5.71%,107 exact | CER7.84%,50 exact |
| True plain beam5/top6 | Same as greedy | Same as greedy |
| HP1010 forward4gram, alpha0.30, beam5/top6 | CER3.62%,117 exact | CER6.58%,53 exact |

LM scoring, rather than beam aggregation alone, supplies the observed gain. Menu-clustered 95% bootstrap intervals for CER difference exclude zero, but only5 reused test menus limit confidence. Test5 contains just **one vertical crop, one character**. HP710 versus HP1010 comparisons confound corpus size with context/schema changes.

## Routing tradeoffs

Candidate R1 uses greedy when the canonical **greedy prediction**, after trimming whitespace, has <=2 characters; otherwise uses HP1010. It never uses ground-truth length. R2 additionally rejects numeric-span changes.

- R1 retains validation/test CER3.62%/6.58%, and long512 clean/noisy0.92%/2.48%.
- Clean-short CER returns from global-LM9.83% to greedy7.33%; noisy-short13.33% versus greedy13.50%. The tested short-string harmful changes are eliminated, not a guarantee for unseen short text or deletion-shortened predictions.
- Monetary numeric-span exact recognition improves127→143/160 with Global/R1:17 fixes and1 damaged amount. R1 also retains7 general-numeric leading-junk harms. Numeric-span correctness and currency correctness are distinct.
- R2 removes more numeric harms, but forfeits40 beneficial digit-probe corrections, including all17 monetary fixes. R1 is a preferred accuracy compromise, **not strict dominance over R2**.
- One already-wrong test line gets worse under LM decoding. Zero previously-exact lines damaged does not mean zero harmful changes.

## Confirmed implementation problems and candidate status

1. Existing threaded inspector mutates shared LM digit mode: race confirmed. Candidate uses immutable models and frozen per-request DecodeConfig; concurrent mocked HTTP tests passed.
2. Existing alpha<=0 shortcut is greedy, not plain beam. Candidate makes decoding mode explicit; tiny brute-force tests pass with and without LM.
3. HP1010 uses `<DIG>`/`<UNK>` contexts. Turning off digit-neutral scoring on those same tables creates an invalid raw-digit diagnostic. Candidate rejects mode/schema mismatches; a genuine digit-biased comparison requires compatible trained tables.
4. Current serialization already persists digit_neutral. Candidate retains this and adds explicit schema/provenance guards; legacy files require explicit handling.
5. Schema-v2 HP1010 packaging preserves all ngram tables and all129 val10 predictions. Original model files remain intact.

Two candidate patches independently pass `git apply --check`; they applied in a disposable mirror, with7 existing tests,4 candidate tests and mocked HTTP/concurrency checks passing. Published code/patches live in [reviews/char_lm_20261119](../../reviews/char_lm_20261119/README.md). These fixes are review artifacts, **not active source changes**.

## Cost and rollout limits

Desktop CPU R1 decoding approximately3.56ms/crop, excluding vision/transfer: roughly178ms for50 crops. Immutable Python LM load approximately1.72s, observed process-RSS increase approximately110MB; this is not a mobile benchmark. No fresh blind real-menu set is available. Add real vertical coverage and actual-device profiling before a deployment claim.

Historical validation artifacts remain locally under `outputs/recognition/validation/char_lm_20261119/` and `char_lm_followup_20261119/`. Full datasets, cached image/logit arrays and per-image predictions are intentionally not published to Git.
