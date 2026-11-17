> Historical experiment record. Status, sources, vocabulary, geometry and gates below belong to this version; they are not current deployment instructions. See [current implementation/status](current_status.md) for V75/V76, V77 authorization and production V43. Referenced cached assets may have been retired; do not regenerate or overwrite them merely to follow this record.

# V52: CTC prefix beam and optional lexicon scoring

## Completed results

Both models completed all12probe cohorts,5,265records per decoder per model.
All six decoder configurations reuse identical frozen logits. No training,
production change or label edits occurred. Literal greedy predictions exactly
match saved source evaluations on every shared probe. Brute-force prefix tests
passed;21,060lexicon-versus-acoustic digit-sequence checks passed.

| Model / decoder | YOLO CER | Exact /156 | Held-out digit recall | Clean1char CER | Clean2char CER |
|---|---:|---:|---:|---:|---:|
| V43 greedy | 41.50% | 50 | 87.72% | 48.50% | 22.75% |
| V43 beam5 | 41.19% | 52 | 88.09% | 45.00% | 21.75% |
| V43 beam10 | 41.08% | 52 | 88.09% | 45.00% | 21.50% |
| V43 beam25 | 41.19% | 52 | 88.09% | 45.00% | 21.50% |
| V43 beam25 + lexicon0.3 | 40.77% | 52 | 88.09% | 46.00% | 21.50% |
| V48 final raw greedy | 39.73% | 52 | 85.42% | 49.50% | 20.75% |
| V48 final raw beam10 | 39.00% | 53 | 85.59% | 48.50% | 20.25% |
| V48 final raw beam25 + lexicon0.3 | 39.00% | 54 | 85.59% | 48.50% | 20.25% |

These are normalized/pooled decoding results, not the earlier literal39.94%V48
or41.61%V43 numbers. Gains compare like-for-like decoding semantics. All proposed
decoder configurations pass the three source-relative development guards; this
does not mean the17model retention gates were recomputed or production approved.

V43 beam25+lexicon0.3 improves YOLO by0.73pp,13-menu paired bootstrap95%CI
[-1.29,-0.21]. V48 beam10 improves0.73pp,CI[-1.35,-0.21]. These are exploratory
intervals on reused development menus, without adjustment for configuration
selection. They are not an independent production validation.

Opening parentheses remain0/13forV43 in every decoding arm. V48 pooled greedy
has2/13; beam search has3/13. On the120-line clean held-out bracket probe,
V43 CER63.92%→62.57%,V48 49.55%→46.86%withbeam10. Search does not solve the
reviewed qualifier failures. Real standalone-price exactness on60YOLOlines rises
35→36forV43 and36→37forV48. Dedicated80synthetic-price-kind exactness stays
77forV43 but declines76→75forV48; this narrower subset excludes the separate
currency-format probes. Overall held-out digit CER improves13.73%→11.31%V43
and16.74%→14.24%V48 withbeam10.

Lexicon reranking helps familiar synthetic text more than YOLO: V43 clean512
CER1.24%→0.60%,noisy512 1.94%→1.18%,originalsynthetic6.28%→5.16%withlexicon0.3.
Unknown terms remain allowed; the score only ranks already available sequences.
The current dictionary includes substantial generated grammar and general food
vocabulary, so this synthetic benefit should not be assumed to transfer to menus.

The optimistic reference-selected best candidate from each beam25 list has
V43 CER29.82%,81/156exact;V48 CER28.68%,82/156exact. This is an oracle diagnostic,
not a real decoder. Even perfect reranking of these particular lists cannot reach
the proposed10%release target. It does not bound larger beams or stronger search.

V43 YOLO desktop decoder-only mean latency is3.24ms/line beam5,6.34ms beam10,
15.98ms beam25,16.00ms beam25+lexicon0.3; compilation and model forward excluded.
Top40frame pruning discards0.0145%mass on average,5.65%in the worst V43YOLOframe.
This is approximate search, not an exhaustive CTC or mobile benchmark.

Recommendation: keep V43 promoted. Beam10 is the practical default candidate
for later integration; lexicon0.3 is an optional domain-sensitive reranker if
its extra compute is acceptable. Do not promote V48 on this modest gain or
start a broad training run. Next recognition work should target small,
tightly spaced parenthesized qualifiers and retained short-text alignment.
Fresh human-verified real-menu confirmation is still pending.

Artifacts: `report.json`, `analysis.md`, per-probe predictions/timing/paired CIs,
and independent `completion_verification.json` under the V52 artifact root.

## Protocol

Forward-only evaluation of promoted V43 and V48 final raw2000. V48 is explicitly
the final research checkpoint, not its retention-selected EMA1000 `best.pt`.
No training, dictionary writes, label edits, vocabulary expansion or promotion.

Compare pooled-parenthesis greedy with CTC prefix beams5/10/25. The decoder
aggregates blank/nonblank probabilities, handles blank-separated repeats, and
uses log-space updates. Three tests verify exact brute-force path probabilities,
repeated-character handling, and unknown-word/digit-preservation behavior.

Search uses at most40nonblank candidates/frame and a12nat relative cutoff,
without renormalizing discarded probability mass. Per-probe top40discarded
mass is audited. CUDA FP16 logits are shared across all decoding arms. CPU
decoding timings exclude JIT compilation, disk loading and model forward;
they are desktop diagnostics, not measured phone latency.

The optional lexicon stage only reranks the acoustic beam25 N-best list with
weights0.1/0.3 and a bounded bonus for non-overlapping dictionary character
coverage. Maximum bonus is1/3nats respectively. Unknown terms receive zero
bonus and remain allowed. It does not force fuzzy replacements. Candidate
digit sequences must equal the acoustic beam25 winner, so dictionary scoring
cannot invent an alternative price. Acoustic beam search itself is unrestricted.
The existing `culinary_menu.db` is opened read-only and hash recorded; dictionary
terms are not collected from validation labels. The dictionary includes general
food/JMdict and generated-grammar terms; it is not a curated language model.

Probes cover156YOLOholdout crops, clean parentheses known/held-out fonts, digits
and prices known/held-out fonts, original/historical synthetic, clean/noisy short
Japanese, noisy application images, and clean/noisy11–14char512lines. Raw greedy
predictions are retained alongside normalized reference/pooled-CTC outputs.
Report CER/D-I-S, exact lines, opening/closing recall, digit macro recall,
digit-sequence exactness, exact price lines, length breakdowns, paired bootstrap
intervals and CPU decoding latency. Development selection minimizes YOLOCER
subject to no greater than1pp deterioration in digit recall or clean1/2char CER
relative to the same model's greedy baseline; it is not production promotion.

The156crop set has been inspected and reused for model decisions. It is now a
development benchmark. No untouched human-verified real-menu confirmation set
is available in this task; no generalization claim may substitute synthetic
held-out fonts or the inspected menus for that future confirmation.

Runner: `scripts/recognition/evaluate_v52_beam.py`.
Decoder: `src/recognition/prefix_beam.py`.
Artifacts: `outputs/recognition/evaluation/v52_beam_lexicon/`.
