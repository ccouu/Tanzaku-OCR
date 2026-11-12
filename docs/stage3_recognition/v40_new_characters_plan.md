# V40: append missing VLM characters and train for2000updates

User authorized vocabulary expansion and targeted training, starting from
promoted V38. V39's recalibrated BN candidate is not used.

The57characters absent from V38 across all VLM labels are appended without
changing any existing character ID or blank0:1722→1779characters,1780CTCoutputs.
The two additions found only in non-OK rows are valid Japanese characters
`久` and `。`; they receive synthetic supervision, not non-OK crop images.

Versioned vocabulary: `data/processed/recognition/vocab_v40.json`; exact
checkpoint companion: `outputs/recognition/training/v40_new_characters_2000/vocab.json`.
Historical vocabularies remain unchanged to keep V38 and old datasets usable.
V38's promoted model/registry is not replaced by this training request.

Mixture over2000updates/effectivebatch32/64000presentations:

| Stream | Images | Fraction |
|---|---:|---:|
| Targeted new-character synthetic |32000|50%|
| YOLO OK crops |16000|25%|
| Original synthetic replay |8000|12.5%|
| Application-clean replay |4000|6.25%|
| Application-noisy replay |4000|6.25%|

Target corpus:80unique renders per new character (4560training), cycling
supported15training fonts;12independent probes per character (684images),
including four heldout font families. Glyph coverage checked before rendering;
no tofu fallback. Spaces occur internally in visible text lines. Punctuation,
case and width variants remain distinct. Training contexts derive only from
training-menu labels; generic contexts supply absent characters. Synthetic
probe is not an unseen-label/character test.

The plan guarantees every new character >=100positive label encounters;
actual planned minimum is532. Counts include all streams and are checked
against actual batch texts at completion. Every new classifier row must have
nonzero learned weights. All521required synthetic terms are also included.

All1717OKYOLOlabels now fit the vocabulary:1561train/156heldout across the
same13heldoutmenus. Every eligible training crop is encountered. Non-OK crops
never trained. Compare new156-crop holdout against initialization on the same
156crops, not directly against V38's smaller135-crop figure. Labels remain
VLM-produced, not independently verified.

Old classifier rows are copied exactly. New57rows start at zero weights/bias-12
and use separate AdamW groups at1e-3; existing head3e-6/encoder5e-7. First200
updates train only new rows, with pretrained parameters bit-identical audited.
Then full fine-tuning, always freezing147V38BNbuffers. Warmup200/cosine2000to.1,
EMA.998; microbatch4 accumulates each32-sample logical update. No BN-stat swap.

Evaluate every500updates, raw/EMA in both precisions. Report all57per-character
recalls and missing successes, fullYOLOtrain/holdout, old synthetic and short
retention. Selection balances heldout-font new-character macro recall and
YOLOholdoutCER, requiring the user-authorized17-gate policy in both precisions.
No automatic promotion. Standard exported classifier tensors are concatenated
so saved models load with the unchanged MenuRecognizer architecture.

Entry: `scripts/recognition/train_v40_new_characters.py`; preparation:
`scripts/recognition/prepare_v40.py`; launcher `scripts/recognition/run_v40.ps1`.
Output `outputs/recognition/training/v40_new_characters_2000/`. Actual coverage
in `new_character_coverage.json`; initial/final raw/EMA and selected checkpoint
are separately recorded. One completion/failure message; no constant polling.

## Completion and independent review

Completed2000updates/64000presentations in approximately29minutes. Independent
plan/checkpoint/hash audit passed: actual new-character floor532(max15388),
all57rows updated, all147BNbuffers fixed,1722oldIDs/blank0preserved and57IDs
appended. Runtime warmup assertion at200passed; AdamW counters independently
confirm2000new-row updates and1800pretrained updates. All521terms/all1561YOLO
train crops covered. V38,old vocab and current-model registry unchanged.

Heldout-font new-character recall0→53.51%raw/46.78%EMA. Original V38 re-evaluated
on the same156-crop holdout:51.82%→47.97%raw/48.91%EMA(FP16). Menu-cluster95%CI
for raw change[-7.81,+.58]pp includeszero. Common135crops41.19→39.59%; newly
covered21crops78.68→69.12%. Only2/36new-character occurrences onYOLOholdout
correctly recognized (12observedcharacters,17spaces; not all57classes tested).

Old retention losses are substantial: seen5.33→9.21%,historical10.67→18.05%,
application-noisy11.21→18.67%FP16. Raw fails11FP16/12BF16 revised gates; EMA
fails11each. `best.pt` remains initialexpandedV38 at update0,not trainedV40.
Do not promote V40 or loosen gates further based on these results.

Initialization defect discovered in review:zero new weights withconstant-12
bias do not guarantee old predictions. On weak frames all old logits can be
below-12; all new rows tie, so first newID(space) wins. Initial FP16 predictions
changed on11seen-font and45historical lines with new spaces. The original
parity-note's rounding explanation is insufficient. Weight/ID parity was true,
prediction parity was not. Future initialization should copy blank-row weights
with a negative bias offset, keeping each new logit belowblank initially.

Keep expanded dictionary, restart future learning from V38 with corrected init,
freeze old features/rows initially, increase replay/teacher consistency and use
one-new-character contexts to bound token imbalance. No next run launched.
Full independent `review.md`, `completion_verification.json` and
`review_paired_analysis.json` are in the run output directory.
