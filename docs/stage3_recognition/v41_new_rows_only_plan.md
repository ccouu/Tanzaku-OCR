# V41: corrected initialization and new-row-only pilot

User approved the revised plan after V40 review. Start from promoted V38,
not trained V40. Keep the append-only1779-character vocabulary and blank0.
Train only57new classifier weight/bias rows for all2000updates. Encoder,
projection/temporal blocks, BiLSTM, old classifier rows, BNaffine and147BN
buffers stay bit-identical. Whole pretrained model stays in eval mode.

New rows copy the blank row weights and blankbias-8. Their logits initially
equal blanklogit-8, avoiding V40's constant-12weak-frame space emissions.
Before any optimizer step, evaluate the standard fused1780-output model and
require zero new-token emissions and exact original-V38 prediction parity
in FP16 and BF16, plus revised retention eligibility. Do not train on failure.

The500target batches use2668fixed V40training renders filtered to exactly one
new character/line, at most5characters, native128/256buckets. Least-seen target
scheduling gives280–281targeted examples per newcharacter, rather than V40's
accidental Latin/space amplification from mixed-new-character contexts. Keep
the identical684new-character font probes, including4heldoutfont families,
and156YOLOholdout from13disjoint menus. No new images rendered; no holdout
images or labels added to training. VLM labels remain provisionally trusted.

| Stream | Images | Fraction |
|---|---:|---:|
| Controlled targeted new-character |16000|25%|
| YOLO train crops |16000|25%|
| Original synthetic replay |16000|25%|
| Application-clean replay |8000|12.5%|
| Application-noisy replay |8000|12.5%|

Total2000updates/64000presentations,logicalbatch32/microbatch4. Every1561YOLO
train crop included,all521required synthetic terms protected. Target counts
and all-stream actualnew-character encounters audited at completion.

AdamW newrows1e-3,warmup200,cosine2000to.1,EMA.998. Full-softmax KL consistency
against frozen old logits (weight1) on old-only labels, including valid blank
frames but excluding artificial right padding. This penalizes false new-token
mass; it is not KL after renormalizing the student over old classes, which
would not constrain new emissions. New-character labels are exempt from this
teacher loss. Ordinary supervised CTC remains for all samples.

Evaluate raw/EMA every500updates, both precisions. Same V40score and revised
17gates, no further cap relaxation. Final artifacts export concatenated rows
under standard MenuRecognizer keys. The pilot changes the package of
initialization/freezing/replay/context/distillation, not one isolated variable;
interpret comparisons accordingly. No automatic promotion or new long run.

Output `outputs/recognition/pilots/v41_new_rows_only_2000/`; launcher
`scripts/recognition/run_v41.ps1`. One-shot Herdr completion/failure; avoid
constant polling. V38 and historical runs/dictionaries stay unchanged.

## Completed and independently audited

Finished2000updates/64000presentations. Both-precision exactzero-update parity
and zero new emissions passed. Actual target counts280–281perclass; total
added-character encounters280–1855. Every57row changed from blank-rowinit.
Every pretrained parameter, oldclassifier row, BNaffine and147buffers remained
bit-identical in saved raw/EMA/selected/step states. Optimizer contains only two
newclassifier tensors. Datahash, menu/pixel disjointness, all521term coverage
and all1561YOLOtraincrop exposure verified.

Selectedbest.pt is trainedEMA1000,passesall17gates in FP16/BF16. Heldfontnewchar
recall52.05%,precision91.28%. Finalraw2000recall85.67%,precision90.99%,F1 88.25%
(293/342correct,322emissions), materially aboveV40R53.51%/P85.51%/F1 65.83%.
All57classes have at least one success among six heldfont examples/class.
FinalEMA R81.58%/P92.08%. Finalraw/EMA pass FP16gates but BF16onechar62/200
exceeds60cap bytwoedits:火→久火(newinsertion),株→抹(newsubstitution).

YOLO156holdout remains51.82%CER for selected/finalraw/finalEMA,comparedwith
identicalsourceV38. NovelYOLOchar success selected0/36/final1/36,withonly12
observedclasses and17space occurrences. Newfont recognition improvement did
not transfer to these menu crops. FullprobeCER52.09%raw reflects continuing
errors on old surrounding characters despitehighnew-character recall; it is
not a complete-line accuracy win overV40's20.51%probeCER.

V38/registryunchanged,no autopromotion,no training running. Audit/review files
in output:review.md,completion_verification.json,review_comparison.json,
short_text_differences.json. Nextfocus verified appearance-matched menu examples;
new-row bias/interpolation can separately test the syntheticrecall/shortgate
tradeoff. No new experiment launched during review.
