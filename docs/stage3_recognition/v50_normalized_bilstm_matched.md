> Historical experiment record. Status, sources, vocabulary, geometry and gates below belong to this version; they are not current deployment instructions. See [current implementation/status](current_status.md) for V75/V76, V77 authorization and production V43. Referenced cached assets may have been retired; do not regenerate or overwrite them merely to follow this record.

# V50: normalized-parenthesis matched BiLSTM experiment

## Completed result — 2026-11-13

Both arms completed 2,000 updates / 64,000 presentations: control 32.77 minutes,
BiLSTM 33.67 minutes. No further training is running. V43 remains promoted.
The independent CPU audit is `scripts/recognition/review_v50.py`; its outputs are
`completion_verification.json` and `paired_analysis.json` in the V50 root.

Final raw FP16 comparison below uses pooled canonical decoding for CER and bracket
recall, literal decoding for the unchanged short-text retention gates, and the
existing literal digit evaluator for digit macro recall.

| Metric | Frozen control | BiLSTM LR 1e-5 |
|---|---:|---:|
| Tiny Latin training CER | 22.55% | 16.61% |
| Multifont held-out Latin CER | 63.12% | 64.25% |
| New 33 Latin letters: macro recall | 33.69% | 23.25% |
| Existing 19 Latin letters: macro recall | 63.66% | 56.83% |
| Clean 120-line bracket probe CER | 59.88% | 47.46% |
| Opening bracket recall on that probe | 12/120 (10.00%) | 25/120 (20.83%) |
| Closing bracket recall on that probe | 84/120 (70.00%) | 96/120 (80.00%) |
| Digit held-out macro recall | 86.72% | 87.25% |
| 156-crop YOLO holdout CER | 41.81% | 40.67% |
| YOLO exact lines | 47/156 | 48/156 |
| YOLO opening / closing bracket hits | 0/13; 6/13 | 0/13; 6/13 |
| Pooled original 1-char CER, literal | 33.00% | 37.00% |
| Pooled original 2-char CER, literal | 15.75% | 17.00% |
| Historical-crop CER, literal | 14.13% | 14.75% |

The clean bracket CER improvement is 12.43 percentage points (paired line
bootstrap 95% CI [-14.92, -10.01]); deletions fall from 308 to 226. Exact lines
are still only 6/120 versus 3/120. On the separate older 80-line bracket probe,
opening recall rises 32.50% to 41.25%, closing recall 78.75% to 82.50%.
This is evidence of useful recurrent adaptation on synthetic parentheses, but
actual YOLO bracket hits are identical between arms and openings remain zero.

The YOLO CER delta is -1.15 pp, with a 13-menu paired bootstrap CI [-2.83, +0.12].
BF16 improves only -0.21 pp (CI [-2.10, +1.22]); neither interval excludes zero.
YOLO substitutions and insertions decrease, while deletions rise 222 to 230.
This single-seed development holdout uses VLM labels; it does not establish a
reliable deployment gain. V43 already scores 41.50% canonical / 41.61% literal
CER and 50/156 exact. BiLSTM final scores 40.67% canonical / 40.77% literal with
48/156 exact, so improved aggregate edits coexist with fewer exact lines.

Latin fitting improves but held-out generalization does not: new-33 recall
falls 10.45 pp and existing-19 recall falls 6.83 pp. Held-out Latin CER worsens
1.13 pp (CI [-0.76, +3.00]); a smaller training loss alone is not a success.
The data support competing adaptation demands, not a proven architectural
inability to recognize Latin. No multi-seed inference is available.

Control `best.pt` is raw update 1,500, passing source-relative guards while
still failing 11/17 absolute literal gates. The BiLSTM arm has **no best.pt**:
every raw/EMA evaluation failed the joint guards. Final raw fails 12 gates in
both precisions; final EMA fails 11 but still exceeds the 1-char source-relative
cap. V43's 1-char baseline is 33.50%; the cap is 35.50%, versus BiLSTM 37.00%.
Final raw additionally fails `fresh_short_noisy2` (25.75% source to 27.75%).
EMA does not rescue selection. Do not relabel the final checkpoint as selected.

Audit verified identical V49 batch hashes, identical zero-update weights and all
three saved evaluation modes between arms, zero literal V43/V49 prediction
differences in FP16/BF16, all 66 selected rows changed, all 147 BN buffers plus
BN affine/CNN/projection/temporal/unselected rows bit-identical in exported raw
and EMA and both underlying training states. All ten recurrent tensors changed
only in the BiLSTM treatment. Optimizer rates and 2,000 steps match the protocol.
All 6,000 digit images, 1,561 YOLO train crops and 521 exact terms were consumed
(each term >=2); 21,635 distinct consumed image hashes and 156 held-out hashes
were verified, with menu/image-hash disjointness. 3,426 presentations had their
parenthesis label normalized; source rows/images and all saved IDs were retained.

Recommendation: keep V43 promoted and retain V50 control raw1,500 as a research
candidate only. Do not extend the combined BiLSTM recipe into a long run. If
another pilot is authorized, separate Latin and parentheses adaptation again,
and protect clean short-text alignment explicitly; require improvement on real
opening-bracket examples before claiming the synthetic gain transfers.

## Original protocol

User authorized the next matched BiLSTM experiment and explicitly requested
parenthesis normalization. To avoid confounding that normalization with recurrent
adaptation, run two independent2,000updatearms from V43finalraw:

1.`control`:normalizedparentheses,entireBiLSTMfrozen.
2.`bilstm`:identicalnormalization,onlycomplete residualBiLSTMbranch(including
outputprojection)unfrozenatLR1e-5.

EacharmreusesV49's exact savedbatchplan(byte-identicalSHA256),sampleorder,seed,
64,000presentations,66physicalrowgroups,7streams,100warmup,cosine2,000to0.1,
EMA.998 andbaseV43initializer. Latin52rowsLR1e-3;10digit+4physicalbracketrows
LR1e-5. No V49/V47/V48trainedcheckpointwarmstart. CNN,projection,orientation,
temporalblocks,BNaffine/147buffers,blank,andallunselectedclassifierrowsfrozen.
OnlytheunfrozenrecurrentmoduleenterstrainingmodeforcuDNNbackward;noLSTMdropout.
Noautomaticpromotion,additionalunfreezing,orcleanup.

## Canonical parentheses in both arms

-Maptraininglabels`（→(`and`）→)`;keepallsource-labelrows,renderedimages,geometry,
 vocabularyIDs/blank0andclassifierdimensionsunchanged. Fullwidthdigitsalready
 canonicalizedtoASCIIbyV49preparation;nootherNFKCnormalization.
-Poolwidth-pairlogitsusinglogaddexpbeforeCTCsoftmax/greedycollapse. Thisgroups
 probabilitymassratherthanmerelychangingstringsafterdecoding,andavoidsa
 spuriousduplicatewhenargmaxalternatesbetweenwidthvariantswithoutablank.
 Genuineblank-separatedrepeatedbracketsarestillpreserved.
-Bothphysicalrowsremainamongthe66trainablerows,sofullwidthvisualexamplesremain
 represented. Thetwoequivalentoutputcolumnsareomittedonlyinsidethelosstensor,
 withtemporarylocaltargetindexmapping;checkpoint/vocabularyIDsnevermove.
-Thefinite-gradienttestexposedPyTorchCTCbackwardNaNsforexplicit−infinityalias
 columns;loss-onlycompactionfixesthiswithoutdiscardingprobabilitymass.
-V43teacherKLremainsweight1,T2onvalidnonLatin-labeledreplay. Poolbothstudent
 andteacherwidth-pairmassattheactualtemperature,soKLdoesnotopposeequivalence.
 Allotherclasses/newLatinmassremaininthefullnormalizer;notold-onlyKL.

NormalizingsemanticschangesV49'sliteralprotocol,soonlythecomparisonbetweenthe
twoV50armsisolatesBiLSTMadaptation. V49isanadditionalhistoricalreference,not
thematchedcontrolforthiscombinednormalizationchange.

## Evaluation and selection

Oneforwardpasssaves3scoringmodesat0/500/1,000/1,500/2,000,raw/EMA,FP16/BF16:

-`validation`:literalrawCTCpredictionsandoriginalreferencewidths.
-`text_normalized_validation`:normalizereferenceandrawdecodedstrings,comparable
 toearlierposthocparenthesisscoring.
-`canonical_validation`:poolwidth-pairprobabilitybeforeCTCcollapseandscore
 againstcanonicalreferences;thismatchesthetrainingsemantics.

Requirezero-updateRAWpredictionparitywithV43andV49;recordinitialcanonical
effectsseparatelyasnormalization-only,notatraininggain. Botharmsstartfromsame
physicalweightsandcanonicaldecoding. Keep17literalretentiongatesandV49rawsource-
relativeYOLO/digit/historical/shorttextguardsunchanged;rankeligiblecheckpoints
bytheV49weightedCERrecipeusingcanonicalvalidation. Reportrawgatefailuresand
canonicalmetricsseparately;neverclaimoldliteralthresholdsarecanonicalbaselinegates.

Focusnew33versusexisting19Latin,opening/closingparentheses,digits/prices/repeats,
shortJapanese,156YOLOholdout,andselectedversusfinal. All6,000digitimages,1,561
YOLOtraincropsand521exacttermsmandatoryforeachfullplan;sourceimagesinclude
3,200ASCII+3,200fullwidthbracketpresentations,nowcanonicalizedinlosstargets.

Output:`outputs/recognition/pilots/v50_normalized_matched_2000/control/`and
`bilstm/`. Launcher:`scripts/recognition/run_v50.ps1`;logs:`outputs/recognition/v50.*.log`.
Sourcearchiveandrunconfigarepreservedperarm. Sevenprobability/CTCgradient/
padding/repeat/rowgroup/exporttestspassed;seven-streamGPUforward/backwardsmoke
passedforbothnormalizedfrozencontrolandunfrozenBiLSTM.

LaunchoutsideCodexrestrictedtokenusingapprovednormal-userexecutionforHerdr
completiondelivery. NoHerdrrestartorACLchange. One-shotcompletion/failurenotice,
localfallbackmessage,nopollingloop. V43remainspromoted.
