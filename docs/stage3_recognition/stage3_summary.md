# Stage 3 — recognition implementation and experiment summary

Current snapshot: through V76 reviewed results and V77 authorization. Read [current implementation/status](current_status.md) for exact paths, hashes, cohorts and caveats, or [compact handoff](HANDOFF.md) to continue work. Historical per-version reports remain unchanged apart from scope notices.

**2026-11-17 addition:** the user renamed former H4 V81 to **L2 V01**, an independent random-start H4 lineage. Two-phase glyph-first/menu training completed (2+8 epochs, 1.6M presentations); all ten epoch hashes and the Phase1→Phase2 source edge were verified after rename. See [L2 V01 data/training/results](l2_v01_dataset_training_summary.md) and [family lineage](model_lineage.md). This addition supersedes no production checkpoint; the original-version sequence below remains a scoped historical summary.

## Deployment and architecture

V43 raw2000 remains the explicitly promoted model in `src/recognition/current_model.json`; V38 is rollback. Research advances do not automatically change production. The recognizer is MobileNetV4 Conv Small with orientation conditioning, residual temporal convolutions and a packed residual BiLSTM128 per direction, CTC output, RGB H64 and effective width stride4. V43 uses height mean + Conv1d. V60 introduced learnedConv2d(960,256,(2,1),stride=(2,1)) height collapse without changing temporal resolution. Mobile accelerator behavior remains unmeasured.

## Experiment sequence

| Sequence | Purpose and reviewed outcome |
|---|---|
| V27–V43 | Font/bucket/margin studies, selective/full fine-tuning, frozenBN, coverage and price experiments. V38 and then V43 were explicitly promoted for menu-domain gains with acknowledged retention trade-offs. |
| V44–V52 | Latin/parentheses/extra-long, selective classifier rows, normalized decoding, resolution and beam/lexicon diagnostics. No later promotion; source-fallback best.pt is distinct from trained candidates. |
| V53–V59 | Random-start mean-pooling recognizer, then short/digit/price/parenthesis/background/curated-menu adaptation. Improved synthetic performance did not establish camera or vertical generalization. [Dataset/weight settings](v53_v59_dataset_summary.md). |
| V60–V65 | Random-start learned height collapse; V61short/digit pilot → V62broad replay → V63price1–4/kanji → V65scale/margin/menu backgrounds. V64 matched recurrent-only/full fine-tuning separately. V65 startedV63, notV64. All lacked native-vertical training. [Dataset/weight settings](v60_v75_dataset_training_summary.md). |
| V66–V67 | Vertical adaptation of horizontal-specialized V65; significant horizontal drift. V67teacherKL selected an early retention pass but verticalCER stayed~90%. |
| V68 | Fresh randomV60 architecture,196k50/50nativeH/V,15fullpasses,allBNstatistics train. Epoch13EMAselected, syntheticH/V6.12%/7.46%; major vertical breakthrough. |
| V69–V71 | Sequential V61/V62/V63 recipes with vertical coverage; V71recovered digits/prices, but no retention-eligible checkpoint. V69actual63944presentations rather than64000; report deviation. |
| V72–V73 | Matched short-menu arms then five scale/margin/menu-background passes. V72fullfine-tune raw1500eligible; V73rawepoch4continuationnoteligible. RealmenuCER21.07%, but exact-term audit found475/521. |
| V74 | TwoYOLO25%passes,1561real+4683replay/epoch. All521restored; fourpriorityterms H/V coverage. Epoch2raweligible; heldoutCER gains modest. |
| V75 / V75B | Addnew50clean menus; five passes on2053real+6159replay. V75epoch4raweligible,newtestCER40.18%→34.82%. UpdatingBN inV75B worsenednewtest to42.86%; noeligiblecandidate. |
| V76 | Fivepasses fromV75 with syntheticgaps4..Gmax. Attempt1halted because33oldrealcropswereheight-letterboxed. Attempt2uses2086nativeH64windows+6258synthetic/epoch,1315updates. Epoch1raweligible; matchednativeoldtesttiesV75, newtest improves byoneexactline; min4spacingdiagnostic regresses. |
| V77 | Authorizedmatched500-update control/tight4–8px mixture arms fromV75; no reviewed results inthissnapshot. Noautomaticlongcontinuation. |
| **L2 V01 (formerly H4 V81)** | Independent random-start H4: 160k glyph corpus, two epochs → 160k 30%glyph/70%menu corpus, eight epochs; 50/50 native H/V throughout. All weights/BN statistics update. Phase2 epoch8 RAW clean1char/2char CER8%/5%; epoch6 RAW is a stronger camera candidate (new60 CER22.20% versus epoch8 25.42%). No YOLO training or promotion. |

## Current conclusions

V75epoch4raw is the preferred reviewed research source for camera/YOLO adaptation, not a production release. Keep native vertical examples and meaningful one/two-character labels in replay. Training only horizontal images made flag1 and sideways glyphs OOD; changing labels/flags alone does not create native vertical data.

Width-cap preprocessing was not universally split. Current required training/inference geometry is proportionalH64, rightpaddingonly, windows<=512 with100pxoverlap where needed, and correctly aligned partial labels. Shared crop exports retain legacy letterboxing, so data-path audits remain mandatory. V76trainedwindows and label-independent inference windows have separate contracts.

Sparse short terms, tight long-line spacing, repeated price digits, ambiguous VLM labels and OOV punctuation remain unresolved. Required521term coverage is exactfull-line and checked aftersampling, not a vocabulary-size or substring promise. V75's 32-image priority-font probe is a small known-label transfer diagnostic, not a general Japanese vocabulary test.

FrozenBN protects running statistics, not every learned feature. EMA must copy protected tensors exactly; audits cover raw andEMA. V75Bsupports frozenstatistics for this recipe but is not a universal result for everydataset/model.

The reusable Strategyrefactor's model is compatible; its generictrainer/Colabrunner are still blocked by the [request-changes review](../../outputs/recognition/review/refactor_v68_v73_review/REVIEW.md). Helpertests did not test module import, fullresume or mockedremote lifecycle. Preserve historicalactualoptimizer versus intendedpolicy differences. No additional code/refactor/modelchanges are made by this documentation audit.

## Reading map and evidence

- [Current status, metrics and preprocessing caveats](current_status.md)
- [V76–V77 crop geometry and spacing summary](v76_v77_dataset_training_summary.md)
- [Historical experiment index](README.md#historical-experiment-records)
- [AGY/Colab operating guide](../agy_colab_training_workflow.md)
- [Crop/export guide](../data/yolo_crop_canvas_pipeline.md)

Full run reports reside under `outputs/recognition/experiments/`; V76post_completion_audit distinguishes native versus legacygeometry. Reviewcorrections remain separate from training labels. Scores use development cohorts repeatedlyinspected/selected; provisionalVLMlabels and fewverticalmenus limit generalization claims. Independenthumanverifiedmenu testing, finaldictionary/translationvalidation and mobileexport/profiling remain pending.
