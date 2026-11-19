# Stage 3 compact handoff

> **2026-11-19:** user explicitly promoted **L2_v06 epoch2 EMA** to the registry. Loader now selects H4 architecture and verified vocabulary/state; V43 rollback preserved. [Promotion](l2_v06_promotion.md), [decoder validation and unapplied fixes](char_lm_validation.md). Earlier V43/current-model statements below are historical. No decoder patches or production-server restart were included.

> **Latest Implementation Plan (2026-11-17):** See [docs/implementation_plan.md](../implementation_plan.md) for the active `L2_v06` (`checkpoints/l2_v06/epoch_02.pt` EMA) baseline, HotPepper menu scraping (`src/data/hotpepper_menu_scraper.py`), LLM morpheme/slot extraction, 4-gram Char-LM CTC beam search (`val10=0.0432`, `test5=0.0721`), and Morpheme Trie lookup roadmap.

Updated through V76 review / V77 authorization. [Current status](current_status.md) is the authoritative documentation index; read it before using historical plans.

**2026-11-17 addition:** [L2 V01](l2_v01_dataset_training_summary.md) replaces former H4 V81 as the user-facing name. Current folders are `h4_k41_l2_v01_p1_rand160k` and `h4_k41_l2_v01_p2_v71mix160k`. Phase2 inherits Phase1 epoch2 RAW; both train all weights and BN statistics. All ten epoch hashes verified after rename. Internal `v81` filenames are historical and preserved. Follow the [lineage guide](model_lineage.md) rather than interpreting larger version numbers as descendants. No promotion.

## Models and pending work

- **Production remains V43 raw2000**, 1,779 characters + blank0; registry `src/recognition/current_model.json`; V38 rollback preserved. No V53–V77 checkpoint has been promoted by this work.
- **Preferred YOLO-adaptation reference: V75 epoch04 raw**, SHA `275d530bc4df0ea63ed812e3bf6b4afeda4f8a695267b3d1513673bf3eedd94b`, under `outputs/recognition/experiments/v75_v74_new_clean_yolo_fullpasses5_colab/downloads/`.
- **V76 Attempt2 epoch01 raw** SHA `215c60668bab8750394d71e9c71305fab8162321a4830685092edbc1c6e0de38` is an eligible alternative, not a clear upgrade. Native-split old-test CER ties V75 at34.09%; new-test CER34.82%→33.93%, one extra exact line. Minimum4px spacing did not improve.
- **V77 authorized:** two independent500-update arms from V75, control standard spacing, treatment6000 standard +6000 tight4–8px synthetic presentations; each arm4000 real +12000 synthetic. All2086 corrected training windows covered, no height padding; all521 terms. No reviewed completion result here; use its latest receipt/audit for state, not this static page.
- **V75B updated BN** and failed retention: new-test CER42.86% versus frozen V75's34.82%. Keep frozen statistics for the current adaptation recipe.

## Data and preprocessing

- Old data `data/processed/gemini_banana_mobile_menus_50_crops/` covers001–130; V74 splits1561train/68val/88test by117/6/7menus.
- New data `data/raw/gemini_banana_mobile_menus_train_50/yolo_crops/`:702OK,615in-vocab,492train-eligible/40menus;5val/5testmenus. Preserve full66/75OKreferences for scoring and separate in-vocab metrics. Namespace collection+menu/crop IDs.
- All521 exact standalone terms required in replay and consumed samples; priority海胆/軟骨/雲丹/馬刺 at least2H+2V/font/appearance coverage per pass. V72/V73 had a real46-term gap despite the erroneous audit count12; do not reuse that counter.
- H64 proportionally, right replicated padding to128/256/384/512. Over512: split with100H64px overlap, correct partial training labels. Test split boundaries are label-independent. Never height-letterbox to cap width in the corrected path.
- Existing shared `recognition_input()` exporter still height-letterboxes extreme lines; generic `preprocess()` can compress width. Corrected V76 uses native display crops and isolated adapters; `predict_long(...,mode='overlap512')` must be explicit. Do not infer integration from helper existence.
- Native vertical: upright glyphs top-to-bottom, vertical punctuation, thenCCW90 into strip,flag1; horizontalflag0. EffectiveWstride4, validTceil(width/4) separately passed to packed BiLSTM/loss/decode.
- Keep source labels and vocabIDs. Fullwidthdigits→ASCII and canonical parenthesis mass pooling are established; comma removal was only discussed, not authorized/applied. Eight missing characters affect87newOKcrops; details in current status.

## Code, training and review

- `src/recognition/model.py`: mean default for V43; explicitconv2d/use_bilstm for V60+research. Exact model parity passed; generic trainer/runner did not. **Do not launch trainer_v2 until the recorded P1/P2 gate is fixed and integrated tests pass.** [Review](../../outputs/recognition/review/refactor_v68_v73_review/REVIEW.md).
- Adaptation: all weights/BNaffine learn;147BNbuffers frozen inRAW+EMA. encoder2e-7/others1e-6, corrected .01matrix/0normbias decay,clip1,batch32,BF16/fp32CTC,EMA.998. Fresh changed-data continuation uses its declared actual horizon; technical resume restores fullstate.
- Existing validated experiments are reused through thin adapters while the refactor gate is unresolved. Do not copy trainer/model bodies or modify protected source merely to launch a new run. Follow user-approved plan and current skills.
- V75 review page: `outputs/recognition/review/v75_old_new_test/index.html`;163old/new test crops, corrections saved via export/import and local storage. Server URL in server.json is temporary. Review exports do not automatically change labels or preserve future blind-test status.
- VLM labels remain provisional; NOISE/TRUNCATED/UNCERTAIN excluded. Confirmed text-free crops are detector hard negatives, not nonempty CTC targets.
- GitHub `https://github.com/ccouu/Tanzaku-OCR.git`; do not infer current commit/push state from historical reports. Data/outputs/checkpoints/credentials/environments remain ignored. Documentation changes do not authorize git push, cleanup or promotion.

## Execution and reporting

Use `.venv-recognition/Scripts/python.exe`, ownedColabA100 with `colab sessions` inventory, detached worker+jobID/singleowner controller, real e2e smoke/first update ACK, dynamic AGY discovery for repair notifications and coordinator notifications only when requested. No LLM polling. All epoch files/config/vocab/corpus/render/splits/coverage/logs must download atomically with SHA256 before owned runtime stop. Full local evaluation follows; evaluator failure does not justify retraining.

Report source/eligible selection/experimental continuation/finalraw/finalEMA distinctly. Preserve all guards and original outputs, precision/cohort/geometry names and menu-level confidence intervals. The1920pairedshort and2400all-length H/V probes differ. Realmenus101–130 are mostly training after V74; report train/test subsets. Human-verified independent camera/menu testing, label review and mobile export/profiling remain outstanding.
