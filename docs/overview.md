# Pipeline overview

The project implements Japanese menu OCR with separate data preparation, text detection, recognition and planned mobile export/translation validation. The current recognizer is MobileNetV4 with temporal convolutions and a packed residual BiLSTM; a mobile ViT was an early project direction, not the trained architecture.

1. **Stage1:** lexical resources, fonts, procedural synthetic data, generated menu collections, real photographs and augmentation. Its original9000-line/1000-page report is a historical snapshot, not the size of current recognition corpora. [Data guide](stage1_data/README.md).
2. **Stage2:** YOLO11n-OBB horizontal/vertical text boundaries, manual review and refined crop/contact-sheet pipeline. Current cropper default is the V5all130 checkpoint under `outputs/detection/training/banana130_finetune_v5/`, selected on117train/13validationmenus. Detector validation is not OCR transcription accuracy or a blind test. [Detection guide](stage2_detection/README.md).
3. **Stage3:** implemented recognition training and evaluation through V76, with V77authorized. **Production remains V43raw2000**; V75raw4 is the preferred reviewed YOLO-adaptation research reference, not promoted. V60+research uses learnedConv2dheightcollapse; promotedV43 usesmeanpooling. Both have widthstride4/BiLSTM128perdirection. [Current implementation/status](stage3_recognition/current_status.md).
4. **Stage4:** mobile conversion, accelerator parity/fallback/latency and end-to-end translation remain unvalidated. Do not present desktop timings or planned budgets as achieved mobile performance.

Nativeverticalrecognition now has actualtrainingcoverage. LatestgeometryrequiresproportionalH64 andsplittingover512 with100pxoverlap,notheightletterboxing. The sharedYOLOexporterstillcontainslegacyheight-fitbehavior; correctedV76adaptersarenotuniversalintegration. See [crop guide](data/yolo_crop_canvas_pipeline.md).

TwoYOLOcropcollections retainprovisionalVLMlabels and menu-levelsplits; namespaces prevent samefilenamecollisions. The culinarydatabase is separate from checkpoint-specificvocabularies. Reviewexports do not automatically change labels. The reusabletrainer/Colabrunner refactor is under [request-changes review](../outputs/recognition/review/refactor_v68_v73_review/REVIEW.md), despite modelparity and passinghelpertests.

For execution read [AGY/Colab workflow](agy_colab_training_workflow.md). Olderimplementationproposals andtheirunmettargets remain [historical context](history/original_implementation_plan.md), not instructions to reset current models or rerun datasets.
