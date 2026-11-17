# Stage 2 detection

Current crop/contact-sheet default: **V5 all130** at `outputs/detection/training/banana130_finetune_v5/yolo11n_menu_obb_banana130_best.pt`,117train/13validationmenus. CombinedvalidationmAP50–95=0.929709; these are selection/development scores, not independent tests or proof of stroke-tightness. V3 remains a comparison/recall reference; root `checkpoints/yolo11n_menu_obb_best.pt` was not overwritten by the V5 candidate.

Use [crop/contact-sheet pipeline](../data/yolo_crop_canvas_pipeline.md) for detection, duplicate/overlap handling, VLM prompts and status parsing. Recognition now has separate old001–130 and new50menu collections; generated/parsedlabels remainprovisional,notautomaticallyhumanverified. Never merge collections by menu basename.

Original manualreview entrypoints remain:

```powershell
.\.venv-recognition\Scripts\python.exe -m src.detection.review.obb_review_ui
.\.venv-recognition\Scripts\python.exe -m src.detection.verify_stage2 --report outputs/reorganization/stage2_audit.json
```

Default review root `data/annotations/menu_obb/` is the original consolidated collection. Later banana batches have separate reviewed roots/snapshots; inspect the target run's training_manifest before editing, rather than assuming the default covers all130menus. Proposals are not reviewed annotations. Preserve frozen snapshots and use new preparation output directories.

[Summary and historical evidence](stage2_summary.md) · [Original review guide](stage2_review.md) · [Historical colab_v2 training](stage2_training_v2.md) · [First50 V3](banana50_latest_finetune.md) · [Second50 V4 comparison](banana51_100_finetune.md) · [Stage3 current status](../stage3_recognition/current_status.md).
