# Pipeline overview

Stage 1 prepares resources and data. Stage 2 detects horizontal and vertical text boundaries. Stage 3 will recognize text; Stage 4 will export the validated pipeline for mobile use.

Stage 1 delivered 9,000 synthetic recognition lines and 1,000 three-class synthetic detector pages, alongside generated menus, real photographs and separate augmentation runs. Real recognition transcriptions and the hybrid dataset remain pending. [Stage 1 evidence](stage1_data/stage1_summary.md).

Stage 2 delivered proposals, explicit review tools, 1,213 reviewed pages, frozen datasets, and a trained YOLO11n-OBB checkpoint. The current training snapshot has 862 train / 351 validation images with source-connected splits and two classes. Validation mAP50 is 0.8427 and mAP50–95 is 0.6521; the planned gates remain unmet. Fresh-photo predictions await inspection. [Stage 2 evidence](stage2_detection/stage2_summary.md).

The three-class synthetic/proposal format includes `price_tag`; the current detector trains `dish_horizontal=0` and `dish_vertical=1`. Existing price conversions and historical annotations are retained. Detector review is independent of verified text transcription.

Stage3's active V27 recognizer combines MobileNetV4, temporal convolutions and
a packed residual BiLSTM. Its20-epoch full-pass run reached4.48% seen-font and
5.27% held-out-font validation CER, but failed provisional GenAI-menu retention.
Earlier experiments are archived. Reviewed real transcriptions, independent
real-menu evaluation and mobile validation remain pending.
[Stage 3 results](stage3_recognition/stage3_summary.md).

Use the [workflow](../WORKFLOW.md), [Stage 1 guide](stage1_data/README.md), and [Stage 2 guide](stage2_detection/README.md) for current entrypoints. The [original implementation plan](history/original_implementation_plan.md) is historical context.
