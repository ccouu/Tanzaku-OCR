# V36 completed: selected raw update7000

Two full epochs completed in about36minutes:8002updates,256000presentations.
An independent saved-artifact audit verified every selected record once per
epoch,405YOLOtraining crops/pass,38heldout crops from5disjoint menus,all521exact
terms/pass,initialFP16/BF16prediction parity,and147fixedV30BNbuffers in raw/EMA
and selected/final checkpoints. All98raw BN affine tensors changed. Source and
input hashes were verified unchanged. No training is running.

Preferred development checkpoint:
`outputs/recognition/training/v36_yolo_fullpasses2/best.pt` =raw update7000.
It matches step_07000.pt model weights exactly and passes all17gates in both
FP16andBF16. BF16two-char is54/400edits,exactly at its cap. Finalraw8002and
finalEMA8002passFP16but each failsBF16two-char at55/400,one edit over cap.
Do not substitute last.pt/final_ema.pt for the retention-selected best.pt.

FP16CER (startingaverage3→selectedraw7000):

| Probe | Start | Selected |
|---|---:|---:|
| Clean training probe |3.05%|2.14%|
| Seen fonts |4.85%|4.85%|
| Held-out fonts |5.32%|5.09%|
| Historical |10.03%|9.77%|
| Application clean |10.18%|8.73%|
| Application noisy |12.48%|11.52%|
| Menu clean |29.83%|28.00%|
| Menu noisy |39.33%|35.33%|
| Confirmation clean |25.13%|23.50%|
| Confirmation noisy |29.83%|27.47%|
| Strong appearance |63.04%|61.61%|
| New YOLO train |58.91%|48.98%|
| New YOLO heldout |79.29%|74.75%|
| Legacy YOLO provisional |89.14%|87.33%|

NewYOLOtrain exact41→75/405;holdout remains3/38;legacy remains0/38.
Eight of the nine recovered holdout edits occur on menu050,one on020;three
other menus unchanged. Holdout vertical CER unchanged58.82%/12crops,while
horizontal improves86.39→80.27%/26crops. Deletions remain108/148holdout edits.
FP16pooled one-char worsens27.5→29%;two-char improves13.5→13.25%.

This is useful application/retention progress,not solved real-menu recognition.
Training/holdout gap widens,but baseline difficulty differs;no late-run YOLO
aggregate regression establishes a classic overfitting curve. Only810YOLOcrop
presentations occurred across both passes (0.3164%ofimages/pass). There is no
matched no-YOLO control,so improvements cannot be assigned causally to the crops,
schedule,orEMA. Repeated development probes and a five-menu holdout limit claims.

Do not automatically extend training. Complete the launched Stage2box review,
inspect complete/correct heldout crops separately from boundary/truncation errors,
then isolate further YOLOadaptation if needed. No production/mobile promotion.

Full four-checkpoint table,paired bootstrap intervals,per-menu results and caveats:
`outputs/recognition/training/v36_yolo_fullpasses2/review.md`.
Independent checks:`completion_verification.json`;stratified YOLO errors:
`yolo_stratified_analysis.json`;paired results:`paired_vs_initial.json`.
