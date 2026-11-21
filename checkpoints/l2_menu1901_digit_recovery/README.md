# L2_menu1901_digit_recovery

Preferred **research** recognizer, previously referred to as “Stage A epoch 3 RAW” of `l2_1901_digits4_real40_1000`. This descriptive package name avoids confusing it with other Stage A runs.

- Weights: `epoch_03_raw.pt`, Git LFS; select **`raw`** from the package. It also contains EMA/training state; the recommended state is RAW.
- SHA-256: `7fcf4b34bb50e552b48b86dd5dad29265378f318c8e5a525a8b6261edda878bb`.
- Matching vocabulary: `vocab.json`, 1901 characters plus blank0, classifier output size 1902. Vocabulary SHA-256: `8ef5a6004cef9c187c88f2ef8a9461e35be5632b1d97b87d2c433f99558df0bc`.
- Architecture: MobileNetV4 H4, learned Conv2d(960,256,(4,1)) height collapse, temporal blocks and one residual packed BiLSTM128/direction, effective width stride4.
- Input: RGB, ImageNet normalization, proportional native H64, replicated right extension to128/256/384/512, valid sequence lengths `ceil(valid_width/4)`, orientation0 horizontal /1 native vertical rotated CCW once. Overwidth lines use the reviewed overlap512 route with100 H64-pixel overlap and label-independent inference seams.

The checkpoint and vocabulary are byte-identical copies of the historical experiment artifacts, which remain preserved. Metadata documents the exact state, ancestry and limitations; this package does not promote the model, change `src/recognition/current_model.json`, or switch the inspector.

New-menu validation CER43.58%, old val10 CER3.62%, digits CER5.73%. Parentheses9.58% and noisy1char20.50% fail original retention caps. A compatible1901 Char-LM/end-to-end evaluation is pending; do not attach the1779 vocabulary/LM unchanged. English letters `Q V j p q w x` are still absent.

After cloning, run `git lfs pull` to retrieve the binary. Training images, scraped photos, labels, databases and experiment outputs are not included. See [preferred checkpoint](../../docs/stage3_recognition/preferred_checkpoint.md) and [L2 training history](../../docs/stage3_recognition/l2_v01_v02_dataset_training_summary.md).
