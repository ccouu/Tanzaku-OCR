# L2_v06 registry promotion — 2026-11-19

User explicitly authorized **L2_v06 epoch2 EMA** as the current recognizer.

- Checkpoint: `checkpoints/l2_v06/epoch_02.pt`, SHA-256 `d1592d6b29e81476b82235072eb2efd7c84bd4a4ce5bf9f964b6407741948349`.
- State key: **`ema`**, not RAW.
- Architecture: `h4_k41`, encoder H4, Conv2d(960,256,(4,1)), width stride4, one residual packed BiLSTM128 per direction,1780 output classes including blank0.
- Vocabulary: `checkpoints/l2_v06/vocab.json`,1779 characters, SHA-256 `40d532789de881c88edf12f6ce857120159a85f568325036b5ea89378e6e7bc9`.
- Registry: `src/recognition/current_model.json`; loader now resolves architecture, state key and verified sidecar vocabulary explicitly, with legacy mean-pooling checkpoint support retained.
- Baseline canonical-greedy FP32: val10 CER5.71%,107/129 exact; reused test5 CER7.84%,50/62 exact. This is not a fresh blind production certification.

V43 weights are unchanged and remain the rollback model. Its prior complete registry is preserved at `checkpoints/l2_v06/previous_registry_v43.json`; restoring that registry requires the V43 checkpoint present locally. V38 remains an older fallback asset. No automatic fallback occurs on missing files or checksum/schema errors.

The published neural checkpoint uses Git LFS. Vocabulary and the existing HP1010 forward/backward LM model files are companion artifacts; training images, menus, crop annotations, scraped title datasets and experiment outputs remain excluded.

**Decoder policy is separate:** the registry retains canonical greedy as its baseline. Char-LM/stateless/R1 fixes remain isolated candidates in `reviews/char_lm_20261119/`; this promotion neither applies them nor restarts the inspector. Its explicit local UI decode mode may still differ from the registry baseline. See [decoder validation](char_lm_validation.md).

Promotion provenance is in `checkpoints/l2_v06/promotion.json`. Source checkpoint bytes were verified without modification.
