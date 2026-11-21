# Preferred recognition checkpoint — research candidate

Updated **2026-11-21**. This is the easy-to-find pointer for our latest recommendation; it is **not a production registry** and is not read automatically by inference loaders.

## Preferred candidate

**L2_menu1901_digit_recovery — RAW** is the stable descriptive model name. Its historical experiment identity is `l2_1901_digits4_real40_1000`, Stage A epoch 3 RAW (`sa_ep03_raw`). The original experiment checkpoint is preserved unchanged.

| Field | Exact value |
|---|---|
| Checkpoint | `checkpoints/l2_menu1901_digit_recovery/epoch_03_raw.pt` |
| State key | **`raw`**; do not load EMA merely because it shares the checkpoint file |
| File SHA-256 | `7fcf4b34bb50e552b48b86dd5dad29265378f318c8e5a525a8b6261edda878bb` |
| Vocabulary | `checkpoints/l2_menu1901_digit_recovery/vocab.json` |
| Vocabulary SHA-256 | `8ef5a6004cef9c187c88f2ef8a9461e35be5632b1d97b87d2c433f99558df0bc` |
| Output classes | 1901 characters + blank ID 0 = **1902** |
| Architecture | H4/k4×1 learned height collapse, MobileNetV4, two temporal blocks, one residual packed BiLSTM128 per direction; width stride 4 |
| Training state | All parameters learned; 147 BN buffers frozen; source A.2 epoch 2 EMA |
| Evaluation | Canonical greedy FP32; new val 43.58%, old val10 3.62%, digits 5.73%, new122 macro recall 90.44% |
| Why preferred | Strong digit recovery and new-menu validation; Stage B did not establish an overall improvement |
| Promotion blockers | Parentheses 9.58% exceeds 7.89% cap; noisy 1-char 20.50% exceeds 18.00% cap; compatible 1901 Char-LM/end-to-end evaluation not yet validated |
| Formal selection | `selected=null`; **preferred diagnostic/research candidate, not gate-eligible or promoted** |

The newer **Stage B step 1000** final is not automatically better. Do not substitute it, Stage A epoch 4, or an EMA state without explicitly changing this recommendation.

## Other model roles

- **Promoted registry:** L2_v06 epoch 2 EMA, `checkpoints/l2_v06/epoch_02.pt`, 1779 characters; source of truth `src/recognition/current_model.json`.
- **Inspector override:** Dishlex50 epoch 5 RAW, 1779 characters, SHA `359fb52cc75b5de3ffc618d75d2b5b501afcf23eeedaab6a4a0599d9a45ea161`. This research pointer does not change the live inspector.
- **Rollback:** V43, retained through the promotion record.

**Compatibility:** the preferred checkpoint needs its 1901 vocabulary and 1902-output classifier. Do not attach the old 1779 model vocabulary/LM unchanged. Seven English letters remain outside even this expanded vocabulary: `Q V j p q w x`.

See [complete L2 datasets and checkpoint history](l2_v01_v02_dataset_training_summary.md), and the latest run's `outputs/recognition/experiments/l2_1901_digits4_real40_1000/{HANDOVER.md,evaluation/evaluation_summary.json}`. File hash verified locally on 2026-11-21; holdout labels are provisional and character inventory was label-informed across splits. No automatic promotion is authorized by this document.
