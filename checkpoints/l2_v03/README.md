# L2_V03 Checkpoint (epoch_05.pt)

- **Architecture:** h4_k41 (MenuRecognizer, H=4, k=(4,1), 1-layer ResidualBiLSTM, 1779 chars + blank 0)
- **Recipe:** 21.9k/epoch (40% YOLO 3x across 225 menus w/ 2x blur+noise + 60% L2_v02 synthetic) from l2_v02 epoch_04.pt raw
- **Evaluated Metrics:** val10=0.0724 (raw, 101/129) / 0.0738 (ema, 102/129), test5=0.0972 (48/62), 1ch_c=0.090, 1ch_n=0.185, dig=0.1065, brk=0.0853, hr=0.0497, l512c=0.0392 (ema), l512n=0.0820 (ema), r4=0.060
- **Checkpoint SHA-256:** d12ce539efcaae205f76a45d94c5870594b59d9ac847eea5e55b7d8b2b4c4e3c
- **States:** raw, ema
