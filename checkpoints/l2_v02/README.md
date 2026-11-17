# L2_V02 Checkpoint (epoch_04.pt)

- **Architecture:** h4_k41 (MenuRecognizer, H=4, k=(4,1), 1-layer ResidualBiLSTM, 1779 chars + blank 0)
- **Recipe:** 128k synthetic/epoch (15% digits, 15% kanji numerals w/ zero, 35% random across buckets, 35% dish vocab, 50/50 H/V) from l2_v01 p2_ep08__ema
- **Evaluated Metrics:** dig=0.1101, brk=0.0853, 1ch_c=0.065 (187/200), 1ch_n=0.110, r1=0.061, r2=0.057, r4=0.039, fish_seq_100=0.0143 (95/100)
- **Checkpoint SHA-256:** ff9c019fe251a2976ad82cb68c97e9c8a0235f095e79db495ea032a6ea30c387
- **States:** raw, ema
