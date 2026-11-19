# L2_V06 Checkpoint (epoch_02.pt, ema)

- **Status:** Promoted (`ema` state)
- **Architecture:** `h4_k41` (`MenuRecognizer`, H=4, k=(4,1), 1-layer ResidualBiLSTM, 1779 chars + blank 0)
- **Recipe:** 58,400 samples/epoch (20% YOLO 4x + 40% mined within-5% probability pair lines + 40% general synthetic replay) from `l2_v05` `epoch_10.pt` (`raw`)
- **Evaluated Metrics (`ep02__ema`):** `val10=0.0571` (`107/129`), `test5=0.0784` (`50/62`), `1ch_c=0.085`, `1ch_n=0.155`, `dig=0.1087`, `brk=0.0734`, `hr=0.0511`, `l512c=0.0368`, `l512n=0.0708`, `r1=0.089`, `r2=0.079`, `r4=0.064`
- **Checkpoint SHA-256:** `d1592d6b29e81476b82235072eb2efd7c84bd4a4ce5bf9f964b6407741948349`
- **States:** `raw`, `ema`
