# V29 application-render mixture pilot

V27 epoch20 continuation; fresh AdamW; encoder5e-7/head3e-6, constant; normal training BatchNorm. Each32-image batch:16 original,8 clean application renders,8 textured/color mild-noise renders. No additional morphology. Original images unchanged.

| Step | Seen CER | Held-out CER | Application clean CER | Noisy CER | Original CER | Historical CER | YOLO CER | Strong CER | Retention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 4.48% | 5.27% | 28.85% | 44.42% | 4.70% | 11.27% | 96.38% | 66.96% | pass |
| 500 | 4.97% | 6.27% | 20.30% | 27.21% | 5.07% | 11.03% | 92.76% | 67.68% | pass |
| 1000 | 5.42% | 5.91% | 17.27% | 24.24% | 5.57% | 10.97% | 90.95% | 68.04% | pass |
| 1500 | 4.85% | 6.23% | 16.91% | 22.55% | 5.24% | 11.36% | 90.05% | 66.43% | pass |
| 2000 | 4.70% | 6.00% | 15.70% | 20.91% | 4.89% | 10.17% | 90.50% | 65.00% | pass |

Best eligible checkpoint: update2000; step0 remains eligible. Selection equally weights clean-font composite, application clean CER, application noisy CER. Retention caps are starting V27 +1 absolute percentage point on four historical metrics.

The generated training transcriptions exclude monitored validation text. Original training records retain the existing historical split. The application probe was designed and previously examined before this experiment; it is a development set, not independent final evidence. No matched original-only continuation arm was run. YOLO labels are provisional. All decoding is raw greedy CTC, primary blank bias0; beta0.2 is secondary in JSON.

See run_config.json, sampling.json, metrics.jsonl, report.json and step checkpoints for full audit.
