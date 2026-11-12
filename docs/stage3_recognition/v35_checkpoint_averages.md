# V35 checkpoint averaging result

Completed two predeclared uniform parameter averages with zero training updates:

- average4: steps 1500, 2000, 3500, 4001.
- average3: steps 2000, 3500, 4001.

All learned weights, including BN gamma/beta, were averaged with FP64 accumulation
and cast back to their original dtype. All buffers were copied from V30; all 147
BN buffers were verified equal in every member. Optimizer states were not averaged.
Original checkpoint hashes were checked unchanged. No best.pt was replaced.

| Candidate | FP16 gates | BF16 gates | FP16 pooled 1-char CER | FP16 pooled 2-char CER |
|---|---:|---:|---:|---:|
| V35 step 4001 | 17/17 | 15/17 | 29.0% | 13.25% |
| average4 | 17/17 | 15/17 | 29.5% | 13.5% |
| average3 | 17/17 | 17/17 | 27.5% | 13.5% |

Each precision uses its matching V30 baseline and the existing +1 percentage-point
gate caps. Average3 BF16 meets both short-text caps exactly: 60/200 one-char edits
and 54/400 two-char edits. There is no safety margin on these small pools.

Average3 is the preferred averaged development candidate because it clears both
precision suites, but it does not dominate the final checkpoint. FP16 historical
CER improves 10.56% to 10.03%, strong appearance 64.29% to 63.04%, and confirmation
clean 25.60% to 25.13%. Application clean worsens 9.82% to 10.18%, menu clean
28.17% to 29.83%, menu noisy 38.50% to 39.33%, and YOLO 88.24% to 89.14%.
The final checkpoint remains useful for comparison on application/menu images.

Paired transcription-cluster bootstrap intervals versus step 4001 support both
the historical improvement (FP16 -0.53pp, 95% interval [-0.80, -0.27]) and menu-clean
regression (+1.67pp, [0.16, 3.26]) without multiplicity adjustment. These repeatedly
used probes are development evidence, not independent confirmation. Mobile export
and device precision parity remain untested. No long training was launched.

Candidate: `outputs/recognition/evaluation/v35_checkpoint_averages/average3.pt`.
Full table: `outputs/recognition/evaluation/v35_checkpoint_averages/analysis.md`.
Audits: `manifest.json`, `verification.json`, `retention.json`, `paired_analysis.json`
in the same output directory. Checkpoints are inference-only and contain no
resumable optimizer state.
