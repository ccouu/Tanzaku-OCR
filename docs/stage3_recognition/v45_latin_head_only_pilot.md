> Historical experiment record. Status, sources, vocabulary, geometry and gates below belong to this version; they are not current deployment instructions. See [current implementation/status](current_status.md) for V75/V76, V77 authorization and production V43. Referenced cached assets may have been retired; do not regenerate or overwrite them merely to follow this record.

# V45: Stage 1A Latin feasibility

Authorized and completed on 2026-11-13. This is a head-only feasibility experiment, not a promoted model or an authorization for Stage 1B.

Completed1,000updates/32,000presentations in about28minutes. FP16 Latin training
CER76.63%at500→74.34%at1,000; heldout macrorecall19.81%→23.92%; YOLOholdout
CER42.44%→42.34%. All oldweights/BNaffine/147buffers remained bit-identical,
all33newrows changed. Neither trainedcheckpoint met source-relative retention
selection, so no `best.pt` exists. No Stage1B or furthertraining is running.
Herdr notification delivery failed due to server/client permissions; training
and saved results completed successfully.

- Source: V43 `final_raw.pt`, with its 1,779 character IDs preserved. Append the same 33 missing Latin letters used in V44.
- Only the 33 new classifier weight/bias rows train. All pretrained weights, old classifier rows, BN affine parameters, and 147 BN buffers remain bit-identical. The feature extractor stays in evaluation mode.
- New rows copy the blank weights, with bias eight below blank. Initial prediction parity with V43 is required in FP16 and BF16.
- Maximum 1,000 updates / 32,000 presentations, logical batch 32, microbatch 4. AdamW LR 1e-3, 100-update warmup, cosine to 0.1 of peak. No EMA arm.
- Each 500-update half: 40% clean Latin, 20% original synthetic, 10% application clean, 10% application noisy, 20% clean YOLO crops. All 1,561 YOLO training crops appear in each half; 156 menu-disjoint heldout crops remain excluded.
- Latin targets reuse V44's clean renders, restricted to 1–6 characters and 128/256px buckets. Replay retains native buckets. No new 768px training or added distortion.
- Supervised CTC applies throughout. Full-softmax teacher KL, weight 1.0 / temperature 2, applies on valid frames of all synthetic and YOLO replay labels without a new Latin letter. No teacher suppression on new-letter target labels.
- Evaluate at updates 0, 500, 1,000: full Latin training CER, fixed 208-image fitting slice, known/heldout fonts, new-33 versus existing-19 recall, single/repeated/word diagnostics, false Latin emissions, digit probes, YOLO holdout, and all 17 retention gates in both precisions.
- Stop at 500 only if Latin training CER <10%, all-52 heldout macro recall >70%, and no additional retention failures relative to V43 in either precision. Otherwise finish 1,000; never automatically start Stage 1B.
- `best.pt` is selected by Latin training CER among trained checkpoints with no additional failed gates versus V43. V43 already fails absolute retention: source-relative selection is **not** an absolute gate pass or unchanged-prediction guarantee. If none qualifies, no `best.pt` is created.

Code: `scripts/recognition/train_v45_latin_head.py`, launcher `scripts/recognition/run_v45.ps1`.

Output: `outputs/recognition/pilots/v45_latin_head_only_1000/`. Preliminary four-stream GPU forward/backward smoke audit is saved separately at `outputs/recognition/pilots/v45_latin_head_only_smoke/smoke.json`; smoke updates do not seed training.

V43 was subsequently explicitly promoted by the user; see [promotion decision](v43_promotion.md).
V45 was not promoted. V38, V43/V44 source checkpoints and vocabularies are preserved.
