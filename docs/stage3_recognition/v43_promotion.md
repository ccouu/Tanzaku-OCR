> Historical experiment record. Status, sources, vocabulary, geometry and gates below belong to this version; they are not current deployment instructions. See [current implementation/status](current_status.md) for V75/V76, V77 authorization and production V43. Referenced cached assets may have been retired; do not regenerate or overwrite them merely to follow this record.

# V43 promotion — 2026-11-13

User explicitly authorized promoting **V43 final raw, update2,000**, after a matched
comparison against V38. This accepts improved YOLO-menu recognition and weaker
synthetic retention for the intended application.

- Active artifact: `checkpoints/menu_recognizer_v43.pt`.
- SHA256: `41823c3d508967e50273834fabe2f27a85ea88137b7df3d2539cce735d9986a1`.
- Source: `outputs/recognition/pilots/v43_digits_prices_2000/final_raw.pt`.
- Source SHA256: `ec27f47bf0c68c6cafca08777c23eb4387947e2cf2a3286008412d9099e2f49b`.
- Registry: `src/recognition/current_model.json`; existing `load_current_recognizer` verifies checksum and loads V43.
- Vocabulary: embedded1,779characters + blank0; V38's1,722IDs unchanged,57appended. Do not substitute a V44/V45dictionary.
- Same architecture, H64RGB, replicated padding,128/256/384/512buckets,validT=ceil(width/4),greedyCTC blankpenalty0. Full-width decimal training labels normalize to ASCII.
- Preserve `checkpoints/menu_recognizer_v38.pt` unchanged as rollback, SHA256 `59d04ddcc44ec6f3388cce17c221ae4ced2b6c4fd2a39e1397752492fc10ddad`.

Matched FP16 evidence (CER lower is better):

| Metric | V38 | V43 |
|---|---:|---:|
|156-crop YOLO holdout CER|51.82%|41.61%|
|Exact YOLO lines|24/156|50/156|
|YOLO digit macro recall|42.91%|65.52%|
|Heldout-font digit macro recall|66.06%|87.72%|
|Clean11–14character512 CER|6.48%|1.24%|
|Historical crop CER|10.67%|13.88%|
|Pooled synthetic1character CER|28.00%|33.50%|
|Pooled synthetic2character CER|13.50%|15.25%|

YOLO paired crops:71improved/13worsened/72unchanged. Menu-cluster CER difference
−10.22pp,95%CI[−14.84,−5.17]. BF16 confirms the same aggregate improvement.
Holdout menus were reused during development; labels are VLM-generated, not
independently human-verified production evidence. V43 still lacks33Latinletters;
Latin and parentheses remain unresolved weaknesses.

V43 fails11existing `menu_adaptation_v2` retention gates in both precisions.
**This is a specific user-authorized promotion override, not a gate pass.**
No thresholds or future checkpoint-selection rules were relaxed. `best.pt` in
the V43 training directory remains the V42fallback; do not use it as V43final.

Promotion strips duplicate optimizer/raw/EMA recovery payloads and retains exact
source model tensors, embedded vocabulary and config. Strict loading, finite CPU
forward passes for all four buckets,147BNbuffers equal V38, active-registry loader
and unchanged source/V38 checksums were verified.

Audit record: `outputs/recognition/pilots/v43_digits_prices_2000/promotion.json`.
Previous registry: `promotion_previous_registry.json` in that directory. Matched
comparison: `outputs/recognition/evaluation/v38_vs_v43_promotion/comparison.md`.
Reproducible promotion command: `.venv-recognition/Scripts/python.exe -m scripts.recognition.promote_v43`.
