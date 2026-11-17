> Historical experiment record. Status, sources, vocabulary, geometry and gates below belong to this version; they are not current deployment instructions. See [current implementation/status](current_status.md) for V75/V76, V77 authorization and production V43. Referenced cached assets may have been retired; do not regenerate or overwrite them merely to follow this record.

# V43 digit and price pilot

Authorized 2,000-update continuation from **V42 final raw, update 2,000**.
Production remains V38. This tests targeted digit adaptation, not vocabulary expansion.

64,000 presentations, homogeneous width-bucket batches of 32, microbatch 4:

| Stream | Share | Presentations |
|---|---:|---:|
| Balanced digit anchors and prices | 30% | 19,200 |
| YOLO training crop jitter | 25% | 16,000 |
| True 11–14-character, 512px replay | 20% | 12,800 |
| Controlled new-character replay | 10% | 6,400 |
| Original synthetic | 7.5% | 4,800 |
| Application synthetic | 7.5% | 4,800 |

Generate 6,000 training renders (600 per digit anchor), using 15 known fonts.
All training renders are visited; each digit anchors exactly 1,920 presentations.
Templates include isolated digits, double/triple repeated digits, short numbers,
yen prices and prices ending in 円. Token frequencies need not be equal because
realistic prices contain more zeros. Include both light/dark clean backgrounds,
variable safe margins, and 20% mildly textured/blurred renders. Some full-width
digit glyphs are deliberately rendered, but their labels use ASCII digits.

Normalize **only U+FF10–U+FF19 → ASCII 0–9** in every training stream.
Preserve all source files and keep original rendered text in the run-local manifest.
Do not apply broad NFKC or remap Japanese numerals. Vocabulary IDs remain unchanged.

Retain all 1,561 YOLO training crops, 521 exact curated terms, and new-character
replay. Preserve the 156-crop, 13-menu-disjoint YOLO holdout and its geometry.
Evaluate 1,000 new-seed digit renders: 500 known-font and 500 heldout-font examples
across the existing four heldout fonts. These are familiar templates, not unseen-label
or production evidence. Check train/probe pixel hashes for overlap.

Keep V42's optimizer recipe: encoder 5e-7, temporal/recurrent/head 3e-6,
appended classifier rows 3e-5; 200-update warmup, cosine to 0.1 over 2,000 updates,
EMA 0.998. Freeze all 147 BN running-stat buffers, train BN affine parameters.
Frozen V42 teacher KL is applied only to ordinary and long synthetic replay.
Use valid CTC lengths, packed BiLSTM, FP32 CTC loss, and greedy beta=0 decoding.

Evaluate raw and EMA every 500 updates in FP16/BF16. Report per-digit recall,
deletions/insertions/substitutions and confusions on digit probes and YOLO crops;
also break out isolated/repeated/price templates and full-width rendered glyphs.
Track clean YOLO holdout, long512, new-character recall and all retention gates.
Selection score is 0.4×(1−heldout-font digit macro recall) + 0.4×YOLO holdout CER
+ 0.2×heldout-font long512 CER, with both-precision retention eligibility required.
The V42 source already fails retention: an unchanged initial `best.pt` is a
**non-eligible fallback**, never evidence of a passing trained checkpoint.
No automatic promotion. Preserve final raw/EMA separately.

Output: `outputs/recognition/pilots/v43_digits_prices_2000/`.
Launcher: `scripts/recognition/run_v43.ps1`; one-shot Herdr success/failure notification,
no periodic result polling.

## Completed results and independent audit

Completed all 2,000 updates and 64,000 presentations. Final raw/EMA snapshots
confirm the exact stream counts, all 6,000 digit training images, all 1,561 YOLO
training crop identities, and 1,920 presentations per digit anchor. All 521 required
terms occur in the consumed plan, minimum once. All 147 BN buffers remain fixed.
Initial FP16/BF16 prediction parity passed. Normalized run-local labels contain
no full-width decimal digits; source labels and vocabulary IDs remain unchanged.
These are sampled presentations, not full passes through every broad-corpus record.

| FP16 metric | V42 source | V43 final raw | V43 final EMA |
|---|---:|---:|---:|
| YOLO holdout CER, 156 crops |45.46%|**41.61%**|41.92%|
| YOLO training CER |34.48%|31.22%|31.72%|
| Heldout-font digit CER |34.29%|13.73%|15.05%|
| Heldout-font digit macro recall |69.28%|87.72%|86.71%|
| YOLO holdout digit macro recall |57.81%|65.52%|65.00%|
| Seen-font CER |5.94%|7.00%|6.82%|
| Historical-crop CER |12.53%|13.88%|13.71%|
| Fresh short clean CER |28.83%|31.33%|31.00%|
| Long512 clean CER |1.72%|1.24%|1.20%|
| Long512 noisy CER |2.38%|1.94%|2.00%|

YOLO holdout exact lines rise 43 → 50/156. Deletions fall 249 → 219 and
insertions 34 → 26; substitutions change 153 → 154. The 13-menu paired bootstrap
CER change is −3.86 percentage points, 95% CI [−5.98, −1.92]. The holdout consists
of reused development menus with VLM labels, not independent human-verified truth.

Heldout-font single-digit CER improves 28.75% → 7.50%; double repeats
19.38% → 3.12%; triple repeats 45.56% → 21.11%; short numbers 23.33% → 3.33%;
円 prices 10.39% → 1.19%; ¥ prices 63.58% → 32.54%. Triple repeats and yen-prefixed
prices remain difficult. On YOLO holdout, digit 4 improves only 1/7 → 2/7;
digits 7 and 9 remain 3/6 and 2/6. These samples are too small for firm per-class
generalization. Full-width glyph style is correlated with template in this renderer,
so style-specific CER is not an isolated full-width versus ASCII comparison.

Final raw fails 11 retention gates in both precisions; final EMA fails 10 FP16 /
11 BF16. **No trained V43 checkpoint is gate eligible.** `best.pt` is unchanged
V42 at V43 update 0. Final raw is the better final menu/digit candidate and was
explicitly selected by the user as V44's source. On2026-11-13 the user subsequently
promoted final raw after the matched V38 comparison. Active artifact:
`checkpoints/menu_recognizer_v43.pt`. This specific retention override does not
change the gates. See [promotion decision](v43_promotion.md).

The vocabulary already contains ASCII digits 0–9 at IDs 3–12. V43's actual
digit-token exposure is unequal despite balanced anchors: 0 occurs 21,835 times,
versus 3,880 for 7. Parentheses occur in only 1,335/64,000 presentations (2.09%).
The alphabet audit found **33 missing Latin letters**, motivating V44.

Audit artifacts: `review.md`, `completion_verification.json`, `sampling.json`,
`initial_parity.json`, `paired_analysis.json`, and both final evaluation files in
the output directory. Production registry remains V38.
