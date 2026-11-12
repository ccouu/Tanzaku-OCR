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
