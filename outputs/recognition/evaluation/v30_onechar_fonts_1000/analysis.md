# Single-character font analysis

Model: V30 control best checkpoint, update2000. Exactly1,000 new single-character
images were evaluated, without training or decoder changes. All19 available
fonts were included:15 trained and4 held-out. Two additional Noto settings test
weight400 versus the application renderer's defaults.

The matched ranking uses23 shared character identities per font setting, each
with clean and textured/color/mild-noise views:966 images. Seventeen extra pairs
bring the total to1,000; they are excluded from font rankings. Native font size
and margin fractions are matched per character across settings. Fonts explicitly
support every glyph; no fallback fonts or clipped ink are used. All crops are
resized proportionally to H64,edge-padded to128 and decoded with valid lengths.

## Which fonts are problematic?

| Font | Clean CER | Textured/noisy CER | Interpretation on this probe |
|---|---:|---:|---|
| Yusei Magic |69.57%|86.96%|Worst clean and noisy results |
| Sawarabi Gothic |60.87%|69.57%|Poor even on clean images |
| Hachi Maru Pop |56.52%|65.22%|Rounded/stylized family is difficult |
| Aoyagi Souseki |52.17%|78.26%|Large appearance-related deterioration |
| Zen Kurenaido |47.83%|78.26%|Large appearance-related deterioration |
| Yuji Mai |39.13%|39.13%|Relatively stronger |
| Yuji Boku |34.78%|34.78%|Relatively stronger |
| Yuji Syuku |26.09%|34.78%|Best clean result |

Held-out Yu Gothic clean CER is56.52%,Meiryo52.17%,Yu Mincho52.17%,and
UD Digi Kyokasho39.13%. Thus both trained and held-out fonts can fail. Brush
style alone does not explain the problem: several Yuji brush fonts outperform
standard digital fonts here.

Rankings are conditional on23 characters and one layout per character. One edit
changes a font/view CER by4.35 points. Close rankings are not reliable population
estimates; font and layout interactions are not isolated by a single rendering.

## Character problems shared across fonts

Across the966 matched images,CER is51.55%; clean49.07%,textured/noisy54.04%.
There are309 substitutions,113 deletions and76 insertions. Across all1,000
images,503 are exactly correct and CER is51.60%. CER and exact error rate differ
because one image can produce multiple character edits.

- `ゅ`: wrong in all21 settings on clean images,usually `ゆ` (17 settings).
- `ぴ`: predicted as `ひ` in12 clean settings; the small handakuten is missed.
- `瑠`: predicted as `溜` in14 clean settings; similar complex glyphs are confused.
- `セ`: only2/21 clean settings exactly correct;9 predict `せ`,8 predict blank.
- `句`:13 clean settings output blank; others include `匂`,`白`,`旬`.
- `華`:13 clean settings output `華華`; the glyph is recognized but repeated.
- `季`:3 clean settings output `季節`,adding a character absent from the crop.
- `シ`: correct in every clean and noisy setting.

This is not solely a bad-font problem or blank overprediction. Small kana versus
full-size kana may lose useful relative-size evidence when an isolated crop is
normalized; that is a hypothesis requiring a paired size/context test. Diacritics,
similar-glyph discrimination and duplicate emissions also matter. The occurrence
of expanded words does not by itself establish that BiLSTM language priors caused
the error.

Several difficult Kanji also have limited original V27 training counts, e.g.
`済`33,`鈴`44,`錦`55,`瑠`89. This audit counted only the original64,000-record
corpus,not cumulative V29/V30 presentations,so those numbers are not total model
exposure. Common `い` has2,360 occurrences yet still fails often; frequency alone
does not explain all errors.

## Variable-weight check

| Setting | Clean CER | Textured/noisy CER |
|---|---:|---:|
| Noto Sans default100 |47.83%|39.13%|
| Noto Sans400 |56.52%|47.83%|
| Noto Serif default200 |52.17%|47.83%|
| Noto Serif400 |47.83%|56.52%|

The renderer weight mismatch exists,but weight400 does not consistently improve
this checkpoint's results. It should not be treated as the established main cause
of the single-character problem.

## Recommendation

Keep font coverage; do not remove difficult families based on this small probe.
Prioritize Yusei Magic,Sawarabi Gothic,Hachi Maru Pop,and appearance robustness
for Aoyagi/Zen when generating fresh variants. In parallel,test matched confused
glyph pairs (`ゅ/ゆ`,`ぴ/ひ`,`セ/せ`,`瑠/溜`) across glyph size and crop margins,
including small kana within multi-character context. Separately inspect temporal
paths for repeated glyphs such as `華`. More identical-image replay and stronger
blank penalties do not address these distinct failure modes.

No training or model promotion was performed. The checkpoint and inference state
were verified unchanged. See summary.md for all font settings and review.html
for every matched image and prediction.
