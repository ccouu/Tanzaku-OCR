# Does meaningful menu terminology improve V30 single-character recognition?

## Test and result

Sixteen valid single-character terms were selected after consulting Japanese
dictionary and food/menu sources:円,茶,酒,水,湯,氷,米,肉,魚,塩,酢,卵,貝,牛,豚,鶏.
Meanings/readings can depend on context; e.g.牛 names cattle and is used in beef
contexts,while円 is the Japanese currency unit. Sources are linked in summary.md
and manifest.json.

V30 control best,update2000,was evaluated on21 font settings (all19 fonts plus
two Noto weight variants),two random native layouts and paired clean/noisy views.
Each term therefore has42 clean and42 noisy images. Images were resized
proportionally to H64 and edge-padded to128. The checkpoint and inference state
were unchanged; decoding remained raw greedy CTC at blank penalty0.

| Group | Clean CER | Noisy CER |
|---|---:|---:|
| Meaningful menu terms |20.83%|31.10%|
| Frequency-matched other Kanji |17.56%|27.38%|

There are1,344 term images and1,344 controls,2,688 total. Matching uses original
character frequency,standalone counts,and actual V29/V30 replay occurrence
counts. Original term/control frequencies differ by at most about3% per pair.
Control characters also have meanings; some are themselves menu-relevant
(e.g.粥). This is an occurrence-matched other-Kanji reference,not a valid
meaningful-versus-meaningless semantic manipulation.

Menu terms are3.27 percentage points worse on clean and3.72 worse on noisy
images. Resampling the16 matched term/control pairs,keeping all font/layout
repetitions clustered,gives95% intervals[-10.27,+16.82] and[-11.16,+17.26]
points respectively (5,000 bootstrap draws). There is no clear advantage for
the selected menu-term group over this reference. Glyph complexity,identity and
stroke geometry are not matched,so this does not isolate a semantic effect.

The earlier random-character audit reached49.07% clean and54.04% noisy CER,
but it included small kana and a different Kanji/size/margin mix. The improvement
against that audit cannot be attributed specifically to meaningfulness.

## Large differences between meaningful terms

| Term | Meaning | Clean exact /42 | Noisy exact /42 |
|---|---|---:|---:|
| 円 | yen |9|2|
| 茶 | tea |42|42|
| 肉 | meat |42|41|
| 塩 | salt |41|39|
| 米 | rice |40|33|
| 貝 | shellfish |40|39|
| 卵 | egg |39|37|
| 魚 | fish |22|26|
| 豚 | pig/pork context |17|10|
| 鶏 | chicken |34|14|

For円,26/42 clean and31/42 noisy images decode to an empty string. Other outputs
include包,丁 and月. It occurs185 times in the original V27 corpus,but only twice
as a one-character line. V29 plus V30-control replay contains374 occurrences,
with zero standalone円 presentations. Sparse standalone exposure is a plausible
hypothesis,but this diagnostic does not prove it causes the failure; earlier
training exposure is not fully counted.

For茶,all84 images across all fonts/layouts/appearances are correct. Original
frequency is843 and current replay1,556,including8 standalone presentations.
Frequency alone is insufficient:湯 has only39 original occurrences yet much
lower clean CER than円. Visual identity and crop interactions still matter.

For魚,outputs sometimes add漁 (`漁魚`),and牛 often becomes生牛. For鶏,mild
appearance changes cause confusions with鴨,鶉 and related complex glyphs.
Thus blank suppression,substitutions and extra emissions are distinct issues.

## Font ranking depends on the character set

On these meaningful terms,the weakest clean results are Aoyagi Souseki43.75%,
Noto Sans default10037.50%,Yusei Magic34.38%,Hachi Maru Pop31.25%. Sawarabi
Gothic improves to18.75%,and Noto Sans400 reaches9.38%. The latter contrasts
with its ranking on the earlier23-character audit. Font weakness is conditional
on the characters/layouts; a universal bad-font blacklist is not supported.

## What this says about terms knowledge

The experiment shows that many practical menu terms are recognized much better
than the original broad sample. However,a semantic label is not supplied to the
recognizer,and no dictionary is used during decoding. These results do not show
that the model recognizes a character because it understands its meaning.
They are consistent with differences in learned glyph exposure,visual
discrimination and training-string priors,which remain confounded here.

The most informative next diagnostic would hold the target glyph pixels fixed
and compare円 alone with円 inside a price such as500円,plus matched non-price
context. Keep target scale,margins,physical canvas,position and valid length
controlled. Target-alignment accuracy should be measured separately from
whole-string CER. That would test contextual benefit more directly,although
visual/receptive-field context still needs control. No further model training
was run.
