# V30 meaningful single-character menu terms

16valid menu-relevant single-character terms,all19 fonts plus2 Noto weight variants,2native layouts,paired clean/noisy views.1,344 term images and1,344 frequency-matched other-Kanji control images. No training,lexicon or decoder change.

| Group | Clean CER | Clean exact /672 | Noisy CER | Noisy exact /672 |
|---|---:|---:|---:|---:|
| term | 20.83% | 533 | 31.10% | 469 |
| control | 17.56% | 556 | 27.38% | 500 |

## Meaningful terms

| Term | Meaning | Clean CER | Noisy CER | Original char /standalone count | V29+V30 replay char /standalone count |
|---|---|---:|---:|---:|---:|
| 円 | yen | 80.95% | 95.24% | 185 /2 | 374 /0 |
| 茶 | tea | 0.00% | 0.00% | 843 /2 | 1556 /8 |
| 酒 | alcohol/sake | 9.52% | 26.19% | 627 /2 | 1368 /2 |
| 水 | water | 11.90% | 35.71% | 402 /2 | 849 /1 |
| 湯 | hot water | 14.29% | 23.81% | 39 /2 | 78 /2 |
| 氷 | ice | 21.43% | 47.62% | 90 /2 | 160 /4 |
| 米 | rice | 4.76% | 21.43% | 329 /2 | 660 /0 |
| 肉 | meat | 0.00% | 2.38% | 504 /2 | 1012 /4 |
| 魚 | fish | 47.62% | 38.10% | 493 /1 | 994 /6 |
| 塩 | salt | 2.38% | 7.14% | 441 /2 | 1064 /6 |
| 酢 | vinegar | 16.67% | 19.05% | 124 /2 | 226 /6 |
| 卵 | egg | 7.14% | 11.90% | 103 /2 | 156 /12 |
| 貝 | shellfish | 4.76% | 7.14% | 256 /1 | 505 /3 |
| 牛 | cattle/beef context | 33.33% | 7.14% | 271 /2 | 574 /6 |
| 豚 | pig/pork context | 59.52% | 80.95% | 70 /2 | 134 /6 |
| 鶏 | chicken | 19.05% | 73.81% | 200 /1 | 409 /1 |

## Fonts on meaningful terms

| Font setting | Clean CER | Noisy CER |
|---|---:|---:|
| AoyagiSouseki | 43.75% | 56.25% |
| NotoSansJP[wght]_default100 | 37.50% | 43.75% |
| YuseiMagic-Regular | 34.38% | 43.75% |
| HachiMaruPop-Regular | 31.25% | 28.12% |
| KouzanBrushFont | 28.12% | 43.75% |
| KaiseiTokumin-Regular | 21.88% | 28.12% |
| meiryo | 21.88% | 40.62% |
| yumin | 21.88% | 21.88% |
| YujiBoku-Regular | 18.75% | 28.12% |
| YujiMai-Regular | 18.75% | 28.12% |
| YujiSyuku-Regular | 18.75% | 28.12% |
| SawarabiGothic-Regular | 18.75% | 31.25% |
| YuGothR | 18.75% | 31.25% |
| ShipporiMincho-Bold | 15.62% | 18.75% |
| Yomogi-Regular | 15.62% | 34.38% |
| ZenKurenaido-Regular | 15.62% | 31.25% |
| UDDigiKyokashoN-R | 15.62% | 21.88% |
| NotoSerifJP[wght]_weight400 | 12.50% | 21.88% |
| KleeOne-SemiBold | 9.38% | 21.88% |
| NotoSerifJP[wght]_default200 | 9.38% | 28.12% |
| NotoSansJP[wght]_weight400 | 9.38% | 21.88% |

## Interpretation limits

This tests recognition of menu-relevant standalone characters,not semantic understanding. Controls also have meanings. Training frequency is approximately matched,while glyph shape and visual complexity differ. Textures and renderer come from the same generator family used during V29/V30 training; results are a development diagnostic. The previous random test included small kana and rare Kanji,so direct improvement against it mixes several changes.

Original occurrence counts refer to the64,000-record V27 corpus. Replay counts are the actual V29 plus V30-control presentations; they omit V26/pretraining and earlier experiments. Font/layout repetitions do not create independent term identities.

[Images and predictions](review.html) · [Complete report](report.json) · [Matching and sources](manifest.json)

## Lexical verification sources

- [Daijisen / Kotobank: 円](https://kotobank.jp/word/%E5%86%86-37934)
- [Gunma restaurant association multilingual food/menu guide](https://gunma-insyoku.or.jp/documents/itadakimasu.pdf)
- [Japan Sushi Association official textbook: tea/rice/water/vinegar](https://sushiken.or.jp/textbook/official/ch06)
- [National Diet Library authority: tea](https://id.ndl.go.jp/auth/ndlsh/%E8%8C%B6)
- [Kotobank: 卵](https://kotobank.jp/word/%E5%8D%B5-94296)
- [Kotobank: 鶏](https://kotobank.jp/word/%E9%B6%8F-460974)
- [Wiktionary Japanese: 湯](https://ja.wiktionary.org/wiki/%E6%B9%AF)
- [Wiktionary Japanese: 氷](https://ja.wiktionary.org/wiki/%E6%B0%B7)
- [Salt Industry Center: cooking seasonings](https://www.shiojigyo.com/recipe/tips/other/)
