# Independent Japanese Lexical and Menu Review Summary (Argon / Gemini)

## 1. Executive Summary & Review Scope
- **Target File**: `outputs/recognition/curation/v31_short_terms/candidates.jsonl`
- **Total Candidates Evaluated**: **5069** (1- and 2-character short terms)
- **Review Mode**: Authorized independent review for Mobile Vision Transformer (ViT) handwritten and printed Japanese menu recognition.
- **Coverage**: **100% of candidate IDs (5069/5069) covered exactly once** in [`argon_review.jsonl`](file:///outputs/recognition/curation/v31_short_terms/argon_review.jsonl).
- **Deliverables Produced**:
  1. [`argon_review.jsonl`](file:///outputs/recognition/curation/v31_short_terms/argon_review.jsonl): Full per-row classifications (`practical_menu`, `valid_uncommon`, `irrelevant`, `uncertain`) with confidence and brief reasons.
  2. [`argon_supplements.json`](file:///outputs/recognition/curation/v31_short_terms/argon_supplements.json): Curated high-value missing/corrupted short menu terms, price counters, tax indicators, and hiragana/katakana staples.
  3. [`argon_summary.md`](file:///outputs/recognition/curation/v31_short_terms/argon_summary.md): Methodology report, contamination analysis, and classification statistics.

---

## 2. Review Methodology: Manual vs. Rule-Based Disclosures
In strict accordance with steering directives, the review applied an honest multi-tiered methodology:
1. **Exhaustive Manual Lexical Review of ALL 560 One-Character Candidates**:
   - Every single 1-character term (275 in-vocabulary + 285 out-of-vocabulary) was individually inspected against authentic Japanese culinary usage, kanji orthography, and restaurant menu cropping practicality.
   - Identified **155 high-value single-character terms** (portions like `丁`, `串`, `丼`, `切`, `本`, `杯`, `枚`; core ingredients like `塩`, `卵`, `肉`, `魚`, `葱`, `茶`, `酒`, `生`, `鍋`, `麺`).
   - Categorized historical/archaic culinary vessels, tea implements, and rare botanical/fish species as `valid_uncommon`.
   - Identified severe JMdict headword-merge artifacts (e.g., `弾`, `珠`, `球`, `璧` conflated with `玉` for okonomiyaki) and classified them as `uncertain`.
2. **Detailed Lexical Screening of Batch 1 (IDs 4 to 19571; rows 0..999)**:
   - Evaluated the first 1,000 entries with manual scrutiny for food terminology vs. extraction artifacts.
3. **Curated Identification of Modern 2-Character Menu Staples (Batches 2–5)**:
   - A curated whitelist of high-confidence modern 2-character Japanese culinary staples (`寿司`, `焼鳥`, `刺身`, `牛丼`, `豚骨`, `天ぷら`, `定食`, etc.) was cross-referenced across all remaining batches to ensure legitimate food items were preserved as `practical_menu` with `high` confidence.
4. **Systematic Rule-Based Identification of Non-Food Irrelevance**:
   - High-confidence rule triggers screened out unmistakable non-menu senses caused by English keyword collisions in the automated JMdict extractor (e.g. `"for the sake of"`, road/tree `"fork"`, wood/paper `"grain"`, printing/armor `"plate"`, toilet `"bowl"`, `"chickenpox"`, `"horseshoe crab"`).
5. **Conservative Deferral (`uncertain`, `low` confidence)**:
   - Entries that were neither confirmed modern staples nor obvious non-food false positives were **conservatively marked as `uncertain` with `low` confidence** rather than guessing or prematurely discarding them.
   - This ensures safe downstream training intersection with the independent parent shortlist without claiming exhaustive manual deliberation of all 5,069 rows.

---

## 3. Quantitative Breakdown

### Classification Counts
| Classification | Total Count | Percentage |
| :--- | :--- | :--- |
| **`practical_menu`** | **749** | 14.78% |
| **`valid_uncommon`** | **478** | 9.43% |
| **`irrelevant`** | **368** | 7.26% |
| **`uncertain`** | **3474** | 68.53% |
| **Total** | **5069** | **100.00%** |

### Confidence Breakdown
| Confidence | Count | Percentage |
| :--- | :--- | :--- |
| **`high`** | **1264** | 24.94% |
| **`medium`** | **345** | 6.81% |
| **`low`** | **3460** | 68.26% |
| **Total** | **5069** | **100.00%** |

### Out-of-Vocabulary (OOV) Candidates
- **Total OOV Candidates (`in_vocab: false`)**: **1407** (37.23% of total)
- **Status**: Every OOV entry is preserved in the review output and explicitly prefixed with `[OOV: missing char '...']` in `brief_reason`.
- **OOV Distribution**:
  - `practical_menu`: 68
  - `valid_uncommon`: 185
  - `irrelevant`: 202
  - `uncertain`: 952

---

## 4. Key Linguistic Contamination Patterns Discovered in `candidates.jsonl`

### A. English Substring Keyword Collisions
The upstream extractor searched for English keywords in JMdict glosses, causing widespread false positives:
- `"sake"` matched: `為`, `為に` ("for the sake of"), `仮に`, `後生`, `捨身`, `畜生`.
- `"fork"` matched: `岐`, `巷`, `衢` ("fork in the road"), `二股`, `三又`, `音叉` ("tuning fork"), `猫又` ("fork-tailed cat").
- `"grain"` matched: `T`, `Y`, `T目`, `Y目`, `木目`, `柾目`, `板目`, `石目`, `芝目`, `粒度`, `粒界` ("wood grain / paper grain").
- `"plate"` matched: `乾板`, `凹版`, `鋼板`, `甲鉄`, `爪甲`, `本塁`, `生還` ("home plate / armor plate / printing plate").
- `"bowl"` matched: `便器`, `一穴` ("toilet bowl"), `碁笥` ("go-stone bowl"), `伏鉢`, `火皿`.
- `"kitchen"` matched: `1K`, `3K`, `DK`, `K` (real-estate floorplans), `貝塚` ("kitchen midden").
- `"chicken"` matched: `水痘` ("chickenpox"), `端金` ("chicken feed"), `金網` ("chicken wire").
- `"nut"` matched: `キチ`, `基地`, `狂い`, `馬鹿`, `書痴` ("fanatic / car nut / book nut").

### B. JMdict Headword-Merging & Overwrite Artifacts
- **Okonomiyaki conflation**: `弾` (23881), `珠` (33665), `球` (33669), `璧` (33683) all inherited the gloss `"okonomiyaki"` because JMdict grouped them under a shared phonetic reading entry for `たま` (ball/egg). On menus, only `玉` is used.
- **Secondary Entry Overwrite (`INSERT OR REPLACE`)**:
  - `ピザ` (5909): Overwritten by slang entry meaning *"fat, ugly person"* instead of pizza.
  - `上` (8322): Overwritten by reading `かみ` (*"wife; mistress of a restaurant"*) instead of `じょう` (*deluxe/upper menu tier*).
  - `中生` (8445): Overwritten by reading `なかて` (*"mid-season crops"*) instead of `ちゅうなま` (*medium draft beer mug*).
  - `大福` (18042): Overwritten by reading `おおぶく` (*"New Year tea"*) instead of `だいふく` (*daifuku mochi*).
  - `生` (33804): Reading assigned `なまり` (*"boiled and dried bonito"*) instead of `なま` (*draft beer / raw*).
  - `本` (27584): Gloss assigned `もと` (*"soup stock"*) instead of counter `ほん` (*skewers/bottles*).
  - `杯` (28934): Reading assigned `つき` (*"shallow bowl"*) instead of counter `はい` (*drinks/bowls*).
  - `枚` (29015): Gloss assigned *"counter for fields or rice paddies"* instead of flat slices / dish counter.
  - `缶` (38016): Gloss assigned `ほとぎ` (*"fat-bodied earthenware water jar"*) instead of can.
  - `麺` (47079): Gloss assigned *"flour"* (JMdict sense 2) instead of noodles.

---

## 5. Missing Term Supplementation (`argon_supplements.json`)
To remediate critical gaps in the 5,069 candidate pool, [`argon_supplements.json`](file:///outputs/recognition/curation/v31_short_terms/argon_supplements.json) provides 62 essential terms:
- **Currency & Tax**: `円`, `税込`, `税別`, `税抜`, `時価`.
- **Portion & Customization**: `大盛`, `並盛`, `小盛`, `特盛`, `替玉`, `追飯`, `固め`, `多め`, `少め`, `抜き`.
- **Service & Ordering**: `無料`, `有料`, `売切`, `完売`, `限定`, `名物`, `人気`, `単品`, `各種`.
- **Core Ingredients / Hiragana Staples**: `水`, `お水`, `氷`, `お湯`, `炭酸`, `水割`, `湯割`, `樽生`, `ピザ`, `上`, `中生`, `大福`, `貝`, `鰤`, `鰯`, `鮎`, `そば`, `すし`, `うに`, `えび`, `かに`, `たこ`, `いか`, `ネギ`, `タレ`, `バラ`, `モモ`, `日替`, `炙り`.

---

## 6. Downstream Training Recommendations
1. **Conservative Intersection**: Filter candidate set by taking the intersection of Argon `practical_menu` (`high` confidence) with the parent shortlist.
2. **OOV Inclusion**: Train models on OOV menu items (flagged in review) by expanding the recognition vocabulary tokenizer rather than discarding them, as terms like `〆`, `奴`, `斤`, `鮃`, `鱧`, `麹` are ubiquitous in authentic restaurant signage.
3. **Synthetic Supplement Augmentation**: Ingest [`argon_supplements.json`](file:///outputs/recognition/curation/v31_short_terms/argon_supplements.json) into text generation pipelines to generate realistic menu price crops (`○○○円`, `税込`) and customization labels.
