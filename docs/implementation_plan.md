# Implementation Plan & Session Handoff (2026-11-17)

> **2026-11-19 validation and registry update:** L2_v06 epoch2 EMA is now explicitly selected in `src/recognition/current_model.json`, resolving the former V43 mismatch. [Promotion record](stage3_recognition/l2_v06_promotion.md). The later frozen HP1010 validation reached val10 CER0.0362/test5 CER0.0658; R1 routing protects short predictions with numeric tradeoffs. [Full summary](stage3_recognition/char_lm_validation.md). Stateless/schema/R1 patches are published as review candidates only, not applied; older pilot numbers and roadmap below are historical to this handoff.

This document captures the current state of the **Mobile Menu OCR + Translation Pipeline** (`L2_v06` recognizer, HotPepper menu scraping, LLM term/slot extraction, Char-LM CTC Beam Search, and Morpheme Trie translation lookup) so any new session can continue with zero prior conversation context.

---

## 1. Current Promoted Model & Evaluation Baseline

- **Current Promoted Checkpoint:** **`L2_v06` (`epoch_02.pt`, `ema` state)**
  - Path: `checkpoints/l2_v06/epoch_02.pt` (`54,390,324` bytes, SHA-256 `d1592d6b29e81476b82235072eb2efd7c84bd4a4ce5bf9f964b6407741948349`)
  - Vocab: `checkpoints/l2_v06/vocab.json` (`1,779` characters + `blank=0`, SHA-256 `40d532789de881c88edf12f6ce857120159a85f568325036b5ea89378e6e7bc9`)
  - Architecture: `h4_k41` (`src/recognition/model.py`, `ARCH_SPECS["h4_k41"]`, `MenuRecognizer`, `H=4`, `Conv2d(960,256,k=(4,1))`, 1-layer `ResidualBiLSTM`, `147` BN buffers frozen during adaptation).
- **Previous Checkpoints (Preserved for Comparison):**
  - `L2_v05`: `checkpoints/l2_v05/epoch_10.pt` (`raw`, SHA-256 `ef42aa1f80408a18c0deabbe95c0c12599312d5efc1e6eae786eb84dc5d4183f`)
  - `L2_v04`: `checkpoints/l2_v04/epoch_04.pt` (`raw`, SHA-256 `19bf06df3b0f8901b93a6c2554b9ff01d751211f51ec6fb14b735df3a0925de1`)
  - `L2_v03`: `checkpoints/l2_v03/epoch_05.pt` (`raw`, SHA-256 `d12ce539efcaae205f76a45d94c5870594b59d9ac847eea5e55b7d8b2b4c4e3c`)
  - `L2_v02`: `checkpoints/l2_v02/epoch_04.pt`
- **Training Lineage (`L2_v02` → `L2_v06`):**
  - `L2_v03` (5 ep): 40% YOLO 3x + 60% synthetic replay.
  - `L2_v04` (5 ep): 20% YOLO 4x (with roll/tilt + blur/noise) + 40% hard-pair confusables + 40% synthetic replay.
  - `L2_v05` (10 ep): Same `L2_v04` recipe continued for 10 epochs (`val10`: `0.0682` → `0.0543`, `108/129`).
  - `L2_v06` (4 ep, promoted `ep02__ema`): Mined `1,749` unique within-5%-probability pairs from `L2_v05` on YOLO train crops (`20%` YOLO 4x + `40%` mined pair-contrast lines + `40%` general replay).
- **Local FP32 RTX 3070 Evaluation (`L2_v06 ep02_ema` vs `L2_v05 ep10_raw`):**
  - `val10` (10 held-out YOLO menus, 129 crops): `0.0571` (`107/129`) vs `0.0543` (`108/129`)
  - `test5` (5 held-out YOLO menus, 62 crops): **`0.0784` (`50/62`)** vs `0.0909` (`49/62`)
  - `1ch_c` / `1ch_n`: `0.085` / `0.155` vs `0.090` / `0.165`
  - `dig` / `brk`: `0.1087` / `0.0734` vs `0.1094` / `0.0838`
  - `l512c` / `l512n`: `0.0368` / `0.0708` vs `0.0344` / `0.0682`

---

## 2. Side-by-Side Inspector Webtool

- **Script:** `scripts/recognition/serve_l2_v02_inspector.py`
- **Launch:** `.venv-recognition\Scripts\python.exe -u scripts\recognition\serve_l2_v02_inspector.py --port 8765`
- **Behavior:** Runs YOLO11n-OBB (`conf=0.35`) + V5 line rectification, then infers both **`L2_v06 (ep02_ema)`** and **`L2_v05 (ep10_raw)`** on every crop and displays top-5 timestep logits side by side (highlighting timesteps where top-1 tokens disagree).

---

## 3. Next-Stage Architecture: Scraping + LLM Extraction + Decoder + Trie Lookup

**Core Principle (Verified Experimentally):**
Do **not** bake lexicon frequency penalties into the OCR model weights. Keep the vision model visually balanced (`P_vision`) and apply domain priors at **decode time** via **Char-LM CTC Beam Search** + **Morpheme Trie Lookup**.
- Unigram character logit biasing degraded `val10` (`0.0571` → `0.0738`).
- Context-aware Char-LM CTC Beam Search improved **both** `val10` (`0.0571` → `0.0432`, `112/129`) and `test5` (`0.0784` → `0.0721`, `51/62`, or `0.0690` at `a=0.10`) with **zero retraining**.

### Step A: Menu Corpus Scraping (HotPepper Gourmet)
- **Why HotPepper instead of Tabelog:** `tabelog.com` gates automated HTTP requests behind a Cloudflare challenge (`403`, `Cf-Mitigated: challenge`). `www.hotpepper.jp` returns `200`, allows `food/`, `drink/`, `course/`, and `lunch/` in `robots.txt` (`RobotFileParser.can_fetch == True`), and has no Cloudflare challenge.
- **Implemented Scraper:**
  - Core module: `src/data/hotpepper_menu_scraper.py` (`RobotFileParser`, polite `1.0–1.8s` delays, `HTMLParser`).
  - CLI: `scripts/data/scrape_hotpepper_menus.py --count <N> --seed <S>`
  - Pilot run (10 random restaurants): `data/raw/hotpepper_menus/sample_10_menus.jsonl` (`1,077` items), `data/raw/hotpepper_menus/sample_10_summary.json`.
- **Next Action for Step A:**
  - Scale scraping across **100–300+ restaurants** spanning diverse genres (izakaya, washoku, sushi, ramen, yakiniku, cafe, yoshoku, etc.) via HotPepper area/genre pages or the free official Recruit Web Service API (`webservice.recruit.co.jp/doc/hotpepper/reference.html`) for discovery.
  - Deduplicate items per restaurant so large chains do not skew corpus frequencies.

### Step B: Title Filtering & LLM Morpheme / Slot Extraction
- **Pass `title` only (plus `tab` and `price` as metadata):**
  - `title` holds real menu-line text (`さつまいもクリチサラダ`, `レモンサワー`, `盛り合わせ`), whereas `description` is conversational prose (`〜ます`, `〜です`) that pollutes a menu-line LM.
  - Filter out promotional blurbs without prices (e.g., >28 chars or ending in `ます`/`です`/`可能`/`どうぞ`), strip `【...】` promo tags and decorative symbols (`♪★◆`), and strip trailing course prices.
- **LLM / Tokenizer Extraction Goal:**
  - Split compound menu titles into **meaningful culinary morphemes** (e.g., `築地より 漬物盛合わせ` → `築地` / `より` / `漬物` / `盛合わせ`, NOT awkward mid-word splits like `盛合`+`わせ`).
  - Tag each term with its **menu slot/role** (`origin_modifier`, `core_dish`, `connector`, `prep_suffix`, `size_unit`, `price`) and collect **per-restaurant term frequencies** and **slot transition probabilities**.

### Step C: 4-Gram Char-LM CTC Prefix Beam Search Decoder
- **Measured Performance on `191` Held-Out YOLO Crops (`L2_v06 ep02_ema`, pure Python CPU):**
  - Greedy baseline: `0.108 ms/crop` (`val10=0.0571 (107/129)`, `test5=0.0784 (50/62)`)
  - 3-gram (`bw=10, topk=12`): `8.658 ms/crop` (`val10=0.0446 (111/129)`, `test5=0.0721 (51/62)`)
  - **4-gram (`bw=10, topk=12`):** `8.715 ms/crop` (**`val10=0.0432 (112/129)`**, **`test5=0.0721 (51/62)`**)
  - **4-gram (`bw=5, topk=6` — mobile-friendly):** **`2.639 ms/crop` (`39.6 ms` per 15-crop menu)** with identical **`val10=0.0432 (112/129)`**, **`test5=0.0721 (51/62)`**
- **Next Action for Step C:**
  - Promote a reusable decoder module under `src/recognition/` (e.g., `char_lm_beam.py`) supporting serialized 4-gram LM tables built from `menu_lexicon.json` + cleaned HotPepper titles.
  - Note: Bidirectional `(Fwd+Bwd)` rescoring tied `FwdOnly` at `beam_width=15`; test whether `Bidir` or a lightweight **Morpheme Trie** adds gains when `beam_width` is tight (`3–5`) on mobile.

### Step D: Morpheme Trie Lookup & Translation Matching
- **Why needed:** In an audit of all `3,111` YOLO crops (`1,354` unique strings), only **`17.5%` (`237/1,354`)** matched `menu_lexicon.json` exact as full lines, because real menu lines combine dishes, modifiers, sizes, and prices (`あじたたき定食 1320円`, `お好み握り(一貫)`).
- **Design:**
  1. Build a **Morpheme Trie** over the LLM-extracted culinary morphemes (mapped to English translations).
  2. During/after CTC Beam Search, segment decoded lines (or score beam paths directly) through the Morpheme Trie, weighted by **slot/role transitions** (`origin → core → suffix → price`).
  3. **Preserve an OOV + price/number fallback path** so unknown dish names, English headers (`COFFEE & ESPRESSO`), and prices (`+200円`, `1280円`) are never forced into wrong dictionary entries.
  4. Support **visual-equivalence / low-cost substitutions** for near-identical glyphs (`ニ`↔`二`, `ッ`↔`ツ`, `ー`↔`一`, `0`↔`〇`) when matching dictionary terms.

---

## 4. Immediate Next Steps for a New Session

1. **Scale HotPepper scraping:** Run `scripts/data/scrape_hotpepper_menus.py` across 100–300+ multi-genre restaurants (expanding `DISCOVERY_PAGES` or pagination/API) and save deduplicated raw JSONL.
2. **Build LLM/morpheme extraction pipeline:** Clean `title`s, extract culinary morphemes + slot roles (`origin`, `core`, `suffix`, `price`), and export a versioned `menu_morpheme_lm.json` + translation table.
3. **Implement reusable `src/recognition/` Char-LM + Morpheme-Trie decoder:** Integrate 4-gram Char-LM CTC beam search (`bw=5, topk=6`) + Morpheme Trie segmentation/lookup with OOV fallback, and benchmark on `val10` (`129`), `test5` (`62`), and real-camera menu images.
4. **Update inspector webtool (`serve_l2_v02_inspector.py`)** to optionally toggle `Greedy` vs `4-gram Char-LM Beam + Trie` and show translated morpheme chunks alongside logits.
