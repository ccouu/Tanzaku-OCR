> Paths updated for the reorganized workspace. Historical artifact contents and provenance paths are retained.

# Stage 1 Summary — Japanese Menu OCR and Translation

## Scope and completion status

Implemented Stage 1 lexicon building, synthetic rendering, handwritten-menu
generation, real-photo collection, text-preserving augmentation, and integrity
checks. The base dataset audit passed in **synthetic mode**; the separate
photo-collection and augmentation audits also passed. This is a data-preparation
milestone, not a claim that the real photos are annotated or a detector is trained.

The current plan separates Stage 2 YOLO text-boundary annotation/training,
Stage 3 CTC recognizer training, and Stage 4 mobile export. None of those model
training or export stages was implemented in the work summarized here.

Herdr rejected the host connection because its server required an administrator
client. Work therefore proceeded sequentially through the authorized solo path,
without changing server permissions or elevating the orchestration environment.

## Delivered artifacts

| Artifact | Location | Result |
| --- | --- | --- |
| Culinary SQLite database | `data/processed/lexicon/culinary_menu.db` | 47,216 canonical entries |
| Vocabulary | `data/processed/synthetic_base_v1/vocab.json` | 1,722 characters; CTC blank ID 0 |
| Character frequency report | `data/processed/synthetic_base_v1/character_frequency.json` | Every vocabulary character occurs at least 10 times |
| Labeled synthetic line crops | `data/processed/synthetic_base_v1/ctc_lines/` | 9,000 grayscale crops with JSONL labels |
| Synthetic detector pages | `data/processed/synthetic_base_v1/yolo_obb/` | 1,000 pages with YOLO-OBB annotations |
| Generated handwritten menus | `data/raw/genai_seeds/menu_images/` | 100 distinct PNGs and prompt/provenance sidecars |
| Retained menu-seed API ledger | `data/raw/genai_seeds/usage_ledger.json` | 100 completed menu-image calls; zero failures in this ledger |
| Real menu photos | `data/raw/tabelog_menus/` | 400 images from 400 restaurants; 400 distinct hashes and paired sidecars |
| Real restaurant interiors | `data/raw/tabelog_interiors/` | 200 images from 200 restaurants; 200 distinct hashes and paired sidecars |
| Earlier augmentation run | `data/processed/augmentation/run_1100/` | 1,100 augmented menus, 1,100 composites, 9,000 derived CTC crops |
| Current 500-pair run | `data/processed/augmentation/run_500/` | 500 augmented menus, 500 composites, 4,050 derived CTC crops |
| Input and stride contract | `data/processed/synthetic_base_v1/stride_config.json` | `(2, 64, 384)` input; width stride 4; `T=96` |
| Packaged Stage 1 dataset | `data/processed/synthetic_base_v1/menu_dataset_10k.tar.gz` | Synthetic dataset plus generated menu seeds |
| Audit results | `data/processed/synthetic_base_v1/verification_report.json` | Status: `passed` |

The archive filename denotes the eventual 10,000-line hybrid target. Its base
supervised corpus contains **9,000 lines**, not 10,000. Full pages, whole-menu
images, restaurant backgrounds, and derived line crops are different sample
units and must not be added together as independent labeled lines.

The base archive includes the synthetic dataset and 100 generated menu seeds.
The later Tabelog collections and augmentation runs are stored separately and
have not been merged into that archive. Derived CTC crops reuse existing source
text; they do not supply additional independently annotated real-menu lines.

## Lexicon, lookup, and price parsing

Implemented [lexicon_builder.py](../../src/data/lexicon_builder.py) to combine the
licensed JMdict culinary candidate subset with deterministic variants assembled
from curated ingredients, modifiers, and cooking methods. Source and category
fields distinguish generated combinations from dictionary entries. Generated
combinations are not independently attested restaurant dishes, and broad JMdict
keyword matches need editorial review. Allergen and dietary fields remain
`unknown` rather than making unsupported recipe claims.

Lookup uses exact indexing, a character-bigram inverted index, and Levenshtein
ranking, with known OCR confusions used to break ties. The five-step price
parser handles market prices, tax metadata, portion counters, minimum/range
markers, and both positional and multiplicative Kanji numerals. Tested examples
include `八五〇円（税込）`, `一串 二百五十円`, and `時価`.

## Fonts and procedural rendering

Downloaded and verified the seven specified fonts: KouzanBrushFont,
AoyagiSouseki, Yuji Boku, Yuji Mai, Shippori Mincho Bold, Hachi Maru Pop, and
Klee One SemiBold. Publisher URLs, license notices, family names, and checksums
are retained under `data/resources/fonts/`. JMdict attribution and license information
are retained with the source and derived lexicon.

Implemented [menu_grammar.py](../../src/data/menu_grammar.py) and
[synthetic_renderer.py](../../src/data/synthetic_renderer.py) with character quotas,
glyph fallback, elastic deformation, ink bleed/erosion, perspective changes,
lighting gradients, and procedural wood, washi, and chalk backgrounds.

Vertical text is drawn upright from top to bottom, with vertical punctuation
substitution, before rotating the entire strip 90 degrees counterclockwise.
Each PNG is width 384 by height 64. The tensor helper adds a constant orientation
plane to normalized grayscale, producing `(2, 64, 384)` without requiring a model.

| Split | Lines | Horizontal | Rotated vertical |
| --- | ---: | ---: | ---: |
| Training | 7,500 | 3,750 | 3,750 |
| Validation | 1,000 | 500 | 500 |
| Synthetic test | 500 | 250 | 250 |

The 640×640 detector pages contain 3,000 horizontal-dish boxes, 3,000
vertical-dish boxes, and 3,000 price boxes, using normalized four-corner OBB labels.

## GenAI handwritten-menu collection

The initial procedural build used zero GenAI calls. Following the explicit
retry request, the full lifetime budget was reallocated to **100 handwritten-menu
image requests**, replacing the original plan's 40 item / 40 texture / 20 reserve
allocation for this collection.

Used the built-in imagegen tool in small batches, then batches of 12, with a
shared prompt template and variations across restaurant types, handwriting,
orientation, materials, and lighting. Saved every completed image into the
workspace, with its prompt, intended text, dimensions, synthetic provenance,
and SHA-256 checksum. The collection manifest is
`data/raw/genai_seeds/menu_images/collection_manifest.json`.

All 100 menu-image requests completed successfully. Budget reservations are
persisted before dispatch; failed requests would also remain charged. The
collector enforces a shared 100-call ceiling across its retained seed categories.

**These menu images are unreviewed synthetic seeds.** Prompt text is not a
verified transcription of the generated pixels. They are included in the
archive, but were not converted into supervised line labels or detector boxes
and do not count as real-photo samples.

### Retired negative-image experiment and accounting

The separate GenAI-negative background experiment was abandoned. Its
`data/raw/genai_seeds/negative_images/` folder, two dedicated generation scripts,
related test, and compiled caches were removed at the user's request. No
dedicated `src/` module depended on that feature. The base archive was checked
and contained no entries from the removed folder. Positive menu seeds and the
augmentation pipeline were retained.

The retained ledger's **100 calls describe the menu-seed collection**, not a
reconstructed all-time project/provider usage total including the retired
negative experiment. Scraping and augmentation make no LLM API calls. Running
them through an assistant can still consume tokens for orchestration, log
reading, responses, and any visual review; local Python execution itself does not.

## Real-photo gathering and review

Extended [tabelog_scraper.py](../../scripts/data/tabelog_scraper.py), backed by
[tabelog_collection.py](../../src/data/tabelog_collection.py), with two separate modes:

| Mode | Public gallery subpath | Selected collection | Review status |
| --- | --- | --- | --- |
| `interior` | `dtlphotolst/3/smp2/` | 200 photos / 200 restaurants | Visually screened as interiors; not approved automatically as text-free negatives |
| `menu` | `dtlmenu/photo/` | 400 photos / 400 restaurants | Original 100 visually screened; latest 300 deliberately left for manual review |

The scraper uses the appropriate image class for each gallery, follows provided
full-size image links, spaces requests, checks robots rules, and stops on access
blocks/rate limits. It validates image decoding, minimum dimensions, restaurant
uniqueness, and content hashes. Resumption skips existing venues and recorded
review exclusions. Each image has a JSON sidecar with source URLs, photo ID,
caption, time, dimensions, checksum, and annotation/review status.

Earlier visual review used the assistant's vision on contact sheets and selected
full-size images; it was not OCR or a separate Gemini classification call.
Obvious non-interior/non-menu candidates were quarantined with reasons and
replaced. For the latest **300 additional menus**, the user explicitly requested
**no visual rejection**. That batch was downloaded without visual screening or
quarantine. Automated file-validity and uniqueness checks still ran.

The menu collection's [additional-batch report](../../data/raw/tabelog_menus/additional_300_report.json)
confirms 300 additions, 400 total images, and the original 100 image files
unchanged. There are 20 menu contact-sheet pages and 10 interior contact-sheet
pages for review. No real menu transcriptions or text polygons are verified:
menu records retain `text=null`, `annotations=[]`, and
`transcription_verified=false`. Source-photo availability does not establish
reuse licensing; that status remains recorded in the metadata.

## Real-photo prelabel and merge hooks

Added [prelabel_real_menus.py](../../scripts/prelabel_real_menus.py) for inference
with local pretrained YOLO or PaddleOCR DBNet assets and export to X-AnyLabeling.
Added [merge_real_dataset.py](../../scripts/merge_real_dataset.py) to validate and
publish a separate hybrid dataset after review. The merge checks transcriptions,
image hashes, textures, and source-photo identities for split leakage.

Real photographs have now been collected, but the pending target is still
**1,000 reviewed real text-line crops**, not 1,000 photographs: 500 for training
and 500 for testing. The planned hybrid corpus totals 10,000 lines: 8,000 synthetic
training, 500 real training, 1,000 synthetic validation, and 500 real test.
Reviewed full-page real detector holdout data also remains pending. Pretrained
backend inference and the complete real-data merge were not exercised with
actual photographs.

## Menu augmentation and interior compositing

Implemented [menu_augmentation.py](../../src/data/menu_augmentation.py) with a reusable
[CLI](../../scripts/data/augment_menus.py) and [audit](../../scripts/data/verify_menu_augmentation.py).
Inputs are the 100 generated menus, 1,000 known synthetic pages, and the selected
Tabelog interior photos. Raw Tabelog menu photos have not been automatically
promoted into labeled augmentation/CTC inputs.

The current geometric model pinhole-projects the menu plane using:

- **Pitch:** magnitude 0–30°, randomly forward/backward (signed −30° to +30°).
- **Roll:** −10° to +10°; **yaw:** −10° to +10°.
- **Scene scale:** an actual random 0.5–1.2× factor, with interiors randomly
  selected within the sample's split. Values above 1.0 are not clamped.

Complete menu borders remain visible. The transform is applied to existing text
polygons as well as pixels. Augmented menus are saved as transparent PNGs under
`menu/`; restaurant composites are saved under `composites/`. Every image has
a paired **JSON text/provenance label**, including pose, matrices, source text
or unverified intended text, and transformed geometry. **No new YOLO `.txt`
labels or detector-class assignments are created**; those remain Stage 2 work.

The current `data/processed/augmentation/run_500/` run contains:

| Output | Count / result |
| --- | --- |
| Augmented menus and final composites | 500 of each, with paired JSON labels |
| Pair splits | 400 train / 44 validation / 56 test |
| Known transformed text regions / CTC crops | 4,050 / 4,050 |
| Generated-menu samples with unverified intended text | 50; no fabricated CTC crops |
| Distinct restaurant backgrounds used | 187 |
| Samples enlarged above 1.0× | 140 |
| Sampled scale range | 0.5013–1.1998× |

CTC crops are grayscale 384×64 images with exact existing synthetic text labels
and orientation flags; vertical crops follow the established 90° CCW protocol.
Generated whole-menu prompt text is retained without treating it as verified
pixel transcription. Menu-content groups and restaurant backgrounds are
split-disjoint **within each run**. The earlier 1,100-pair run remains available;
do not combine runs blindly or count their correlated crops as independent data.
See [augmentation documentation](menu_augmentation.md) for parameters and schemas.

## Verification results and limits

The most recent recorded full regression run passed **52 tests**. The original
menu-generation milestone passed 35 tests; subsequent scraper and augmentation
coverage expanded the suite. Checks cover price parsing, fuzzy lookup,
budget persistence and concurrency, vertical rendering, orientation tensors,
OBB geometry, split integrity, character frequency, source checksums, generated
menu metadata, archive-to-disk content equality, real-photo collection/resumption,
3D projection and scale fidelity, text preservation, paired labels, and split isolation.

Passing audit evidence is stored separately for each artifact:

- Base archive: `data/processed/synthetic_base_v1/verification_report.json`.
- Real menus: `data/raw/tabelog_menus/verification_report.json` and `additional_300_report.json`.
- Interiors: `data/raw/tabelog_interiors/verification_report.json`.
- Augmentation: `data/processed/augmentation/run_1100/verification_report.json` and `data/processed/augmentation/run_500/verification_report.json`.

These are integrity/geometry audits, not proof that unreviewed real/generated
menu text is accurate, that later batches were visually screened, or that a
trained model meets detection/OCR/mobile targets. The base audit's
`real_data_pending=true` refers to reviewed real line crops and hybrid assembly,
despite raw photos now being available.

The currently recorded audit reports exact-lookup p95 of **0.0168 ms** and
one-typo lookup p95 of **0.1791 ms**. These are warm desktop measurements over a
small query set, not mobile benchmarks or exhaustive worst-case guarantees.

The recorded procedural build duration is **68.414 seconds before archive
packaging**, approximately 79 seconds including packaging in the measured run.
This exceeded the plan's under-60-second target. The audit's passing status does
not mean that runtime target or the optional real-data workflow was completed.

Archive SHA-256:

```text
13e49c1636182b6ed25f3ba0e26f0fe1222b3216c4e33bb42252f728bad876fe
```

Re-run local verification without making new GenAI calls:

```powershell
python -m pytest -q
python -m scripts.data.verify_stage1_dataset --max-genai-calls 100
python -m scripts.data.verify_tabelog_collection --data-root data/raw/tabelog_menus --target 400
python -m scripts.data.verify_tabelog_collection --data-root data/raw/tabelog_interiors --target 200
python -m scripts.data.verify_menu_augmentation --data-root data/processed/augmentation/run_500
```

## Remaining handoff

Manual review of the latest 300 menu candidates is pending. All real-menu
transcriptions and text-boundary annotations still need preparation/review;
the 10,000-line hybrid merge and verified real holdout have not been built.
Lexicon editorial review, unknown allergen/dietary fields, background signage,
and source licensing remain recorded limitations. Any final dataset assembly
must group related source menus/restaurant photos across all included runs to
prevent leakage. Later-stage execution requires a separate instruction.
