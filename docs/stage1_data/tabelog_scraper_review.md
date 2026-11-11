> Paths updated for the reorganized workspace. Historical artifact contents and provenance paths are retained.

# Tabelog Photo Scraper Review

The scraper now supports both restaurant interiors and real menu photographs.
Use `--photo-kind interior` (default) or `--photo-kind menu`; each mode selects
its own gallery path, photo-image HTML class, output directory and metadata type.
Collections cannot be mixed in the same output directory.

## Issues corrected

- The original download function swallowed failures while its caller incremented
  the success counter. Counts now increase only after a decoded image is saved.
- The original collected up to five images per restaurant. The default now
  collects ten images from ten distinct restaurant IDs, one image per venue.
- Removed the Playwright/browser dependency and import-time directory creation.
  The public English pages expose the required information in server-rendered
  HTML and were accessible during this run. The original Japanese listing
  returned an HTTP 403 challenge.
- Replaced broad image selectors with interior-gallery photo extraction,
  excluding maps, logos, and other page images. Actual full-size anchor URLs
  take precedence over thumbnails and lazy-loading attributes; no CDN resize
  paths are guessed.
- Added relative pagination handling, bounded crawling, a two-second default
  request interval, robots checks, explicit stops on blocked/rate-limited access,
  image decoding/resolution checks, SHA-256 deduplication, and safe resumption.
- Added JSON provenance, a collection manifest, and a contact sheet. The output
  path is anchored to the repository rather than the invoking shell directory.

The entry point remains `scripts/data/tabelog_scraper.py`; implementation lives
in `src/data/tabelog_collection.py`.

## Collection result

Saved **10 valid, distinct photos from 10 different restaurants** under
`data/raw/tabelog_interiors/`. Each selected photo has a JSON sidecar containing
its restaurant ID/name, restaurant page, interior gallery, original image URL,
photo ID, caption, collection time, dimensions, and checksum.

Visual review found a food close-up incorrectly filed in one upstream interior
gallery. That venue (`13245796`) was excluded, its candidate retained under
`rejected/`, and a different restaurant supplied the replacement interior.
Exclusions persist across resumed collections. Diagnostic page snapshots are
kept separately under `diagnostics/`.

Three focused tests passed, covering parser behavior, rejected/duplicate
downloads, successful counting, one image per restaurant, and resumption.
The live collection also passed ten-restaurant and ten-checksum uniqueness checks.

## Usage and review status

```powershell
python -m scripts.data.tabelog_scraper --target 10 --delay 2
python -m scripts.data.tabelog_scraper --target 200 --delay 2
python -m scripts.data.tabelog_scraper --photo-kind menu --target 100 --delay 2
python -m scripts.data.verify_tabelog_collection --target 200
python -m scripts.data.verify_tabelog_collection --data-root data/raw/tabelog_menus --target 100
python -m pytest tests/test_tabelog_scraper.py -q
```

Direct invocation with `python scripts/data/tabelog_scraper.py` also works.
Use `--exclude-restaurant RESTAURANT_ID` to skip a venue rejected during review;
quarantine any existing selected photo and remove its manifest entry first.

Review `contact_sheet.jpg` and `manifest.json`. These are real restaurant-interior
candidates, not automatically approved text-free negatives: some images contain
signage, lantern lettering, or tabletop materials. No empty YOLO labels were
created and no files were merged into the Stage 1 training archive. Source
photo reuse licensing is recorded as not established.

## Larger collections

The scraper accepts up to 1,000 requested photos and scales default page/venue
limits to the target. A 200-photo request extends the existing ten-photo
collection to 200 total photos, with one per restaurant. It resumes by validating
existing checksums and skips known restaurant IDs and review exclusions.

Contact sheets contain at most 20 images per page, named
`contact_sheet_001.jpg`, `contact_sheet_002.jpg`, and so on. Their index is
`contact_sheets.json`; `contact_sheet.jpg` remains an alias of the first page.
The offline collection audit checks counts, unique restaurant IDs and hashes,
dimensions, paired metadata, source URLs and contact-sheet pages.

The expanded run retained 200 selected interior photos from 200 distinct
restaurants. Visual review quarantined upstream advertising, exterior shots,
icons, food/drink close-ups, and other non-room candidates; replacements came
from different restaurants. Rejected files and review reasons remain under
`rejected/` and are excluded from the collection manifest and augmentation inputs.
The collection has ten contact-sheet pages.

To persist future review exclusions before resuming:

```powershell
python -m scripts.data.review_tabelog_collection --rejections-json path/to/restaurant_id_to_reason.json
python -m scripts.data.tabelog_scraper --target 200 --delay 2
```

The unused GenAI-negative image folder and its dedicated generation scripts,
test, and compiled caches were removed. No dedicated source module under `src/`
depended on that workflow. The positive generated-menu collection, its usage
ledger, and the text-preserving augmentation pipeline were retained. The Stage 1
archive was checked and contains no entries from the removed negative folder.

## Real menu-photo collection

Menu mode visits each restaurant's `dtlmenu/photo/` page, such as
`https://tabelog.com/en/aichi/A2306/A230601/23075911/dtlmenu/photo/`.
It extracts `rstdtl-photo-list__img` images and uses their provided full-size
anchor URLs instead of the smaller thumbnails. Menu captions are preferred
over generic photo captions. Restaurants with no available menu photos are
skipped; only successfully decoded, unique images count toward the target.

The default output is `data/raw/tabelog_menus/`:

- `menu_<restaurant-id>.jpg` (or the decoded PNG/WebP extension).
- A paired JSON file with restaurant/gallery/image URLs, photo ID, caption,
  collection time, dimensions, checksum and review status.
- `manifest.json`, paginated contact sheets and `contact_sheets.json`.
- `verification_report.json` after running the offline audit.

Real menu images are initially untranscribed: `text=null`, `annotations=[]`,
and `transcription_verified=false`. They are not automatically assigned CTC
transcriptions or YOLO text-boundary labels. Review the actual pixels and
annotate them before adding them to supervised training splits. They are kept
separately from the restaurant-background collection and generated menu seeds.

The completed menu collection contains 100 images from 100 distinct restaurant
IDs, with 100 unique file checksums and five contact-sheet pages. Visual review
quarantined ten non-menu or heavily obscured candidates and replaced them with
menu photos from other venues. Transcriptions remain unverified. Review exclusions are independent
between the interior and menu datasets, so a venue rejected for its interior
photo can still supply a valid menu photo.
