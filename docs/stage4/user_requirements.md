# Stage4 part1 — Minimal camera menu-translation app

Date:2026-11-23. **Implementation authorized:** shared Flutter UI, working Windows desktop demo and iOS Core ML foundation. Native iOS build/device verification requires macOS/Xcode/iPhone access; desktop results cannot establish it.

## Goal and target

Point the camera at a Japanese menu, press a shutter button, and process the captured whole menu. Show detected text-line boxes on the resulting still image; tap one to read Japanese and English. Measure whole-page latency rather than only a selected crop.

Foundation: **Flutter UI + Core ML inference on iPhone15**, Japanese-to-English and on-device. Windows runs the same Flutter interaction with a local Python reference backend; desktop speed is not iPhone speed. Pin exact iOS/Xcode versions during compatibility checks. Android comes later. Use the [core experiment Step1000 RAW2105 candidate](../stage3_recognition/preferred_checkpoint.md); it is not production-promoted.

## Main screen and interaction

1. Open to rear-camera preview with Settings and shutter buttons. Handle permission denial. Windows supports webcam preview/capture and image import for repeatable tests or absent hardware.
2. Press shutter: snapshot one complete camera stream buffer, freeze the image and start the whole-menu pipeline. No detection/OCR runs continuously on preview frames.
3. Detect/postprocess all lines, then recognize, decode and translate every eligible crop/window. Show progress and isolate per-crop failures. Draw tappable polygons on the still as results arrive.
4. Tap a box to show cached crop/Japanese/English. Tapping does not trigger first recognition.
5. Dismiss returns to the same page; Retake returns to preview. Retake/cancel/settings reprocessing must not mix page results.

**Capture-triggered full-page processing is the default.** Account for every eligible detected line, all long-line windows and failures in completion/timing. Report safety limits instead of silently skipping crops. No remote upload, account or onboarding carousel.

## Capture resolution and whole-page timing

“Same resolution as live camera” means actual stream-buffer pixel dimensions, not screen logical pixels. Save the full frame at that negotiated width/height, modulo orientation swapping, without a separate higher-resolution still-photo mode or screenshot downsampling. Preserve full field of view, not just the aspect-fill viewport. Save losslessly to a bounded app-private cache when shutter is pressed. Detector1024letterboxing and recognizerH64 resizing remain separate downstream preprocessing.

Show captured dimensions, page elapsed time, total lines/windows and completion/failure counts. Timing details cover capture/normalization, encoding/persistence, detection, postprocessing, recognition, decoding and lookup. Distinguish cold/warm runs, wall time versus summed concurrent work, and cached reprocessing versus fresh inference. Page timing ends only after all crops complete/fail or the job is cancelled.

## Immediate settings

| Setting | Default | Control and behavior |
|---|---|---|
| YOLO confidence |0.35| Slider0.10–0.90,step0.05; applies to next capture or explicit Reprocess of the still |
| CharLM alpha |0.30| Slider0.00–0.60,step0.05; re-decodes cached page logits and repeats affected lookups |
| Decoder |CharLM R1| Canonical greedy / CharLM R1; alpha ignored in greedy mode |
| Reset |—| Restore defaults |

Persist locally. Label alpha “Language-model weight”; explain that higher values favor likely words and can change correct text/numbers. Beam width5/top-k6 stay fixed. Alpha0 plain beam is not greedy. Snapshot settings per page job; use explicit Reprocess for the still instead of changing benchmark configuration mid-run. Cache misses rerun inference and are disclosed. Detector confidence is not OCR certainty.

## Popup and translation behavior

- Japanese is selected decoder output; fuzzy dictionary alternatives must not silently replace it.
- Prefer reviewed whole-phrase English; otherwise show exact component translations, preserving particles/unknown spans and prices. Do not imply a fluent complete translation when coverage is partial.
- A recognized dictionary term lacking English shows **“Await for translation”**. No dictionary hit shows **“No lexicon match”**.
- Optional compact Suggestions area may expose fuzzy/confusable alternatives; automatic correction remains off. No mandatory edit-cost tables or developer plots in the main flow.
- Lookup never changes numeric/currency spans. CharLM itself can alter numbers: retain greedy text and flag price-like disagreements rather than claiming numeric safety.
- False-positive background crops may show “No readable text”; never invent translations. Missing English is not an application error.

## Geometry and privacy

Support horizontal and native vertical Japanese, portrait/landscape device orientation, and lines exceeding512normalized pixels. Preserve RGB proportionalH64, replicate buckets, valid lengths and once-only native vertical CCW; split long lines with existing100px-overlap512 logic, not width-driven height letterboxing. Box overlays and taps must account for preview aspect-fill and camera orientation.

Device orientation is distinct from a sideways menu inside the image. No runtime VLM/cloud rotation service is part of the app. Uncertain sideways/mixed pages prompt repositioning; consider manual90-degree correction later only if phone tests require it.

Frames/crops stay local. Save the image only on explicit shutter capture, in app-private cache rather than the photo gallery. No continuous recording, remote upload, analytics images or scraping. Retain the current page and document bounded transient-data cleanup. Windows uses only its localhost Python worker; iPhone runs in-process/on-device. Handle missing camera, loading/errors, no detections, cancellation and thermal slowdown without accumulating jobs.

## Acceptance targets for the first device

These are **proposed targets**, not measured performance:

| Area | Acceptance |
|---|---|
| Interaction | Preview -> shutter -> all-line progress/results -> cached popup -> retake; desktop image import also works |
| Geometry/resolution | Saved size matches configured stream size; correct still-image polygons/hit tests under orientation/letterbox/zoom; no screenshot or high-resolution still substitution |
| Settings | Confidence and alpha affect their respective stages, persist and reset correctly |
| Speed measurement | Responsive preview; whole-page cold/warm p50/p95 for15/30/50-line pages and actual line/window counts. Establish device baseline before promising a page latency; old3Hz detection/1.5s tap targets are superseded |
| Parity | Golden Python/model/decoder/lookup fixtures pass the implementation plan's conversion gates |
| Coverage | H/V,1/2 characters, repeated digits/prices, blank/background and >512cases tested |
| Stability/privacy | Offline operation, ten-minute camera use without unbounded memory, correct background/foreground and permission handling |

If measurements miss targets, report the bottleneck and revise workload deliberately; do not silently change weights/geometry or claim ANE acceleration. Experimental model quality and mostly exposed genAI results do not justify real-world accuracy or dietary/allergen guarantees.

## Out of scope for part1

Android, AppStore release, retraining, model architecture changes, continuous live detection/OCR, translated text painted over every line, cloud LLM translation, accounts, restaurant discovery and training-feedback backend. Full-page recognition with tappable still-image boxes is in scope. Implementation and a desktop demo are now authorized; iPhone export/performance remains unverified until tested.
