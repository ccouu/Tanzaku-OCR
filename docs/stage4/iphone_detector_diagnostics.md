# iPhone detector diagnostics — build 1.0.1+2

This build investigates “processing finishes but no boxes appear”. No model weights, vocabulary, decoder or translation assets are changed. It retains the validated FP32 Core ML models.

## On the phone

1. Install the new unsigned IPA through the same Windows signing tool used for the first build. Confirm app version **1.0.1**, build **2** in the exported native diagnostics.
2. Tap the **bug/diagnostics** button in the camera screen's top bar. It is available before taking a photo and after capture.
3. Leave confidence at **0.35** and CPU-only **off**. Tap **Run Bundled menu-101 Test**. The Python/Mac reference detects **15 boxes**. A matching count alone is not proof of coordinate or preprocessing parity; inspect the source overlay and actual detector input too.
4. Tap **Copy JSON Report** or **Save JSON**. Return the report and any visible error. Preview images are omitted from exports by default; enable them only if you want to include the photo. Nothing is uploaded automatically.
5. If the bundled test fails or finds no boxes, repeat once with **CPU-only on**. This separates native compute-backend behavior from preprocessing/postprocessing issues.
6. Capture/import your menu, reopen diagnostics, and choose **Run Current Capture**. Compare it with the bundled result. A lower-confidence run is optional evidence, not an automatic production threshold change.

## What the report measures

- Stage and exact error; app/build/OS version and requested compute units.
- Source dimensions and UIImage orientation, actual 1024 detector-input PNG, input RGB means and min/max, letterbox geometry.
- Core ML output dimensions, strides, type, finite/nonfinite values, maximum class confidence, and pre-NMS candidate counts at 0.05/0.15/0.35.
- Dart post-NMS count and polygons, effective threshold, native prediction timing and complete diagnostic wall time.

Detection-only diagnostics do not load the recognizer, CharLM, lexicon or translation DB. A native/model/byte-decoding failure is shown explicitly, rather than interpreted as a successful empty menu. The ordinary whole-page screen now also shows terminal `run.error` prominently.

## Changes under test

MethodChannel Float32 decoding now respects arbitrary Uint8List offsets, including unaligned byte views. Core ML output serialization and feature-prefix slicing respect MLMultiArray strides. The detector loads separately from the OCR models. The new screen exposes the existing native image preprocessing without claiming it has matched the Python implementation on a physical phone.

Mocked Flutter tests establish UI and transfer behavior only. Actual iPhone detection, native preprocessing, camera orientation, overlay alignment and latency must be confirmed from device reports. Do not retrain or relax model-parity gates based only on an empty overlay.
