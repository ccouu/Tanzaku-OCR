# Stage 4 GitHub Actions macOS Core ML Parity & iOS IPA Deployment Guide

**User-selected route,2026-11-23:** build an **unsigned release device IPA on GitHub**, then sign/install it on Windows using a free Apple ID in Sideloadly (AltStore Classic alternative). Follow [Windows signing instructions](windows_free_apple_id_installation.md). This supersedes signed-in-CI as the default for the personal trial: no Apple signing secrets or paid-team profile is needed in CI. The unsigned IPA still must be signed locally before installation; native inference/resource/parity gates remain mandatory. Older signed-CI sections below describe an optional alternative, not the chosen route.

Date: 2026-11-23 *(local investigation & workflow draft — zero git push, zero CI trigger, zero public asset release performed)*

## 1. Repository State & Visibility (Read-Only Inspection)
- **Repository:** `https://github.com/ccouu/Tanzaku-OCR`
- **Visibility:** **Public** (`private: false`, default branch `main`, remote size `~102 MB`, last pushed `2026-11-21T06:02:11Z`, `HEAD = cd4c6af`).
- **Local Working Tree Gap:** All Stage 3 2,105 short-recovery results, Lexicon V3 updates, Stage 4 Flutter desktop demo, Core ML export tooling, Colab `.mlpackage` artifacts, and golden fixtures exist **locally only** and have NOT been pushed to `origin/main`.
- **Public-Repo Safety Rule:** Because `ccouu/Tanzaku-OCR` is public, **do not** push restaurant photo datasets, unvetted model weights, or signing credentials. Use `workflow_dispatch` only (never `pull_request` with signing secrets), least-privilege `permissions: { contents: read }`, and explicit checksummed artifact delivery URLs.

## 2. Official GitHub-Hosted macOS Runners & Billing Boundaries
- **Recommended Pinned Runner:** Pin an explicit Apple Silicon macOS runner label (e.g., `macos-15`, `arm64`) rather than floating `macos-latest`, and log `uname -a`, `sw_vers`, and `xcodebuild -version` at job start for reproducibility.
- **Account- and Plan-Dependent Billing (No Hardcoded Rate Guesses):**
  - Standard GitHub-hosted runners are generally free for **public** repositories, whereas **private** repositories consume included monthly compute allowances at higher macOS minute multipliers and incur per-minute overages once exhausted.
  - **Artifact storage** and **Actions cache storage** are billed/quota-limited separately from compute minutes and vary by GitHub account plan (Free, Pro, Team, Enterprise).
  - Verify current runner specs, minute multipliers, and storage quotas directly on GitHub's official billing and runner reference pages before running private or high-volume CI jobs.
- **Cost & Storage Safeguards:** Always set `timeout-minutes` (`45` max), `concurrency: { cancel-in-progress: true }`, and short artifact retention (`retention-days: 1` for intermediate handoff bundles, `7` for parity reports).

## 3. Explicit Asset Delivery & Validated Bundle Handoff (No Assumed Local `outputs/`)
Because `origin/main` does not track the pinned Stage 4 models, vocab, CharLM, Lexicon V3, SQLite DB, or golden fixtures, the draft workflow (`stage4_ios_coreml_parity_and_build.draft.yml`) enforces a strict two-bundle contract:
1. **Job 1 (`macos_coreml_parity_and_unsigned_ios_build`):**
   - Downloads and SHA-256 verifies **both** `coreml_mlpackages_bundle.tar.gz` (Colab-converted FP32/FP16 `.mlpackage` models) **and** `stage4_reference_validation_bundle.tar.gz` (containing ONLY the 6 pinned Stage 4 reference assets, frozen `golden_fixtures/`, and `menu-101.jpg` mapped to their exact repo-relative paths — zero restaurant dataset bulk).
   - Verifies all 6 pinned assets via `verify_pinned_stage4_bundle(strict=True)`.
   - Runs mandatory Mac Core ML `.predict()` parity (`scripts/stage4/verify_coreml_runtime_parity.py` for `fp32` and `fp16`) **BEFORE** `flutter build ios --release --no-codesign`.
   - Stages and uploads a checksummed `stage4-validated-ios-runtime-bundle` handoff artifact (`fp16/*.mlpackage` + 4 non-model runtime assets + parity reports + `validated_handoff_manifest.json`).
2. **Job 2 (`optional_signed_ios_ipa`):**
   - Downloads `stage4-validated-ios-runtime-bundle` from Job 1, verifies `passed == true && runtime_verified == true` and every SHA-256 in `validated_handoff_manifest.json`, and compiles `.mlpackage` → `.mlmodelc` via `xcrun coremlcompiler compile` into `apps/menu_camera_flutter/ios/Runner/Stage4Bundle/`.
   - **Enforces a native-plugin readiness guard:** refuses to produce a functional signed OCR IPA while `CoreMLMenuPlugin.swift` still contains `COREML_RUNTIME_UNVERIFIED` / `unverifiedCoreMLPipeline` (unless explicitly overridden via `allow_ui_only_scaffold_ipa: true`, in which case the uploaded artifact is explicitly named `stage4-ui-only-scaffold-ipa-NOT-FUNCTIONAL-OCR`).

## 4. Unsigned Archive vs Installable Signed IPA (Critical Distinctions)
1. **Unsigned Build (`flutter build ios --no-codesign` / `xcodebuild archive CODE_SIGNING_ALLOWED=NO`):**
   - Proves Swift/Flutter compilation and produces an unsigned `.app` / `.xcarchive`.
   - **NOT installable on a physical iPhone.** Zipping an unsigned `Payload/*.app` as `.ipa` does **not** make it installable.
   - `"ad-hoc code signing"` (`codesign -s -`) is **not** Apple Ad Hoc provisioned distribution.
2. **Installable Signed IPA (Ad Hoc or Development):**
   - Requires:
     1. Apple Distribution (for Ad Hoc) or Apple Development certificate + private key (`.p12` + password),
     2. Explicit App ID / Bundle ID,
     3. Provisioning profile (`.mobileprovision`) containing the target iPhone's **UDID** and matching entitlements,
     4. `ExportOptions.plist` (`method: ad-hoc` or `development`, `teamID`, `signingStyle: manual`, `provisioningProfiles` matching the active Xcode version),
     5. Ephemeral keychain import + `xcodebuild archive` + `xcodebuild -exportArchive` + mandatory cleanup (`if: always()`).
   - **Free Personal Team vs Paid Apple Developer Program:**
     - A **Free Personal Team** is intended for local Xcode development signing on personal devices with shorter profile expiry and stricter device/app quotas, and does not provide Apple Developer portal Ad Hoc `.mobileprovision` export for CI or TestFlight distribution.
     - A **Paid Apple Developer Program** membership is required for portal Ad Hoc `.mobileprovision` profiles, TestFlight, and App Store Connect distribution. Exact fees, expiry windows, and device/app limits are account- and policy-dependent; check Apple's official Developer Program documentation and the user's existing membership before recommending any purchase.

## 5. Windows-to-iPhone Installation Reality (GitHub VM Cannot USB-Attach Local iPhone)
- A GitHub-hosted macOS runner **cannot** attach a user's local iPhone over USB, and Mac Core ML `.predict()` / iOS Simulator **cannot** measure physical iPhone 15 camera stream parity, Neural Engine (ANE) speed, peak RAM, or 10-minute thermal behavior.
- Concrete installation paths from Windows once a **signed `.ipa`** is produced:
  1. **Official Apple TestFlight** (if enrolled in Apple Developer Program): upload signed build to App Store Connect; user installs OTA on iPhone 15 via the official TestFlight app without USB or a local Mac.
  2. **Official Ad Hoc OTA (`itms-services://`) over HTTPS:** host the Ad Hoc signed `.ipa` + manifest `.plist` over HTTPS for a device whose UDID is in the provisioning profile.
  3. **Optional Local Resigning Tools on Windows (e.g., Sideloadly / AltStore):** separate optional route using the user's own Apple ID, USB cable, iPhone Developer Mode, and manual profile trust (subject to Apple's Personal Team signing rules; not a bypass of Apple signing).

## 6. App Native Backend Readiness Honesty
- **Important:** `apps/menu_camera_flutter/ios/Runner/CoreMLMenuPlugin.swift` currently implements manifest verification, ordered resolution parity validation, AVFoundation stream buffer capture, and a serialized `DispatchQueue`, but **intentionally returns structured `COREML_ASSETS_NOT_BUNDLED` / `COREML_RUNTIME_UNVERIFIED` errors** until full Swift inference is wired and parity-gated on macOS.
- Therefore, **even a signed `.ipa` built today would NOT be functional on-device OCR yet** — remaining Swift work:
  1. Load compiled `V5DetectorOBB1024.mlmodelc`, `RecognizerFeatureExtractor.mlmodelc`, and `RecognizerRecurrentHead.mlmodelc` via Core ML `MLModel`.
  2. Port 1024 letterbox + ProbIoU rotated NMS + `line_crops` rectification & H64 bucketing (`128/256/384/512`) + host `valid_T` prefix slicing + `>512px` 100px-overlap window merging to Swift.
  3. Port canonical CTC folding + Unicode-scalar CharLM V2 R1 beam decoder (with 326 expanded-char fallback) + Lexicon V3 Trie/DP + read-only SQLite `culinary_menu.db` lookup to Swift.
  4. Pass `scripts/stage4/verify_coreml_runtime_parity.py` (FP32 + FP16) on macOS before bundling `.mlpackage` assets into Xcode.

## 7. Primary Official References Checked (2026-11-23)
- GitHub-hosted runners: https://docs.github.com/en/actions/reference/runners/github-hosted-runners
- GitHub Actions billing: https://docs.github.com/en/billing/concepts/product-billing/github-actions
- Apple Ad Hoc provisioning profile: https://developer.apple.com/help/account/provisioning-profiles/create-an-ad-hoc-provisioning-profile
- Apple distributing apps to development devices: https://developer.apple.com/documentation/xcode/distributing-apps-to-your-development-devices
- Xcode Organizer / ExportOptions reference: https://help.apple.com/xcode/mac/current/en.lproj/dev60b6fbbc7.html
