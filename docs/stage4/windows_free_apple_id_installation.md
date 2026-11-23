# iPhone trial: GitHub unsigned build -> Windows signing

Updated2026-11-23. **User-selected installation approach:** build on a GitHub-hosted macOS runner without signing credentials, download an unsigned IPA, then sign/install locally on Windows with the user's own free Apple ID. This skips signing in CI; it does not skip the signature required by iOS. No jailbreak, certificate-sharing service or signing bypass is needed for our own app.

## Recommended first route: Sideloadly

For this single-app trial, use Sideloadly first; AltStore **Classic** is an alternative. AltStore PAL is a separate marketplace product, not this local IPA workflow. The tools are third-party software; install only from their official sites and enter Apple credentials/2FA yourself in the signing tool. Never put an Apple password in this repo, chat, GitHub secrets or build logs for this route.

### 1. GitHub builds the actual device app

The macOS workflow checks out the scoped application/export code, retrieves checksummed model/runtime assets, runs actual native CoreML parity before accepting a functional build, compiles models into app resources and builds the Flutter **release arm64 iPhone app**, not an iOS Simulator binary:

```bash
cd apps/menu_camera_flutter
flutter pub get
flutter analyze
flutter test
flutter build ios --release --no-codesign
```

Use the unsigned `build/ios/iphoneos/Runner.app`. Package the COMPLETE app, including Frameworks, compiled model resources and Flutter assets, into a ZIP with top-level `Payload/Runner.app` and `.ipa` extension. Preserve permissions/symlinks; do not zip only the executable, omit nested frameworks or use the simulator build. An example packaging step, to be tested by CI:

```bash
mkdir -p "$RUNNER_TEMP/stage4_ipa/Payload"
ditto build/ios/iphoneos/Runner.app "$RUNNER_TEMP/stage4_ipa/Payload/Runner.app"
cd "$RUNNER_TEMP/stage4_ipa"
ditto -c -k --keepParent Payload "$RUNNER_TEMP/MenuCamera-unsigned.ipa"
shasum -a 256 "$RUNNER_TEMP/MenuCamera-unsigned.ipa"
```

This unsigned IPA is **input to the local signing tool**, not directly installable. No `xcodebuild -exportArchive` development/ad-hoc export or signing secret is required in this chosen CI path. Keep an unsigned compile-only smoke artifact distinct from the functional OCR artifact; a build with missing resources or unfinished native inference must never be labelled functional.

Publish the IPA, its checksum, resource inventory and CI parity report as short-retention workflow artifacts. No restaurant-image datasets, signing material or credentials belong in the commit/artifact. Do not quietly replace unavailable models with the desktop Python endpoint. Model/lexicon asset delivery remains an explicit checksummed CI input; public repository status does not itself publish local assets.

### 2. Prepare Windows and iPhone

Use the current official Sideloadly Windows installer. Its setup guide currently calls for Apple-distributed iTunes/iCloud components rather than Microsoft Store versions; inspect existing installation first, and let the user perform any required system install/replacement. Do not automatically uninstall Apple's software or change system security settings.

Connect the unlocked iPhone by USB and accept the trust/pairing prompt. Download the GitHub artifact, extract its outer artifact ZIP to obtain `MenuCamera-unsigned.ipa`, and verify its SHA before opening it. Do not pass the outer GitHub artifact ZIP to the signer.

### 3. Sign and install locally

In Sideloadly, select the iPhone, load the IPA, enter the user's Apple ID and complete any Apple authentication prompts personally, then install. Trust the corresponding developer profile and enable iPhone Developer Mode if iOS requests them. Confirm exact menus against the device's iOS version; these are user-controlled device changes.

Keep the same Apple ID and stable app bundle identifier for updates. Changing identifiers creates a new installation identity and may affect local data and free-account limits. Use only entitlements/capabilities actually needed by our camera/CoreML/offline app; unsupported paid-team capabilities or app extensions can prevent free-account signing. Do not grant unrelated permissions or alter installed profile entitlements blindly.

### 4. Refresh and verify

Sideloadly's current FAQ states free-account apps expire after7days and free accounts are limited to3 installed sideloaded apps. Refresh/re-sign before expiry; its refresh daemon requires the computer and a usable USB/Wi-Fi connection. These are upstream limits to recheck if Apple/tool behavior changes, not application promises.

After launch, verify Airplane Mode processing, camera stream/saved dimensions, all-line results, H/V/long-line handling, correct asset hashes and full-page cold/warm timings on the actual iPhone. Successful signing or installation does not prove inference parity or performance.

## AltStore Classic alternative

Install AltServer on Windows using its official guide, install AltStore Classic on the paired iPhone, then import the unsigned IPA into Classic with AltServer available. It also signs using the user's Apple account and requires refreshes. Classic itself occupies one of the limited sideloaded app slots. This route adds a phone-side application and may suit ongoing refresh management; it is not necessary to install both tools for our first trial.

## Readiness and status boundaries

- Colab Linux `.mlpackage` conversion: completed; still needs native Mac prediction/parity.
- GitHub unsigned device build and packaging: planned/setup work; only an actual CI run can establish success.
- Native iOS inference implementation: in progress; do not treat exported models or a Flutter UI scaffold as a complete backend.
- Local signing/device trust and physical phone tests: require the user and connected device; not performed by the GitHub VM.
- No paid Apple Developer membership, CI signing certificates or AppStore release is required for the chosen personal sideload trial. Availability depends on current Apple/tool support and successful authentication.

## Sources checked2026-11-23

- [Sideloadly official download and Windows prerequisites](https://sideloadly.io/index.html).
- [Sideloadly official FAQ: account limits, refresh and updates](https://sideloadly.io/faq).
- [AltStore Classic Windows installation](https://faq.altstore.io/altstore-classic/how-to-install-altstore-windows).
- [AltStore Classic operation and refresh](https://faq.altstore.io/altstore-classic/your-altstore).
- [Flutter iOS build/deployment guide](https://docs.flutter.dev/deployment/ios).

No installer execution, Apple login, device security change, GitHub publication or workflow run was performed by writing this guide.
