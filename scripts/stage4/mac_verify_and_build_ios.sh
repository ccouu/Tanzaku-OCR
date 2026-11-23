#!/usr/bin/env bash
# Ready one-command macOS verification, Core ML conversion, runtime .predict() parity gate,
# and iOS build script.
# Run ONLY on an authorized macOS host with Xcode and Flutter iOS SDK installed.
# FAILS BEFORE `flutter build ios` if Core ML runtime parity is missing or fails.
# Does NOT bypass Apple signing or distribute to the App Store.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/outputs/stage4/coreml_export_iphone_20261123"
VENV_DIR="${HOME}/.cache/stage4_coreml_export_venv"

echo "=== 1. Check macOS / Xcode / Flutter prerequisites ==="
if [ "$(uname -s)" != "Darwin" ]; then
  echo "[FAIL] This runbook requires macOS (Darwin) for Core ML .predict() runtime parity and Xcode iOS build." >&2
  exit 1
fi
uname -a
xcodebuild -version
flutter doctor -v
flutter devices

echo "=== 2. Create isolated Core ML export venv (never touches desktop .venv-recognition) ==="
if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip
pip install -r "${ROOT_DIR}/scripts/stage4/export_requirements_macos_linux.txt"

echo "=== 3. Verify PyTorch split-wrapper parity & convert .mlpackage (FP32 + FP16) ==="
python3 "${ROOT_DIR}/scripts/stage4/export_coreml_models.py" \
  --out-dir "${OUT_DIR}" \
  --device cpu \
  --export

echo "=== 4. MANDATORY Mac Core ML .predict() runtime parity gate (FP32 + FP16) ==="
echo "    (Aborts BEFORE flutter build ios if any package is missing or any gate fails)"
python3 "${ROOT_DIR}/scripts/stage4/verify_coreml_runtime_parity.py" \
  --out-dir "${OUT_DIR}" \
  --precision fp32
python3 "${ROOT_DIR}/scripts/stage4/verify_coreml_runtime_parity.py" \
  --out-dir "${OUT_DIR}" \
  --precision fp16

echo "=== 5. Run Flutter unit/widget tests & iOS no-codesign build check ==="
cd "${ROOT_DIR}/apps/menu_camera_flutter"
flutter pub get
flutter analyze
flutter test
flutter build ios --no-codesign

echo "=== 6. Next step for physical iPhone 15 ==="
echo "Core ML runtime parity PASSED for FP32 and FP16. Note: CoreMLMenuPlugin.swift still requires"
echo "wiring the verified .mlpackage inference + CharLM/Lexicon Swift execution on macOS before"
echo "physical iPhone 15 Airplane Mode validation."
