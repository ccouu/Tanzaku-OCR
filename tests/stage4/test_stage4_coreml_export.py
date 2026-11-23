"""Focused tests for Stage 4 Core ML export wrappers, mathematical parity, and honest OS boundaries."""
from __future__ import annotations

from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pytest
import torch

from src.recognition.lexicon_lookup.menu_pipeline import LOCAL_YOLO_PKGS, V5YoloDetector
from src.stage4.coreml_export import (
    DETECTOR_CLASS_MAP,
    DETECTOR_IMGSZ,
    DetectorOBBExportWrapper,
    RECOGNIZER_BUCKETS,
    RecognizerFeatureExtractorWrapper,
    RecognizerRecurrentHeadWrapper,
    discover_export_environment,
    export_coreml_models,
    load_pinned_recognizer,
    verify_coreml_runtime_parity,
    verify_detector_obb_wrapper_parity,
    verify_recognizer_split_wrapper_parity,
)

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_JSON = ROOT / "outputs/stage4/coreml_export_iphone_20261123/golden_fixtures/coreml_golden_parity_fixtures.json"


def test_recognizer_split_wrapper_parity_all_4_buckets_and_valid_lengths():
    rep = verify_recognizer_split_wrapper_parity(root=ROOT, device="cpu", atol=1e-4)
    assert rep["passed"] is True
    assert rep["buckets_tested"] == list(RECOGNIZER_BUCKETS)
    assert rep["cases_tested"] == 44
    assert rep["greedy_mismatches"] == 0
    assert rep["bn_147_buffers_checked"] == 147
    assert rep["bn_147_buffers_identical"] is True
    assert rep["raw_state_sha256_unchanged"] is True
    assert rep["worst_abs_diff"] <= 1e-4


def test_detector_v5_obb_1024_wrapper_and_rotated_nms_parity():
    rep = verify_detector_obb_wrapper_parity(root=ROOT)
    assert rep["passed"] is True
    assert rep["raw_output_shape"] == [1, 7, 21504]
    assert rep["class_map"] == DETECTOR_CLASS_MAP
    assert rep["match_recall"] >= 0.95


def test_export_environment_reports_honest_os_boundary(tmp_path: Path):
    env = discover_export_environment()
    assert env["os"] == platform.system()
    if platform.system() == "Windows":
        assert env["coreml_conversion_os_supported"] is False
        assert env["coreml_runtime_prediction_supported"] is False
        res = export_coreml_models(out_dir=tmp_path / "mlpkg", root=ROOT)
        assert res["status"] == "blocked"
        assert "macOS and Linux" in res["reason"]


def test_coreml_runtime_parity_fails_clearly_off_mac(tmp_path: Path):
    if platform.system() == "Darwin":
        pytest.skip("Off-Mac blocker test runs on non-macOS hosts")
    rep = verify_coreml_runtime_parity(
        mlpackage_dir=tmp_path / "mlpackages",
        golden_fixtures_path=GOLDEN_JSON,
        precision="fp32",
        root=ROOT,
    )
    assert rep["status"] == "blocked_off_mac"
    assert rep["passed"] is False
    assert rep["runtime_verified"] is False
    assert "macOS" in rep["reason"]

    # CLI must also exit non-zero off-Mac
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage4/verify_coreml_runtime_parity.py"),
            "--out-dir",
            str(tmp_path / "cli_out"),
            "--precision",
            "fp32",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "blocked_off_mac" in proc.stdout


def test_coreml_runtime_parity_mock_mac_predict_pass_and_fail_gates(tmp_path: Path):
    assert GOLDEN_JSON.is_file(), f"Missing frozen golden fixtures: {GOLDEN_JSON}"
    rec_model, bundle, _ = load_pinned_recognizer(root=ROOT, device="cpu")
    feat_mod = RecognizerFeatureExtractorWrapper(rec_model).eval()
    head_mod = RecognizerRecurrentHeadWrapper(rec_model).eval()

    if LOCAL_YOLO_PKGS.is_dir() and str(LOCAL_YOLO_PKGS) not in sys.path:
        sys.path.append(str(LOCAL_YOLO_PKGS))
    det = V5YoloDetector(Path(bundle["assets"]["detector_v5_obb"]["path"]), device="cpu", imgsz=DETECTOR_IMGSZ)
    det_mod = DetectorOBBExportWrapper(det._ensure_loaded().model.eval()).eval()

    class _MockMLModel:
        def __init__(self, fn):
            self._fn = fn

        def predict(self, inputs: dict):
            return self._fn(inputs)

    def _passing_loader(path: Path):
        name = path.name
        if "FeatureExtractor" in name:
            def _f(inp):
                with torch.inference_mode():
                    out = feat_mod(
                        torch.from_numpy(np.asarray(inp["image"], dtype=np.float32)),
                        torch.from_numpy(np.asarray(inp["orientation"], dtype=np.float32)),
                    ).numpy()
                return {"temporal_features": out}
            return _MockMLModel(_f)
        if "RecurrentHead" in name:
            def _h(inp):
                with torch.inference_mode():
                    out = head_mod(torch.from_numpy(np.asarray(inp["valid_features"], dtype=np.float32))).numpy()
                return {"logits": out}
            return _MockMLModel(_h)
        def _d(inp):
            with torch.inference_mode():
                out = det_mod(torch.from_numpy(np.asarray(inp["image_rgb01"], dtype=np.float32))).numpy()
            return {"raw_obb": out}
        return _MockMLModel(_d)

    rep_ok = verify_coreml_runtime_parity(
        mlpackage_dir=tmp_path / "mlpkg",
        golden_fixtures_path=GOLDEN_JSON,
        precision="fp32",
        root=ROOT,
        model_loader=_passing_loader,
        _allow_mock_off_mac=True,
    )
    assert rep_ok["status"] == "passed"
    assert rep_ok["passed"] is True
    assert rep_ok["mock_execution"] is True
    # Mock off-Mac MUST NOT claim runtime_verified=True!
    assert rep_ok["runtime_verified"] is False
    assert rep_ok["recognizer_synthetic_matrix"]["cases_tested"] == 26
    assert rep_ok["recognizer_synthetic_matrix"]["greedy_mismatches"] == 0
    assert rep_ok["recognizer_golden_fixtures"]["cases_tested"] == 10
    assert rep_ok["recognizer_golden_fixtures"]["mismatches"] == []
    assert rep_ok["long_line_gt512_match"] is True
    assert rep_ok["detector_obb_1024_runtime"]["match_recall"] >= 0.95

    # Corrupted logits MUST trip the parity gate!
    def _corrupted_loader(path: Path):
        base_m = _passing_loader(path)
        if "RecurrentHead" in path.name:
            return _MockMLModel(lambda inp: {"logits": base_m.predict(inp)["logits"] + 5.0})
        return base_m

    rep_bad = verify_coreml_runtime_parity(
        mlpackage_dir=tmp_path / "mlpkg",
        golden_fixtures_path=GOLDEN_JSON,
        precision="fp32",
        root=ROOT,
        model_loader=_corrupted_loader,
        _allow_mock_off_mac=True,
    )
    assert rep_bad["status"] == "failed_parity_gate"
    assert rep_bad["passed"] is False
    assert rep_bad["runtime_verified"] is False


def test_mac_runbook_enforces_runtime_parity_before_ios_build():
    lines = (ROOT / "scripts/stage4/mac_verify_and_build_ios.sh").read_text(encoding="utf-8").splitlines()
    cmd_lines = [
        ln.strip()
        for ln in lines
        if ln.strip() and not ln.lstrip().startswith("#") and not ln.lstrip().startswith("echo ")
    ]
    cmd_text = "\n".join(cmd_lines)
    assert "set -euo pipefail" in cmd_text
    idx_parity = cmd_text.index("verify_coreml_runtime_parity.py")
    idx_ios_build = cmd_text.index("flutter build ios")
    assert idx_parity < idx_ios_build
    assert "--precision fp32" in cmd_text and "--precision fp16" in cmd_text


def test_active_ios_unsigned_ipa_workflow_contracts_and_readiness_guards():
    wf_path = ROOT / ".github/workflows/stage4_ios_unsigned_ipa.yml"
    assert wf_path.is_file(), f"Missing active unsigned IPA workflow: {wf_path}"
    wf = wf_path.read_text(encoding="utf-8")

    # 1. Pinned macos-15 ARM64, Flutter 3.50.1, read-only permissions, short retention, zero Apple secrets/P12
    assert "runs-on: macos-15" in wf
    assert 'flutter-version: "3.50.1"' in wf
    assert "contents: read" in wf
    assert "retention-days: 7" in wf
    for forbidden_secret in ("APPLE_SIGNING_CERT", "APPLE_PROVISIONING_PROFILE", "cert.p12", "exportArchive"):
        assert forbidden_secret not in wf, f"Unexpected CI signing material in unsigned Windows-signing workflow: {forbidden_secret}"

    # 2. Separate compile-only smoke vs functional unsigned IPA, both clearly labelled unsigned-for-Windows-signing
    assert "compile_only_smoke:" in wf
    assert "functional_unsigned_ipa:" in wf
    assert "stage4-ios-compile-only-smoke-unsigned-for-Windows-signing" in wf
    assert "stage4-ios-functional-unsigned-for-Windows-signing" in wf
    assert "ditto -c -k --sequesterRsrc --keepParent Payload" in wf

    # 3. Functional path enforces checksummed bundles, Mac Core ML .predict() parity BEFORE build,
    #    coremlcompiler -> .mlmodelc, and capability check of all 4 registered Swift modules
    assert "mlpackages_bundle_url:" in wf and "mlpackages_bundle_sha256:" in wf
    assert "reference_validation_bundle_url:" in wf and "reference_validation_bundle_sha256:" in wf
    assert "verify_pinned_stage4_bundle" in wf
    assert "verify_coreml_runtime_parity.py" in wf
    assert "xcrun coremlcompiler compile" in wf
    for swift_file in ("CoreMLMenuPlugin.swift", "Stage4NativeCamera.swift", "Stage4CoreMLInference.swift", "Stage4SQLiteStore.swift"):
        assert swift_file in wf
        assert (ROOT / "apps/menu_camera_flutter/ios/Runner" / swift_file).is_file()

    # 4. Deployment doc avoids hardcoded billing/pricing/Personal-Team guesses
    doc = (ROOT / "docs/stage4/github_actions_ios_deployment.md").read_text(encoding="utf-8")
    for forbidden in ("$0.062", "$0.25", "$0.07", "$99", "10×", "7-day", "max 3 apps"):
        assert forbidden not in doc, f"Hardcoded billing/limit guess found in deployment doc: {forbidden}"
    assert "account- and plan-dependent" in doc.lower() or "account- and policy-dependent" in doc.lower()
