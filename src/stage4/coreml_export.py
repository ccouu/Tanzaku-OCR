"""Stage 4 Core ML export wrappers, mathematical parity verification, and macOS/Linux conversion driver.

Design invariants:
  - Uses ONLY the pinned Stage 4 bundle from `src.stage4.bundle_manifest.verify_pinned_stage4_bundle`
    and shared `MenuRecognizer` (`ARCH_SPECS['h4_k41']`) + `load_recognizer_state` + `resolve_checkpoint_state`.
  - Zero weight edits, zero BN recalibration, zero quantization (FP32 reference + FP16 only; no INT8),
    zero vocab/ID remapping (2,105 chars + CTC blank = 2,106 outputs).
  - Exact split recognizer export preserves packed BiLSTM valid-prefix semantics without tracing a
    single frozen length or running backward recurrence through padding:
      1. `RecognizerFeatureExtractorWrapper`: runs MobileNetV4 + learned H4 Conv2d projection +
         orientation conditioning + 2 TemporalBlocks on the EXACT physical replicate-padded bucket
         canvas `[1, 3, 64, W_bucket]` (`W_bucket in {128, 256, 384, 512}`), returning `[1, 256, T_bucket]`.
      2. Host slices valid prefix `temporal_features[:, :, :valid_T]` where `valid_T = ceil(valid_W / 4)`
         BEFORE backward LSTM execution.
      3. `RecognizerRecurrentHeadWrapper`: runs the exact unpadded BiLSTM (128/dir) + residual Linear(256,256)
         + classifier Conv1d(256, 2106, 1) on `[1, 256, valid_T]`, returning `[valid_T, 1, 2106]`.
  - Detector export preserves V5 YOLO11n-OBB 1024 letterbox, class mapping (`0: dish_horizontal`,
    `1: dish_vertical`), radian angles, and rotated ProbIoU NMS matching Ultralytics reference.
  - Honest platform boundaries: official `coremltools` supports conversion on macOS/Linux only,
    and native Core ML `.predict()` requires macOS/iOS. Never fabricates Windows Core ML predictions.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Optional

import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recognition.bn_calibration import snapshot_bn_buffers, verify_bn_buffers_identical
from src.recognition.checkpointing import resolve_checkpoint_state
from src.recognition.lexicon_lookup.menu_pipeline import LOCAL_YOLO_PKGS, V5YoloDetector
from src.recognition.model import ARCH_SPECS, MenuRecognizer, load_recognizer_state
from src.stage4.bundle_manifest import sha256_file, verify_pinned_stage4_bundle

RECOGNIZER_BUCKETS: tuple[int, ...] = (128, 256, 384, 512)
RECOGNIZER_VALID_T_MIN = 1
RECOGNIZER_VALID_T_MAX = 128
DETECTOR_IMGSZ = 1024
DETECTOR_CLASS_MAP: dict[int, str] = {0: "dish_horizontal", 1: "dish_vertical"}
DETECTOR_ANGLE_UNITS = "radians_ultralytics_xywhr"
TARGET_IOS_VERSION = "iOS17"


def compute_model_state_sha256(model: nn.Module) -> str:
    """Deterministic SHA-256 across all parameter and buffer tensors (float32/int64 bytes)."""
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class RecognizerFeatureExtractorWrapper(nn.Module):
    """Part 1 of exact split export: CNN + learned H4 projection + orientation + 2 TemporalBlocks.

    Inputs:
      - image: float32 `[1, 3, 64, W_bucket]` (ImageNet-normalized RGB on physical replicate-padded canvas)
      - orientation: float32 `[1]` (`0.0` horizontal, `1.0` vertical)
    Output:
      - temporal_features: float32 `[1, 256, W_bucket // 4]`
    """

    def __init__(self, source_model: MenuRecognizer) -> None:
        super().__init__()
        if source_model.height_pool != "conv2d_h4" or source_model.expected_h != 4:
            raise ValueError(
                f"Expected h4_k41 conv2d_h4 recognizer, got height_pool={source_model.height_pool!r}, "
                f"expected_h={source_model.expected_h}"
            )
        self.encoder = source_model.encoder
        self.projection = source_model.projection
        self.condition = source_model.condition
        self.temporal = source_model.temporal
        self.use_orientation = bool(source_model.use_orientation)

    def forward(self, image: torch.Tensor, orientation: torch.Tensor) -> torch.Tensor:
        feat_4d = self.encoder.forward_features(image)
        features = self.projection(feat_4d).squeeze(2)
        # Avoid dynamic `expand`/`tile` reps that fail Core ML EnumeratedShapes type inference;
        # elementwise broadcast over `zeros_like(features[:, :1, :])` is bit-identical.
        zeros_flag = torch.zeros_like(features[:, :1, :])
        if self.use_orientation:
            flag = zeros_flag + orientation.to(features.dtype).reshape(-1, 1, 1)
        else:
            flag = zeros_flag
        features = self.condition(torch.cat((features, flag), dim=1))
        return self.temporal(features)


class RecognizerRecurrentHeadWrapper(nn.Module):
    """Part 2 of exact split export: unpadded valid-prefix ResidualBiLSTM + 2,106 classifier.

    Host MUST slice `temporal_features[:, :, :valid_T]` where `valid_T = ceil(valid_W / 4)`
    before invoking this module, ensuring the backward LSTM starts from the exact valid end frame
    with zero padding leakage (mathematically identical to `pack_padded_sequence`).

    Input:
      - valid_features: float32 `[1, 256, valid_T]` (`1 <= valid_T <= 128`)
    Output:
      - logits: float32 `[valid_T, 1, 2106]`
    """

    def __init__(self, source_model: MenuRecognizer) -> None:
        super().__init__()
        if source_model.recurrent is None:
            raise ValueError("Source MenuRecognizer has recurrent=None; expected ResidualBiLSTM")
        self.lstm = source_model.recurrent.lstm
        self.output = source_model.recurrent.output
        self.classifier = source_model.classifier

    def forward(self, valid_features: torch.Tensor) -> torch.Tensor:
        seq = valid_features.transpose(1, 2)  # [1, valid_T, 256]
        encoded, _ = self.lstm(seq)           # [1, valid_T, 256]
        residual = self.output(encoded).transpose(1, 2)  # [1, 256, valid_T]
        fused = valid_features + residual
        return self.classifier(fused).permute(2, 0, 1)   # [valid_T, 1, 2106]


class DetectorOBBExportWrapper(nn.Module):
    """Fixed-1024 V5 YOLO11n-OBB raw prediction wrapper preserving class scores and radian angle.

    Input:
      - image_rgb01: float32 `[1, 3, 1024, 1024]` RGB in `[0.0, 1.0]` (letterboxed with 114/255 pad)
    Output:
      - raw_obb: float32 `[1, 7, 21504]` (`[cx, cy, w, h, cls0_prob, cls1_prob, angle_rad]`)
    """

    def __init__(self, yolo_inner_model: nn.Module) -> None:
        super().__init__()
        self.model = yolo_inner_model

    def forward(self, image_rgb01: torch.Tensor) -> torch.Tensor:
        out = self.model(image_rgb01)
        return out[0] if isinstance(out, (tuple, list)) else out


def load_pinned_recognizer(root: Path = ROOT, device: str = "cpu") -> tuple[MenuRecognizer, dict[str, Any], str]:
    """Load pinned Step1000 RAW 2105 recognizer via shared loader with strict SHA-256 and 147-BN check."""
    bundle = verify_pinned_stage4_bundle(root, strict=True)
    assets = bundle["assets"]
    vocab = json.loads(Path(assets["vocab_2105"]["path"]).read_text(encoding="utf-8"))
    chars = vocab["characters"]
    assert len(chars) == 2105
    sd, _, _, _ = resolve_checkpoint_state(
        Path(assets["ocr_checkpoint_raw"]["path"]),
        arm_key="raw",
        expected_sha256=assets["ocr_checkpoint_raw"]["expected_sha256"],
    )
    model = MenuRecognizer(len(chars) + 1, use_bilstm=True, **ARCH_SPECS["h4_k41"])
    load_recognizer_state(model, sd)
    bn_ref = snapshot_bn_buffers(model)
    assert len(bn_ref) == 147
    model.to(torch.device(device)).eval()
    state_sha = compute_model_state_sha256(model)
    return model, bundle, state_sha


def verify_recognizer_split_wrapper_parity(
    *,
    root: Path = ROOT,
    device: str = "cpu",
    atol: float = 1e-4,
) -> dict[str, Any]:
    """Prove exact PyTorch parity of split wrappers vs source MenuRecognizer across all 4 buckets,
    both orientations, and diverse valid widths (including very short non-multiples-of-4 & T=128).
    """
    model, bundle, state_sha_before = load_pinned_recognizer(root=root, device=device)
    bn_ref = snapshot_bn_buffers(model)
    feat_wrapper = RecognizerFeatureExtractorWrapper(model).eval()
    rec_wrapper = RecognizerRecurrentHeadWrapper(model).eval()

    dev = torch.device(device)
    gen = torch.Generator(device="cpu").manual_seed(20261123)

    # Test cases per physical bucket: very short, non-multiple-of-4, mid, near-boundary, exact boundary
    bucket_valid_widths: dict[int, tuple[int, ...]] = {
        128: (3, 5, 11, 17, 63, 125, 128),
        256: (129, 131, 177, 253, 256),
        384: (257, 261, 319, 381, 384),
        512: (385, 389, 447, 509, 512),
    }

    cases_report: list[dict[str, Any]] = []
    worst_abs_diff = 0.0
    worst_rel_diff = 0.0
    greedy_mismatches = 0

    with torch.inference_mode():
        for bucket_w, valid_widths in bucket_valid_widths.items():
            # Deterministic synthetic normalized tensor matching physical canvas [1, 3, 64, bucket_w]
            img = torch.randn((1, 3, 64, bucket_w), generator=gen, dtype=torch.float32).to(dev)
            for ori in (0, 1):
                ori_t = torch.tensor([float(ori)], dtype=torch.float32, device=dev)
                ori_long = torch.tensor([ori], dtype=torch.int64, device=dev)
                temporal_feats = feat_wrapper(img, ori_t)
                expected_t_phys = bucket_w // 4
                assert tuple(temporal_feats.shape) == (1, 256, expected_t_phys)

                for vw in valid_widths:
                    vt = (vw + 3) // 4
                    assert 1 <= vt <= expected_t_phys
                    ref_logits = model(img, ori_long, torch.tensor([vt], dtype=torch.int64, device=dev))[:vt]
                    split_logits = rec_wrapper(temporal_feats[:, :, :vt])
                    assert ref_logits.shape == split_logits.shape == (vt, 1, 2106)

                    diff = (ref_logits - split_logits).abs()
                    max_abs = float(diff.max().item())
                    denom = ref_logits.abs().clamp_min(1e-6)
                    max_rel = float((diff / denom).max().item())
                    ref_argmax = ref_logits[:, 0, :].argmax(dim=-1).tolist()
                    split_argmax = split_logits[:, 0, :].argmax(dim=-1).tolist()
                    argmax_eq = ref_argmax == split_argmax
                    if not argmax_eq:
                        greedy_mismatches += 1
                    worst_abs_diff = max(worst_abs_diff, max_abs)
                    worst_rel_diff = max(worst_rel_diff, max_rel)
                    cases_report.append(
                        {
                            "bucket_w": bucket_w,
                            "T_physical": expected_t_phys,
                            "orientation_flag": ori,
                            "valid_width": vw,
                            "valid_T": vt,
                            "max_abs_diff": round(max_abs, 9),
                            "argmax_identical": argmax_eq,
                        }
                    )

    bn_count = verify_bn_buffers_identical(model, bn_ref)
    bn_identical = bool(bn_count == 147)
    state_sha_after = compute_model_state_sha256(model)
    passed = (
        bn_identical
        and state_sha_before == state_sha_after
        and greedy_mismatches == 0
        and worst_abs_diff <= atol
    )
    return {
        "passed": passed,
        "device": str(dev),
        "atol": atol,
        "cases_tested": len(cases_report),
        "buckets_tested": list(RECOGNIZER_BUCKETS),
        "worst_abs_diff": worst_abs_diff,
        "worst_rel_diff": worst_rel_diff,
        "greedy_mismatches": greedy_mismatches,
        "bn_147_buffers_checked": int(bn_count),
        "bn_147_buffers_identical": bn_identical,
        "raw_state_sha256_before": state_sha_before,
        "raw_state_sha256_after": state_sha_after,
        "raw_state_sha256_unchanged": state_sha_before == state_sha_after,
        "bundle_id": bundle["bundle_id"],
        "cases": cases_report,
    }


def letterbox_rgb_square(
    rgb: Image.Image,
    target_size: int = DETECTOR_IMGSZ,
    pad_value: int = 114,
) -> tuple[np.ndarray, dict[str, float]]:
    """Ultralytics-compatible square letterbox preserving aspect ratio and returning inverse transform."""
    src_w, src_h = rgb.size
    gain = min(target_size / float(src_w), target_size / float(src_h))
    new_w = int(round(src_w * gain))
    new_h = int(round(src_h * gain))
    dw = (target_size - new_w) / 2.0
    dh = (target_size - new_h) / 2.0
    left = int(round(dw - 0.1))
    top = int(round(dh - 0.1))

    resized = rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_size, target_size), (pad_value, pad_value, pad_value))
    canvas.paste(resized, (left, top))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))[None, ...]  # [1, 3, target_size, target_size]
    meta = {"gain": gain, "pad_x": float(left), "pad_y": float(top), "src_w": float(src_w), "src_h": float(src_h)}
    return chw, meta


def xywhr_to_corners(cx: float, cy: float, w: float, h: float, angle_rad: float) -> list[list[float]]:
    """Convert Ultralytics xywhr (center, width, height, angle in radians) to 4 clockwise corners."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    vec1 = (w / 2.0 * cos_a, w / 2.0 * sin_a)
    vec2 = (-h / 2.0 * sin_a, h / 2.0 * cos_a)
    return [
        [cx + vec1[0] + vec2[0], cy + vec1[1] + vec2[1]],
        [cx + vec1[0] - vec2[0], cy + vec1[1] - vec2[1]],
        [cx - vec1[0] - vec2[0], cy - vec1[1] - vec2[1]],
        [cx - vec1[0] + vec2[0], cy - vec1[1] + vec2[1]],
    ]


def _probiou_matrix(obb1: np.ndarray, obb2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Exact ProbIoU between rotated boxes `[N, 5]` (`cx, cy, w, h, angle_rad`), matching Ultralytics."""
    x1, y1, w1, h1, r1 = obb1[:, 0:1], obb1[:, 1:2], obb1[:, 2:3], obb1[:, 3:4], obb1[:, 4:5]
    x2, y2, w2, h2, r2 = obb2[:, 0:1].T, obb2[:, 1:2].T, obb2[:, 2:3].T, obb2[:, 3:4].T, obb2[:, 4:5].T

    a1 = (w1 ** 2) / 12.0
    b1 = (h1 ** 2) / 12.0
    c1, s1 = np.cos(r1), np.sin(r1)
    cov_a1 = a1 * c1 * c1 + b1 * s1 * s1
    cov_b1 = a1 * s1 * s1 + b1 * c1 * c1
    cov_c1 = (a1 - b1) * c1 * s1

    a2 = (w2 ** 2) / 12.0
    b2 = (h2 ** 2) / 12.0
    c2, s2 = np.cos(r2), np.sin(r2)
    cov_a2 = a2 * c2 * c2 + b2 * s2 * s2
    cov_b2 = a2 * s2 * s2 + b2 * c2 * c2
    cov_c2 = (a2 - b2) * c2 * s2

    denom = (cov_a1 + cov_a2) * (cov_b1 + cov_b2) - (cov_c1 + cov_c2) ** 2 + eps
    t1 = (
        ((cov_a1 + cov_a2) * (y1 - y2) ** 2 + (cov_b1 + cov_b2) * (x1 - x2) ** 2) / denom
    ) * 0.25
    t2 = (((cov_c1 + cov_c2) * (x2 - x1) * (y1 - y2)) / denom) * 0.5
    det1 = np.maximum(cov_a1 * cov_b1 - cov_c1 ** 2, 0.0)
    det2 = np.maximum(cov_a2 * cov_b2 - cov_c2 ** 2, 0.0)
    t3 = np.log(np.maximum(denom / (4.0 * np.sqrt(det1 * det2) + eps), 1.0)) * 0.5
    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(np.maximum(1.0 - np.exp(-bd), eps))
    return 1.0 - hd


def decode_and_rotated_nms_obb(
    raw_obb: np.ndarray,
    letterbox_meta: dict[str, float],
    *,
    confidence: float = 0.35,
    iou_threshold: float = 0.70,
    max_det: int = 300,
) -> list[dict[str, Any]]:
    """Decode raw `[1, 7, N]` V5 YOLO11n-OBB output with class-aware ProbIoU rotated NMS and inverse letterbox."""
    arr = np.asarray(raw_obb, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    # arr shape: [7, N] -> [N, 7]: cx, cy, w, h, p0, p1, angle_rad
    preds = arr.T
    cls_scores = preds[:, 4:6]
    cls_ids = np.argmax(cls_scores, axis=1)
    confs = cls_scores[np.arange(len(preds)), cls_ids]
    keep_mask = confs >= float(confidence)
    if not np.any(keep_mask):
        return []

    kept_boxes = np.concatenate([preds[keep_mask, :4], preds[keep_mask, 6:7]], axis=1)  # [M, 5]
    kept_confs = confs[keep_mask]
    kept_cls = cls_ids[keep_mask]

    order = np.argsort(-kept_confs)
    selected: list[int] = []
    for idx in order:
        if len(selected) >= max_det:
            break
        box_i = kept_boxes[idx : idx + 1]
        cls_i = kept_cls[idx]
        suppressed = False
        for s_idx in selected:
            if kept_cls[s_idx] != cls_i:
                continue
            iou = float(_probiou_matrix(box_i, kept_boxes[s_idx : s_idx + 1])[0, 0])
            if iou > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            selected.append(int(idx))

    gain = letterbox_meta["gain"]
    pad_x = letterbox_meta["pad_x"]
    pad_y = letterbox_meta["pad_y"]
    src_w = letterbox_meta["src_w"]
    src_h = letterbox_meta["src_h"]

    results: list[dict[str, Any]] = []
    for out_idx, s_idx in enumerate(selected):
        cx, cy, w, h, ang = [float(v) for v in kept_boxes[s_idx]]
        cx_src = (cx - pad_x) / gain
        cy_src = (cy - pad_y) / gain
        w_src = w / gain
        h_src = h / gain
        pts = xywhr_to_corners(cx_src, cy_src, w_src, h_src, ang)
        pts_clamped = [
            [round(max(0.0, min(src_w, x)), 2), round(max(0.0, min(src_h, y)), 2)]
            for x, y in pts
        ]
        cls_id = int(kept_cls[s_idx])
        results.append(
            {
                "id": f"raw_{out_idx:03d}",
                "class_id": cls_id,
                "label": DETECTOR_CLASS_MAP[cls_id],
                "confidence": round(float(kept_confs[s_idx]), 4),
                "xywhr_src": [round(cx_src, 2), round(cy_src, 2), round(w_src, 2), round(h_src, 2), round(ang, 6)],
                "points": pts_clamped,
            }
        )
    return results


def verify_detector_obb_wrapper_parity(
    *,
    root: Path = ROOT,
    image_path: Optional[Path] = None,
    confidence: float = 0.35,
    iou_threshold: float = 0.70,
) -> dict[str, Any]:
    """Verify V5 YOLO11n-OBB 1024 export wrapper + rotated ProbIoU NMS against Ultralytics reference."""
    bundle = verify_pinned_stage4_bundle(root, strict=True)
    if LOCAL_YOLO_PKGS.is_dir() and str(LOCAL_YOLO_PKGS) not in sys.path:
        sys.path.append(str(LOCAL_YOLO_PKGS))

    w_path = Path(bundle["assets"]["detector_v5_obb"]["path"])
    det = V5YoloDetector(w_path, device="cpu", imgsz=DETECTOR_IMGSZ, iou=iou_threshold)
    yolo = det._ensure_loaded()
    assert dict(yolo.names) == DETECTOR_CLASS_MAP, f"Unexpected class map: {yolo.names}"

    sample_img_path = image_path or (root / "data/raw/gemini_banana_mobile_menus_50/menu-101.jpg")
    if sample_img_path.is_file():
        rgb = Image.open(sample_img_path).convert("RGB")
        img_name = sample_img_path.name
    else:
        rgb = Image.new("RGB", (768, 1024), (245, 240, 230))
        img_name = "synthetic_blank_768x1024"

    ref_dets, _ = det.detect(rgb, confidence=confidence)

    wrapper = DetectorOBBExportWrapper(yolo.model.eval()).eval()
    chw, lb_meta = letterbox_rgb_square(rgb, target_size=DETECTOR_IMGSZ)
    with torch.inference_mode():
        raw_out = wrapper(torch.from_numpy(chw)).detach().cpu().numpy()

    wrap_dets = decode_and_rotated_nms_obb(
        raw_out, lb_meta, confidence=confidence, iou_threshold=iou_threshold
    )

    # Match each reference detection to closest wrapper detection by ProbIoU / center distance
    ref_count = len(ref_dets)
    wrap_count = len(wrap_dets)
    matched = 0
    if ref_count > 0 and wrap_count > 0:
        for rd in ref_dets:
            rpts = np.asarray(rd["points"], dtype=np.float32)
            rc = rpts.mean(axis=0)
            best_dist = min(
                float(np.linalg.norm(rc - np.asarray(wd["points"], dtype=np.float32).mean(axis=0)))
                for wd in wrap_dets
                if wd["label"] == rd["label"]
            ) if any(wd["label"] == rd["label"] for wd in wrap_dets) else 1e9
            if best_dist <= 8.0:
                matched += 1

    recall = (matched / ref_count) if ref_count > 0 else 1.0
    passed = (
        tuple(raw_out.shape) == (1, 7, 21504)
        and abs(wrap_count - ref_count) <= max(1, int(0.05 * ref_count))
        and recall >= 0.95
    )
    return {
        "passed": passed,
        "image_name": img_name,
        "imgsz": DETECTOR_IMGSZ,
        "raw_output_shape": list(raw_out.shape),
        "class_map": DETECTOR_CLASS_MAP,
        "angle_units": DETECTOR_ANGLE_UNITS,
        "ultralytics_ref_count": ref_count,
        "wrapper_rotated_nms_count": wrap_count,
        "matched_within_8px": matched,
        "match_recall": round(recall, 4),
    }


def discover_export_environment() -> dict[str, Any]:
    """Safely inspect OS, Python, PyTorch, coremltools, Xcode, and Flutter iOS prerequisites."""
    sys_name = platform.system()
    ct_version: Optional[str] = None
    try:
        import coremltools as ct  # type: ignore

        ct_version = getattr(ct, "__version__", "unknown")
    except Exception:
        ct_version = None

    xcodebuild = shutil.which("xcodebuild")
    return {
        "os": sys_name,
        "os_release": platform.release(),
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "coremltools_installed": ct_version is not None,
        "coremltools_version": ct_version,
        "coreml_conversion_os_supported": sys_name in ("Darwin", "Linux"),
        "coreml_runtime_prediction_supported": sys_name == "Darwin",
        "xcodebuild_available": xcodebuild is not None,
        "target_ios_version": TARGET_IOS_VERSION,
    }


def export_coreml_models(
    *,
    out_dir: Path,
    root: Path = ROOT,
    fp16: bool = True,
    verify_runtime_on_mac: bool = True,
) -> dict[str, Any]:
    """Export V5 YOLO11n-OBB (1024) + split RecognizerFeatureExtractor + RecognizerRecurrentHead to .mlpackage.

    Runs actual conversion ONLY on supported macOS/Linux with `coremltools` installed.
    On Windows or missing `coremltools`, returns a structured `blocked` status without fabricating outputs.
    """
    env = discover_export_environment()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not env["coreml_conversion_os_supported"]:
        return {
            "status": "blocked",
            "reason": (
                f"Official coremltools conversion is supported on macOS and Linux only; "
                f"current host is {env['os']} {env['os_release']}."
            ),
            "environment": env,
            "artifacts": {},
        }
    if not env["coremltools_installed"]:
        return {
            "status": "blocked",
            "reason": (
                "coremltools is not installed in this isolated environment. "
                "Install scripts/stage4/export_requirements_macos_linux.txt in an isolated macOS/Linux venv."
            ),
            "environment": env,
            "artifacts": {},
        }

    import coremltools as ct  # type: ignore

    bundle = verify_pinned_stage4_bundle(root, strict=True)
    rec_model, _, rec_state_sha = load_pinned_recognizer(root=root, device="cpu")
    feat_mod = RecognizerFeatureExtractorWrapper(rec_model).eval()
    head_mod = RecognizerRecurrentHeadWrapper(rec_model).eval()

    if LOCAL_YOLO_PKGS.is_dir() and str(LOCAL_YOLO_PKGS) not in sys.path:
        sys.path.append(str(LOCAL_YOLO_PKGS))
    det = V5YoloDetector(Path(bundle["assets"]["detector_v5_obb"]["path"]), device="cpu", imgsz=DETECTOR_IMGSZ)
    yolo = det._ensure_loaded()
    det_mod = DetectorOBBExportWrapper(yolo.model.eval()).eval()

    skip_load = not env["coreml_runtime_prediction_supported"]
    target = ct.target.iOS17
    precisions = [("fp32", ct.precision.FLOAT32)]
    if fp16:
        precisions.append(("fp16", ct.precision.FLOAT16))

    exported: dict[str, Any] = {}
    with torch.inference_mode():
        traced_feat = torch.jit.trace(
            feat_mod,
            (torch.zeros(1, 3, 64, 512, dtype=torch.float32), torch.zeros(1, dtype=torch.float32)),
        )
        traced_head = torch.jit.trace(
            head_mod,
            (torch.zeros(1, 256, 32, dtype=torch.float32),),
        )
        dummy_det = torch.zeros(1, 3, DETECTOR_IMGSZ, DETECTOR_IMGSZ, dtype=torch.float32)
        # Warm up Ultralytics OBB head anchor/stride cache for 1024x1024 before tracing
        ref_det_out = det_mod(dummy_det)
        traced_det = torch.jit.trace(det_mod, (dummy_det,), check_trace=False)
        assert torch.allclose(traced_det(dummy_det), ref_det_out, atol=1e-5), "Traced OBB 1024 output diverged!"

    for prec_name, ct_prec in precisions:
        prec_dir = out_dir / prec_name
        prec_dir.mkdir(parents=True, exist_ok=True)

        # 1. RecognizerFeatureExtractor (EnumeratedShapes across the 4 physical buckets)
        feat_ml = ct.convert(
            traced_feat,
            convert_to="mlprogram",
            minimum_deployment_target=target,
            compute_precision=ct_prec,
            skip_model_load=skip_load,
            inputs=[
                ct.TensorType(
                    name="image",
                    shape=ct.EnumeratedShapes(shapes=[[1, 3, 64, w] for w in RECOGNIZER_BUCKETS]),
                    dtype=np.float32,
                ),
                ct.TensorType(name="orientation", shape=(1,), dtype=np.float32),
            ],
            outputs=[ct.TensorType(name="temporal_features", dtype=np.float32)],
        )
        feat_path = prec_dir / f"RecognizerFeatureExtractor_{prec_name}.mlpackage"
        feat_ml.save(str(feat_path))

        # 2. RecognizerRecurrentHead (RangeDim 1..128 for exact host-sliced valid_T)
        head_ml = ct.convert(
            traced_head,
            convert_to="mlprogram",
            minimum_deployment_target=target,
            compute_precision=ct_prec,
            skip_model_load=skip_load,
            inputs=[
                ct.TensorType(
                    name="valid_features",
                    shape=(1, 256, ct.RangeDim(lower_bound=RECOGNIZER_VALID_T_MIN, upper_bound=RECOGNIZER_VALID_T_MAX, default=32)),
                    dtype=np.float32,
                ),
            ],
            outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        )
        head_path = prec_dir / f"RecognizerRecurrentHead_{prec_name}.mlpackage"
        head_ml.save(str(head_path))

        # 3. V5 YOLO11n-OBB Detector (fixed 1024x1024)
        det_ml = ct.convert(
            traced_det,
            convert_to="mlprogram",
            minimum_deployment_target=target,
            compute_precision=ct_prec,
            skip_model_load=skip_load,
            inputs=[ct.TensorType(name="image_rgb01", shape=(1, 3, DETECTOR_IMGSZ, DETECTOR_IMGSZ), dtype=np.float32)],
            outputs=[ct.TensorType(name="raw_obb", dtype=np.float32)],
        )
        det_path = prec_dir / f"V5DetectorOBB1024_{prec_name}.mlpackage"
        det_ml.save(str(det_path))

        exported[prec_name] = {
            "recognizer_feature_extractor": str(feat_path),
            "recognizer_recurrent_head": str(head_path),
            "detector_v5_obb_1024": str(det_path),
            "runtime_status": "exported_unverified_runtime" if skip_load else "exported_macos_loadable",
        }

    manifest = {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_id": bundle["bundle_id"],
        "source_assets_sha256": {k: v["actual_sha256"] for k, v in bundle["assets"].items()},
        "recognizer_raw_state_sha256": rec_state_sha,
        "environment": env,
        "skip_model_load": skip_load,
        "artifacts": exported,
    }
    manifest_path = out_dir / "coreml_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "exported_unverified_runtime" if skip_load else "exported_macos",
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = cur
    return prev[-1]


def _canonical_greedy_from_logits(
    logits_t1c: np.ndarray,
    char_to_id: dict[str, int],
    id_to_char: dict[int, str],
) -> str:
    """Fold parenthesis logits BEFORE CTC collapse, then canonicalize digits/parentheses."""
    from src.recognition.canonical_ctc import fold_parenthesis_logits
    from src.recognition.label_normalization import normalize_digits, normalize_parentheses

    t_tensor = torch.from_numpy(np.asarray(logits_t1c, dtype=np.float32))
    folded = fold_parenthesis_logits(t_tensor, char_to_id, drop_aliases=False)[:, 0, :]
    ids = folded.argmax(dim=-1).tolist()
    collapsed: list[int] = []
    prev = 0
    for tok in ids:
        if tok != 0 and tok != prev:
            collapsed.append(tok)
        prev = tok
    raw_text = "".join(id_to_char.get(t, "") for t in collapsed)
    return normalize_digits(normalize_parentheses(raw_text)).strip()


COREML_PRECISION_TOLERANCES: dict[str, dict[str, float]] = {
    "fp32": {"recognizer_logit_atol": 1e-3, "detector_raw_atol": 1e-2, "max_cer_delta_pp": 0.20},
    "fp16": {"recognizer_logit_atol": 5e-2, "detector_raw_atol": 1e-1, "max_cer_delta_pp": 0.20},
}


def verify_coreml_runtime_parity(
    *,
    mlpackage_dir: Path,
    golden_fixtures_path: Path,
    precision: str = "fp32",
    root: Path = ROOT,
    model_loader: Any = None,
    _allow_mock_off_mac: bool = False,
) -> dict[str, Any]:
    """Mac-only Core ML runtime `.predict()` parity validator.

    Loads exported `RecognizerFeatureExtractor_<prec>.mlpackage`,
    `RecognizerRecurrentHead_<prec>.mlpackage`, and `V5DetectorOBB1024_<prec>.mlpackage`
    and compares actual Core ML `.predict()` outputs against exact PyTorch source models
    and frozen golden fixtures across:
      - All 4 physical buckets (128, 256, 384, 512), H/V orientations, edge valid_T values
      - All frozen OCR golden fixtures (blank, price/brackets, repeated digits, expanded Unicode, V/H)
      - Long >512px overlap-100 multi-window merged canonical greedy fixture
      - V5 YOLO11n-OBB 1024 raw tensor + ProbIoU rotated NMS parity on menu-101.jpg

    Fails clearly off-Mac (`status="blocked_off_mac"`, `passed=False`, `runtime_verified=False`)
    unless `_allow_mock_off_mac=True` is passed explicitly in unit tests with a mock `model_loader`.
    """
    from src.recognition.data import IMAGENET_MEAN, IMAGENET_STD, preprocess
    from src.recognition.long_line_inference import bucket_image, owned_tokens, window_ranges

    prec = precision.strip().lower()
    if prec not in COREML_PRECISION_TOLERANCES:
        raise ValueError(f"Unsupported precision={precision!r}; expected 'fp32' or 'fp16'")
    tol = COREML_PRECISION_TOLERANCES[prec]
    env = discover_export_environment()

    if not env["coreml_runtime_prediction_supported"] and not _allow_mock_off_mac:
        return {
            "status": "blocked_off_mac",
            "passed": False,
            "runtime_verified": False,
            "mock_execution": False,
            "precision": prec,
            "reason": (
                f"Core ML runtime .predict() requires macOS (Darwin); current host is {env['os']} {env['os_release']}. "
                "Export loadability is not runtime parity; run on macOS before iOS build."
            ),
            "environment": env,
        }

    prec_dir = Path(mlpackage_dir).resolve() / prec
    feat_path = prec_dir / f"RecognizerFeatureExtractor_{prec}.mlpackage"
    head_path = prec_dir / f"RecognizerRecurrentHead_{prec}.mlpackage"
    det_path = prec_dir / f"V5DetectorOBB1024_{prec}.mlpackage"

    if model_loader is None:
        missing_pkgs = [str(p) for p in (feat_path, head_path, det_path) if not p.exists()]
        if missing_pkgs:
            return {
                "status": "failed_missing_mlpackage",
                "passed": False,
                "runtime_verified": False,
                "mock_execution": False,
                "precision": prec,
                "reason": f"Missing exported Core ML package(s): {missing_pkgs}",
                "environment": env,
            }
        import coremltools as ct  # type: ignore

        model_loader = lambda p: ct.models.MLModel(str(p))

    feat_ml = model_loader(feat_path)
    head_ml = model_loader(head_path)
    det_ml = model_loader(det_path)

    golden_path = Path(golden_fixtures_path).resolve()
    if not golden_path.is_file():
        return {
            "status": "failed_missing_goldens",
            "passed": False,
            "runtime_verified": False,
            "mock_execution": bool(_allow_mock_off_mac),
            "precision": prec,
            "reason": f"Missing golden fixtures JSON: {golden_path}",
            "environment": env,
        }
    golden_doc = json.loads(golden_path.read_text(encoding="utf-8"))
    golden_root = golden_path.parent

    rec_model, bundle, _ = load_pinned_recognizer(root=root, device="cpu")
    vocab = json.loads(Path(bundle["assets"]["vocab_2105"]["path"]).read_text(encoding="utf-8"))
    chars = vocab["characters"]
    c2i = {c: i + 1 for i, c in enumerate(chars)}
    i2c = {v: k for k, v in c2i.items()}

    # 1. Synthetic multi-bucket / multi-valid_T / H+V / edge-case matrix via .predict()
    gen = torch.Generator(device="cpu").manual_seed(20261123)
    bucket_widths: dict[int, tuple[int, ...]] = {
        128: (3, 5, 17, 128),
        256: (129, 177, 256),
        384: (257, 381, 384),
        512: (385, 509, 512),
    }
    synthetic_worst_abs = 0.0
    synthetic_greedy_mismatches = 0
    synthetic_cases = 0

    with torch.inference_mode():
        for bw, vws in bucket_widths.items():
            img_t = torch.randn((1, 3, 64, bw), generator=gen, dtype=torch.float32)
            img_np = img_t.numpy()
            for ori in (0, 1):
                feat_out = feat_ml.predict(
                    {"image": img_np, "orientation": np.array([float(ori)], dtype=np.float32)}
                )["temporal_features"]
                feat_np = np.asarray(feat_out, dtype=np.float32)
                for vw in vws:
                    vt = (vw + 3) // 4
                    ref_logits = rec_model(
                        img_t, torch.tensor([ori], dtype=torch.int64), torch.tensor([vt], dtype=torch.int64)
                    )[:vt].numpy()
                    cm_logits = np.asarray(
                        head_ml.predict({"valid_features": np.ascontiguousarray(feat_np[:, :, :vt])})["logits"],
                        dtype=np.float32,
                    )
                    max_abs = float(np.max(np.abs(ref_logits - cm_logits)))
                    synthetic_worst_abs = max(synthetic_worst_abs, max_abs)
                    ref_txt = _canonical_greedy_from_logits(ref_logits, c2i, i2c)
                    cm_txt = _canonical_greedy_from_logits(cm_logits, c2i, i2c)
                    if ref_txt != cm_txt:
                        synthetic_greedy_mismatches += 1
                    synthetic_cases += 1

    # 2. Frozen OCR golden fixtures (blank, price/brackets, repeated digits, expanded Unicode, V/H)
    golden_worst_abs = 0.0
    golden_mismatches: list[dict[str, Any]] = []
    ref_edits = cm_edits = total_truth_chars = 0

    with torch.inference_mode():
        for row in golden_doc["ocr_golden_fixtures"]:
            img_p = golden_root / row["image_rel"]
            ori = int(row["orientation_flag"])
            bucket = int(row["bucket"])
            vt = int(row["valid_T"])
            with Image.open(img_p) as im:
                tensor, _, _ = preprocess(im.convert("RGB"), IMAGENET_MEAN, IMAGENET_STD, "batch", max_width=bucket)
            t_in = tensor[None]
            ref_logits = rec_model(
                t_in, torch.tensor([ori], dtype=torch.int64), torch.tensor([vt], dtype=torch.int64)
            )[:vt].numpy()
            feat_np = np.asarray(
                feat_ml.predict(
                    {"image": t_in.numpy(), "orientation": np.array([float(ori)], dtype=np.float32)}
                )["temporal_features"],
                dtype=np.float32,
            )
            cm_logits = np.asarray(
                head_ml.predict({"valid_features": np.ascontiguousarray(feat_np[:, :, :vt])})["logits"],
                dtype=np.float32,
            )
            max_abs = float(np.max(np.abs(ref_logits - cm_logits)))
            golden_worst_abs = max(golden_worst_abs, max_abs)
            cm_greedy = _canonical_greedy_from_logits(cm_logits, c2i, i2c)
            expected_greedy = str(row["greedy_ocr"])
            truth = str(row["truth"])
            if truth:
                total_truth_chars += len(truth)
                ref_edits += _levenshtein(expected_greedy, truth)
                cm_edits += _levenshtein(cm_greedy, truth)
            if cm_greedy != expected_greedy:
                golden_mismatches.append(
                    {"record_id": row["record_id"], "expected": expected_greedy, "coreml": cm_greedy}
                )

        # 3. Long >512px overlap-100 multi-window fixture
        lf = golden_doc["long_line_multi_window_fixture"]
        long_p = golden_root / lf["image_rel"]
        with Image.open(long_p) as lim:
            ranges = window_ranges(lim.width, max_width=512, overlap=100)
            canon_paths = []
            from src.recognition.canonical_ctc import fold_parenthesis_logits
            from src.recognition.label_normalization import normalize_digits, normalize_parentheses

            for left, right in ranges:
                fitted, meta = bucket_image(lim.crop((left, 0, right, 64)), max_width=512)
                tensor, _, _ = preprocess(fitted, IMAGENET_MEAN, IMAGENET_STD, "batch", max_width=meta["canvas"])
                vl = int(meta["input_length"])
                feat_np = np.asarray(
                    feat_ml.predict(
                        {"image": tensor[None].numpy(), "orientation": np.array([0.0], dtype=np.float32)}
                    )["temporal_features"],
                    dtype=np.float32,
                )
                cm_logits = np.asarray(
                    head_ml.predict({"valid_features": np.ascontiguousarray(feat_np[:, :, :vl])})["logits"],
                    dtype=np.float32,
                )
                folded = fold_parenthesis_logits(torch.from_numpy(cm_logits), c2i, drop_aliases=False)
                canon_paths.append(folded[:vl, 0].argmax(-1).tolist())
            cm_long_text = normalize_digits(
                normalize_parentheses("".join(i2c.get(t, "") for t in owned_tokens(canon_paths, ranges)))
            ).strip()
        long_line_match = cm_long_text == lf["merged_canonical_greedy_ocr"]

    # 4. Detector V5 YOLO11n-OBB 1024 runtime .predict() parity
    if LOCAL_YOLO_PKGS.is_dir() and str(LOCAL_YOLO_PKGS) not in sys.path:
        sys.path.append(str(LOCAL_YOLO_PKGS))
    det = V5YoloDetector(Path(bundle["assets"]["detector_v5_obb"]["path"]), device="cpu", imgsz=DETECTOR_IMGSZ, iou=0.70)
    yolo = det._ensure_loaded()
    det_wrapper = DetectorOBBExportWrapper(yolo.model.eval()).eval()
    sample_img_path = root / "data/raw/gemini_banana_mobile_menus_50/menu-101.jpg"
    rgb = Image.open(sample_img_path).convert("RGB") if sample_img_path.is_file() else Image.new("RGB", (768, 1024), (245, 240, 230))
    chw, lb_meta = letterbox_rgb_square(rgb, target_size=DETECTOR_IMGSZ)
    with torch.inference_mode():
        ref_raw_obb = det_wrapper(torch.from_numpy(chw)).detach().cpu().numpy()
    cm_raw_obb = np.asarray(det_ml.predict({"image_rgb01": chw})["raw_obb"], dtype=np.float32)
    det_max_abs = float(np.max(np.abs(ref_raw_obb - cm_raw_obb)))
    ref_dets = decode_and_rotated_nms_obb(ref_raw_obb, lb_meta, confidence=0.35, iou_threshold=0.70)
    cm_dets = decode_and_rotated_nms_obb(cm_raw_obb, lb_meta, confidence=0.35, iou_threshold=0.70)
    det_matched = 0
    for rd in ref_dets:
        rc = np.asarray(rd["points"], dtype=np.float32).mean(axis=0)
        same_cls = [wd for wd in cm_dets if wd["label"] == rd["label"]]
        if same_cls and min(float(np.linalg.norm(rc - np.asarray(wd["points"], dtype=np.float32).mean(axis=0))) for wd in same_cls) <= 8.0:
            det_matched += 1
    det_recall = (det_matched / len(ref_dets)) if ref_dets else 1.0

    ref_cer_pp = (100.0 * ref_edits / total_truth_chars) if total_truth_chars > 0 else 0.0
    cm_cer_pp = (100.0 * cm_edits / total_truth_chars) if total_truth_chars > 0 else 0.0
    cer_delta_pp = round(cm_cer_pp - ref_cer_pp, 4)

    passed = (
        synthetic_worst_abs <= tol["recognizer_logit_atol"]
        and golden_worst_abs <= tol["recognizer_logit_atol"]
        and synthetic_greedy_mismatches == 0
        and len(golden_mismatches) == 0
        and long_line_match
        and cer_delta_pp <= tol["max_cer_delta_pp"]
        and det_max_abs <= tol["detector_raw_atol"]
        and det_recall >= 0.95
    )

    return {
        "status": "passed" if passed else "failed_parity_gate",
        "passed": passed,
        "runtime_verified": bool(env["coreml_runtime_prediction_supported"] and not _allow_mock_off_mac and passed),
        "mock_execution": bool(_allow_mock_off_mac),
        "precision": prec,
        "tolerances": tol,
        "environment": env,
        "recognizer_synthetic_matrix": {
            "cases_tested": synthetic_cases,
            "worst_abs_diff": synthetic_worst_abs,
            "greedy_mismatches": synthetic_greedy_mismatches,
        },
        "recognizer_golden_fixtures": {
            "cases_tested": len(golden_doc["ocr_golden_fixtures"]),
            "worst_abs_diff": golden_worst_abs,
            "mismatches": golden_mismatches,
            "ref_cer_pp": round(ref_cer_pp, 4),
            "coreml_cer_pp": round(cm_cer_pp, 4),
            "cer_delta_pp": cer_delta_pp,
        },
        "long_line_gt512_match": long_line_match,
        "detector_obb_1024_runtime": {
            "raw_max_abs_diff": det_max_abs,
            "ref_count": len(ref_dets),
            "coreml_count": len(cm_dets),
            "matched_within_8px": det_matched,
            "match_recall": round(det_recall, 4),
        },
    }
