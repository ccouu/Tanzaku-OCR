"""Remote Colab Linux Core ML export worker invoked by shared `src.recognition.colab_runner`.

Creates an isolated Linux virtualenv with tested PyTorch + coremltools versions,
verifies all 6 pinned Stage 4 assets, proves 44-case PyTorch split-wrapper parity and
V5 YOLO11n-OBB 1024 rotated-NMS parity, converts all 6 FP32/FP16 `.mlpackage` models
(with `skip_model_load=True`, honestly labelled `exported_unverified_runtime`),
inventories every package file with SHA-256, archives the packages, and publishes
job-owned `artifact_inventory.json` + `DONE.json` for SHA-verified retrieval.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print(f"[CMD] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def _setup_isolated_venv(venv_dir: Path, lock_out: Path) -> Path:
    vpy = venv_dir / "bin" / "python"
    uv_bin = shutil.which("uv")
    py312 = shutil.which("python3.12") or shutil.which("python3.11") or sys.executable

    if uv_bin:
        if not vpy.is_file():
            print(f"[STEP 1] Creating isolated Python 3.12 Linux venv via uv at {venv_dir}...", flush=True)
            _run([uv_bin, "venv", "--python", py312, str(venv_dir)])
        print("[STEP 1] Installing isolated tested Core ML export deps via uv (CPU PyTorch 2.5.1 + coremltools 8.3.0)...", flush=True)
        _run([
            uv_bin, "pip", "install",
            "--python", str(vpy),
            "--index-strategy", "unsafe-best-match",
            "--extra-index-url", "https://download.pytorch.org/whl/cpu",
            "torch==2.5.1+cpu",
            "torchvision==0.20.1+cpu",
            "coremltools==8.3.0",
            "timm==1.0.31",
            "ultralytics==8.3.40",
            "numpy==1.26.4",
            "pillow>=10.0,<12.0",
            "opencv-python-headless>=4.8,<4.12",
            "scipy>=1.11,<1.16",
        ])
        lock_out.parent.mkdir(parents=True, exist_ok=True)
        cp = subprocess.run([uv_bin, "pip", "freeze", "--python", str(vpy)], check=True, capture_output=True, text=True)
        lock_out.write_text(cp.stdout, encoding="utf-8")
        print(f"[STEP 1] Saved toolchain lock ({len(cp.stdout.splitlines())} packages) to {lock_out}", flush=True)
        return vpy

    if not vpy.is_file():
        print(f"[STEP 1] Creating isolated Linux venv at {venv_dir}...", flush=True)
        _run([py312, "-m", "venv", str(venv_dir)])
    vpip = venv_dir / "bin" / "pip"
    print("[STEP 1] Installing isolated tested Core ML export dependencies (CPU PyTorch 2.5.1 + coremltools 8.3.0)...", flush=True)
    _run([str(vpy), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])
    _run([
        str(vpip), "install",
        "--index-url", "https://download.pytorch.org/whl/cpu",
        "torch==2.5.1", "torchvision==0.20.1",
    ])
    _run([
        str(vpip), "install",
        "coremltools==8.3.0",
        "timm==1.0.31",
        "ultralytics==8.3.40",
        "numpy==1.26.4",
        "pillow>=10.0,<12.0",
        "opencv-python-headless>=4.8,<4.12",
        "scipy>=1.11,<1.16",
    ])
    lock_out.parent.mkdir(parents=True, exist_ok=True)
    cp = subprocess.run([str(vpip), "freeze"], check=True, capture_output=True, text=True)
    lock_out.write_text(cp.stdout, encoding="utf-8")
    print(f"[STEP 1] Saved toolchain lock ({len(cp.stdout.splitlines())} packages) to {lock_out}", flush=True)
    return vpy


def _materialize_pinned_ckpt(root: Path) -> None:
    src = root / "payload" / "source_ckpt.pt"
    dst = root / "outputs/recognition/experiments/l2_selected2105_short_recovery_1000/downloads/short_recovery/step_1000.pt"
    if not dst.is_file():
        if not src.is_file():
            raise FileNotFoundError(f"Missing bundled checkpoint at {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def _run_inside_venv(root: Path, out_dir: Path, job_id: str) -> None:
    t0 = time.perf_counter()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.stage4.bundle_manifest import verify_pinned_stage4_bundle
    from src.stage4.coreml_export import (
        discover_export_environment,
        export_coreml_models,
        verify_detector_obb_wrapper_parity,
        verify_recognizer_split_wrapper_parity,
    )

    print(f"[STEP 2] Verifying all 6 pinned Stage 4 assets inside {root}...", flush=True)
    bundle = verify_pinned_stage4_bundle(root, strict=True)
    env = discover_export_environment()
    print(f"[STEP 2] Environment: {json.dumps(env)}", flush=True)

    print("[STEP 3] Running 44-case PyTorch split-wrapper parity proof across all 4 buckets & valid lengths...", flush=True)
    rec_parity = verify_recognizer_split_wrapper_parity(root=root, device="cpu", atol=1e-4)
    if not rec_parity["passed"]:
        raise RuntimeError(f"Recognizer split-wrapper parity FAILED: {json.dumps(rec_parity, indent=2)}")
    print(
        f"[STEP 3] Recognizer parity PASSED: cases={rec_parity['cases_tested']}, "
        f"worst_abs_diff={rec_parity['worst_abs_diff']:.2e}, bn147={rec_parity['bn_147_buffers_identical']}",
        flush=True,
    )

    print("[STEP 4] Running V5 YOLO11n-OBB 1024 wrapper + ProbIoU rotated NMS parity proof...", flush=True)
    det_parity = verify_detector_obb_wrapper_parity(root=root)
    if not det_parity["passed"]:
        raise RuntimeError(f"Detector OBB 1024 parity FAILED: {json.dumps(det_parity, indent=2)}")
    print(
        f"[STEP 4] Detector parity PASSED: ref={det_parity['ultralytics_ref_count']}, "
        f"wrap={det_parity['wrapper_rotated_nms_count']}, recall={det_parity['match_recall']}",
        flush=True,
    )

    mlpkg_dir = out_dir / "mlpackages"
    print(f"[STEP 5] Converting FP32 + FP16 .mlpackage models into {mlpkg_dir} (skip_model_load=True on Linux)...", flush=True)
    export_res = export_coreml_models(out_dir=mlpkg_dir, root=root, fp16=True)
    if export_res["status"] not in ("exported_unverified_runtime", "exported_macos"):
        raise RuntimeError(f"Core ML export failed or blocked: {json.dumps(export_res, indent=2)}")
    print(f"[STEP 5] Export status: {export_res['status']}", flush=True)

    # Complete per-file SHA-256 inventory of every file inside every .mlpackage
    print("[STEP 6] Building complete per-file .mlpackage SHA-256 inventory and tar.gz archive...", flush=True)
    pkg_file_inv: dict[str, dict[str, Any]] = {}
    pkg_summary: dict[str, dict[str, Any]] = {}
    for prec in ("fp32", "fp16"):
        prec_dir = mlpkg_dir / prec
        if not prec_dir.is_dir():
            continue
        for pkg in sorted(prec_dir.glob("*.mlpackage")):
            rel_pkg = pkg.relative_to(mlpkg_dir).as_posix()
            n_files = 0
            tot_bytes = 0
            for f in sorted(pkg.rglob("*")):
                if f.is_file():
                    rel_f = f.relative_to(mlpkg_dir).as_posix()
                    sz = f.stat().st_size
                    pkg_file_inv[rel_f] = {"sha256": _sha256_file(f), "bytes": sz}
                    n_files += 1
                    tot_bytes += sz
            pkg_summary[rel_pkg] = {"file_count": n_files, "total_bytes": tot_bytes}

    file_inv_path = out_dir / "mlpackage_file_inventory.json"
    file_inv_path.write_text(
        json.dumps({"packages": pkg_summary, "files": pkg_file_inv}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    archive_path = out_dir / "coreml_mlpackages_bundle.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tf:
        for rel_f in sorted(pkg_file_inv.keys()):
            tf.add(mlpkg_dir / rel_f, arcname=rel_f, recursive=False)
        tf.add(mlpkg_dir / "coreml_export_manifest.json", arcname="coreml_export_manifest.json", recursive=False)

    manifest_copy = out_dir / "coreml_export_manifest.json"
    shutil.copy2(mlpkg_dir / "coreml_export_manifest.json", manifest_copy)

    summary = {
        "job_id": job_id,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "bundle_id": bundle["bundle_id"],
        "environment": env,
        "export_status": export_res["status"],
        "runtime_verified": False,
        "runtime_verification_note": (
            "Converted on Colab Linux with skip_model_load=True. "
            "Linux conversion is NOT native .mlmodelc compilation and NOT Core ML .predict() parity; "
            "run scripts/stage4/verify_coreml_runtime_parity.py on macOS before iOS build."
        ),
        "recognizer_split_wrapper_parity": {k: v for k, v in rec_parity.items() if k != "cases"},
        "detector_obb_1024_parity": det_parity,
        "packages_summary": pkg_summary,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": _sha256_file(archive_path),
    }
    summary_path = out_dir / "colab_export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Flat path-to-SHA inventory for colab_runner.retrieve_terminal_inventory
    artifact_files = [
        archive_path,
        manifest_copy,
        file_inv_path,
        summary_path,
        out_dir / "toolchain_lock.txt",
    ]
    inventory: dict[str, str] = {}
    for fp in artifact_files:
        if fp.is_file():
            inventory[fp.relative_to(root).as_posix()] = _sha256_file(fp)
    inv_path = out_dir / "artifact_inventory.json"
    inv_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Write job-owned DONE.json ONLY after all artifacts & inventory are flushed
    done_path = root / "DONE.json"
    done_path.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "status": "completed",
                "artifact_manifest": inv_path.relative_to(root).as_posix(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] Colab Core ML export complete in {summary['elapsed_s']}s; wrote {done_path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Remote Colab Linux Core ML export worker")
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--out-subdir", default="export_out")
    ap.add_argument("--venv-dir", default="/content/stage4_coreml_venv_py312")
    ap.add_argument("--inside-venv", action="store_true")
    args = ap.parse_args(argv)

    root = Path.cwd().resolve()
    out_dir = (root / args.out_subdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.inside_venv:
        print(f"[COLAB_EXPORT] Starting job_id={args.job_id} on {platform.platform()}", flush=True)
        _materialize_pinned_ckpt(root)
        vpy = _setup_isolated_venv(Path(args.venv_dir), out_dir / "toolchain_lock.txt")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root)
        env["MPLBACKEND"] = "Agg"
        _run(
            [
                str(vpy),
                "-u",
                "-m",
                "src.stage4.colab_coreml_export_worker",
                "--job-id",
                args.job_id,
                "--out-subdir",
                args.out_subdir,
                "--venv-dir",
                args.venv_dir,
                "--inside-venv",
            ],
            cwd=root,
            env=env,
        )
        return 0

    _run_inside_venv(root, out_dir, args.job_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
