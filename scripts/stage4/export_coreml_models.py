"""Thin CLI for Stage 4 Core ML export, wrapper parity proof, and environment discovery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage4.coreml_export import (
    discover_export_environment,
    export_coreml_models,
    verify_detector_obb_wrapper_parity,
    verify_recognizer_split_wrapper_parity,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 4 Core ML export & PyTorch wrapper parity CLI")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/stage4/coreml_export_iphone_20261123")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--verify-wrapper-only", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--no-fp16", action="store_true")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    env = discover_export_environment()
    rec_parity = verify_recognizer_split_wrapper_parity(root=ROOT, device=args.device)
    det_parity = verify_detector_obb_wrapper_parity(root=ROOT)

    report = {
        "environment": env,
        "recognizer_split_wrapper_parity": {k: v for k, v in rec_parity.items() if k != "cases"},
        "recognizer_cases_count": len(rec_parity["cases"]),
        "detector_obb_wrapper_parity": det_parity,
        "export_result": None,
    }
    if not args.verify_wrapper_only and args.export:
        report["export_result"] = export_coreml_models(
            out_dir=args.out_dir / "mlpackages",
            root=ROOT,
            fp16=not args.no_fp16,
        )

    out_path = args.out_dir / "export_cli_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not (rec_parity["passed"] and det_parity["passed"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
