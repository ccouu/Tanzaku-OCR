"""Mac-only Core ML runtime `.predict()` parity validator CLI.

Fails clearly with non-zero exit code off-Mac, on missing .mlpackage artifacts,
or on any logit / canonical-greedy / CER / rotated-OBB parity gate failure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage4.coreml_export import verify_coreml_runtime_parity


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Mac-only Core ML runtime .predict() parity gate")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/stage4/coreml_export_iphone_20261123")
    ap.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    ap.add_argument(
        "--goldens",
        type=Path,
        default=ROOT / "outputs/stage4/coreml_export_iphone_20261123/golden_fixtures/coreml_golden_parity_fixtures.json",
    )
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rep = verify_coreml_runtime_parity(
        mlpackage_dir=args.out_dir / "mlpackages",
        golden_fixtures_path=args.goldens,
        precision=args.precision,
        root=ROOT,
    )
    out_file = args.out_dir / f"coreml_runtime_parity_{args.precision}.json"
    out_file.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if not (rep.get("passed") and rep.get("runtime_verified")):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
