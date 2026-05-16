#!/usr/bin/env python3
"""P59: metadata-only dispatch probe for dual NVFP4 artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.nvfp4_layout import detect_nvfp4_layout  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="Model directory to classify")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    reports = [detect_nvfp4_layout(path).to_dict() for path in args.model]
    result = {
        "schema_version": "lynn-engine-p59-nvfp4-layout-dispatch-probe-v1",
        "reports": reports,
        "fail_loud_ok": all(r["recommended_loader"] != "none" for r in reports),
        "layout_kinds": [r["layout_kind"] for r in reports],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
