#!/usr/bin/env python3
"""CLI wrapper for Lynn Engine Phase 4 P1 quant manifest scanning."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.quant_manifest import save_manifest, scan_checkpoint


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="HF/safetensors checkpoint directory")
    ap.add_argument("--out", help="write manifest JSON to this path")
    ap.add_argument("--max-samples", type=int, default=24)
    args = ap.parse_args()

    if args.out:
        manifest = save_manifest(args.model_dir, args.out, max_samples=args.max_samples)
    else:
        manifest = scan_checkpoint(args.model_dir, max_samples=args.max_samples)
    print(manifest.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
