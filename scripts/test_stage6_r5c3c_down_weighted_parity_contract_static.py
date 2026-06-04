#!/usr/bin/env python3
"""GPU-free static checks for the R5-C3C down/weighted parity contract."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "reports" / "stage6" / "R5C3C_DOWN_WEIGHTED_PARITY_CONTRACT_20260604.md"


def main() -> int:
    failures: list[str] = []
    if not CONTRACT.exists():
        failures.append(f"missing {CONTRACT.relative_to(ROOT)}")
    else:
        text = CONTRACT.read_text(encoding="utf-8")
        for needle in [
            "PASS_R5C3C_DOWN_WEIGHTED_PARITY_SMOKE",
            "banked_down_projection_numeric_parity=true",
            "banked_weighted_topk_numeric_parity=true",
            "banked_grouped_moe_fp4_mma_poc=false",
            "banked_kernel_speed=false",
            "banked_default_promotion=false",
            "R5-C3B",
            "R5-C3C",
            "SwiGLU",
            "down projection",
            "weighted top-k",
            "does not bank full active-MoE FP4-MMA speed",
        ]:
            if needle not in text:
                failures.append(f"contract missing {needle!r}")
    if failures:
        print("Stage 6 R5-C3C down/weighted parity contract static check FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Stage 6 R5-C3C down/weighted parity contract static check PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
