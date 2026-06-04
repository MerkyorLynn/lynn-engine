#!/usr/bin/env python3
"""R5-B: e8m0 repack bridge for Lynn per-16 NVFP4 scales.

R5-A proved two things on R6000:

* per-16 grouping can be preserved by padding each 16-value group to group32;
* current Lynn E4M3-like scales cannot be used zero-copy as e8m0 scales.

R5-B tests the next bridge: dequantize the current per-16 value contract,
requantize each per-16 group into E2M1 codes with an e8m0 power-of-two scale,
then run the padded group32 path through `tl.dot_scaled`.

This is still a synthetic, model-free numeric gate.  It may bank repack numeric
feasibility only; it does not bank a grouped-MoE kernel, speed, or default path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.r5a_stage6_per16_layout_bridge import (  # noqa: E402
    _E2M1_MAG,
    _bench,
    _dequant_codes,
    _dot_scaled_mn,
    _expand_packed_per16_to_group32,
    _inventory,
    _make_codes,
    _make_scale16,
    _metrics,
    _pack_codes,
    _time_stats,
)


def _choose_exp(raw: torch.Tensor, mode: str) -> torch.Tensor:
    log2 = torch.log2(raw.float().clamp_min(1.0e-30))
    if mode == "nearest":
        return torch.round(log2)
    if mode == "floor":
        return torch.floor(log2)
    if mode == "ceil":
        return torch.ceil(log2)
    raise ValueError(f"unknown exponent mode {mode!r}")


def _requantize_per16_to_e8m0(
    codes: torch.Tensor,
    scale16: torch.Tensor,
    *,
    exponent_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    values = _dequant_codes(codes, scale16)
    rows, k = values.shape
    groups = k // 16
    grouped = values.reshape(rows, groups, 16).float()
    max_abs = grouped.abs().amax(dim=-1).clamp_min(1.0e-30)
    raw_scale = max_abs / float(_E2M1_MAG[-1].item())
    exponent = _choose_exp(raw_scale, exponent_mode).clamp(-126, 127)
    scale = torch.pow(torch.full_like(exponent, 2.0), exponent)
    normalized = grouped / scale.unsqueeze(-1)
    table = _E2M1_MAG.to(values.device)
    mag = torch.argmin((normalized.abs().unsqueeze(-1) - table.view(1, 1, 1, -1)).abs(), dim=-1)
    sign = (normalized < 0).to(torch.uint8) << 3
    repacked_codes = (mag.to(torch.uint8) | sign).reshape(rows, k).contiguous()
    repacked_packed = _pack_codes(repacked_codes)
    repacked_scale = (exponent + 127).clamp(0, 255).to(torch.uint8).contiguous()
    padded = _expand_packed_per16_to_group32(repacked_packed)
    approx = _dequant_codes(repacked_codes, scale)
    diff = approx.float() - values.float()
    return padded, repacked_scale, {
        "value_rel_l2": float(torch.linalg.vector_norm(diff).item() / torch.linalg.vector_norm(values.float()).clamp_min(1e-12).item()),
        "value_cosine": float(F.cosine_similarity(approx.float().flatten(), values.float().flatten(), dim=0).item()),
        "scale_exponent_min": float(exponent.min().item()),
        "scale_exponent_max": float(exponent.max().item()),
    }


def _run_case(args: argparse.Namespace, m: int, n: int, k: int, device: torch.device) -> dict[str, Any]:
    groups16 = k // 16
    act_codes = _make_codes(m, k, device=device, seed=args.seed + m + 17)
    weight_codes = _make_codes(n, k, device=device, seed=args.seed + n + 31)
    act_scale16 = _make_scale16(m, groups16, device=device, case="e4m3_like")
    weight_scale16 = _make_scale16(n, groups16, device=device, case="e4m3_like")
    act_ref = _dequant_codes(act_codes, act_scale16)
    weight_ref = _dequant_codes(weight_codes, weight_scale16)

    def scalar_reference() -> torch.Tensor:
        return torch.matmul(act_ref, weight_ref.t())

    ref, ref_times = _bench(scalar_reference, args.warmup, args.repeats)
    rows: list[dict[str, Any]] = []
    for mode in [item.strip() for item in args.exponent_modes.split(",") if item.strip()]:
        act_padded, act_scale, act_repack = _requantize_per16_to_e8m0(act_codes, act_scale16, exponent_mode=mode)
        weight_padded, weight_scale, weight_repack = _requantize_per16_to_e8m0(weight_codes, weight_scale16, exponent_mode=mode)

        def candidate() -> torch.Tensor:
            return _dot_scaled_mn(
                act_padded,
                act_scale,
                weight_padded,
                weight_scale,
                block_m=args.block_m,
                block_n=args.block_n,
                block_k_packed=args.block_k_packed,
            )

        out, times = _bench(candidate, args.warmup, args.repeats)
        rows.append({
            "candidate": f"e8m0_repack_{mode}",
            "metrics": _metrics(out, ref),
            "timing_ms": _time_stats(times),
            "act_repack": act_repack,
            "weight_repack": weight_repack,
            "bytes": {
                "packed_ratio_vs_original": 2.0,
                "scale_ratio_vs_original": 0.25,
            },
        })
    best = min(rows, key=lambda row: float(row["metrics"]["rel_l2"]))
    return {
        "scale_case": "e4m3_like",
        "shape": {"M": m, "N": n, "K": k, "groups16": groups16},
        "reference_timing_ms": _time_stats(ref_times),
        "candidates": rows,
        "best_candidate": best["candidate"],
        "best_rel_l2": best["metrics"]["rel_l2"],
        "best_cosine": best["metrics"]["cosine"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--m-values", default="1,16,64")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--block-m", type=int, default=16)
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--block-k-packed", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--exponent-modes", default="nearest,floor,ceil")
    ap.add_argument("--rel-l2-max", type=float, default=0.08)
    ap.add_argument("--cosine-min", type=float, default=0.995)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("R5-B requires CUDA")
    if args.k % 32 != 0:
        raise ValueError("--k must be divisible by 32")
    device = torch.device("cuda")
    m_values = [int(x.strip()) for x in args.m_values.split(",") if x.strip()]
    t0 = time.time()
    cases = [_run_case(args, m, args.n, args.k, device) for m in m_values]
    elapsed_s = time.time() - t0
    best_rows = [min(case["candidates"], key=lambda row: float(row["metrics"]["rel_l2"])) for case in cases]
    numeric_ok = all(
        row["metrics"]["rel_l2"] <= args.rel_l2_max and row["metrics"]["cosine"] >= args.cosine_min
        for row in best_rows
    )
    result = {
        "schema": "lynn-stage6-r5b-e8m0-repack-bridge-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "inventory": _inventory(),
        "elapsed_seconds": elapsed_s,
        "dimensions": {"H": args.k, "I": args.n // 2, "gate_up_rows": args.n, "m_values": m_values},
        "thresholds": {"rel_l2_max": args.rel_l2_max, "cosine_min": args.cosine_min},
        "cases": cases,
        "passes": {
            "e4m3_like_e8m0_repack_numeric_ok": bool(numeric_ok),
            "banked_repack_numeric": bool(numeric_ok),
            "banked_grouped_moe_fp4_mma_poc": False,
            "banked_kernel_speed": False,
            "banked_default_promotion": False,
            "all": bool(numeric_ok),
        },
        "decision": "PASS_R5B_E8M0_REPACK_NUMERIC" if numeric_ok else "FAIL_R5B_E8M0_REPACK_NUMERIC",
        "notes": [
            "R5-B banks only synthetic e8m0 repack numeric feasibility.",
            "The path still doubles packed K through padded group32 and does not bank grouped-MoE kernel speed.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passes"]["all"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
