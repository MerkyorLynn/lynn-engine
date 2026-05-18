#!/usr/bin/env python3
"""P162 - Search simple FP32 reduction trees for packed MoE gate/up.

P161 showed individual terms are bit-exact while the 256-term partial sum
differs from Triton.  This probe tries common deterministic FP32 reduction
trees against Triton's actual `tl.sum` result and the native CUDA result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p160_native_packed_moe_gateup_partial_trace import _triton_partial_gateup  # noqa: E402
from benchmarks.p161_native_packed_moe_gateup_term_trace import (  # noqa: E402
    _load_fixture,
    _target_from_p160,
    _triton_terms,
)


def _f32(x: float | np.float32) -> np.float32:
    return np.float32(x)


def _left_fold(vals: np.ndarray) -> np.float32:
    acc = _f32(0.0)
    for value in vals:
        acc = _f32(acc + value)
    return acc


def _right_fold(vals: np.ndarray) -> np.float32:
    acc = _f32(0.0)
    for value in vals[::-1]:
        acc = _f32(acc + value)
    return acc


def _pairwise_halving(vals: np.ndarray) -> np.float32:
    arr = vals.astype(np.float32, copy=True)
    n = arr.shape[0]
    while n > 1:
        half = n // 2
        arr[:half] = np.float32(arr[:half] + arr[half:n])
        n = half
    return _f32(arr[0])


def _pairwise_halving_reversed(vals: np.ndarray) -> np.float32:
    return _pairwise_halving(vals[::-1].copy())


def _chunk_left_then_pairwise(vals: np.ndarray, chunk: int) -> np.float32:
    chunks = []
    for start in range(0, vals.shape[0], chunk):
        chunks.append(_left_fold(vals[start : start + chunk]))
    return _pairwise_halving(np.asarray(chunks, dtype=np.float32))


def _chunk_pairwise_then_left(vals: np.ndarray, chunk: int) -> np.float32:
    chunks = []
    for start in range(0, vals.shape[0], chunk):
        chunks.append(_pairwise_halving(vals[start : start + chunk]))
    return _left_fold(np.asarray(chunks, dtype=np.float32))


def _warp_then_pairwise(vals: np.ndarray, warp: int = 32) -> np.float32:
    chunks = []
    for start in range(0, vals.shape[0], warp):
        chunks.append(_pairwise_halving(vals[start : start + warp]))
    return _pairwise_halving(np.asarray(chunks, dtype=np.float32))


def _warp_then_left(vals: np.ndarray, warp: int = 32) -> np.float32:
    chunks = []
    for start in range(0, vals.shape[0], warp):
        chunks.append(_pairwise_halving(vals[start : start + warp]))
    return _left_fold(np.asarray(chunks, dtype=np.float32))


def _candidates() -> list[tuple[str, Callable[[np.ndarray], np.float32]]]:
    funcs: list[tuple[str, Callable[[np.ndarray], np.float32]]] = [
        ("left_fold", _left_fold),
        ("right_fold", _right_fold),
        ("pairwise_halving", _pairwise_halving),
        ("pairwise_halving_reversed", _pairwise_halving_reversed),
        ("warp32_then_pairwise", _warp_then_pairwise),
        ("warp32_then_left", _warp_then_left),
    ]
    for chunk in (2, 4, 8, 16, 32, 64, 128):
        funcs.append((f"chunk{chunk}_left_then_pairwise", lambda vals, c=chunk: _chunk_left_then_pairwise(vals, c)))
        funcs.append((f"chunk{chunk}_pairwise_then_left", lambda vals, c=chunk: _chunk_pairwise_then_left(vals, c)))
    return funcs


def main() -> int:
    ap = argparse.ArgumentParser(description="P162 packed MoE gate/up reduction tree search.")
    ap.add_argument("--packed-fixtures", required=True)
    ap.add_argument("--p160-report", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from engine.native_cuda import load_lynn_native_extension

    packed_dir = Path(args.packed_fixtures)
    target = _target_from_p160(Path(args.p160_report))
    data = _load_fixture(packed_dir / target["fixture_file"], args.device)
    x = data["hidden_in"].to(torch.bfloat16).view(-1).contiguous()
    gu_packed = data["slot_gate_up_packed"].contiguous()
    gu_scale = data["slot_gate_up_scale"].to(torch.float16).contiguous()
    gu_global = data["slot_gate_up_global_scale"].to(torch.float16).contiguous()

    triton_terms = _triton_terms(
        x,
        gu_packed,
        gu_scale,
        gu_global,
        slot=target["slot"],
        row=target["row"],
        kind=target["kind"],
        hidden_block=target["hidden_block"],
    ).contiguous()
    triton_partial = _triton_partial_gateup(
        x,
        gu_packed,
        gu_scale,
        gu_global,
        block_inter=8,
        block_hidden=256,
        num_warps=4,
    )
    triton_target = float(
        triton_partial[target["slot"], target["hidden_block"], target["kind"], target["row"]].item()
    )

    ext = load_lynn_native_extension(verbose=False)
    native_partial = ext.moe_slot_packed_nvfp4_partial_accum_triton_order_probe(
        x, gu_packed, gu_scale, gu_global
    ).contiguous()
    native_target = float(
        native_partial[target["slot"], target["hidden_block"], target["kind"], target["row"]].item()
    )

    terms = triton_terms.detach().cpu().numpy().astype(np.float32)
    rows: list[dict[str, Any]] = []
    for name, fn in _candidates():
        value = float(fn(terms))
        rows.append(
            {
                "name": name,
                "value": value,
                "abs_diff_vs_triton": abs(value - triton_target),
                "abs_diff_vs_native": abs(value - native_target),
                "matches_triton": value == triton_target,
                "matches_native": value == native_target,
            }
        )
    rows.sort(key=lambda row: float(row["abs_diff_vs_triton"]))

    report = {
        "schema": "lynn-p162-moe-gateup-reduction-tree-search-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "packed_fixtures": str(packed_dir),
        "p160_report": args.p160_report,
        "target": target,
        "triton_target": triton_target,
        "native_target": native_target,
        "target_abs_diff": abs(triton_target - native_target),
        "best": rows[0],
        "matched_triton": [row for row in rows if row["matches_triton"]],
        "matched_native": [row for row in rows if row["matches_native"]],
        "candidates": rows,
    }
    if report["matched_triton"]:
        report["diagnosis"] = "SIMPLE_TREE_MATCHED_TRITON"
    elif report["matched_native"]:
        report["diagnosis"] = "SIMPLE_TREE_MATCHED_NATIVE_ONLY"
    else:
        report["diagnosis"] = "NO_SIMPLE_TREE_MATCHED_TRITON"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        f"[p162] diagnosis={report['diagnosis']} "
        f"triton={triton_target:.9f} native={native_target:.9f} "
        f"best={rows[0]['name']} diff={rows[0]['abs_diff_vs_triton']:.2e}"
    )
    print(f"[p162] report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
