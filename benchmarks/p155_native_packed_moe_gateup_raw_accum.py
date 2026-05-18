#!/usr/bin/env python3
"""P155 - Raw gate/up accumulator diagnostics for native packed MoE.

P154 showed that a Triton-like hidden-block reduction improves native
gate/up exactness but does not close it.  This probe compares the raw FP32
gate_acc/up_acc tensors before SiLU/BF16 store, using an inline Triton kernel
that mirrors P147 gate/up and writes both raw accumulators and the BF16 inter.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 512


if HAS_TRITON:

    @triton.jit
    def _e2m1_from_nibble_fast(nibble):
        mag = nibble & 0x07
        sign = (nibble & 0x08) != 0
        mag_f = mag.to(tl.float32)
        val = tl.where(
            mag <= 4,
            mag_f * 0.5,
            tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0)),
        )
        return tl.where(sign, -val, val)

    @triton.jit
    def _p155_gateup_raw_reference_kernel(
        x_ptr,
        expert_ids_ptr,
        gate_up_packed_ptr,
        gate_up_scale_ptr,
        global_scale_ptr,
        raw_ptr,
        inter_ptr,
        PACKED_STRIDE_E: tl.constexpr,
        PACKED_STRIDE_M: tl.constexpr,
        PACKED_STRIDE_N: tl.constexpr,
        SCALE_STRIDE_E: tl.constexpr,
        SCALE_STRIDE_M: tl.constexpr,
        SCALE_STRIDE_G: tl.constexpr,
        RAW_STRIDE_K: tl.constexpr,
        RAW_STRIDE_KIND: tl.constexpr,
        RAW_STRIDE_I: tl.constexpr,
        INTER_STRIDE_K: tl.constexpr,
        INTER_STRIDE_I: tl.constexpr,
        HIDDEN: tl.constexpr,
        INTERMEDIATE: tl.constexpr,
        BLOCK_INTER: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
    ):
        slot = tl.program_id(0)
        block_i = tl.program_id(1)
        expert = tl.load(expert_ids_ptr + slot)
        inter_offsets = block_i * BLOCK_INTER + tl.arange(0, BLOCK_INTER)
        inter_mask = inter_offsets < INTERMEDIATE
        h_offsets = tl.arange(0, BLOCK_HIDDEN)
        global_scale = tl.load(global_scale_ptr).to(tl.float32)

        gate_acc = tl.zeros((BLOCK_INTER,), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_INTER,), dtype=tl.float32)

        for h0 in range(0, HIDDEN, BLOCK_HIDDEN):
            cols = h0 + h_offsets
            col_mask = cols < HIDDEN
            packed_cols = cols // 2
            scale_cols = cols // 16
            x = tl.load(x_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

            gate_rows = inter_offsets
            up_rows = INTERMEDIATE + inter_offsets
            gate_packed_offsets = (
                expert * PACKED_STRIDE_E
                + gate_rows[:, None] * PACKED_STRIDE_M
                + packed_cols[None, :] * PACKED_STRIDE_N
            )
            up_packed_offsets = (
                expert * PACKED_STRIDE_E
                + up_rows[:, None] * PACKED_STRIDE_M
                + packed_cols[None, :] * PACKED_STRIDE_N
            )
            gate_scale_offsets = (
                expert * SCALE_STRIDE_E
                + gate_rows[:, None] * SCALE_STRIDE_M
                + scale_cols[None, :] * SCALE_STRIDE_G
            )
            up_scale_offsets = (
                expert * SCALE_STRIDE_E
                + up_rows[:, None] * SCALE_STRIDE_M
                + scale_cols[None, :] * SCALE_STRIDE_G
            )

            gate_packed = tl.load(
                gate_up_packed_ptr + gate_packed_offsets,
                mask=inter_mask[:, None] & col_mask[None, :],
                other=0,
            )
            up_packed = tl.load(
                gate_up_packed_ptr + up_packed_offsets,
                mask=inter_mask[:, None] & col_mask[None, :],
                other=0,
            )
            gate_nibble = tl.where((cols[None, :] & 1) == 0, gate_packed & 0x0F, (gate_packed >> 4) & 0x0F)
            up_nibble = tl.where((cols[None, :] & 1) == 0, up_packed & 0x0F, (up_packed >> 4) & 0x0F)
            gate_w = _e2m1_from_nibble_fast(gate_nibble)
            up_w = _e2m1_from_nibble_fast(up_nibble)
            gate_scale = tl.load(
                gate_up_scale_ptr + gate_scale_offsets,
                mask=inter_mask[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            up_scale = tl.load(
                gate_up_scale_ptr + up_scale_offsets,
                mask=inter_mask[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            gate_acc += tl.sum(gate_w * (gate_scale / global_scale) * x[None, :], axis=1)
            up_acc += tl.sum(up_w * (up_scale / global_scale) * x[None, :], axis=1)

        tl.store(
            raw_ptr + slot * RAW_STRIDE_K + inter_offsets * RAW_STRIDE_I,
            gate_acc,
            mask=inter_mask,
        )
        tl.store(
            raw_ptr + slot * RAW_STRIDE_K + RAW_STRIDE_KIND + inter_offsets * RAW_STRIDE_I,
            up_acc,
            mask=inter_mask,
        )
        inter = (gate_acc * tl.sigmoid(gate_acc) * up_acc).to(tl.bfloat16)
        tl.store(inter_ptr + slot * INTER_STRIDE_K + inter_offsets * INTER_STRIDE_I, inter, mask=inter_mask)


def _load_fixture(path: Path, device: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load as load_buffer
    from safetensors.torch import load_file

    if len(path.suffixes) >= 2 and path.suffixes[-2:] == [".safetensors", ".gz"]:
        with gzip.open(str(path), "rb") as f:
            raw = f.read()
        return {k: v.to(device) for k, v in load_buffer(raw).items()}
    return load_file(str(path), device=device)


def _metric(ref: torch.Tensor, cand: torch.Tensor) -> dict[str, float | int | list[int]]:
    rf = ref.float().flatten()
    cf = cand.float().flatten()
    diff = rf - cf
    abs_diff = diff.abs()
    ref_norm = torch.linalg.vector_norm(rf).clamp_min(1e-12)
    cand_norm = torch.linalg.vector_norm(cf).clamp_min(1e-12)
    diff_norm = torch.linalg.vector_norm(diff)
    max_abs = float(abs_diff.max())
    max_index = int(abs_diff.argmax())
    return {
        "max_abs": max_abs,
        "mean_abs": float(abs_diff.mean()),
        "rel_l2": float(diff_norm / ref_norm),
        "cosine": float(torch.dot(rf, cf) / (ref_norm * cand_norm)),
        "exact": 1 if max_abs == 0.0 else 0,
        "max_index": [max_index],
    }


def _bench_ms(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _ref_file_for(reference_dir: Path, layer_id: int, prompt_id: int) -> Path:
    path = reference_dir / f"layer_{layer_id:02d}_prompt_{prompt_id:02d}_triton_stage.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"P147 reference not found: {path}")
    return path


def _triton_raw_gateup(
    x: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global_scale: torch.Tensor,
    *,
    block_inter: int,
    block_hidden: int,
    num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for the P155 raw reference kernel")
    top_k = int(gate_up_packed.shape[0])
    slot_ids = torch.arange(top_k, device=x.device, dtype=torch.int32)
    raw = torch.empty((top_k, 2, INTERMEDIATE_SIZE), device=x.device, dtype=torch.float32)
    inter = torch.empty((top_k, INTERMEDIATE_SIZE), device=x.device, dtype=torch.bfloat16)
    grid = (top_k, triton.cdiv(INTERMEDIATE_SIZE, block_inter))
    _p155_gateup_raw_reference_kernel[grid](
        x.contiguous(),
        slot_ids,
        gate_up_packed.contiguous(),
        gate_up_scale.contiguous(),
        gate_up_global_scale.to(device=x.device).contiguous(),
        raw,
        inter,
        gate_up_packed.stride(0),
        gate_up_packed.stride(1),
        gate_up_packed.stride(2),
        gate_up_scale.stride(0),
        gate_up_scale.stride(1),
        gate_up_scale.stride(2),
        raw.stride(0),
        raw.stride(1),
        raw.stride(2),
        inter.stride(0),
        inter.stride(1),
        HIDDEN=HIDDEN_SIZE,
        INTERMEDIATE=INTERMEDIATE_SIZE,
        BLOCK_INTER=block_inter,
        BLOCK_HIDDEN=block_hidden,
        num_warps=num_warps,
    )
    return raw, inter


def _raw_to_inter(raw: torch.Tensor) -> torch.Tensor:
    return (torch.nn.functional.silu(raw[:, 0, :]) * raw[:, 1, :]).to(torch.bfloat16).contiguous()


def _sum_exact(rows: list[dict[str, Any]], key: str) -> int:
    return sum(int(r[key]["exact"]) for r in rows)


def _max_abs(rows: list[dict[str, Any]], key: str) -> float:
    return max(float(r[key]["max_abs"]) for r in rows)


def _variant_diagnosis(rows: list[dict[str, Any]], prefix: str) -> str:
    total = len(rows)
    gate_exact = _sum_exact(rows, f"{prefix}_raw_gate_vs_triton")
    up_exact = _sum_exact(rows, f"{prefix}_raw_up_vs_triton")
    inter_exact = _sum_exact(rows, f"{prefix}_inter_vs_triton")
    if gate_exact == total and up_exact == total and inter_exact == total:
        return "EXACT"
    if gate_exact == total and up_exact == total:
        return "SILU_BF16_BOUNDARY_DRIFT"
    return "RAW_ACCUM_DRIFT"


def main() -> int:
    ap = argparse.ArgumentParser(description="P155 raw gate/up accumulator diagnostics.")
    ap.add_argument("--packed-fixtures", required=True)
    ap.add_argument("--p147-reference-dir", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--gate-block-inter", type=int, default=8)
    ap.add_argument("--gate-block-hidden", type=int, default=256)
    ap.add_argument("--gate-num-warps", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from engine.native_cuda import load_lynn_native_extension

    packed_dir = Path(args.packed_fixtures)
    ref_dir = Path(args.p147_reference_dir) if args.p147_reference_dir else None
    manifest = json.loads((packed_dir / "manifest.json").read_text())

    ext = load_lynn_native_extension(verbose=False)
    for name in (
        "moe_slot_packed_nvfp4_raw_accum_probe",
        "moe_slot_packed_nvfp4_raw_accum_triton_order_probe",
        "moe_slot_packed_nvfp4_inter_probe",
        "moe_slot_packed_nvfp4_inter_triton_order_probe",
    ):
        if not hasattr(ext, name):
            raise RuntimeError(f"native extension lacks {name}")

    print("[p155] Native packed MoE raw gate/up accumulator diagnostics")
    print(f"[p155] packed_fixtures={packed_dir}")
    print(f"[p155] p147_reference_dir={ref_dir or '<none>'}")

    rows: list[dict[str, Any]] = []
    for entry in manifest["fixtures"]:
        fixture_file = entry["fixture_file"]
        layer_id = int(entry["layer_id"])
        prompt_id = int(entry["prompt_id"])
        data = _load_fixture(packed_dir / fixture_file, args.device)

        x = data["hidden_in"].to(torch.bfloat16).view(-1).contiguous()
        gu_packed = data["slot_gate_up_packed"].contiguous()
        gu_scale = data["slot_gate_up_scale"].to(torch.float16).contiguous()
        gu_global = data["slot_gate_up_global_scale"].to(torch.float16).contiguous()

        def triton_raw_fn() -> tuple[torch.Tensor, torch.Tensor]:
            return _triton_raw_gateup(
                x,
                gu_packed,
                gu_scale,
                gu_global,
                block_inter=args.gate_block_inter,
                block_hidden=args.gate_block_hidden,
                num_warps=args.gate_num_warps,
            )

        def native_raw_fn() -> torch.Tensor:
            return ext.moe_slot_packed_nvfp4_raw_accum_probe(x, gu_packed, gu_scale, gu_global)

        def native_inter_fn() -> torch.Tensor:
            return ext.moe_slot_packed_nvfp4_inter_probe(x, gu_packed, gu_scale, gu_global)

        def native_order_raw_fn() -> torch.Tensor:
            return ext.moe_slot_packed_nvfp4_raw_accum_triton_order_probe(x, gu_packed, gu_scale, gu_global)

        def native_order_inter_fn() -> torch.Tensor:
            return ext.moe_slot_packed_nvfp4_inter_triton_order_probe(x, gu_packed, gu_scale, gu_global)

        triton_raw, triton_inter = triton_raw_fn()
        native_raw = native_raw_fn().contiguous()
        native_inter = native_inter_fn().contiguous()
        order_raw = native_order_raw_fn().contiguous()
        order_inter = native_order_inter_fn().contiguous()

        row: dict[str, Any] = {
            "fixture_file": fixture_file,
            "layer_id": layer_id,
            "prompt_id": prompt_id,
            "triton_raw_ms": _bench_ms(triton_raw_fn, warmup=args.warmup, iters=args.iters),
            "native_raw_ms": _bench_ms(native_raw_fn, warmup=args.warmup, iters=args.iters),
            "native_inter_ms": _bench_ms(native_inter_fn, warmup=args.warmup, iters=args.iters),
            "triton_order_raw_ms": _bench_ms(native_order_raw_fn, warmup=args.warmup, iters=args.iters),
            "triton_order_inter_ms": _bench_ms(native_order_inter_fn, warmup=args.warmup, iters=args.iters),
            "native_raw_gate_vs_triton": _metric(triton_raw[:, 0, :], native_raw[:, 0, :]),
            "native_raw_up_vs_triton": _metric(triton_raw[:, 1, :], native_raw[:, 1, :]),
            "native_inter_vs_triton": _metric(triton_inter, native_inter),
            "native_raw_to_inter_vs_native_inter": _metric(native_inter, _raw_to_inter(native_raw)),
            "triton_order_raw_gate_vs_triton": _metric(triton_raw[:, 0, :], order_raw[:, 0, :]),
            "triton_order_raw_up_vs_triton": _metric(triton_raw[:, 1, :], order_raw[:, 1, :]),
            "triton_order_inter_vs_triton": _metric(triton_inter, order_inter),
            "triton_order_raw_to_inter_vs_native_inter": _metric(order_inter, _raw_to_inter(order_raw)),
        }
        if ref_dir is not None:
            ref = _load_fixture(_ref_file_for(ref_dir, layer_id, prompt_id), args.device)
            row["triton_inter_vs_p147"] = _metric(
                ref["triton_inter"].to(torch.bfloat16).contiguous(),
                triton_inter,
            )
        rows.append(row)
        print(
            f"  L{layer_id:02d}/P{prompt_id:02d} "
            f"raw_base=({row['native_raw_gate_vs_triton']['max_abs']:.2e},"
            f"{row['native_raw_up_vs_triton']['max_abs']:.2e}) "
            f"raw_order=({row['triton_order_raw_gate_vs_triton']['max_abs']:.2e},"
            f"{row['triton_order_raw_up_vs_triton']['max_abs']:.2e}) "
            f"inter=({row['native_inter_vs_triton']['max_abs']:.2e},"
            f"{row['triton_order_inter_vs_triton']['max_abs']:.2e})",
            flush=True,
        )

    total = len(rows)
    report = {
        "schema": "lynn-p155-native-packed-moe-gateup-raw-accum-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "packed_fixtures": str(packed_dir),
        "p147_reference_dir": str(ref_dir) if ref_dir is not None else None,
        "kernel_config": {
            "gate_block_inter": args.gate_block_inter,
            "gate_block_hidden": args.gate_block_hidden,
            "gate_num_warps": args.gate_num_warps,
        },
        "total": total,
        "native_diagnosis": _variant_diagnosis(rows, "native"),
        "triton_order_diagnosis": _variant_diagnosis(rows, "triton_order"),
        "native_raw_gate_exact": _sum_exact(rows, "native_raw_gate_vs_triton"),
        "native_raw_up_exact": _sum_exact(rows, "native_raw_up_vs_triton"),
        "native_inter_exact": _sum_exact(rows, "native_inter_vs_triton"),
        "triton_order_raw_gate_exact": _sum_exact(rows, "triton_order_raw_gate_vs_triton"),
        "triton_order_raw_up_exact": _sum_exact(rows, "triton_order_raw_up_vs_triton"),
        "triton_order_inter_exact": _sum_exact(rows, "triton_order_inter_vs_triton"),
        "native_raw_gate_max_abs_max": _max_abs(rows, "native_raw_gate_vs_triton"),
        "native_raw_up_max_abs_max": _max_abs(rows, "native_raw_up_vs_triton"),
        "native_inter_max_abs_max": _max_abs(rows, "native_inter_vs_triton"),
        "triton_order_raw_gate_max_abs_max": _max_abs(rows, "triton_order_raw_gate_vs_triton"),
        "triton_order_raw_up_max_abs_max": _max_abs(rows, "triton_order_raw_up_vs_triton"),
        "triton_order_inter_max_abs_max": _max_abs(rows, "triton_order_inter_vs_triton"),
        "triton_raw_ms_mean": sum(r["triton_raw_ms"] for r in rows) / total if total else None,
        "native_raw_ms_mean": sum(r["native_raw_ms"] for r in rows) / total if total else None,
        "native_inter_ms_mean": sum(r["native_inter_ms"] for r in rows) / total if total else None,
        "triton_order_raw_ms_mean": sum(r["triton_order_raw_ms"] for r in rows) / total if total else None,
        "triton_order_inter_ms_mean": sum(r["triton_order_inter_ms"] for r in rows) / total if total else None,
        "results": rows,
    }
    if ref_dir is not None:
        report["triton_inter_vs_p147_exact"] = _sum_exact(rows, "triton_inter_vs_p147")
        report["triton_inter_vs_p147_max_abs_max"] = _max_abs(rows, "triton_inter_vs_p147")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        "[p155] native="
        f"{report['native_diagnosis']} order={report['triton_order_diagnosis']} "
        f"raw_exact={report['triton_order_raw_gate_exact']}/{total},"
        f"{report['triton_order_raw_up_exact']}/{total} "
        f"inter_exact={report['triton_order_inter_exact']}/{total}"
    )
    print(f"[p155] report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
