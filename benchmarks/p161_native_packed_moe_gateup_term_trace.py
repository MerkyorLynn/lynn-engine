#!/usr/bin/env python3
"""P161 - Per-term gate/up trace for packed-NVFP4 MoE.

P160 localized native-vs-Triton drift to a single 256-hidden block.  P161 goes
one level deeper for a selected `(slot, hidden_block, gate/up, row)` and dumps
the 256 individual FP32 products before reduction.

If terms are exact, the remaining blocker is the reduction tree.  If terms
already differ, the blocker is FP4 decode / scale-global arithmetic lowering.
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
BLOCK_HIDDEN = 256


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
    def _p161_gateup_term_reference_kernel(
        x_ptr,
        gate_up_packed_ptr,
        gate_up_scale_ptr,
        global_scale_ptr,
        terms_ptr,
        PACKED_STRIDE_K: tl.constexpr,
        PACKED_STRIDE_M: tl.constexpr,
        PACKED_STRIDE_N: tl.constexpr,
        SCALE_STRIDE_K: tl.constexpr,
        SCALE_STRIDE_M: tl.constexpr,
        SCALE_STRIDE_G: tl.constexpr,
        SLOT: tl.constexpr,
        ROW: tl.constexpr,
        KIND: tl.constexpr,
        HIDDEN_BLOCK: tl.constexpr,
        INTERMEDIATE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK)
        hidden_offsets = HIDDEN_BLOCK * BLOCK + offsets
        packed_cols = hidden_offsets // 2
        scale_cols = hidden_offsets // 16
        weight_row = ROW + KIND * INTERMEDIATE

        packed_offsets = (
            SLOT * PACKED_STRIDE_K
            + weight_row * PACKED_STRIDE_M
            + packed_cols * PACKED_STRIDE_N
        )
        scale_offsets = (
            SLOT * SCALE_STRIDE_K
            + weight_row * SCALE_STRIDE_M
            + scale_cols * SCALE_STRIDE_G
        )
        packed = tl.load(gate_up_packed_ptr + packed_offsets)
        nibble = tl.where((hidden_offsets & 1) == 0, packed & 0x0F, (packed >> 4) & 0x0F)
        w = _e2m1_from_nibble_fast(nibble)
        scale = tl.load(gate_up_scale_ptr + scale_offsets).to(tl.float32)
        global_scale = tl.load(global_scale_ptr).to(tl.float32)
        x = tl.load(x_ptr + hidden_offsets).to(tl.float32)
        terms = w * (scale / global_scale) * x
        tl.store(terms_ptr + offsets, terms)


def _load_fixture(path: Path, device: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load as load_buffer
    from safetensors.torch import load_file

    if len(path.suffixes) >= 2 and path.suffixes[-2:] == [".safetensors", ".gz"]:
        with gzip.open(str(path), "rb") as f:
            raw = f.read()
        return {k: v.to(device) for k, v in load_buffer(raw).items()}
    return load_file(str(path), device=device)


def _metric(ref: torch.Tensor, cand: torch.Tensor) -> dict[str, Any]:
    rf = ref.float().contiguous()
    cf = cand.float().contiguous()
    diff = rf - cf
    abs_diff = diff.abs()
    ref_norm = torch.linalg.vector_norm(rf.flatten()).clamp_min(1e-12)
    cand_norm = torch.linalg.vector_norm(cf.flatten()).clamp_min(1e-12)
    max_abs = float(abs_diff.max())
    max_index = int(abs_diff.flatten().argmax())
    return {
        "max_abs": max_abs,
        "mean_abs": float(abs_diff.mean()),
        "rel_l2": float(torch.linalg.vector_norm(diff.flatten()) / ref_norm),
        "cosine": float(torch.dot(rf.flatten(), cf.flatten()) / (ref_norm * cand_norm)),
        "exact": 1 if max_abs == 0.0 else 0,
        "max_index": max_index,
        "ref_at_max": float(rf.flatten()[max_index].item()),
        "cand_at_max": float(cf.flatten()[max_index].item()),
    }


def _triton_terms(
    x: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global_scale: torch.Tensor,
    *,
    slot: int,
    row: int,
    kind: int,
    hidden_block: int,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for P161")
    terms = torch.empty((BLOCK_HIDDEN,), device=x.device, dtype=torch.float32)
    _p161_gateup_term_reference_kernel[(1,)](
        x.contiguous(),
        gate_up_packed.contiguous(),
        gate_up_scale.contiguous(),
        gate_up_global_scale.to(device=x.device).contiguous(),
        terms,
        gate_up_packed.stride(0),
        gate_up_packed.stride(1),
        gate_up_packed.stride(2),
        gate_up_scale.stride(0),
        gate_up_scale.stride(1),
        gate_up_scale.stride(2),
        SLOT=slot,
        ROW=row,
        KIND=kind,
        HIDDEN_BLOCK=hidden_block,
        INTERMEDIATE=INTERMEDIATE_SIZE,
        BLOCK=BLOCK_HIDDEN,
        num_warps=8,
    )
    return terms


def _target_from_p160(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    worst = report["worst_fixture"]
    partial_metric = worst["partial_metric"]
    slot, hidden_block, kind, row = [int(x) for x in partial_metric["max_index"]]
    return {
        "fixture_file": worst["fixture_file"],
        "layer_id": int(worst["layer_id"]),
        "prompt_id": int(worst["prompt_id"]),
        "slot": slot,
        "hidden_block": hidden_block,
        "kind": kind,
        "row": row,
        "source_p160_max_abs": float(partial_metric["max_abs"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="P161 per-term packed MoE gate/up trace.")
    ap.add_argument("--packed-fixtures", required=True)
    ap.add_argument("--p160-report", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from engine.native_cuda import load_lynn_native_extension

    packed_dir = Path(args.packed_fixtures)
    target = _target_from_p160(Path(args.p160_report))
    data = _load_fixture(packed_dir / target["fixture_file"], args.device)

    ext = load_lynn_native_extension(verbose=False)
    if not hasattr(ext, "moe_slot_packed_nvfp4_term_trace_probe"):
        raise RuntimeError("native extension lacks moe_slot_packed_nvfp4_term_trace_probe")

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
    native_terms = ext.moe_slot_packed_nvfp4_term_trace_probe(
        x,
        gu_packed,
        gu_scale,
        gu_global,
        target["slot"],
        target["row"],
        target["kind"],
        target["hidden_block"],
    ).contiguous()
    terms_metric = _metric(triton_terms, native_terms)
    diagnosis = "REDUCTION_TREE_DRIFT" if terms_metric["exact"] else "TERM_ARITHMETIC_DRIFT"

    top_diffs = torch.topk((triton_terms - native_terms).abs(), k=8)
    top = []
    for idx_t, val_t in zip(top_diffs.indices.detach().cpu().tolist(), top_diffs.values.detach().cpu().tolist()):
        top.append(
            {
                "offset": int(idx_t),
                "hidden": int(target["hidden_block"] * BLOCK_HIDDEN + idx_t),
                "abs_diff": float(val_t),
                "triton": float(triton_terms[idx_t].item()),
                "native": float(native_terms[idx_t].item()),
            }
        )

    report = {
        "schema": "lynn-p161-native-packed-moe-gateup-term-trace-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "packed_fixtures": str(packed_dir),
        "p160_report": args.p160_report,
        "target": target,
        "terms_vs_triton": terms_metric,
        "diagnosis": diagnosis,
        "top_abs_diffs": top,
        "triton_sum_torch": float(triton_terms.sum().item()),
        "native_sum_torch": float(native_terms.sum().item()),
        "torch_sum_abs_diff": float((triton_terms.sum() - native_terms.sum()).abs().item()),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        f"[p161] diagnosis={diagnosis} "
        f"term_exact={terms_metric['exact']} max_abs={terms_metric['max_abs']:.2e} "
        f"target={target}"
    )
    print(f"[p161] report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
