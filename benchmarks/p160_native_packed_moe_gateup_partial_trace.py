#!/usr/bin/env python3
"""P160 - Per-hidden-block gate/up partial-sum trace for packed-NVFP4 MoE.

P155/P156 proved the native packed gate/up path drifts before the BF16 inter
store.  This probe splits that FP32 dot product into 256-hidden chunks and
compares Triton reference partial sums against the native Triton-order kernel.

The output answers one narrow question:

* if every 256-hidden partial is exact, the remaining drift is the final
  hidden-block accumulation tree;
* if a partial is already different, the drift starts inside that block's
  FP4 decode / scale / multiply / reduce sequence.
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
    def _p160_gateup_partial_reference_kernel(
        x_ptr,
        expert_ids_ptr,
        gate_up_packed_ptr,
        gate_up_scale_ptr,
        global_scale_ptr,
        partial_ptr,
        PACKED_STRIDE_E: tl.constexpr,
        PACKED_STRIDE_M: tl.constexpr,
        PACKED_STRIDE_N: tl.constexpr,
        SCALE_STRIDE_E: tl.constexpr,
        SCALE_STRIDE_M: tl.constexpr,
        SCALE_STRIDE_G: tl.constexpr,
        PARTIAL_STRIDE_K: tl.constexpr,
        PARTIAL_STRIDE_B: tl.constexpr,
        PARTIAL_STRIDE_KIND: tl.constexpr,
        PARTIAL_STRIDE_I: tl.constexpr,
        HIDDEN: tl.constexpr,
        INTERMEDIATE: tl.constexpr,
        BLOCK_INTER: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
    ):
        slot = tl.program_id(0)
        block_i = tl.program_id(1)
        block_h = tl.program_id(2)
        expert = tl.load(expert_ids_ptr + slot)

        inter_offsets = block_i * BLOCK_INTER + tl.arange(0, BLOCK_INTER)
        inter_mask = inter_offsets < INTERMEDIATE
        h_offsets = block_h * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
        col_mask = h_offsets < HIDDEN
        packed_cols = h_offsets // 2
        scale_cols = h_offsets // 16

        x = tl.load(x_ptr + h_offsets, mask=col_mask, other=0.0).to(tl.float32)
        global_scale = tl.load(global_scale_ptr).to(tl.float32)

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

        mask = inter_mask[:, None] & col_mask[None, :]
        gate_packed = tl.load(gate_up_packed_ptr + gate_packed_offsets, mask=mask, other=0)
        up_packed = tl.load(gate_up_packed_ptr + up_packed_offsets, mask=mask, other=0)
        gate_nibble = tl.where(
            (h_offsets[None, :] & 1) == 0,
            gate_packed & 0x0F,
            (gate_packed >> 4) & 0x0F,
        )
        up_nibble = tl.where(
            (h_offsets[None, :] & 1) == 0,
            up_packed & 0x0F,
            (up_packed >> 4) & 0x0F,
        )
        gate_w = _e2m1_from_nibble_fast(gate_nibble)
        up_w = _e2m1_from_nibble_fast(up_nibble)
        gate_scale = tl.load(gate_up_scale_ptr + gate_scale_offsets, mask=mask, other=0.0).to(tl.float32)
        up_scale = tl.load(gate_up_scale_ptr + up_scale_offsets, mask=mask, other=0.0).to(tl.float32)
        gate_partial = tl.sum(gate_w * (gate_scale / global_scale) * x[None, :], axis=1)
        up_partial = tl.sum(up_w * (up_scale / global_scale) * x[None, :], axis=1)

        base = slot * PARTIAL_STRIDE_K + block_h * PARTIAL_STRIDE_B + inter_offsets * PARTIAL_STRIDE_I
        tl.store(partial_ptr + base, gate_partial, mask=inter_mask)
        tl.store(partial_ptr + base + PARTIAL_STRIDE_KIND, up_partial, mask=inter_mask)


def _load_fixture(path: Path, device: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load as load_buffer
    from safetensors.torch import load_file

    if len(path.suffixes) >= 2 and path.suffixes[-2:] == [".safetensors", ".gz"]:
        with gzip.open(str(path), "rb") as f:
            raw = f.read()
        return {k: v.to(device) for k, v in load_buffer(raw).items()}
    return load_file(str(path), device=device)


def _ref_file_for(reference_dir: Path, layer_id: int, prompt_id: int) -> Path:
    path = reference_dir / f"layer_{layer_id:02d}_prompt_{prompt_id:02d}_triton_stage.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"P147 reference not found: {path}")
    return path


def _left_fold_sum_hidden_blocks(partials: torch.Tensor) -> torch.Tensor:
    acc = partials[:, 0, :, :].clone()
    for block_idx in range(1, partials.shape[1]):
        acc = acc + partials[:, block_idx, :, :]
    return acc.contiguous()


def _raw_to_inter(raw: torch.Tensor) -> torch.Tensor:
    return (torch.nn.functional.silu(raw[:, 0, :]) * raw[:, 1, :]).to(torch.bfloat16).contiguous()


def _metric(ref: torch.Tensor, cand: torch.Tensor) -> dict[str, Any]:
    rf = ref.float().contiguous()
    cf = cand.float().contiguous()
    diff = rf - cf
    abs_diff = diff.abs()
    ref_norm = torch.linalg.vector_norm(rf.flatten()).clamp_min(1e-12)
    cand_norm = torch.linalg.vector_norm(cf.flatten()).clamp_min(1e-12)
    max_abs = float(abs_diff.max())
    flat_index = int(abs_diff.flatten().argmax())
    coord_reversed: list[int] = []
    rem = flat_index
    for size in reversed(abs_diff.shape):
        coord_reversed.append(rem % int(size))
        rem //= int(size)
    coord = list(reversed(coord_reversed))
    return {
        "max_abs": max_abs,
        "mean_abs": float(abs_diff.mean()),
        "rel_l2": float(torch.linalg.vector_norm(diff.flatten()) / ref_norm),
        "cosine": float(torch.dot(rf.flatten(), cf.flatten()) / (ref_norm * cand_norm)),
        "exact": 1 if max_abs == 0.0 else 0,
        "max_index": coord,
        "ref_at_max": float(rf[tuple(coord)].item()),
        "cand_at_max": float(cf[tuple(coord)].item()),
    }


def _triton_partial_gateup(
    x: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global_scale: torch.Tensor,
    *,
    block_inter: int,
    block_hidden: int,
    num_warps: int,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for the P160 partial reference kernel")
    top_k = int(gate_up_packed.shape[0])
    hidden_blocks = triton.cdiv(HIDDEN_SIZE, block_hidden)
    slot_ids = torch.arange(top_k, device=x.device, dtype=torch.int32)
    partial = torch.empty((top_k, hidden_blocks, 2, INTERMEDIATE_SIZE), device=x.device, dtype=torch.float32)
    grid = (top_k, triton.cdiv(INTERMEDIATE_SIZE, block_inter), hidden_blocks)
    _p160_gateup_partial_reference_kernel[grid](
        x.contiguous(),
        slot_ids,
        gate_up_packed.contiguous(),
        gate_up_scale.contiguous(),
        gate_up_global_scale.to(device=x.device).contiguous(),
        partial,
        gate_up_packed.stride(0),
        gate_up_packed.stride(1),
        gate_up_packed.stride(2),
        gate_up_scale.stride(0),
        gate_up_scale.stride(1),
        gate_up_scale.stride(2),
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        partial.stride(3),
        HIDDEN=HIDDEN_SIZE,
        INTERMEDIATE=INTERMEDIATE_SIZE,
        BLOCK_INTER=block_inter,
        BLOCK_HIDDEN=block_hidden,
        num_warps=num_warps,
    )
    return partial


def _trace_worst_vector(ref: torch.Tensor, cand: torch.Tensor, coord: list[int]) -> dict[str, Any]:
    slot, _, kind, row = coord
    ref_vals = ref[slot, :, kind, row].detach().cpu().tolist()
    cand_vals = cand[slot, :, kind, row].detach().cpu().tolist()
    return {
        "slot": slot,
        "kind": "gate" if kind == 0 else "up",
        "row": row,
        "triton_partials": ref_vals,
        "native_partials": cand_vals,
        "diff_partials": [float(a - b) for a, b in zip(ref_vals, cand_vals)],
    }


def _fixture_entries(manifest: dict[str, Any], *, max_fixtures: int, fixture_substring: str | None) -> list[dict[str, Any]]:
    entries = list(manifest["fixtures"])
    if fixture_substring:
        entries = [entry for entry in entries if fixture_substring in entry["fixture_file"]]
    if max_fixtures:
        entries = entries[:max_fixtures]
    return entries


def main() -> int:
    ap = argparse.ArgumentParser(description="P160 per-hidden-block gate/up partial-sum trace.")
    ap.add_argument("--packed-fixtures", required=True)
    ap.add_argument("--p147-reference-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-fixtures", type=int, default=0, help="0 means all selected fixtures.")
    ap.add_argument("--fixture-substring", default=None, help="Optional substring filter, e.g. layer_28_prompt_00.")
    ap.add_argument("--gate-block-inter", type=int, default=8)
    ap.add_argument("--gate-block-hidden", type=int, default=256)
    ap.add_argument("--gate-num-warps", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from engine.native_cuda import load_lynn_native_extension

    packed_dir = Path(args.packed_fixtures)
    ref_dir = Path(args.p147_reference_dir)
    manifest = json.loads((packed_dir / "manifest.json").read_text())
    entries = _fixture_entries(
        manifest,
        max_fixtures=args.max_fixtures,
        fixture_substring=args.fixture_substring,
    )
    if not entries:
        raise RuntimeError("No fixtures selected for P160")

    ext = load_lynn_native_extension(verbose=False)
    for name in (
        "moe_slot_packed_nvfp4_partial_accum_triton_order_probe",
        "moe_slot_packed_nvfp4_raw_accum_triton_order_probe",
    ):
        if not hasattr(ext, name):
            raise RuntimeError(f"native extension lacks {name}")

    print("[p160] Per-hidden-block packed MoE gate/up partial-sum trace")
    print(f"[p160] packed_fixtures={packed_dir}")
    print(f"[p160] p147_reference_dir={ref_dir}")
    print(f"[p160] selected={len(entries)}")

    rows: list[dict[str, Any]] = []
    for entry in entries:
        fixture_file = entry["fixture_file"]
        layer_id = int(entry["layer_id"])
        prompt_id = int(entry["prompt_id"])
        data = _load_fixture(packed_dir / fixture_file, args.device)
        ref = _load_fixture(_ref_file_for(ref_dir, layer_id, prompt_id), args.device)

        x = data["hidden_in"].to(torch.bfloat16).view(-1).contiguous()
        gu_packed = data["slot_gate_up_packed"].contiguous()
        gu_scale = data["slot_gate_up_scale"].to(torch.float16).contiguous()
        gu_global = data["slot_gate_up_global_scale"].to(torch.float16).contiguous()

        triton_partial = _triton_partial_gateup(
            x,
            gu_packed,
            gu_scale,
            gu_global,
            block_inter=args.gate_block_inter,
            block_hidden=args.gate_block_hidden,
            num_warps=args.gate_num_warps,
        ).contiguous()
        native_partial = ext.moe_slot_packed_nvfp4_partial_accum_triton_order_probe(
            x, gu_packed, gu_scale, gu_global
        ).contiguous()
        native_raw = ext.moe_slot_packed_nvfp4_raw_accum_triton_order_probe(
            x, gu_packed, gu_scale, gu_global
        ).contiguous()

        triton_raw_from_partial = _left_fold_sum_hidden_blocks(triton_partial)
        native_raw_from_partial = _left_fold_sum_hidden_blocks(native_partial)
        triton_inter_from_partial = _raw_to_inter(triton_raw_from_partial)
        native_inter_from_partial = _raw_to_inter(native_raw_from_partial)

        partial_metric = _metric(triton_partial, native_partial)
        block_metrics = []
        for block_idx in range(triton_partial.shape[1]):
            block_metric = _metric(triton_partial[:, block_idx, :, :], native_partial[:, block_idx, :, :])
            block_metric["hidden_block"] = block_idx
            block_metrics.append(block_metric)
        worst_block = max(block_metrics, key=lambda item: float(item["max_abs"]))

        row = {
            "fixture_file": fixture_file,
            "layer_id": layer_id,
            "prompt_id": prompt_id,
            "partial_vs_triton": partial_metric,
            "block_metrics": block_metrics,
            "worst_block": worst_block,
            "worst_vector": _trace_worst_vector(triton_partial, native_partial, partial_metric["max_index"]),
            "triton_partial_inter_vs_p147": _metric(
                ref["triton_inter"].to(torch.bfloat16).contiguous(),
                triton_inter_from_partial,
            ),
            "native_partial_inter_vs_p147": _metric(
                ref["triton_inter"].to(torch.bfloat16).contiguous(),
                native_inter_from_partial,
            ),
            "native_partial_sum_vs_native_raw": _metric(native_raw, native_raw_from_partial),
            "triton_partial_sum_vs_native_raw": _metric(native_raw, triton_raw_from_partial),
        }
        rows.append(row)
        print(
            f"  L{layer_id:02d}/P{prompt_id:02d} "
            f"partial_max={partial_metric['max_abs']:.2e} "
            f"worst_block={worst_block['hidden_block']} "
            f"native_inter_vs_p147={row['native_partial_inter_vs_p147']['max_abs']:.2e}",
            flush=True,
        )

    exact_count = sum(int(row["partial_vs_triton"]["exact"]) for row in rows)
    max_partial = max(float(row["partial_vs_triton"]["max_abs"]) for row in rows)
    worst_row = max(rows, key=lambda row: float(row["partial_vs_triton"]["max_abs"]))
    if exact_count == len(rows):
        diagnosis = "FINAL_HIDDEN_BLOCK_ACCUMULATION_DRIFT"
    else:
        diagnosis = "WITHIN_HIDDEN_BLOCK_DRIFT"

    report = {
        "schema": "lynn-p160-native-packed-moe-gateup-partial-trace-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "packed_fixtures": str(packed_dir),
        "p147_reference_dir": str(ref_dir),
        "kernel_config": {
            "gate_block_inter": args.gate_block_inter,
            "gate_block_hidden": args.gate_block_hidden,
            "gate_num_warps": args.gate_num_warps,
        },
        "selected": len(rows),
        "partial_exact": exact_count,
        "partial_max_abs_max": max_partial,
        "diagnosis": diagnosis,
        "worst_fixture": {
            "fixture_file": worst_row["fixture_file"],
            "layer_id": worst_row["layer_id"],
            "prompt_id": worst_row["prompt_id"],
            "partial_metric": worst_row["partial_vs_triton"],
            "worst_block": worst_row["worst_block"],
            "worst_vector": worst_row["worst_vector"],
        },
        "results": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        f"[p160] diagnosis={diagnosis} partial_exact={exact_count}/{len(rows)} "
        f"max_abs={max_partial:.2e}"
    )
    print(f"[p160] report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
