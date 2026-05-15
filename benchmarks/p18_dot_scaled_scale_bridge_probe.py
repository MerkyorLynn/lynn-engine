#!/usr/bin/env python3
"""P18: probe whether Lynn per-16 scales can bridge into `tl.dot_scaled`.

P17 proved Triton can execute the raw packed E2M1 gate/up shape very quickly
when scales are already in the e8m0 group32 layout expected by
`tl.dot_scaled`. Lynn's current NVFP4 artifacts are different: packed weights
are quantized with per-16 floating/e4m3-style scales.

This probe intentionally does **not** change the production engine. It tests a
cheap bridge:

* keep the current per-16 packed E2M1 weight codes,
* fold each pair of per-16 scales into one group32 scale,
* round that scale to e8m0 byte form,
* run the selected top-k gate/up dot through `tl.dot_scaled`,
* compare against the current scalar bridge using the exact same packed
  weights and per-16 scales.

If this fails numerically, it is a strong signal that 155+ TPS requires either
an engine-native e8m0/group32 quant artifact or a custom CUDA/CUTLASS path that
accepts Lynn's existing per-16 scale contract.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p10e_packed_active_expert_probe import (  # noqa: E402
    _load_grouped,
    _prefill_to_layer_input,
)
from engine.full_forward import _rms_norm  # noqa: E402
from engine.nvfp4_runtime import _quantize_activation_to_fp4  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.nvfp4_linear import nvfp4_matvec_packed  # noqa: E402


@triton.jit
def _dot_scaled_selected_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    K_PACKED_TOTAL: tl.constexpr,
    N: tl.constexpr,
    GROUPS_TOTAL: tl.constexpr,
    BLOCK_K_PACKED: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kp_base = tl.arange(0, BLOCK_K_PACKED)
    offs_g_base = tl.arange(0, BLOCK_K_PACKED // 16)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for kp0 in range(0, K_PACKED_TOTAL, BLOCK_K_PACKED):
        offs_kp = kp0 + offs_kp_base
        group_start = kp0 // 16
        offs_g = group_start + offs_g_base

        a = tl.load(a_ptr + offs_kp, mask=offs_kp < K_PACKED_TOTAL, other=0)
        b = tl.load(
            b_ptr + offs_n[None, :] * K_PACKED_TOTAL + offs_kp[:, None],
            mask=(offs_n[None, :] < N) & (offs_kp[:, None] < K_PACKED_TOTAL),
            other=0,
        )
        a_s = tl.load(a_scale_ptr + offs_g, mask=offs_g < GROUPS_TOTAL, other=127)
        b_s = tl.load(
            b_scale_ptr + offs_n[:, None] * GROUPS_TOTAL + offs_g[None, :],
            mask=(offs_n[:, None] < N) & (offs_g[None, :] < GROUPS_TOTAL),
            other=127,
        )
        out = tl.dot_scaled(
            a[None, :],
            a_s[None, :],
            "e2m1",
            b,
            b_s,
            "e2m1",
            lhs_k_pack=True,
            rhs_k_pack=True,
        )
        acc += tl.reshape(out, (BLOCK_N,))

    tl.store(c_ptr + offs_n, acc, mask=offs_n < N)


def _bench(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
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


def _fold_pair(scale16: torch.Tensor, mode: str) -> torch.Tensor:
    if scale16.shape[-1] % 2 != 0:
        raise ValueError(f"expected even per-16 scale groups, got {tuple(scale16.shape)}")
    pair = scale16.float().reshape(*scale16.shape[:-1], scale16.shape[-1] // 2, 2).clamp_min(1e-30)
    if mode == "max":
        return pair.max(dim=-1).values
    if mode == "mean":
        return pair.mean(dim=-1)
    if mode == "geom":
        return torch.sqrt(pair[..., 0] * pair[..., 1])
    raise ValueError(f"unknown fold mode {mode!r}")


def _to_e8m0_bytes(scale: torch.Tensor) -> torch.Tensor:
    # P17 empirical contract: byte 127 behaves as scale 1.0 for each operand.
    byte = torch.round(torch.log2(scale.float().clamp_min(1e-30))) + 127.0
    return byte.clamp(0, 255).to(torch.uint8).contiguous()


def _dot_scaled_selected(
    act_packed: torch.Tensor,
    act_scale_e8m0: torch.Tensor,
    selected_packed: torch.Tensor,
    selected_scale_e8m0: torch.Tensor,
    *,
    block_k_packed: int,
    block_n: int,
) -> torch.Tensor:
    n = int(selected_packed.shape[0])
    k_packed = int(selected_packed.shape[1])
    groups = int(selected_scale_e8m0.shape[1])
    out = torch.empty((n,), device=selected_packed.device, dtype=torch.float32)
    _dot_scaled_selected_kernel[(triton.cdiv(n, block_n),)](
        act_packed.contiguous(),
        act_scale_e8m0.contiguous(),
        selected_packed.contiguous(),
        selected_scale_e8m0.contiguous(),
        out,
        K_PACKED_TOTAL=k_packed,
        N=n,
        GROUPS_TOTAL=groups,
        BLOCK_K_PACKED=block_k_packed,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return out


def _silu_inter(raw: torch.Tensor, *, top_k: int) -> torch.Tensor:
    gate, up = raw.reshape(top_k, 1024).chunk(2, dim=1)
    return (F.silu(gate.float()) * up.float()).to(torch.bfloat16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--fold-modes", default="max,mean,geom")
    ap.add_argument("--block-k-packed", type=int, default=256)
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    model_dir = Path(args.model)
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    h_layer, _ = _prefill_to_layer_input(runner, args.layer, args.prompt)
    w = runner.layer_weights[args.layer]
    cfg = runner.layer_cfgs[args.layer]
    h_moe = _rms_norm(h_layer, w["post_attention_layernorm.weight"])
    hidden_2d = h_moe.view(-1, h_moe.shape[-1])[:1].contiguous()

    router_logits = F.linear(hidden_2d, w["mlp.gate.weight"])
    _, expert_indices = torch.topk(router_logits, int(cfg["num_experts_per_tok"]), dim=-1)
    expert_ids = expert_indices[0].to(torch.long)
    top_k = int(expert_ids.numel())

    gate_up_packed, gate_up_scale, gate_up_global = _load_grouped(
        model_dir,
        f"model.language_model.layers.{args.layer}.mlp.experts.gate_up_proj",
        runner.device,
    )
    selected_packed = gate_up_packed[expert_ids].reshape(-1, gate_up_packed.shape[-1]).contiguous()
    selected_scale16 = gate_up_scale[expert_ids].reshape(-1, gate_up_scale.shape[-1]).float().contiguous()
    selected_effective16 = (selected_scale16 / gate_up_global.to(selected_scale16.device).float()).contiguous()

    act_packed, act_scale16 = _quantize_activation_to_fp4(hidden_2d)
    act_packed = act_packed[0].contiguous()
    act_scale16 = act_scale16[0].float().contiguous()

    def scalar_bridge_raw() -> torch.Tensor:
        return nvfp4_matvec_packed(
            hidden_2d[0],
            selected_packed,
            selected_scale16,
            gate_up_global,
            block_m=16,
            block_n=128,
        )

    ref_raw = scalar_bridge_raw()
    ref_inter = _silu_inter(ref_raw, top_k=top_k)
    rows = []
    for mode in [x.strip() for x in args.fold_modes.split(",") if x.strip()]:
        act_s = _to_e8m0_bytes(_fold_pair(act_scale16[None, :], mode)[0])
        weight_s = _to_e8m0_bytes(_fold_pair(selected_effective16, mode))

        def candidate_raw() -> torch.Tensor:
            return _dot_scaled_selected(
                act_packed,
                act_s,
                selected_packed,
                weight_s,
                block_k_packed=args.block_k_packed,
                block_n=args.block_n,
            )

        cand_raw = candidate_raw()
        cand_inter = _silu_inter(cand_raw, top_k=top_k)
        raw_diff = cand_raw.float() - ref_raw.float()
        inter_diff = cand_inter.float() - ref_inter.float()
        rows.append(
            {
                "fold_mode": mode,
                "raw": {
                    "max_abs": float(raw_diff.abs().max().item()),
                    "mean_abs": float(raw_diff.abs().mean().item()),
                    "rel_l2": float(torch.linalg.vector_norm(raw_diff).item() / torch.linalg.vector_norm(ref_raw.float()).item()),
                    "cosine": float(F.cosine_similarity(cand_raw.float(), ref_raw.float(), dim=0).item()),
                },
                "inter": {
                    "max_abs": float(inter_diff.abs().max().item()),
                    "mean_abs": float(inter_diff.abs().mean().item()),
                    "rel_l2": float(torch.linalg.vector_norm(inter_diff).item() / torch.linalg.vector_norm(ref_inter.float()).item()),
                    "cosine": float(F.cosine_similarity(cand_inter.float().flatten(), ref_inter.float().flatten(), dim=0).item()),
                },
                "timing_ms": {
                    "dot_scaled_bridge_raw_ms": _bench(candidate_raw, args.warmup, args.iters),
                },
            }
        )

    result = {
        "schema_version": "lynn-engine-p18-dot-scaled-scale-bridge-probe-v1",
        "model": args.model,
        "layer": args.layer,
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "shape": {
            "top_k": top_k,
            "hidden": int(hidden_2d.shape[-1]),
            "selected_packed": list(selected_packed.shape),
            "scale16": list(selected_scale16.shape),
            "scale32": [int(selected_scale16.shape[0]), int(selected_scale16.shape[1] // 2)],
        },
        "timing_ms": {
            "scalar_bridge_raw_ms": _bench(scalar_bridge_raw, args.warmup, args.iters),
        },
        "rows": rows,
        "pass": bool(any(row["inter"]["cosine"] > 0.995 and row["inter"]["rel_l2"] < 0.05 for row in rows)),
        "notes": [
            "This bridge reuses current per-16 packed codes and only folds scales to group32/e8m0.",
            "A fail means the current artifact cannot be safely consumed by dot_scaled without re-quantization or custom scale handling.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
