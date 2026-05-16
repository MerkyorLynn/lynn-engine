#!/usr/bin/env python3
"""P54: stronger e8m0/group32 scale-search probe for vendor-layout feasibility.

P18-B tested a naive BF16 -> E2M1/e8m0 group32 re-quantization and failed
quality. P54 asks a stricter question before we commit fully to a custom
per-16 kernel:

* If we search nearby e8m0 scale exponents per group32, does the official-stack
  layout become accurate enough?
* If even an activation-aware dot upper bound fails, the scale contract itself
  is the blocker and Lynn needs a per-16 grouped native-FP4 kernel.

This is an offline feasibility probe. It does not change serving code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p10e_packed_active_expert_probe import _prefill_to_layer_input  # noqa: E402
from benchmarks.p18_dot_scaled_scale_bridge_probe import _bench, _dot_scaled_selected, _silu_inter  # noqa: E402
from benchmarks.p18b_dot_scaled_e8m0_requant_probe import _quantize_e2m1_e8m0_group32  # noqa: E402
from engine.full_forward import _rms_norm  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


_E2M1_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _base_exponent(grouped: torch.Tensor) -> torch.Tensor:
    max_abs = grouped.abs().amax(dim=-1).clamp_min(1e-30)
    return torch.round(torch.log2(max_abs / 6.0)).clamp(-126, 127)


def _pack_with_exponent(x: torch.Tensor, exponent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack `[rows,K]` with pre-selected e8m0 exponents `[rows,K//32]`."""
    if x.ndim != 2:
        raise ValueError(f"x must be [rows,K], got {tuple(x.shape)}")
    if x.shape[1] % 32 != 0:
        raise ValueError(f"K must be divisible by 32, got {x.shape[1]}")
    rows, k = x.shape
    groups = k // 32
    if tuple(exponent.shape) != (rows, groups):
        raise ValueError(f"exponent must be {(rows, groups)}, got {tuple(exponent.shape)}")

    table = _E2M1_TABLE.to(x.device)
    grouped = x.float().reshape(rows, groups, 32)
    scale = torch.pow(torch.full_like(exponent.float(), 2.0), exponent.float()).clamp_min(1e-30)
    normalized = grouped.abs() / scale.unsqueeze(-1)
    mag = torch.argmin((normalized.unsqueeze(-1) - table.view(1, 1, 1, -1)).abs(), dim=-1)
    sign = (grouped < 0).to(torch.uint8) * 8
    codes = (mag.to(torch.uint8) | sign).reshape(rows, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    scale_bytes = (exponent + 127).clamp(0, 255).to(torch.uint8).contiguous()
    return packed, scale_bytes


def _search_exponents_recon(
    x: torch.Tensor,
    offsets: torch.Tensor,
    *,
    chunk_rows: int,
) -> torch.Tensor:
    """Choose e8m0 exponents by minimizing per-group reconstruction error."""
    rows, k = x.shape
    groups = k // 32
    table = _E2M1_TABLE.to(x.device)
    out = torch.empty((rows, groups), device=x.device, dtype=torch.float32)
    offsets = offsets.to(x.device).float()

    for start in range(0, rows, chunk_rows):
        chunk = x[start : start + chunk_rows].float().reshape(-1, groups, 32)
        base = _base_exponent(chunk)
        candidates = (base.unsqueeze(-1) + offsets.view(1, 1, -1)).clamp(-126, 127)
        scale = torch.pow(torch.full_like(candidates, 2.0), candidates).clamp_min(1e-30)
        normalized = chunk.abs().unsqueeze(2) / scale.unsqueeze(-1)
        mag = torch.argmin((normalized.unsqueeze(-1) - table.view(1, 1, 1, 1, -1)).abs(), dim=-1)
        recon = table[mag] * scale.unsqueeze(-1)
        recon = torch.where(chunk.unsqueeze(2) < 0, -recon, recon)
        err = torch.mean((recon - chunk.unsqueeze(2)) ** 2, dim=-1)
        best = torch.argmin(err, dim=-1)
        out[start : start + chunk.shape[0]] = torch.gather(candidates, 2, best.unsqueeze(-1)).squeeze(-1)
    return out


def _search_exponents_dot_upper(
    weight: torch.Tensor,
    hidden: torch.Tensor,
    offsets: torch.Tensor,
    *,
    chunk_rows: int,
) -> torch.Tensor:
    """Activation-aware upper-bound exponent search for selected weight rows.

    This is intentionally stronger than a deployable static artifact: for each
    group32 it picks the exponent that best preserves the current dot
    contribution against the current hidden vector. If this still fails, a
    static e8m0/group32 artifact is very unlikely to work without retraining.
    """
    rows, k = weight.shape
    groups = k // 32
    table = _E2M1_TABLE.to(weight.device)
    hidden_g = hidden.float().reshape(groups, 32)
    out = torch.empty((rows, groups), device=weight.device, dtype=torch.float32)
    offsets = offsets.to(weight.device).float()

    for start in range(0, rows, chunk_rows):
        chunk = weight[start : start + chunk_rows].float().reshape(-1, groups, 32)
        base = _base_exponent(chunk)
        candidates = (base.unsqueeze(-1) + offsets.view(1, 1, -1)).clamp(-126, 127)
        scale = torch.pow(torch.full_like(candidates, 2.0), candidates).clamp_min(1e-30)
        normalized = chunk.abs().unsqueeze(2) / scale.unsqueeze(-1)
        mag = torch.argmin((normalized.unsqueeze(-1) - table.view(1, 1, 1, 1, -1)).abs(), dim=-1)
        recon = table[mag] * scale.unsqueeze(-1)
        recon = torch.where(chunk.unsqueeze(2) < 0, -recon, recon)
        ref_dot = torch.sum(chunk * hidden_g.unsqueeze(0), dim=-1)
        cand_dot = torch.sum(recon * hidden_g.view(1, groups, 1, 32), dim=-1)
        err = (cand_dot - ref_dot.unsqueeze(-1)).abs()
        best = torch.argmin(err, dim=-1)
        out[start : start + chunk.shape[0]] = torch.gather(candidates, 2, best.unsqueeze(-1)).squeeze(-1)
    return out


def _metrics(cand_raw: torch.Tensor, ref_raw: torch.Tensor, *, top_k: int) -> dict[str, dict[str, float]]:
    ref_inter = _silu_inter(ref_raw, top_k=top_k)
    cand_inter = _silu_inter(cand_raw, top_k=top_k)
    raw_diff = cand_raw.float() - ref_raw.float()
    inter_diff = cand_inter.float() - ref_inter.float()
    return {
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
    }


def _run_layer(
    runner: LynnIncrementalRunner,
    *,
    layer: int,
    prompt: str,
    offsets: torch.Tensor,
    chunk_rows: int,
    block_k_packed: int,
    block_n: int,
    warmup: int,
    iters: int,
) -> dict:
    h_layer, _ = _prefill_to_layer_input(runner, layer, prompt)
    w = runner.layer_weights[layer]
    cfg = runner.layer_cfgs[layer]
    h_moe = _rms_norm(h_layer, w["post_attention_layernorm.weight"])
    hidden_2d = h_moe.view(-1, h_moe.shape[-1])[:1].contiguous()
    hidden = hidden_2d[0]

    router_logits = F.linear(hidden_2d, w["mlp.gate.weight"])
    _, expert_indices = torch.topk(router_logits, int(cfg["num_experts_per_tok"]), dim=-1)
    expert_ids = expert_indices[0].to(torch.long)
    top_k = int(expert_ids.numel())

    gate_up = w["mlp.experts.gate_up_proj"][expert_ids].contiguous()
    selected_weight = gate_up.reshape(-1, gate_up.shape[-1]).contiguous()

    ref_raw = torch.matmul(selected_weight.float(), hidden.float())

    act_naive_packed, act_naive_scale = _quantize_e2m1_e8m0_group32(hidden_2d)
    weight_naive_packed, weight_naive_scale = _quantize_e2m1_e8m0_group32(selected_weight)
    act_search_exp = _search_exponents_recon(hidden_2d, offsets, chunk_rows=chunk_rows)
    weight_recon_exp = _search_exponents_recon(selected_weight, offsets, chunk_rows=chunk_rows)
    weight_dot_exp = _search_exponents_dot_upper(selected_weight, hidden, offsets, chunk_rows=chunk_rows)
    act_search_packed, act_search_scale = _pack_with_exponent(hidden_2d, act_search_exp)
    weight_recon_packed, weight_recon_scale = _pack_with_exponent(selected_weight, weight_recon_exp)
    weight_dot_packed, weight_dot_scale = _pack_with_exponent(selected_weight, weight_dot_exp)

    def run_dot(act_packed: torch.Tensor, act_scale: torch.Tensor, weight_packed: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
        return _dot_scaled_selected(
            act_packed[0].contiguous(),
            act_scale[0].contiguous(),
            weight_packed,
            weight_scale,
            block_k_packed=block_k_packed,
            block_n=block_n,
        )

    def naive_raw() -> torch.Tensor:
        return run_dot(act_naive_packed, act_naive_scale, weight_naive_packed, weight_naive_scale)

    def recon_raw() -> torch.Tensor:
        return run_dot(act_search_packed, act_search_scale, weight_recon_packed, weight_recon_scale)

    def dot_upper_raw() -> torch.Tensor:
        return run_dot(act_search_packed, act_search_scale, weight_dot_packed, weight_dot_scale)

    rows = {}
    for name, fn in [
        ("naive_maxabs", naive_raw),
        ("search_recon_mse", recon_raw),
        ("search_dot_upper_bound", dot_upper_raw),
    ]:
        cand = fn()
        m = _metrics(cand, ref_raw, top_k=top_k)
        m["timing_ms"] = _bench(fn, warmup, iters)
        m["pass"] = bool(m["inter"]["cosine"] > 0.995 and m["inter"]["rel_l2"] < 0.08)
        rows[name] = m

    return {
        "layer": layer,
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "shape": {
            "top_k": top_k,
            "hidden": int(hidden_2d.shape[-1]),
            "selected_weight": list(selected_weight.shape),
        },
        "offsets": [int(x) for x in offsets.cpu().tolist()],
        "methods": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="28")
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--offsets", default="-4,-3,-2,-1,0,1,2,3,4")
    ap.add_argument("--chunk-rows", type=int, default=512)
    ap.add_argument("--block-k-packed", type=int, default=256)
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=80)
    args = ap.parse_args()

    layers = _parse_ints(args.layers)
    offsets = torch.tensor(_parse_ints(args.offsets), dtype=torch.float32, device="cuda")
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    layer_rows = [
        _run_layer(
            runner,
            layer=layer,
            prompt=args.prompt,
            offsets=offsets,
            chunk_rows=args.chunk_rows,
            block_k_packed=args.block_k_packed,
            block_n=args.block_n,
            warmup=args.warmup,
            iters=args.iters,
        )
        for layer in layers
    ]

    best_methods = []
    for row in layer_rows:
        best = max(row["methods"].items(), key=lambda item: item[1]["inter"]["cosine"])
        best_methods.append(
            {
                "layer": row["layer"],
                "best_method": best[0],
                "best_inter_cosine": best[1]["inter"]["cosine"],
                "best_inter_rel_l2": best[1]["inter"]["rel_l2"],
                "best_pass": best[1]["pass"],
            }
        )
    result = {
        "schema_version": "lynn-engine-p54-e8m0-scale-search-probe-v1",
        "model": args.model,
        "layers": layers,
        "prompt": args.prompt,
        "layer_results": layer_rows,
        "summary": {
            "all_layers_pass_best": bool(all(x["best_pass"] for x in best_methods)),
            "best_methods": best_methods,
        },
        "notes": [
            "search_recon_mse is a deployable-style static reconstruction search.",
            "search_dot_upper_bound is activation-aware and should be treated as an optimistic upper bound, not a production artifact.",
            "If search_dot_upper_bound fails, e8m0/group32 is unlikely to be enough for Lynn's current 27B artifact without retraining or a custom per-16 kernel.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
