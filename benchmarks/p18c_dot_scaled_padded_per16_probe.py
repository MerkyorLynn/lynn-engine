#!/usr/bin/env python3
"""P18-C: exact per-16 grouping via padded group32 `tl.dot_scaled`.

`tl.dot_scaled` uses e8m0 scales with group size 32. Lynn's current artifact
uses per-16 scales. P18-A folded two per-16 groups into one group32 group and
failed numerically. This probe tries a different bridge:

* expand each original per-16 group into a group32 group,
* place the original 16 FP4 values in the first half,
* pad the second half with zeros,
* keep one scale per original per-16 group, rounded to e8m0.

This doubles K and therefore compute/memory for the affected packed matrices,
but it removes the scale-pair folding error. If it passes, it is a possible
intermediate route to native tensor cores while preserving per-16 grouping.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p10e_packed_active_expert_probe import _load_grouped, _prefill_to_layer_input  # noqa: E402
from benchmarks.p18_dot_scaled_scale_bridge_probe import _bench, _dot_scaled_selected, _silu_inter, _to_e8m0_bytes  # noqa: E402
from engine.full_forward import _rms_norm  # noqa: E402
from engine.nvfp4_runtime import _quantize_activation_to_fp4  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.nvfp4_linear import nvfp4_matvec_packed  # noqa: E402


def _expand_packed_per16_to_group32(packed: torch.Tensor) -> torch.Tensor:
    """Expand `[rows,K/2]` per-16 packed bytes to padded group32 bytes."""
    if packed.shape[-1] % 8 != 0:
        raise ValueError(f"packed bytes must be divisible by 8, got {tuple(packed.shape)}")
    rows = packed.shape[0]
    groups16 = packed.shape[1] // 8
    src = packed.reshape(rows, groups16, 8)
    out = torch.zeros((rows, groups16, 16), device=packed.device, dtype=torch.uint8)
    out[:, :, :8] = src
    return out.reshape(rows, groups16 * 16).contiguous()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
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
    act_packed = act_packed.contiguous()
    act_scale16 = act_scale16.float().contiguous()

    expanded_weight = _expand_packed_per16_to_group32(selected_packed)
    expanded_act = _expand_packed_per16_to_group32(act_packed)[0]
    weight_scale_e8m0 = _to_e8m0_bytes(selected_effective16)
    act_scale_e8m0 = _to_e8m0_bytes(act_scale16)[0]

    def scalar_bridge_raw() -> torch.Tensor:
        return nvfp4_matvec_packed(
            hidden_2d[0],
            selected_packed,
            selected_scale16,
            gate_up_global,
            block_m=16,
            block_n=128,
        )

    def padded_dot_scaled_raw() -> torch.Tensor:
        return _dot_scaled_selected(
            expanded_act,
            act_scale_e8m0,
            expanded_weight,
            weight_scale_e8m0,
            block_k_packed=args.block_k_packed,
            block_n=args.block_n,
        )

    ref_raw = scalar_bridge_raw()
    cand_raw = padded_dot_scaled_raw()
    ref_inter = _silu_inter(ref_raw, top_k=top_k)
    cand_inter = _silu_inter(cand_raw, top_k=top_k)
    raw_diff = cand_raw.float() - ref_raw.float()
    inter_diff = cand_inter.float() - ref_inter.float()
    result = {
        "schema_version": "lynn-engine-p18c-dot-scaled-padded-per16-probe-v1",
        "model": args.model,
        "layer": args.layer,
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "shape": {
            "top_k": top_k,
            "selected_packed": list(selected_packed.shape),
            "expanded_weight": list(expanded_weight.shape),
            "expanded_act": list(expanded_act.shape),
            "scale_groups": list(weight_scale_e8m0.shape),
        },
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
            "scalar_bridge_raw_ms": _bench(scalar_bridge_raw, args.warmup, args.iters),
            "padded_dot_scaled_raw_ms": _bench(padded_dot_scaled_raw, args.warmup, args.iters),
        },
        "pass": bool(
            F.cosine_similarity(cand_inter.float().flatten(), ref_inter.float().flatten(), dim=0).item() > 0.995
            and (torch.linalg.vector_norm(inter_diff).item() / torch.linalg.vector_norm(ref_inter.float()).item()) < 0.05
        ),
        "notes": [
            "This keeps current per-16 packed codes but doubles K by padding each per-16 group to group32.",
            "The remaining error is e4m3/float scale to e8m0 power-of-two rounding, not group folding.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
