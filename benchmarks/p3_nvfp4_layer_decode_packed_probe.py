#!/usr/bin/env python3
"""P3-G probe: one decode layer with packed NVFP4 attention + active MoE experts.

P3-A..F proved the individual packed bridge pieces:
  - single packed Linear matvec
  - linear-attention decode projections
  - dual gate/up projections
  - one active expert FFN

P3-G stitches those pieces into a real layer-level decode block. It compares:
  resident slow-dequant NVFP4 layer 0
  vs.
  packed NVFP4 linear-attention projections + packed active MoE experts

This is still a scalar bridge correctness/plumbing probe, not a tensor-core FP4
GEMM benchmark. The resident dequant path remains the oracle.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.full_forward import _rms_norm
from engine.incremental_decode import decode_linear_attn
from engine.loader import load_qwen36_layer
from engine.moe_optimized import moe_forward_decode_optimized
from engine.nvfp4_runtime import PackedNVFP4Linear
from engine.qwen36_linear_attn_block import (
    CONV_DIM,
    CONV_KERNEL,
    HEAD_K_DIM,
    HEAD_V_DIM,
    NUM_V_HEADS,
)
from triton_kernels.nvfp4_linear import nvfp4_dual_matvec_packed


LINEAR_ATTN_WEIGHT_NAMES = [
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.out_proj.weight",
]


def _cfg_from(model_dir: Path) -> tuple[dict[str, Any], list[str]]:
    config = json.loads((model_dir / "config.json").read_text())
    text_config = config["text_config"]
    rope = text_config.get("rope_parameters", {})
    return {
        "hidden_size": text_config["hidden_size"],
        "num_attention_heads": text_config["num_attention_heads"],
        "num_key_value_heads": text_config["num_key_value_heads"],
        "head_dim": text_config["head_dim"],
        "num_experts": text_config["num_experts"],
        "num_experts_per_tok": text_config["num_experts_per_tok"],
        "rope_theta": rope.get("rope_theta", text_config.get("rope_theta", 1e6)),
        "partial_rotary_factor": rope.get(
            "partial_rotary_factor", text_config.get("partial_rotary_factor", 1.0)
        ),
    }, text_config["layer_types"]


def _compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.float().flatten()
    bf = b.float().flatten()
    diff = af - bf
    denom = torch.linalg.vector_norm(bf).clamp_min(1e-12)
    cosine = torch.dot(af, bf) / (
        torch.linalg.vector_norm(af).clamp_min(1e-12)
        * torch.linalg.vector_norm(bf).clamp_min(1e-12)
    )
    return {
        "mean_abs": float(diff.abs().mean()),
        "max_abs": float(diff.abs().max()),
        "rmse": float(torch.sqrt(torch.mean(diff.square()))),
        "rel_l2": float(torch.linalg.vector_norm(diff) / denom),
        "cosine": float(cosine),
        "sign_match": float((torch.signbit(af) == torch.signbit(bf)).float().mean()),
    }


def _bench(fn, *, warmup: int, iters: int) -> float:
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


def _load_packed_linear(v8_dir: Path, layer: int, short_name: str, device: str) -> PackedNVFP4Linear:
    base = f"model.language_model.layers.{layer}.{short_name.removesuffix('.weight')}"
    with safe_open(v8_dir / "model.safetensors", framework="pt", device="cpu") as st:
        return PackedNVFP4Linear.from_safetensors(st, base, name=short_name, device=device)


@dataclass(slots=True)
class PackedExpert:
    expert_id: int
    gate: PackedNVFP4Linear
    up: PackedNVFP4Linear
    down: PackedNVFP4Linear

    @classmethod
    def from_safetensors(cls, v8_dir: Path, layer: int, expert_id: int, device: str) -> "PackedExpert":
        prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert_id}"
        with safe_open(v8_dir / "model.safetensors", framework="pt", device="cpu") as st:
            return cls(
                expert_id=expert_id,
                gate=PackedNVFP4Linear.from_safetensors(
                    st, f"{prefix}.gate_proj", name=f"expert.{expert_id}.gate", device=device
                ),
                up=PackedNVFP4Linear.from_safetensors(
                    st, f"{prefix}.up_proj", name=f"expert.{expert_id}.up", device=device
                ),
                down=PackedNVFP4Linear.from_safetensors(
                    st, f"{prefix}.down_proj", name=f"expert.{expert_id}.down", device=device
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one active expert FFN for a single hidden vector."""
        if x.ndim != 1:
            raise ValueError(f"PackedExpert.forward expects [hidden], got {tuple(x.shape)}")
        gate_out, up_out = nvfp4_dual_matvec_packed(
            x,
            self.gate.weight_packed,
            self.gate.weight_scale,
            self.gate.weight_global_scale,
            self.up.weight_packed,
            self.up.weight_scale,
            self.up.weight_global_scale,
        )
        inter = F.silu(gate_out).to(x.dtype) * up_out.to(x.dtype)
        return self.down(inter).reshape(-1)


@dataclass(slots=True)
class PackedSharedExpert:
    gate: PackedNVFP4Linear
    up: PackedNVFP4Linear
    down: PackedNVFP4Linear

    @classmethod
    def from_safetensors(cls, v8_dir: Path, layer: int, device: str) -> "PackedSharedExpert":
        prefix = f"model.language_model.layers.{layer}.mlp.shared_expert"
        with safe_open(v8_dir / "model.safetensors", framework="pt", device="cpu") as st:
            return cls(
                gate=PackedNVFP4Linear.from_safetensors(
                    st, f"{prefix}.gate_proj", name="shared_expert.gate", device=device
                ),
                up=PackedNVFP4Linear.from_safetensors(
                    st, f"{prefix}.up_proj", name="shared_expert.up", device=device
                ),
                down=PackedNVFP4Linear.from_safetensors(
                    st, f"{prefix}.down_proj", name="shared_expert.down", device=device
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 1:
            raise ValueError(f"PackedSharedExpert.forward expects [hidden], got {tuple(x.shape)}")
        gate_out, up_out = nvfp4_dual_matvec_packed(
            x,
            self.gate.weight_packed,
            self.gate.weight_scale,
            self.gate.weight_global_scale,
            self.up.weight_packed,
            self.up.weight_scale,
            self.up.weight_global_scale,
        )
        inter = F.silu(gate_out).to(x.dtype) * up_out.to(x.dtype)
        return self.down(inter).reshape(-1)


def _route(h: torch.Tensor, w: dict, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    h_flat = h.view(-1, h.shape[-1])
    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(router_logits, cfg["num_experts_per_tok"], dim=-1)
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(h.dtype)
    return routing_weights, expert_indices


def _packed_moe_decode(
    h: torch.Tensor,
    resident_w: dict,
    cfg: dict,
    packed_experts: dict[int, PackedExpert],
    packed_shared: PackedSharedExpert | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """MoE decode path with packed active experts and resident shared expert.

    Router and shared expert intentionally remain resident tensors in P3-G.
    That keeps this milestone focused on whether packed active experts can be
    wired into the real layer route without changing routing semantics.
    """
    if h.shape[0] != 1 or h.shape[1] != 1:
        raise NotImplementedError("P3-G packed MoE only covers B=1,T=1 decode")

    h_flat = h.view(1, h.shape[-1])
    routing_weights, expert_indices = _route(h, resident_w, cfg)
    expert_ids = [int(x) for x in expert_indices[0].tolist()]

    moe_out = torch.zeros_like(h_flat)
    for slot, expert_id in enumerate(expert_ids):
        expert = packed_experts[expert_id]
        ffn_e = expert.forward(h_flat[0]).reshape(1, -1).to(h_flat.dtype)
        moe_out = moe_out + ffn_e * routing_weights[0, slot]

    shared_mode = "none"
    if "mlp.shared_expert.gate_proj.weight" in resident_w:
        if packed_shared is None:
            gate_s = F.linear(h_flat, resident_w["mlp.shared_expert.gate_proj.weight"])
            up_s = F.linear(h_flat, resident_w["mlp.shared_expert.up_proj.weight"])
            shared_ffn = F.linear(
                F.silu(gate_s) * up_s,
                resident_w["mlp.shared_expert.down_proj.weight"],
            )
            shared_mode = "resident"
        else:
            shared_ffn = packed_shared.forward(h_flat[0]).reshape(1, -1).to(h_flat.dtype)
            shared_mode = "packed"
        if "mlp.shared_expert_gate.weight" in resident_w:
            shared_gate = torch.sigmoid(F.linear(h_flat, resident_w["mlp.shared_expert_gate.weight"]))
            shared_ffn = shared_ffn * shared_gate
        moe_out = moe_out + shared_ffn

    trace = {
        "expert_ids": expert_ids,
        "routing_weights": [float(x) for x in routing_weights[0].float().tolist()],
        "unique_experts": sorted(set(expert_ids)),
        "shared_expert": shared_mode,
    }
    return moe_out.view_as(h), trace


def _layer_decode_resident(
    h_new: torch.Tensor,
    w: dict,
    cfg: dict,
    recurrent_state: torch.Tensor,
    conv_state: torch.Tensor,
) -> dict[str, torch.Tensor]:
    residual = h_new
    h_norm = _rms_norm(h_new, w["input_layernorm.weight"])
    attn_out, new_recurrent, new_conv = decode_linear_attn(
        h_norm, w, recurrent_state, conv_state
    )
    h_after_attn = residual + attn_out

    residual = h_after_attn
    h_moe_norm = _rms_norm(h_after_attn, w["post_attention_layernorm.weight"])
    moe_out = moe_forward_decode_optimized(h_moe_norm, w, cfg)
    h_out = residual + moe_out
    return {
        "h_norm": h_norm,
        "attn_out": attn_out,
        "h_after_attn": h_after_attn,
        "h_moe_norm": h_moe_norm,
        "moe_out": moe_out,
        "h_out": h_out,
        "recurrent_state": new_recurrent,
        "conv_state": new_conv,
    }


def _layer_decode_packed(
    h_new: torch.Tensor,
    w: dict,
    cfg: dict,
    packed_experts: dict[int, PackedExpert],
    recurrent_state: torch.Tensor,
    conv_state: torch.Tensor,
    packed_shared: PackedSharedExpert | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    residual = h_new
    h_norm = _rms_norm(h_new, w["input_layernorm.weight"])
    attn_out, new_recurrent, new_conv = decode_linear_attn(
        h_norm, w, recurrent_state, conv_state
    )
    h_after_attn = residual + attn_out

    residual = h_after_attn
    h_moe_norm = _rms_norm(h_after_attn, w["post_attention_layernorm.weight"])
    moe_out, trace = _packed_moe_decode(h_moe_norm, w, cfg, packed_experts, packed_shared)
    h_out = residual + moe_out
    return {
        "h_norm": h_norm,
        "attn_out": attn_out,
        "h_after_attn": h_after_attn,
        "h_moe_norm": h_moe_norm,
        "moe_out": moe_out,
        "h_out": h_out,
        "recurrent_state": new_recurrent,
        "conv_state": new_conv,
    }, trace


def _make_inputs(device: str, dtype: torch.dtype, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    h_new = torch.randn(1, 1, 2048, device=device, dtype=dtype, generator=gen)
    recurrent_state = (
        torch.randn(
            1,
            NUM_V_HEADS,
            HEAD_K_DIM,
            HEAD_V_DIM,
            device=device,
            dtype=torch.float32,
            generator=gen,
        )
        * 0.01
    )
    conv_state = torch.randn(
        1, CONV_DIM, CONV_KERNEL - 1, device=device, dtype=dtype, generator=gen
    ) * 0.01
    return h_new, recurrent_state, conv_state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v8", required=True, help="NVFP4 v8-RTN checkpoint dir")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--seed", type=int, default=20260514)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--cosine-threshold", type=float, default=0.999)
    ap.add_argument("--rel-l2-threshold", type=float, default=0.02)
    ap.add_argument(
        "--pack-shared-expert",
        action="store_true",
        help="Also route shared_expert gate/up/down through packed NVFP4 bridge.",
    )
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    v8_dir = Path(args.v8)
    cfg, layer_types = _cfg_from(v8_dir)
    layer_type = layer_types[args.layer]
    if layer_type != "linear_attention":
        raise ValueError(f"P3-G currently targets a linear_attention layer, got {layer_type!r}")

    resident_w, _ = load_qwen36_layer(
        str(v8_dir),
        args.layer,
        num_experts=cfg["num_experts"],
        device=args.device,
        dequant_dtype=dtype,
    )
    packed_w = copy.copy(resident_w)
    for name in LINEAR_ATTN_WEIGHT_NAMES:
        packed_w[name] = _load_packed_linear(v8_dir, args.layer, name, args.device)

    h_new, recurrent_state, conv_state = _make_inputs(args.device, dtype, args.seed)

    # First run resident to know the router IDs for the oracle path. Then run a
    # packed-attention prefix once to know packed active expert IDs, and load the
    # union. This keeps timing free from safetensors I/O.
    resident = _layer_decode_resident(
        h_new,
        resident_w,
        cfg,
        recurrent_state.clone(),
        conv_state.clone(),
    )
    _, resident_route = _route(resident["h_moe_norm"], resident_w, cfg)

    packed_prefix_w = packed_w
    packed_prefix = _layer_decode_resident(
        h_new,
        packed_prefix_w,
        cfg,
        recurrent_state.clone(),
        conv_state.clone(),
    )
    _, packed_route = _route(packed_prefix["h_moe_norm"], resident_w, cfg)
    active_ids = sorted(
        set(int(x) for x in resident_route[0].tolist())
        | set(int(x) for x in packed_route[0].tolist())
    )
    packed_experts = {
        expert_id: PackedExpert.from_safetensors(v8_dir, args.layer, expert_id, args.device)
        for expert_id in active_ids
    }
    packed_shared = (
        PackedSharedExpert.from_safetensors(v8_dir, args.layer, args.device)
        if args.pack_shared_expert
        else None
    )

    packed, packed_trace = _layer_decode_packed(
        h_new,
        packed_w,
        cfg,
        packed_experts,
        recurrent_state.clone(),
        conv_state.clone(),
        packed_shared,
    )
    torch.cuda.synchronize()

    # Isolate packed MoE on the resident h_moe_norm too. If full-layer parity ever
    # fails due to router sensitivity, this tells us whether active expert FFN
    # wiring is still correct.
    packed_moe_same_input, same_input_trace = _packed_moe_decode(
        resident["h_moe_norm"], resident_w, cfg, packed_experts, packed_shared
    )

    result: dict[str, Any] = {
        "schema_version": "lynn-engine-p3-nvfp4-layer-decode-packed-probe-v1",
        "v8_model": str(v8_dir),
        "layer": args.layer,
        "layer_type": layer_type,
        "device": {
            "name": torch.cuda.get_device_name(args.device),
            "capability": list(torch.cuda.get_device_capability(args.device)),
        },
        "packed_components": {
            "linear_attention": LINEAR_ATTN_WEIGHT_NAMES,
            "active_moe_experts": active_ids,
            "shared_expert": "packed-nvfp4-bridge" if packed_shared else "resident-dequant-oracle",
            "router": "resident-dequant-oracle",
        },
        "router": {
            "resident_expert_ids": [int(x) for x in resident_route[0].tolist()],
            "packed_expert_ids": packed_trace["expert_ids"],
            "same_input_expert_ids": same_input_trace["expert_ids"],
            "topk_exact_match": [int(x) for x in resident_route[0].tolist()] == packed_trace["expert_ids"],
            "active_union_size": len(active_ids),
        },
        "comparisons": {
            "attn_out": _compare(packed["attn_out"], resident["attn_out"]),
            "h_after_attn": _compare(packed["h_after_attn"], resident["h_after_attn"]),
            "moe_out_same_h_moe_norm": _compare(packed_moe_same_input, resident["moe_out"]),
            "moe_out_full_layer": _compare(packed["moe_out"], resident["moe_out"]),
            "final_layer_output": _compare(packed["h_out"], resident["h_out"]),
            "recurrent_state": _compare(packed["recurrent_state"], resident["recurrent_state"]),
            "conv_state": _compare(packed["conv_state"], resident["conv_state"]),
        },
        "timing_ms": {
            "resident_layer_decode": _bench(
                lambda: _layer_decode_resident(
                    h_new,
                    resident_w,
                    cfg,
                    recurrent_state.clone(),
                    conv_state.clone(),
                )["h_out"],
                warmup=args.warmup,
                iters=args.iters,
            ),
            "packed_layer_decode_bridge": _bench(
                lambda: _layer_decode_packed(
                    h_new,
                    packed_w,
                    cfg,
                    packed_experts,
                    recurrent_state.clone(),
                    conv_state.clone(),
                    packed_shared,
                )[0]["h_out"],
                warmup=args.warmup,
                iters=args.iters,
            ),
        },
        "notes": [
            "P3-G is a layer-level scalar bridge: packed linear-attention projections plus packed active MoE experts.",
            "Router and shared expert stay resident-dequant to isolate packed active-weight plumbing.",
            "This is not a Blackwell tensor-core FP4 GEMM path; production throughput belongs to the next native kernel milestone.",
        ],
    }
    result["timing_ms"]["speedup"] = (
        result["timing_ms"]["resident_layer_decode"]
        / result["timing_ms"]["packed_layer_decode_bridge"]
    )

    final_cmp = result["comparisons"]["final_layer_output"]
    same_input_cmp = result["comparisons"]["moe_out_same_h_moe_norm"]
    result["verdict"] = (
        "PASS"
        if (
            final_cmp["cosine"] >= args.cosine_threshold
            and final_cmp["rel_l2"] <= args.rel_l2_threshold
            and same_input_cmp["cosine"] >= args.cosine_threshold
        )
        else "FAIL"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
