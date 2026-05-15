"""Packed NVFP4 MoE decode path."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.nvfp4_runtime import dual_scalar_bridge
from triton_kernels.nvfp4_moe import (
    nvfp4_grouped_down_weighted_sum,
    nvfp4_grouped_gate_up_silu,
)


def moe_forward_decode_packed_nvfp4(h: torch.Tensor, w: dict, cfg: dict) -> torch.Tensor:
    """Decode-only MoE using packed NVFP4 expert weights.

    Router and shared expert stay on the existing BF16 resident path for now;
    active routed experts consume packed NVFP4 weights directly. This is the
    first production-shaped bridge from P10-H into the resident runner.
    """
    if h.shape[1] != 1:
        raise NotImplementedError("packed NVFP4 MoE path is decode-only")
    required = (
        "mlp.experts._gate_up_packed",
        "mlp.experts._gate_up_scale",
        "mlp.experts._gate_up_global_scale",
        "mlp.experts._down_packed",
        "mlp.experts._down_scale",
        "mlp.experts._down_global_scale",
    )
    missing = [key for key in required if key not in w]
    if missing:
        raise KeyError(f"packed NVFP4 MoE aliases missing: {missing}")

    h_flat = h.reshape(-1, h.shape[-1])
    if h_flat.shape[0] != 1:
        raise NotImplementedError("packed NVFP4 MoE path currently supports batch=1")

    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(
        router_logits,
        int(cfg["num_experts_per_tok"]),
        dim=-1,
    )
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32)[0]
    expert_ids = expert_indices[0].to(torch.long)
    hidden = h_flat[0]

    inter = nvfp4_grouped_gate_up_silu(
        hidden,
        expert_ids,
        w["mlp.experts._gate_up_packed"],
        w["mlp.experts._gate_up_scale"],
        w["mlp.experts._gate_up_global_scale"],
        block_inter=8,
        block_hidden=64,
    )
    moe_out = nvfp4_grouped_down_weighted_sum(
        inter,
        expert_ids,
        routing_weights,
        w["mlp.experts._down_packed"],
        w["mlp.experts._down_scale"],
        w["mlp.experts._down_global_scale"],
        block_hidden=8,
        block_inter=256,
    ).reshape_as(h_flat)

    if "mlp.shared_expert.gate_proj.weight" in w:
        if (
            "mlp.shared_expert.gate_proj.weight.packed" in w
            and "mlp.shared_expert.up_proj.weight.packed" in w
            and "mlp.shared_expert.down_proj.weight.packed" in w
        ):
            gate_s, up_s = dual_scalar_bridge(
                h_flat[0],
                w["mlp.shared_expert.gate_proj.weight.packed"],
                w["mlp.shared_expert.up_proj.weight.packed"],
            )
            shared = w["mlp.shared_expert.down_proj.weight.packed"](
                (F.silu(gate_s) * up_s).to(h.dtype)
            ).reshape_as(h_flat)
        else:
            gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
            up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
            shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
        if "mlp.shared_expert_gate.weight" in w:
            shared = shared * torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
        moe_out = moe_out + shared

    return moe_out.to(h.dtype).reshape_as(h)
