#!/usr/bin/env python3
"""P15: segment profile for packed-NVFP4 MoE decode.

The full-token graph path is currently bottlenecked inside 4-layer graph
groups. P13/P14 proved that graph-state plumbing is correctness-sensitive, so
P15 moves back to a safer question: where inside the already-correct MoE decode
does time go?

This probe measures, for a realistic one-token hidden state at a selected
layer:

  - router linear + top-k + softmax
  - active routed experts packed NVFP4 kernels
  - shared expert BF16 path
  - shared expert packed scalar-bridge path
  - current production `moe_forward_decode_packed_nvfp4`

It intentionally does not change engine behavior. Use it to justify the next
opt-in backend switch instead of guessing.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.full_forward import _decode_layer, _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.incremental_decode import decode_full_attn, decode_linear_attn  # noqa: E402
from engine.moe_packed_nvfp4 import moe_forward_decode_packed_nvfp4  # noqa: E402
from engine.nvfp4_runtime import dual_scalar_bridge  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402
from triton_kernels.nvfp4_moe import (  # noqa: E402
    nvfp4_grouped_down_weighted_sum,
    nvfp4_grouped_gate_up_silu,
    nvfp4_grouped_gate_up_silu_fast_decode,
)


def _bench(fn: Callable[[], Any], warmup: int, iters: int) -> float:
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


def _prefill(runner: LynnIncrementalRunner, state: LynnInferenceState, prompt: str) -> tuple[int, int]:
    ids = _encode_prompt(runner.tokenizer, prompt, runner.device, use_chat_template=False)
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)
    for i in range(runner.n_layers):
        h = _prefill_layer(h, pos, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
    state.seq_len = int(ids.shape[1])
    h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    logits = runner._lm_head_logits(h_final)
    return int(logits[0].argmax().item()), int(ids.shape[1])


def _prepare_layer_moe_input(
    runner: LynnIncrementalRunner,
    state: LynnInferenceState,
    token_id: int,
    decode_position: int,
    layer_idx: int,
) -> torch.Tensor:
    """Return the target layer's post-attention RMSNorm input for MoE.

    Previous layers are advanced with the production decode layer. For the
    target layer itself, only the attention sub-block is executed so that the
    returned tensor is exactly what MoE receives.
    """
    token = torch.tensor([[token_id]], device=runner.device, dtype=torch.long)
    pos_tensor = torch.tensor([[decode_position]], device=runner.device, dtype=torch.long)
    h = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])

    for i in range(layer_idx):
        h = _decode_layer(
            h,
            pos_tensor,
            LAYER_TYPES[i],
            runner.layer_weights[i],
            runner.layer_cfgs[i],
            state,
            i,
        )

    w = runner.layer_weights[layer_idx]
    cfg = runner.layer_cfgs[layer_idx]
    residual = h
    h_norm = _rms_norm(h, w["input_layernorm.weight"])
    if LAYER_TYPES[layer_idx] == "linear_attention":
        recurrent_backend = os.environ.get("LYNN_LINEAR_ATTN_RECURRENT_BACKEND", "torch")
        attn_out, new_state, new_conv = decode_linear_attn(
            h_norm,
            w,
            state.recurrent_state[layer_idx],
            state.conv_state[layer_idx],
            recurrent_backend=recurrent_backend,
        )
        # Keep setup semantics aligned with production, although MoE profiling
        # below does not depend on future recurrent state.
        if os.environ.get("LYNN_LINEAR_STATE_UPDATE", "assign") == "inplace":
            target = state.recurrent_state[layer_idx]
            if target.data_ptr() != new_state.data_ptr():
                target.copy_(new_state)
            state.conv_state[layer_idx].copy_(new_conv)
        else:
            state.update_linear_attn_state(layer_idx, new_state, new_conv)
    else:
        K, V = state.kv_cache[layer_idx]
        attn_out = decode_full_attn(
            h_norm,
            pos_tensor,
            w,
            cfg,
            K,
            V,
            cached_seq_len=state.seq_len,
        )
    h_after_attn = residual + attn_out
    return _rms_norm(h_after_attn, w["post_attention_layernorm.weight"]).contiguous()


def _router_topk(h_flat: torch.Tensor, w: dict, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(
        router_logits,
        int(cfg["num_experts_per_tok"]),
        dim=-1,
    )
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32)[0]
    return expert_indices[0].to(torch.long), routing_weights


def _active_packed(
    h_flat: torch.Tensor,
    w: dict,
    cfg: dict,
    *,
    gate_block_inter: int = 8,
    gate_block_hidden: int = 64,
    down_block_hidden: int = 8,
    down_block_inter: int = 256,
) -> torch.Tensor:
    expert_ids, routing_weights = _router_topk(h_flat, w, cfg)
    hidden = h_flat[0]
    gate_up = (
        nvfp4_grouped_gate_up_silu_fast_decode
        if os.environ.get("LYNN_NATIVE_GATEUP_BACKEND") == "triton_fast_decode"
        else nvfp4_grouped_gate_up_silu
    )
    inter = gate_up(
        hidden,
        expert_ids,
        w["mlp.experts._gate_up_packed"],
        w["mlp.experts._gate_up_scale"],
        w["mlp.experts._gate_up_global_scale"],
        block_inter=gate_block_inter,
        block_hidden=gate_block_hidden,
    )
    return nvfp4_grouped_down_weighted_sum(
        inter,
        expert_ids,
        routing_weights,
        w["mlp.experts._down_packed"],
        w["mlp.experts._down_scale"],
        w["mlp.experts._down_global_scale"],
        block_hidden=down_block_hidden,
        block_inter=down_block_inter,
    ).reshape_as(h_flat)


def _shared_bf16(h_flat: torch.Tensor, w: dict) -> torch.Tensor | None:
    if "mlp.shared_expert.gate_proj.weight" not in w:
        return None
    gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
    up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
    shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
    if "mlp.shared_expert_gate.weight" in w:
        shared = shared * torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
    return shared


def _shared_packed_scalar_bridge(h_flat: torch.Tensor, w: dict) -> torch.Tensor | None:
    required = (
        "mlp.shared_expert.gate_proj.weight.packed",
        "mlp.shared_expert.up_proj.weight.packed",
        "mlp.shared_expert.down_proj.weight.packed",
    )
    if any(key not in w for key in required):
        return None
    gate_s, up_s = dual_scalar_bridge(
        h_flat[0],
        w["mlp.shared_expert.gate_proj.weight.packed"],
        w["mlp.shared_expert.up_proj.weight.packed"],
    )
    shared = w["mlp.shared_expert.down_proj.weight.packed"](
        (F.silu(gate_s) * up_s).to(h_flat.dtype)
    ).reshape_as(h_flat)
    if "mlp.shared_expert_gate.weight" in w:
        shared = shared * torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
    return shared


def _shared_packed_native_fast_2d(h_flat: torch.Tensor, w: dict) -> torch.Tensor | None:
    required = (
        "mlp.shared_expert.gate_proj.weight.packed",
        "mlp.shared_expert.up_proj.weight.packed",
        "mlp.shared_expert.down_proj.weight.packed",
    )
    if any(key not in w for key in required):
        return None
    gate_s = w["mlp.shared_expert.gate_proj.weight.packed"].forward_native_fast_2d(h_flat).to(h_flat.dtype).unsqueeze(0)
    up_s = w["mlp.shared_expert.up_proj.weight.packed"].forward_native_fast_2d(h_flat).to(h_flat.dtype).unsqueeze(0)
    shared = w["mlp.shared_expert.down_proj.weight.packed"].forward_native_fast_2d(
        (F.silu(gate_s) * up_s).to(h_flat.dtype)
    ).to(h_flat.dtype).unsqueeze(0)
    if "mlp.shared_expert_gate.weight" in w:
        shared = shared * torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
    return shared


def _diff(a: torch.Tensor | None, b: torch.Tensor | None) -> dict[str, Any] | None:
    if a is None or b is None:
        return None
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    return {
        "max_abs": float((af - bf).abs().max().item()),
        "mean_abs": float((af - bf).abs().mean().item()),
        "cosine": float(F.cosine_similarity(af, bf, dim=0).item()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--gate-block-inter", type=int, default=8)
    ap.add_argument("--gate-block-hidden", type=int, default=64)
    ap.add_argument("--down-block-hidden", type=int, default=8)
    ap.add_argument("--down-block-inter", type=int, default=256)
    ap.add_argument(
        "--sweep-active-blocks",
        action="store_true",
        help="Benchmark a small active-expert block-size grid in the same resident process.",
    )
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    if not (0 <= args.layer < runner.n_layers):
        raise ValueError(f"--layer must be in [0, {runner.n_layers}); got {args.layer}")

    state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
    token_id, decode_position = _prefill(runner, state, args.prompt)
    snap = runner._snapshot_state(state)
    h_moe = _prepare_layer_moe_input(runner, state, token_id, decode_position, args.layer)
    h_flat = h_moe.reshape(-1, h_moe.shape[-1])
    w = runner.layer_weights[args.layer]
    cfg = runner.layer_cfgs[args.layer]

    # Materialize once for diffs and output sanity.
    current = moe_forward_decode_packed_nvfp4(h_moe, w, cfg).reshape_as(h_flat)
    active = _active_packed(
        h_flat,
        w,
        cfg,
        gate_block_inter=args.gate_block_inter,
        gate_block_hidden=args.gate_block_hidden,
        down_block_hidden=args.down_block_hidden,
        down_block_inter=args.down_block_inter,
    )
    shared_bf16 = _shared_bf16(h_flat, w)
    shared_scalar = _shared_packed_scalar_bridge(h_flat, w)
    shared_native = _shared_packed_native_fast_2d(h_flat, w)
    expert_ids, routing_weights = _router_topk(h_flat, w, cfg)

    timings: dict[str, float] = {}

    def router_only():
        _router_topk(h_flat, w, cfg)

    timings["router_topk_ms"] = _bench(router_only, args.warmup, args.iters)
    timings["active_packed_ms"] = _bench(
        lambda: _active_packed(
            h_flat,
            w,
            cfg,
            gate_block_inter=args.gate_block_inter,
            gate_block_hidden=args.gate_block_hidden,
            down_block_hidden=args.down_block_hidden,
            down_block_inter=args.down_block_inter,
        ),
        args.warmup,
        args.iters,
    )
    if shared_bf16 is not None:
        timings["shared_bf16_ms"] = _bench(lambda: _shared_bf16(h_flat, w), args.warmup, args.iters)
    if shared_scalar is not None:
        timings["shared_packed_scalar_bridge_ms"] = _bench(
            lambda: _shared_packed_scalar_bridge(h_flat, w),
            args.warmup,
            args.iters,
        )
    if shared_native is not None:
        timings["shared_packed_native_fast_2d_ms"] = _bench(
            lambda: _shared_packed_native_fast_2d(h_flat, w),
            args.warmup,
            args.iters,
        )
    timings["current_full_moe_ms"] = _bench(
        lambda: moe_forward_decode_packed_nvfp4(h_moe, w, cfg),
        args.warmup,
        args.iters,
    )

    expected_bf16 = active + shared_bf16 if shared_bf16 is not None else active
    expected_scalar = active + shared_scalar if shared_scalar is not None else None
    expected_native = active + shared_native if shared_native is not None else None

    active_block_sweep = []
    if args.sweep_active_blocks:
        configs = []
        for gate_inter in (8, 16, 32):
            for gate_hidden in (64, 128, 256):
                for down_hidden in (8, 16, 32):
                    for down_inter in (128, 256, 512):
                        configs.append((gate_inter, gate_hidden, down_hidden, down_inter))
        seen = set()
        for gate_inter, gate_hidden, down_hidden, down_inter in configs:
            if (gate_inter, gate_hidden, down_hidden, down_inter) in seen:
                continue
            seen.add((gate_inter, gate_hidden, down_hidden, down_inter))
            try:
                out = _active_packed(
                    h_flat,
                    w,
                    cfg,
                    gate_block_inter=gate_inter,
                    gate_block_hidden=gate_hidden,
                    down_block_hidden=down_hidden,
                    down_block_inter=down_inter,
                )
                diff = _diff(out, active)
                ms = _bench(
                    lambda gate_inter=gate_inter, gate_hidden=gate_hidden, down_hidden=down_hidden, down_inter=down_inter: _active_packed(
                        h_flat,
                        w,
                        cfg,
                        gate_block_inter=gate_inter,
                        gate_block_hidden=gate_hidden,
                        down_block_hidden=down_hidden,
                        down_block_inter=down_inter,
                    ),
                    max(2, args.warmup // 2),
                    max(20, args.iters // 4),
                )
                active_block_sweep.append(
                    {
                        "gate_block_inter": gate_inter,
                        "gate_block_hidden": gate_hidden,
                        "down_block_hidden": down_hidden,
                        "down_block_inter": down_inter,
                        "ms": ms,
                        "tps_equiv": 1000.0 / ms,
                        "diff_vs_default": diff,
                    }
                )
            except Exception as exc:  # pragma: no cover - diagnostic output.
                active_block_sweep.append(
                    {
                        "gate_block_inter": gate_inter,
                        "gate_block_hidden": gate_hidden,
                        "down_block_hidden": down_hidden,
                        "down_block_inter": down_inter,
                        "error": repr(exc),
                    }
                )
        active_block_sweep.sort(key=lambda row: row.get("ms", float("inf")))

    result = {
        "schema_version": "lynn-engine-p15-moe-packed-segment-profile-v1",
        "model": args.model,
        "layer": args.layer,
        "layer_type": LAYER_TYPES[args.layer],
        "decode_position": decode_position,
        "native_prepared": runner.packed_decode_native_prepared,
        "packed_decode_backend": runner.packed_decode_backend,
        "moe_impl": runner.moe_impl,
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "routing_weights": [float(x) for x in routing_weights.float().tolist()],
        "kernel_config": {
            "gate_block_inter": args.gate_block_inter,
            "gate_block_hidden": args.gate_block_hidden,
            "down_block_hidden": args.down_block_hidden,
            "down_block_inter": args.down_block_inter,
        },
        "timings_ms": timings,
        "timings_tps_equiv": {k.replace("_ms", "_tps"): 1000.0 / v for k, v in timings.items()},
        "active_block_sweep_top10": active_block_sweep[:10],
        "current_vs_active_plus_shared_bf16": _diff(current, expected_bf16),
        "current_vs_active_plus_shared_scalar": _diff(current, expected_scalar),
        "current_vs_active_plus_shared_native": _diff(current, expected_native),
        "shared_scalar_vs_bf16": _diff(shared_scalar, shared_bf16),
        "shared_native_vs_bf16": _diff(shared_native, shared_bf16),
        "shared_native_vs_scalar": _diff(shared_native, shared_scalar),
        "note": "Profiles MoE subsegments only; attention/setup is run once outside the timed region.",
    }
    # Guard against accidental mutation in setup being optimized away unused.
    runner._restore_state(state, snap)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
