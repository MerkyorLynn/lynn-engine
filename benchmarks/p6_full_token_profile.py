#!/usr/bin/env python3
"""P6-F: full-token resident decode profile.

Profiles one warm decode token through all 40 layers using already-resident
weights. This is the first profile after P6-D/E showed recurrent fusion helps
but does not yet reach the 50 TPS target.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.full_forward import _decode_layer, _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402


def _prepare_decode_moe_fast_layout(layer_weights: list[dict], layer_cfgs: list[dict]) -> None:
    """Prepare one-time MoE expert layout for indexed/triton decode kernels.

    Full 35B checkpoints keep per-expert tensors, while the variable 27B
    skeleton already stores fused gate_up/down tensors. The fast decode kernels
    consume `_gate_stacked` / `_up_stacked` / `_down_stacked` aliases in both
    cases.
    """
    from triton_kernels.moe_expert_ffn import stack_expert_weights

    for w, cfg in zip(layer_weights, layer_cfgs):
        if "mlp.experts._gate_stacked" in w:
            continue
        if "mlp.experts.gate_up_proj" in w and "mlp.experts.down_proj" in w:
            gate_stacked, up_stacked = w["mlp.experts.gate_up_proj"].chunk(2, dim=1)
            # Zero-copy aliases: the Triton kernel consumes explicit strides, so
            # copying all 40 layers would only waste ~18 GB and OOM 96 GB cards.
            w["mlp.experts._gate_stacked"] = gate_stacked
            w["mlp.experts._up_stacked"] = up_stacked
            w["mlp.experts._down_stacked"] = w["mlp.experts.down_proj"]
            cfg["num_experts"] = int(w["mlp.experts.gate_up_proj"].shape[0])
        else:
            stack_expert_weights(w, num_experts=cfg["num_experts"])


def _bench(fn, warmup: int, iters: int) -> float:
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


def _prefill(runner: LynnIncrementalRunner, prompt: str, *, use_chat_template: bool):
    ids = _encode_prompt(runner.tokenizer, prompt, runner.device, use_chat_template=use_chat_template)
    T = ids.shape[1]
    state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(T, device=runner.device, dtype=torch.long).unsqueeze(0)
    for i in range(runner.n_layers):
        h = _prefill_layer(h, pos, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
    state.seq_len = T
    h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    logits = F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"])
    next_id = int(logits[0].argmax().item())
    return next_id, state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--chat-template", action="store_true")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    requested_moe_impl = os.environ.get("LYNN_MOE_IMPL", "optimized")
    # Historical versions forced both indexed_bmm and triton back to
    # `optimized` before runner construction, then restored the requested
    # backend after load. That is no longer safe: the resident runner may
    # prewarm CUDA graphs during __init__, and the Python optimized MoE path
    # calls torch.unique, which is illegal during stream capture. Keep Triton
    # active from construction onward; only indexed_bmm remains single-prompt
    # incompatible with the reusable resident runner.
    if requested_moe_impl == "indexed_bmm":
        os.environ["LYNN_MOE_IMPL"] = "optimized"
    runner = LynnIncrementalRunner(args.model, device=args.device, dtype=dtype, verbose=False)
    os.environ["LYNN_MOE_IMPL"] = requested_moe_impl
    if requested_moe_impl in ("indexed_bmm", "triton"):
        _prepare_decode_moe_fast_layout(runner.layer_weights, runner.layer_cfgs)
    next_id, state = _prefill(runner, args.prompt, use_chat_template=args.chat_template)
    token = torch.tensor([[next_id]], device=args.device, dtype=torch.long)
    pos_id = state.seq_len
    pos_tensor = torch.tensor([[pos_id]], device=args.device, dtype=torch.long)

    def embed():
        return F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])

    h0 = embed()

    layer_lat = []
    h = h0
    # Sequential one-shot profile mutates a cloned state object once.
    # For per-layer benchmarks we call each layer repeatedly with the same input
    # and state slot; that captures layer kernel cost, not full generation text.
    for i in range(runner.n_layers):
        w = runner.layer_weights[i]
        cfg = runner.layer_cfgs[i]
        layer_type = LAYER_TYPES[i]
        state_i = state
        fn = lambda i=i, h=h, w=w, cfg=cfg, layer_type=layer_type, state_i=state_i: _decode_layer(
            h, pos_tensor, layer_type, w, cfg, state_i, i
        )
        lat = _bench(fn, args.warmup, args.iters)
        layer_lat.append({"layer": i, "type": layer_type, "latency_ms": lat})
        # Advance once for a realistic next layer input.
        h = _decode_layer(h, pos_tensor, layer_type, w, cfg, state, i)

    def full_token():
        h = embed()
        for i in range(runner.n_layers):
            h = _decode_layer(h, pos_tensor, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
        h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
        return F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"])

    full_ms = _bench(full_token, args.warmup, args.iters)
    by_type = {}
    for item in layer_lat:
        rec = by_type.setdefault(item["type"], {"count": 0, "sum_ms": 0.0, "max_ms": 0.0})
        rec["count"] += 1
        rec["sum_ms"] += item["latency_ms"]
        rec["max_ms"] = max(rec["max_ms"], item["latency_ms"])
    for rec in by_type.values():
        rec["avg_ms"] = rec["sum_ms"] / rec["count"]

    result = {
        "schema_version": "lynn-engine-p6-full-token-profile-v1",
        "model": args.model,
        "recurrent_backend": os.environ.get("LYNN_LINEAR_ATTN_RECURRENT_BACKEND", "torch"),
        "moe_impl": os.environ.get("LYNN_MOE_IMPL", "optimized"),
        "device": torch.cuda.get_device_name(args.device),
        "load_seconds": runner.load_seconds,
        "full_token_ms": full_ms,
        "estimated_tps": 1000.0 / full_ms,
        "by_type": by_type,
        "top_layers": sorted(layer_lat, key=lambda x: x["latency_ms"], reverse=True)[:10],
        "layers": layer_lat,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
