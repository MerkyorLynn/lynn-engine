#!/usr/bin/env python3
"""P8-I: torch.compile full-token resident decode probe.

This is a ceiling probe, not a production serving path. It measures whether
`torch.compile(..., mode="reduce-overhead")` can beat the manual CUDA graph
full-token ceiling while keeping the same resident weights and cache shapes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
import torch._inductor.config as inductor_config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.full_forward import _decode_layer, _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402


def _prepare_decode_moe_fast_layout(layer_weights: list[dict[str, Any]], layer_cfgs: list[dict[str, Any]]) -> None:
    from triton_kernels.moe_expert_ffn import stack_expert_weights

    for w, cfg in zip(layer_weights, layer_cfgs):
        if "mlp.experts._gate_stacked" in w:
            continue
        if "mlp.experts.gate_up_proj" in w and "mlp.experts.down_proj" in w:
            gate_stacked, up_stacked = w["mlp.experts.gate_up_proj"].chunk(2, dim=1)
            w["mlp.experts._gate_stacked"] = gate_stacked
            w["mlp.experts._up_stacked"] = up_stacked
            w["mlp.experts._down_stacked"] = w["mlp.experts.down_proj"]
            cfg["num_experts"] = int(w["mlp.experts.gate_up_proj"].shape[0])
        else:
            stack_expert_weights(w, num_experts=cfg["num_experts"])


def _prefill(runner: LynnIncrementalRunner, prompt: str):
    ids = _encode_prompt(runner.tokenizer, prompt, runner.device, use_chat_template=False)
    state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)
    for i in range(runner.n_layers):
        h = _prefill_layer(h, pos, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
    state.seq_len = ids.shape[1]
    h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    logits = F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"])
    return int(logits[0].argmax().item()), state


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--mode", default="reduce-overhead")
    ap.add_argument(
        "--inductor-cudagraph-trees",
        action="store_true",
        help="Allow Inductor cudagraph trees. Disabled by default because mutable decode state can trip allocator assertions.",
    )
    args = ap.parse_args()

    # Manual CUDA graph probes are handled separately. For this compile probe,
    # Inductor cudagraph trees are noisy with mutable decode state and can fail
    # before returning a usable timing. Keep them off unless explicitly testing
    # that PyTorch path.
    inductor_config.triton.cudagraph_trees = bool(args.inductor_cudagraph_trees)

    requested_moe = os.environ.get("LYNN_MOE_IMPL", "optimized")
    if requested_moe == "triton":
        os.environ["LYNN_MOE_IMPL"] = "optimized"
    t0 = time.time()
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    load_seconds = time.time() - t0
    os.environ["LYNN_MOE_IMPL"] = requested_moe
    if requested_moe in {"triton", "indexed_bmm"}:
        _prepare_decode_moe_fast_layout(runner.layer_weights, runner.layer_cfgs)

    next_id, state = _prefill(runner, args.prompt)
    token = torch.tensor([[next_id]], device=runner.device, dtype=torch.long)
    pos_tensor = torch.tensor([[int(state.seq_len)]], device=runner.device, dtype=torch.long)

    def full_token():
        h = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
        for i in range(runner.n_layers):
            h = _decode_layer(h, pos_tensor, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
        h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
        return F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"])

    eager_ms = _bench(full_token, args.warmup, max(10, args.iters // 5))

    compiled = torch.compile(full_token, mode=args.mode, fullgraph=False)
    compiled_t0 = time.time()
    compiled()
    torch.cuda.synchronize()
    compiled_first_s = time.time() - compiled_t0
    compiled_ms = _bench(compiled, args.warmup, args.iters)

    verdict = "PASS_SPEEDUP" if compiled_ms < eager_ms else "PASS_NO_SPEEDUP"
    result = {
        "schema_version": "lynn-engine-p8-torch-compile-full-token-probe-v1",
        "model": args.model,
        "device": torch.cuda.get_device_name("cuda"),
        "load_seconds": load_seconds,
        "compile_mode": args.mode,
        "inductor_cudagraph_trees": bool(args.inductor_cudagraph_trees),
        "moe_impl": requested_moe,
        "linear_recurrent_backend": os.environ.get("LYNN_LINEAR_ATTN_RECURRENT_BACKEND", "torch"),
        "linear_recurrent_inplace": os.environ.get("LYNN_LINEAR_ATTN_RECURRENT_INPLACE", "0"),
        "linear_gqa_recurrent": os.environ.get("LYNN_LINEAR_ATTN_GQA_RECURRENT", "0"),
        "qk_norm_rope_backend": os.environ.get("LYNN_QK_NORM_ROPE_BACKEND", "torch"),
        "eager_ms": eager_ms,
        "eager_tps": 1000.0 / eager_ms,
        "compiled_first_s": compiled_first_s,
        "compiled_ms": compiled_ms,
        "compiled_tps": 1000.0 / compiled_ms,
        "speedup": eager_ms / compiled_ms,
        "verdict": verdict,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
