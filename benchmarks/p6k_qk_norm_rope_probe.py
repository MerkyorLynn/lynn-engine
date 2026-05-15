#!/usr/bin/env python3
"""P6-K: fused Q/K norm+RoPE correctness and latency probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.full_forward import _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.incremental_decode import _apply_partial_rope, _build_rope_cos_sin, _linear  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402
from triton_kernels.qk_norm_rope import HAS_TRITON, qk_norm_rope_triton  # noqa: E402


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=120)
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    next_id, state = _prefill(runner, "用一句话解释 MoE active parameters")
    token = torch.tensor([[next_id]], device=runner.device, dtype=torch.long)
    h0 = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
    w = runner.layer_weights[args.layer]
    cfg = runner.layer_cfgs[args.layer]
    h_norm = _rms_norm(h0, w["input_layernorm.weight"])

    B = 1
    H_Q = cfg["num_attention_heads"]
    H_KV = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    rotary_dim = int(head_dim * cfg["partial_rotary_factor"])
    q_full = _linear(h_norm, w["self_attn.q_proj.weight"])
    k_new = _linear(h_norm, w["self_attn.k_proj.weight"])
    q, _gate = q_full.view(B, 1, H_Q, head_dim * 2).chunk(2, dim=-1)
    q = q.transpose(1, 2)
    k = k_new.view(B, 1, H_KV, head_dim).transpose(1, 2)
    pos_tensor = torch.tensor([[state.seq_len]], device=runner.device, dtype=torch.long)
    cos, sin = _build_rope_cos_sin(pos_tensor, rotary_dim, cfg["rope_theta"], runner.device, runner.dtype)

    def ref_q():
        return _apply_partial_rope(_rms_norm(q, w["self_attn.q_norm.weight"]), cos, sin, rotary_dim)

    def ref_k():
        return _apply_partial_rope(_rms_norm(k, w["self_attn.k_norm.weight"]), cos, sin, rotary_dim)

    q_ref = ref_q()
    k_ref = ref_k()
    q_tri = qk_norm_rope_triton(q, w["self_attn.q_norm.weight"], cos, sin, rotary_dim)
    k_tri = qk_norm_rope_triton(k, w["self_attn.k_norm.weight"], cos, sin, rotary_dim)

    result = {
        "schema_version": "lynn-engine-p6k-qk-norm-rope-probe-v1",
        "model": args.model,
        "layer": args.layer,
        "device": torch.cuda.get_device_name("cuda"),
        "has_triton": HAS_TRITON,
        "head_dim": head_dim,
        "rotary_dim": rotary_dim,
        "diff": {
            "q_max_abs": float((q_tri.float() - q_ref.float()).abs().max().item()),
            "q_cosine": float(F.cosine_similarity(q_tri.float().flatten(), q_ref.float().flatten(), dim=0).item()),
            "k_max_abs": float((k_tri.float() - k_ref.float()).abs().max().item()),
            "k_cosine": float(F.cosine_similarity(k_tri.float().flatten(), k_ref.float().flatten(), dim=0).item()),
        },
        "latency_ms": {
            "torch_q_norm_rope": _bench(ref_q, args.warmup, args.iters),
            "torch_k_norm_rope": _bench(ref_k, args.warmup, args.iters),
            "triton_q_norm_rope": _bench(lambda: qk_norm_rope_triton(q, w["self_attn.q_norm.weight"], cos, sin, rotary_dim), args.warmup, args.iters),
            "triton_k_norm_rope": _bench(lambda: qk_norm_rope_triton(k, w["self_attn.k_norm.weight"], cos, sin, rotary_dim), args.warmup, args.iters),
        },
    }
    result["latency_ms"]["torch_qk_total"] = result["latency_ms"]["torch_q_norm_rope"] + result["latency_ms"]["torch_k_norm_rope"]
    result["latency_ms"]["triton_qk_total"] = result["latency_ms"]["triton_q_norm_rope"] + result["latency_ms"]["triton_k_norm_rope"]
    result["speedup_qk_total"] = result["latency_ms"]["torch_qk_total"] / result["latency_ms"]["triton_qk_total"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
