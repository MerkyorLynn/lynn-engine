#!/usr/bin/env python3
"""P168: Qwen3.6 linear/GDN decode segment census across linear-attention layers.

This is a measurement-only probe. It extends P10-C from one layer to a sampled
or full set of linear-attention layers so the next exact kernel island is chosen
from measured repeated cost instead of one-layer intuition.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.full_forward import _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import LynnInferenceState  # noqa: E402
from engine.incremental_decode import _linear, _linear_conv_update_decode, _rms_norm_gated_decode  # noqa: E402
from engine.qwen36_linear_attn_block import (  # noqa: E402
    HEAD_K_DIM,
    HEAD_V_DIM,
    KEY_DIM,
    NUM_K_HEADS,
    NUM_V_HEADS,
    VALUE_DIM,
    V_PER_K,
)
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402
from triton_kernels.gated_delta import (  # noqa: E402
    recurrent_gated_delta_fused_prepare,
    recurrent_gated_delta_fused_prepare_gqa,
)

DEFAULT_PROMPT = "用一句话解释 MoE active parameters"
SEGMENT_KEYS = [
    "fused_native_fp4_inproj",
    "conv_update",
    "split_qkv_repeat",
    "recurrent_fused_prepare",
    "gated_rmsnorm",
    "out_proj_bf16",
    "full_core_recomposed",
]


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


def _summ(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"mean": None, "median": None, "min": None, "max": None, "sum": None}
    return {
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "min": min(xs),
        "max": max(xs),
        "sum": sum(xs),
    }


def _env_snapshot() -> dict[str, str | None]:
    names = [
        "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4",
        "LYNN_LINEAR_ATTN_CONV_BACKEND",
        "LYNN_LINEAR_ATTN_RECURRENT_BACKEND",
        "LYNN_LINEAR_ATTN_RECURRENT_INPLACE",
        "LYNN_LINEAR_ATTN_GQA_RECURRENT",
        "LYNN_RMSNORM_GATED_BACKEND",
        "LYNN_LINEAR_BLOCK_GRAPH",
        "LYNN_LINEAR_BLOCK_GRAPH_REUSE",
        "LYNN_LINEAR_STATE_UPDATE",
        "LYNN_NATIVE_FP4_LM_HEAD",
    ]
    return {name: os.environ.get(name) for name in names}


def _prefill(runner: LynnIncrementalRunner, prompt: str, max_seq_len: int):
    ids = _encode_prompt(runner.tokenizer, prompt, runner.device, use_chat_template=False)
    state = LynnInferenceState.from_config(
        runner.cfg,
        batch=1,
        max_seq_len=max_seq_len,
        device=runner.device,
        dtype=runner.dtype,
    )
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)
    for i in range(runner.n_layers):
        h = _prefill_layer(h, pos, runner.layer_types[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
    state.seq_len = int(ids.shape[1])
    h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    logits = runner._lm_head_logits(h_final)
    return int(logits[0].argmax().item()), state, int(ids.shape[1])


def _profile_layer(runner: LynnIncrementalRunner, state: LynnInferenceState, token_id: int, layer: int, warmup: int, iters: int) -> dict[str, Any]:
    if runner.layer_types[layer] != "linear_attention":
        raise ValueError(f"layer {layer} is {runner.layer_types[layer]!r}, expected linear_attention")
    token = torch.tensor([[token_id]], device=runner.device, dtype=torch.long)
    h0 = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
    w = runner.layer_weights[layer]
    h_new = _rms_norm(h0, w["input_layernorm.weight"])
    B = h_new.shape[0]

    fused_key = "linear_attn._in_proj_qkv_z_b_a.weight"
    if fused_key not in w:
        raise RuntimeError(f"{fused_key} missing; set LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1")

    use_gqa_recurrent = V_PER_K > 1 and os.environ.get("LYNN_LINEAR_ATTN_GQA_RECURRENT", "0") == "1"

    def fused_inproj():
        proj_all = _linear(h_new, w[fused_key])
        return torch.split(
            proj_all,
            [KEY_DIM + KEY_DIM + VALUE_DIM, VALUE_DIM, NUM_V_HEADS, NUM_V_HEADS],
            dim=-1,
        )

    mixed_new, z, b, a = fused_inproj()
    mixed_new_t = mixed_new.transpose(1, 2)

    def conv_update():
        return _linear_conv_update_decode(mixed_new_t, state.conv_state[layer], w["linear_attn.conv1d.weight"])

    out_conv, _ = conv_update()

    def split_qkv():
        q, k, v = torch.split(out_conv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
        q = q.reshape(B, 1, NUM_K_HEADS, HEAD_K_DIM)
        k = k.reshape(B, 1, NUM_K_HEADS, HEAD_K_DIM)
        v = v.reshape(B, 1, NUM_V_HEADS, HEAD_V_DIM)
        if V_PER_K > 1 and not use_gqa_recurrent:
            q = q.repeat_interleave(V_PER_K, dim=2)
            k = k.repeat_interleave(V_PER_K, dim=2)
        return q, k, v

    q, k, v = split_qkv()
    z = z.reshape(B, 1, NUM_V_HEADS, HEAD_V_DIM)
    beta = b.sigmoid()
    g = -w["linear_attn.A_log"].float().exp() * F.softplus(a.float() + w["linear_attn.dt_bias"].float())

    def recurrent():
        if use_gqa_recurrent:
            return recurrent_gated_delta_fused_prepare_gqa(q, k, v, g, beta, state.recurrent_state[layer])
        return recurrent_gated_delta_fused_prepare(q, k, v, g, beta, state.recurrent_state[layer])

    core_attn_out, _ = recurrent()

    def gated_norm():
        flat_x = core_attn_out.reshape(-1, HEAD_V_DIM)
        flat_z = z.reshape(-1, HEAD_V_DIM)
        y = _rms_norm_gated_decode(flat_x, w["linear_attn.norm.weight"], flat_z)
        return y.reshape(B, 1, NUM_V_HEADS * HEAD_V_DIM)

    core_normed = gated_norm()

    def out_proj():
        return _linear(core_normed, w["linear_attn.out_proj.weight"])

    def full_core():
        mixed, z0, b0, a0 = fused_inproj()
        mixed_t = mixed.transpose(1, 2)
        conv, _ = _linear_conv_update_decode(mixed_t, state.conv_state[layer], w["linear_attn.conv1d.weight"])
        q0, k0, v0 = torch.split(conv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
        q0 = q0.reshape(B, 1, NUM_K_HEADS, HEAD_K_DIM)
        k0 = k0.reshape(B, 1, NUM_K_HEADS, HEAD_K_DIM)
        v0 = v0.reshape(B, 1, NUM_V_HEADS, HEAD_V_DIM)
        if V_PER_K > 1 and not use_gqa_recurrent:
            q0 = q0.repeat_interleave(V_PER_K, dim=2)
            k0 = k0.repeat_interleave(V_PER_K, dim=2)
        z0 = z0.reshape(B, 1, NUM_V_HEADS, HEAD_V_DIM)
        beta0 = b0.sigmoid()
        g0 = -w["linear_attn.A_log"].float().exp() * F.softplus(a0.float() + w["linear_attn.dt_bias"].float())
        if use_gqa_recurrent:
            attn, _ = recurrent_gated_delta_fused_prepare_gqa(q0, k0, v0, g0, beta0, state.recurrent_state[layer])
        else:
            attn, _ = recurrent_gated_delta_fused_prepare(q0, k0, v0, g0, beta0, state.recurrent_state[layer])
        flat_x = attn.reshape(-1, HEAD_V_DIM)
        flat_z = z0.reshape(-1, HEAD_V_DIM)
        normed = _rms_norm_gated_decode(flat_x, w["linear_attn.norm.weight"], flat_z)
        normed = normed.reshape(B, 1, NUM_V_HEADS * HEAD_V_DIM)
        return _linear(normed, w["linear_attn.out_proj.weight"])

    timing = {
        "fused_native_fp4_inproj": _bench(fused_inproj, warmup, iters),
        "conv_update": _bench(conv_update, warmup, iters),
        "split_qkv_repeat": _bench(split_qkv, warmup, iters),
        "recurrent_fused_prepare": _bench(recurrent, warmup, iters),
        "gated_rmsnorm": _bench(gated_norm, warmup, iters),
        "out_proj_bf16": _bench(out_proj, warmup, iters),
        "full_core_recomposed": _bench(full_core, max(1, warmup // 2), max(10, iters // 4)),
    }
    return {
        "layer": layer,
        "use_gqa_recurrent": use_gqa_recurrent,
        "timing_ms": timing,
        "top_segments": sorted(
            [{"segment": k, "latency_ms": v} for k, v in timing.items() if k != "full_core_recomposed"],
            key=lambda row: row["latency_ms"],
            reverse=True,
        ),
    }


def _parse_layers(raw: str | None, runner: LynnIncrementalRunner) -> list[int]:
    linear = [i for i, kind in enumerate(runner.layer_types) if kind == "linear_attention"]
    if not raw or raw == "all":
        return linear
    if raw == "first-of-block":
        return [i for i in linear if i % 4 == 0]
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            out.extend(range(start, end + 1))
        else:
            out.append(int(part))
    bad = [i for i in out if i not in linear]
    if bad:
        raise ValueError(f"non-linear-attention layers requested: {bad}")
    return sorted(dict.fromkeys(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="all", help="all, first-of-block, comma list, or ranges")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, max_seq_len=args.max_seq_len, verbose=False)
    layers = _parse_layers(args.layers, runner)
    next_id, state, prompt_tokens = _prefill(runner, args.prompt, args.max_seq_len)

    layer_reports = []
    for layer in layers:
        print(f"[p168] profiling layer {layer}", flush=True)
        layer_reports.append(_profile_layer(runner, state, next_id, layer, args.warmup, args.iters))

    by_segment = {key: [row["timing_ms"][key] for row in layer_reports] for key in SEGMENT_KEYS}
    segment_summary = {key: _summ(vals) for key, vals in by_segment.items()}
    non_full_sum = sum(float(segment_summary[key]["sum"] or 0.0) for key in SEGMENT_KEYS if key != "full_core_recomposed")
    full_core_sum = float(segment_summary["full_core_recomposed"]["sum"] or 0.0)
    hot_segments = sorted(
        [
            {
                "segment": key,
                "sum_ms_across_profiled_layers": float(stats["sum"] or 0.0),
                "mean_ms": stats["mean"],
                "share_of_non_full_sum": (float(stats["sum"] or 0.0) / non_full_sum) if non_full_sum else None,
            }
            for key, stats in segment_summary.items()
            if key != "full_core_recomposed"
        ],
        key=lambda row: row["sum_ms_across_profiled_layers"],
        reverse=True,
    )
    report = {
        "schema_version": "lynn-engine-p168-qwen36-linear-core-segment-census-v1",
        "model": args.model,
        "device": torch.cuda.get_device_name("cuda"),
        "env": _env_snapshot(),
        "prompt": args.prompt,
        "prompt_tokens": prompt_tokens,
        "layers": layers,
        "layer_count": len(layers),
        "warmup": args.warmup,
        "iters": args.iters,
        "segment_summary_ms": segment_summary,
        "aggregate": {
            "non_full_segment_sum_ms_across_profiled_layers": non_full_sum,
            "full_core_sum_ms_across_profiled_layers": full_core_sum,
            "full_core_mean_ms_per_profiled_layer": segment_summary["full_core_recomposed"]["mean"],
        },
        "hot_segments": hot_segments,
        "layers_detail": layer_reports,
        "notes": [
            "Measurement-only census; it does not modify resident defaults.",
            "Layer input is a shared one-token decode seed, so this ranks operation/layout cost rather than semantic layer activation difficulty.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate": report["aggregate"], "hot_segments": hot_segments}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
