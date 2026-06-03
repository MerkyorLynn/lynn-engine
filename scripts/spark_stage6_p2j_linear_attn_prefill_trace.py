#!/usr/bin/env python3
"""Stage 6 Phase 2-J: linear-attention prefill segment trace.

P2-H/P2-I showed p2e_hybrid can survive selected-layer prefill, but mixed
prefill still goes through the old torch-only linear-attention prefill path.
This trace isolates one linear-attention layer and measures each prefill segment
so the next native-kernel target is evidence-driven.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.incremental_decode import (  # noqa: E402
    _chunk_gated_delta_with_state,
    _rms_norm_gated_decode,
    prefill_linear_attn,
)
from engine.loader import load_qwen36_layer  # noqa: E402
from engine.qwen36_linear_attn_block import (  # noqa: E402
    CONV_KERNEL,
    HEAD_K_DIM,
    HEAD_V_DIM,
    KEY_DIM,
    NUM_K_HEADS,
    NUM_V_HEADS,
    VALUE_DIM,
    V_PER_K,
)
from scripts.spark_stage6_p1_dense_projection_poc import _diff_stats  # noqa: E402


DEFAULT_MODEL = "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526"
GIB = 1024**3


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _model_cfg(model_dir: Path) -> dict[str, Any]:
    cfg = json.loads((model_dir / "config.json").read_text())
    return cfg.get("text_config", cfg)


def _cuda_mem_gib() -> float:
    return float(torch.cuda.memory_allocated() / GIB)


def _time_wall_ms(fn: Callable[[], Any]) -> tuple[Any, float]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000.0


def _time_step(rows: list[dict[str, Any]], name: str, fn: Callable[[], Any]) -> Any:
    before = _cuda_mem_gib()
    out, ms = _time_wall_ms(fn)
    rows.append({
        "name": name,
        "wall_ms": ms,
        "mem_before_gib": before,
        "mem_after_gib": _cuda_mem_gib(),
    })
    return out


def _bench_wall_ms(fn: Callable[[], Any], *, repeats: int) -> dict[str, Any]:
    times: list[float] = []
    for _ in range(repeats):
        _out, ms = _time_wall_ms(fn)
        times.append(ms)
        del _out
        torch.cuda.empty_cache()
    times_sorted = sorted(times)
    return {
        "repeats": repeats,
        "median_ms": float(times_sorted[len(times_sorted) // 2]),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "all_ms": times,
    }


def _trace_prefill(h: torch.Tensor, w: dict[str, Any], *, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    B, T, _ = h.shape
    rows: list[dict[str, Any]] = []

    mixed = _time_step(rows, "qkv_projection", lambda: F.linear(h, w["linear_attn.in_proj_qkv.weight"]))
    mixed_t = _time_step(rows, "transpose_qkv", lambda: mixed.transpose(1, 2).contiguous())
    conv_w = w["linear_attn.conv1d.weight"]
    pad = CONV_KERNEL - 1

    def conv_step() -> tuple[torch.Tensor, torch.Tensor]:
        mixed_padded = F.pad(mixed_t, (pad, 0))
        mixed_conv = F.conv1d(mixed_padded, conv_w, bias=None, padding=0, groups=mixed_t.shape[1])
        mixed_conv = F.silu(mixed_conv)
        new_conv_state = mixed_t[:, :, max(0, T - (CONV_KERNEL - 1)):].contiguous()
        if new_conv_state.shape[-1] < CONV_KERNEL - 1:
            pad_amt = (CONV_KERNEL - 1) - new_conv_state.shape[-1]
            new_conv_state = F.pad(new_conv_state, (pad_amt, 0))
        return mixed_conv.transpose(1, 2).contiguous(), new_conv_state

    mixed_conv, new_conv_state = _time_step(rows, "causal_depthwise_conv_silu", conv_step)

    def split_step() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = torch.split(mixed_conv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
        q = q.reshape(B, T, NUM_K_HEADS, HEAD_K_DIM)
        k = k.reshape(B, T, NUM_K_HEADS, HEAD_K_DIM)
        v = v.reshape(B, T, NUM_V_HEADS, HEAD_V_DIM)
        return q, k, v

    q, k, v = _time_step(rows, "split_qkv_reshape", split_step)
    z = _time_step(
        rows,
        "z_projection",
        lambda: F.linear(h, w["linear_attn.in_proj_z.weight"]).reshape(B, T, NUM_V_HEADS, HEAD_V_DIM),
    )
    beta = _time_step(
        rows,
        "beta_projection_sigmoid",
        lambda: F.linear(h, w["linear_attn.in_proj_b.weight"]).sigmoid(),
    )

    def g_step() -> torch.Tensor:
        a = F.linear(h, w["linear_attn.in_proj_a.weight"])
        neg_exp_A_log = w.get("linear_attn._neg_exp_A_log")
        if neg_exp_A_log is None:
            neg_exp_A_log = -w["linear_attn.A_log"].float().exp()
        return neg_exp_A_log * F.softplus(a.float() + w["linear_attn.dt_bias"].float())

    g = _time_step(rows, "g_projection_decay", g_step)

    if V_PER_K > 1:
        q, k = _time_step(
            rows,
            "repeat_qk_for_gqa",
            lambda: (q.repeat_interleave(V_PER_K, dim=2), k.repeat_interleave(V_PER_K, dim=2)),
        )

    core_attn_out, last_state = _time_step(
        rows,
        "chunk_gated_delta_with_state",
        lambda: _chunk_gated_delta_with_state(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=chunk_size,
            use_qk_l2norm=True,
            initial_state=None,
            output_final_state=True,
        ),
    )

    def norm_step() -> torch.Tensor:
        flat_x = core_attn_out.reshape(-1, HEAD_V_DIM)
        flat_z = z.reshape(-1, HEAD_V_DIM)
        flat_y = _rms_norm_gated_decode(flat_x, w["linear_attn.norm.weight"], flat_z)
        return flat_y.reshape(B, T, NUM_V_HEADS * HEAD_V_DIM)

    normed = _time_step(rows, "rmsnorm_gated", norm_step)
    out = _time_step(rows, "out_projection", lambda: F.linear(normed, w["linear_attn.out_proj.weight"]))
    return out, last_state, new_conv_state, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seq-lens", default="16,64,128")
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = "cuda"
    model_dir = Path(args.model)
    text_cfg = _model_cfg(model_dir)
    num_experts = int(text_cfg.get("num_experts", 256))
    torch.manual_seed(args.seed)

    w, _cfg = load_qwen36_layer(
        str(model_dir),
        args.layer,
        num_experts=num_experts,
        device=device,
        dequant_dtype=torch.bfloat16,
    )
    hidden = int(w["linear_attn.in_proj_qkv.weight"].shape[1])
    seq_lens = _parse_ints(args.seq_lens)

    print("=============== STAGE 6 PHASE 2-J LINEAR-ATTN PREFILL TRACE ===============", flush=True)
    print(f"model      : {model_dir}", flush=True)
    print(f"layer      : {args.layer}", flush=True)
    print(f"seq_lens   : {seq_lens}", flush=True)
    print(f"chunk_size : {args.chunk_size}", flush=True)

    traces: dict[str, Any] = {}
    for seq_len in seq_lens:
        h = (torch.randn((1, seq_len, hidden), device=device, dtype=torch.float32) * 0.35).to(torch.bfloat16)
        ref_out, ref_state, ref_conv = prefill_linear_attn(h, w, chunk_size=args.chunk_size)
        traced_out, traced_state, traced_conv, rows = _trace_prefill(h, w, chunk_size=args.chunk_size)
        full_bench = _bench_wall_ms(lambda h=h: prefill_linear_attn(h, w, chunk_size=args.chunk_size), repeats=args.repeats)
        traced_total_ms = sum(float(r["wall_ms"]) for r in rows)
        for r in rows:
            r["pct_of_trace"] = float(r["wall_ms"] / traced_total_ms * 100.0) if traced_total_ms else 0.0
        diff_out = _diff_stats(traced_out, ref_out)
        diff_state = _diff_stats(traced_state, ref_state)
        diff_conv = _diff_stats(traced_conv, ref_conv)
        traces[str(seq_len)] = {
            "shape": {"hidden": hidden, "seq_len": seq_len, "chunk_size": args.chunk_size},
            "full_prefill_wall": full_bench,
            "trace_total_ms": traced_total_ms,
            "segments": rows,
            "numeric": {
                "out_vs_prefill_linear_attn": diff_out,
                "state_vs_prefill_linear_attn": diff_state,
                "conv_vs_prefill_linear_attn": diff_conv,
            },
        }
        top = sorted(rows, key=lambda x: float(x["wall_ms"]), reverse=True)[:4]
        print(f"[T={seq_len}] full={full_bench['median_ms']:.2f}ms trace={traced_total_ms:.2f}ms", flush=True)
        for row in top:
            print(f"  {row['name']}: {row['wall_ms']:.2f}ms ({row['pct_of_trace']:.1f}%)", flush=True)
        print(
            f"  numeric: cos={diff_out['cosine']:.9f} rel_l2={diff_out['rel_l2']:.3e} "
            f"argmax={diff_out['argmax_match']}",
            flush=True,
        )

    numeric_pass = all(
        v["numeric"]["out_vs_prefill_linear_attn"]["cosine"] > 0.999999
        and v["numeric"]["out_vs_prefill_linear_attn"]["argmax_match"]
        and v["numeric"]["state_vs_prefill_linear_attn"]["cosine"] > 0.999999
        and v["numeric"]["conv_vs_prefill_linear_attn"]["cosine"] > 0.999999
        for v in traces.values()
    )
    result = {
        "schema": "lynn-stage6-p2j-linear-attn-prefill-trace-v1",
        "model": str(model_dir),
        "layer": args.layer,
        "seed": args.seed,
        "seq_lens": seq_lens,
        "chunk_size": args.chunk_size,
        "traces": traces,
        "passes": {
            "numeric": bool(numeric_pass),
            "all": bool(numeric_pass),
        },
        "notes": [
            "This isolates BF16 linear-attention prefill only; it does not exercise P2E MoE.",
            "Segment times are wall-clock with cuda synchronize around each step, so Python loop overhead is included.",
        ],
    }
    print("=============== RESULT JSON ===============", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
