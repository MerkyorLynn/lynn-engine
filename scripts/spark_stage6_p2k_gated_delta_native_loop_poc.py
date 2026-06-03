#!/usr/bin/env python3
"""Stage 6 Phase 2-KA: gated-delta native recurrent-loop PoC.

P2-J showed the linear-attention prefill wall is the torch-only
`chunk_gated_delta_with_state` segment. This PoC asks the narrow first question:
can the existing Triton single-token recurrent gated-delta kernel be reused as a
prefill primitive by looping over T tokens?

This is intentionally a lower-bound. If it is correct but slow, the verdict is
"do not promote; write a real chunk-level prefill kernel".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.incremental_decode import _chunk_gated_delta_with_state  # noqa: E402
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
from triton_kernels.gated_delta import recurrent_gated_delta_fused_prepare_gqa  # noqa: E402


DEFAULT_MODEL = "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526"


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _model_cfg(model_dir: Path) -> dict[str, Any]:
    cfg = json.loads((model_dir / "config.json").read_text())
    return cfg.get("text_config", cfg)


def _time_wall_ms(fn: Callable[[], Any]) -> tuple[Any, float]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000.0


def _bench_wall_ms(fn: Callable[[], Any], *, repeats: int) -> dict[str, Any]:
    times: list[float] = []
    for _ in range(repeats):
        out, ms = _time_wall_ms(fn)
        times.append(ms)
        del out
        torch.cuda.empty_cache()
    times_sorted = sorted(times)
    return {
        "repeats": repeats,
        "median_ms": float(times_sorted[len(times_sorted) // 2]),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "all_ms": times,
    }


def _prepare_linear_attn_inputs(h: torch.Tensor, w: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, _ = h.shape
    mixed = F.linear(h, w["linear_attn.in_proj_qkv.weight"]).transpose(1, 2)
    mixed = F.conv1d(
        F.pad(mixed, (CONV_KERNEL - 1, 0)),
        w["linear_attn.conv1d.weight"],
        bias=None,
        padding=0,
        groups=mixed.shape[1],
    )
    mixed = F.silu(mixed).transpose(1, 2).contiguous()
    q, k, v = torch.split(mixed, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    q = q.reshape(B, T, NUM_K_HEADS, HEAD_K_DIM)
    k = k.reshape(B, T, NUM_K_HEADS, HEAD_K_DIM)
    v = v.reshape(B, T, NUM_V_HEADS, HEAD_V_DIM)
    beta = F.linear(h, w["linear_attn.in_proj_b.weight"]).sigmoid()
    a = F.linear(h, w["linear_attn.in_proj_a.weight"])
    neg_exp_A_log = w.get("linear_attn._neg_exp_A_log")
    if neg_exp_A_log is None:
        neg_exp_A_log = -w["linear_attn.A_log"].float().exp()
    g = neg_exp_A_log * F.softplus(a.float() + w["linear_attn.dt_bias"].float())
    return q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta.contiguous()


def _reference_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if V_PER_K > 1:
        q_ref = q.repeat_interleave(V_PER_K, dim=2)
        k_ref = k.repeat_interleave(V_PER_K, dim=2)
    else:
        q_ref, k_ref = q, k
    return _chunk_gated_delta_with_state(
        q_ref,
        k_ref,
        v,
        g,
        beta,
        chunk_size=chunk_size,
        use_qk_l2norm=True,
        initial_state=None,
        output_final_state=True,
    )


def _native_recurrent_loop(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    inplace: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    old = os.environ.get("LYNN_LINEAR_ATTN_RECURRENT_INPLACE")
    os.environ["LYNN_LINEAR_ATTN_RECURRENT_INPLACE"] = "1" if inplace else "0"
    try:
        state = torch.zeros(
            (1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM),
            device=q.device,
            dtype=torch.float32,
        )
        outs: list[torch.Tensor] = []
        for t in range(int(q.shape[1])):
            out_t, state = recurrent_gated_delta_fused_prepare_gqa(
                q[:, t : t + 1, :, :].contiguous(),
                k[:, t : t + 1, :, :].contiguous(),
                v[:, t : t + 1, :, :].contiguous(),
                g[:, t : t + 1, :].contiguous(),
                beta[:, t : t + 1, :].contiguous(),
                state,
            )
            outs.append(out_t)
        return torch.cat(outs, dim=1), state
    finally:
        if old is None:
            os.environ.pop("LYNN_LINEAR_ATTN_RECURRENT_INPLACE", None)
        else:
            os.environ["LYNN_LINEAR_ATTN_RECURRENT_INPLACE"] = old


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

    print("=============== STAGE 6 PHASE 2-KA GATED-DELTA NATIVE LOOP POC ===============", flush=True)
    print(f"model      : {model_dir}", flush=True)
    print(f"layer      : {args.layer}", flush=True)
    print(f"seq_lens   : {seq_lens}", flush=True)
    print(f"chunk_size : {args.chunk_size}", flush=True)

    rows: list[dict[str, Any]] = []
    numeric: dict[str, Any] = {}
    for seq_len in seq_lens:
        h = (torch.randn((1, seq_len, hidden), device=device, dtype=torch.float32) * 0.35).to(torch.bfloat16)
        q, k, v, g, beta = _prepare_linear_attn_inputs(h, w)
        ref_out, ref_state = _reference_chunk(q, k, v, g, beta, chunk_size=args.chunk_size)

        def ref_fn():
            return _reference_chunk(q, k, v, g, beta, chunk_size=args.chunk_size)

        def alloc_fn():
            return _native_recurrent_loop(q, k, v, g, beta, inplace=False)

        def inplace_fn():
            return _native_recurrent_loop(q, k, v, g, beta, inplace=True)

        alloc_out, alloc_state = alloc_fn()
        inplace_out, inplace_state = inplace_fn()
        numeric[f"T{seq_len}_alloc_out_vs_chunk"] = _diff_stats(alloc_out, ref_out)
        numeric[f"T{seq_len}_alloc_state_vs_chunk"] = _diff_stats(alloc_state, ref_state)
        numeric[f"T{seq_len}_inplace_out_vs_chunk"] = _diff_stats(inplace_out, ref_out)
        numeric[f"T{seq_len}_inplace_state_vs_chunk"] = _diff_stats(inplace_state, ref_state)

        ref_bench = _bench_wall_ms(ref_fn, repeats=args.repeats)
        alloc_bench = _bench_wall_ms(alloc_fn, repeats=args.repeats)
        inplace_bench = _bench_wall_ms(inplace_fn, repeats=args.repeats)
        row = {
            "seq_len": seq_len,
            "chunk_reference_ms": ref_bench["median_ms"],
            "native_loop_alloc_ms": alloc_bench["median_ms"],
            "native_loop_inplace_ms": inplace_bench["median_ms"],
            "alloc_vs_chunk": ref_bench["median_ms"] / alloc_bench["median_ms"] if alloc_bench["median_ms"] else None,
            "inplace_vs_chunk": ref_bench["median_ms"] / inplace_bench["median_ms"] if inplace_bench["median_ms"] else None,
            "estimated_kernel_launches": seq_len,
            "ref_bench": ref_bench,
            "alloc_bench": alloc_bench,
            "inplace_bench": inplace_bench,
        }
        rows.append(row)
        print(
            f"[T={seq_len}] chunk={row['chunk_reference_ms']:.2f}ms "
            f"native_alloc={row['native_loop_alloc_ms']:.2f}ms "
            f"native_inplace={row['native_loop_inplace_ms']:.2f}ms",
            flush=True,
        )
        print(
            f"  alloc cos={numeric[f'T{seq_len}_alloc_out_vs_chunk']['cosine']:.9f} "
            f"rel_l2={numeric[f'T{seq_len}_alloc_out_vs_chunk']['rel_l2']:.3e}; "
            f"inplace cos={numeric[f'T{seq_len}_inplace_out_vs_chunk']['cosine']:.9f} "
            f"rel_l2={numeric[f'T{seq_len}_inplace_out_vs_chunk']['rel_l2']:.3e}",
            flush=True,
        )

    numeric_pass = all(v["cosine"] > 0.999 and v["argmax_match"] for v in numeric.values())
    speed_pass = all((r["inplace_vs_chunk"] or 0.0) >= 1.0 for r in rows)
    result = {
        "schema": "lynn-stage6-p2ka-gated-delta-native-loop-poc-v1",
        "model": str(model_dir),
        "layer": args.layer,
        "seed": args.seed,
        "seq_lens": seq_lens,
        "chunk_size": args.chunk_size,
        "rows": rows,
        "numeric": numeric,
        "passes": {
            "numeric": bool(numeric_pass),
            "speed_vs_chunk_reference": bool(speed_pass),
            "all": bool(numeric_pass and speed_pass),
        },
        "notes": [
            "This is a lower-bound reuse of the existing single-token Triton recurrent kernel.",
            "It intentionally launches one recurrent kernel per token; a true prefill kernel must process a chunk/block per launch.",
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
