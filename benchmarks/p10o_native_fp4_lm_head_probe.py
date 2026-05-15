#!/usr/bin/env python3
"""P10-O: native FP4 lm_head feasibility probe.

The P10-N serving-shaped path crosses 100 TPS only when benchmark-only state
restore is removed. The strict full path is pinned just below 100 TPS by the
final BF16 lm_head (~0.668 ms). The published 27B NVFP4 artifact does not pack
lm_head, so this probe quantizes lm_head at runtime once and measures whether a
native FP4 lm_head can safely replace the BF16 projection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.full_forward import _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.nvfp4_runtime import _compact_scale_to_swizzled_fp8  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402
from triton_kernels.nvfp4_linear import quantize_fp4_m1_native  # noqa: E402


_E2M1_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _bench(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
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


def _quantize_weight_to_fp4(weight: torch.Tensor, *, chunk_rows: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2 or weight.shape[1] % 16 != 0:
        raise ValueError(f"expected [V, K] with K divisible by 16, got {tuple(weight.shape)}")
    table = _E2M1_TABLE.to(device=weight.device)
    v, k = weight.shape
    groups = k // 16
    packed = torch.empty((v, k // 2), device=weight.device, dtype=torch.uint8)
    scale = torch.empty((v, groups), device=weight.device, dtype=torch.float32)
    for start in range(0, v, chunk_rows):
        end = min(start + chunk_rows, v)
        wg = weight[start:end].float().reshape(end - start, groups, 16)
        scale_chunk = (wg.abs().amax(dim=-1) / float(table[-1])).clamp_min(1e-8)
        normalized = wg.abs() / scale_chunk.unsqueeze(-1)
        mag = torch.argmin((normalized.unsqueeze(-1) - table.view(1, 1, 1, -1)).abs(), dim=-1)
        sign = (wg < 0).to(torch.uint8) * 8
        codes = (mag.to(torch.uint8) | sign).reshape(end - start, k)
        packed[start:end] = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
        scale[start:end] = scale_chunk
    return packed, scale.contiguous()


def _prefill_last_hidden(runner: LynnIncrementalRunner, prompt: str) -> torch.Tensor:
    ids = _encode_prompt(runner.tokenizer, prompt, runner.device, use_chat_template=False)
    state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)
    for i in range(runner.n_layers):
        h = _prefill_layer(h, pos, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
    h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    return h_final[:, -1, :].contiguous()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--quant-chunk-rows", type=int, default=4096)
    args = ap.parse_args()

    if not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(torch, "_scaled_mm"):
        raise RuntimeError("native FP4 requires torch.float4_e2m1fn_x2 and torch._scaled_mm")

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    h = _prefill_last_hidden(runner, args.prompt)
    lm_head = runner.outside["lm_head.weight"].contiguous()

    quant_start = torch.cuda.Event(enable_timing=True)
    quant_end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    quant_start.record()
    packed_w, compact_scale = _quantize_weight_to_fp4(lm_head, chunk_rows=args.quant_chunk_rows)
    scale_b = _compact_scale_to_swizzled_fp8(compact_scale, outer_dim=lm_head.shape[0], k=lm_head.shape[1])
    quant_end.record()
    torch.cuda.synchronize()
    quant_ms = float(quant_start.elapsed_time(quant_end))

    act_packed, scale_a = quantize_fp4_m1_native(h)

    def bf16_lm_head() -> torch.Tensor:
        return F.linear(h, lm_head)

    def native_lm_head() -> torch.Tensor:
        return torch._scaled_mm(
            act_packed.view(torch.float4_e2m1fn_x2),
            packed_w.view(torch.float4_e2m1fn_x2).t(),
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.float16,
        )

    ref = bf16_lm_head()
    native = native_lm_head().to(ref.dtype)
    diff = (native.float() - ref.float()).abs()
    ref_top = torch.topk(ref.float()[0], k=args.top_k)
    nat_top = torch.topk(native.float()[0], k=args.top_k)
    ref_ids = [int(x) for x in ref_top.indices.tolist()]
    nat_ids = [int(x) for x in nat_top.indices.tolist()]
    overlap = len(set(ref_ids) & set(nat_ids))

    timing = {
        "bf16_lm_head_ms": _bench(bf16_lm_head, args.warmup, args.iters),
        "native_fp4_lm_head_ms": _bench(native_lm_head, args.warmup, args.iters),
    }
    timing["native_vs_bf16_ratio"] = timing["bf16_lm_head_ms"] / timing["native_fp4_lm_head_ms"]

    result = {
        "schema_version": "lynn-engine-p10o-native-fp4-lm-head-probe-v1",
        "model": args.model,
        "shape": {
            "hidden": list(h.shape),
            "lm_head": list(lm_head.shape),
            "packed_lm_head": list(packed_w.shape),
            "scale_b": list(scale_b.shape),
        },
        "one_time_quantize_ms": quant_ms,
        "diff": {
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "rel_l2": float(torch.linalg.vector_norm(native.float() - ref.float()).item() / torch.linalg.vector_norm(ref.float()).item()),
            "cosine": float(F.cosine_similarity(native.float().flatten(), ref.float().flatten(), dim=0).item()),
        },
        "topk": {
            "k": args.top_k,
            "overlap": overlap,
            "overlap_ratio": overlap / args.top_k,
            "ref_top_ids": ref_ids,
            "native_top_ids": nat_ids,
            "top1_match": bool(ref_ids[0] == nat_ids[0]),
        },
        "timing_ms": timing,
        "pass": bool(ref_ids[0] == nat_ids[0] and overlap / args.top_k >= 0.8),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
