#!/usr/bin/env python3
"""Stage 6 Phase 2: grouped MoE packed-prefill census.

P1-A closed dense scalar-dequant batched projections on Spark. P2 moves to the
real prefill bottleneck: the routed MoE expert path. This harness compares one
real layer across:

* BF16 prefill MoE with resident stacked expert shadows.
* P0.1 stream_bf16 proof path, after deleting those BF16 shadows.
* Existing small-M grouped packed oracle, also after deleting the shadows.

It is a census/profiling gate, not a promotion path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.full_forward import (  # noqa: E402
    _moe_forward,
    _moe_forward_packed_prefill_stream_bf16,
)
from engine.loader import load_qwen36_layer  # noqa: E402
from engine.moe_packed_nvfp4 import moe_forward_verify_smallm_nvfp4  # noqa: E402
from engine.nvfp4_runtime import load_grouped_nvfp4_weight  # noqa: E402
from scripts.spark_stage6_p1_dense_projection_poc import (  # noqa: E402
    _bench_cuda,
    _diff_stats,
    _nbytes,
)


DEFAULT_MODEL = "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526"
GIB = 1024**3


def _parse_batches(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _model_cfg(model_dir: Path) -> dict[str, Any]:
    cfg = json.loads((model_dir / "config.json").read_text())
    text_cfg = cfg.get("text_config", cfg)
    return text_cfg


def _attach_packed_moe(model_dir: Path, layer_idx: int, w: dict[str, Any], *, device: str) -> dict[str, Any]:
    base = f"model.language_model.layers.{layer_idx}.mlp.experts"
    gate_up_packed, gate_up_scale, gate_up_global = load_grouped_nvfp4_weight(
        model_dir,
        f"{base}.gate_up_proj",
        device=device,
    )
    down_packed, down_scale, down_global = load_grouped_nvfp4_weight(
        model_dir,
        f"{base}.down_proj",
        device=device,
    )
    w["mlp.experts._gate_up_packed"] = gate_up_packed
    w["mlp.experts._gate_up_scale"] = gate_up_scale
    w["mlp.experts._gate_up_global_scale"] = gate_up_global
    w["mlp.experts._down_packed"] = down_packed
    w["mlp.experts._down_scale"] = down_scale
    w["mlp.experts._down_global_scale"] = down_global
    return {
        "gate_up_packed": gate_up_packed,
        "gate_up_scale": gate_up_scale,
        "gate_up_global_scale": gate_up_global,
        "down_packed": down_packed,
        "down_scale": down_scale,
        "down_global_scale": down_global,
    }


def _tensor_bytes(rows: list[torch.Tensor]) -> int:
    return sum(_nbytes(t) for t in rows if isinstance(t, torch.Tensor))


def _cuda_mem_gib() -> float:
    return float(torch.cuda.memory_allocated() / GIB)


def _one_peak(fn: Callable[[], torch.Tensor]) -> dict[str, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = _cuda_mem_gib()
    y = fn()
    torch.cuda.synchronize()
    after = _cuda_mem_gib()
    peak = float(torch.cuda.max_memory_allocated() / GIB)
    del y
    torch.cuda.empty_cache()
    return {"before_gib": before, "after_gib": after, "peak_gib": peak}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--batches", default="1,4,16,64")
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = "cuda"
    model_dir = Path(args.model)
    batches = _parse_batches(args.batches)
    torch.manual_seed(args.seed)
    os.environ.setdefault("LYNN_ROUTER_TOPK_SORTED", "0")

    text_cfg = _model_cfg(model_dir)
    num_experts = int(text_cfg.get("num_experts", 256))
    top_k = int(text_cfg.get("num_experts_per_tok", 8))

    print("=============== STAGE 6 PHASE 2 GROUPED MOE PREFILL CENSUS ===============", flush=True)
    print(f"model   : {model_dir}", flush=True)
    print(f"layer   : {args.layer}", flush=True)
    print(f"batches : {batches}", flush=True)
    print(f"experts : {num_experts} top_k={top_k}", flush=True)

    w, layer_cfg = load_qwen36_layer(
        str(model_dir),
        args.layer,
        num_experts=num_experts,
        device=device,
        dequant_dtype=torch.bfloat16,
    )
    layer_cfg["num_experts"] = int(layer_cfg.get("num_experts", num_experts))
    layer_cfg["num_experts_per_tok"] = top_k
    layer_cfg["is_moe"] = True
    packed = _attach_packed_moe(model_dir, args.layer, w, device=device)
    hidden = int(w["mlp.gate.weight"].shape[1])

    bf16_shadow_bytes = _tensor_bytes([
        w.get("mlp.experts.gate_up_proj"),
        w.get("mlp.experts.down_proj"),
    ])
    packed_bytes = _tensor_bytes(list(packed.values()))

    print(f"hidden       : {hidden}", flush=True)
    print(f"BF16 shadow  : {bf16_shadow_bytes / GIB:.3f} GiB", flush=True)
    print(f"packed expert: {packed_bytes / GIB:.3f} GiB", flush=True)

    xs: dict[int, torch.Tensor] = {
        b: (torch.randn((1, b, hidden), device=device, dtype=torch.float32) * 0.35).to(torch.bfloat16)
        for b in batches
    }
    torch.cuda.synchronize()

    refs: dict[int, torch.Tensor] = {}
    numeric: dict[str, Any] = {}
    for b, h in xs.items():
        refs[b] = _moe_forward(h, w, layer_cfg).detach()
        stream = _moe_forward_packed_prefill_stream_bf16(h, w, layer_cfg).detach()
        smallm = moe_forward_verify_smallm_nvfp4(h, w, layer_cfg).detach()
        numeric[str(b)] = {
            "stream_vs_bf16": _diff_stats(stream, refs[b]),
            "smallm_vs_bf16": _diff_stats(smallm, refs[b]),
        }
        print(
            f"[numeric M={b}] "
            f"stream cos={numeric[str(b)]['stream_vs_bf16']['cosine']:.9f} "
            f"smallm cos={numeric[str(b)]['smallm_vs_bf16']['cosine']:.9f}",
            flush=True,
        )
        del stream, smallm

    bench_bf16: dict[str, Any] = {}
    for b, h in xs.items():
        bench_bf16[str(b)] = _bench_cuda(
            lambda h=h: _moe_forward(h, w, layer_cfg),
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )

    del w["mlp.experts.gate_up_proj"], w["mlp.experts.down_proj"]
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mem_after_delete = _cuda_mem_gib()

    bench_stream: dict[str, Any] = {}
    bench_smallm: dict[str, Any] = {}
    peak_stream: dict[str, Any] = {}
    peak_smallm: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for b, h in xs.items():
        peak_stream[str(b)] = _one_peak(lambda h=h: _moe_forward_packed_prefill_stream_bf16(h, w, layer_cfg))
        peak_smallm[str(b)] = _one_peak(lambda h=h: moe_forward_verify_smallm_nvfp4(h, w, layer_cfg))
        bench_stream[str(b)] = _bench_cuda(
            lambda h=h: _moe_forward_packed_prefill_stream_bf16(h, w, layer_cfg),
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        bench_smallm[str(b)] = _bench_cuda(
            lambda h=h: moe_forward_verify_smallm_nvfp4(h, w, layer_cfg),
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        bf = bench_bf16[str(b)]["median_us"]
        st = bench_stream[str(b)]["median_us"]
        sm = bench_smallm[str(b)]["median_us"]
        rows.append({
            "batch": b,
            "bf16_median_us": bf,
            "stream_median_us": st,
            "smallm_median_us": sm,
            "stream_vs_bf16": bf / st if st else float("nan"),
            "smallm_vs_bf16": bf / sm if sm else float("nan"),
            "smallm_vs_stream": st / sm if sm else float("nan"),
            "bf16_us_per_token": bf / b,
            "stream_us_per_token": st / b,
            "smallm_us_per_token": sm / b,
        })
        print(
            f"[bench M={b}] bf16={bf:.2f}us stream={st:.2f}us smallm={sm:.2f}us "
            f"smallm_vs_stream={st / sm if sm else float('nan'):.3f}x",
            flush=True,
        )

    numeric_pass = all(
        numeric[str(b)]["stream_vs_bf16"]["cosine"] > 0.999
        and numeric[str(b)]["smallm_vs_bf16"]["cosine"] > 0.999
        for b in batches
    )
    no_shadow_pass = "mlp.experts.gate_up_proj" not in w and "mlp.experts.down_proj" not in w

    result = {
        "schema": "lynn-stage6-p2-grouped-moe-prefill-census-v1",
        "model": str(model_dir),
        "layer": args.layer,
        "seed": args.seed,
        "batches": batches,
        "shape": {
            "hidden": hidden,
            "num_experts": layer_cfg["num_experts"],
            "top_k": top_k,
            "expert_intermediate": int(packed["gate_up_packed"].shape[1] // 2),
        },
        "bytes": {
            "bf16_grouped_expert_shadow": bf16_shadow_bytes,
            "packed_grouped_expert_total": packed_bytes,
            "bf16_to_packed_ratio": bf16_shadow_bytes / packed_bytes if packed_bytes else None,
            "mem_after_deleting_bf16_shadow_gib": mem_after_delete,
        },
        "numeric": numeric,
        "bench": {
            "rows": rows,
            "bf16_prefill": bench_bf16,
            "stream_bf16_no_resident_shadow": bench_stream,
            "smallm_grouped_no_resident_shadow": bench_smallm,
        },
        "memory": {
            "stream_peak": peak_stream,
            "smallm_peak": peak_smallm,
        },
        "passes": {
            "numeric_cos_gt_0.999": bool(numeric_pass),
            "bf16_shadow_deleted_before_packed_benches": bool(no_shadow_pass),
        },
        "notes": [
            "stream_bf16 dequants the whole layer into temporary BF16 weights per call.",
            "smallm groups by unique expert but still dequants each selected expert into a wide temporary.",
            "This census informs the first real P2 kernel; it is not a serving path.",
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
