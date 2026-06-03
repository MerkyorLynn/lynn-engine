#!/usr/bin/env python3
"""Stage 6 Phase 2-F: opt-in one-layer packed-prefill replacement verify.

P2-E proved the scheduler/retune composition in a harness. P2-F verifies the
same idea through the engine dispatch path:

    LYNN_PACKED_PREFILL_SLOW=1
    LYNN_PACKED_PREFILL_SLOW_MODE=p2e_hybrid

The active BF16 expert shadows are deleted before the packed modes run.
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

from engine.full_forward import _moe_forward  # noqa: E402
from engine.loader import load_qwen36_layer  # noqa: E402
from scripts.spark_stage6_p1_dense_projection_poc import (  # noqa: E402
    _bench_cuda,
    _diff_stats,
    _nbytes,
)
from scripts.spark_stage6_p2_grouped_moe_prefill_census import _attach_packed_moe  # noqa: E402


DEFAULT_MODEL = "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526"
GIB = 1024**3


def _parse_batches(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _model_cfg(model_dir: Path) -> dict[str, Any]:
    cfg = json.loads((model_dir / "config.json").read_text())
    return cfg.get("text_config", cfg)


def _tensor_bytes(items: list[torch.Tensor | None]) -> int:
    return sum(_nbytes(t) for t in items if isinstance(t, torch.Tensor))


def _cuda_mem_gib() -> float:
    return float(torch.cuda.memory_allocated() / GIB)


def _peak_once(fn: Callable[[], torch.Tensor]) -> dict[str, float]:
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


def _set_packed_prefill_env(mode: str, layer: int) -> dict[str, str | None]:
    updates = {
        "LYNN_PACKED_PREFILL_SLOW": "1",
        "LYNN_PACKED_PREFILL_SLOW_MODE": mode,
        "LYNN_PACKED_PREFILL_P2E_LAYERS": str(layer),
        "LYNN_PACKED_PREFILL_P2E_BLOCK_T": "32",
        "LYNN_PACKED_PREFILL_P2E_BLOCK_INTER": "8",
        "LYNN_PACKED_PREFILL_P2E_BLOCK_HIDDEN": "128",
        "LYNN_PACKED_PREFILL_P2E_NUM_WARPS": "4",
        "LYNN_PACKED_PREFILL_P2E_DOWN_BLOCK_HIDDEN": "8",
        "LYNN_PACKED_PREFILL_P2E_DOWN_BLOCK_INTER": "512",
        "LYNN_PACKED_PREFILL_P2E_DOWN_NUM_WARPS": "8",
    }
    old = {k: os.environ.get(k) for k in updates}
    os.environ.update(updates)
    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--batches", default="16,64")
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = "cuda"
    model_dir = Path(args.model)
    batches = _parse_batches(args.batches)
    torch.manual_seed(args.seed)

    text_cfg = _model_cfg(model_dir)
    num_experts = int(text_cfg.get("num_experts", 256))
    top_k = int(text_cfg.get("num_experts_per_tok", 8))
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
    layer_cfg["layer_idx"] = int(args.layer)
    packed = _attach_packed_moe(model_dir, args.layer, w, device=device)

    hidden = int(w["mlp.gate.weight"].shape[1])
    xs: dict[int, torch.Tensor] = {
        b: (torch.randn((1, b, hidden), device=device, dtype=torch.float32) * 0.35).to(torch.bfloat16)
        for b in batches
    }
    active_bf16_bytes = _tensor_bytes([w.get("mlp.experts.gate_up_proj"), w.get("mlp.experts.down_proj")])
    packed_active_bytes = _tensor_bytes(list(packed.values()))

    print("=============== STAGE 6 PHASE 2-F ONE-LAYER REPLACEMENT VERIFY ===============", flush=True)
    print(f"model        : {model_dir}", flush=True)
    print(f"layer        : {args.layer}", flush=True)
    print(f"batches      : {batches}", flush=True)
    print(f"shape        : hidden={hidden} experts={num_experts} top_k={top_k}", flush=True)
    print(f"BF16 active  : {active_bf16_bytes / GIB:.3f} GiB", flush=True)
    print(f"packed active: {packed_active_bytes / GIB:.3f} GiB", flush=True)

    os.environ["LYNN_PACKED_PREFILL_SLOW"] = "0"
    refs = {b: _moe_forward(h, w, layer_cfg).detach() for b, h in xs.items()}
    bench_bf16 = {
        str(b): _bench_cuda(lambda h=h: _moe_forward(h, w, layer_cfg), warmup=args.warmup, iters=args.iters, repeats=args.repeats)
        for b, h in xs.items()
    }

    del w["mlp.experts.gate_up_proj"], w["mlp.experts.down_proj"]
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mem_after_delete = _cuda_mem_gib()

    numeric: dict[str, Any] = {}
    bench_stream: dict[str, Any] = {}
    bench_p2e: dict[str, Any] = {}
    peak_stream: dict[str, Any] = {}
    peak_p2e: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for b, h in xs.items():
        old = _set_packed_prefill_env("stream_bf16", args.layer)
        stream_fn = lambda h=h: _moe_forward(h, w, layer_cfg)
        stream = stream_fn().detach()
        numeric[f"stream_M{b}_vs_bf16"] = _diff_stats(stream, refs[b])
        peak_stream[str(b)] = _peak_once(stream_fn)
        bench_stream[str(b)] = _bench_cuda(stream_fn, warmup=args.warmup, iters=args.iters, repeats=args.repeats)
        _restore_env(old)

        old = _set_packed_prefill_env("p2e_hybrid", args.layer)
        p2e_fn = lambda h=h: _moe_forward(h, w, layer_cfg)
        p2e = p2e_fn().detach()
        numeric[f"p2e_M{b}_vs_bf16"] = _diff_stats(p2e, refs[b])
        numeric[f"p2e_M{b}_vs_stream"] = _diff_stats(p2e, stream)
        peak_p2e[str(b)] = _peak_once(p2e_fn)
        bench_p2e[str(b)] = _bench_cuda(p2e_fn, warmup=args.warmup, iters=args.iters, repeats=args.repeats)
        _restore_env(old)

        bf = bench_bf16[str(b)]["median_us"]
        st = bench_stream[str(b)]["median_us"]
        p2 = bench_p2e[str(b)]["median_us"]
        rows.append({
            "batch": b,
            "bf16_full_us": bf,
            "stream_bf16_us": st,
            "p2e_hybrid_us": p2,
            "p2e_vs_bf16": bf / p2 if p2 else None,
            "p2e_vs_stream": st / p2 if p2 else None,
        })
        print(
            f"[M={b}] bf16={bf:.2f}us stream={st:.2f}us p2e={p2:.2f}us "
            f"p2e_vs_bf16={bf / p2 if p2 else float('nan'):.3f}x",
            flush=True,
        )

    numeric_pass = all(v["cosine"] > 0.999 and v["argmax_match"] for v in numeric.values())
    no_shadow_pass = "mlp.experts.gate_up_proj" not in w and "mlp.experts.down_proj" not in w
    speed_pass = all(r["p2e_vs_bf16"] >= 1.0 and r["p2e_vs_stream"] >= 10.0 for r in rows)
    result = {
        "schema": "lynn-stage6-p2f-one-layer-replacement-verify-v1",
        "model": str(model_dir),
        "layer": args.layer,
        "seed": args.seed,
        "batches": batches,
        "env": {
            "LYNN_PACKED_PREFILL_SLOW": "1",
            "LYNN_PACKED_PREFILL_SLOW_MODE": "p2e_hybrid",
            "LYNN_PACKED_PREFILL_P2E_LAYERS": str(args.layer),
            "LYNN_PACKED_PREFILL_P2E_BLOCK_INTER": "8",
        },
        "shape": {"hidden": hidden, "num_experts": num_experts, "top_k": top_k},
        "bytes": {
            "bf16_layer_active_experts": active_bf16_bytes,
            "packed_layer_active_experts": packed_active_bytes,
            "mem_after_deleting_bf16_active_gib": mem_after_delete,
        },
        "numeric": numeric,
        "bench": {
            "rows": rows,
            "bf16_full_moe": bench_bf16,
            "stream_bf16": bench_stream,
            "p2e_hybrid": bench_p2e,
        },
        "memory": {
            "stream_peak": peak_stream,
            "p2e_peak": peak_p2e,
        },
        "passes": {
            "numeric": bool(numeric_pass),
            "no_bf16_active_shadow": bool(no_shadow_pass),
            "speed_vs_bf16_and_stream": bool(speed_pass),
            "all": bool(numeric_pass and no_shadow_pass and speed_pass),
        },
        "notes": [
            "This verifies engine dispatch, not only harness composition.",
            "Default remains off. p2e_hybrid only runs when LYNN_PACKED_PREFILL_SLOW=1 and mode=p2e_hybrid.",
            "Layer filtering is controlled by LYNN_PACKED_PREFILL_P2E_LAYERS.",
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
