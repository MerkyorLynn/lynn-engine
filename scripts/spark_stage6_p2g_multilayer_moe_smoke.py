#!/usr/bin/env python3
"""Stage 6 Phase 2-G: multi-layer p2e_hybrid MoE no-reload smoke.

P2-F verified one layer through engine dispatch. P2-G chains several MoE layers
on synthetic hidden states to measure cumulative numeric drift, memory, and
latency after deleting active BF16 expert shadows.
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


def _parse_layers(text: str) -> list[int]:
    out: list[int] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if "-" in raw:
            lo, hi = raw.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(raw))
    return sorted(dict.fromkeys(out))


def _model_cfg(model_dir: Path) -> dict[str, Any]:
    cfg = json.loads((model_dir / "config.json").read_text())
    return cfg.get("text_config", cfg)


def _cuda_mem_gib() -> float:
    return float(torch.cuda.memory_allocated() / GIB)


def _tensor_bytes(items: list[torch.Tensor | None]) -> int:
    return sum(_nbytes(t) for t in items if isinstance(t, torch.Tensor))


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


def _set_mode(mode: str, layers: list[int]) -> dict[str, str | None]:
    updates = {
        "LYNN_PACKED_PREFILL_SLOW": "1",
        "LYNN_PACKED_PREFILL_SLOW_MODE": mode,
        "LYNN_PACKED_PREFILL_P2E_LAYERS": ",".join(str(x) for x in layers),
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


def _run_layers(h: torch.Tensor, layers: list[tuple[dict[str, Any], dict[str, Any]]]) -> torch.Tensor:
    out = h
    for w, cfg in layers:
        # Keep synthetic hidden states in a realistic residual-scale regime.
        # Chaining raw MoE outputs collapses toward zero and makes cosine
        # meaningless; transformer blocks add the FFN output to the residual.
        out = out + _moe_forward(out, w, cfg)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layers", default="0-3")
    ap.add_argument("--batches", default="16,64")
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = "cuda"
    model_dir = Path(args.model)
    layer_ids = _parse_layers(args.layers)
    batches = _parse_batches(args.batches)
    torch.manual_seed(args.seed)

    text_cfg = _model_cfg(model_dir)
    num_experts = int(text_cfg.get("num_experts", 256))
    top_k = int(text_cfg.get("num_experts_per_tok", 8))
    loaded: list[tuple[dict[str, Any], dict[str, Any]]] = []
    packed_bytes = 0
    active_bf16_bytes = 0
    for layer_idx in layer_ids:
        w, cfg = load_qwen36_layer(
            str(model_dir),
            layer_idx,
            num_experts=num_experts,
            device=device,
            dequant_dtype=torch.bfloat16,
        )
        cfg["num_experts"] = int(cfg.get("num_experts", num_experts))
        cfg["num_experts_per_tok"] = top_k
        cfg["is_moe"] = True
        cfg["layer_idx"] = int(layer_idx)
        packed = _attach_packed_moe(model_dir, layer_idx, w, device=device)
        active_bf16_bytes += _tensor_bytes([w.get("mlp.experts.gate_up_proj"), w.get("mlp.experts.down_proj")])
        packed_bytes += _tensor_bytes(list(packed.values()))
        loaded.append((w, cfg))

    hidden = int(loaded[0][0]["mlp.gate.weight"].shape[1])
    xs: dict[int, torch.Tensor] = {
        b: (torch.randn((1, b, hidden), device=device, dtype=torch.float32) * 0.35).to(torch.bfloat16)
        for b in batches
    }
    print("=============== STAGE 6 PHASE 2-G MULTI-LAYER MOE SMOKE ===============", flush=True)
    print(f"model        : {model_dir}", flush=True)
    print(f"layers       : {layer_ids}", flush=True)
    print(f"batches      : {batches}", flush=True)
    print(f"BF16 active  : {active_bf16_bytes / GIB:.3f} GiB", flush=True)
    print(f"packed active: {packed_bytes / GIB:.3f} GiB", flush=True)

    os.environ["LYNN_PACKED_PREFILL_SLOW"] = "0"
    refs = {b: _run_layers(h, loaded).detach() for b, h in xs.items()}
    bench_bf16 = {
        str(b): _bench_cuda(lambda h=h: _run_layers(h, loaded), warmup=args.warmup, iters=args.iters, repeats=args.repeats)
        for b, h in xs.items()
    }

    for w, _cfg in loaded:
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
        old = _set_mode("stream_bf16", layer_ids)
        stream_fn = lambda h=h: _run_layers(h, loaded)
        stream = stream_fn().detach()
        numeric[f"stream_M{b}_vs_bf16"] = _diff_stats(stream, refs[b])
        peak_stream[str(b)] = _peak_once(stream_fn)
        bench_stream[str(b)] = _bench_cuda(stream_fn, warmup=args.warmup, iters=args.iters, repeats=args.repeats)
        _restore_env(old)

        old = _set_mode("p2e_hybrid", layer_ids)
        p2e_fn = lambda h=h: _run_layers(h, loaded)
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
            f"[M={b}] layers={len(layer_ids)} bf16={bf:.2f}us stream={st:.2f}us "
            f"p2e={p2:.2f}us p2e_vs_bf16={bf / p2 if p2 else float('nan'):.3f}x",
            flush=True,
        )

    numeric_pass = all(v["cosine"] > 0.999 and v["argmax_match"] for v in numeric.values())
    no_shadow_pass = all("mlp.experts.gate_up_proj" not in w and "mlp.experts.down_proj" not in w for w, _ in loaded)
    speed_pass = all(r["p2e_vs_stream"] >= 10.0 for r in rows)
    result = {
        "schema": "lynn-stage6-p2g-multilayer-moe-smoke-v1",
        "model": str(model_dir),
        "layers": layer_ids,
        "seed": args.seed,
        "batches": batches,
        "shape": {"hidden": hidden, "num_experts": num_experts, "top_k": top_k},
        "bytes": {
            "bf16_active_experts": active_bf16_bytes,
            "packed_active_experts": packed_bytes,
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
            "speed_vs_stream": bool(speed_pass),
            "all": bool(numeric_pass and no_shadow_pass and speed_pass),
        },
        "notes": [
            "This is MoE-only multi-layer smoke with residual addition on synthetic hidden states, not full transformer prefill.",
            "The speed gate requires beating stream_bf16 by at least 10x; BF16 parity is reported but not required for P2-G.",
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
