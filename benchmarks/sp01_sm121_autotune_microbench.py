#!/usr/bin/env python3
"""SP-01 microbench: Triton autotune vs static MoE kernels on Spark sm_121.

Validates kernel-level parity + speedup of the autotuned gate_up + down MoE
kernels against the production static configs. Run on Spark inside the engine
venv after pulling `spark/sm121-port`. The user-facing TPS bench
(`benchmarks/lynn_27b_vs_35b.py`) is separate — this one isolates the
kernel-level delta so we can attribute server-TPS movement.

Output: JSON report under `--out`, including the Triton-autotune-picked
configs and isolated per-launch latencies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p10e_packed_active_expert_probe import (  # noqa: E402
    _load_grouped,
    _prefill_to_layer_input,
)
from engine.full_forward import _rms_norm  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.nvfp4_moe import (  # noqa: E402
    _grouped_down_weighted_sum_kernel_sp01_autotuned,
    _grouped_gate_up_silu_kernel_sp01_autotuned,
    nvfp4_grouped_down_weighted_sum,
    nvfp4_grouped_down_weighted_sum_sp01_autotuned,
    nvfp4_grouped_gate_up_silu,
    nvfp4_grouped_gate_up_silu_sp01_autotuned,
)


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


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    denom = af.norm() * bf.norm()
    if denom.item() == 0.0:
        return 0.0
    return float((af @ bf) / denom)


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-9))


def _picked_config(kernel) -> dict:
    """Extract the autotune-picked config from a Triton autotuner."""
    cache = getattr(kernel, "cache", None)
    if not cache:
        return {"status": "no cache populated yet"}
    entries = []
    for key, cfg in cache.items():
        entries.append(
            {
                "key": [int(k) if hasattr(k, "__int__") else str(k) for k in key],
                "kwargs": dict(cfg.kwargs),
                "num_warps": cfg.num_warps,
                "num_stages": cfg.num_stages,
            }
        )
    return {"entries": entries}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to Lynn 27B NVFP4 model dir")
    ap.add_argument("--layer", type=int, default=6, help="layer index to probe")
    ap.add_argument("--prompt", default="解释 MoE 中的 active parameters 概念")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args()

    model_dir = Path(args.model)
    print(f"[sp01] loading runner from {model_dir}", flush=True)
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)

    print(f"[sp01] prefilling layer {args.layer} for prompt input", flush=True)
    h_layer, _ = _prefill_to_layer_input(runner, args.layer, args.prompt)
    w = runner.layer_weights[args.layer]
    cfg = runner.layer_cfgs[args.layer]
    h_moe = _rms_norm(h_layer, w["post_attention_layernorm.weight"])
    h_flat = h_moe.view(-1, h_moe.shape[-1])
    hidden = h_flat[0]
    top_k = int(cfg["num_experts_per_tok"])

    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(router_logits, top_k, dim=-1)
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32)[0]
    expert_ids = expert_indices[0].to(torch.long)

    gate_up_packed, gate_up_scale, gate_up_global = _load_grouped(
        model_dir,
        f"model.language_model.layers.{args.layer}.mlp.experts.gate_up_proj",
        runner.device,
    )
    down_packed, down_scale, down_global = _load_grouped(
        model_dir,
        f"model.language_model.layers.{args.layer}.mlp.experts.down_proj",
        runner.device,
    )

    # --- Static baseline ---
    def static_gateup() -> torch.Tensor:
        return nvfp4_grouped_gate_up_silu(
            hidden, expert_ids, gate_up_packed, gate_up_scale, gate_up_global,
            block_inter=8, block_hidden=64,
        )

    inter_static = static_gateup()

    def static_down() -> torch.Tensor:
        return nvfp4_grouped_down_weighted_sum(
            inter_static, expert_ids, routing_weights,
            down_packed, down_scale, down_global,
            block_hidden=8, block_inter=256,
        )

    out_static = static_down()

    # --- Autotuned ---
    print("[sp01] first autotuned gate_up call — Triton sweeping 17 configs", flush=True)
    t0 = time.time()
    inter_auto = nvfp4_grouped_gate_up_silu_sp01_autotuned(
        hidden, expert_ids, gate_up_packed, gate_up_scale, gate_up_global,
    )
    torch.cuda.synchronize()
    gateup_autotune_first_ms = (time.time() - t0) * 1000.0

    print("[sp01] first autotuned down call — Triton sweeping 17 configs", flush=True)
    t0 = time.time()
    out_auto = nvfp4_grouped_down_weighted_sum_sp01_autotuned(
        inter_auto, expert_ids, routing_weights,
        down_packed, down_scale, down_global,
    )
    torch.cuda.synchronize()
    down_autotune_first_ms = (time.time() - t0) * 1000.0

    # Parity vs static
    gateup_cos = _cosine(inter_auto, inter_static)
    gateup_rel = _rel_l2(inter_auto, inter_static)
    out_cos = _cosine(out_auto, out_static)
    out_rel = _rel_l2(out_auto, out_static)

    print(f"[sp01] gateup cos={gateup_cos:.7f} rel_l2={gateup_rel:.2e}")
    print(f"[sp01] active-MoE cos={out_cos:.7f} rel_l2={out_rel:.2e}")

    # --- Steady-state timings ---
    print(f"[sp01] timing static gate_up (warmup={args.warmup} iters={args.iters})", flush=True)
    static_gateup_ms = _bench(static_gateup, args.warmup, args.iters)
    print(f"[sp01] timing static down", flush=True)
    static_down_ms = _bench(static_down, args.warmup, args.iters)
    print(f"[sp01] static gate_up+down = {static_gateup_ms + static_down_ms:.4f} ms/layer", flush=True)

    def autotuned_gateup() -> torch.Tensor:
        return nvfp4_grouped_gate_up_silu_sp01_autotuned(
            hidden, expert_ids, gate_up_packed, gate_up_scale, gate_up_global,
        )

    inter_auto_cached = autotuned_gateup()

    def autotuned_down() -> torch.Tensor:
        return nvfp4_grouped_down_weighted_sum_sp01_autotuned(
            inter_auto_cached, expert_ids, routing_weights,
            down_packed, down_scale, down_global,
        )

    print(f"[sp01] timing autotuned gate_up", flush=True)
    auto_gateup_ms = _bench(autotuned_gateup, args.warmup, args.iters)
    print(f"[sp01] timing autotuned down", flush=True)
    auto_down_ms = _bench(autotuned_down, args.warmup, args.iters)
    print(f"[sp01] autotuned gate_up+down = {auto_gateup_ms + auto_down_ms:.4f} ms/layer", flush=True)

    speedup_gateup = static_gateup_ms / auto_gateup_ms if auto_gateup_ms > 0 else float("nan")
    speedup_down = static_down_ms / auto_down_ms if auto_down_ms > 0 else float("nan")
    speedup_combined = (
        (static_gateup_ms + static_down_ms) / (auto_gateup_ms + auto_down_ms)
        if (auto_gateup_ms + auto_down_ms) > 0
        else float("nan")
    )

    print(f"[sp01] gate_up speedup = {speedup_gateup:.3f}x")
    print(f"[sp01] down speedup    = {speedup_down:.3f}x")
    print(f"[sp01] combined        = {speedup_combined:.3f}x")

    report = {
        "type": "sp01_sm121_autotune_microbench",
        "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "model_dir": str(model_dir),
        "layer": args.layer,
        "prompt": args.prompt,
        "warmup": args.warmup,
        "iters": args.iters,
        "top_k": top_k,
        "autotune_picked": {
            "gate_up": _picked_config(_grouped_gate_up_silu_kernel_sp01_autotuned),
            "down": _picked_config(_grouped_down_weighted_sum_kernel_sp01_autotuned),
        },
        "first_call_ms": {
            "gate_up_autotune_sweep": gateup_autotune_first_ms,
            "down_autotune_sweep": down_autotune_first_ms,
        },
        "static_ms": {
            "gate_up": static_gateup_ms,
            "down": static_down_ms,
            "combined": static_gateup_ms + static_down_ms,
            "config_gate_up": {"BLOCK_INTER": 8, "BLOCK_HIDDEN": 64, "num_warps": 4},
            "config_down": {"BLOCK_HIDDEN": 8, "BLOCK_INTER": 256, "num_warps": 4},
        },
        "autotuned_ms": {
            "gate_up": auto_gateup_ms,
            "down": auto_down_ms,
            "combined": auto_gateup_ms + auto_down_ms,
        },
        "speedup": {
            "gate_up": speedup_gateup,
            "down": speedup_down,
            "combined": speedup_combined,
        },
        "parity": {
            "gate_up_cosine_vs_static": gateup_cos,
            "gate_up_rel_l2_vs_static": gateup_rel,
            "active_moe_cosine_vs_static": out_cos,
            "active_moe_rel_l2_vs_static": out_rel,
        },
        "promotion_gate": {
            "cosine_min_ok": min(gateup_cos, out_cos) >= 0.9999,
            "combined_speedup_threshold": 1.05,
            "combined_speedup_ok": speedup_combined >= 1.05,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[sp01] report written: {out_path}")
    print(f"[sp01] promotion_gate.cosine_min_ok = {report['promotion_gate']['cosine_min_ok']}")
    print(f"[sp01] promotion_gate.combined_speedup_ok = {report['promotion_gate']['combined_speedup_ok']}")
    return 0 if report["promotion_gate"]["cosine_min_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
