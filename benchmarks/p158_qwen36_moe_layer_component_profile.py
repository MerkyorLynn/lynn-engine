#!/usr/bin/env python3
"""P158 · Profile real Qwen3.6 MoE layer components.

P157 corrected the active top-k Triton stage timing to about 0.052 ms on p138
fixtures. This probe loads the real resident 35B layer weights and measures the
whole MoE sub-layer around that exact active stage: router/top-k, active
gate/down, shared expert, shared gate/add, and total layer MoE.
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.nvfp4_runtime import dual_scalar_bridge  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.nvfp4_moe import (  # noqa: E402
    nvfp4_grouped_down_weighted_sum,
    nvfp4_grouped_gate_up_silu_fast_decode,
)


def _load_fixture(path: Path, device: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load as load_buffer
    from safetensors.torch import load_file

    if len(path.suffixes) >= 2 and path.suffixes[-2:] == [".safetensors", ".gz"]:
        with gzip.open(str(path), "rb") as f:
            raw = f.read()
        return {k: v.to(device) for k, v in load_buffer(raw).items()}
    return load_file(str(path), device=device)


def _bench_ms(fn: Callable[[], Any], *, warmup: int, iters: int) -> float:
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


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return statistics.mean(values) if values else None


def _shared_expert_forward(h_flat: torch.Tensor, w: dict[str, Any]) -> torch.Tensor | None:
    if "mlp.shared_expert.gate_proj.weight" not in w:
        return None
    if (
        "mlp.shared_expert.gate_proj.weight.packed" in w
        and "mlp.shared_expert.up_proj.weight.packed" in w
        and "mlp.shared_expert.down_proj.weight.packed" in w
    ):
        gate_s, up_s = dual_scalar_bridge(
            h_flat[0],
            w["mlp.shared_expert.gate_proj.weight.packed"],
            w["mlp.shared_expert.up_proj.weight.packed"],
        )
        return w["mlp.shared_expert.down_proj.weight.packed"]((F.silu(gate_s) * up_s).to(h_flat.dtype)).reshape_as(h_flat)
    if "mlp.shared_expert._gate_up_proj.weight" in w:
        gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
        gate_s, up_s = gate_up_s.chunk(2, dim=-1)
    else:
        gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
        up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
    return F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])


def _shared_finalize(h_flat: torch.Tensor, moe_out: torch.Tensor, shared: torch.Tensor | None, w: dict[str, Any]) -> torch.Tensor:
    if shared is None:
        return moe_out
    if "mlp.shared_expert_gate.weight" in w:
        shared = shared * torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
    return moe_out + shared


def _profile_one(
    *,
    h_flat: torch.Tensor,
    w: dict[str, Any],
    fixture: dict[str, torch.Tensor],
    top_k: int,
    warmup: int,
    iters: int,
) -> dict[str, float | int]:
    def router_fn() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = F.linear(h_flat, w["mlp.gate.weight"])
        route, ids = torch.topk(logits, top_k, dim=-1, sorted=False)
        route = F.softmax(route, dim=-1, dtype=torch.float32)[0].contiguous()
        ids = ids[0].to(torch.int32).contiguous()
        return logits, ids, route

    logits, expert_ids, routing_weights = router_fn()
    hidden = h_flat[0].contiguous()
    # p138 fixtures are already repacked into slot order for the active top-k
    # expert set. Use slot ids for those slot-packed tensors while timing the
    # real router above as its own component.
    slot_top_k = int(fixture["slot_gate_up_packed"].shape[0])
    slot_ids = torch.arange(slot_top_k, device=hidden.device, dtype=torch.int32)
    slot_routing_weights = fixture["routing_weights"].to(torch.float32).contiguous()

    def gate_fn() -> torch.Tensor:
        return nvfp4_grouped_gate_up_silu_fast_decode(
            hidden,
            slot_ids,
            fixture["slot_gate_up_packed"].contiguous(),
            fixture["slot_gate_up_scale"].contiguous(),
            fixture["slot_gate_up_global_scale"].to(hidden.device).contiguous(),
            block_inter=8,
            block_hidden=256,
            num_warps=4,
        )

    inter = gate_fn().contiguous()

    def down_fn() -> torch.Tensor:
        return nvfp4_grouped_down_weighted_sum(
            inter,
            slot_ids,
            slot_routing_weights,
            fixture["slot_down_packed"].contiguous(),
            fixture["slot_down_scale"].contiguous(),
            fixture["slot_down_global_scale"].to(hidden.device).contiguous(),
            block_hidden=8,
            block_inter=512,
            num_warps=8,
        )

    def active_fn() -> torch.Tensor:
        inter_local = gate_fn()
        return nvfp4_grouped_down_weighted_sum(
            inter_local,
            slot_ids,
            slot_routing_weights,
            fixture["slot_down_packed"].contiguous(),
            fixture["slot_down_scale"].contiguous(),
            fixture["slot_down_global_scale"].to(hidden.device).contiguous(),
            block_hidden=8,
            block_inter=512,
            num_warps=8,
        ).reshape_as(h_flat)

    active_out = active_fn()

    def shared_fn() -> torch.Tensor | None:
        return _shared_expert_forward(h_flat, w)

    shared = shared_fn()

    def finalize_fn() -> torch.Tensor:
        return _shared_finalize(h_flat, active_out, shared, w)

    def total_fn() -> torch.Tensor:
        logits_local, _ids_local, _route_local = router_fn()
        del logits_local
        inter_local = nvfp4_grouped_gate_up_silu_fast_decode(
            hidden,
            slot_ids,
            fixture["slot_gate_up_packed"].contiguous(),
            fixture["slot_gate_up_scale"].contiguous(),
            fixture["slot_gate_up_global_scale"].to(hidden.device).contiguous(),
            block_inter=8,
            block_hidden=256,
            num_warps=4,
        )
        active_local = nvfp4_grouped_down_weighted_sum(
            inter_local,
            slot_ids,
            slot_routing_weights,
            fixture["slot_down_packed"].contiguous(),
            fixture["slot_down_scale"].contiguous(),
            fixture["slot_down_global_scale"].to(hidden.device).contiguous(),
            block_hidden=8,
            block_inter=512,
            num_warps=8,
        ).reshape_as(h_flat)
        shared_local = _shared_expert_forward(h_flat, w)
        return _shared_finalize(h_flat, active_local, shared_local, w)

    router_ms = _bench_ms(router_fn, warmup=warmup, iters=iters)
    gate_ms = _bench_ms(gate_fn, warmup=warmup, iters=iters)
    down_ms = _bench_ms(down_fn, warmup=warmup, iters=iters)
    active_ms = _bench_ms(active_fn, warmup=warmup, iters=iters)
    shared_ms = _bench_ms(shared_fn, warmup=warmup, iters=iters) if shared is not None else 0.0
    finalize_ms = _bench_ms(finalize_fn, warmup=warmup, iters=iters)
    total_ms = _bench_ms(total_fn, warmup=warmup, iters=iters)

    return {
        "router_ms": router_ms,
        "gateup_ms": gate_ms,
        "down_ms": down_ms,
        "active_combined_ms": active_ms,
        "shared_expert_ms": shared_ms,
        "finalize_ms": finalize_ms,
        "total_moe_ms": total_ms,
        "active_plus_shared_plus_router_ms": router_ms + active_ms + shared_ms + finalize_ms,
        "topk": top_k,
        "slot_topk": slot_top_k,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Profile Qwen3.6 MoE layer components.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--packed-fixtures", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--max-fixtures", type=int, default=18)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    t0 = time.time()
    runner = LynnIncrementalRunner(args.model, device=args.device, dtype=dtype, max_seq_len=4096, verbose=True)
    load_seconds = time.time() - t0

    packed_dir = Path(args.packed_fixtures)
    manifest = json.loads((packed_dir / "manifest.json").read_text())
    rows: list[dict[str, Any]] = []
    for entry in manifest["fixtures"][: args.max_fixtures]:
        layer_id = int(entry["layer_id"])
        prompt_id = int(entry["prompt_id"])
        data = _load_fixture(packed_dir / entry["fixture_file"], args.device)
        h_flat = data["hidden_in"].to(dtype).view(1, -1).contiguous()
        w = runner.layer_weights[layer_id]
        cfg = runner.layer_cfgs[layer_id]
        if not cfg.get("is_moe", int(cfg.get("num_experts", 0) or 0) > 0):
            continue
        row = {
            "fixture_file": entry["fixture_file"],
            "layer_id": layer_id,
            "prompt_id": prompt_id,
            **_profile_one(
                h_flat=h_flat,
                w=w,
                fixture=data,
                top_k=int(cfg["num_experts_per_tok"]),
                warmup=args.warmup,
                iters=args.iters,
            ),
        }
        rows.append(row)
        print(
            f"  L{layer_id:02d}/P{prompt_id:02d} total={row['total_moe_ms']:.4f} "
            f"router={row['router_ms']:.4f} active={row['active_combined_ms']:.4f} "
            f"shared={row['shared_expert_ms']:.4f} finalize={row['finalize_ms']:.4f}",
            flush=True,
        )

    report = {
        "schema": "lynn-p158-qwen36-moe-layer-component-profile-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "packed_fixtures": str(packed_dir),
        "load_seconds": load_seconds,
        "total": len(rows),
        "means": {
            "router_ms": _mean(rows, "router_ms"),
            "gateup_ms": _mean(rows, "gateup_ms"),
            "down_ms": _mean(rows, "down_ms"),
            "active_combined_ms": _mean(rows, "active_combined_ms"),
            "shared_expert_ms": _mean(rows, "shared_expert_ms"),
            "finalize_ms": _mean(rows, "finalize_ms"),
            "total_moe_ms": _mean(rows, "total_moe_ms"),
            "active_plus_shared_plus_router_ms": _mean(rows, "active_plus_shared_plus_router_ms"),
        },
        "results": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(out_path), "means": report["means"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
