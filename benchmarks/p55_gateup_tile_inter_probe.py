#!/usr/bin/env python3
"""P55: tile-inter CUDA scalar gate/up probe.

P48 showed that down projection can get a local win from a non-atomic tiled
shape, but tiny accumulation drift can still flip full-generate greedy output.
P55 tests the analogous gate/up-side shape before we jump into the real
per-16 grouped native-FP4 tensor-core kernel:

    x[2048] + selected experts -> inter[top_k,512]

The tile-inter kernel computes multiple intermediate rows per CUDA block,
reusing the hidden vector load and reducing block count. It is still scalar
math, so a win here is an ABI/tile-shape signal, not the final 155TPS kernel.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p10e_packed_active_expert_probe import _load_grouped, _prefill_to_layer_input  # noqa: E402
from engine.full_forward import _rms_norm  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.nvfp4_moe import nvfp4_grouped_gate_up_silu  # noqa: E402


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


def _parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _load_native(build_dir: str):
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    python_bin = Path(sys.executable).resolve().parent
    os.environ["PATH"] = f"{python_bin}:{os.environ.get('PATH', '')}"
    cuda_home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    nvcc = Path(cuda_home) / "bin" / "nvcc"
    if nvcc.exists():
        os.environ["PATH"] = f"{nvcc.parent}:{os.environ.get('PATH', '')}"
    Path(build_dir).mkdir(parents=True, exist_ok=True)
    return load(
        name="lynn_native_p55_gateup_tile_inter",
        sources=[
            str(ROOT / "csrc" / "lynn_native" / "bindings.cpp"),
            str(ROOT / "csrc" / "lynn_native" / "smoke_kernel.cu"),
            str(ROOT / "csrc" / "lynn_native" / "moe_scalar_kernel.cu"),
        ],
        build_directory=build_dir,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
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


def _diff(ref: torch.Tensor, out: torch.Tensor) -> dict:
    rf = ref.float().reshape(-1)
    of = out.float().reshape(-1)
    delta = of - rf
    return {
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rel_l2": float(torch.linalg.vector_norm(delta).item() / torch.linalg.vector_norm(rf).item()),
        "cosine": float(F.cosine_similarity(rf, of, dim=0).item()),
    }


def _run_layer(
    module,
    runner: LynnIncrementalRunner,
    model_dir: Path,
    *,
    layer: int,
    prompt: str,
    tile_inters: list[int],
    warmup: int,
    iters: int,
) -> dict:
    h_layer, _ = _prefill_to_layer_input(runner, layer, prompt)
    w = runner.layer_weights[layer]
    cfg = runner.layer_cfgs[layer]
    h_moe = _rms_norm(h_layer, w["post_attention_layernorm.weight"])
    h_flat = h_moe.view(-1, h_moe.shape[-1])
    hidden = h_flat[0].contiguous()
    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    _, expert_indices = torch.topk(
        router_logits,
        int(cfg["num_experts_per_tok"]),
        dim=-1,
        sorted=False,
    )
    expert_ids = expert_indices[0].to(torch.int32).contiguous()
    gate_up_packed, gate_up_scale, gate_up_global = _load_grouped(
        model_dir,
        f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj",
        runner.device,
    )

    def triton_ref() -> torch.Tensor:
        return nvfp4_grouped_gate_up_silu(
            hidden,
            expert_ids,
            gate_up_packed,
            gate_up_scale,
            gate_up_global,
            block_inter=8,
            block_hidden=256,
            num_warps=4,
        )

    def cuda_scalar() -> torch.Tensor:
        return module.gate_up_silu_scalar(
            hidden,
            expert_ids,
            gate_up_packed,
            gate_up_scale,
            gate_up_global,
        )

    ref = triton_ref()
    triton_ms = _bench(triton_ref, warmup, iters)
    scalar_out = cuda_scalar()
    scalar_ms = _bench(cuda_scalar, warmup, iters)
    variants = {
        "cuda_scalar": {
            "diff_vs_triton_ref": _diff(ref, scalar_out),
            "ms": scalar_ms,
            "speedup_vs_triton": triton_ms / scalar_ms,
        }
    }
    for tile_inter in tile_inters:
        def cuda_tile(tile_inter: int = tile_inter) -> torch.Tensor:
            return module.gate_up_silu_tile_inter_scalar(
                hidden,
                expert_ids,
                gate_up_packed,
                gate_up_scale,
                gate_up_global,
                int(tile_inter),
            )

        out = cuda_tile()
        ms = _bench(cuda_tile, warmup, iters)
        variants[f"tile_inter_{tile_inter}"] = {
            "diff_vs_triton_ref": _diff(ref, out),
            "ms": ms,
            "speedup_vs_triton": triton_ms / ms,
            "speedup_vs_cuda_scalar": scalar_ms / ms,
        }

    best = min(variants.items(), key=lambda item: item[1]["ms"])
    return {
        "layer": layer,
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "triton_ref_ms": triton_ms,
        "variants": variants,
        "best_variant": best[0],
        "best_ms": best[1]["ms"],
        "best_speedup_vs_triton": triton_ms / best[1]["ms"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="4,16,28,36")
    ap.add_argument("--tile-inters", default="1,2,4,8")
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--build-dir", default="/tmp/lynn_engine_native_build/p55_gateup_tile_inter")
    args = ap.parse_args()

    t0 = time.time()
    module = _load_native(args.build_dir)
    build_s = time.time() - t0
    model_dir = Path(args.model)
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    layers = _parse_ints(args.layers)
    tile_inters = _parse_ints(args.tile_inters)
    cases = [
        _run_layer(
            module,
            runner,
            model_dir,
            layer=layer,
            prompt=args.prompt,
            tile_inters=tile_inters,
            warmup=args.warmup,
            iters=args.iters,
        )
        for layer in layers
    ]
    all_exactish = all(
        variant["diff_vs_triton_ref"]["cosine"] >= 0.999999
        for case in cases
        for variant in case["variants"].values()
    )
    best_rows = [
        {
            "layer": case["layer"],
            "best_variant": case["best_variant"],
            "best_ms": case["best_ms"],
            "best_speedup_vs_triton": case["best_speedup_vs_triton"],
        }
        for case in cases
    ]
    result = {
        "schema_version": "lynn-engine-p55-gateup-tile-inter-probe-v1",
        "model": args.model,
        "layers": layers,
        "tile_inters": tile_inters,
        "build_seconds": build_s,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "nvcc_version": _run(["nvcc", "--version"]),
        "cases": cases,
        "summary": {
            "all_variants_exactish": all_exactish,
            "best_rows": best_rows,
            "best_speedup_vs_triton_max": max(row["best_speedup_vs_triton"] for row in best_rows),
            "best_speedup_vs_triton_min": min(row["best_speedup_vs_triton"] for row in best_rows),
        },
        "notes": [
            "Tile-inter scalar kernels test ABI/tile shape only; they are not native FP4 tensor-core kernels.",
            "A speedup here would guide grouped kernel tiling; a slowdown closes this scalar shortcut.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all_exactish else 1


if __name__ == "__main__":
    raise SystemExit(main())
