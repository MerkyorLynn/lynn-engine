#!/usr/bin/env python3
"""P29: native CUDA down weighted-sum contract probe.

P28 proved the native extension can compute gate/up intermediate tensors from
the real packed NVFP4 expert layout. P29 covers the second half of active MoE:

    inter[top_k, 512] + expert_ids[top_k] + routing_weights[top_k]
        + down_packed/down_scale/global
        -> hidden[2048]

As in P28, this is a scalar CUDA reference. The goal is contract parity against
the production Triton path, not speed promotion.
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
from triton_kernels.nvfp4_moe import (  # noqa: E402
    nvfp4_grouped_down_weighted_sum,
    nvfp4_grouped_gate_up_silu,
)


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


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
        name="lynn_native_p29_down",
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
    routing_weights, expert_indices = torch.topk(
        router_logits,
        int(cfg["num_experts_per_tok"]),
        dim=-1,
        sorted=False,
    )
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32)[0].contiguous()
    expert_ids = expert_indices[0].to(torch.int32).contiguous()

    gate_up_packed, gate_up_scale, gate_up_global = _load_grouped(
        model_dir,
        f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj",
        runner.device,
    )
    down_packed, down_scale, down_global = _load_grouped(
        model_dir,
        f"model.language_model.layers.{layer}.mlp.experts.down_proj",
        runner.device,
    )
    inter = nvfp4_grouped_gate_up_silu(
        hidden,
        expert_ids,
        gate_up_packed,
        gate_up_scale,
        gate_up_global,
        block_inter=8,
        block_hidden=256,
        num_warps=4,
    )

    def triton_ref() -> torch.Tensor:
        return nvfp4_grouped_down_weighted_sum(
            inter,
            expert_ids,
            routing_weights,
            down_packed,
            down_scale,
            down_global,
            block_hidden=8,
            block_inter=512,
            num_warps=8,
        )

    def cuda_candidate() -> torch.Tensor:
        return module.down_weighted_sum_scalar(
            inter,
            expert_ids,
            routing_weights,
            down_packed,
            down_scale,
            down_global,
        )

    ref = triton_ref()
    out = cuda_candidate()
    return {
        "layer": layer,
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "diff_vs_triton_ref": _diff(ref, out),
        "triton_ref_ms": _bench(triton_ref, warmup, iters),
        "cuda_scalar_ms": _bench(cuda_candidate, warmup, iters),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[28])
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--build-dir", default="/tmp/lynn_engine_native_build/p29_down")
    args = ap.parse_args()

    t0 = time.time()
    module = _load_native(args.build_dir)
    build_s = time.time() - t0
    model_dir = Path(args.model)
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    cases = [
        _run_layer(
            module,
            runner,
            model_dir,
            layer=layer,
            prompt=args.prompt,
            warmup=args.warmup,
            iters=args.iters,
        )
        for layer in args.layers
    ]
    for case in cases:
        case["cuda_vs_triton_speedup"] = case["triton_ref_ms"] / case["cuda_scalar_ms"]
    promote = all(case["diff_vs_triton_ref"]["cosine"] >= 0.999999 for case in cases)
    result = {
        "schema_version": "lynn-engine-p29-native-down-cuda-probe-v1",
        "model": args.model,
        "layers": args.layers,
        "build_seconds": build_s,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "nvcc_version": _run(["nvcc", "--version"]),
        "cases": cases,
        "promote_contract": promote,
        "promote_speed": all(case["cuda_scalar_ms"] < case["triton_ref_ms"] for case in cases),
        "notes": [
            "CUDA scalar down is a native-extension contract probe, not the final fast kernel.",
            "It completes the gate/up + down active-MoE native extension data contract.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if promote else 1


if __name__ == "__main__":
    raise SystemExit(main())
