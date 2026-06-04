#!/usr/bin/env python3
"""R5-A: synthetic Lynn per-16 NVFP4 -> block-scaled FP4 layout bridge.

This is the first runnable gate after the R6000 FP4-MMA census.  It is model
free by design: use the real Lynn active-MoE gate/up dimensions, generate
deterministic packed E2M1 + per-16 scales, then compare candidate group32
block-scaled routes against an exact per-16 scalar reference.

The probe does not promote a kernel.  It answers the bridge question:

* Can two Lynn per-16 scale groups be folded into one block-scaled group32?
* Can a padded per-16 -> group32 route preserve the layout semantics?
* Does e8m0 scale rounding make current Lynn E4M3-style scales non-zero-copy?
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F

import triton
import triton.language as tl


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@triton.jit
def _dot_scaled_mn_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    M_TOTAL: tl.constexpr,
    K_PACKED_TOTAL: tl.constexpr,
    N_TOTAL: tl.constexpr,
    GROUPS_TOTAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_PACKED: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kp_base = tl.arange(0, BLOCK_K_PACKED)
    offs_g_base = tl.arange(0, BLOCK_K_PACKED // 16)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kp0 in range(0, K_PACKED_TOTAL, BLOCK_K_PACKED):
        offs_kp = kp0 + offs_kp_base
        group_start = kp0 // 16
        offs_g = group_start + offs_g_base
        a = tl.load(
            a_ptr + offs_m[:, None] * K_PACKED_TOTAL + offs_kp[None, :],
            mask=(offs_m[:, None] < M_TOTAL) & (offs_kp[None, :] < K_PACKED_TOTAL),
            other=0,
        )
        b = tl.load(
            b_ptr + offs_n[None, :] * K_PACKED_TOTAL + offs_kp[:, None],
            mask=(offs_n[None, :] < N_TOTAL) & (offs_kp[:, None] < K_PACKED_TOTAL),
            other=0,
        )
        a_s = tl.load(
            a_scale_ptr + offs_m[:, None] * GROUPS_TOTAL + offs_g[None, :],
            mask=(offs_m[:, None] < M_TOTAL) & (offs_g[None, :] < GROUPS_TOTAL),
            other=127,
        )
        b_s = tl.load(
            b_scale_ptr + offs_n[:, None] * GROUPS_TOTAL + offs_g[None, :],
            mask=(offs_n[:, None] < N_TOTAL) & (offs_g[None, :] < GROUPS_TOTAL),
            other=127,
        )
        acc += tl.dot_scaled(
            a,
            a_s,
            "e2m1",
            b,
            b_s,
            "e2m1",
            lhs_k_pack=True,
            rhs_k_pack=True,
        )

    tl.store(
        c_ptr + offs_m[:, None] * N_TOTAL + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < N_TOTAL),
    )


_E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _run(cmd: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        return {
            "cmd": cmd,
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    except Exception as exc:  # noqa: BLE001 - inventory must not crash the probe.
        return {"cmd": cmd, "ok": False, "error": repr(exc)}


def _inventory() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "git_head": _run(["git", "rev-parse", "HEAD"]),
        "git_status": _run(["git", "status", "--short"]),
        "nvidia_smi": _run(["nvidia-smi"]),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        payload["torch"] = {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "memory_gib": props.total_memory / (1024**3),
        }
    return payload


def _prepare_path() -> None:
    python_bin = Path(sys.executable).resolve().parent
    os.environ["PATH"] = f"{python_bin}:{os.environ.get('PATH', '')}"
    for extra_bin in (Path.home() / "miniconda3" / "bin", Path("/root/miniconda3/bin"), Path("/usr/local/cuda/bin")):
        if extra_bin.exists():
            os.environ["PATH"] = f"{extra_bin}:{os.environ.get('PATH', '')}"


def _make_codes(rows: int, k: int, *, device: torch.device, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    mag = torch.randint(0, 8, (rows, k), device=device, generator=gen, dtype=torch.uint8)
    sign = torch.randint(0, 2, (rows, k), device=device, generator=gen, dtype=torch.uint8) << 3
    return (mag | sign).contiguous()


def _pack_codes(codes: torch.Tensor) -> torch.Tensor:
    if codes.shape[-1] % 2 != 0:
        raise ValueError("last dimension must be even to pack E2M1 nibbles")
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def _unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    out = torch.empty((*packed.shape[:-1], packed.shape[-1] * 2), device=packed.device, dtype=torch.uint8)
    out[..., 0::2] = low
    out[..., 1::2] = high
    return out


def _dequant_codes(codes: torch.Tensor, scale16: torch.Tensor) -> torch.Tensor:
    table = _E2M1_MAG.to(codes.device)
    mag = (codes & 0x07).long()
    sign = torch.where((codes & 0x08) != 0, -1.0, 1.0).to(torch.float32)
    values = table[mag] * sign
    groups = codes.shape[-1] // 16
    if scale16.shape[-1] != groups:
        raise ValueError(f"scale16 last dimension {scale16.shape[-1]} does not match expected per-16 groups {groups}")
    scale = scale16.float().reshape(*scale16.shape[:-1], groups, 1).expand(*scale16.shape[:-1], groups, 16)
    return values.reshape(*values.shape[:-1], groups, 16).float().mul(scale).reshape_as(values.float())


def _make_scale16(rows: int, groups: int, *, device: torch.device, case: str) -> torch.Tensor:
    idx = torch.arange(rows * groups, device=device, dtype=torch.float32).reshape(rows, groups)
    exponents = ((idx.remainder(9.0)) - 4.0) / 2.0
    base = torch.pow(torch.full_like(exponents, 2.0), exponents)
    if case == "power2":
        return base.contiguous()
    if case == "e4m3_like":
        mantissas = torch.tensor([1.0, 1.125, 1.25, 1.5, 1.75], device=device, dtype=torch.float32)
        return (base * mantissas[(idx.long() % len(mantissas))]).contiguous()
    raise ValueError(f"unknown scale case {case!r}")


def _fold_pair(scale16: torch.Tensor, mode: str) -> torch.Tensor:
    pair = scale16.float().reshape(*scale16.shape[:-1], scale16.shape[-1] // 2, 2)
    if mode == "max":
        return pair.max(dim=-1).values
    if mode == "mean":
        return pair.mean(dim=-1)
    if mode == "geom":
        return torch.sqrt(pair[..., 0] * pair[..., 1])
    raise ValueError(f"unknown fold mode {mode!r}")


def _to_e8m0_bytes(scale: torch.Tensor) -> torch.Tensor:
    byte = torch.round(torch.log2(scale.float().clamp_min(1.0e-30))) + 127.0
    return byte.clamp(0, 255).to(torch.uint8).contiguous()


def _expand_packed_per16_to_group32(packed: torch.Tensor) -> torch.Tensor:
    if packed.shape[-1] % 8 != 0:
        raise ValueError(f"packed bytes must be divisible by 8, got {tuple(packed.shape)}")
    rows = packed.shape[0]
    groups16 = packed.shape[1] // 8
    src = packed.reshape(rows, groups16, 8)
    out = torch.zeros((rows, groups16, 16), device=packed.device, dtype=torch.uint8)
    out[:, :, :8] = src
    return out.reshape(rows, groups16 * 16).contiguous()


def _dot_scaled_mn(
    act_packed: torch.Tensor,
    act_scale_e8m0: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale_e8m0: torch.Tensor,
    *,
    block_m: int,
    block_n: int,
    block_k_packed: int,
) -> torch.Tensor:
    m = int(act_packed.shape[0])
    n = int(weight_packed.shape[0])
    k_packed = int(weight_packed.shape[1])
    groups = int(weight_scale_e8m0.shape[1])
    out = torch.empty((m, n), device=act_packed.device, dtype=torch.float32)
    _dot_scaled_mn_kernel[(triton.cdiv(m, block_m), triton.cdiv(n, block_n))](
        act_packed.contiguous(),
        act_scale_e8m0.contiguous(),
        weight_packed.contiguous(),
        weight_scale_e8m0.contiguous(),
        out,
        M_TOTAL=m,
        K_PACKED_TOTAL=k_packed,
        N_TOTAL=n,
        GROUPS_TOTAL=groups,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K_PACKED=block_k_packed,
        num_warps=4,
    )
    return out


def _bench(fn: Callable[[], torch.Tensor], warmup: int, repeats: int) -> tuple[torch.Tensor, list[float]]:
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    times_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
    assert out is not None
    return out, times_ms


def _metrics(candidate: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    diff = candidate.float() - ref.float()
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rel_l2": float(torch.linalg.vector_norm(diff).item() / torch.linalg.vector_norm(ref.float()).clamp_min(1e-12).item()),
        "cosine": float(F.cosine_similarity(candidate.float().flatten(), ref.float().flatten(), dim=0).item()),
    }


def _time_stats(times_ms: list[float]) -> dict[str, float]:
    return {
        "median_ms": float(statistics.median(times_ms)),
        "mean_ms": float(statistics.fmean(times_ms)),
        "min_ms": float(min(times_ms)),
        "max_ms": float(max(times_ms)),
    }


def _candidate_row(
    name: str,
    fn: Callable[[], torch.Tensor],
    ref: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
    packed_bytes: int,
    scale_bytes: int,
    original_packed_bytes: int,
    original_scale_bytes: int,
) -> dict[str, Any]:
    out, times = _bench(fn, warmup, repeats)
    row = {
        "candidate": name,
        "metrics": _metrics(out, ref),
        "timing_ms": _time_stats(times),
        "bytes": {
            "packed_bytes": int(packed_bytes),
            "scale_bytes": int(scale_bytes),
            "original_packed_bytes": int(original_packed_bytes),
            "original_scale_bytes": int(original_scale_bytes),
            "packed_ratio_vs_original": float(packed_bytes / max(1, original_packed_bytes)),
            "scale_ratio_vs_original": float(scale_bytes / max(1, original_scale_bytes)),
        },
    }
    return row


def _run_case(args: argparse.Namespace, scale_case: str, m: int, n: int, k: int, device: torch.device) -> dict[str, Any]:
    groups16 = k // 16
    act_codes = _make_codes(m, k, device=device, seed=args.seed + m + 17)
    weight_codes = _make_codes(n, k, device=device, seed=args.seed + n + 31)
    act_scale16 = _make_scale16(m, groups16, device=device, case=scale_case)
    weight_scale16 = _make_scale16(n, groups16, device=device, case=scale_case)
    act_packed = _pack_codes(act_codes)
    weight_packed = _pack_codes(weight_codes)

    act_ref = _dequant_codes(act_codes, act_scale16)
    weight_ref = _dequant_codes(weight_codes, weight_scale16)

    def scalar_reference() -> torch.Tensor:
        return torch.matmul(act_ref, weight_ref.t())

    ref, ref_times = _bench(scalar_reference, args.warmup, args.repeats)

    original_packed_bytes = act_packed.numel() + weight_packed.numel()
    original_scale_bytes = act_scale16.numel() * act_scale16.element_size() + weight_scale16.numel() * weight_scale16.element_size()
    rows: list[dict[str, Any]] = []
    for fold_mode in ("max", "mean", "geom"):
        act_scale32 = _to_e8m0_bytes(_fold_pair(act_scale16, fold_mode))
        weight_scale32 = _to_e8m0_bytes(_fold_pair(weight_scale16, fold_mode))
        rows.append(_candidate_row(
            f"fold_pair_group32_{fold_mode}",
            lambda act_scale32=act_scale32, weight_scale32=weight_scale32: _dot_scaled_mn(
                act_packed,
                act_scale32,
                weight_packed,
                weight_scale32,
                block_m=args.block_m,
                block_n=args.block_n,
                block_k_packed=args.block_k_packed,
            ),
            ref,
            warmup=args.warmup,
            repeats=args.repeats,
            packed_bytes=original_packed_bytes,
            scale_bytes=act_scale32.numel() + weight_scale32.numel(),
            original_packed_bytes=original_packed_bytes,
            original_scale_bytes=original_scale_bytes,
        ))

    padded_act = _expand_packed_per16_to_group32(act_packed)
    padded_weight = _expand_packed_per16_to_group32(weight_packed)
    act_scale_e8m0 = _to_e8m0_bytes(act_scale16)
    weight_scale_e8m0 = _to_e8m0_bytes(weight_scale16)
    rows.append(_candidate_row(
        "padded_per16_group32",
        lambda: _dot_scaled_mn(
            padded_act,
            act_scale_e8m0,
            padded_weight,
            weight_scale_e8m0,
            block_m=args.block_m,
            block_n=args.block_n,
            block_k_packed=args.block_k_packed,
        ),
        ref,
        warmup=args.warmup,
        repeats=args.repeats,
        packed_bytes=padded_act.numel() + padded_weight.numel(),
        scale_bytes=act_scale_e8m0.numel() + weight_scale_e8m0.numel(),
        original_packed_bytes=original_packed_bytes,
        original_scale_bytes=original_scale_bytes,
    ))

    best = min(rows, key=lambda row: float(row["metrics"]["rel_l2"]))
    return {
        "scale_case": scale_case,
        "shape": {"M": m, "N": n, "K": k, "groups16": groups16},
        "reference_timing_ms": _time_stats(ref_times),
        "candidates": rows,
        "best_candidate": best["candidate"],
        "best_rel_l2": best["metrics"]["rel_l2"],
        "best_cosine": best["metrics"]["cosine"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--m-values", default="1,16,64")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--block-m", type=int, default=16)
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--block-k-packed", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--power2-rel-l2-max", type=float, default=1.0e-4)
    ap.add_argument("--e4m3-rel-l2-max-for-zero-copy", type=float, default=2.0e-2)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("R5-A requires CUDA")
    if args.k % 32 != 0:
        raise ValueError("--k must be divisible by 32")

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
    os.environ.setdefault("LYNN_NATIVE_CUDA_ARCH", "sm_120a")
    _prepare_path()
    device = torch.device("cuda")
    m_values = [int(x.strip()) for x in args.m_values.split(",") if x.strip()]
    t0 = time.time()
    cases = [
        _run_case(args, scale_case, m, args.n, args.k, device)
        for scale_case in ("power2", "e4m3_like")
        for m in m_values
    ]
    elapsed_s = time.time() - t0

    power2_padded = [
        row
        for case in cases
        if case["scale_case"] == "power2"
        for row in case["candidates"]
        if row["candidate"] == "padded_per16_group32"
    ]
    e4m3_padded = [
        row
        for case in cases
        if case["scale_case"] == "e4m3_like"
        for row in case["candidates"]
        if row["candidate"] == "padded_per16_group32"
    ]
    fold_rows = [
        row
        for case in cases
        for row in case["candidates"]
        if row["candidate"].startswith("fold_pair_group32")
    ]
    power2_layout_ok = all(row["metrics"]["rel_l2"] <= args.power2_rel_l2_max for row in power2_padded)
    e4m3_zero_copy_ok = all(row["metrics"]["rel_l2"] <= args.e4m3_rel_l2_max_for_zero_copy for row in e4m3_padded)
    fold_pair_ok = all(row["metrics"]["rel_l2"] <= args.power2_rel_l2_max for row in fold_rows)

    decision = "FAIL_R5A_LAYOUT_BRIDGE_INCOMPLETE"
    if power2_layout_ok:
        decision = "PASS_R5A_LAYOUT_BRIDGE_ZERO_COPY_SUPPORTED" if e4m3_zero_copy_ok else "PASS_R5A_LAYOUT_BRIDGE_E8M0_REPACK_REQUIRED"

    result = {
        "schema": "lynn-stage6-r5a-per16-layout-bridge-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "inventory": _inventory(),
        "elapsed_seconds": elapsed_s,
        "dimensions": {
            "H": args.k,
            "I": args.n // 2,
            "gate_up_rows": args.n,
            "m_values": m_values,
        },
        "thresholds": {
            "power2_rel_l2_max": args.power2_rel_l2_max,
            "e4m3_rel_l2_max_for_zero_copy": args.e4m3_rel_l2_max_for_zero_copy,
        },
        "cases": cases,
        "passes": {
            "power2_padded_per16_layout_ok": bool(power2_layout_ok),
            "fold_pair_group32_supported": bool(fold_pair_ok),
            "current_lynn_e4m3_scales_zero_copy_supported": bool(e4m3_zero_copy_ok),
            "banked_layout_bridge": bool(power2_layout_ok),
            "banked_grouped_moe_fp4_mma_poc": False,
            "banked_kernel_speed": False,
            "banked_default_promotion": False,
            "all": bool(power2_layout_ok),
        },
        "decision": decision,
        "notes": [
            "R5-A banks only layout-bridge evidence.",
            "fold_pair_group32 tests the cheap zero-copy idea that collapses two Lynn per-16 groups into one group32 scale.",
            "padded_per16_group32 tests preserving per-16 grouping by padding each group to group32; it doubles packed K bytes.",
            "If current_lynn_e4m3_scales_zero_copy_supported is false, R5-B must use explicit repack or a custom scale path before claiming native FP4-MMA speed.",
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passes"]["all"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
