#!/usr/bin/env python3
"""P103: compile-only probe for SM120a FP8 activation x E2M1 weight MMA.

P102 rejected the BF16/FP16 activation x E2M1 shortcut. The next practical route
is W4A8: keep weights in E2M1/NVFP4 while using FP8 activation. This probe
checks whether the SM120a CuTe atom exposes FP8 x E2M1 combinations for raw and
blockscaled MMA.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.native_cuda import discover_native_include_paths  # noqa: E402


TYPE_ALIASES = {
    "e2m1": "cute::float_e2m1_t",
    "e4m3": "cute::float_e4m3_t",
    "e5m2": "cute::float_e5m2_t",
    "f32": "float",
    "f16": "cute::half_t",
    "ue8m0": "cute::float_ue8m0_t",
}


VARIANTS = [
    {
        "name": "raw_e2m1_e2m1_f32_control",
        "family": "raw",
        "a": "e2m1",
        "b": "e2m1",
        "c": "f32",
        "control": True,
    },
    {
        "name": "block_e2m1_e2m1_f32_ue8m0_control",
        "family": "blockscaled",
        "a": "e2m1",
        "b": "e2m1",
        "c": "f32",
        "sf": "ue8m0",
        "vs": 32,
        "control": True,
    },
    {
        "name": "raw_e4m3_e2m1_f32",
        "family": "raw",
        "a": "e4m3",
        "b": "e2m1",
        "c": "f32",
        "w4a8_candidate": True,
    },
    {
        "name": "raw_e5m2_e2m1_f32",
        "family": "raw",
        "a": "e5m2",
        "b": "e2m1",
        "c": "f32",
        "w4a8_candidate": True,
    },
    {
        "name": "raw_e2m1_e4m3_f32_reverse",
        "family": "raw",
        "a": "e2m1",
        "b": "e4m3",
        "c": "f32",
        "w4a8_candidate": True,
    },
    {
        "name": "raw_e2m1_e5m2_f32_reverse",
        "family": "raw",
        "a": "e2m1",
        "b": "e5m2",
        "c": "f32",
        "w4a8_candidate": True,
    },
    {
        "name": "block_e4m3_e2m1_f32_ue8m0",
        "family": "blockscaled",
        "a": "e4m3",
        "b": "e2m1",
        "c": "f32",
        "sf": "ue8m0",
        "vs": 32,
        "w4a8_candidate": True,
    },
    {
        "name": "block_e5m2_e2m1_f32_ue8m0",
        "family": "blockscaled",
        "a": "e5m2",
        "b": "e2m1",
        "c": "f32",
        "sf": "ue8m0",
        "vs": 32,
        "w4a8_candidate": True,
    },
    {
        "name": "block_e2m1_e4m3_f32_ue8m0_reverse",
        "family": "blockscaled",
        "a": "e2m1",
        "b": "e4m3",
        "c": "f32",
        "sf": "ue8m0",
        "vs": 32,
        "w4a8_candidate": True,
    },
    {
        "name": "block_e2m1_e5m2_f32_ue8m0_reverse",
        "family": "blockscaled",
        "a": "e2m1",
        "b": "e5m2",
        "c": "f32",
        "sf": "ue8m0",
        "vs": 32,
        "w4a8_candidate": True,
    },
]


def _prepare_path() -> None:
    python_bin = Path(sys.executable).resolve().parent
    os.environ["PATH"] = f"{python_bin}:{os.environ.get('PATH', '')}"
    for extra_bin in (
        Path.home() / "miniconda3" / "bin",
        Path("/root/miniconda3/bin"),
        Path("/usr/local/cuda/bin"),
    ):
        if extra_bin.exists():
            os.environ["PATH"] = f"{extra_bin}:{os.environ.get('PATH', '')}"


def _tail(text: str, max_lines: int = 18) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-max_lines:])


def _run(cmd: list[str], timeout_s: int = 60) -> dict[str, object]:
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "seconds": time.time() - t0,
            "stdout_tail": _tail(proc.stdout),
            "stderr_tail": _tail(proc.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": None,
            "seconds": time.time() - t0,
            "stdout_tail": _tail(exc.stdout or ""),
            "stderr_tail": _tail(exc.stderr or ""),
            "timeout": True,
        }


def _cuda_source(variant: dict[str, object]) -> str:
    a_type = TYPE_ALIASES[str(variant["a"])]
    b_type = TYPE_ALIASES[str(variant["b"])]
    c_type = TYPE_ALIASES[str(variant["c"])]
    if variant["family"] == "raw":
        atom = f"cute::SM120_16x8x32_TN<{a_type}, {b_type}, {c_type}>"
    else:
        sf_type = TYPE_ALIASES[str(variant["sf"])]
        vs = int(variant["vs"])
        atom = (
            "cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS"
            f"<{a_type}, {b_type}, {c_type}, {sf_type}, {vs}>"
        )
    return f"""
#include <cute/arch/mma_sm120.hpp>
#include <cute/numeric/numeric_types.hpp>

using ProbeAtom = {atom};

__global__ void p103_compile_probe(float* out) {{
  if (threadIdx.x == 0 && blockIdx.x == 0) {{
    out[0] = static_cast<float>(sizeof(ProbeAtom));
  }}
}}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/tmp/lynn_engine_native_build/p103_fp8_activation_fp4_weight")
    ap.add_argument("--arch", default="sm_120a")
    ap.add_argument("--timeout-s", type=int, default=90)
    args = ap.parse_args()

    _prepare_path()
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc not found")

    build_root = Path(args.build_dir)
    if build_root.exists():
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True, exist_ok=True)

    include_paths = discover_native_include_paths()
    include_args = [arg for path in include_paths for arg in ("-I", path)]
    results: list[dict[str, object]] = []
    for variant in VARIANTS:
        source_path = build_root / f"{variant['name']}.cu"
        obj_path = build_root / f"{variant['name']}.o"
        source_path.write_text(_cuda_source(variant), encoding="utf-8")
        cmd = [
            nvcc,
            "-std=c++17",
            "-O3",
            "--use_fast_math",
            *include_args,
            "-arch",
            args.arch,
            "-c",
            str(source_path),
            "-o",
            str(obj_path),
        ]
        result = _run(cmd, timeout_s=args.timeout_s)
        result.update(
            {
                "name": variant["name"],
                "family": variant["family"],
                "a": variant["a"],
                "b": variant["b"],
                "c": variant["c"],
                "sf": variant.get("sf"),
                "vs": variant.get("vs"),
                "control": bool(variant.get("control", False)),
                "w4a8_candidate": bool(variant.get("w4a8_candidate", False)),
                "cmd": cmd,
            }
        )
        results.append(result)

    controls = [item for item in results if item["control"]]
    candidates = [item for item in results if item["w4a8_candidate"]]
    controls_ok = all(bool(item["ok"]) for item in controls)
    w4a8_supported = any(bool(item["ok"]) for item in candidates)
    supported = [item["name"] for item in candidates if item["ok"]]
    decision = (
        "FP8 activation x E2M1 weight SM120a MMA is exposed; W4A8 is a viable hardware route."
        if w4a8_supported
        else "No FP8 activation x E2M1 weight atom compiled; W4A8 needs a software bridge."
    )
    if not controls_ok:
        decision = "Controls failed; cannot draw a W4A8 hardware conclusion from this run."

    payload = {
        "schema_version": "lynn-engine-p103-sm120a-fp8-activation-fp4-weight-mma-probe-v1",
        "nvcc": nvcc,
        "arch": args.arch,
        "include_paths": include_paths,
        "variants": results,
        "controls_ok": controls_ok,
        "fp8_activation_fp4_weight_supported": w4a8_supported if controls_ok else None,
        "supported_w4a8_variants": supported,
        "decision": decision,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if controls_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
