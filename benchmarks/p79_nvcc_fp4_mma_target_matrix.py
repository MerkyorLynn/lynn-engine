#!/usr/bin/env python3
"""P79: compile-only target matrix for Blackwell E2M1 MMA.

P78 showed that the current torch extension path reaches ptxas but rejects
`.kind::f8f6f4` for the default `sm_120` target. P79 checks whether the same
minimal CuTe FP4 MMA source compiles for any nearby feature-suffixed target
supported by the installed nvcc/ptxas.
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


CUDA_SOURCE = r"""
#include <cute/arch/mma_sm120.hpp>
#include <cute/numeric/numeric_types.hpp>

__global__ void p79_mma_kernel(float* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    const uint32_t a0 = 0u;
    const uint32_t a1 = 0u;
    const uint32_t a2 = 0u;
    const uint32_t a3 = 0u;
    const uint32_t b0 = 0u;
    const uint32_t b1 = 0u;
    const float c0 = 1.0f;
    const float c1 = 2.0f;
    const float c2 = 3.0f;
    const float c3 = 4.0f;

    cute::SM120_16x8x32_TN<
        cute::float_e2m1_t,
        cute::float_e2m1_t,
        float>::fma(
            d0, d1, d2, d3,
            a0, a1, a2, a3,
            b0, b1,
            c0, c1, c2, c3);

    out[0] = d0;
    out[1] = d1;
    out[2] = d2;
    out[3] = d3;
  }
}
"""


DEFAULT_TARGETS = [
    "sm_120",
    "compute_120",
    "sm_120a",
    "compute_120a",
    "sm_120f",
    "compute_120f",
    "sm_121",
    "compute_121",
    "sm_121a",
    "compute_121a",
    "sm_121f",
    "compute_121f",
    "sm_100",
    "compute_100",
    "sm_100a",
    "compute_100a",
    "sm_100f",
    "compute_100f",
    "sm_101",
    "compute_101",
    "sm_101a",
    "compute_101a",
    "sm_101f",
    "compute_101f",
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


def _tail(text: str, max_lines: int = 16) -> str:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/tmp/lynn_engine_native_build/p79_nvcc_fp4_mma_target_matrix")
    ap.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    args = ap.parse_args()

    _prepare_path()
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc not found")

    build_root = Path(args.build_dir)
    build_root.mkdir(parents=True, exist_ok=True)
    source_path = build_root / "p79_fp4_mma_target_matrix.cu"
    source_path.write_text(CUDA_SOURCE, encoding="utf-8")

    include_paths = discover_native_include_paths()
    include_args = [arg for path in include_paths for arg in ("-I", path)]

    arch_list = _run([nvcc, "--list-gpu-arch"])
    code_list = _run([nvcc, "--list-gpu-code"])

    matrix: list[dict[str, object]] = []
    for target in args.targets:
        obj_path = build_root / f"p79_{target}.o"
        cmd = [
            nvcc,
            "-std=c++17",
            "-O3",
            "--use_fast_math",
            *include_args,
            "-arch",
            target,
            "-c",
            str(source_path),
            "-o",
            str(obj_path),
        ]
        result = _run(cmd)
        result.update({"target": target, "cmd": cmd})
        matrix.append(result)

    any_fp4_mma_target_ok = any(item["ok"] for item in matrix)
    result = {
        "schema_version": "lynn-engine-p79-nvcc-fp4-mma-target-matrix-v1",
        "nvcc": nvcc,
        "include_paths": include_paths,
        "nvcc_list_gpu_arch": arch_list,
        "nvcc_list_gpu_code": code_list,
        "targets": matrix,
        "any_fp4_mma_target_ok": any_fp4_mma_target_ok,
        "decision": (
            "At least one tested target compiles the CuTe E2M1 MMA source."
            if any_fp4_mma_target_ok
            else "No tested target compiles the CuTe E2M1 MMA source on this nvcc/ptxas stack."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if any_fp4_mma_target_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
