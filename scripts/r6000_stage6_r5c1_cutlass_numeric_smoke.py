#!/usr/bin/env python3
"""Stage 6 R5-C1 CUTLASS native NVF4 + UE4M3 numeric smoke.

This gate runs CUTLASS example 79d on the R6000 lane:

  examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu

The example is intentionally used as a minimal numeric smoke before writing a
Lynn grouped-MoE kernel. It exercises native NVFP4 E2M1 operands plus UE4M3
scales and validates against CUTLASS host reference. It does not bank a Lynn
kernel, a speed claim, or a runtime default.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
TARGET = "79d_blackwell_geforce_nvfp4_grouped_gemm"
EXAMPLE_REL = "examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu"
HEADER_REL = "include/cutlass/subbyte_reference.h"
ATOMIC_OLD = "__nv_atomic_load_n(ptr_, __NV_ATOMIC_RELAXED)"
ATOMIC_NEW = "__nv_atomic_load_n(ptr_, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE)"


def _tail(text: str, max_lines: int = 80) -> str:
    lines = (text or "").splitlines()
    return "\n".join(lines[-max_lines:])


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout_s: int = 600,
) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_s,
        )
        return {
            "cmd": cmd,
            "cwd": str(cwd),
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout_tail": _tail(proc.stdout),
            "stderr_tail": _tail(proc.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "cwd": str(cwd),
            "ok": False,
            "returncode": None,
            "timeout": True,
            "stdout_tail": _tail(exc.stdout or ""),
            "stderr_tail": _tail(exc.stderr or ""),
        }


def _env(cuda_home: str, python_bin: str) -> dict[str, str]:
    env = os.environ.copy()
    cuda_bin = str(Path(cuda_home) / "bin")
    env["PATH"] = f"{Path(python_bin).parent}:{cuda_bin}:{env.get('PATH', '')}"
    env.setdefault("CUDA_HOME", cuda_home)
    env.setdefault("CUDACXX", str(Path(cuda_home) / "bin" / "nvcc"))
    return env


def _git(cutlass_dir: Path) -> dict[str, Any]:
    if not cutlass_dir.exists():
        return {}
    return {
        "head": _run(["git", "rev-parse", "HEAD"], cwd=cutlass_dir, timeout_s=30),
        "branch": _run(["git", "branch", "--show-current"], cwd=cutlass_dir, timeout_s=30),
        "status": _run(["git", "status", "--short"], cwd=cutlass_dir, timeout_s=30),
    }


@contextmanager
def _temporary_atomic_scope_patch(cutlass_dir: Path, enabled: bool) -> Iterator[dict[str, Any]]:
    """Patch a CUDA 12.8 reference-header API mismatch, then restore it."""
    header = cutlass_dir / HEADER_REL
    info = {
        "enabled": enabled,
        "header": str(header),
        "applied": False,
        "already_patched": False,
        "restored": False,
        "error": None,
    }
    original = ""
    try:
        if enabled and header.exists():
            original = header.read_text(encoding="utf-8")
            if ATOMIC_NEW in original:
                info["already_patched"] = True
            elif ATOMIC_OLD in original:
                header.write_text(original.replace(ATOMIC_OLD, ATOMIC_NEW), encoding="utf-8")
                info["applied"] = True
            else:
                info["error"] = "expected atomic load token not found"
        yield info
    finally:
        if info["applied"]:
            header.write_text(original, encoding="utf-8")
            info["restored"] = True


def _binary_path(build_dir: Path) -> Path:
    return build_dir / "examples" / "79_blackwell_geforce_gemm" / TARGET


def _configure_and_build(
    *,
    cutlass_dir: Path,
    build_dir: Path,
    cuda_home: str,
    python_bin: str,
    timeout_s: int,
    clean_build: bool,
    atomic_scope_patch: bool,
) -> dict[str, Any]:
    if clean_build and build_dir.exists():
        shutil.rmtree(build_dir)
    env = _env(cuda_home, python_bin)
    util_include = cutlass_dir / "tools" / "util" / "include"
    cmake = [
        "cmake",
        "-S",
        str(cutlass_dir),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CUDA_COMPILER={Path(cuda_home) / 'bin' / 'nvcc'}",
        f"-DPython3_EXECUTABLE={python_bin}",
        "-DCUTLASS_NVCC_ARCHS=120a",
        "-DCUTLASS_ENABLE_EXAMPLES=ON",
        "-DCUTLASS_ENABLE_TESTS=OFF",
        "-DCUTLASS_ENABLE_TOOLS=ON",
        f"-DCMAKE_CUDA_FLAGS=-I{util_include}",
    ]
    build = ["cmake", "--build", str(build_dir), "--target", TARGET, "-j", "8"]
    with _temporary_atomic_scope_patch(cutlass_dir, atomic_scope_patch) as patch_info:
        configure_result = _run(cmake, cwd=cutlass_dir, env=env, timeout_s=timeout_s)
        build_result = (
            _run(build, cwd=cutlass_dir, env=env, timeout_s=timeout_s)
            if configure_result["ok"]
            else {"ok": False, "skipped": True}
        )
    return {
        "configure": configure_result,
        "build": build_result,
        "atomic_scope_patch": patch_info,
    }


def _parse_run(stdout: str, stderr: str) -> dict[str, Any]:
    text = f"{stdout}\n{stderr}"
    runtimes = [float(x) for x in re.findall(r"Avg runtime\s*:\s*([0-9.]+)\s*ms", text)]
    tflops = [float(x) for x in re.findall(r"TFLOPS\s*:\s*([0-9.]+)", text)]
    disposition_passed_count = text.count("Disposition: Passed")
    return {
        "cooperative_seen": "Running kernel with Cooperative kernel schedule" in text,
        "pingpong_seen": "Running kernel with Pingpong kernel schedule" in text,
        "host_reference_seen": "Host-side verification is now running" in text,
        "disposition_passed_count": disposition_passed_count,
        "no_noop_device_gate": "requires a GPU" not in text and "not supported" not in text,
        "avg_runtime_ms": runtimes,
        "tflops": tflops,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    cutlass_dir = Path(args.cutlass_dir).expanduser().resolve()
    build_dir = Path(args.build_dir).expanduser().resolve()
    binary = _binary_path(build_dir)
    example = cutlass_dir / EXAMPLE_REL
    build_result: dict[str, Any] = {"skipped": True}
    if args.build:
        build_result = _configure_and_build(
            cutlass_dir=cutlass_dir,
            build_dir=build_dir,
            cuda_home=args.cuda_home,
            python_bin=args.python_bin,
            timeout_s=args.timeout_s,
            clean_build=args.clean_build,
            atomic_scope_patch=not args.no_atomic_scope_patch,
        )

    run_cmd = [
        str(binary),
        f"--m={args.m}",
        f"--n={args.n}",
        f"--k={args.k}",
        f"--groups={args.groups}",
        f"--iterations={args.iterations}",
    ]
    if binary.exists():
        run_result = _run(run_cmd, cwd=build_dir, env=_env(args.cuda_home, args.python_bin), timeout_s=args.timeout_s)
    else:
        run_result = {"ok": False, "returncode": None, "stdout_tail": "", "stderr_tail": "binary missing", "cmd": run_cmd}
    parsed = _parse_run(run_result.get("stdout_tail", ""), run_result.get("stderr_tail", ""))

    build_ok = bool((build_result.get("build") or {}).get("ok")) if args.build else False
    passes = {
        "cutlass_dir_exists": cutlass_dir.exists(),
        "example_79d_exists": example.exists(),
        "binary_exists": binary.exists(),
        "build_invoked": bool(args.build),
        "build_succeeded": build_ok,
        "run_succeeded": bool(run_result.get("ok")),
        "no_noop_device_gate": parsed["no_noop_device_gate"],
        "cooperative_passed": parsed["cooperative_seen"],
        "pingpong_passed": parsed["pingpong_seen"],
        "host_reference_seen": parsed["host_reference_seen"],
        "dispositions_passed_count_ge_2": parsed["disposition_passed_count"] >= 2,
        "banked_numeric_smoke": False,
        "banked_grouped_moe_fp4_mma_poc": False,
        "banked_kernel_speed": False,
        "banked_default_promotion": False,
    }
    passes["banked_numeric_smoke"] = all(
        bool(passes[key])
        for key in [
            "cutlass_dir_exists",
            "example_79d_exists",
            "binary_exists",
            "build_invoked",
            "build_succeeded",
            "run_succeeded",
            "no_noop_device_gate",
            "cooperative_passed",
            "pingpong_passed",
            "host_reference_seen",
            "dispositions_passed_count_ge_2",
        ]
    )
    passes["all"] = bool(passes["banked_numeric_smoke"])
    decision = (
        "PASS_R5C1_CUTLASS_NVF4_UE4M3_NUMERIC_SMOKE"
        if passes["all"]
        else "FAIL_R5C1_CUTLASS_NVF4_UE4M3_NUMERIC_SMOKE"
    )
    return {
        "schema": "lynn-stage6-r5c1-cutlass-numeric-smoke-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cutlass_dir": str(cutlass_dir),
        "build_dir": str(build_dir),
        "binary": str(binary),
        "example": str(example),
        "git": _git(cutlass_dir),
        "shape": {
            "m": args.m,
            "n": args.n,
            "k": args.k,
            "groups": args.groups,
            "iterations": args.iterations,
        },
        "build_result": build_result,
        "run_result": run_result,
        "run_parse": parsed,
        "passes": passes,
        "decision": decision,
        "promotion_boundary": {
            "grouped_moe_fp4_mma_poc": False,
            "kernel_speed": False,
            "default_runtime": False,
        },
        "caveats": [
            "CUTLASS example 79d is a minimal numeric smoke, not a Lynn grouped-MoE kernel.",
            "Avg runtime is recorded for traceability only; R5-C1 does not bank speed.",
            "CUDA 12.8 may require a temporary subbyte_reference atomic scope patch during build; the script restores it.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cutlass-dir", default="/root/autodl-tmp/src/cutlass")
    ap.add_argument("--build-dir", default="/root/autodl-tmp/src/cutlass/build-r5c1-sm120a-tools-on")
    ap.add_argument("--cuda-home", default="/usr/local/cuda-12.8")
    ap.add_argument("--python-bin", default="/root/miniconda3/bin/python")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout-s", type=int, default=900)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--clean-build", action="store_true")
    ap.add_argument("--no-atomic-scope-patch", action="store_true")
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--iterations", type=int, default=1)
    args = ap.parse_args()
    data = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": data["decision"], "passes": data["passes"]}, indent=2))
    return 0 if data["passes"]["all"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
