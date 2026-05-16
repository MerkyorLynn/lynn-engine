#!/usr/bin/env python3
"""P81: verify Lynn native CUDA architecture policy.

P80 proved `sm_120a` can execute E2M1 FP4 MMA. P81 wires the feature target into
the shared native CUDA loader and verifies the policy is observable from a
small probe before larger kernels depend on it.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.native_cuda import (  # noqa: E402
    native_cuda_arch_flags,
    native_cuda_extra_cuda_cflags,
)


def _case(env: dict[str, str | None]) -> dict[str, object]:
    old = {key: os.environ.get(key) for key in env}
    try:
        for key, value in env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return {
            "env": env,
            "arch_flags": native_cuda_arch_flags(),
            "cuda_cflags": native_cuda_extra_cuda_cflags(),
        }
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cases = [
        _case({"LYNN_NATIVE_CUDA_ARCH": None, "LYNN_NATIVE_CUDA_ARCH_AUTO": None}),
        _case({"LYNN_NATIVE_CUDA_ARCH": "sm_120a", "LYNN_NATIVE_CUDA_ARCH_AUTO": None}),
        _case({"LYNN_NATIVE_CUDA_ARCH": None, "LYNN_NATIVE_CUDA_ARCH_AUTO": "1"}),
    ]
    expected_auto = ["-arch=sm_120a"] if torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (12, 0) else []
    ok = (
        cases[0]["arch_flags"] == []
        and cases[1]["arch_flags"] == ["-arch=sm_120a"]
        and cases[2]["arch_flags"] == expected_auto
    )
    result = {
        "schema_version": "lynn-engine-p81-native-cuda-arch-policy-probe-v1",
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "cases": cases,
        "ok": ok,
        "decision": (
            "Native CUDA arch policy is ready for opt-in sm_120a FP4 MMA kernels."
            if ok
            else "Native CUDA arch policy mismatch; inspect env handling before using sm_120a kernels."
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
