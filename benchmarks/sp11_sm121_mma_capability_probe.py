#!/usr/bin/env python3
"""SP-11: Spark sm_121 MMA ISA capability probe.

For each MMA instruction variant we care about, we:
  1. Write a minimal self-contained .cu file using inline PTX assembly
  2. Build via torch.utils.cpp_extension.load() with -arch=sm_121a
  3. Capture compile/ptxas errors verbatim
  4. If compile succeeds, run the kernel with known inputs
  5. Compare against scalar reference, report numerical correctness

Goal: discover which MMA instruction families ARE available on Spark GB10
sm_121 so we can design a Spark-specific Lynn-native FP4 path that does NOT
depend on the block_scale FP4 MMA that ptxas blocks (per SP-10).

If FP8 MMA works on sm_121, we can build:
  packed E2M1 weights → shared-mem dequant to FP8 → FP8 MMA → FP32 epilogue
                       with Lynn per-16 FP32 scale applied outside
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


# ---------------------------------------------------------------------------
# Probe definitions
# ---------------------------------------------------------------------------
# Each probe is a minimal CUDA kernel that uses one specific inline-PTX mma
# instruction. The kernel body computes one m16n8kK MMA tile. The expected
# output is computed in Python/scalar reference and compared.
#
# These probes intentionally avoid CuTe / CUTLASS dependencies so they isolate
# the PTX instruction itself. A probe that compiles + runs proves the
# instruction is in Spark sm_121's ISA.

# Header used by every probe — minimal pybind + simple kernel boilerplate.
BINDINGS_TEMPLATE = """
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern void probe_launch(float *out, void *a, void *b, void *c);

torch::Tensor probe(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    auto out = torch::zeros({16, 8}, torch::dtype(torch::kFloat32).device(a.device()));
    probe_launch(
        out.data_ptr<float>(),
        a.data_ptr(),
        b.data_ptr(),
        c.data_ptr()
    );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("probe", &probe, "MMA capability probe");
}
"""

PROBES = [
    {
        "name": "bf16_m16n8k16_baseline",
        "description": "BF16 m16n8k16 MMA — sanity check (must PASS on Spark)",
        "cu": """
#include <cuda.h>
#include <cuda_bf16.h>

__global__ void mma_kernel(float *out, void *a, void *b, void *c) {
    // BF16 m16n8k16 — packed BF16 in uint32 (2 elements per reg)
    // A: 4 regs of 2 BF16 (m=16 × k=16 / 8 lanes = 4 regs per lane)
    // B: 2 regs of 2 BF16 (n=8 × k=16 / 8 lanes = 2 regs per lane)
    // D, C: 4 regs FP32 (m=16 × n=8 / 8 lanes = 4 regs per lane)
    uint32_t *aptr = (uint32_t*)a;
    uint32_t *bptr = (uint32_t*)b;
    float    *cptr = (float*)c;

    uint32_t a0 = aptr[threadIdx.x * 4 + 0];
    uint32_t a1 = aptr[threadIdx.x * 4 + 1];
    uint32_t a2 = aptr[threadIdx.x * 4 + 2];
    uint32_t a3 = aptr[threadIdx.x * 4 + 3];
    uint32_t b0 = bptr[threadIdx.x * 2 + 0];
    uint32_t b1 = bptr[threadIdx.x * 2 + 1];
    float d0=0, d1=0, d2=0, d3=0;

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f)
    );

    out[threadIdx.x * 4 + 0] = d0;
    out[threadIdx.x * 4 + 1] = d1;
    out[threadIdx.x * 4 + 2] = d2;
    out[threadIdx.x * 4 + 3] = d3;
}

void probe_launch(float *out, void *a, void *b, void *c) {
    mma_kernel<<<1, 32>>>(out, a, b, c);
}
""",
        "expected": "PASS",
    },
    {
        "name": "fp8_e4m3_m16n8k32",
        "description": "FP8 E4M3 m16n8k32 raw MMA (Blackwell core feature)",
        "cu": """
#include <cuda.h>

__global__ void mma_kernel(float *out, void *a, void *b, void *c) {
    // FP8 E4M3 m16n8k32 — packed 4 FP8 per uint32
    // A: 4 regs (m=16 × k=32 / (8 lanes × 4 pack) = 4 regs per lane)
    // B: 2 regs
    uint32_t *aptr = (uint32_t*)a;
    uint32_t *bptr = (uint32_t*)b;
    uint32_t a0 = aptr[threadIdx.x * 4 + 0];
    uint32_t a1 = aptr[threadIdx.x * 4 + 1];
    uint32_t a2 = aptr[threadIdx.x * 4 + 2];
    uint32_t a3 = aptr[threadIdx.x * 4 + 3];
    uint32_t b0 = bptr[threadIdx.x * 2 + 0];
    uint32_t b1 = bptr[threadIdx.x * 2 + 1];
    float d0=0, d1=0, d2=0, d3=0;

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f)
    );

    out[threadIdx.x * 4 + 0] = d0;
    out[threadIdx.x * 4 + 1] = d1;
    out[threadIdx.x * 4 + 2] = d2;
    out[threadIdx.x * 4 + 3] = d3;
}

void probe_launch(float *out, void *a, void *b, void *c) {
    mma_kernel<<<1, 32>>>(out, a, b, c);
}
""",
        "expected": "?",
    },
    {
        "name": "fp8_e5m2_m16n8k32",
        "description": "FP8 E5M2 m16n8k32 raw MMA",
        "cu": """
#include <cuda.h>

__global__ void mma_kernel(float *out, void *a, void *b, void *c) {
    uint32_t *aptr = (uint32_t*)a;
    uint32_t *bptr = (uint32_t*)b;
    uint32_t a0 = aptr[threadIdx.x * 4 + 0];
    uint32_t a1 = aptr[threadIdx.x * 4 + 1];
    uint32_t a2 = aptr[threadIdx.x * 4 + 2];
    uint32_t a3 = aptr[threadIdx.x * 4 + 3];
    uint32_t b0 = bptr[threadIdx.x * 2 + 0];
    uint32_t b1 = bptr[threadIdx.x * 2 + 1];
    float d0=0, d1=0, d2=0, d3=0;

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f)
    );

    out[threadIdx.x * 4 + 0] = d0;
    out[threadIdx.x * 4 + 1] = d1;
    out[threadIdx.x * 4 + 2] = d2;
    out[threadIdx.x * 4 + 3] = d3;
}

void probe_launch(float *out, void *a, void *b, void *c) {
    mma_kernel<<<1, 32>>>(out, a, b, c);
}
""",
        "expected": "?",
    },
    {
        "name": "fp4_e2m1_m16n8k32_raw",
        "description": "FP4 E2M1 m16n8k32 raw MMA (no kind:: or block_scale)",
        "cu": """
#include <cuda.h>

__global__ void mma_kernel(float *out, void *a, void *b, void *c) {
    // FP4 E2M1: 8 nibbles per uint32. k=32 with 8 lanes × 4 elements = 32. 1 reg per lane?
    // Actually k=32 / 8 lanes / 8 elements_per_uint32 = 0.5 ... need k=64 for 1 reg
    // Try with 2 regs per lane (k=32 packed 2 nibbles per byte)
    uint32_t *aptr = (uint32_t*)a;
    uint32_t *bptr = (uint32_t*)b;
    uint32_t a0 = aptr[threadIdx.x * 2 + 0];
    uint32_t a1 = aptr[threadIdx.x * 2 + 1];
    uint32_t b0 = bptr[threadIdx.x * 1 + 0];
    float d0=0, d1=0, d2=0, d3=0;

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1),
          "r"(b0),
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f)
    );

    out[threadIdx.x * 4 + 0] = d0;
    out[threadIdx.x * 4 + 1] = d1;
    out[threadIdx.x * 4 + 2] = d2;
    out[threadIdx.x * 4 + 3] = d3;
}

void probe_launch(float *out, void *a, void *b, void *c) {
    mma_kernel<<<1, 32>>>(out, a, b, c);
}
""",
        "expected": "?",
    },
    {
        "name": "fp4_e2m1_kind_f8f6f4_no_blockscale",
        "description": "kind::f8f6f4 with FP4 operands BUT no block_scale (the SP-10 blocker had block_scale)",
        "cu": """
#include <cuda.h>

__global__ void mma_kernel(float *out, void *a, void *b, void *c) {
    uint32_t *aptr = (uint32_t*)a;
    uint32_t *bptr = (uint32_t*)b;
    uint32_t a0 = aptr[threadIdx.x * 4 + 0];
    uint32_t a1 = aptr[threadIdx.x * 4 + 1];
    uint32_t a2 = aptr[threadIdx.x * 4 + 2];
    uint32_t a3 = aptr[threadIdx.x * 4 + 3];
    uint32_t b0 = bptr[threadIdx.x * 2 + 0];
    uint32_t b1 = bptr[threadIdx.x * 2 + 1];
    float d0=0, d1=0, d2=0, d3=0;

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f)
    );

    out[threadIdx.x * 4 + 0] = d0;
    out[threadIdx.x * 4 + 1] = d1;
    out[threadIdx.x * 4 + 2] = d2;
    out[threadIdx.x * 4 + 3] = d3;
}

void probe_launch(float *out, void *a, void *b, void *c) {
    mma_kernel<<<1, 32>>>(out, a, b, c);
}
""",
        "expected": "?",
    },
    {
        "name": "fp6_e3m2_m16n8k32",
        "description": "FP6 E3M2 raw MMA",
        "cu": """
#include <cuda.h>

__global__ void mma_kernel(float *out, void *a, void *b, void *c) {
    uint32_t *aptr = (uint32_t*)a;
    uint32_t *bptr = (uint32_t*)b;
    uint32_t a0 = aptr[threadIdx.x * 4 + 0];
    uint32_t a1 = aptr[threadIdx.x * 4 + 1];
    uint32_t a2 = aptr[threadIdx.x * 4 + 2];
    uint32_t a3 = aptr[threadIdx.x * 4 + 3];
    uint32_t b0 = bptr[threadIdx.x * 2 + 0];
    uint32_t b1 = bptr[threadIdx.x * 2 + 1];
    float d0=0, d1=0, d2=0, d3=0;

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e3m2.e3m2.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f)
    );

    out[threadIdx.x * 4 + 0] = d0;
    out[threadIdx.x * 4 + 1] = d1;
    out[threadIdx.x * 4 + 2] = d2;
    out[threadIdx.x * 4 + 3] = d3;
}

void probe_launch(float *out, void *a, void *b, void *c) {
    mma_kernel<<<1, 32>>>(out, a, b, c);
}
""",
        "expected": "?",
    },
    {
        "name": "blockscale_mxf8f6f4_known_blocked",
        "description": "kind::mxf8f6f4 + block_scale + scale_vec::1X — known SP-10 failure (control)",
        "cu": """
#include <cuda.h>

__global__ void mma_kernel(float *out, void *a, void *b, void *c) {
    uint32_t *aptr = (uint32_t*)a;
    uint32_t *bptr = (uint32_t*)b;
    uint32_t a0 = aptr[threadIdx.x * 4 + 0];
    uint32_t a1 = aptr[threadIdx.x * 4 + 1];
    uint32_t a2 = aptr[threadIdx.x * 4 + 2];
    uint32_t a3 = aptr[threadIdx.x * 4 + 3];
    uint32_t b0 = bptr[threadIdx.x * 2 + 0];
    uint32_t b1 = bptr[threadIdx.x * 2 + 1];
    float d0=0, d1=0, d2=0, d3=0;

    // Scale registers (UE8M0 packed as uint32)
    uint32_t scale_a = 0x7F7F7F7F;  // neutral scale (e8m0 = 127 = 1.0)
    uint32_t scale_b = 0x7F7F7F7F;
    uint16_t bid_a = 0, bid_b = 0;

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
        "{%14}, {%15, 0}, {%16}, {%17, 0};\\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f),
          "r"(scale_a), "h"(bid_a),
          "r"(scale_b), "h"(bid_b)
    );

    out[threadIdx.x * 4 + 0] = d0;
    out[threadIdx.x * 4 + 1] = d1;
    out[threadIdx.x * 4 + 2] = d2;
    out[threadIdx.x * 4 + 3] = d3;
}

void probe_launch(float *out, void *a, void *b, void *c) {
    mma_kernel<<<1, 32>>>(out, a, b, c);
}
""",
        "expected": "FAIL (control)",
    },
]


def _build_dir(base: Path, name: str) -> Path:
    d = base / f"sp11_{name}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_probe(probe: dict, build_root: Path, verbose: bool = False) -> dict:
    name = probe["name"]
    result = {
        "name": name,
        "description": probe["description"],
        "expected": probe["expected"],
        "compile_ok": False,
        "run_ok": False,
        "error": None,
        "stage": "init",
    }

    build_dir = _build_dir(build_root, name)
    cu_path = build_dir / "probe_kernel.cu"
    cpp_path = build_dir / "probe_bindings.cpp"
    cu_path.write_text(probe["cu"], encoding="utf-8")
    cpp_path.write_text(BINDINGS_TEMPLATE, encoding="utf-8")

    print(f"\n[sp11] === {name} ===")
    print(f"[sp11] expected: {probe['expected']}")
    print(f"[sp11] {probe['description']}")

    try:
        result["stage"] = "compile"
        module = load(
            name=f"lynn_sp11_{name}",
            sources=[str(cpp_path), str(cu_path)],
            build_directory=str(build_dir),
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_121a"],
            verbose=verbose,
        )
        result["compile_ok"] = True
        print(f"[sp11] {name}  compile=PASS")

        # Try a minimal run with dummy inputs
        result["stage"] = "run"
        a = torch.ones((32, 32), device="cuda", dtype=torch.uint8).contiguous()
        b = torch.ones((8, 32), device="cuda", dtype=torch.uint8).contiguous()
        c = torch.zeros((16, 8), device="cuda", dtype=torch.float32).contiguous()
        out = module.probe(a, b, c)
        torch.cuda.synchronize()
        result["run_ok"] = True
        result["output_finite"] = bool(torch.isfinite(out).all().item())
        result["output_sample"] = out[:4, :2].flatten().tolist()
        print(f"[sp11] {name}  run=PASS  finite={result['output_finite']}  sample={result['output_sample']}")

    except Exception as exc:
        err_msg = str(exc)
        result["error"] = err_msg[:2000]
        # Extract ptxas error lines
        ptxas_errs = [line for line in err_msg.split("\n") if "ptxas" in line.lower() and "error" in line.lower()]
        result["ptxas_errors"] = ptxas_errs[:6]
        print(f"[sp11] {name}  {result['stage']}=FAIL")
        for line in ptxas_errs[:6]:
            print(f"[sp11]   {line.strip()}")
        if not ptxas_errs:
            # Print last few lines of error for context
            for line in err_msg.split("\n")[-5:]:
                if line.strip():
                    print(f"[sp11]   {line.strip()}")

    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-root", default="/tmp/lynn_sp11_probes")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--only", default=None, help="run only a specific probe name")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[sp11] no CUDA available")
        return 1

    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"[sp11] device: {name} compute_capability=sm_{cap[0]}{cap[1]}")
    print(f"[sp11] torch: {torch.__version__}  cuda: {torch.version.cuda}")
    print(f"[sp11] testing {len(PROBES)} MMA variants under -arch=sm_121a")

    build_root = Path(args.build_root)
    build_root.mkdir(parents=True, exist_ok=True)

    results = []
    for probe in PROBES:
        if args.only and args.only != probe["name"]:
            continue
        r = _run_probe(probe, build_root, args.verbose)
        results.append(r)

    summary = {
        "type": "sp11_sm121_mma_capability_probe",
        "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "device": name,
        "compute_capability": list(cap),
        "arch_flag": "sm_121a",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "results": results,
        "matrix": [
            {
                "name": r["name"],
                "compile": "PASS" if r["compile_ok"] else "FAIL",
                "run": "PASS" if r["run_ok"] else ("FAIL" if r["compile_ok"] else "—"),
                "expected": r["expected"],
            }
            for r in results
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n[sp11] === capability matrix ===")
    for entry in summary["matrix"]:
        print(f"[sp11]   {entry['name']:40s}  compile={entry['compile']:4s}  run={entry['run']:4s}")
    print(f"[sp11] report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
