#!/usr/bin/env python3
"""P84: sparse register mapping for CuTe SM120 E2M1 MMA.

P83 showed naive all-ones word filling does not produce the expected K=32 dot
contract. P84 sweeps sparse register/nibble patterns to learn which bits affect
which accumulators.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.native_cuda import (  # noqa: E402
    discover_native_include_paths,
    native_cuda_extra_cuda_cflags,
)


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor p84_fp4_mma_mapping();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp4_mma_mapping", &p84_fp4_mma_mapping, "P84 E2M1 MMA sparse mapping");
}
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cute/arch/mma_sm120.hpp>
#include <cute/numeric/numeric_types.hpp>

namespace {

constexpr int kBaselineCases = 2;
constexpr int kALaneCases = 32 * 4 * 8;
constexpr int kBLaneCases = 32 * 2 * 8;
constexpr int kCases = kBaselineCases + kALaneCases + kBLaneCases;

__device__ __forceinline__ void run_mma(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float* out) {
  float d0 = 0.0f;
  float d1 = 0.0f;
  float d2 = 0.0f;
  float d3 = 0.0f;
  cute::SM120_16x8x32_TN<
      cute::float_e2m1_t,
      cute::float_e2m1_t,
      float>::fma(
          d0, d1, d2, d3,
          a0, a1, a2, a3,
          b0, b1,
          0.0f, 0.0f, 0.0f, 0.0f);
  out[0] = d0;
  out[1] = d1;
  out[2] = d2;
  out[3] = d3;
}

__global__ void p84_mapping_kernel(float* out) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x;
  const uint32_t zero = 0x00000000u;
  const uint32_t one_word = 0x22222222u;
  const uint32_t one_nibble = 0x2u;

  uint32_t a[4] = {zero, zero, zero, zero};
  uint32_t b[2] = {zero, zero};

  if (row == 0) {
    // all zero
  } else if (row == 1) {
    a[0] = one_word;
    a[1] = one_word;
    a[2] = one_word;
    a[3] = one_word;
    b[0] = one_word;
    b[1] = one_word;
  } else if (row < kBaselineCases + kALaneCases) {
    const int off = row - kBaselineCases;
    const int selected_lane = off / (4 * 8);
    const int rem = off % (4 * 8);
    const int reg = rem / 8;
    const int nib = rem % 8;
    if (lane == selected_lane) {
      a[reg] = one_nibble << (4 * nib);
    }
    b[0] = one_word;
    b[1] = one_word;
  } else {
    const int off = row - kBaselineCases - kALaneCases;
    const int selected_lane = off / (2 * 8);
    const int rem = off % (2 * 8);
    const int reg = rem / 8;
    const int nib = rem % 8;
    a[0] = one_word;
    a[1] = one_word;
    a[2] = one_word;
    a[3] = one_word;
    if (lane == selected_lane) {
      b[reg] = one_nibble << (4 * nib);
    }
  }

  run_mma(a[0], a[1], a[2], a[3], b[0], b[1], out + (static_cast<int64_t>(row) * 32 + lane) * 4);
}

}  // namespace

torch::Tensor p84_fp4_mma_mapping() {
  auto out = torch::empty({kCases, 32, 4}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
  p84_mapping_kernel<<<kCases, 32>>>(out.data_ptr<float>());
  const cudaError_t launch_err = cudaGetLastError();
  TORCH_CHECK(launch_err == cudaSuccess, "p84_mapping_kernel launch failed: ", cudaGetErrorString(launch_err));
  return out;
}
"""


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


def _build_module(build_root: Path, verbose: bool):
    if build_root.exists():
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True, exist_ok=True)
    cpp_path = build_root / "p84_bindings.cpp"
    cu_path = build_root / "p84_kernel.cu"
    cpp_path.write_text(CPP_SOURCE, encoding="utf-8")
    cu_path.write_text(CUDA_SOURCE, encoding="utf-8")
    return load(
        name="lynn_native_p84_sm120a_fp4_mma_mapping",
        sources=[str(cpp_path), str(cu_path)],
        build_directory=str(build_root),
        extra_cflags=["-O3"],
        extra_cuda_cflags=native_cuda_extra_cuda_cflags(),
        extra_include_paths=discover_native_include_paths(),
        verbose=verbose,
    )


def _row_label(index: int) -> dict[str, object]:
    if index == 0:
        return {"kind": "baseline_zero"}
    if index == 1:
        return {"kind": "baseline_all_one"}
    offset = index - 2
    if offset < 32 * 4 * 8:
        return {
            "kind": "a_onehot",
            "selected_lane": offset // (4 * 8),
            "reg": (offset % (4 * 8)) // 8,
            "nibble": offset % 8,
        }
    offset -= 32 * 4 * 8
    return {
        "kind": "b_onehot",
        "selected_lane": offset // (2 * 8),
        "reg": (offset % (2 * 8)) // 8,
        "nibble": offset % 8,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/tmp/lynn_engine_native_build/p84_sm120a_fp4_mma_mapping")
    ap.add_argument("--include-records", action="store_true")
    ap.add_argument("--full-records", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("P84 requires CUDA")

    os.environ.setdefault("LYNN_NATIVE_CUDA_ARCH", "sm_120a")
    _prepare_path()

    t0 = time.time()
    module = _build_module(Path(args.build_dir), args.verbose)
    build_s = time.time() - t0
    out = module.fp4_mma_mapping()
    torch.cuda.synchronize()
    rows = out.detach().cpu()

    records = []
    compact_records = []
    for idx, values in enumerate(rows.tolist()):
        nonzero = []
        for lane, lane_values in enumerate(values):
            for acc, value in enumerate(lane_values):
                if value != 0.0:
                    nonzero.append({"lane": lane, "acc": acc, "value": float(value)})
        label = _row_label(idx)
        record = {
            **label,
            "row": idx,
            "sample_lanes": [[float(x) for x in lane_values] for lane_values in values[:4]],
            "nonzero_accumulators": nonzero,
            "nonzero_count": len(nonzero),
            "sum": float(rows[idx].sum().item()),
        }
        compact = {
            **label,
            "row": idx,
            "sample_lanes": record["sample_lanes"],
            "nonzero_count": record["nonzero_count"],
            "sum": record["sum"],
        }
        if nonzero:
            compact["first_nonzero"] = nonzero[0]
            compact["last_nonzero"] = nonzero[-1]
        records.append(record)
        compact_records.append(compact)

    baseline_all_one = [[float(x) for x in lane_values] for lane_values in rows[1, :4].tolist()]
    kinds: dict[str, dict[str, object]] = {}
    for record in compact_records:
        kind = str(record["kind"])
        bucket = kinds.setdefault(kind, {
            "count": 0,
            "nonzero_cases": 0,
            "nonzero_count_min": None,
            "nonzero_count_max": None,
            "sum_min": None,
            "sum_max": None,
        })
        bucket["count"] = int(bucket["count"]) + 1
        nonzero_count = int(record["nonzero_count"])
        if nonzero_count:
            bucket["nonzero_cases"] = int(bucket["nonzero_cases"]) + 1
        for key, value in (("nonzero_count", nonzero_count), ("sum", float(record["sum"]))):
            min_key = f"{key}_min"
            max_key = f"{key}_max"
            if bucket[min_key] is None or value < bucket[min_key]:
                bucket[min_key] = value
            if bucket[max_key] is None or value > bucket[max_key]:
                bucket[max_key] = value

    representative = []
    for wanted in (
        {"kind": "baseline_zero"},
        {"kind": "baseline_all_one"},
        {"kind": "a_onehot", "selected_lane": 0, "reg": 0, "nibble": 0},
        {"kind": "a_onehot", "selected_lane": 0, "reg": 0, "nibble": 1},
        {"kind": "b_onehot", "selected_lane": 31, "reg": 1, "nibble": 6},
        {"kind": "b_onehot", "selected_lane": 31, "reg": 1, "nibble": 7},
    ):
        for record in compact_records:
            if all(record.get(k) == v for k, v in wanted.items()):
                representative.append(record)
                break

    ok = any(record["nonzero_count"] for record in compact_records[2:])
    result = {
        "schema_version": "lynn-engine-p84-sm120a-fp4-mma-register-mapping-v1",
        "torch": torch.__version__,
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "include_paths": discover_native_include_paths(),
        "cuda_cflags": native_cuda_extra_cuda_cflags(),
        "build_seconds": build_s,
        "num_cases": len(compact_records),
        "baseline_all_one": baseline_all_one,
        "kind_summary": kinds,
        "representative_records": representative,
        "records": (records if args.full_records else compact_records) if args.include_records else None,
        "records_included": args.include_records,
        "records_are_compact": args.include_records and not args.full_records,
        "ok": ok,
        "decision": (
            "Sparse E2M1 register mapping captured; use records to derive packed-row layout."
            if ok
            else "Sparse E2M1 mapping produced no nonzero output; inspect MMA register setup."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
