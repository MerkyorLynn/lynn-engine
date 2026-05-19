# P198 · Qwen3.5-9B Native FP4×FP8 Build Preflight

**Date:** 2026-05-19
**Author:** Qwen Code (auto-generated)
**Status:** 🟡 PENDING — awaiting R6000 execution

---

## Problem Statement

P190 true FP4×FP8 resident gate fails on R6000 with:

```
FP4 MMA not available: requires SM120a + CuTe headers
```

But P189 capability probe reports `device capability=(12,0)` and CuTe headers
are present. The failure occurs inside `dense_fp4xfp8_poc.cu` at compile time:
the `#if HAS_FP4_MMA` guard evaluates to 0 because the build flags weren't set.

P198 separates this into three distinct failure modes before blaming P190.

## Decision Chain

```
BLOCKED_COMPILE          → extension build failed (compile/link error)
BLOCKED_SYMBOL_MISSING   → extension loaded but FP4 MMA probe symbol missing
BLOCKED_PROBE_FAIL       → probe symbol exists but tiny-tensor smoke failed
READY_FOR_P190           → all checks pass
```

## Env Overlay

P198 sets these env vars before loading the extension:

| Env | Value | Why |
|-----|-------|-----|
| `LYNN_ENABLE_SM120A_FP4_MMA` | `1` | Passes `-DLYNN_ENABLE_SM120A_FP4_MMA=1` to nvcc → enables `#if HAS_FP4_MMA` block |
| `LYNN_NATIVE_CUDA_ARCH_AUTO` | `1` | Triggers `-arch=sm_120a` for SM120 devices |
| `LYNN_NATIVE_CUDA_BUILD_DIR` | `/tmp/.../p198_<stamp>` | Isolates build from production |

## What P198 Checks

1. **Environment:** torch version, CUDA version, device name, capability
2. **CuTe headers:** `cute/arch/mma_sm120.hpp` discoverable via `discover_native_include_paths()`
3. **Build:** `load_lynn_native_extension()` — catches compile/link errors
4. **Symbols:** `hasattr(ext, "dense_fp4xfp8_mma_scaled_probe")` — catches `#if HAS_FP4_MMA=0` stubs
5. **Runtime:** tiny tensor `[1, 8] × [8, 16]` through the real MMA probe

## JSON Schema

```json
{
  "stamp": "20260519_123456",
  "torch_version": "2.x.y",
  "torch_cuda": "12.8",
  "cuda_available": true,
  "device_name": "NVIDIA RTX PRO 6000 Blackwell",
  "capability": [12, 0],
  "cute_header_found": true,
  "cute_include_paths": ["/path/to/deep_gemm/include"],
  "build_dir": "/tmp/lynn_engine_native_build/p198_20260519_123456",
  "extension_loaded": true,
  "extension_path": "/tmp/.../lynn_native_runtime.so",
  "available_symbols": ["dense_fp4xfp8_mma_scaled_probe", ...],
  "missing_symbols": [],
  "probe_result": "pass|fail|skipped",
  "probe_output_shape": [8],
  "probe_output_dtype": "torch.float32",
  "decision": "READY_FOR_P190",
  "elapsed_s": 45.2
}
```

## How to Run

```bash
cd /root/autodl-tmp/lynn-engine
bash scripts/r6000_qwen35_9b_native_fp4_build_preflight.sh

# Custom report dir:
REPORT_DIR=/tmp/p198_test bash scripts/r6000_qwen35_9b_native_fp4_build_preflight.sh
```

## Results

**TODO: fill after R6000 run.**

| Field | Value |
|-------|-------|
| decision | — |
| device_name | — |
| capability | — |
| cute_header_found | — |
| extension_loaded | — |
| available_symbols | — |
| probe_result | — |
| elapsed_s | — |

### If BLOCKED_COMPILE

**TODO: fill if this case triggers.**

The error tail will be in `load_error_tail` in the JSON. Typical causes:
- nvcc not found in PATH
- Missing CUDA headers (`cuda_fp8.h`, `cuda_bf16.h`)
- CuTe header not found (wrong include path)
- SM120a not supported by installed CUDA toolkit version

### If BLOCKED_SYMBOL_MISSING

Extension compiled but `#if HAS_FP4_MMA` evaluated to 0. Check:
- Was `LYNN_ENABLE_SM120A_FP4_MMA=1` set?
- Was `-arch=sm_120a` passed? (needs `LYNN_NATIVE_CUDA_ARCH_AUTO=1`)
- Does `cute/arch/mma_sm120.hpp` exist in include paths?

---

## Relationship to Other Work

| Item | Relation |
|------|----------|
| P190 | P198 unblocks P190's FP4 MMA path |
| P191 | P198 validates the same `dense_fp4xfp8_poc.cu` build |
| P189 | P189 checks device capability; P189 goes further (build + symbol + runtime) |
| P197 | Independent — P197 is W4A16 vs W4A8 drift, P198 is FP4 MMA readiness |
