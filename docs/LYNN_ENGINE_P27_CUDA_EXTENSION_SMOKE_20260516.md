# Lynn Engine P27 — CUDA Extension Smoke Gate (2026-05-16)

P16-P26 narrowed the remaining 155 TPS gap to the active routed expert path.
P24 and P26 also ruled out the most tempting Triton-only shortcuts:

| Route | Result |
|---|---|
| per-16 dequant then `tl.dot` | numerically OK, slower |
| merged-top-k Triton scheduling | numerically OK, ~2.1x slower |

That leaves the next serious path:

```text
custom per-16 grouped native-FP4 active expert kernel
```

Before writing that kernel, P27 proves the R6000 build stack can compile, load,
and launch a Lynn-owned CUDA extension against the active PyTorch/CUDA runtime.

## Probe

Script:

```text
benchmarks/p27_cuda_extension_smoke.py
```

Sources:

```text
csrc/lynn_native/bindings.cpp
csrc/lynn_native/smoke_kernel.cu
```

The kernel is intentionally tiny: `add_one(float32)` over a 1M-element CUDA
tensor. This is not a performance optimization. It is a build/load/launch gate.

## R6000 Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | sm_120 |
| PyTorch | 2.10.0+cu128 |
| CUDA toolkit | 12.8 |
| nvcc | `/usr/local/cuda/bin/nvcc` |
| `TORCH_CUDA_ARCH_LIST` | `12.0` |

One environment issue surfaced and was fixed: the `r6000-eval` environment had
the `ninja` Python package installed, but non-interactive SSH did not include
the environment's `bin` directory in `PATH`. The smoke script now prepends
`Path(sys.executable).parent` before invoking `torch.utils.cpp_extension.load`.

## Result

```json
{
  "pass": true,
  "build_seconds": 43.271867990493774,
  "device_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
  "device_capability": [12, 0],
  "torch_version": "2.10.0+cu128",
  "torch_cuda": "12.8",
  "max_abs": 0.0,
  "avg_ms": 0.004708159863948822
}
```

## Decision

P27 passes.

The next 155 TPS attempt no longer needs to worry about repository or build
plumbing for native code. The remaining work is the actual active expert kernel
contract:

```text
packed E2M1 values + Lynn per-16 scales + top-k routed expert ids
    -> gate/up/down expert output
```

In practice, that means P28 should move from a smoke kernel to a real native
MoE CUDA extension scaffold, then incrementally replace the current scalar
bridge with a quality-gated grouped native-FP4 expert implementation.
