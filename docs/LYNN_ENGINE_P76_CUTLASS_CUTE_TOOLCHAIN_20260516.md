# Lynn Engine P76: CUTLASS/CuTe toolchain gate

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Why P76

P75 closes scalar gate/up launch-shape tuning. The next real active-MoE route
needs a grouped per-16 native-FP4 kernel, likely using CuTe/CUTLASS or a custom
CUDA kernel that calls Blackwell FP4 MMA instructions directly.

Before writing that kernel, P76 verifies whether the current R6000 environment
can actually compile CUTLASS/CuTe sm_120 FP4 code.

## Loader change

`engine/native_cuda.py` now discovers optional native include roots:

```text
LYNN_NATIVE_EXTRA_INCLUDE_DIRS
LYNN_CUTLASS_INCLUDE_DIR
LYNN_DEEP_GEMM_INCLUDE_DIR
<current env>/site-packages/deep_gemm/include
/root/miniconda3/lib/python*/site-packages/deep_gemm/include
/root/autodl-tmp/conda-envs/*/lib/python*/site-packages/deep_gemm/include
```

It also prepends the active Python env and base miniconda `bin` directories to
`PATH` so `ninja` and `nvcc` are visible to `torch.utils.cpp_extension.load`.

## Probe

```bash
python benchmarks/p76_cutlass_cute_toolchain_probe.py \
  --out reports/p16_155/p76_cutlass_cute_toolchain_probe.json
```

The smoke extension includes:

```cpp
#include <cutlass/float8.h>
#include <cutlass/float_subbyte.h>
#include <cute/arch/mma_sm120.hpp>
#include <cute/numeric/numeric_types.hpp>
```

and launches a tiny CUDA kernel that verifies:

- `cutlass::float_e2m1_t` is 4-bit;
- `cutlass::float_ue8m0_t` is 8-bit;
- `cute::SM120_16x8x32_TN<float_e2m1_t,float_e2m1_t,float>` is visible.

## Result

| Item | Value |
|---|---|
| PyTorch | `2.10.0+cu128` |
| CUDA capability | `sm_120` |
| include path | `/root/miniconda3/lib/python3.12/site-packages/deep_gemm/include` |
| required headers | all present |
| compile | PASS |
| build time | 40.33s |
| smoke values | `[4.0, 8.0, 1.0, 120.0]` |

## Decision

The CuTe/CUTLASS toolchain gate is green on R6000.

This does **not** mean a vendor ModelOpt kernel can consume the current
Lynn-native artifact directly. P54 already showed that converting Lynn's
per-16 FP32 scale contract into e8m0/group32 loses too much quality.

What P76 proves is narrower and more useful:

```text
we can now write/compile a Lynn-owned grouped per-16 kernel
using the official CUTLASS/CuTe Blackwell FP4 type system
while preserving Lynn's current artifact layout.
```

## Next path

P77 should start from the smallest useful CuTe/CUTLASS sub-kernel:

1. keep the existing P65/P70/P73 active-MoE ABI;
2. build a tiny selected-row FP4 MMA tile against synthetic data first;
3. then connect it to Lynn packed E2M1 + per-16 scales;
4. only after sub-kernel numerics pass, re-enter the P69 active boundary gate.

The promotion rule remains unchanged:

```text
no microbench-only promotion; P69 first, full-generate/server gates second
```
