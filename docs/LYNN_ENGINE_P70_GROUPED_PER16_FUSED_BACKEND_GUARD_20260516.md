# Lynn Engine P70: grouped per-16 fused backend guard

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Why P70

P68 proved that a two-stage tiled reference path can compose gate/up and down
under the grouped per-16 ABI, but its 1.108x active-MoE boundary speedup is not
strong enough for the 155TPS path. The next implementation must be a true
one-boundary fused kernel, not another runtime toggle around the P68 reference.

P70 reserves an explicit runtime backend name for that target:

```bash
export LYNN_NATIVE_ACTIVE_MOE_BACKEND=grouped_per16_fused
```

This name is separate from:

- `grouped_per16`: broad future ABI guard;
- `grouped_per16_tile_reference`: P68 measurement path;
- `cuda_scalar` / `cuda_tile_inter`: rejected scalar/tile bridge experiments.

## Code

```text
csrc/lynn_native/bindings.cpp
csrc/lynn_native/moe_scalar_kernel.cu
engine/moe_packed_nvfp4.py
benchmarks/p70_grouped_per16_fused_backend_guard.py
```

The native extension now exposes:

```text
active_moe_grouped_per16_fused_contract(...)
```

Today it only performs shape/layout checks and then fails loudly. This is
intentional: the final CUDA/CUTLASS kernel should replace this function body
without changing the Python/runtime contract.

## Gate

```bash
python benchmarks/p70_grouped_per16_fused_backend_guard.py \
  --model /root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final \
  --out reports/p16_155/p70_grouped_per16_fused_backend_guard.json \
  --layer 28
```

Report:

```text
reports/p16_155/p70_grouped_per16_fused_backend_guard.json
```

Result:

| Check | Value |
|---|---|
| Triton baseline norm | 1.18522 |
| backend name | `grouped_per16_fused` |
| shape/layout checks | pass |
| expected exception type | `RuntimeError` from native `TORCH_CHECK` |
| expected message | contains `fused grouped per-16` and `two-stage intermediate tensor` |
| gate pass | true |

## Decision

P70 does not claim speed. It creates the clean replacement point for the actual
fused grouped per-16 active expert kernel:

```text
hidden[2048]
expert_ids[top_k=8]
routing_weights[top_k=8]
gate_up packed/scale/global
down packed/scale/global
-> out[2048]
```

Future P71+ work should fill this backend with a real one-boundary kernel and
then run P69 acceptance before any full-generate/server promotion work.
