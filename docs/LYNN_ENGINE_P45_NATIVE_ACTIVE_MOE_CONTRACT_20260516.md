# Lynn Engine P45 · Native Active-MoE Contract

Date: 2026-05-16

## Summary

P45 adds a one-call CUDA extension ABI for the active MoE path:

```text
active_moe_scalar_contract(
  hidden,
  expert_ids,
  routing_weights,
  gate_up_packed,
  gate_up_scale,
  gate_up_global_scale,
  down_packed,
  down_scale,
  down_global_scale,
) -> moe_out[2048]
```

This is intentionally not a production speed path yet.  The current
implementation delegates to the existing scalar reference gate/up and down
kernels inside the extension.  Its purpose is to freeze the Python/CUDA boundary
for the real grouped/block-diagonal FP4 kernel that follows.

## Evidence

Report:

```text
reports/p16_155/p45_native_active_moe_contract_gate.json
```

Mean latency across layers 2, 8, 14, 20, 28, and 36:

| Path | Mean latency |
|---|---:|
| Current Triton active MoE | 0.058347 ms |
| CUDA scalar two-call reference | 0.065891 ms |
| CUDA scalar one-call contract | 0.065829 ms |

```text
contract_vs_two_call_speedup = 1.0009x
contract_vs_triton_speedup   = 0.8863x
min contract-vs-two-call cosine = 0.99999988
min contract-vs-triton cosine   = 0.99999988
```

## Decision

Do not promote `cuda_scalar_contract` as a production backend.  It is an ABI
scaffold only.

The important outcome is that the extension boundary now has the exact runtime
shape the next kernel needs.  P46 can replace the scalar internals without
touching `engine.moe_packed_nvfp4` or the resident runner contract.

## Runtime Hook

The opt-in diagnostic backend is:

```bash
export LYNN_NATIVE_ACTIVE_MOE_BACKEND=cuda_scalar_contract
export LYNN_MOE_FAST_FIXED=0
```

Keep the production default on Triton until a future kernel passes full gates.

## Next

P46 should implement a first true grouped/block-diagonal active expert kernel
behind this same ABI.  The first milestone does not need to be fully optimized;
it must beat the scalar contract while preserving layer-level parity against
Triton.  Full-generate parity and server TPS gates remain mandatory before any
promotion.
