# Lynn Engine P60: grouped per-16 backend guard

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Why P60

P56/P58 closed the scalar tile-inter shortcut.  The next real route is no
longer "try another env var"; it is a true grouped per-16 active expert kernel.

P60 creates the stable runtime name for that future implementation:

```bash
export LYNN_NATIVE_ACTIVE_MOE_BACKEND=grouped_per16
```

Today, that backend must fail loudly.  It must not silently fall back to:

- `cuda_scalar`;
- `cuda_scalar_contract`;
- `cuda_tile_inter`;
- Triton.

## Code added

```text
engine/moe_packed_nvfp4.py
benchmarks/p60_grouped_per16_backend_guard.py
```

`grouped_per16` is now a recognized backend name, but it raises
`NotImplementedError` with an explicit message:

```text
reserved for the true grouped per-16 native-FP4 active expert kernel
```

## Gate

```bash
python benchmarks/p60_grouped_per16_backend_guard.py \
  --model /root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final \
  --out reports/p16_155/p60_grouped_per16_backend_guard.json
```

The gate verifies:

1. the current Triton baseline path still runs;
2. `grouped_per16` raises the expected fail-loud error;
3. no rejected scalar/tile bridge is used as a hidden fallback.

## Decision

P60 does not claim a speedup. It freezes the production-facing backend ABI for
the next kernel:

```text
hidden[2048]
top-k expert_ids[8]
routing_weights[8]
gate_up packed NVFP4 per-16
down packed NVFP4 per-16
-> active MoE out[2048]
```

The next phase can replace this guard with the real CUDA/CUTLASS kernel while
keeping the same Python/runtime contract and promotion gates.
