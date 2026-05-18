# Qwen3.6-35B W4A16 Kernel Refactor Plan — 2026-05-18

## Context

Lynn Engine Phase 4/5 has proven:
1. NVFP4 v8-RTN dequant path works (P2 slow path)
2. Native `torch._scaled_mm` FP4 can match scalar-bridge numerics (P4/P5)
3. Triton fused MoE expert FFN kernels exist but are not yet validated on GPU (P3.2.3)

**Bottleneck:** Every kernel iteration requires loading the full 35B model
(40 layers × 256 experts × 3 projections = ~24 GB dequant), which takes 3-5
minutes on R6000. This makes tight iteration loops on native grouped GEMM
kernels impractical.

## Stream D Goal

**Reduce native MoE kernel development from "full model load" to "fixture fast target".**

Export real intermediate activations (hidden states, expert routing decisions,
ground truth outputs) from the full model once, then use those tiny fixtures
(~16 KB each) for rapid kernel validation.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  p133: Export (one-time, ~5 min on R6000)                    │
│                                                               │
│  Full 35B model → per (layer, prompt):                       │
│    hidden_in[1, 2048]  expert_ids[8]  routing_weights[8]     │
│    moe_output[1, 2048] (ground truth)                        │
│    + sidecar metadata (NVFP4 scale paths)                    │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  p134: Contract Test (fast, ~10s per fixture)                │
│                                                               │
│  Load fixture → run reference Triton MoE → compare           │
│  Optionally: run candidate native kernel → compare vs ref    │
│                                                               │
│  Metrics: max_abs, mean_abs, rel_l2, cosine, exact, ms      │
│  Gate:    max_abs=0 for Triton self-check                    │
│           max_abs<threshold for native candidate              │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Stream A: Native Grouped Kernel Development                 │
│                                                               │
│  Developer workflow:                                          │
│    1. Write/modify kernel in csrc/lynn_native/               │
│    2. Expose as candidate backend                            │
│    3. Run p134 with --candidate-backend native_grouped       │
│    4. Iterate until GREEN                                    │
│                                                               │
│  No model loading needed! ~10s per test cycle.               │
└─────────────────────────────────────────────────────────────┘
```

## Fixture Format

Each fixture is a safetensors file containing:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `hidden_in` | `[1, 2048]` | BF16 | MoE sublayer input (post-attn-norm) |
| `expert_ids` | `[8]` | int32 | Top-K expert indices from router |
| `routing_weights` | `[8]` | float32 | Softmax routing weights |
| `moe_output` | `[1, 2048]` | BF16 | Ground truth MoE output |

Plus a `manifest.json` with metadata, sidecar paths, and per-fixture info.

## Sidecar Contract

For NVFP4 v8-RTN models, the fixture manifest records the path to packed expert
weights (`.weight_packed`, `.weight_scale`, `.weight_global_scale`) so that
native kernel candidates can:
1. Load the packed weights directly (no dequant)
2. Run native FP4 GEMM on the fixture's `hidden_in`
3. Compare against fixture's `moe_output`

This enables testing the full native quantized path end-to-end.

## Acceptance Criteria

1. `python -m py_compile` passes for both p133 and p134
2. R6000: p133 exports 18+ fixtures (9 layers × 2 prompts)
3. R6000: p134 self-check produces `max_abs=0` for ALL fixtures
4. No changes to default serving behavior
5. No conflicts with Stream A (csrc/lynn_native) or Stream B (engine/) files

## Files

| File | Purpose |
|------|---------|
| `benchmarks/p133_export_active_moe_fixtures.py` | Fixture export from full model |
| `benchmarks/p134_active_moe_fixture_contract.py` | Contract test runner |
| `scripts/r6000_export_qwen36_moe_fixtures.sh` | R6000 automation |
| `reports/qwen36_35b/p133_fixtures/` | Exported fixture files |
| `reports/qwen36_35b/p134_*_report.json` | Contract test reports |
| `docs/QWEN36_W4A16_KERNEL_REFACTOR_PLAN_20260518.md` | This document |

## Stream Isolation

| Stream | Scope | Files |
|--------|-------|-------|
| A | Native CUTLASS/CUDA grouped GEMM | `csrc/lynn_native/*` |
| B | Engine serving runtime | `engine/`, `server/` |
| C | Triton kernel optimization | `triton_kernels/` |
| **D** | **Fixture + Contract harness** | `benchmarks/p133_*`, `benchmarks/p134_*`, `scripts/r6000_*` |

Streams are file-disjoint. D provides the test harness that A/C use for validation.

## Next Steps

1. Run `r6000_export_qwen36_moe_fixtures.sh` on R6000
2. Verify GREEN contract
3. Stream A developer integrates candidate backend
4. Iterate kernel until p134 passes with acceptable thresholds
5. When native kernel passes `max_abs < 5e-3, cosine > 0.999`, merge to production
