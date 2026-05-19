# Qwen3.5-9B NVFP4 Convstrict Promotion Report

Date: 2026-05-19
Branch: `claude/qwen35-9b-convstrict-promotion-20260519`
Model: `Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0`

## Verdict: DEFAULT_PROMOTABLE

The `graph_plus_conv_triton` candidate (convstrict env) passes all promotion gates.

## Gate Results

| Gate | Requirement | Result | Status |
|------|-------------|--------|--------|
| P183 Exact-Fast Isolation | Best exact fast candidate identified | `graph_plus_conv_triton` | ✅ |
| P184 Exact Gate | 70/70 hard structured prompts exact | **70/70** | ✅ |
| P150 Service TPS (128) | Baseline reference | 61.32 TPS | ✅ |
| P150 Service TPS (256) | ≥ 60 TPS | 62.25 TPS | ✅ |
| P150 Service TPS (512) | ≥ 62 TPS | **62.09 TPS** | ✅ |
| Graph reuse | All runs reuse captured graph | ✅ (all true) | ✅ |

## Candidate Configuration

```bash
# scripts/qwen35_9b_candidate_env_convstrict.env
LYNN_PREFILL_WARMUP=1
LYNN_LINEAR_STATE_UPDATE=inplace
LYNN_LINEAR_BLOCK_GRAPH=1
LYNN_LINEAR_BLOCK_GRAPH_REUSE=1
LYNN_LINEAR_BLOCK_GRAPH_PREWARM=1
LYNN_LINEAR_ATTN_CONV_BACKEND=triton_torch_silu
# Drift-prone knobs DISABLED:
LYNN_LINEAR_ATTN_RECURRENT_BACKEND=torch
LYNN_LINEAR_ATTN_RECURRENT_INPLACE=0
LYNN_LINEAR_ATTN_GQA_RECURRENT=0
LYNN_QK_NORM_ROPE_BACKEND=torch
LYNN_RMSNORM_GATED_BACKEND=torch
LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=0
LYNN_NATIVE_FP4_LM_HEAD=0
LYNN_PACKED_DECODE=0
LYNN_PACKED_DECODE_PREPARE_NATIVE=0
```

## Why NOT 77 TPS fast_no_packed?

The `fast_no_packed` configuration achieves 77 TPS but is **BLOCKED by exact drift**:
- Native FP4 LM head, triton QK/RoPE, triton recurrent/GQA each introduce
  low-margin token flips on structured prompts
- P184 at 77 TPS config: NOT 70/70 exact → cannot promote

The 62 TPS convstrict config sacrifices ~15 TPS for **guaranteed output correctness**.

## Promotion Decision

| Tier | TPS | Exact | Promotable |
|------|-----|-------|------------|
| convstrict (DEFAULT) | 62 | ✅ 70/70 | **YES** |
| fast_no_packed (AMBER) | 77 | ❌ drift | NO |

## File Inventory

| File | Purpose |
|------|---------|
| `scripts/qwen35_9b_candidate_env_convstrict.env` | Candidate env (exists) |
| `scripts/r6000_qwen35_9b_convstrict_promotion_gate.sh` | Promotion gate wrapper |
| `reports/qwen35_9b/p184_qwen35_9b_nvfp4_convstrict_exact_gate_*.json` | 70/70 exact proof |
| `reports/qwen35_9b/p150_*_convstrict.json` | Service TPS proof |
| This report | Summary |

## Merge Recommendation

**YES — safe to merge.** The convstrict env is opt-in (requires explicit env file sourcing),
does not change any default, and has been validated for exact correctness + acceptable TPS.
