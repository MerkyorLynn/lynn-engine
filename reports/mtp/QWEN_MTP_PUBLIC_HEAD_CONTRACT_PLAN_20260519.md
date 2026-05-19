# Qwen MTP Public Head Contract Plan

**Date:** 2026-05-19
**Models:** Qwen3.5-9B Dense, Qwen3.6-35B-A3B MoE
**Status:** INVENTORY READY — Phase 1 verification requires Spark/R6000

---

## Plan Overview

```
┌─────────────────────────────────────────┐
│  Step 1: Inventory (CPU-only)           │  ← READY NOW
│  - Scan safetensors for mtp.* keys      │
│  - Record shape, dtype, sha256_prefix   │
│  - Detect model_type, hidden_size       │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│  Step 2: Load & Forward (GPU, Spark)    │  ← NEEDS SPARK
│  - Load MTP head weights                │
│  - Forward pass with main hidden_state  │
│  - Verify output shape matches vocab    │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│  Step 3: Speculative Decode (GPU)       │  ← NEEDS ENGINE INTEGRATION
│  - Run greedy baseline (no MTP)         │
│  - Run speculative with MTP head        │
│  - Measure accept_rate, accept_len      │
│  - Verify output is lossless            │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│  Step 4: Performance Gate               │
│  - Measure TPS with/without MTP         │
│  - Compute speedup                      │
│  - Gate: speedup ≥ 1.3x                 │
└─────────────────────────────────────────┘
```

## Step 1: Inventory (Done)

Tool: `scripts/qwen_mtp_public_head_inventory.py`

Run locally or on any machine with the model files:

```bash
# 9B:
python scripts/qwen_mtp_public_head_inventory.py \
  --model-dir /path/to/Qwen3.5-9B \
  --out reports/mtp/qwen35_9b_mtp_inventory.json

# 35B:
python scripts/qwen_mtp_public_head_inventory.py \
  --model-dir /path/to/Qwen3.6-35B-A3B \
  --out reports/mtp/qwen36_35b_mtp_inventory.json
```

## Step 2: Forward Validation (Spark)

Requires the model loaded on GPU. Spark owner will:
1. Load the model with MTP heads enabled
2. Run a forward pass, capture hidden_state at position N
3. Feed hidden_state through mtp_head[0]
4. Verify output logits shape = [1, vocab_size]
5. Check argmax gives a plausible next-next token

## Step 3: Speculative Decode Verification

### Verification Protocol

```python
# Pseudo-code for verification:
baseline_tokens = greedy_decode(prompt, max_new=64)
speculative_tokens = speculative_decode(prompt, max_new=64, mtp_heads=True)

assert baseline_tokens == speculative_tokens  # lossless
accept_rate = accepted_drafts / total_drafts
accept_len = mean(accepted_lengths)
```

### Acceptance Fields

| Field | Type | Description |
|-------|------|-------------|
| `accept_rate` | float | Fraction of spec tokens accepted (0..1) |
| `accept_len` | float | Mean accepted draft length per step |
| `target_offset` | int | Which MTP head (1 = next-next token) |
| `speedup` | float | TPS(with MTP) / TPS(without MTP) |
| `output_lossless` | bool | Speculative output == greedy output |
| `total_tokens_generated` | int | Total tokens in verification run |
| `total_drafts` | int | Total speculative attempts |
| `model_type` | str | "qwen3" or "qwen3_moe" |
| `hidden_size` | int | Model hidden dimension |
| `mtp_num_hidden_layers` | int | Number of MTP heads |

## Step 4: Performance Gate

| Metric | Phase 1 Target | Notes |
|--------|---------------|-------|
| accept_rate | ≥ 0.60 | Official head, offset=1 |
| accept_len | ≥ 1.5 | Mean tokens accepted per draft |
| speedup | ≥ 1.3x | Must compensate for MTP overhead |
| lossless | TRUE | Non-negotiable for decode correctness |

If Phase 1 targets are not met with the official head, proceed to Phase 2
(retrained head with offset=2).

## Phase 2: Offset=2 Retrain (Future)

Not part of this contract. Documented for planning only:

- Train a new head predicting offset=2 (2 tokens ahead)
- Use Lynn-specific training data (code, Chinese, structured)
- Possibly use a deeper head architecture
- Target: accept_rate ≥ 0.70, accept_len ≥ 2.0, speedup ≥ 1.5x

## Model Comparison

| Model | Type | Hidden | MTP Layers (expected) | MTP Size (est.) |
|-------|------|--------|----------------------|-----------------|
| Qwen3.5-9B | dense | 4096 | 1 | ~1.2 GiB |
| Qwen3.6-35B-A3B | MoE | 2048 | 1 | ~0.6 GiB |

## Files

| Path | Content |
|------|---------|
| `scripts/qwen_mtp_public_head_inventory.py` | CPU-only tensor scanner |
| `scripts/qwen_mtp_public_head_inventory.sh` | Shell wrapper |
| `docs/QWEN_MTP_PUBLIC_HEAD_CONTRACT_20260519.md` | Contract specification |
| `reports/mtp/QWEN_MTP_PUBLIC_HEAD_CONTRACT_PLAN_20260519.md` | This plan |

## Dependencies

- Step 1: No dependencies (CPU, local)
- Step 2: Spark GPU + model loaded
- Step 3: Engine integration (separate PR, not this scaffold)
- Step 4: Same as Step 3

## Non-Goals

- No GGUF/llama.cpp MTP (not supported)
- No training in this contract (Phase 2 is separate)
- No server/HTTP changes
- No kernel changes (csrc untouched)
