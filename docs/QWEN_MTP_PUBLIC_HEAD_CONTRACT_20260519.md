# Qwen MTP Public Head Contract

**Date:** 2026-05-19
**Scope:** Qwen3.5-9B Dense + Qwen3.6-35B-A3B MoE
**Status:** SCAFFOLD — inventory tool ready, Spark investigation in progress

---

## What is MTP (Multi-Token Prediction)?

MTP enables speculative decoding using auxiliary "NextN" prediction heads
that predict tokens N steps ahead. The Qwen3.5/3.6 official models ship
with trained MTP heads that can be used for speculative decoding without
any separate draft model.

### Key Architecture

```
Main model → hidden_state → lm_head → token N
                          ↘ mtp_head[0] → token N+1
                          ↘ mtp_head[1] → token N+2
                          ↘ ...
```

Each MTP head consists of:
- A small transformer layer (norm + self-attention + FFN or just norm + linear)
- A separate lm_head projection for that offset

## Verification Strategy

### Phase 1: Public NextN (no retraining)

Use the official MTP heads as shipped by Qwen:
- offset=1 head predicts token N+1 given hidden state at N
- Accept rate depends on model quality and prompt distribution

**Verification fields:**
| Field | Description | Target |
|-------|-------------|--------|
| `accept_rate` | Fraction of speculated tokens accepted | ≥ 0.60 for offset=1 |
| `accept_len` | Average accepted draft length per step | ≥ 1.5 tokens |
| `target_offset` | Which NextN head (1-indexed) | 1 (first public head) |
| `speedup` | Wall-clock TPS improvement vs greedy | ≥ 1.3x |
| `quality_match` | Greedy output identical with/without MTP | TRUE (spec decode is lossless) |

### Phase 2: Offset=2 Retrain (future)

If offset=1 accept_rate is insufficient, retrain a custom head with:
- offset=2 (predicts 2 tokens ahead)
- Lynn-specific training data
- Possible architecture changes (deeper head, cross-attention)

This phase is explicitly **NOT part of the first release.**

## Inventory Tool

```bash
# Scan model for MTP tensors:
python scripts/qwen_mtp_public_head_inventory.py --model-dir /path/to/model

# With output:
python scripts/qwen_mtp_public_head_inventory.py \
  --model-dir /path/to/model \
  --out reports/mtp/inventory_9b.json
```

### Output Schema

```json
{
  "schema": "lynn-qwen-mtp-public-head-inventory-v1",
  "model_dir": "/path/to/model",
  "model_config": {
    "model_type": "qwen3",
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "mtp_num_hidden_layers": 1
  },
  "total_tensors": 500,
  "mtp_tensors_count": 12,
  "mtp_tensors": [
    {"key": "model.mtp.0.lm_head.weight", "shape": [151936, 4096], "dtype": "BF16", ...}
  ],
  "mtp_total_mib": 1200.5
}
```

## Expected MTP Tensor Layout

### Qwen3.5-9B Dense (if MTP present)

| Key pattern | Shape | Notes |
|-------------|-------|-------|
| `model.mtp.{i}.embed_tokens.weight` | [vocab, hidden] | May be tied |
| `model.mtp.{i}.enorm.weight` | [hidden] | Layer norm |
| `model.mtp.{i}.hnorm.weight` | [hidden] | Layer norm |
| `model.mtp.{i}.eh_proj.weight` | [hidden*2, hidden] | Projection |
| `model.mtp.{i}.lm_head.weight` | [vocab, hidden] | NextN output head |

### Qwen3.6-35B-A3B MoE (confirmed on Spark)

Same layout but with `hidden_size=2048` and `vocab_size=151936`.
MTP heads are dense (not MoE) even when the main model is MoE.

## Acceptance Criteria for Phase 1 Promotion

| Criterion | Threshold | Notes |
|-----------|-----------|-------|
| MTP heads present in model | ≥ 1 head | inventory check |
| Tensor shapes match config | all | automated validation |
| SHA256 stable across loads | reproducible | no random init |
| accept_rate (greedy, 1K tokens) | ≥ 0.60 | measured on Spark/R6000 |
| accept_len mean | ≥ 1.5 | measured on eval prompts |
| Output lossless vs greedy | TRUE | spec decode must be exact |
| TPS speedup | ≥ 1.3x | wall-clock improvement |

## What This Contract Does NOT Cover

1. Does NOT retrain any head (Phase 2 only)
2. Does NOT modify engine serving code (separate PR)
3. Does NOT test on GGUF/llama.cpp (MTP is engine-only)
4. Does NOT promise a specific accept_rate (model-dependent)
5. Does NOT block the 9B first release (MTP is a speed optimization)

## References

| Item | Location |
|------|----------|
| Inventory tool | `scripts/qwen_mtp_public_head_inventory.py` |
| Shell wrapper | `scripts/qwen_mtp_public_head_inventory.sh` |
| Spark MTP repro gate | `docs/QWEN36_35B_SPARK_MTP_REPRO_GATE_20260519.md` |
| Lynn MTP 2048-head spec | `docs/LYNN_ENGINE_MTP_LYNN_2048_HEAD_SPEC_20260516.md` |
| MTP verify ABI | `docs/LYNN_ENGINE_MTP_VERIFY_ABI_20260517.md` |
