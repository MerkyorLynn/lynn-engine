# P134 MoE Fixture Contract Test — R6000 Report

Date: 2026-05-18 (pending R6000 execution)

Machine: RTX PRO 6000 Blackwell

## Purpose

Verify that the Triton active-MoE reference kernel reproduces the stored p133
fixture outputs exactly (max_abs=0). This validates the fixtures as a
deterministic contract gate for native kernel development.

## Test Matrix

| Fixture | Layer | Prompt | Expected max_abs | Expected cosine |
|---------|-------|--------|-----------------|-----------------|
| layer_00_prompt_00 | 0 | "Hello" | 0.0 | 1.0 |
| layer_00_prompt_01 | 0 | "The capital..." | 0.0 | 1.0 |
| layer_04_prompt_00 | 4 | "Hello" | 0.0 | 1.0 |
| layer_04_prompt_01 | 4 | "The capital..." | 0.0 | 1.0 |
| layer_08_prompt_00 | 8 | "Hello" | 0.0 | 1.0 |
| layer_08_prompt_01 | 8 | "The capital..." | 0.0 | 1.0 |
| layer_16_prompt_00 | 16 | "Hello" | 0.0 | 1.0 |
| layer_16_prompt_01 | 16 | "The capital..." | 0.0 | 1.0 |
| layer_20_prompt_00 | 20 | "Hello" | 0.0 | 1.0 |
| layer_20_prompt_01 | 20 | "The capital..." | 0.0 | 1.0 |
| layer_28_prompt_00 | 28 | "Hello" | 0.0 | 1.0 |
| layer_28_prompt_01 | 28 | "The capital..." | 0.0 | 1.0 |
| layer_32_prompt_00 | 32 | "Hello" | 0.0 | 1.0 |
| layer_32_prompt_01 | 32 | "The capital..." | 0.0 | 1.0 |
| layer_36_prompt_00 | 36 | "Hello" | 0.0 | 1.0 |
| layer_36_prompt_01 | 36 | "The capital..." | 0.0 | 1.0 |
| layer_39_prompt_00 | 39 | "Hello" | 0.0 | 1.0 |
| layer_39_prompt_01 | 39 | "The capital..." | 0.0 | 1.0 |

## Execution Modes

### Mode 1: Triton Self-Check (default)

```bash
python benchmarks/p134_active_moe_fixture_contract.py \
    --fixtures reports/qwen36_35b/p133_fixtures \
    --model-dir $MODEL_DIR \
    --max-abs-threshold 0.0 \
    --cosine-threshold 1.0
```

Expected: ALL GREEN (max_abs=0 for every fixture).

Rationale: The reference forward in p134 uses the same `F.linear` + loop-over-active-experts
code path as p133's capture. Given identical inputs + weights, the output must be
bit-exact. Any deviation indicates a bug in the fixture pipeline.

### Mode 2: Native Candidate (future)

```bash
python benchmarks/p134_active_moe_fixture_contract.py \
    --fixtures reports/qwen36_35b/p133_fixtures \
    --model-dir $MODEL_DIR \
    --candidate-backend native_grouped \
    --max-abs-threshold 0.01 \
    --cosine-threshold 0.999
```

Thresholds for native kernel acceptance:
- `max_abs < 5e-3` (FP16 ULP floor for multi-step compute)
- `cosine > 0.999` (strong correlation)
- `rel_l2 < 0.05` (relative error within 5%)

## Contract Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| max_abs | max\|ref - cand\| | Worst-case error |
| mean_abs | mean\|ref - cand\| | Average error |
| rel_l2 | \|\|diff\|\|_2 / \|\|ref\|\|_2 | Scale-invariant error |
| cosine | cos(ref, cand) | Direction agreement |
| exact | 1 if max_abs==0 | Bit-exact flag |
| ref_ms | CUDA timer | Reference latency |
| candidate_ms | CUDA timer | Candidate latency |

## Status

**PENDING** — Awaiting R6000 execution after p133.

## Implications

Once p134 self-check passes GREEN:
1. Fixtures are certified deterministic
2. Stream A developers can use them as admission gate
3. No full model load needed for kernel iteration
4. Expected speedup: ~5 min (full load) → ~10s (fixture load + test)
