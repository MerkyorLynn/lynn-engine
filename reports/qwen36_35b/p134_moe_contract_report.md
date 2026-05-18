# P134 MoE Fixture Contract Test — R6000 Report

Date: 2026-05-18

Machine: RTX PRO 6000 Blackwell

## Purpose

Verify that the Triton active-MoE reference kernel reproduces the stored p133
fixture outputs exactly (max_abs=0). This validates the fixtures as a
deterministic contract gate for native kernel development.  The v2 contract also
accepts a directory of precomputed candidate outputs, so native kernels can
write safetensors and be checked without loading layer weights again.

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
    --fixtures reports/qwen36_35b/p133_fixtures_official_w4a16 \
    --model-dir $MODEL_DIR \
    --max-abs-threshold 0.0 \
    --cosine-threshold 0.999999
```

R6000 result: ALL GREEN (max_abs=0 for every fixture).

Rationale: The reference forward in p134 uses the same `F.linear` + loop-over-active-experts
code path as p133's capture. Given identical inputs + weights, the output must be
bit-exact. Any deviation indicates a bug in the fixture pipeline.

### Mode 2: Native Candidate Backend (future)

```bash
python benchmarks/p134_active_moe_fixture_contract.py \
    --fixtures reports/qwen36_35b/p133_fixtures_official_w4a16 \
    --model-dir $MODEL_DIR \
    --candidate-backend native_grouped \
    --max-abs-threshold 0.01 \
    --cosine-threshold 0.999
```

### Mode 3: Precomputed Candidate Outputs

```bash
python benchmarks/p134_active_moe_fixture_contract.py \
    --fixtures reports/qwen36_35b/p133_fixtures_official_w4a16 \
    --candidate-output-dir /path/to/native_candidate_outputs \
    --max-abs-threshold 0.0 \
    --cosine-threshold 1.0
```

The candidate directory may mirror fixture filenames and contain `moe_output`,
`routed_output`, `candidate_output`, or `output`.  This is the fastest path for
CUDA kernel developers: write outputs once, then let p134 compare them against
the same fixture contract.

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

## R6000 Result

| Check | Result |
|---|---:|
| fixtures | 18 |
| layers | 0, 4, 8, 16, 20, 28, 32, 36, 39 |
| prompts | 2 |
| contract schema | lynn-moe-contract-v2 |
| passed / total | 18 / 18 |
| max abs | 0.0 |
| mean abs | 0.0 |
| rel L2 | 0.0 |
| min cosine | 1.0 |
| mean reference latency | 0.963 ms |
| max reference latency | 0.983 ms |
| candidate-output-dir self-check | GREEN, 18/18 |
| verdict | GREEN |

The fixture set is now a valid fast admission gate for Stream A native active
MoE candidates.  A native candidate should pass P134 before moving to P37,
P25, or hard structured gates.

## Status

**GREEN** — R6000 execution complete.

## Implications

The p134 self-check passed GREEN:
1. Fixtures are certified deterministic.
2. Stream A developers can use them as an admission gate.
3. No full model load is needed for candidate-kernel fixture iteration.
4. The fixture set cuts the first correctness loop from full service gates to a
   small 18-case target before P37/P25 escalation.
