# Stage 6 Phase 4 - native fused MoE zero-shadow ABI contract

Date: 2026-06-04

Verdict: **ABI/PREFLIGHT ONLY; no fused P4 kernel is banked yet.**

P3 proved the zero-reload serving path can be quality-smoked behind an opt-in
server flag, but it still uses the existing packed/Triton active-MoE pieces.
P4 is where the Lynn-owned C++/CUDA hot path becomes a real replacement point:
one native boundary, caller-owned output, packed NVFP4 weights only, and no BF16
expert shadow.

## Why this exists

The language decision is now explicit:

- Python remains the research, orchestration, quantization, and eval layer.
- Triton remains the fastest prototype path.
- CUDA C++ owns the future hot path where launch count, scratch ownership, and
  weight-layout contracts must stop drifting.

This P4 contract does not claim speed. It makes the future fused kernel an
extension symbol that must compile, bind, and fail loudly until the real CUDA
math replaces the placeholder boundary.

## P4 ABI

New native symbol:

```text
active_moe_fused_zero_shadow_out_contract(
  hidden[T, 2048] bf16 contiguous,
  expert_ids[T, top_k] int32 contiguous,
  routing_weights[T, top_k] fp32 contiguous,
  gate_up_packed[E, 1024, 1024] uint8 contiguous,
  gate_up_scale[E, 1024, 128] fp32 contiguous,
  gate_up_global_scale[1] fp32 contiguous,
  down_packed[E, 2048, 256] uint8 contiguous,
  down_scale[E, 2048, 32] fp32 contiguous,
  down_global_scale[1] fp32 contiguous,
  out[T, 2048] bf16 contiguous,
  tile_tokens,
  tile_inter,
  tile_hidden,
) -> void
```

Important properties:

- caller-owned `out`, so the production path can become graph/runtime friendly;
- no BF16 expert weight tensors in the ABI;
- no Python callback or Triton fallback inside the symbol;
- shape/layout guards match the Lynn native per-16 Qwen3.6-35B-A3B active MoE
  contract: `H=2048`, `I=512`, `2I=1024`;
- valid input currently reaches an intentional `P4 fused 4-bit zero-shadow CUDA
  kernel is not implemented yet` error.

## Runtime Bridge

The opt-in backend name is now reserved in `engine/moe_packed_nvfp4.py`:

```bash
LYNN_NATIVE_ACTIVE_MOE_BACKEND=fused_zero_shadow_out_contract
```

It wraps the current decode token as `[1, H]`, passes packed NVFP4 active-expert
weights into `active_moe_fused_zero_shadow_out_contract`, and stops at the same
fail-loud boundary. This backend must remain default-off until the CUDA math is
implemented and the byte-count, numeric, speed, and RC gates pass.

## GPU Preflight

Run through the Spark artifact wrapper after syncing the branch:

```bash
scripts/run_spark_stage6_p4_native_abi_preflight.sh
```

The wrapper records remote HEAD/provenance manifest, `nvidia-smi` before/after,
Docker exit code, `run.log`, `result.json`, and `summary.md` under
`reports/stage6/p4_native_abi_preflight_*`.

Direct in-container command:

```bash
python3 scripts/spark_stage6_p4_native_abi_preflight.py \
  --out reports/stage6/p4_native_abi_preflight/result.json \
  --strict-exit
```

Expected bankable preflight decision:

```text
PASS_ABI_CONTRACT
```

That means the extension built, the symbol exists, a valid packed-NVFP4 tensor
bundle passed all static guards, and the call stopped only at the intentional
not-implemented boundary.

Non-bankable decisions:

| Decision | Meaning |
|---|---|
| `BLOCKED_NO_CUDA` | host is not a CUDA test node |
| `BLOCKED_COMPILE` | native extension did not build/load |
| `BLOCKED_SYMBOL_MISSING` | pybind/source list drift |
| `BLOCKED_GUARD_OR_RUNTIME` | tensor ABI drift or unexpected runtime failure |
| `UNEXPECTED_IMPLEMENTED` | someone removed fail-loud before adding evidence gates |

## Promotion Boundary

P4 is not promoted by this ABI gate. A real fused kernel can be banked only with:

- byte-count proof that it reads packed NVFP4 bytes, not BF16 expert shadows;
- numeric parity against BF16/P3 references;
- e2e TPS or layer microbench with honest launch/latency accounting;
- RC quality smoke before any server/default promotion.

GPU-free static check:

```bash
python3 scripts/test_stage6_p4_native_abi_static.py
```

GPU-free evidence-tooling self-test:

```bash
python3 scripts/test_stage6_p4_evidence_tools.py
```

Formal report writer:

```bash
python3 scripts/write_stage6_p4_native_abi_report.py \
  reports/stage6/p4_native_abi_preflight_<timestamp> \
  --report-out reports/stage6/P4_NATIVE_ABI_PREFLIGHT_RESULT_20260604.md
```
