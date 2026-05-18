# Qwen3.6 35B W4A16 Native MoE P155 Raw Gate/Up Accumulator Diagnostic

Date: 2026-05-19

## Purpose

P155 records native pre-SiLU FP32 `gate_acc` and `up_acc` tensors with shape
`[top_k, 2, 512]` for the packed NVFP4 slot gate/up path.  The benchmark also
builds an inline Triton raw reference kernel that mirrors P147 gate/up and
stores raw gate/up before applying SiLU and BF16 inter rounding.

## Added Native Symbols

- `moe_slot_packed_nvfp4_raw_accum_probe`
- `moe_slot_packed_nvfp4_raw_accum_triton_order_probe`

Both return FP32 `[top_k, 2, 512]`, where `[:, 0, :]` is raw gate and
`[:, 1, :]` is raw up.

## Benchmark

Run on R6000:

```bash
bash scripts/r6000_qwen36_moe_p155_gateup_raw_accum.sh
```

The JSON report compares:

- native raw gate accumulator vs Triton raw gate accumulator;
- native raw up accumulator vs Triton raw up accumulator;
- native post-SiLU BF16 inter vs Triton post-SiLU BF16 inter;
- raw-derived native inter vs the native kernel inter, to isolate the
  SiLU/BF16 boundary after raw accumulation.

## Guardrails

This is a fixture diagnostic only.  It does not run P37/P25 and does not touch
resident runner, incremental decode, or server code.

## R6000 Result

Artifact:
`reports/qwen36_35b/p155_native_packed_gateup_raw_accum_20260519_041757.json`

Verdict: **RAW_ACCUM_DRIFT**

| Check | Exact | Max Abs | Mean Latency |
|---|---:|---:|---:|
| inline Triton inter vs P147 inter | 18/18 | 0 | 0.03439 ms raw |
| native raw gate vs Triton raw gate | 0/18 | 9.536743e-7 | 0.04123 ms raw |
| native raw up vs Triton raw up | 0/18 | 9.536743e-7 | 0.04123 ms raw |
| native post-SiLU inter vs Triton inter | 6/18 | 2.44140625e-4 | 0.04121 ms inter |
| Triton-order raw gate vs Triton raw gate | 0/18 | 9.536743e-7 | 0.06575 ms raw |
| Triton-order raw up vs Triton raw up | 0/18 | 9.536743e-7 | 0.06575 ms raw |
| Triton-order post-SiLU inter vs Triton inter | 10/18 | 2.44140625e-4 | 0.06574 ms inter |

The P155 inline Triton reference exactly matches the P147 reference inter, so
the diagnostic reference is valid. Both native raw-accumulator variants drift
before SiLU. The raw max_abs is small (~1e-6), but it is enough to cross BF16
rounding thresholds after SiLU and produce the 2.44e-4 intermediate jumps seen
in P153/P154.

## Updated Interpretation

The blocker is not down/reduce and not only the BF16 store boundary. The exact
fix must make raw gate/up accumulation bit-equivalent to Triton. Candidate next
experiments should focus on:

1. matching Triton `tl.sum` reduction order exactly for each 256-column block;
2. matching FP4 decode/scale multiply expression order at the scalar operation
   level;
3. emitting a slower reference C++ gate/up path that computes each row in the
   same visible order as Triton to confirm whether exactness is possible before
   optimizing it.
