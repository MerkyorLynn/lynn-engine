# Stage 6 Phase 4 - native fused MoE zero-shadow ABI contract

Date: 2026-06-04

Verdict: **TWO-STAGE REFERENCE/PREFLIGHT ONLY; no fused P4 kernel is banked yet.**

P3 proved the zero-reload serving path can be quality-smoked behind an opt-in
server flag, but it still uses the existing packed/Triton active-MoE pieces.
P4 is where the Lynn-owned C++/CUDA hot path becomes a real replacement point:
one native boundary, caller-owned scratch/output, packed NVFP4 weights only,
and no BF16 expert shadow.

## Why this exists

The language decision is now explicit:

- Python remains the research, orchestration, quantization, and eval layer.
- Triton remains the fastest prototype path.
- CUDA C++ owns the future hot path where launch count, scratch ownership, and
  weight-layout contracts must stop drifting.

This P4 contract does not claim speed. It makes the future fused kernel an
extension symbol with a conservative T=1 two-stage packed-NVFP4 reference path.
The reference path proves the real boundary can return output without BF16
expert shadows, but it is not the final fused/fast kernel.

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
  inter_scratch[T, top_k, 512] bf16 contiguous,
  out[T, 2048] bf16 contiguous,
  tile_tokens,
  tile_inter,
  tile_hidden,
) -> void
```

Important properties:

- caller-owned `inter_scratch` and `out`, so the production path can become
  graph/runtime friendly without internal allocation;
- no BF16 expert weight tensors in the ABI;
- no Python callback or Triton fallback inside the symbol;
- shape/layout guards match the Lynn native per-16 Qwen3.6-35B-A3B active MoE
  contract: `H=2048`, `I=512`, `2I=1024`;
- valid T=1 decode input currently runs the existing graph-safe two-stage
  packed-NVFP4 scalar reference inside the native extension.

## Runtime Bridge

The opt-in backend name is now reserved in `engine/moe_packed_nvfp4.py`:

```bash
LYNN_NATIVE_ACTIVE_MOE_BACKEND=fused_zero_shadow_out_contract
```

It wraps the current decode token as `[1, H]`, passes packed NVFP4 active-expert
weights into `active_moe_fused_zero_shadow_out_contract`, and reaches the
two-stage reference boundary. This backend must remain default-off until the
byte-count, numeric, speed, and RC gates pass.

The bridge requires `LYNN_MOE_ACTIVE_SCRATCH=1`, which lets
`resident_runner.py` preallocate `mlp.experts._active_inter_scratch [top_k,512]`
and `mlp.experts._active_out_scratch [2048]` per layer. The P4 symbol must not
allocate its own hot-path tensors.

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
PASS_TWO_STAGE_REFERENCE_CONTRACT
```

That means the extension built, the symbol exists, a valid packed-NVFP4 tensor
bundle passed all static guards, and the two-stage reference returned finite
output while `banked_fused_kernel=false`.

The preflight also records an ABI byte budget:

- `packed_weight_bytes`: packed NVFP4 expert bytes plus scale tensors admitted
  into the native symbol;
- `bf16_shadow_equivalent_bytes`: equivalent full BF16 active-expert shadow for
  the same fixture shape;
- `forbidden_shadow_tensor_names`: must be empty;
- `passes.zero_shadow_abi=true` and `passes.packed_byte_budget=true`.

This is an ABI/input proof, not an HBM profiler. The eventual fused kernel still
needs a separate runtime byte-count/profiler artifact before `banked_fused_kernel`
can become true.

## Runtime Bridge Preflight

The synthetic ABI preflight proves the native symbol boundary. The runtime
bridge preflight proves the real resident-runner decode path can reach that
boundary with real layer tensors and no active-expert BF16 shadows.

Run through the Spark artifact wrapper after syncing the branch:

```bash
scripts/run_spark_stage6_p4_runtime_bridge_preflight.sh
```

Direct Spark command:

```bash
python3 scripts/spark_stage6_p4_runtime_bridge_preflight.py \
  --model /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526 \
  --out reports/stage6/p4_runtime_bridge_preflight/result.json \
  --strict-exit
```

Expected bankable bridge decision:

```text
PASS_TWO_STAGE_RUNTIME_BRIDGE
```

That means the runner first produced a nonzero Triton baseline, removed the
active-expert BF16 shadow tensors, switched to
`LYNN_NATIVE_ACTIVE_MOE_BACKEND=fused_zero_shadow_out_contract`, returned a
candidate output from the native two-stage reference while using caller-owned
active-MoE scratch, and passed a smoke-level numeric comparison against the
current Triton path.

Bridge summary command:

```bash
python3 scripts/summarize_stage6_p4_runtime_bridge_preflight.py \
  reports/stage6/p4_runtime_bridge_preflight/result.json \
  --markdown-out reports/stage6/p4_runtime_bridge_preflight/summary.md \
  --strict-exit
```

This banks only `banked_runtime_bridge_preflight=true`. It must keep
`banked_fused_kernel=false` and `banked_default_promotion=false`.

Non-bankable decisions:

| Decision | Meaning |
|---|---|
| `BLOCKED_NO_CUDA` | host is not a CUDA test node |
| `BLOCKED_COMPILE` | native extension did not build/load |
| `BLOCKED_SYMBOL_MISSING` | pybind/source list drift |
| `BLOCKED_GUARD_OR_RUNTIME` | tensor ABI drift or unexpected runtime failure |
| `FAIL_REFERENCE_OUTPUT` | two-stage reference returned non-finite output |

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

GPU-free zero-shadow firewall:

```bash
python3 scripts/test_stage6_p4_zero_shadow_firewall.py
```

Formal report writer:

```bash
python3 scripts/write_stage6_p4_native_abi_report.py \
  reports/stage6/p4_native_abi_preflight_<timestamp> \
  --report-out reports/stage6/P4_NATIVE_ABI_PREFLIGHT_RESULT_20260604.md
```
