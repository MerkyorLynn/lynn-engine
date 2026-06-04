# Stage 6 P4B - native fused single-kernel contract

Date: 2026-06-04

Verdict: **CONTRACT ONLY; not implemented yet; no fused kernel is banked.**

P4A reserved and tested the real resident-runner bridge into a native
two-stage, caller-owned scratch/output reference. P4B is the next boundary: the
true fused active-MoE kernel that must not materialize the intermediate
`[top_k, 512]` tensor as a separate stage.

## Backend

New opt-in backend name:

```bash
LYNN_NATIVE_ACTIVE_MOE_BACKEND=fused_zero_shadow_single_kernel_contract
```

The backend is intentionally fail-loud today. It is present to freeze the ABI
and prevent P4A's two-stage reference from being mistaken for the final fused
kernel.

The Python wrapper records runtime evidence before invoking the native symbol:

```text
_p4b_fused_zero_shadow_single_kernel_contract_call_count
_p4b_fused_zero_shadow_single_kernel_contract_last_shapes
```

These counters are allowed to increment even while the native symbol fails
loud. They are for future runtime-bridge evidence only; they do not bank fused
speed.

## ABI

Native symbol:

```text
active_moe_fused_zero_shadow_single_kernel_contract(
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
  tile_experts,
  tile_hidden,
) -> void
```

Required properties:

- no BF16 expert weight tensors in the ABI;
- no `inter_scratch` argument and no P4A two-stage reference call;
- caller-owned output only;
- No output scratch allocation. P4B must not allocate `torch::empty`,
  `torch::zeros`, `TensorOptions`, or any hidden float accumulator tensor inside
  the native entry point;
- must not reuse the historical `active_moe_fused_atomic_scalar_kernel` path:
  that path is single-launch and packed, but it accumulates into a float output
  buffer and therefore is not the P4B zero-scratch/caller-owned BF16 ABI;
- no `atomicAdd(out, ...)` shortcut against the caller-owned BF16 output; the
  first true reference should compute each output element/tile inside its owning
  block and write BF16 once;
- no Python/Triton fallback inside the native symbol;
- dispatch must not fall through the generic Triton backend set;
- `banked_fused_kernel=false` until a real implementation passes byte-count,
  numeric, speed, and RC quality gates.

## Current State

The native function validates shape/layout and then throws:

```text
P4B single-kernel fused zero-shadow contract is not implemented yet; do not
bank fused-kernel speed or promote this backend
```

That is the intended behavior until the fused CUDA/CUTLASS implementation
exists.

## GPU-Free Gate

```bash
python3 scripts/test_stage6_p4b_single_kernel_static.py
python3 scripts/test_stage6_p4b_single_kernel_evidence_tools.py
python3 scripts/test_stage6_p4b_runtime_bridge_tools.py
```

The static gate checks:

- C++ symbol and pybind are present;
- Python backend is present and opt-in only;
- generic active-MoE fallback set does not include
  `fused_zero_shadow_single_kernel_contract`;
- C++ P4B function body does not mention `inter_scratch`;
- C++ P4B function body does not allocate hidden scratch or reuse the old
  float-output atomic scalar kernel;
- C++ P4B function body does not call the two-stage
  `lynn_native_active_moe_grouped_per16_nonatomic_out_reference`;
- fail-loud message is present.

## Spark Preflight

```bash
scripts/run_spark_stage6_p4b_single_kernel_preflight.sh \
  --host dgx-spark \
  --image lynn-eval-base:cu13 \
  --expect-head "$(git rev-parse HEAD)"
```

Expected decision while P4B is still unimplemented:

```text
PASS_SINGLE_KERNEL_FAILLOUD_CONTRACT
```

This banks only `banked_single_kernel_contract_preflight=true`. It must keep
`banked_fused_kernel=false` and `banked_default_promotion=false`. A returned
output is a failure until the real fused CUDA/CUTLASS implementation is present
and a new byte-count/numeric/speed/RC gate replaces this fail-loud contract
gate.

## Runtime Bridge Fail-Loud Gate

```bash
scripts/run_spark_stage6_p4b_runtime_bridge_preflight.sh \
  --host dgx-spark \
  --image lynn-eval-base:cu13 \
  --expect-head "$(git rev-parse HEAD)"
```

Expected decision while P4B is still unimplemented:

```text
PASS_P4B_RUNTIME_BRIDGE_FAILLOUD
```

This gate uses a real resident runner path, first obtains a Triton baseline,
then deletes active BF16 expert shadows and switches to
`LYNN_NATIVE_ACTIVE_MOE_BACKEND=fused_zero_shadow_single_kernel_contract`.
It banks only `banked_p4b_runtime_bridge_preflight=true` when the P4B call
counter advances exactly once, the last-shape evidence is out-only
(`hidden`, `expert_ids`, `out`, no `inter_scratch`), and the native symbol
throws the expected fail-loud not-implemented error.

It must keep `banked_fused_kernel=false` and `banked_default_promotion=false`.
When the real fused implementation lands, this fail-loud gate must be replaced
by an output-returning numeric/speed/RC gate.

## Spark Results

Latest banked fail-loud artifacts:

| Gate | Artifact | Decision | Boundary |
|---|---|---|---|
| Synthetic single-kernel ABI | `reports/stage6/p4b_single_kernel_preflight_20260604_081151/` | `PASS_SINGLE_KERNEL_FAILLOUD_CONTRACT` | Banks contract preflight only; fused kernel remains unbanked |
| Real runtime bridge | `reports/stage6/p4b_runtime_bridge_preflight_20260604_081234/` | `PASS_P4B_RUNTIME_BRIDGE_FAILLOUD` | Banks resident-runner route integrity only; fused kernel remains unbanked |

The runtime artifact proves P4B route selection on the real model at layer 0:
native call delta `1`, last shapes `hidden=[1,2048]`,
`expert_ids=[1,8]`, `out=[1,2048]`, no `inter_scratch`, active BF16 expert
shadows removed, and packed tensors present. The expected fail-loud error is
preserved:

```text
P4B single-kernel fused zero-shadow contract is not implemented yet; do not
bank fused-kernel speed or promote this backend
```

## Promotion Gate

P4B may become banked only when all of these are true:

| Evidence | Required |
|---|---|
| Byte-count | packed NVFP4 bytes + scales only; no BF16 expert shadow |
| Numeric | layer-level cosine/rel-L2/argmax vs P4A/P3 reference |
| Speed | layer microbench or e2e TPS improvement with launch/latency accounting |
| Runtime | resident-runner bridge reaches the P4B symbol, not P4A/Triton |
| Quality | RC smoke before any server/default promotion |

Until then, P4B is a contract and `banked_fused_kernel=false`.
