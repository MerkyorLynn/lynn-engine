# Stage 6 P4B - native fused single-kernel contract

Date: 2026-06-04

Verdict: **opt-in single-CTA numeric reference banked; fused-kernel speed and default promotion remain closed.**

P4A reserved and tested the real resident-runner bridge into a native
two-stage, caller-owned scratch/output reference. P4B is the next boundary: the
true fused active-MoE kernel that must not materialize the intermediate
`[top_k, 512]` tensor as a separate stage.

## Backend

Opt-in backend name:

```bash
LYNN_NATIVE_ACTIVE_MOE_BACKEND=fused_zero_shadow_single_kernel_contract
```

By default, this backend is still intentionally fail-loud. It freezes the ABI
and prevents P4A's two-stage reference from being mistaken for the final fused
kernel.

There is now one explicit correctness-only escape hatch:

```bash
LYNN_NATIVE_P4B_SINGLE_CTA_REFERENCE=1
```

With that variable set, the native symbol runs a first output-returning
single-CTA CUDA reference for `T=1`, `top_k=8`, `tile_tokens=1`,
`tile_experts=1`, `tile_hidden=8`. This is a proof that the out-only P4B ABI can
produce the same BF16 result as the P4A two-stage reference without exposing
`inter_scratch` to the caller. It is **not** a speed gate and must not be used
for default serving.

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

The native function validates shape/layout. With
`LYNN_NATIVE_P4B_SINGLE_CTA_REFERENCE=1`, it launches
`p4b_single_cta_reference_kernel<<<1, 256>>>` and writes caller-owned BF16
`out[T, 2048]` directly.

Without that opt-in variable, it still throws:

```text
P4B single-kernel fused zero-shadow contract is not implemented yet; do not
bank fused-kernel speed or promote this backend
```

That remains the intended default behavior until a real multi-CTA CUDA/CUTLASS
implementation passes byte-count, numeric, speed, and RC gates.

## GPU-Free Gate

```bash
python3 scripts/test_stage6_p4b_single_kernel_static.py
python3 scripts/test_stage6_p4b_single_kernel_evidence_tools.py
python3 scripts/test_stage6_p4b_runtime_bridge_tools.py
python3 scripts/test_stage6_p4b_single_cta_numeric_tools.py
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
- fail-loud message is present;
- single-CTA reference opt-in, Spark wrapper, and numeric summarizer are present.

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
output is a failure for the default fail-loud gate. Output-returning behavior is
tested only by the explicit single-CTA numeric gate below.

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

## Single-CTA Numeric Reference Gate

```bash
scripts/run_spark_stage6_p4b_single_cta_numeric_preflight.sh \
  --host dgx-spark \
  --image lynn-eval-base:cu13 \
  --expect-head "$(git rev-parse HEAD)"
```

Expected decision:

```text
PASS_P4B_SINGLE_CTA_NUMERIC_REFERENCE
```

This gate opts into `LYNN_NATIVE_P4B_SINGLE_CTA_REFERENCE=1`, runs P4A
`active_moe_fused_zero_shadow_out_contract` and P4B
`active_moe_fused_zero_shadow_single_kernel_contract` on the same synthetic
packed-NVFP4 tensors, then checks:

- candidate BF16 output is finite and returned through caller-owned `out`;
- `rel_l2` and `max_abs` versus P4A are within threshold;
- candidate ABI names do not include `inter_scratch`;
- packed weight byte budget is below the BF16 expert-shadow equivalent;
- `banked_single_cta_numeric_preflight=true`;
- `banked_fused_kernel=false` and `banked_default_promotion=false`.

This banks correctness of the first output-returning native P4B path only. It
does not bank latency, bandwidth, e2e TPS, or server/default promotion.

## Spark Results

Latest banked artifacts:

| Gate | Artifact | Decision | Boundary |
|---|---|---|---|
| Synthetic single-kernel ABI | `reports/stage6/p4b_single_kernel_preflight_20260604_083106/` | `PASS_SINGLE_KERNEL_FAILLOUD_CONTRACT` | Banks default fail-loud contract only; fused speed remains unbanked |
| Real runtime bridge | `reports/stage6/p4b_runtime_bridge_preflight_20260604_083152/` | `PASS_P4B_RUNTIME_BRIDGE_FAILLOUD` | Banks resident-runner route integrity only; fused speed remains unbanked |
| Single-CTA numeric reference | `reports/stage6/p4b_single_cta_numeric_preflight_20260604_084451/` | `PASS_P4B_SINGLE_CTA_NUMERIC_REFERENCE` | Banks opt-in correctness reference only; speed/default remain unbanked |

The runtime artifact proves P4B route selection on the real model at layer 0:
native call delta `1`, last shapes `hidden=[1,2048]`,
`expert_ids=[1,8]`, `out=[1,2048]`, no `inter_scratch`, active BF16 expert
shadows removed, and packed tensors present. The expected fail-loud error is
preserved:

```text
P4B single-kernel fused zero-shadow contract is not implemented yet; do not
bank fused-kernel speed or promote this backend
```

The single-CTA numeric artifact proves the opt-in output-returning path on
Spark GB10 (`sm_121`, torch `2.9.1+cu130`, CUDA `13.0`): P4B candidate versus
P4A reference `rel_l2=0.0`, `max_abs=0.0`, `candidate_out=[1,2048]` BF16,
`no_inter_scratch_candidate_abi=true`, packed/BF16 byte ratio
`0.3750001589457194`, elapsed `25.818s`.

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
