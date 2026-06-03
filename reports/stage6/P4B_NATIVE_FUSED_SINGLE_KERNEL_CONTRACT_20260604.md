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
```

The static gate checks:

- C++ symbol and pybind are present;
- Python backend is present and opt-in only;
- generic active-MoE fallback set does not include
  `fused_zero_shadow_single_kernel_contract`;
- C++ P4B function body does not mention `inter_scratch`;
- C++ P4B function body does not call the two-stage
  `lynn_native_active_moe_grouped_per16_nonatomic_out_reference`;
- fail-loud message is present.

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
