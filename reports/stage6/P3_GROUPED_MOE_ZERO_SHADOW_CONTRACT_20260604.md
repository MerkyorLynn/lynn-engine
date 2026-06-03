# Stage 6 Phase 3 — grouped MoE zero-shadow contract

Date: 2026-06-04

Verdict: **CONTRACT ONLY; no P3 kernel is banked yet.**

P2-N proved selected-layer coverage for `p2e_hybrid` active-MoE packed prefill
plus block linear-attention. P2-O is the next resident-runner real-prompt gate.
P3 starts only after P2-O is understood. Its job is not another no-shadow proof:
P2 already proved active expert BF16 shadows can be removed. P3's job is to make
the active expert path production-shaped: fewer temporary dequant buffers, fewer
per-expert launches, and a clean grouped active-expert kernel contract.

## Existing Inputs

The current packed active-MoE layout is already stable:

| Tensor | Contract |
|---|---|
| `mlp.experts._gate_up_packed` | grouped NVFP4, expert-major, `[E, 2I, H/2]` packed E2M1 |
| `mlp.experts._gate_up_scale` | grouped scale, `[E, 2I, H/16]` |
| `mlp.experts._gate_up_global_scale` | scalar/global scale |
| `mlp.experts._down_packed` | grouped NVFP4, expert-major, `[E, H, I/2]` packed E2M1 |
| `mlp.experts._down_scale` | grouped scale, `[E, H, I/16]` |
| `mlp.experts._down_global_scale` | scalar/global scale |
| Router output | `expert_ids [T, top_k]`, `routing_weights [T, top_k]` |

For Qwen3.6-35B-A3B W4A16 in this repo, the active MoE dimensions used by the
existing kernels are `H=2048`, `I=512`, `E=256`, `top_k=8`.

## Existing References

P3 must compare against these already-banked references:

- BF16 active expert shadow reference: `_moe_forward`.
- Packed no-resident-shadow reference: `p2e_hybrid`.
- Current small-M grouped oracle: `moe_forward_verify_smallm_nvfp4`.
- Current prefill gate/up slice: `nvfp4_prefill_gate_up_silu_one_expert`.
- Current down weighted-sum slice: `nvfp4_grouped_down_weighted_sum`.

## P3-A Kernel Contract

The first P3 kernel must consume packed NVFP4 active expert weights directly and
must not require `mlp.experts.gate_up_proj` or `mlp.experts.down_proj`.

Minimum callable shape:

```text
active_moe_grouped_prefill(
  hidden[T, H] bf16,
  expert_ids[T, top_k] int32,
  routing_weights[T, top_k] fp32/bf16,
  gate_up_packed, gate_up_scale, gate_up_global_scale,
  down_packed, down_scale, down_global_scale,
) -> out[T, H] bf16
```

Allowed temporary storage:

- bounded per-token/per-tile scratch;
- bounded `inter[T, top_k, I]` during early P3-A if reported honestly.

Forbidden for a P3 bank:

- rebuilding full-layer BF16 `gate_up` or `down` shadows;
- silently calling `reload_decode_bf16_shadows()`;
- using `mlp.experts.gate_up_proj` or `mlp.experts.down_proj`;
- claiming full RC quality from a layer microbench.

## Gate Ladder

| Gate | Scope | Required evidence |
|---|---|---|
| P3-A | One layer, synthetic hidden, active MoE only | numeric vs BF16 and P2E, no BF16 active shadow, memory peak reported, speed vs P2E reported |
| P3-B | Multi-layer selected prefill | residual-stack numeric, no active shadow, speed not regressed vs P2N |
| P3-C | Resident-runner real prompt | generated-token smoke, no reload, memory release evidence, P2-O-style artifact |
| P3-D | Server/promotion candidate | RC quality battery, rollback flag, README/release matrix update |

P3-A PASS requires all of:

- cosine >= 0.999 against BF16 or current P2E reference;
- argmax match for layer output smoke;
- active BF16 expert shadow absent before the packed path runs;
- memory peak and temporary bytes reported;
- no hidden reload/rebuild of active shadows;
- speed is reported with honest caveat. Speed regression can still be a contract
  pass, but it cannot be promoted beyond P3-A.

## Relationship To P2-O

P2-O remains the next resident-runner gate. P3 work can start as kernel-contract
work, but no P3 result should be used to update README status until P2-O has a
real Spark artifact or the report explicitly states that P2-O is still pending.

## Local Static Check

GPU-free contract sanity:

```bash
python3 scripts/test_stage6_p3_contract_static.py
```

This only verifies that the repo still exposes the APIs and evidence artifacts
this contract depends on. It is not a kernel or speed test.

## P3-A Runnable Probe

The first runnable P3-A artifact is a contract-shaped probe, not a banked fused
kernel:

```bash
scripts/run_spark_stage6_p3a_contract_probe.sh --layer 0 --batches 1,16,64
```

It computes active-MoE router outputs once, builds a BF16 active expert
reference, deletes `mlp.experts.gate_up_proj` and `mlp.experts.down_proj`, then
runs `active_moe_grouped_prefill_p3a(...)` from packed tensors only. A valid
P3-A probe artifact must report:

- `banked_fused_kernel=false`;
- `passes.shadow_absent_at_candidate_start=true`;
- cosine / argmax against BF16 active reference;
- packed bytes, released BF16 bytes, inter scratch estimate, memory peak;
- speed versus BF16 active MoE, with no promotion claim.

The wrapper records remote HEAD or provenance-manifest match, Docker exit code,
`nvidia-smi` before/after, `run.log`, `result.json`, and `summary.md` under
`reports/stage6/p3a_layer*_grouped_moe_contract_probe_*`. The summary helper is:

```bash
python3 scripts/summarize_stage6_p3a_contract_probe.py \
  reports/stage6/p3a_layer*/result.json \
  --markdown-out /tmp/p3a_summary.md \
  --strict-exit
```

`--strict-exit` fails unless `banked_fused_kernel=false`, numeric passes, and the
active BF16 shadow is absent at candidate start.

Formal report writer:

```bash
python3 scripts/write_stage6_p3a_report.py \
  reports/stage6/p3a_layer*_grouped_moe_contract_probe_* \
  --report-out reports/stage6/P3A_GROUPED_MOE_CONTRACT_PROBE_20260604.md
```

The report writer may bank only the P3-A contract probe. It must not promote P3
or claim a fused grouped-MoE kernel.
