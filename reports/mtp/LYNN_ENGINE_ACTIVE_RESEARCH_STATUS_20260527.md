# Lynn Engine Active Research Status

Date: 2026-05-27 Asia/Shanghai

This note updates the 2026-05-20 product pivot. The pivot is still true:
Lynn client should use llama.cpp/GGUF as the default local inference backend
until Lynn engine clears the same-model same-hardware speed bar.

It does **not** mean Lynn engine is abandoned. The active engine track is now
narrower and more concrete:

```text
Qwen3.6-35B-A3B / Qwen3.5-9B
  + Lynn APEX-MTP draft sidecar
  + Nemotron-style verify -> accept -> crop/repair runtime
  + T=1-equivalent K=N verifier kernels
```

## Current TL;DR

| Item | Status |
|---|---|
| Nemotron 8B as a ship model | Rejected for Lynn product use; useful as algorithm reference only. |
| Nemotron algorithm control flow | Portable. The useful pieces are verify/accept/crop, full-accept fast commit, prefix repair, and K=N verifier structure. |
| Qwen35 APEX-MTP K=2 runtime | Token-exact on Spark W4A16 + official MTP sidecar with conservative row-wise bridge. |
| Throughput today | Not yet a win in Lynn Python runner. K2 verifier cost still dominates. |
| Main blocker | Batched full-attention verifier numerics: PyTorch batched attention and batched `o_proj` are not T=1-equivalent. |
| Active next work | Build T=1-equivalent dual-row attention and `o_proj` kernels, then re-run real 35B maintenance smoke. |
| Production guardrail | Brain V2 MIMO stays first; llama.cpp APEX-MTP service stays restored as second fallback after experiments. |

## What Nemotron Contributed

The Nemotron-Labs-Diffusion-8B work separated two things that are easy to
confuse:

1. The **model-specific draft source**: bidirectional diffusion/self-spec draft
   requires joint AR-diffusion training plus a linear-spec LoRA. This does not
   transfer directly to Qwen/Llama causal checkpoints.
2. The **runtime skeleton**: draft, verify, accept, crop/repair, and move the
   decode state forward. This is model-agnostic and directly useful for Lynn
   APEX-MTP.

The practical conclusion is:

```text
Do not copy Nemotron's bidirectional draft path into Qwen base weights.
Do copy the runtime structure into Lynn's existing APEX-MTP path.
```

Nemotron 8B itself also failed Lynn product gates in SGLang chat/eval mode:
tool calling did not produce usable `tool_calls`, MCQ eval emitted repeated
`</think>` spam, and the attractive raw TPS numbers were inflated by repetitive
generation. That does not invalidate the algorithm extraction; it confirms this
should be a Lynn-native APEX-MTP effort, not a Nemotron model switch.

## What Is Already In Lynn Engine

The active branch is:

```text
codex/qwen35-mtp-block-verify
```

Implemented or prototyped:

| Area | Result |
|---|---|
| K=N speculative skeleton | `decode_block_to_logits_and_hidden` and `speculative_step_kn_batched` can verify multi-token draft blocks. |
| Reject/crop semantics | Conservative restore-and-replay path protects token exactness for partial rejects. |
| Full-accept fast commit | K=2 fully accepted blocks can keep verifier state directly. |
| Prefix repair | K=2 one-token partial accept can repair the committed prefix with a smaller exact block. |
| Safe full-attention bridge | `LYNN_FULL_ATTN_K2_BACKEND=rowwise_gate_bridge` keeps K2 token-exact by row-wise QKV/RoPE, attention, and `o_proj`, while batching only the safe gate multiply. |
| Row-wise `o_proj` prototype | `triton_kernels/rowwise_linear.py` proves one Triton launch can match two T1 row-wise launches bit-for-bit under the same backend. |
| Row-wise prefix attention prototype | `triton_kernels/rowwise_attention.py` proves K2 prefix attention can keep independent T1-equivalent online-softmax accumulators in one launch. |
| Safe maintenance wrapper | `scripts/spark_run_qwen35_mtp_maintenance_smoke.sh` masks/stops/restores APEX service around heavy 35B experiments. |

Primary reports:

```text
reports/mtp/QWEN35_MTP_BLOCK_VERIFY_OVERNIGHT_20260526.md
reports/mtp/qwen35_mtp_safe_k2_bridge_20260526_230739.json
reports/mtp/qwen35_mtp_kn_full_accept_fast_20260526_233834.json
reports/mtp/qwen35_mtp_k2_prefix_repair_20260526_235627.json
reports/mtp/qwen35_k2_rowwise_linear_kernel_20260527_020517.json
reports/mtp/qwen35_k2_rowwise_attention_kernel_20260527_023349.json
reports/mtp/qwen35_k2_rowwise_attention_kernel_stride_20260527_023632.json
reports/mtp/qwen35_k2_rowwise_attention_kernel_dynamicn_20260527_025345.json
reports/mtp/qwen35_mtp_k2_rowwise_gate_real2_20260527_021806.json
reports/mtp/qwen35_mtp_k2_rowwise_attention_kernel_warm_20260527_024429.json
reports/mtp/qwen35_mtp_k2_rowwise_attention_dynamicn_warm_20260527_025502.json
```

## Spark Evidence So Far

| Experiment | Exactness | Key Number | Meaning |
|---|---:|---:|---|
| Strict K2 bridge | 6/6 | 16.27 TPS | Correct but too much replay/row-wise cost. |
| K2 full-accept fast commit | 6/6 | 24.36 TPS | +49.5% over strict K2; direct state commit works for K=2. |
| K2 prefix repair | 6/6 | 24.79 TPS | Partial-accept repair is exact; speed gain is small but direction is right. |
| Manual GQA fast path | 5/6 | 25.90 TPS | Approximate fast path breaks parity and is still below baseline. |
| Row-wise `o_proj` kernel probe | 8/8 | 0.18 ms vs 0.75 ms | One dual-row Triton launch can replace two T1 launches under the same backend. |
| Row-wise prefix-attention kernel probe | 8/8 | 0.077 ms vs 0.123 ms | K2 prefix attention can be exact and about 1.6x faster than two T1 launches in the micro fixture. |
| Dynamic-N prefix-attention probe | 8/8 | 0.115 ms vs 0.162 ms | Runtime `N` removes per-seq-len specialization risk, at the cost of a slower micro-kernel. |
| Real 35B rowwise-gate smoke | 6/6 | K2 21.28 TPS | Correct end-to-end, but verifier attention remains too expensive. |
| Real 35B rowwise-attention smoke | 6/6 | K2 22.99 TPS, 0.83x warmed baseline | Attention kernel improves the exact K2 bridge by about 8%, but still does not beat the warmed T1 baseline. |
| Real 35B dynamic-N smoke | 6/6 | K2 23.00 TPS, 0.62x warmed baseline | Current code path is exact and avoids baseline length-JIT artifacts, but K2 remains below baseline. |

Current Python runner baseline on this short smoke is about 31-34 TPS, while the
active llama.cpp APEX-MTP fallback has been observed around 66.7 tok/s with
draft acceptance around 0.47. So the Lynn Python K2 path is a correctness and
research result today, not a production speed win yet.

The 2026-05-27 warmed rowwise-attention smoke makes this sharper. Under the
same experimental rowwise attention/`o_proj` contract, K2 stayed 6/6 exact and
rose from 21.28 TPS to 22.99 TPS, but the warmed baseline was 27.82 TPS. The
kernel direction is useful; it has not yet crossed the production speed bar.

The current committed kernel uses runtime `N` rather than specializing on
sequence length. This removes a misleading per-prompt JIT artifact and gives a
cleaner comparison: warmed baseline 37.03 TPS, K2 23.00 TPS, both 6/6 exact.
So the active branch is now correctness-clean and less benchmark-fragile, but
the K2 verifier still costs too much.

## Why It Still Matters

The runtime result changes the problem from "can this algorithm work on Qwen?"
to a sharper engineering target:

```text
Make the K2/K4 verifier as cheap as the accept rate deserves.
```

That is exactly the kind of engine problem Lynn can own. The APEX sidecar
already gives a real trained draft source. The extracted Nemotron runtime tells
us how to spend accepted draft tokens safely. The remaining gap is kernel and
service-loop cost, not conceptual feasibility.

## Current Blocker

Fast K2 parity was split into components:

| Component | Finding |
|---|---|
| Batched gate multiply | Safe. |
| Batched `o_proj` matmul | Not T=1-equivalent enough for deterministic verifier parity. |
| Batched attention / SDPA | Not T=1-equivalent, and in the tested GQA prefix shape was not faster either. |
| Row-wise bridge | Exact, but too slow. |

Therefore the next kernels are not "generic faster matmul" work. They need to
preserve the T=1 accumulation contract while sharing launch/weight-load cost
across two rows:

1. Dual-row prefix attention with independent online-softmax accumulators.
2. Dual-row `o_proj` with independent row accumulators.
3. Both enabled for baseline T1 and K2 verifier when checking token exactness.

## Training Direction

For Qwen-family diffusion/self-spec training, the conclusion is separate:

| Route | Recommendation |
|---|---|
| Qwen3.5 9B continued AR-diffusion pretrain | Technically correct, but requires real A100/H100 budget. |
| Qwen3.5 4B LoRA-only diffusion PoC | Good cheap gate to estimate accept ceiling before renting larger hardware. |
| Qwen3.6-35B-A3B MoE diffusion training | Park for now; MoE plus bidirectional diffusion has too many first-dollar unknowns. |
| Lynn APEX-MTP K expansion | Mainline now; it uses an already-trained draft source and the new runtime skeleton. |

If the runtime kernels become cheap and K=2/K=4 accept is still insufficient,
then the right next spend is draft-side training, not another verifier rewrite.

## Next 72 Hours

1. Profile the remaining exact verifier cost after the rowwise-attention smoke:
   QKV/RoPE row-wise projection, rowwise `o_proj`, MTP sidecar forward, and
   Python service loop.
2. Try the next cheap exact bridge: fuse or batch the safe parts around rowwise
   `o_proj` without changing the T1 accumulation contract.
3. If K2 still loses to baseline, profile exact verifier cost by component and
   decide whether to move the path into the production llama.cpp APEX service
   loop or keep optimizing Lynn Python runner first.
4. Keep production restored after every smoke:

```bash
systemctl is-active lynn-apex-mtp-llamacpp.service
```

## Product Framing

Lynn engine remains an R&D engine, but it is not idle. The current active
framing is:

```text
llama.cpp ships product fallback today.
Lynn engine develops the next self-spec / APEX-MTP acceleration path.
```

That is a narrower promise than the pre-5/20 engine plan, and a stronger one.
It keeps the product safe while preserving the path toward a Lynn-owned speed
advantage on Qwen3.6-35B-A3B and Qwen3.5-9B.
