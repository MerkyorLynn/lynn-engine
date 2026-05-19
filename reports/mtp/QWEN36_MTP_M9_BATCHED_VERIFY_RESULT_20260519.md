# Qwen3.6-35B MTP M9 Batched Verify Result

**Date:** 2026-05-19 (smoke completed 23:52)
**Source JSON:** `/tmp/mtp_smoke_concat_m9_20260519_234434.json` (Spark, 93 KiB)
**PID 3237857:** completed (no longer running)
**Model:** `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000`
**Sidecar:** `/home/merkyor/models/mtp_sidecars/qwen36-35b-a3b-mtp-official-lynn-fused/mtp.safetensors`
**Schema:** `lynn-mtp-speculative-smoke-v2`
**Config:** max_new=128, n_prompts=6

---

## Summary Numbers (exact, from JSON)

| Mode | exact_match_rate | mean_decode_tps | mean_spec_effective_tps | mean_spec_accept_rate | mean_shadow_accept_rate |
|------|------------------|-----------------|-------------------------|-----------------------|--------------------------|
| baseline | 1.000 (6/6) | **37.996** | — | — | — |
| shadow | 0.000 (0/6) | 33.910 | — | — | 0.00911 (0.91%) |
| spec_k1 (sequential) | **0.000 (0/6)** | — | **24.801** | **0.7688 (76.88%)** | — |
| spec_k1_batched (M9) | **0.000 (0/6)** | — | **12.558** | **0.1539 (15.39%)** | — |

## Gates (from JSON)

| Gate | Pass | Value |
|------|------|-------|
| `correctness_spec_k1_matches_baseline` | **FALSE** | spec_k1 final tokens diverge from baseline |
| `correctness_spec_k1_batched_matches_baseline` | **FALSE** | batched final tokens diverge from baseline |
| `tps_ratio_spec_over_baseline` | — | 0.6527 (24.8 / 38.0) |
| `tps_ratio_spec_batched_over_baseline` | — | 0.3305 (12.6 / 38.0) |

## Verdict: **CLOSED — multiple independent failures**

`spec_k1_batched accept = 15.4%` is below the 20% threshold and below the
60% promote bar. Worse, **`spec_k1` (sequential, the previously-77% path)
also failed correctness**: prefix_match_len=1, exact_match=False on all 6 prompts.

This is a regression vs the prior 77.22% accept run referenced in commit
`acd9fb5`. Something between then and this M9 run changed the acceptance
accounting OR the verification pipeline.

---

## Per-Prompt Smoking Gun

Looking at `configs[*].rows[0]` (first prompt — "Explain Q4_K_M vs NVFP4"):

```
baseline new_ids head: [271, 238434, 183833, 220, 44459, 163050, 7411, 129063]
baseline completion:  "Telauvez samteni compar那双时至今サリー最新更新..."
                                   ^^^ baseline is producing GARBAGE

spec_k1 new_ids head: [271, 248068, 198, 8160, 579, 264, 7047, 1817]
spec_k1 completion:   "<think>\nHere's a thinking process: ..."
                                   ^^^ spec_k1 produces COHERENT text

spec_k1_batched ids:  [271, 48, 17, 9802, 1203, 369, 9238, 220]
spec_k1_batched text: "Q2_K_M is mixed 6-bit -bit -bit4 -bit4..."
                                   ^^^ batched produces DIFFERENT garbage
```

All three modes start with token id `271` (newline), then diverge. Their
"new_ids" are completely different streams.

**The baseline itself is incoherent on this prompt.** baseline `exact_match=True`
in the schema means baseline matches its own greedy reference (deterministic),
not that the output is meaningful. The model on Spark appears to be the
`from-r6000` checkpoint and may have a layer-mapping or RoPE issue independent
of MTP.

## Root Cause Analysis

### Layer 1: Baseline output is incoherent

The baseline path (no speculative, no MTP) produces text like
`"Telauvez samteni compar那双时至今サリー最新更新一直公司成员 online"` for a clean
English prompt. This is not a quantization artifact at the level we'd see from
W4A16 NVFP4 — it looks more like:
- Wrong RoPE/position offset, OR
- Layer ordering bug in the from-r6000 conversion, OR
- The wrong model being loaded (sidecar vs base mismatch?)

### Layer 2: spec_k1 produces different (more coherent) text

Sequential spec_k1 yields `"<think>\nHere's a thinking process..."` which IS
coherent — the model can think. But its tokens don't match baseline, hence
`exact_match=False`. With accept_rate=76.88%, the MTP head is correlating with
something — probably its OWN internal verification path, which is using a
different code path from baseline.

This means `spec_k1` and `baseline` are running through different forward paths,
and the previously-claimed "77.22% accept proves official head works" was
**measured against the spec path's own verifier, not against baseline**.

### Layer 3: spec_k1_batched is yet a third path

`spec_k1_batched` produces *different* garbage from baseline AND from spec_k1.
Even with the M9 fix (3386937: K=2 MoE per-token T=1 loop), the K=2 attention
or norm path is different enough to produce a third distinct output stream.

## Deployed Spark Code Inspection

Per task spec, since `spec_k1_batched accept < 20%`, I inspected the deployed code:

### `/home/merkyor/lynn-engine/engine/full_forward.py` (lines 380-410)

```python
# MoE for K=2: the packed_nvfp4 backend (Spark Config D default) is
# T=1-only — its fused Triton kernel hard-codes h.shape[1] == 1.
# ... per-position T=1 MoE for backend consistency ...
base_moe_fn = moe_fn if moe_fn is not None else _resolve_decode_moe_impl(
    os.environ.get("LYNN_MOE_IMPL", "optimized")
)
if h_norm.shape[1] == 1:
    moe_out = base_moe_fn(h_norm, w, cfg)
else:
    # Per-position T=1 MoE for backend consistency.
    moe_per_token = [
        base_moe_fn(h_norm[:, t : t + 1, :].contiguous(), w, cfg)
        for t in range(h_norm.shape[1])
    ]
    moe_out = torch.cat(moe_per_token, dim=1)
```

**The M9 fix is correctly deployed.** The K=2 path uses `base_moe_fn` (== the
configured runner backend) and loops T=1 per position. This is consistent with
T=1 baseline — for the MoE block.

But this only addresses MoE. The K=2 path also runs:
- attention with q/k/v projections
- norms
- residuals

Any of these could still differ between K=2 and T=1×2.

### `/home/merkyor/lynn-engine/engine/resident_runner.py`

The runner correctly passes `moe_fn=self.decode_moe_fn` at lines 887, 1039
(and at 929 falls back via `_resolve_decode_moe_impl(os.environ.get("LYNN_MOE_IMPL"))`
when fast_dispatch is off). Wiring is consistent.

## What Still Could Be Wrong

| Suspect | Evidence | How to check |
|---------|----------|--------------|
| RoPE position calculation in spec verify | All three modes diverge after token 0 (`271`) | Compare position_ids passed to attention in baseline vs spec_k1 forward |
| Linear-attn state update under K=2 | 30/40 layers are linear-attn; state mutation order matters | Check `linear_state_update` semantics for K=2 |
| Sidecar vs base model layer count | Sidecar is the fused MTP head, but baseline could be picking up layer_count from sidecar metadata | Diff the loaded `model.layers` count vs config |
| The model checkpoint itself | baseline output is incoherent for a coherent prompt | Run a Triton-baseline non-Lynn-Engine comparison |

## Threshold Check (per task)

| Threshold | Required | Actual | Pass |
|-----------|----------|--------|------|
| spec_k1_batched accept ≥ 20% (task soft floor) | ≥0.20 | 0.154 | NO |
| spec_k1_batched accept ≥ 60% (promote) | ≥0.60 | 0.154 | NO |
| spec_k1_batched correctness | TRUE | FALSE | NO |
| spec TPS > baseline | >37.996 | 12.558 / 24.801 | NO |

## Recommended Next Investigation (no kernel edits)

1. **Confirm baseline coherence**: run the same 6 prompts through llama.cpp on
   the same machine to verify the model checkpoint itself is fine.
2. **Diff position_ids**: log `pos_tensor` passed to attention in both T=1 and
   K=2 paths for a single token; they must be identical for the same generation
   step.
3. **Disable linear-attn graph reuse for K=2**: temporarily force the recurrent
   state path to recompute from scratch for K=2 to isolate state-update bugs.
4. **Run shadow-only against vLLM-reference**: get a known-good MTP accept
   number for this model+sidecar combo before chasing batched recovery.

## Files

| Path | Content |
|------|---------|
| `/tmp/mtp_smoke_concat_m9_20260519_234434.json` | Raw smoke result (Spark) |
| `/tmp/mtp_smoke_concat_m9_20260519_234434.log` | Run log (Spark, 2.8 KiB) |
| `/home/merkyor/lynn-engine/engine/full_forward.py:380-410` | M9 K=2 MoE per-token loop (correctly deployed) |
| `/home/merkyor/lynn-engine/engine/resident_runner.py:887,929,1039` | moe_fn wiring (correctly deployed) |
| This report | Engineering status |

## Bottom Line

The M9 K=2 MoE backend fix is correctly deployed but **insufficient**. The
batched path still diverges from baseline (accept 15.4%, correctness FAIL).
More importantly, **even the sequential spec_k1 path now fails correctness**
on this Spark checkpoint, which is a regression from the prior 77% claim.
The previously-cited "77.22% spec_k1 accept proves official head works" was
measuring spec accept against its own verifier, not against baseline tokens.

MTP is **not** ready to enter the resident serving path. Real next step is
checkpoint sanity, not more batched-verify tweaks.
