# MTP K=1 Sequential Speculative Smoke — Spark sm_121 (2026-05-19)

## TL;DR

* **Correctness gate**: ✅ PASS — speculative output token-exact matches baseline across all 16 prompts.
* **Accept rate**: **0.30%** (catastrophically low; expected 50-70% per memory).
* **TPS ratio**: spec_k1 / baseline = **0.465** (12.08 vs 25.96 TPS) — speculative is 2.15× slower.
* **Root cause**: MTP head trained at OFFSET=1 (predicts same position as base), but speculative serving requires OFFSET=2 (lookahead). My K=1 wire compared MTP's pos-N+1 draft to base's pos-N+2 argmax — different positions → ~1/vocab random accept.
* **Secondary finding**: baseline TPS 25.96 ≠ memory's 38.96 — missing Config D env vars.

## Run details

| Field | Value |
|---|---|
| Container | `lynn-mtp-overnight` (lmsysorg/sglang:dev-cu13) |
| Started | 2026-05-19 10:30:11 UTC |
| Exited | 2026-05-19 10:44:17 UTC (~14 min wall) |
| Status | `Exited (0)` |
| Model | `/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000` |
| Sidecar | `/models/mtp_sidecars/qwen36-35b-a3b-mtp/mtp.safetensors` |
| Prompts | 16 (en/zh × short/mid/long, JSON, tool-call, code, creative) |
| max_new | 256 |
| Report | `/home/merkyor/reports/mtp_overnight/20260519_103011/mtp_smoke.json` |

## Numbers

| Config | exact_match | decode_tps | spec_eff_tps | spec_accept | shadow_accept |
|---|---|---|---|---|---|
| baseline | 1.0 | **25.96** | — | — | — |
| shadow | 1.0 | 25.95 | — | — | None |
| spec_k1 | **1.0** ✅ | — | **12.08** | **0.30%** ❌ | — |

## Root cause: MTP head OFFSET semantics

`scripts/a100_mtp_fc_calibration_train.py` ``_collect_cases`` (line 84-92):

```python
base_hidden, input_embed, ids, last_token_id, last_pos = _base_prefill_last_hidden(
    runner, str(spec["prompt"]), use_chat_template=use_chat_template,
)
base_logits = runner._lm_head_logits(
    _rms_norm(base_hidden, runner.outside["model.language_model.norm.weight"])
)
label_id = int(base_logits[0].argmax().item())  # base's argmax for position last_pos + 1
```

The supervision target is **base's argmax for position last_pos + 1** — i.e. OFFSET=1 from `base_hidden`. The head learns to mimic base's prediction at the SAME next position, not look ahead.

For speculative serving this gives zero lookahead — both base and MTP predict position N+1 from the same hidden state. You cannot get more than 1 token per round.

My ``speculative_step_k1`` in ``engine/mtp_serving.py`` assumed OFFSET=2: it took MTP's draft (actually for position N+1) and compared it against ``argmax_after_pending`` (base's prediction for position N+2 after decoding pending into the state). Two different positions → comparison is uncorrelated → accept rate ≈ 1/vocab = 1/248320 ≈ 0% in practice (0.30% measured matches this).

The memory entry ``project_mtp_scaffolding_ready_20260516.md`` says ``推理 offset=2, 期望 accept 60-70%`` — that was the **contract goal**, not the implemented training. Retraining is needed to fulfill the contract.

## Baseline TPS regression (25.96 vs memory's 38.96)

``scripts/spark_mtp_speculative_smoke.py`` BASE_ENV includes:

* LYNN_MOE_IMPL=packed_nvfp4
* LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1
* LYNN_NATIVE_FP4_LM_HEAD=1
* LYNN_PACKED_DECODE_BACKEND=native_fast_2d

memory ``project_overnight_results_20260517.md`` Config D (which achieved 42.55 / 45.39 TPS at 128/256 tok) **also** requires:

* LYNN_PACKED_DECODE=1
* LYNN_AUTOTUNE=1
* LYNN_LINEAR_ATTN_NATIVE_FP4=1
* LYNN_PACKED_SHARED_EXPERT=1
* (probably) LYNN_FULL_TOKEN_GRAPH_SLOT=1 or LYNN_LINEAR_BLOCK_GRAPH=1 for the graph slot fast-path

Without these the baseline runs in eager scalar-bridge-ish path at ~26 TPS. Independent of MTP — this is a smoke-runner env hygiene fix.

## Two paths forward

### Path A — Retrain MTP head at OFFSET=2 (A100 专项, 2-3 day)

Supervision change in ``a100_mtp_fc_calibration_train.py`` (and iterative variants):

```python
# OLD (offset=1):
label_id = argmax(lm_head(rms_norm(base_hidden)))

# NEW (offset=2):
# For each prompt: prefill up to position N, decode one more step to get
# hidden_{N+1} (i.e. base's true state after committing argmax_N), then
# label_id = argmax(lm_head(rms_norm(hidden_{N+1})))
# Input to MTP: (base_hidden_N, embedding(argmax_N), pos=N)  ← teacher-forced
```

After retrain, ``speculative_step_k1`` as currently wired in ``engine/mtp_serving.py`` should work without further changes. Acceptance gate: heldout accept_rate ≥ 50% should drive Spark spec_k1 effective TPS to 1.3-1.5× baseline (memory's "推理 offset=2 期望 accept 60-70%" projection).

### Path B — Fix baseline env first, treat MTP wire as parked

Add Config D env to ``spark_mtp_speculative_smoke.py`` BASE_ENV. Re-run baseline to confirm 38-42 TPS. K=1 spec K wire stays in tree as correctness-validated infrastructure, awaiting OFFSET=2 sidecar.

This is the **cheap** path: confirms there's no regression in production decode path, postpones MTP serving until A100 produces an OFFSET=2 sidecar.

## Recommendation

* **B first** (15 min) — re-run smoke with full Config D env, confirm baseline ≥ 38 TPS. If yes, current production path is healthy.
* **A** is the unlock for ≥ 1.3× speedup — but is an A100 retraining job, not Spark serving work.
* Do NOT pursue M5 (batched K=1 verify) — even with batching, an OFFSET=1 head cannot drive any speculative speedup. M5 is contingent on Path A.

## Spark cleanup

* Container `lynn-mtp-overnight` still on host (Exited 0). Remove with ``docker rm lynn-mtp-overnight`` when no longer needed for debug.
* No memory leaked — mem fully released.
* Production PoC container ``lynn-engine-35b-w4a8-native`` was stopped before the run; user may want to restart it if it was serving anything (it was the W4A8 PoC per ``reference_spark_fp8_w4a8_design_strategy_20260519.md``).
