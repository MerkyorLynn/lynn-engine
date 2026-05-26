# Qwen35 MTP Block Verify Overnight Status

Date: 2026-05-26 00:45 Asia/Shanghai

Morning update: 2026-05-26 09:35 Asia/Shanghai

Post-reboot smoke update: 2026-05-26 10:15 Asia/Shanghai

## Target

Make the Nemotron-style verify/accept/crop runtime useful for Lynn's own Qwen
line, with the same implementation surface available to:

- Qwen3.6-35B-A3B with the existing official MTP sidecar.
- Qwen3.5-9B once a compatible draft head or trained sidecar exists.

The key split is deliberate: runtime ABI first, draft-head quality second. If
K>1 accept quality is the blocker, it becomes a training task, not a runtime
rewrite.

Recommended product decision after the parallel Nemotron research branch:

- Pick **A+B**.
- A: keep production work on Lynn K=N + APEX-MTP expansion.
- B: run a cheap Qwen3.5 4B LoRA PoC later to measure the diffusion/self-spec
  accept ceiling before paying for Qwen3.5 9B continued pretraining.
- Park Qwen3.6-35B-A3B diffusion training for now; MoE plus bidirectional
  diffusion is too much uncertainty for the first training dollar.

## Local Branch

Branch:

```text
codex/qwen35-mtp-block-verify
```

Changed files:

```text
engine/full_forward.py
engine/incremental_decode.py
engine/mtp_serving.py
engine/mtp_sidecar.py
engine/resident_runner.py
scripts/a100_pack_lynn_native_nvfp4.py
scripts/spark_mtp_speculative_smoke.py
scripts/spark_wait_qwen35_mtp_smoke.sh
reports/mtp/QWEN35_MTP_BLOCK_VERIFY_OVERNIGHT_20260526.md
```

Local syntax gate:

```text
python3 -m compileall -q engine/full_forward.py engine/incremental_decode.py engine/mtp_serving.py engine/mtp_sidecar.py engine/resident_runner.py scripts/a100_pack_lynn_native_nvfp4.py scripts/spark_mtp_speculative_smoke.py
```

Status: passed.

## Runtime Work Completed

Implemented a shared K=N speculative verification skeleton:

- `decode_full_attn_block(...)`: prefix-causal block verifier for full-attention layers.
- `decode_linear_attn_block(...)`: sequential Gated Delta-Net rollout for block verification.
- `_decode_layer_block(...)`: layer-level K=N decode path, delegating K=1/K=2 to existing hot paths.
- `decode_block_to_logits_and_hidden(...)`: runs `[pending, draft_1, ..., draft_k]` and returns all-position logits.
- `speculative_step_kn_batched(...)`: verify/accept/replay rollback path for K>1.
- `LYNN_MTP_SPECULATIVE_K`: opt-in env knob; default behavior remains K=1.
- Smoke report fields for `draft_tokens_proposed`, `accepted_draft_tokens`, and `draft_accept_rate`.
- Conservative K>N state repair: even full-block accepts restore the pre-block
  recurrent/conv snapshot and replay the committed prefix through canonical T=1
  decode before continuing. This protects token exactness while the K>N verifier
  is still experimental.

The K>1 draft source is intentionally marked experimental: the current official
sidecar is one-token MTP. Recursive chaining lets the runtime be tested, but it
does not prove the head was trained for multi-token self-drafting.

## Spark Artifact Recovery

Spark no longer had the historical Lynn-native W4A16 Qwen35 artifact:

```text
/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000
```

Rebuilt it from the existing BF16 Qwen35 checkpoint using the Lynn-native packer.
The original packer OOM-killed Spark because it quantized whole tensors at once.
Patched `scripts/a100_pack_lynn_native_nvfp4.py` with row-chunk quantization.

New artifact:

```text
/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526
```

Pack result:

```json
{
  "output_shards": 7,
  "quantized_count": 553,
  "kept_count": 492,
  "elapsed_seconds": 400.4462103843689
}
```

Validation:

```text
size: 23G
layout: lynn_native_per16_variable
backend: lynn_native_per16
manifest: lynn-variable-nvfp4-pack-v1
```

This restores the W4A16 input needed for Lynn MTP experiments.

## Spark Smoke State

Attempted a minimal real-model smoke:

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e LYNN_MOE_IMPL=packed_nvfp4 \
  -e LYNN_MOE_FAST_FIXED=1 \
  -e LYNN_NATIVE_ACTIVE_MOE_BACKEND=triton \
  -e LYNN_NATIVE_GATEUP_BACKEND=triton_fast_decode \
  -e LYNN_NATIVE_DOWN_BACKEND=triton \
  -e LYNN_ROUTER_TOPK_SORTED=0 \
  -e LYNN_LINEAR_ATTN_RECURRENT_BACKEND=triton_fused_prepare \
  -e LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1 \
  -e LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1 \
  -e LYNN_LINEAR_STATE_UPDATE=inplace \
  -e LYNN_LINEAR_BLOCK_GRAPH=0 \
  -e LYNN_LINEAR_BLOCK_GRAPH_REUSE=0 \
  -e LYNN_LINEAR_BLOCK_GRAPH_PREWARM=0 \
  -e LYNN_PACKED_DECODE=0 \
  -e LYNN_PACKED_SHARED_EXPERT=0 \
  -e LYNN_NATIVE_FP4_LM_HEAD=1 \
  -e LYNN_QK_NORM_ROPE_BACKEND=triton_pair \
  -e LYNN_RMSNORM_GATED_BACKEND=triton \
  -e LYNN_FULL_ATTN_ROPE_CACHE=1 \
  -v /home/merkyor/lynn-engine:/workspace \
  -v /home/merkyor/models:/models \
  -w /workspace \
  lynn-eval-base:cu13 \
  python3 scripts/spark_mtp_speculative_smoke.py \
    --model /models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526 \
    --sidecar /models/mtp_sidecars/qwen36-35b-a3b-mtp-official-lynn-fused/mtp.safetensors \
    --out reports/mtp/qwen35_mtp_k4_smoke_20260526_0018.json \
    --max-new 8 \
    --spec-k-list 4
```

Observed:

- The runner loaded via Lynn variable NVFP4 slow-dequant path first.
- This is the existing Lynn resident design: BF16 resident weights plus packed
  aliases for decode kernels.
- Current Spark GPU was already occupied by Nemotron SGLang (~52GB), a
  llama-server (~30GB), and ASR/TTS services. The smoke entered unhealthy memory
  pressure and SSH banner exchange started timing out.

Conclusion: do not treat this failed smoke as an algorithm result. It is a
resource contention/run-path result.

Morning connectivity diagnosis:

- `dgx` / `dgx-spark` via Tencent frp `127.0.0.1:2224`: connection refused.
- `dgx-via-ssh` via Tencent autossh `127.0.0.1:2222`: connection refused.
- Tencent currently listens only on N5 reverse port `127.0.0.1:2223`.
- `dgx-via-n5` reaches TCP port 22 on `192.168.100.26`, but SSH banner times
  out from N5. This points to Spark sshd/host load, not a local code issue.
- Last local sync to Spark happened before the conservative K>N replay repair
  and Config-D-ish smoke env refinements, so those final fixes still need rsync
  once SSH recovers.

Post-reboot result:

- Spark SSH recovered after host reboot, and the latest engine/scripts were
  synced to `/home/merkyor/lynn-engine`.
- Qwen35 W4A16 + official MTP sidecar smoke ran successfully on Spark.
- APEX llama.cpp service was temporarily stopped during the 35B Python runner
  smoke to avoid swap pressure, then restarted after the run.

Primary artifact:

```text
reports/mtp/qwen35_mtp_kn_strict_20260526_100603.json
```

Strict verifier smoke means full-attention K=2/K=block uses the canonical T=1
primitive internally. This is the correctness-first default until the true
batched full-attention verifier is bit-stable.

Summary:

| Config | Exact vs baseline | TPS | Accept | Draft accept |
|---|---:|---:|---:|---:|
| baseline | 100% | 33.65 | n/a | n/a |
| shadow | 100% | 34.31 | n/a | n/a |
| spec_k1 | 100% | 29.58 | 74.04% | 74.04% |
| spec_k1_batched_strict | 100% | 29.25 | 77.16% | 77.16% |
| spec_k2_batched_strict | 100% | 16.29 | 2.38% | 63.34% |
| spec_k4_batched_strict | 100% | 13.72 | 0.00% | 45.14% |

Control artifact without strict full-attention fallback:

```text
reports/mtp/qwen35_mtp_kn_smoke_retry_20260526_095832.json
```

That run reproduced the runtime gap:

- `spec_k1` sequential stayed exact.
- `spec_k1_batched`, `spec_k2_batched`, and `spec_k4_batched` failed token
  exactness when the fast batched full-attention verifier was enabled.
- Therefore the current blocker is batched full-attention verifier numerical
  parity, not MTP draft quality, KV crop, or reject replay.

Code default was changed accordingly:

- `LYNN_FULL_ATTN_K2_BACKEND` now defaults to `t1_loop`.
- `LYNN_FULL_ATTN_BLOCK_BACKEND` now defaults to `t1_loop`.
- Fast paths remain available only by explicit env opt-in:
  `LYNN_FULL_ATTN_K2_BACKEND=k2` and `LYNN_FULL_ATTN_BLOCK_BACKEND=block`.

## Late-Night K2 Fast-Path Diagnosis

Follow-up sweep after the strict smoke isolated the fast K2 drift more tightly.
The full-attention K2 fast path first diverges at layer 3 even at zero advance:

| Artifact | Extra env | Safe combos | Finding |
|---|---|---|---|
| `qwen35_m18_layer_sweep_20260526_223235.json` | default `triton_pair` QK/RoPE | `t1_full_attn_only`, `t1_both` | Fast `decode_full_attn_k2` drifts at L3 full-attention. |
| `qwen35_m18_layer_sweep_qk_torch_20260526_223856.json` | `LYNN_QK_NORM_ROPE_BACKEND=torch` | `t1_full_attn_only`, `t1_both` | Drift remains, so it is not just Triton-pair vs torch QK/RoPE. |
| `qwen35_m18_layer_sweep_rowwise_t1_20260526_224505.json` | `LYNN_FULL_ATTN_K2_PROBE=rowwise_t1` | `t1_full_attn_only`, `t1_both` | Row-wise attention/gate/o-proj alone reduces one position but does not restore exactness. |
| `qwen35_m18_layer_sweep_rowwise_qkv_20260526_225400.json` | `LYNN_FULL_ATTN_K2_PROBE=rowwise_qkv` | `t1_full_attn_only`, `t1_both` | Row-wise QKV/RoPE alone also does not restore exactness. |
| `qwen35_m18_layer_sweep_rowwise_qkv_t1_20260526_230101.json` | `LYNN_FULL_ATTN_K2_PROBE=rowwise_qkv_rowwise_t1` | all four combos | Exact parity returns: logits and recurrent state diffs are zero in the M18 probe. |

Conclusion: the K2 verifier mismatch is a composition effect across batched
QKV/RoPE plus batched attention/gate/o-proj, not KV crop, reject replay,
training, or linear-attention state handling. A correctness bridge now exists:

```text
LYNN_FULL_ATTN_K2_BACKEND=rowwise_bridge
```

This mode keeps the K2 verifier shell but computes full-attention QKV/RoPE and
attention/o-proj row-wise inside `decode_full_attn_k2`. It is a diagnostic
parity bridge, not a throughput solution.

End-to-end bridge smoke:

```text
reports/mtp/qwen35_mtp_safe_k2_bridge_20260526_230739.json
```

| Config | Exact vs baseline | TPS | Accept | Draft accept |
|---|---:|---:|---:|---:|
| baseline | 100% | 34.29 | n/a | n/a |
| spec_k1 | 100% | 29.20 | 74.04% | 74.04% |
| spec_k1_batched | 100% | 28.92 | 77.16% | 77.16% |
| spec_k2_batched + rowwise bridge | 100% | 16.27 | 2.38% | 63.34% |

The bridge confirms correctness but does not improve speed over the strict
T=1 fallback. The next performance task is therefore a real batched full-attn
K2 kernel/implementation that matches T=1 numerics, not more row-wise fallback.

Additional SDPA isolation on 2026-05-27 tested `manual_gqa` for both T=1 and
K2 full-attention decode:

```text
reports/mtp/qwen35_m18_layer_sweep_manual_gqa_20260527_000731.json
reports/mtp/qwen35_mtp_k2_manual_gqa_fast_20260527_001547.json
```

Layer sweep result:

| Combo | Result |
|---|---|
| `k2_both` + `manual_gqa` | drift, first bad L5 `linear_attention` |
| `t1_full_attn_only` + `manual_gqa` | exact |
| `t1_linear_attn_only` + `manual_gqa` | drift, first bad L5 `linear_attention` |
| `t1_both` + `manual_gqa` | exact |

End-to-end smoke result with `LYNN_FULL_ATTN_DECODE_BACKEND=manual_gqa`,
`LYNN_FULL_ATTN_K2_BACKEND=k2`, full-accept fast commit, and prefix repair:

| Config | Exact vs baseline | TPS | Accept | Draft accept |
|---|---:|---:|---:|---:|
| baseline | 100% | 33.66 | n/a | n/a |
| spec_k1 | 100% | 29.39 | 77.16% | 77.16% |
| spec_k1_batched | 66.67% | 31.54 | 80.05% | 80.05% |
| spec_k2_batched | 83.33% | 25.90 | 62.40% | 69.30% |

Conclusion: replacing SDPA with the existing Python `manual_gqa` path is not
enough. It improves some accept/TPS numbers but breaks token-exactness in the
batched verifier. The remaining K2 parity work must separate batched
projection/RoPE, attention, gating, and `o_proj` numerics rather than treating
SDPA as the only source of drift.

Two component probes were added to split the exact row-wise bridge:

```text
LYNN_FULL_ATTN_K2_PROBE=rowwise_qkv_rowwise_attn_batched_o
LYNN_FULL_ATTN_K2_PROBE=rowwise_qkv_batched_attn_rowwise_o
```

`scripts/spark_mtp_m18_k2_layer_sweep_probe.py` now also supports
`--probe-mode-list`, so future component sweeps can reuse one model load instead
of relaunching the 35B container for each probe.

Artifacts:

```text
reports/mtp/qwen35_m18_layer_sweep_rowwise_qkv_rowwise_attn_batched_o_20260527_002523.json
reports/mtp/qwen35_m18_layer_sweep_rowwise_qkv_batched_attn_rowwise_o_20260527_003147.json
```

Component result:

| Probe | Meaning | Result |
|---|---|---|
| `rowwise_qkv_rowwise_attn_batched_o` | QKV/RoPE + attention row-wise; gate/`o_proj` batched | drift at L5 |
| `rowwise_qkv_batched_attn_rowwise_o` | QKV/RoPE + gate/`o_proj` row-wise; attention batched | drift at L5 |
| `rowwise_qkv_rowwise_t1` | QKV/RoPE + attention + gate/`o_proj` all row-wise | exact |

So the fast K2 problem has at least two independent numerical contributors:
batched attention and batched output projection/gating. A production fast K2
path must make both components T=1-equivalent, or choose a controlled
approximate mode as a deliberate product/model change.

Follow-up split of gate vs `o_proj`:

```text
reports/mtp/qwen35_m18_layer_sweep_gate_o_split_20260527_004529.json
```

| Probe | Meaning | Result |
|---|---|---|
| `rowwise_qkv_rowwise_attn_batched_gate_rowwise_o` | only gate is batched; `o_proj` row-wise | exact |
| `rowwise_qkv_rowwise_attn_rowwise_gate_batched_o` | gate row-wise; only `o_proj` batched | drift at L5 |

This isolates the output side further: batched sigmoid/gating is safe, while
the `[2, hidden] @ o_proj.weight` batched matmul is not T=1-equivalent enough
for deterministic MTP parity. The remaining independent blocker is batched
attention itself. The next fast-K2 kernel work should therefore target:

1. A dual-row attention kernel that preserves T=1-equivalent accumulation.
2. A dual-row `o_proj` kernel/dispatch that shares weight loads but accumulates
   each row exactly like the T=1 path.

The non-strict fast-K2 control does not justify accepting approximate drift:

```text
reports/mtp/qwen35_mtp_kn_smoke_retry_20260526_095832.json
```

| Config | Exact vs baseline | TPS | Notes |
|---|---:|---:|---|
| spec_k1_batched fast-K2 | 33.33% | 29.47 | Similar speed to strict, but token exactness breaks. |
| spec_k2_batched fast-K2 | 83.33% | 16.59 | Only +1.8% over strict K2, still far below baseline. |
| spec_k4_batched fast block | 50.00% | 14.27 | Approximate and still slow. |

So the shortcut "ship approximate fast-K2 as a slightly different model" is not
attractive: it loses deterministic parity without buying meaningful TPS.

## K>N Full-Accept Fast Commit

The first real K>N speed recovery is to avoid replay when the verifier accepts
the entire proposed block. This is gated by:

```text
LYNN_MTP_KN_FULL_ACCEPT_FAST_COMMIT=1
```

Behavior:

- If `accepted_count == draft_count`, keep the verifier's final state and use
  the last verifier row's hidden/logits for `next_base_hidden` and
  `next_pending_id`.
- If the block is partially accepted or rejected, keep the existing conservative
  restore-and-replay path.

Spark smoke:

```text
reports/mtp/qwen35_mtp_kn_full_accept_fast_20260526_233834.json
```

| Config | Exact vs baseline | TPS | Accept | Draft accept |
|---|---:|---:|---:|---:|
| baseline | 100% | 33.42 | n/a | n/a |
| spec_k1 | 100% | 29.29 | 74.04% | 74.04% |
| spec_k1_batched | 100% | 28.99 | 77.16% | 77.16% |
| spec_k2_batched + full-accept fast commit | 100% | 24.36 | 56.57% | 63.80% |

Compared with the strict K2 replay smoke, K2 effective TPS improves from
16.29 to 24.36 tok/s (**+49.5%**) while preserving 6/6 token exactness.
The mean is still below baseline, but high-accept prompts already cross it:
prompt 003 reaches 35.42 tok/s. This confirms the runtime now has the right
shape: better draft training directly converts into throughput instead of
being swallowed by replay.

K4 was tested with the same optimization:

```text
reports/mtp/qwen35_mtp_k4_full_accept_fast_20260526_234710.json
```

Result: **do not promote K4 direct commit yet.**

| Config | Exact vs baseline | TPS | Accept | Draft accept |
|---|---:|---:|---:|---:|
| spec_k4_batched + full-accept fast commit | 83.33% | 15.51 | 19.12% | 44.38% |

One prompt diverged, and mean TPS remains far below baseline. The runtime now
defaults the fast-commit cap to K=2 via:

```text
LYNN_MTP_KN_FULL_ACCEPT_FAST_COMMIT_MAX_K=2
```

K>2 needs separate block-state parity work before direct state commit is safe.

## K2 Partial-Accept Prefix Repair

The next replay reduction is partial-accept repair for K=2. When exactly one
draft token is accepted, the committed prefix is `[pending, accepted_draft]`.
The new opt-in path restores the pre-block recurrent/conv snapshot, then
repairs that committed prefix with a smaller exact block verifier instead of
falling back to canonical token-by-token T=1 replay.

Knobs:

```text
LYNN_MTP_KN_PREFIX_BLOCK_REPAIR=1
LYNN_MTP_KN_PREFIX_BLOCK_REPAIR_MAX_LEN=2
```

Spark smoke:

```text
reports/mtp/qwen35_mtp_k2_prefix_repair_20260526_235627.json
```

| Config | Exact vs baseline | TPS | Accept | Draft accept |
|---|---:|---:|---:|---:|
| baseline | 100% | 33.43 | n/a | n/a |
| spec_k1 | 100% | 29.24 | 74.04% | 74.04% |
| spec_k1_batched | 100% | 28.94 | 77.16% | 77.16% |
| spec_k2_batched + full-accept fast commit + prefix repair | 100% | 24.79 | 58.16% | 65.10% |

This keeps the K2 gate at 6/6 exact and nudges K2 effective TPS from 24.36 to
24.79 tok/s over the full-accept-only smoke. The improvement is small, so keep
this path opt-in for now. The more important result is that K2 direct-state
commit and K2 prefix-state repair both preserve token parity when capped to
two-token blocks.

## Next Smoke Command

Run once Spark is free enough, ideally after Nemotron SGLang exits:

```bash
docker run --rm --gpus all --ipc=host \
  -v /home/merkyor/lynn-engine:/workspace \
  -v /home/merkyor/models:/models \
  -w /workspace \
  lynn-eval-base:cu13 \
  python3 scripts/spark_mtp_speculative_smoke.py \
    --model /models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526 \
    --sidecar /models/mtp_sidecars/qwen36-35b-a3b-mtp-official-lynn-fused/mtp.safetensors \
    --out reports/mtp/qwen35_mtp_k4_smoke_$(date +%Y%m%d_%H%M%S).json \
    --max-new 32 \
    --spec-k-list 2,4
```

Or use the non-destructive waiter, which does not stop any service:

```bash
MODEL=/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526 \
SIDECAR=/home/merkyor/models/mtp_sidecars/qwen36-35b-a3b-mtp-official-lynn-fused/mtp.safetensors \
MAX_USED_MIB_BEFORE_RUN=60000 \
MAX_NEW=32 \
SPEC_K_LIST=2,4 \
scripts/spark_wait_qwen35_mtp_smoke.sh
```

Success criteria:

- Baseline runs without resource pressure.
- `spec_k1` and `spec_k1_batched` remain token-exact vs baseline.
- `spec_k2_batched`/`spec_k4_batched` either pass token-exact or fail with a
  specific traceback/prefix boundary.
- If token-exact passes but `draft_accept_rate` falls off with K, the next step
  is draft-head training, not verifier rewrite.

## Training Path If Needed

If the official one-token sidecar does not chain cleanly to K=4/K=8, train the
draft side rather than forcing runtime tricks:

1. Start with Qwen35-A3B K=2/K=4 MTP continuation training.
2. Freeze the base model initially; train an offset-k draft adapter/head against
   teacher-forced base continuations.
3. Use heldout prompt continuations to measure accept rate, not just CE loss.
4. Promote only if `draft_accept_rate` plus block-verify cost beats graph
   baseline TPS.
5. Mirror the same ABI for Qwen3.5-9B with a dense-model sidecar.

Hardware estimate:

- Dual A100 is enough for adapter/head training and accept-rate experiments.
- H100 shortens iteration time and makes longer K/longer context batches easier.
- Full joint AR-diffusion continued pretrain is a separate, much larger project;
  start with targeted MTP/block-draft training first.
