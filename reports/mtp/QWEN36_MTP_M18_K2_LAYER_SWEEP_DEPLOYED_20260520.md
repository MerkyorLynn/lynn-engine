# Qwen3.6-35B MTP M18 Layer-Type Sweep — Deployed, Pending GPU Window

**Date:** 2026-05-20
**Status:** Code shipped. Probe attempted on Spark; aborted on CUDA OOM.

## What landed

- Commit `087b910` on `origin/main`:
  - `engine/full_forward.py:_decode_layer_k2` — adds opt-in
    `LYNN_MTP_K2_LINEAR_ATTN_MODE` env (default `k2`, alt `t1_loop`).
    Mirrors the existing `LYNN_FULL_ATTN_K2_BACKEND=t1_loop` knob and
    pairs together to give 4 K=2 layer-type configurations.
  - `scripts/spark_mtp_m18_k2_layer_sweep_probe.py` — sweep probe that
    runs the M16 bisect helper with each of the 4 mode combinations.

## Goal

User direction: 按层打开 K2 batching，找出 full-attn / linear-attn 哪一段
能安全批量化。M17's `t1_canonical` is a binary switch (all T=1 or all K=2)
— it does not localize which layer type carries the residual drift.

The 4-combination sweep gives a per-layer-type pass/fail signal:

| combo                  | full-attn       | linear-attn                                          | meaning |
|---|---|---|---|
| `k2_both`              | `decode_full_attn_k2` (SDPA over [B,2]) | `decode_linear_attn_k2` (per-pos internal, end-of-block state.update) | current default |
| `t1_full_attn_only`    | `t1_loop` (2× `decode_full_attn`)        | `k2` internal                                        | M13 setup |
| `t1_linear_attn_only`  | `k2` SDPA                                 | `t1_loop` (2× `decode_linear_attn` with state.update_linear_attn_state interleaved) | **NEW M18** |
| `t1_both`              | `t1_loop`                                  | `t1_loop`                                            | per-layer t1_canonical equivalent |

A combo with `first_bad_layer == None` against canonical 2× T=1
sequential is "safe to batch" for that layer type.

## Spark attempt 2026-05-20 ~09:02

- Launched in comfyui venv on host: PID 3413236
- Probe loaded ~36 of 40 layers (BF16 dequant), then OOMed on layer 37+
- `torch.OutOfMemoryError: Tried to allocate 2.00 GiB. GPU 0 has 119.63 GiB total of which 3.43 GiB is free.`
- Other GPU memory holders at OOM time: probe itself ~77 GB (most of 35B model dequant resident), voice services (lynn-tts/lynn-asr) restarting cycle held ~1-5 GB

The voice services (`sensevoice_server.py` PID 3289447, `cosyvoice_server.py`,
`emotion_server`, `qwen3-asr-spike`) restart periodically. When 35B model load
overlaps a voice restart cycle, GPU mem peaks past 119 GB unified budget and
the smaller process (or any incremental allocation) OOMs.

## Recommended re-run

Re-run when voice services are stable + no other heavy GPU workload:

```bash
cd /home/merkyor/lynn-engine
PYTHONPATH=/home/merkyor/lynn-engine \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
LYNN_MTP_SIDECAR=/home/merkyor/models/mtp_sidecars/qwen36-35b-a3b-mtp-official-lynn-fused/mtp.safetensors \
/home/merkyor/comfyui/ComfyUI/.venv/bin/python -u \
    scripts/spark_mtp_m18_k2_layer_sweep_probe.py \
    --model /home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000 \
    --sidecar /home/merkyor/models/mtp_sidecars/qwen36-35b-a3b-mtp-official-lynn-fused/mtp.safetensors \
    --advance 0 \
    --out /tmp/mtp_m18_k2_layer_sweep_$(date +%Y%m%d_%H%M%S).json
```

Expected runtime: ~5–10 min once GPU is free (model load 3–4 min + 4
prefill+bisect cycles ~1 min each).

## Outcome interpretation guide

After the JSON lands, the `safe_combinations` array names every combo with
`first_bad_layer == None`:

- If `t1_full_attn_only` is safe but `t1_linear_attn_only` is not → the
  K=2 full-attention SDPA is the only divergence source; current `t1_loop`
  full-attn fix is sufficient at the layer level.
- If `t1_linear_attn_only` is safe but `t1_full_attn_only` is not → the
  end-of-block-only state update in `decode_linear_attn_k2` is the residual
  source; promote `LYNN_MTP_K2_LINEAR_ATTN_MODE=t1_loop` to default to
  match sequential's interleaved state.update.
- If both are safe individually but `k2_both` is not → there is an
  interaction (e.g. full-attn drift propagates into linear-attn input,
  pushing it over threshold despite linear-attn k2 being individually
  clean). Use `t1_loop` for both as the safe baseline; promote each
  layer-type fix incrementally.
- If only `t1_both` is safe → the per-layer-type K=2 paths are
  individually unsafe; the right unlock is a true batched K=2 kernel for
  each layer type (longer-horizon work).

## Cross-reference

- M16 bisect: `reports/mtp/QWEN36_MTP_M16_K2_BISECT_RESULT_20260520.md`
- M17 t1_canonical scaffold: `reports/mtp/QWEN36_MTP_M17_CANONICAL_K2_RESULT_20260520.md`
- M13 probe env-gap analysis: `reports/mtp/QWEN36_MTP_M13_PROBE_ENV_GAP_HYPOTHESIS_20260520.md`
- Memory: `project_lynn_engine_t1_only_kernel_contract_20260519` (T=1-only kernel contract framing)
