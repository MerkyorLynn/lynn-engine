# Qwen3.6-35B-A3B Spark MTP Reproduction Gate · 2026-05-19

## Decision

Spark 35B work should test MTP first, not another round of small kernel
switches.  The SGLang data point to reproduce is not "FP8 is fast" by itself;
it is "FP8 plus NEXTN/MTP speculative decode reaches roughly 60-69 tok/s with
about 70%+ accept rate".

This does not change the 9B release mainline.  Qwen3.5-9B remains the first
release path for Mac Q4_K_M and NVIDIA NVFP4.  Qwen3.6-35B-A3B becomes a Spark
MTP reproduction track until the accept-rate and quality gates are proven.

## Why This Matters

Current measured baselines:

| Stack | Mode | Decode TPS | Notes |
|---|---:|---:|---|
| Lynn-native Spark | W4A16 NVFP4, no MTP | ~38.96 | Single-token forward path |
| SGLang Spark | FP8, MTP NEXTN | ~60-69 | Reported accept rate ~71-82% |
| Lynn-native R6000 | 35B W4A16 safe | ~107-108 | Different hardware, no MTP credit |

If the SGLang number reproduces, most Spark uplift is speculative decoding
rather than the quantization format alone.  A Lynn MTP path with similar accept
rate would be a larger lever than another 5-10% kernel micro-optimization on
Spark.

## Reproduction Requirements

Run the same model and prompt set with MTP disabled and enabled.

Required outputs:

| Required Field | Pass Bar |
|---|---|
| `base_decode_tps` | Recorded for no-MTP baseline |
| `mtp_decode_tps` | >= 1.5x baseline or >= 60 tok/s on Spark |
| `accept_rate` | >= 60% minimum, >= 70% target |
| `accept_len_mean` | >= 2.0 minimum, >= 2.5 target |
| `quality_smoke` | No obvious structured/tool/coding regression |
| `args` | Full launch command and SGLang version |

The result is not promotion-ready if MTP is a silent no-op, if accept metrics are
missing, or if throughput improves by measuring characters/s rather than
tokens/s.

## Suggested SGLang Probe Args

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.6-35B-A3B-FP8 \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-scheduler-strategy extra_buffer \
  --page-size 64 \
  --cpu-offload-gb 0 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 2
```

Record a paired no-MTP run by removing the speculative flags while keeping all
other settings unchanged.

## Lynn Follow-Up If Green

If Spark SGLang MTP is GREEN:

1. Treat official 35B MTP sidecar as a serving-integration target, not just a
   diagnostic asset.
2. Re-open Lynn MTP state alignment with a paired SGLang trace: token ids,
   draft ids, accepted ids, hidden/state rollback markers.
3. Do not count MTP in Lynn TPS until local accept-rate and end-to-end tokens/s
   pass the same gate.
4. Keep 9B release work unblocked; 35B MTP is a parallel Spark track.

## Current Lynn Caution

Earlier Lynn-side official 35B MTP sweeps passed shape/forward smoke but had
local iterative accept at 0/24.  That means the blocker is likely integration
alignment, state rollback, tokenizer/template, or scheduler semantics.  It is
not safe to assume that simply loading the sidecar will reproduce SGLang's MTP
multiplier.
