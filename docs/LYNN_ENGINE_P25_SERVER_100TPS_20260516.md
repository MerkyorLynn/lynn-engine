# Lynn Engine P25 — OpenAI Server 100 TPS Decode Gate (2026-05-16)

P23 moved the internal strict full path to the 118 TPS class and the replay
ceiling to the 123 TPS class. P25 verifies that the OpenAI-compatible server can
also reach the 100 TPS decode class when launched with the correct graph env.

## Why P25 Exists

The first server sanity used the P16 benchmark env only. That env is good for
single-token graph ceiling probes, but it missed the reusable linear block graph
switches used by the service path:

```bash
export LYNN_LINEAR_BLOCK_GRAPH=1
export LYNN_LINEAR_BLOCK_GRAPH_REUSE=1
export LYNN_LINEAR_BLOCK_GRAPH_PREWARM=1
```

Without those switches, the server fell back to an eager-ish decode path and
reported only about 28 decode tok/s. With the graph switches enabled, the same
server reaches the 100 TPS decode class.

## Server Env

```bash
source /tmp/lynn_p16_env.sh
export LYNN_LINEAR_BLOCK_GRAPH=1
export LYNN_LINEAR_BLOCK_GRAPH_REUSE=1
export LYNN_LINEAR_BLOCK_GRAPH_PREWARM=1
export LYNN_FULL_TOKEN_GRAPH_SLOT=0
export LYNN_PACKED_DECODE=0
export LYNN_PACKED_DECODE_PREPARE_NATIVE=0
```

Server:

```bash
python -m server.openai_http \
  --model /root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final \
  --served-name Lynn-27B-NVFP4-P25-Server \
  --host 127.0.0.1 \
  --port 18155
```

`/health`:

```json
{
  "status": "ok",
  "model": "Lynn-27B-NVFP4-P25-Server",
  "runtime": {
    "moe_impl": "packed_nvfp4",
    "packed_nvfp4_moe_aliases_attached": 40,
    "packed_decode_backend": "scalar_bridge",
    "native_fp4_lm_head_enabled": true,
    "runtime_warnings": []
  }
}
```

## Result

Probe:

```bash
python benchmarks/p25_server_decode_tps_probe.py \
  --url http://127.0.0.1:18155/v1 \
  --model Lynn-27B-NVFP4-P25-Server \
  --max-tokens 64 128 256 512 \
  --runs 2 \
  --out reports/p16_155/p25_server_decode_tps_probe.json
```

| Completion cap | Wall TPS mean | Decode TPS mean | Prefill mean |
|---:|---:|---:|---:|
| 64 | 48.95 | 99.42 | 0.659 s |
| 128 | 66.08 | 99.53 | 0.649 s |
| 256 | 79.33 | 99.69 | 0.650 s |
| 512 | 87.73 | 99.25 | 0.656 s |

Median decode step time is consistently about **9.95 ms/token**.

## Interpretation

This is a service-level result, not just a benchmark ceiling:

- OpenAI-compatible `/v1/completions`;
- real tokenizer + prefill;
- reusable linear-block graphs;
- packed NVFP4 active MoE;
- native FP4 lm_head;
- no global `LYNN_PACKED_DECODE` regression path.

The current server can sustain roughly **100 decode tok/s**. End-to-end request
TPS is lower for short completions because prefill costs about 0.65 s per
request. Long completions amortize this cost and reach about 88 wall tok/s at
512 generated tokens.

## Remaining Gap to 155

P25 closes the "server wiring" gap. The remaining 155 target is not blocked by
FastAPI or the OpenAI wrapper. It is blocked by the active routed expert kernel.

Current boundary:

```text
server decode:       ~100 tok/s
strict full graph:   118.73 tok/s
replay ceiling:      123.78 tok/s
skip-active upper:   173.84 tok/s
non-MoE upper:       208.78 tok/s
```

The next real speed step is therefore still the same conclusion from P16/P18/P23/P24:

```text
custom per-16 grouped native-FP4 active expert kernel
```

P25 also adds a small server-specific benchmark script and fixes unsupported
temperature requests to return HTTP 400 instead of an internal server error.

