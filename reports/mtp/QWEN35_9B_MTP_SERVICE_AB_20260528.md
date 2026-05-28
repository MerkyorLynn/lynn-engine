# Qwen3.5 9B APEX-MTP Service A/B (2026-05-28)

## Result

The 35B APEX-MTP serving pattern transfers to Qwen3.5 9B for **single-stream**
decode. On Spark, the best 9B setting is again `draft-mtp n_max=4`.

| Mode | Single Wall TPS | Single Server TPS | Single Accept | 4-Way Wall TPS | 4-Way Server TPS | 4-Way Accept |
|---|---:|---:|---:|---:|---:|---:|
| AR on non-MTP server | 36.61 | 38.18 | n/a | **120.58** | **31.83** | n/a |
| AR on MTP-capable server (`n_max=0`) | 32.45 | 34.12 | n/a | 32.13 | 8.25 | n/a |
| MTP `n_max=1` | 35.76 | 36.92 | 91.84% | 46.67 | 12.45 | 86.63% |
| MTP `n_max=2` | 46.58 | 48.51 | 79.45% | 62.72 | 16.91 | 76.85% |
| MTP `n_max=4` | **60.95** | **64.20** | 64.15% | 76.52 | 21.55 | 60.54% |

Compared with the non-MTP AR server, `n_max=4` gives about **+66% single-stream
wall TPS**: 36.61 -> 60.95 TPS.

The concurrency lesson matches the 35B run: MTP is useful for low queue depth,
but it is not the 4-way serving default. Non-MTP AR still has the better
4-way aggregate throughput: 120.58 vs 76.52 total TPS.

## Environment

- Host: `dgx-spark`
- Model: `/home/merkyor/lynn-quant/qwen35-9b/gguf/Qwen3.5-9B-Q4_K_M-imatrix-mtp.gguf`
- Model size: 5.4 GiB
- llama.cpp branch: `/home/merkyor/build/llama.cpp` `codex/apex-mtp-service-ab`
- Temp binary: `/home/merkyor/build/llama.cpp/build-cuda-sm121-codex-ab/bin/llama-server`
- Temp port: `18099`
- Production 35B fallback stayed active on `18098`

## Artifact Provenance Statement

This experiment used a local GGUF artifact named
`Qwen3.5-9B-Q4_K_M-imatrix-mtp.gguf`.

Objective facts established during the run:

- The base model family is Qwen3.5 9B.
- The GGUF contains an embedded MTP-capable block:
  `qwen35.nextn_predict_layers` is present and the file contains
  `blk.32.nextn.*` tensors.
- llama.cpp successfully initialized `draft-mtp` from this GGUF:
  `common_speculative_impl_draft_mtp: adding speculative implementation
  'draft-mtp'`.
- Request-level `speculative.n_max` changed measured draft behavior:
  `n_max=1/2/4` produced non-zero `draft_n` and `draft_n_accepted`.
- The benchmark data in this report was produced by Lynn on Spark.

What this run does **not** establish:

- It does not identify the training dataset used for the embedded MTP head.
- It does not prove that the MTP head was trained by Lynn.
- It does not redistribute or document any private MTP training data.
- It does not replace the original Qwen base-model attribution or license.

Safe public wording:

```text
This artifact is based on Qwen3.5 9B and is packaged as a local experimental
GGUF with an embedded MTP-capable block for llama.cpp `draft-mtp` serving. The
Spark benchmark below describes the observed serving behavior of this artifact.
Base-model attribution remains with Qwen. MTP-head training-data provenance is
not encoded in the local manifest used for this experiment, so this page does
not claim a specific MTP training dataset.
```

The 9B service used:

```bash
llama-server \
  -m /home/merkyor/lynn-quant/qwen35-9b/gguf/Qwen3.5-9B-Q4_K_M-imatrix-mtp.gguf \
  --host 127.0.0.1 \
  --port 18099 \
  -a qwen35-9b-q4km-imatrix-mtp \
  --ctx-size 32768 \
  --parallel 4 \
  --threads 4 \
  --n-gpu-layers 999 \
  -fa on \
  --jinja \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --reasoning auto \
  --reasoning-budget -1 \
  --spec-type draft-mtp \
  --spec-draft-n-max 4 \
  --metrics
```

Important footnote: do **not** start this branch with
`--spec-type none,draft-mtp`. The parser returns `NONE` as soon as it sees
`none`, so the server logs `no implementations specified for speculative
decoding` and every request reports `draft_n=0`. The valid A/B setup is:

- Start server with `--spec-type draft-mtp`.
- Disable AR per request with `"speculative.n_max": 0`.
- Sweep MTP per request with `"speculative.n_max": 1`, `2`, or `4`.

## Artifacts

```text
reports/mtp/qwen35_9b_service_ab_draftmtp_20260528_165625/
  summary.json
  invalid_nested_payload_summary_topkey.json
```

Remote raw run:

```text
/home/merkyor/lynn-apex-mtp-runs/qwen35_9b_service_ab_draftmtp_20260528_165625
```

## Conclusion

For 9B, APEX-MTP should be treated as a **single-stream acceleration mode**:

```text
queue depth <= 1 -> enable draft-mtp, n_max=4
queue depth >= 2 -> prefer non-MTP AR service
```

This is strong enough to justify carrying the 35B dynamic-MTP-admission policy
over to the 9B serving path, but not enough to replace the high-concurrency
Q4_K_M-imatrix AR default.
