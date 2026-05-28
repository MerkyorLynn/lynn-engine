# HF / ModelScope APEX-MTP Release Checklist

## What To Upload

For a llama.cpp-ready APEX-MTP release, the artifact must include MTP capability:

- A GGUF with `qwen35moe.nextn_predict_layers = 1` and the trailing MTP block
  tensors embedded, or
- A base model plus a clearly named MTP sidecar and loader instructions.

For the current Spark path, embedded GGUF is cleaner because `llama-server`
starts with:

```bash
--spec-type draft-mtp --spec-draft-n-max 4
```

and does not require users to manually pair a separate sidecar.

## What Not To Upload

Do not upload private training data. For model hubs, the useful "data" is
release metadata:

- quantization recipe,
- imatrix provenance,
- benchmark tables,
- launch commands,
- license and base model attribution.

If redistribution of the calibration set is allowed, include the imatrix file
or dataset reference. If not, record only the calibration recipe and path/name
used during quantization.

## Required Model Card Fields

- Base: Qwen3.6-35B-A3B
- License: Apache-2.0, matching the base model metadata
- Quantization: current `I-Balanced` imatrix, plus future Q4_K_M imatrix if built
- MTP: embedded APEX-MTP / `nextn_predict_layers=1`
- llama.cpp command:

```bash
llama-server \
  -m Qwen3.6-35B-A3B-APEX-MTP-I-Balanced.gguf \
  --ctx-size 262144 \
  --parallel 4 \
  --n-gpu-layers 999 \
  -fa on \
  --jinja \
  --spec-type draft-mtp \
  --spec-draft-n-max 4 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0
```

## Current Spark Numbers

From `reports/mtp/service_ab_20260528_115837/summary.json`:

| Mode | Single Wall TPS | 4-Way Wall TPS | Notes |
|---|---:|---:|---|
| AR baseline | 60.65 | 124.80 | best current concurrency path |
| APEX-MTP `n_max=4` | 77.01 | 85.62 | best current single-stream path |

Existing formal 32K thinking-on quality summaries for the same APEX
I-Balanced route are copied under
`reports/qwen36_35b/apex_quality32k_20260521/`:

| Gate | Result |
|---|---:|
| MMLU 500 5-shot | 90.00% |
| GPQA Diamond 198 | 78.79% naive / 83.87% excl parse fail |
| Tool-call thinking-on | 12 / 15 |

Recommended model-card wording:

```text
On Spark GB10 with llama.cpp sm_121/cu13, APEX-MTP improves single-stream
generation by about 27% over AR in a short HTTP benchmark. For 4-way concurrent
serving, plain AR is currently faster; use dynamic MTP admission or route
high-concurrency traffic to AR.
```

## Q4_K_M Follow-Up

Build a Q4_K_M imatrix GGUF only if the MTP block stays embedded. The expected
benefit is higher single-stream TPS due to lower memory bandwidth, but it needs
fresh quality and accept-rate validation before replacing `I-Balanced`.
