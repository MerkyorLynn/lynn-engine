# Qwen3.6-35B-A3B APEX-MTP Quality32K Refresh

Date: 2026-05-28 Asia/Shanghai

## Current Production Service

Spark is currently serving the fastest single-stream APEX-MTP route:

```text
Service: lynn-apex-mtp-llamacpp.service
Model: /home/merkyor/models/Qwen3.6-35B-A3B-APEX-MTP-GGUF/Qwen3.6-35B-A3B-APEX-MTP-I-Balanced.gguf
Port: 18098
Speculation: --spec-type draft-mtp --spec-draft-n-max 4
Role: Brain V2 fallback #2
```

Sanity check on 2026-05-28:

| Request | Wall TPS | Server TPS | Draft accepted |
|---:|---:|---:|---:|
| 1 | 72.75 | 82.13 | 68 / 104 |
| 2 | 76.19 | 84.54 | 69 / 95 |
| 3 | 80.77 | 90.40 | 71 / 94 |

Median wall TPS: **76.19**.

This matches the short A/B result in
`reports/mtp/LLAMA_CPP_APEX_MTP_SERVICE_AB_20260528.md`, where MTP `n_max=4`
reached **77.01 wall TPS** vs **60.65 wall TPS** AR on the same service loop.

## Existing Formal 32K Thinking-On Results

These were already completed on Spark for the same APEX I-Balanced route:

| Gate | N | Result | Notes |
|---|---:|---:|---|
| MMLU 500 5-shot | 500 | **90.00%** | 450 / 500, parse_fail 0 |
| GPQA Diamond | 198 | **78.79% naive** / **83.87% excl parse fail** | 156 / 198, parse_fail 12 |
| Tool-call thinking-on | 15 | **12 / 15** | ship_gate PASS |
| Tool-call thinking-off | 15 | 11 / 15 | ship_gate FAIL |

Copied summaries:

```text
reports/qwen36_35b/apex_quality32k_20260521/
  mmlu500_thinkon32k_20260521_223238.summary.json
  gpqa_diamond198_thinkon32k_20260521_223238.summary.json
  toolcall_thinking_on_20260521_222423.summary.json
  toolcall_thinking_off_20260522_111058.summary.json
```

## Fresh Refresh Run

A fresh serialized 32K thinking-on refresh was started on Spark to align with
the current 77 TPS production service:

```text
PID: 252228
Run dir: /home/merkyor/eval/reports/apex_quality32k_20260528_121431
Local runner: scripts/spark_apex_quality32k_eval_20260528.sh
Remote runner: /home/merkyor/eval/scripts/spark_apex_quality32k_eval_20260528.sh
```

The run order is intentionally serialized to avoid 32K KV-cache pressure:

```text
1. GPQA parse smoke, 2 questions, 4K cap
2. V8 stage1 tool-call, thinking-on, 32K
3. V8 stage5 coding/tool-call, thinking-on, 32K
4. V8 stage4 research holdout, thinking-on, 32K
5. V9 all, runs=2, thinking-on, 32K cap
6. MMLU 500 5-shot, thinking-on, 32K
7. GPQA Diamond 198, thinking-on, 32K
```

The previous full GPQA 32K run took about **10.55 hours**, so this refresh is
expected to run long.

## Serving Policy

The result so far supports a split policy:

```text
single / low queue depth -> APEX-MTP n_max=4
multi-slot / high queue  -> AR or request-level n_max=0
```

APEX-MTP is a strong single-stream accelerator on Spark, but current llama.cpp
4-slot concurrency still favors AR.
