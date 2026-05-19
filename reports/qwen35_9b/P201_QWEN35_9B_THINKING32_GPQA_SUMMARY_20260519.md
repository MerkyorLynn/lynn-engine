# P201 Qwen3.5-9B Thinking32 GPQA Live Summary

Status: runner merged, R6000 execution in progress.

Purpose: summarize the long-running Qwen3.5-9B Q4_K_M GPQA Diamond
thinking-on 32K JSONL while it is still being written. The summarizer is
safe to rerun repeatedly and tolerates a truncated final JSONL line.

Default R6000 command:

```bash
cd /root/autodl-tmp/lynn-engine-main
bash scripts/r6000_qwen35_9b_gpqa_thinking32_summarize.sh
```

Local/offline summary command:

```bash
python3 scripts/p201_gpqa_thinking32_live_summarizer.py \
  --jsonl reports/qwen35_9b/thinking32/qwen35_9b_q4km_gpqa_thinking32_20260519_182901.jsonl \
  --out reports/qwen35_9b/p201_gpqa_summary.json \
  --md-out reports/qwen35_9b/p201_gpqa_summary.md
```

If `--jsonl` is omitted, the Python helper discovers the newest
`*thinking32*gpqa*.jsonl` or `*gpqa*thinking32*.jsonl` under
`reports/qwen35_9b` (including the `thinking32/` subdirectory). It keeps the
legacy positional JSONL argument for existing live-watch usage.

Current long-run source:

```text
/root/autodl-tmp/reports/qwen35_9b/thinking32/qwen35_9b_q4km_gpqa_thinking32_20260519_182901.jsonl
```

The summary JSON and Markdown include progress, accuracy, parse_fail,
excl_parse_fail / accuracy_excluding_parse_fail, elapsed-time mean/median/p95,
prompt/completion/total token mean/median/p95, raw character distribution,
per-subject accuracy, parse-fail leaders, and hard-subject details for
chemistry/physics-heavy categories.
