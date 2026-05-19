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

Current long-run source:

```text
/root/autodl-tmp/reports/qwen35_9b/thinking32/qwen35_9b_q4km_gpqa_thinking32_20260519_182901.jsonl
```

The summary JSON includes progress, naive accuracy, parse-fail adjusted
accuracy, elapsed-time distribution, raw character distribution, completion
token distribution, per-subject accuracy, parse-fail leaders, and hard-subject
details for chemistry/physics-heavy categories.
