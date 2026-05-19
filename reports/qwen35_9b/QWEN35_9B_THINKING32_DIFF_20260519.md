# Qwen3.5-9B Thinking-On 32K Diff

Status: R6000 run is still in progress. This note records the rolling result
after the first 15 GPQA Diamond samples from the Q4_K_M CUDA 32K run.

## Current Rolling Result

| Variant | GPQA Diamond | Parse fail | Notes |
| --- | ---: | ---: | --- |
| Q4_K_M thinking-off | 37.37% | n/a | A/B/C/D short-answer gate |
| Q4_K_M thinking-on 16K | 50.00% naive / 83.33% excl parse-fail | 40.00% | 50-sample run; many organic chemistry questions did not close within 16K |
| Q4_K_M thinking-on 32K | 93.33% naive / 100.00% excl parse-fail | 6.67% | rolling 15/50 sample; not final |

## Interpretation

The 16K run looked weak in naive accuracy mostly because 40% of the questions
failed to produce a parseable final answer before the token budget. With 32K,
the early parse-fail rate has dropped to 1/15. This supports the current gate
policy: report both naive accuracy and `accuracy_excluding_parse_fail`, and keep
32K thinking-on as the serious GPQA mode for 9B.

The only current parse-fail is an Organic Chemistry symmetry question where the
model loops near the answer instead of closing. That keeps parse recovery and
organic-chem-specific prompt tightening on the follow-up list.

## Artifacts

- R6000 JSONL:
  `/root/autodl-tmp/reports/qwen35_9b/q4km_cuda_thinking32_gpqa50_20260519_121803.jsonl`
- Rolling summary:
  `/root/autodl-tmp/reports/qwen35_9b/q4km_cuda_thinking32_gpqa50_20260519_121803.partial.summary.json`
- Summarizer:
  `scripts/summarize_openai_mcq_thinking32.py`
