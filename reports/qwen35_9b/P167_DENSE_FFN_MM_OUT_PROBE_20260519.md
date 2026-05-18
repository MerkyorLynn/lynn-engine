# P167 Qwen3.5-9B Dense FFN `mm(out=...)` Probe

Date: 2026-05-19

## Purpose

After P164/P165 closed the existing packed NVFP4 wrappers, P167 tested the
most conservative exact boundary change: keep the BF16/dequant math path and
replace `F.linear` temporaries with caller-owned `torch.mm(..., out=...)`
buffers for gate, up, intermediate, and down.

## R6000 Result

| Metric | Result |
|---|---:|
| Exact | 8/8 |
| Max abs | 0.0 |
| Min cosine | 1.0 |
| Reference FFN | 0.21573 ms |
| `mm(out=...)` FFN | 0.21568 ms |
| Mean speedup | 1.00024x |
| Decision | closed, flat |

## Decision

Do not promote.  The exactness result is useful because it proves
caller-owned Torch matmul buffers preserve the P160 contract, but there is no
meaningful speed gain.  The 9B dense FFN speed path therefore requires a real
fused kernel or offline weight-layout repack rather than Torch allocation
avoidance.

## Artifacts

- `benchmarks/p167_qwen35_9b_dense_ffn_mm_out_probe.py`
- `scripts/r6000_qwen35_9b_dense_ffn_p167_mm_out_probe.sh`
- `reports/qwen35_9b/p167_dense_ffn_mm_out_probe_20260519_0725_mm_out.json`
