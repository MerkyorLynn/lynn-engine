# Qwen3.6-35B P163 Router Boundary Promotion

**Date:** 2026-05-19  
**Model:** official Qwen3.6-35B-A3B Lynn-native W4A16 NVFP4  
**Candidate:** `LYNN_ROUTER_TOPK_OUT_BUFFER=1`

## Microprobe

Artifact:
`reports/qwen36_35b/p163_qwen36_router_boundary_probe_20260519_0527.json`

P163 keeps the Torch router math but uses caller-owned `torch.topk(..., out=...)`
buffers for the decode shape.

| Metric | Value |
|---|---:|
| Fixtures | 18 |
| Exact ids/routes/logits | 18 / 18 |
| Default router mean | 0.04087 ms |
| Out-buffer router mean | 0.03945 ms |
| Delta | -0.00142 ms/layer |
| topk-only delta | -0.00139 ms |

## Promotion Gate

Artifact:
`reports/qwen36_35b/r6000_qwen36_w4a16_router_topk_out_20260519_0534_router_topk_out_promotion_summary.json`

| Gate | Result |
|---|---:|
| P37 exact | true |
| P37 median speedup | 1.0019x |
| P25 512 decode TPS | 108.06 |
| hard structured | 40 / 40 |
| hard structured mean decode TPS | 108.38 |
| Decision | `DEFAULT_CANDIDATE` |

## Default Change

`LYNN_ROUTER_TOPK_OUT_BUFFER=1` is now part of the safe W4A16 default profile
for Qwen3.6-35B. The win is small but clean: it is exact, structured-safe, and
raises the default safety line from the 107 TPS class to the 108 TPS class.

Future candidate gates now use `SAFE_DEFAULT_TPS=108.0` and
`DEFAULT_P25_TPS=109.0` so the next default promotion must beat this new line.
