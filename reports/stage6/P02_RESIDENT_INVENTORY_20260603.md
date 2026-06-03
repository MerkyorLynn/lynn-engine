# Stage 6 P0.2 — resident BF16 inventory

**Date:** 2026-06-03
**Host:** Spark GB10 (`dgx-via-n5`)
**Model:** `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526`
**Runner:** `scripts/spark_stage6_p02_resident_inventory.py` in docker `lynn-eval-base:cu13`, `PYTHONNOUSERSITE=1`.
**Remote run dir:** `/home/merkyor/lynn-engine/reports/stage6/p02_resident_inventory_20260603_235101`
**APEX safety:** APEX was idle before the run, stopped for the inventory pass, then restored on `:18098`; post-run `/health` returned `{"status":"ok"}`.

## Verdict

**P0.2 PASSED as an inventory gate.** After the 60 GiB grouped-MoE BF16 shadow
release, only **4.72 GiB** of BF16 resident tensors remain. No speed claim is
made here; this gate decides the next kernel order.

The result is important because it narrows Stage 6:

- The 60 GiB MoE expert shadow is already solved by the release/no-reload path.
- Remaining resident BF16 is small enough to inventory exactly.
- The largest remaining targets are projection and embedding/lm-head residents,
  not router weights.
- The script found **0.0 GiB** of packed-alias candidates in this mode, so the
  next reductions require explicit packed-prefill/lookup paths rather than just
  flipping an alias-release flag.

## Summary

| check | result |
|---|---:|
| resident after load | 88.16 GiB |
| BF16 total before release | 64.72 GiB |
| release | 60.00 GiB / 80 tensors |
| resident after release | 28.16 GiB |
| BF16 total after release | 4.72 GiB |
| packed-alias candidate bytes | 0.00 GiB |
| docker status | 0 |
| APEX restored | yes |

## Remaining BF16 By Category

| category | resident BF16 |
|---|---:|
| `linear_attn.projection` | 1.884 GiB |
| `outside.embed` | 0.947 GiB |
| `outside.lm_head` | 0.947 GiB |
| `full_attn.projection` | 0.508 GiB |
| `moe.shared_expert` | 0.391 GiB |
| `moe.router` | 0.039 GiB |
| `layernorm` | 0.0003 GiB |
| `outside.norm` | 0.000004 GiB |

Before release, `moe.grouped_expert_shadow` alone was **60.00 GiB**. It is gone
from the after-release table.

## Top Tensors

| category | layer | key | shape | GiB | packed alias |
|---|---:|---|---:|---:|---|
| `outside.lm_head` | outside | `lm_head.weight` | `248320x2048` | 0.947 | false |
| `outside.embed` | outside | `model.language_model.embed_tokens.weight` | `248320x2048` | 0.947 | false |
| `linear_attn.projection` | 0 | `linear_attn.in_proj_qkv.weight` | `8192x2048` | 0.031 | false |
| `linear_attn.projection` | 1 | `linear_attn.in_proj_qkv.weight` | `8192x2048` | 0.031 | false |
| `linear_attn.projection` | 2 | `linear_attn.in_proj_qkv.weight` | `8192x2048` | 0.031 | false |
| `full_attn.projection` | 3 | `self_attn.q_proj.weight` | `8192x2048` | 0.031 | false |

The full top-40 tensor list is in the remote run log.

## Engineering Decision

P0.2 promotes the next work in this order:

| phase | target | why |
|---|---|---|
| P1 | Batched packed-NVFP4 prefill projections for linear-attn/full-attn | largest remaining layer-resident BF16 block: 2.392 GiB combined |
| P1b | Packed embedding / tied lm-head lookup path, if semantics allow | 1.894 GiB outside resident; high memory value but needs careful token/logit semantics |
| P2 | Grouped M>1 packed MoE prefill kernel | replaces the 20.75 s `stream_bf16` proof path and avoids per-request 23-24 s reload |
| Later | Router / norms | only ~0.039 GiB + tiny tensors; not first-order memory or latency leverage |

Do not promote any P1/P2 kernel on memory alone. The gate remains:
token-exact against BF16/`stream_bf16`, no hidden reload, bounded peak memory,
and measured prefill latency.
