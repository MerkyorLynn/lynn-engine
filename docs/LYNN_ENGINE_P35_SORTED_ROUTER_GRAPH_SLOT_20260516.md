# Lynn Engine P35 · Sorted router restores full-token graph-slot parity

Date: 2026-05-16  
Hardware: RTX PRO 6000 Blackwell Server Edition (`sm_120`)  
Model: `lynn-27b-variable-recovery-step5000-nvfp4-final`

## Context

P20 promoted `torch.topk(sorted=False)` because it was exact for the eager/full-graph path and improved the R6000 graph ceiling. P14-C later showed that graph-owned full-token slots drifted badly in the current runtime. P35 tests whether router ordering is the missing graph-slot contract.

## Experiment

Runtime profile:

```bash
export LYNN_MOE_IMPL=packed_nvfp4
export LYNN_NATIVE_FP4_LM_HEAD=1
export LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1
export LYNN_LINEAR_ATTN_RECURRENT_BACKEND=triton_fused_prepare
export LYNN_LINEAR_STATE_UPDATE=inplace
export LYNN_PACKED_DECODE=0
export LYNN_PACKED_SHARED_EXPERT=0
export LYNN_ROUTER_TOPK_SORTED=1
```

Gate:

```bash
python benchmarks/p10t_runner_graph_slot_gate.py \
  --model /root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final \
  --out reports/p16_155/p35_p10t_sorted_graph_slot_multiprompt_gate.json \
  --prompts-jsonl /tmp/p35_prompts.jsonl \
  --prefix-new 8 16 32
```

Prompt set:

- Chinese MoE explanation
- Python recursive factorial
- RoPE vs ALiBi comparison
- English arithmetic

Each prompt runs prefix lengths 8 / 16 / 32, for 12 graph-slot parity checks total.

## Result

| Metric | Value |
|---|---:|
| Strict rows | 12/12 PASS |
| `max_abs` | 0.0 for all rows |
| top-1 match | 12/12 |
| top-10 overlap | 10/10 for all rows |
| graph replay TPS min | 97.74 |
| graph replay TPS median | 100.89 |
| graph replay TPS mean | 102.52 |
| graph replay TPS max | 111.51 |

## Interpretation

`LYNN_ROUTER_TOPK_SORTED=1` restores full-token graph-slot parity. The graph slot contract is therefore stricter than the eager/full-graph contract:

- eager/full-graph path can safely use unsorted top-k from P20;
- full-token graph-slot path needs sorted top-k to keep replay exact.

This is a useful fork rather than a contradiction. P20 remains valid for the current stable serving path. P35 gives the graph-slot line a correctness-preserving way forward.

## Decision

Do not globally revert P20. Instead, future graph-slot serving modes should force or internally require:

```bash
LYNN_ROUTER_TOPK_SORTED=1
```

The next engineering target is to implement a graph-slot serving mode that explicitly selects the sorted-router contract, then benchmark end-to-end wall TPS against the current P25 OpenAI server path.

