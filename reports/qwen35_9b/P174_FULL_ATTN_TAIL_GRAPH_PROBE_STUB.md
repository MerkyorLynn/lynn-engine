# P174 Qwen3.5-9B Full-Attention Tail Graph Probe

Scope: read-only benchmark for `/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0`.

The probe isolates the dense tail after a full-attention layer's attention/residual add:

`post_attention_layernorm -> dense FFN -> residual add`

## R6000 Result

P174 passed on all 8 full-attention layers:

| Metric | Result |
|---|---:|
| Full-attention layers probed | 8/8 |
| Tail output exact | 8/8 |
| Alt-input tail output exact | 8/8 |
| Suffix logits exact | 8/8 |
| Local greedy exact | 8/8 |
| Eager tail | ~0.247 ms/layer |
| Graph replay | ~0.227-0.230 ms/layer |
| Graph replay + input copy | ~0.229-0.232 ms/layer |

The boundary is clean and reusable: graph replay preserves exact tail outputs
and suffix logits. The gain is modest, about 0.017-0.019 ms per full-attention
layer after input copy. Across 8 full-attention layers this is roughly
0.13-0.15 ms/token, so it is useful as a 9B opt-in building block but not a
large speed lever by itself.

## Run

```bash
bash scripts/r6000_qwen35_9b_full_attn_tail_graph_probe.sh
```

The JSON includes discovered full-attention layers, exactness (`max_abs` and
local greedy parity where possible), eager timing, CUDA graph capture/replay
timing, and a `verdict`.

## Artifacts

- `benchmarks/p174_qwen35_9b_full_attn_tail_graph_probe.py`
- `scripts/r6000_qwen35_9b_full_attn_tail_graph_probe.sh`
- `reports/qwen35_9b/p174_qwen35_9b_full_attn_tail_graph_probe_20260519_1250.json`
- `reports/qwen35_9b/p174_qwen35_9b_full_attn_tail_graph_probe_20260519_1252_all8.json`
